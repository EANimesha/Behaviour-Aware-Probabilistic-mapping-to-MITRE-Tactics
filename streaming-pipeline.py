"""
Streaming Evaluation Pipeline (v4)
=====================================

Complete operational streaming with source-IP FIFO windows.

Evaluations:
  1. Online Adaptation: Static vs online GMM with per-flow AND window metrics
  2. Drift Detection: Log-likelihood monitoring with gradual drift
  3. Novelty Detection: Hold-out attack type detection
  4. Window-Level Streaming: Source-IP FIFO buffers with soft tactic aggregation

Usage:
  python streaming_eval.py bot_iot
  python streaming_eval.py cicids2018
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from copy import deepcopy
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, roc_curve
)

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'

if DATASET == 'cicids2018':
    from src.config.config_cicids import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
    )
elif DATASET == 'unsw_nb15':
    from src.config.config_unsw import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
    )
else:
    from src.config.config import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
    )

from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
from src.streaming.streaming_gmm import StreamingDPGMM
from src.knowledge_base.kb import KnowledgeBase
from src.evaluation.ground_truth import GroundTruthMapper

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "streaming_results")
os.makedirs(EVAL_DIR, exist_ok=True)

BATCH_SIZE_STREAM = 1000
WINDOW_SIZE = 50
TACTIC_THRESHOLD = 0.10

HOLDOUT_ATTACKS = {
    'bot_iot': 'Theft',
    'cicids2018': 'Bot',
    'unsw_nb15': 'Backdoor',
}


# ══════════════════════════════════════════════════════
# Source-IP FIFO Window Manager
# ══════════════════════════════════════════════════════

class SourceIPWindowManager:
    """Maintains per-source-IP FIFO buffers for streaming windows."""

    def __init__(self, window_size=50, tactic_threshold=0.10):
        self.window_size = window_size
        self.tactic_threshold = tactic_threshold
        self.buffers = defaultdict(lambda: {
            'tactic_dists': [],
            'true_tactics': [],
            'true_attacks': [],
        })
        self.completed_windows = []

    def add_flows(self, src_ips, tactic_dists, true_tactics=None,
                  true_attacks=None):
        """Add flows to per-source-IP buffers. Emit windows when full."""
        new_windows = []

        for i in range(len(src_ips)):
            src = src_ips[i]
            buf = self.buffers[src]
            buf['tactic_dists'].append(tactic_dists[i])
            if true_tactics is not None:
                buf['true_tactics'].append(true_tactics[i])
            if true_attacks is not None:
                buf['true_attacks'].append(true_attacks[i])

            # Emit window when buffer reaches window_size
            if len(buf['tactic_dists']) >= self.window_size:
                window = self._emit_window(src, buf)
                new_windows.append(window)
                # Reset buffer (non-overlapping windows)
                self.buffers[src] = {
                    'tactic_dists': [],
                    'true_tactics': [],
                    'true_attacks': [],
                }

        self.completed_windows.extend(new_windows)
        return new_windows

    def _emit_window(self, src_ip, buf):
        """Create a window from buffered flows."""
        dists = buf['tactic_dists']
        n = len(dists)

        # Aggregate: mean of per-flow tactic distributions
        agg = defaultdict(float)
        for d in dists:
            for t, s in d.items():
                agg[t] += s / n

        # Normalize
        total = sum(agg.values())
        if total > 0:
            agg = {t: s / total for t, s in agg.items()}

        primary = max(agg, key=agg.get) if agg else 'Unknown'
        sorted_tactics = sorted(agg.items(), key=lambda x: -x[1])
        top2 = [t for t, _ in sorted_tactics[:2]]
        final_set = [t for t, s in agg.items() if s >= self.tactic_threshold]

        window = {
            'src_ip': src_ip,
            'n_flows': n,
            'tactic_scores': dict(agg),
            'primary_tactic': primary,
            'top2_tactics': top2,
            'final_tactics': final_set,
        }

        if buf['true_tactics']:
            tc = Counter(buf['true_tactics'])
            majority = tc.most_common(1)[0]
            window['true_primary'] = majority[0]
            window['true_purity'] = majority[1] / n
            # True distribution
            true_dist = {t: c / n for t, c in tc.items()}
            window['true_distribution'] = true_dist
        if buf['true_attacks']:
            window['true_attacks'] = buf['true_attacks']

        return window


def evaluate_windows(windows, tactic_list):
    """Compute window-level metrics from completed windows."""
    if not windows:
        return {}

    top1_correct = 0
    top2_correct = 0
    cosines = []
    soft_jaccards = []
    kl_divs = []

    for w in windows:
        if 'true_primary' not in w:
            continue

        pred = w['tactic_scores']
        true_dist = w.get('true_distribution', {})
        true_primary = w['true_primary']

        # Top-1
        if w['primary_tactic'] == true_primary:
            top1_correct += 1

        # Top-2
        if true_primary in w.get('top2_tactics', []):
            top2_correct += 1

        # Cosine similarity
        p_vec = np.array([pred.get(t, 0) for t in tactic_list])
        t_vec = np.array([true_dist.get(t, 0) for t in tactic_list])
        p_norm = np.linalg.norm(p_vec)
        t_norm = np.linalg.norm(t_vec)
        if p_norm > 0 and t_norm > 0:
            cosines.append(np.dot(p_vec, t_vec) / (p_norm * t_norm))

        # Soft Jaccard
        mins = sum(min(pred.get(t, 0), true_dist.get(t, 0))
                   for t in set(list(pred.keys()) + list(true_dist.keys())))
        maxs = sum(max(pred.get(t, 0), true_dist.get(t, 0))
                   for t in set(list(pred.keys()) + list(true_dist.keys())))
        if maxs > 0:
            soft_jaccards.append(mins / maxs)

        # KL divergence
        eps = 1e-10
        kl = sum(true_dist.get(t, eps) *
                 np.log(true_dist.get(t, eps) / max(pred.get(t, 0), eps))
                 for t in true_dist)
        kl_divs.append(kl)

    n = len([w for w in windows if 'true_primary' in w])
    if n == 0:
        return {}

    return {
        'n_windows': n,
        'top1_accuracy': top1_correct / n,
        'top2_accuracy': top2_correct / n,
        'mean_cosine': np.mean(cosines) if cosines else 0,
        'mean_soft_jaccard': np.mean(soft_jaccards) if soft_jaccards else 0,
        'mean_kl_div': np.mean(kl_divs) if kl_divs else 0,
    }


# ══════════════════════════════════════════════════════
# Encoding helper
# ══════════════════════════════════════════════════════

def encode_flows(joint_enc, df, device, checkpoint):
    """Encode DataFrame through trained encoder → Z."""
    preprocessor = Preprocessor(clip_extremes=(DATASET != 'bot_iot'))
    if 'preprocessor_scaler' in checkpoint:
        preprocessor.scaler = checkpoint['preprocessor_scaler']
        preprocessor.fitted = True
        preprocessor.final_feature_names = checkpoint.get('feature_names', [])
        X, src, dst = preprocessor.transform_new(df)
    else:
        data = preprocessor.preprocess_dataset(df)
        X = data['X_train']
        src, dst = data['src_train'], data['dst_train']

    cfg = checkpoint['config']
    builder = NetworkGraphBuilder(seq_length=cfg.get('seq_length', SEQ_LENGTH))
    graph, temporal_seqs = builder.build_graph(X, src, dst)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    with torch.no_grad():
        node_emb = joint_enc.context_encoder(graph, node_features)
        context_emb = builder.get_ip_embeddings(
            node_emb, src, dst).to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    t_mask = torch.ones(len(X), temporal_seqs.shape[1]).to(device)

    all_z = []
    with torch.no_grad():
        for s in range(0, len(X_tensor), VAE_BATCH_SIZE):
            e = min(s + VAE_BATCH_SIZE, len(X_tensor))
            h_f = joint_enc.feature_encoder(X_tensor[s:e])
            h_b = joint_enc.behaviour_encoder(temporal_seqs[s:e], t_mask[s:e])
            h_c = context_emb[s:e]
            if hasattr(joint_enc, 'norm_f'):
                h_f = joint_enc.norm_f(h_f)
                h_c = joint_enc.norm_c(h_c)
                h_b = joint_enc.norm_b(h_b)
            h_fusion = torch.cat([h_f, h_c, h_b], dim=1)
            z = joint_enc.encoder(h_fusion)
            all_z.append(z.cpu())

    return torch.cat(all_z).numpy(), src, dst


def load_model_and_data(device='cpu'):
    """Load checkpoint, prepare chronological data."""
    print(f"\n{'='*70}")
    print(f"STREAMING EVALUATION — {DATASET}")
    print(f"{'='*70}")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    feature_enc = FeatureEncoder(cfg['feature_dim'], cfg['feature_output_dim'])
    context_enc = ContextEncoder(
        cfg.get('node_feature_dim', NODE_FEATURE_DIM), 32,
        cfg['context_output_dim'], 0.0)
    behaviour_enc = TransformerBehaviourEncoder(
        cfg['feature_dim'], 64, cfg['behaviour_output_dim'], 2, 4, 0.0)
    joint_enc = JointEncoder(
        feature_enc, context_enc, behaviour_enc,
        cfg['feature_output_dim'], cfg['context_output_dim'],
        cfg['behaviour_output_dim'], cfg['z_dim'],
        n_behaviours=cfg.get('n_behaviours', 5),
        lambda_bhv=cfg.get('lambda_bhv', 1.0))
    joint_enc.load_state_dict(checkpoint['joint_encoder_state'])
    joint_enc.to(device); joint_enc.eval()

    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    if DATASET in ('unsw_nb15', 'cicids2018') and 'Attack' in df.columns:
        from src.data.balancer import FastChronologicalBalancer
        if DATASET == 'unsw_nb15':
            from src.config.config_unsw import (
                MAX_BENIGN_SAMPLES, MAX_ATTACK_PER_CLASS,
                MIN_ATTACK_PER_CLASS, MAX_TOTAL_SAMPLES)
        else:
            from src.config.config_cicids import (
                MAX_BENIGN_SAMPLES, MAX_ATTACK_PER_CLASS,
                MIN_ATTACK_PER_CLASS, MAX_TOTAL_SAMPLES)
        balancer = FastChronologicalBalancer(
            max_total_samples=MAX_TOTAL_SAMPLES,
            max_benign_samples=MAX_BENIGN_SAMPLES,
            max_attack_per_class=MAX_ATTACK_PER_CLASS,
            min_attack_per_class=MIN_ATTACK_PER_CLASS)
        df, _ = balancer.balance(df)

    # Chronological split
    n_train = int(len(df) * 0.8)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test = df.iloc[n_train:].reset_index(drop=True)
    y_train = df_train['Attack'].values
    y_test = df_test['Attack'].values

    print(f"  Train: {len(df_train):,}, Test: {len(df_test):,}")

    print("  Encoding test flows...")
    Z_test, src_test, dst_test = encode_flows(
        joint_enc, df_test, device, checkpoint)
    print(f"  Z_test: {Z_test.shape}")

    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']

    kb = KnowledgeBase()
    kb.load(KB_PATH)

    return (joint_enc, gmm, kb, checkpoint,
            df_train, df_test, Z_test, y_train, y_test,
            src_test, dst_test)


# ══════════════════════════════════════════════════════
# EVAL 1: Online Adaptation with Windows
# ══════════════════════════════════════════════════════

def eval_online_adaptation(Z_test, y_test, src_test, gmm, kb):
    """Static vs online vs dynamic-KB GMM: per-flow AND window metrics."""
    print(f"\n{'='*70}")
    print("EVALUATION 1: ONLINE ADAPTATION (per-flow + windows)")
    print(f"{'='*70}")

    from src.knowledge_base.dynamic_kb import DynamicKB, compute_batch_feature_stats
    from src.mapping.soft_cluster_scorer import SoftClusterScorer

    gt_mapper = GroundTruthMapper(DATASET)
    true_tactics = gt_mapper.map_labels(y_test)
    tactic_list = gt_mapper.get_tactic_list()

    n_batches = len(Z_test) // BATCH_SIZE_STREAM
    if n_batches < 2:
        print("  Not enough data"); return {}

    static_gmm = deepcopy(gmm)
    online_gmm = deepcopy(gmm)
    dynamic_gmm = deepcopy(gmm)

    scorer = SoftClusterScorer(dataset=DATASET)
    dkb = DynamicKB(kb, scorer, dynamic_gmm.model,
                     drift_threshold=0.5, dataset=DATASET)

    static_wm = SourceIPWindowManager(WINDOW_SIZE, TACTIC_THRESHOLD)
    online_wm = SourceIPWindowManager(WINDOW_SIZE, TACTIC_THRESHOLD)
    dynamic_wm = SourceIPWindowManager(WINDOW_SIZE, TACTIC_THRESHOLD)

    records = []

    for b in range(n_batches):
        s = b * BATCH_SIZE_STREAM
        e = min(s + BATCH_SIZE_STREAM, len(Z_test))
        Z_b = Z_test[s:e]
        y_b = y_test[s:e]
        t_b = true_tactics[s:e]
        src_b = src_test[s:e]

        # Static GMM + static KB
        s_clusters = static_gmm.model.predict(Z_b)
        s_ll = static_gmm.model.score_samples(Z_b).mean()
        s_dists, s_preds = [], []
        for i in range(len(s_clusters)):
            entry = kb.lookup(int(s_clusters[i]))
            s_dists.append(entry['tactic_scores'] if entry else {'Unknown': 1.0})
            s_preds.append(entry['primary_tactic'] if entry else 'Unknown')
        s_acc = accuracy_score(t_b, s_preds)
        static_wm.add_flows(src_b, s_dists, true_tactics=t_b, true_attacks=y_b)

        # Online GMM + static KB
        o_clusters = online_gmm.model.predict(Z_b)
        o_ll = online_gmm.model.score_samples(Z_b).mean()
        o_dists, o_preds = [], []
        for i in range(len(o_clusters)):
            entry = kb.lookup(int(o_clusters[i]))
            o_dists.append(entry['tactic_scores'] if entry else {'Unknown': 1.0})
            o_preds.append(entry['primary_tactic'] if entry else 'Unknown')
        o_acc = accuracy_score(t_b, o_preds)
        online_wm.add_flows(src_b, o_dists, true_tactics=t_b, true_attacks=y_b)

        # Dynamic GMM + dynamic KB
        d_clusters = dynamic_gmm.model.predict(Z_b)
        d_ll = dynamic_gmm.model.score_samples(Z_b).mean()
        d_dists, d_preds = dkb.predict_flows(d_clusters)
        d_acc = accuracy_score(t_b, d_preds)
        dynamic_wm.add_flows(src_b, d_dists, true_tactics=t_b, true_attacks=y_b)

        # Record
        for label, acc, ll in [('Static', s_acc, s_ll),
                                ('Online', o_acc, o_ll),
                                ('Dynamic', d_acc, d_ll)]:
            # Compute purity for this method's clusters
            gm = {'Static': static_gmm, 'Online': online_gmm,
                   'Dynamic': dynamic_gmm}[label]
            cls = gm.model.predict(Z_b)
            purities = []
            for k in np.unique(cls):
                mask = cls == k
                if mask.sum() < 2: continue
                maj = Counter(y_b[mask]).most_common(1)[0]
                purities.append(maj[1] / mask.sum())
            purity = np.mean(purities) if purities else 0

            records.append({
                'batch': b + 1, 'method': label,
                'purity': purity, 'flow_accuracy': acc, 'loglik': ll,
            })

        # Update AFTER prediction
        online_gmm.update_online_em(Z_b)
        dynamic_gmm.update_online_em(Z_b)

        # Dynamic KB check and update
        feat_stats = compute_batch_feature_stats(
            Z_b, d_clusters, dynamic_gmm.model)
        updated, new = dkb.check_and_update(
            dynamic_gmm.model, Z_b, d_clusters, feat_stats, y_b)

        if (b + 1) % max(1, n_batches // 5) == 0:
            dkb_stats = dkb.get_stats()
            print(f"  Batch {b+1:>2d}/{n_batches}: "
                  f"Acc S={s_acc:.3f}/O={o_acc:.3f}/D={d_acc:.3f} "
                  f"| LL S={s_ll:.1f}/O={o_ll:.1f}/D={d_ll:.1f} "
                  f"| KB updates={dkb_stats['n_updates']}, "
                  f"new={dkb_stats['n_new_clusters']}")

    # Cumulative window metrics
    df_rec = pd.DataFrame(records)

    static_cum = evaluate_windows(static_wm.completed_windows, tactic_list)
    online_cum = evaluate_windows(online_wm.completed_windows, tactic_list)
    dynamic_cum = evaluate_windows(dynamic_wm.completed_windows, tactic_list)

    dkb_stats = dkb.get_stats()

    print(f"\n  Cumulative window metrics:")
    for label, cum in [('Static', static_cum), ('Online', online_cum),
                        ('Dynamic', dynamic_cum)]:
        print(f"    {label:8s}: {cum.get('n_windows',0)} windows, "
              f"Top-1={cum.get('top1_accuracy',0):.4f}, "
              f"Top-2={cum.get('top2_accuracy',0):.4f}, "
              f"Cosine={cum.get('mean_cosine',0):.4f}, "
              f"KL={cum.get('mean_kl_div',0):.4f}")

    print(f"\n  Dynamic KB stats:")
    print(f"    Cluster re-scores: {dkb_stats['n_updates']}")
    print(f"    New clusters added: {dkb_stats['n_new_clusters']}")
    print(f"    Final KB size: {dkb_stats['kb_size']}")

    # Build metrics dict BEFORE plotting (plotting references it)
    metrics = {
        'adapt_n_batches': n_batches,
        'adapt_static_flow_acc': df_rec[df_rec['method']=='Static']['flow_accuracy'].mean(),
        'adapt_online_flow_acc': df_rec[df_rec['method']=='Online']['flow_accuracy'].mean(),
        'adapt_dynamic_flow_acc': df_rec[df_rec['method']=='Dynamic']['flow_accuracy'].mean(),
        'adapt_static_ll': df_rec[df_rec['method']=='Static']['loglik'].mean(),
        'adapt_online_ll': df_rec[df_rec['method']=='Online']['loglik'].mean(),
        'adapt_dynamic_ll': df_rec[df_rec['method']=='Dynamic']['loglik'].mean(),
        'adapt_ll_improvement': df_rec[df_rec['method']=='Online']['loglik'].mean() -
                                df_rec[df_rec['method']=='Static']['loglik'].mean(),
        'adapt_static_win_top1': static_cum.get('top1_accuracy', 0),
        'adapt_online_win_top1': online_cum.get('top1_accuracy', 0),
        'adapt_dynamic_win_top1': dynamic_cum.get('top1_accuracy', 0),
        'adapt_static_win_top2': static_cum.get('top2_accuracy', 0),
        'adapt_online_win_top2': online_cum.get('top2_accuracy', 0),
        'adapt_dynamic_win_top2': dynamic_cum.get('top2_accuracy', 0),
        'adapt_static_win_cosine': static_cum.get('mean_cosine', 0),
        'adapt_online_win_cosine': online_cum.get('mean_cosine', 0),
        'adapt_dynamic_win_cosine': dynamic_cum.get('mean_cosine', 0),
        'adapt_static_win_kl': static_cum.get('mean_kl_div', 0),
        'adapt_online_win_kl': online_cum.get('mean_kl_div', 0),
        'adapt_dynamic_win_kl': dynamic_cum.get('mean_kl_div', 0),
        'dkb_n_updates': dkb_stats['n_updates'],
        'dkb_n_new_clusters': dkb_stats['n_new_clusters'],
        'dkb_final_size': dkb_stats['kb_size'],
    }

    # ── Comprehensive plots: user selects best for paper ──

    df_rec = pd.DataFrame(records)
    static_df = df_rec[df_rec['method'] == 'Static']
    online_df = df_rec[df_rec['method'] == 'Online']

    # Figure 1: Per-batch metrics (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot 1: Cluster Purity
    axes[0, 0].plot(static_df['batch'], static_df['purity'], 'b-o', ms=5,
                     label='Static GMM', alpha=0.8, linewidth=2)
    axes[0, 0].plot(online_df['batch'], online_df['purity'], 'r-s', ms=5,
                     label='Online GMM', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Batch (chronological)', fontsize=10)
    axes[0, 0].set_ylabel('Cluster Purity', fontsize=10)
    axes[0, 0].set_title('Cluster Purity Over Batches', fontsize=11)
    axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Per-Flow Tactic Accuracy
    axes[0, 1].plot(static_df['batch'], static_df['flow_accuracy'], 'b-o',
                     ms=5, label='Static GMM', alpha=0.8, linewidth=2)
    axes[0, 1].plot(online_df['batch'], online_df['flow_accuracy'], 'r-s',
                     ms=5, label='Online GMM', alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Batch (chronological)', fontsize=10)
    axes[0, 1].set_ylabel('Tactic Accuracy', fontsize=10)
    axes[0, 1].set_title('Per-Flow Tactic Accuracy Over Batches', fontsize=11)
    axes[0, 1].legend(fontsize=9); axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Log-Likelihood
    axes[1, 0].plot(static_df['batch'], static_df['loglik'], 'b-o', ms=5,
                     label='Static GMM', alpha=0.8, linewidth=2)
    axes[1, 0].plot(online_df['batch'], online_df['loglik'], 'r-s', ms=5,
                     label='Online GMM', alpha=0.8, linewidth=2)
    axes[1, 0].fill_between(static_df['batch'].values,
                              static_df['loglik'].values,
                              online_df['loglik'].values,
                              alpha=0.15, color='green')
    axes[1, 0].set_xlabel('Batch (chronological)', fontsize=10)
    axes[1, 0].set_ylabel('Mean Log-Likelihood', fontsize=10)
    axes[1, 0].set_title('Log-Likelihood Over Batches', fontsize=11)
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary bar chart
    bar_labels = ['Per-Flow Acc', 'Win Top-1', 'Win Cosine']
    static_bars = [
        metrics['adapt_static_flow_acc'],
        static_cum.get('top1_accuracy', 0),
        static_cum.get('mean_cosine', 0),
    ]
    online_bars = [
        metrics['adapt_online_flow_acc'],
        online_cum.get('top1_accuracy', 0),
        online_cum.get('mean_cosine', 0),
    ]
    x = np.arange(len(bar_labels))
    w = 0.35
    axes[1, 1].bar(x - w/2, static_bars, w, label='Static',
                    color='steelblue', alpha=0.7)
    axes[1, 1].bar(x + w/2, online_bars, w, label='Online',
                    color='coral', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(bar_labels, fontsize=10)
    axes[1, 1].set_ylabel('Score', fontsize=10)
    axes[1, 1].set_title('Streaming Performance Summary', fontsize=11)
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.3, axis='y')
    ll_imp = metrics['adapt_ll_improvement']
    axes[1, 1].annotate(f'LL improvement: {ll_imp:+.2f}',
                          xy=(0.95, 0.95), xycoords='axes fraction',
                          fontsize=10, ha='right', va='top',
                          bbox=dict(boxstyle='round', facecolor='lightgreen',
                                    alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'streaming_adaptation.png'), dpi=150)
    plt.close()

    # Figure 2: Cumulative window metrics over batches
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))

    # Recalculate cumulative window Top-1 per batch
    s_wm_cum = SourceIPWindowManager(WINDOW_SIZE, TACTIC_THRESHOLD)
    o_wm_cum = SourceIPWindowManager(WINDOW_SIZE, TACTIC_THRESHOLD)
    s_cum_top1, o_cum_top1 = [], []

    for bb in range(n_batches):
        ss = bb * BATCH_SIZE_STREAM
        ee = min(ss + BATCH_SIZE_STREAM, len(Z_test))
        src_b = src_test[ss:ee]
        t_b = true_tactics[ss:ee]

        for gm, wm, wlist in [(static_gmm, s_wm_cum, s_cum_top1),
                                (online_gmm, o_wm_cum, o_cum_top1)]:
            cls = gm.model.predict(Z_test[ss:ee])
            dists = []
            for i in range(len(cls)):
                entry = kb.lookup(int(cls[i]))
                dists.append(entry['tactic_scores'] if entry
                             else {'Unknown': 1.0})
            wm.add_flows(src_b, dists, true_tactics=t_b)
            cum = evaluate_windows(wm.completed_windows, tactic_list)
            wlist.append(cum.get('top1_accuracy', 0))

    axes2[0].plot(range(1, n_batches+1), s_cum_top1, 'b-o', ms=5,
                   label='Static GMM', alpha=0.8, linewidth=2)
    axes2[0].plot(range(1, n_batches+1), o_cum_top1, 'r-s', ms=5,
                   label='Online GMM', alpha=0.8, linewidth=2)
    axes2[0].set_xlabel('Batch (chronological)', fontsize=10)
    axes2[0].set_ylabel('Cumulative Top-1 Accuracy', fontsize=10)
    axes2[0].set_title('Cumulative Window Top-1 Over Batches', fontsize=11)
    axes2[0].legend(fontsize=9); axes2[0].grid(True, alpha=0.3)

    # Window metrics summary
    win_labels = ['Top-1', 'Top-2', 'Cosine', 'Soft Jac']
    s_win = [static_cum.get('top1_accuracy', 0),
             static_cum.get('top2_accuracy', 0),
             static_cum.get('mean_cosine', 0),
             static_cum.get('mean_soft_jaccard', 0)]
    o_win = [online_cum.get('top1_accuracy', 0),
             online_cum.get('top2_accuracy', 0),
             online_cum.get('mean_cosine', 0),
             online_cum.get('mean_soft_jaccard', 0)]
    x2 = np.arange(len(win_labels))
    axes2[1].bar(x2 - w/2, s_win, w, label='Static', color='steelblue',
                  alpha=0.7)
    axes2[1].bar(x2 + w/2, o_win, w, label='Online', color='coral', alpha=0.7)
    axes2[1].set_xticks(x2)
    axes2[1].set_xticklabels(win_labels, fontsize=10)
    axes2[1].set_ylabel('Score', fontsize=10)
    axes2[1].set_title('Window-Level Streaming Metrics', fontsize=11)
    axes2[1].set_ylim(0, 1.1)
    axes2[1].legend(fontsize=9); axes2[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'streaming_windows.png'), dpi=150)
    plt.close()

    return metrics


# ══════════════════════════════════════════════════════
# EVAL 2: Drift Detection
# ══════════════════════════════════════════════════════

def eval_drift_detection(Z_test, y_test, gmm):
    """Gradual drift simulation with LL monitoring."""
    print(f"\n{'='*70}")
    print("EVALUATION 2: DRIFT DETECTION")
    print(f"{'='*70}")

    gmm_eval = deepcopy(gmm)
    n_normal = int(len(Z_test) * 0.6)
    Z_normal = Z_test[:n_normal]
    Z_drift_base = Z_test[n_normal:]

    n_drift = len(Z_drift_base)
    feature_std = np.std(Z_test, axis=0)
    rng = np.random.RandomState(42)
    drift_scales = np.linspace(0.0, 1.0, n_drift)
    noise = rng.randn(*Z_drift_base.shape) * feature_std * 0.5
    noise *= drift_scales[:, np.newaxis]
    Z_drift = Z_drift_base + noise

    Z_stream = np.vstack([Z_normal, Z_drift])
    n_batches = len(Z_stream) // BATCH_SIZE_STREAM
    drift_batch = n_normal // BATCH_SIZE_STREAM

    logliks = []
    batch_labels = []

    for b in range(n_batches):
        s = b * BATCH_SIZE_STREAM
        e = min(s + BATCH_SIZE_STREAM, len(Z_stream))
        ll = gmm_eval.model.score_samples(Z_stream[s:e]).mean()
        logliks.append(ll)
        batch_labels.append('drift' if b >= drift_batch else 'normal')

    normal_lls = [logliks[b] for b in range(min(drift_batch, len(logliks)))]
    mu_bl = np.mean(normal_lls)
    std_bl = np.std(normal_lls) if len(normal_lls) > 1 else abs(mu_bl) * 0.1
    threshold = mu_bl - 2 * std_bl

    detection_batch = None
    for b in range(drift_batch, len(logliks)):
        if logliks[b] < threshold:
            detection_batch = b; break
    latency = (detection_batch - drift_batch) if detection_batch else -1

    true_drift = np.array([1 if l == 'drift' else 0 for l in batch_labels])
    pred_drift = np.array([1 if ll < threshold else 0 for ll in logliks])
    tp = ((pred_drift == 1) & (true_drift == 1)).sum()
    fp = ((pred_drift == 1) & (true_drift == 0)).sum()
    fn = ((pred_drift == 0) & (true_drift == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    print(f"  Normal: {drift_batch} batches, Drift: {n_batches-drift_batch}")
    print(f"  Baseline LL: {mu_bl:.2f} ± {std_bl:.2f}")
    print(f"  Threshold: {threshold:.2f}")
    if detection_batch is not None:
        print(f"  Detected at batch {detection_batch}, Latency: {latency}")
    else:
        print(f"  NOT detected")
    print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ['steelblue' if l == 'normal' else 'coral' for l in batch_labels]
    ax.bar(range(n_batches), logliks, color=colors, alpha=0.7, width=0.8,
           edgecolor='white', linewidth=0.5)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold (μ-2σ = {threshold:.1f})')
    ax.axvline(x=drift_batch - 0.5, color='black', linestyle=':',
               linewidth=2, label=f'Drift injection (batch {drift_batch})')
    if detection_batch is not None:
        ax.axvline(x=detection_batch, color='green', linestyle='-.',
                   linewidth=2, label=f'Detection (batch {detection_batch})')
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor='steelblue', alpha=0.7, label='Normal'),
                Patch(facecolor='coral', alpha=0.7, label='Drifted')]
    ax.legend(handles=handles, fontsize=8, loc='lower left')
    ax.set_xlabel('Batch Number'); ax.set_ylabel('Mean Log-Likelihood')
    ax.set_title('Drift Detection via Log-Likelihood Monitoring')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'drift_detection.png'), dpi=150)
    plt.close()

    return {
        'drift_injection_batch': drift_batch,
        'drift_detection_batch': detection_batch if detection_batch else -1,
        'drift_detection_latency': latency,
        'drift_precision': precision, 'drift_recall': recall, 'drift_f1': f1,
    }


# ══════════════════════════════════════════════════════
# EVAL 3: Novelty Detection (hold-out)
# ══════════════════════════════════════════════════════

def eval_novelty_detection(joint_enc, gmm, kb, checkpoint,
                            df_train, df_test, device):
    """Hold out one attack, retrain GMM, detect held-out as novel."""
    print(f"\n{'='*70}")
    print("EVALUATION 3: NOVELTY DETECTION (hold-out)")
    print(f"{'='*70}")

    holdout = HOLDOUT_ATTACKS.get(DATASET, 'Theft')
    gt_mapper = GroundTruthMapper(DATASET)

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    y_full = df_full['Attack'].values

    if holdout not in set(y_full):
        available = sorted(set(y_full) - {'Benign'})
        holdout = available[-1]
    print(f"  Holdout: '{holdout}' ({(y_full==holdout).sum()} flows)")

    known_mask = y_full != holdout
    df_known = df_full[known_mask].reset_index(drop=True)
    y_known = df_known['Attack'].values

    train_idx, test_idx = train_test_split(
        np.arange(len(df_known)), train_size=0.8,
        stratify=y_known, random_state=42)
    df_known_train = df_known.iloc[train_idx].reset_index(drop=True)
    df_known_test = df_known.iloc[test_idx].reset_index(drop=True)
    df_holdout = df_full[~known_mask].reset_index(drop=True)

    print(f"  GMM training: {len(df_known_train)} known flows")
    print(f"  Test: {len(df_known_test)} known + {len(df_holdout)} holdout")

    print("  Encoding known training...")
    Z_known, _, _ = encode_flows(joint_enc, df_known_train, device, checkpoint)

    print("  Fitting novelty GMM...")
    gmm_cfg = checkpoint['gmm_config']
    novelty_gmm = BayesianGaussianMixture(
        n_components=gmm_cfg['max_components'],
        covariance_type=gmm_cfg['covariance_type'],
        weight_concentration_prior_type='dirichlet_process',
        max_iter=300, random_state=42, n_init=3)
    novelty_gmm.fit(Z_known)
    print(f"  Active clusters: {np.sum(novelty_gmm.weights_ > 0.01)}")

    train_lls = novelty_gmm.score_samples(Z_known)
    novelty_threshold = np.percentile(train_lls, 5)
    print(f"  Threshold (5th pct): {novelty_threshold:.2f}")

    df_test_all = pd.concat([df_known_test, df_holdout], ignore_index=True)
    y_test_all = df_test_all['Attack'].values

    print("  Encoding test...")
    Z_test_all, _, _ = encode_flows(joint_enc, df_test_all, device, checkpoint)

    test_lls = novelty_gmm.score_samples(Z_test_all)
    novelty_flags = test_lls < novelty_threshold
    anomaly_scores = -test_lls

    is_holdout = (y_test_all == holdout).astype(int)
    true_tactics = gt_mapper.map_labels(y_test_all)
    is_attack = (true_tactics != 'None').astype(int)

    auc_holdout = roc_auc_score(is_holdout, anomaly_scores) \
        if is_holdout.sum() > 0 and (1-is_holdout).sum() > 0 else 0
    auc_attack = roc_auc_score(is_attack, anomaly_scores) \
        if is_attack.sum() > 0 and (1-is_attack).sum() > 0 else 0

    holdout_rate = novelty_flags[is_holdout == 1].mean() if is_holdout.sum() > 0 else 0
    known_fpr = novelty_flags[(is_attack == 1) & (is_holdout == 0)].mean() \
        if ((is_attack == 1) & (is_holdout == 0)).sum() > 0 else 0
    benign_fpr = novelty_flags[is_attack == 0].mean() \
        if (is_attack == 0).sum() > 0 else 0

    print(f"\n  AUC-ROC (holdout): {auc_holdout:.4f}")
    print(f"  Holdout detect:   {holdout_rate:.4f}")
    print(f"  Known FPR:        {known_fpr:.4f}")
    print(f"  Benign FPR:       {benign_fpr:.4f}")

    # Per-attack rates
    print(f"\n  Per-attack flagging:")
    attack_rates = {}
    for a in sorted(set(y_test_all)):
        mask = y_test_all == a
        rate = novelty_flags[mask].mean()
        marker = '★' if a == holdout else ''
        print(f"    {a:25s}: {rate:.4f} ({novelty_flags[mask].sum()}/{mask.sum()}) {marker}")
        attack_rates[a] = rate

    # Plot (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    if is_holdout.sum() > 0 and (1-is_holdout).sum() > 0:
        fpr, tpr, _ = roc_curve(is_holdout, anomaly_scores)
        axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc_holdout:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title(f'Novelty ROC: "{holdout}" as Unknown')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    ll_known = test_lls[is_holdout == 0]
    ll_hold = test_lls[is_holdout == 1]
    axes[1].hist(ll_known, bins=50, alpha=0.6, density=True,
                 color='steelblue', label='Known')
    if len(ll_hold) > 0:
        axes[1].hist(ll_hold, bins=30, alpha=0.6, density=True,
                     color='coral', label=f'{holdout}')
    axes[1].axvline(x=novelty_threshold, color='red', ls='--', lw=1.5,
                    label=f'Threshold={novelty_threshold:.1f}')
    axes[1].set_xlabel('Log-Likelihood'); axes[1].set_ylabel('Density')
    axes[1].set_title('LL Distribution'); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    attacks = sorted(attack_rates.keys())
    rates = [attack_rates[a] for a in attacks]
    colors = ['coral' if a == holdout else 'steelblue' for a in attacks]
    axes[2].barh(attacks, rates, color=colors, alpha=0.7)
    axes[2].axvline(x=0.05, color='red', ls='--', alpha=0.5)
    axes[2].set_xlabel('Flag Rate'); axes[2].set_title('Per-Attack Detection')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'novelty_detection.png'), dpi=150)
    plt.close()

    return {
        'novelty_holdout': holdout,
        'novelty_auc_holdout': auc_holdout,
        'novelty_auc_attack': auc_attack,
        'novelty_holdout_rate': holdout_rate,
        'novelty_known_fpr': known_fpr,
        'novelty_benign_fpr': benign_fpr,
    }


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    (joint_enc, gmm, kb, checkpoint,
     df_train, df_test, Z_test, y_train, y_test,
     src_test, dst_test) = load_model_and_data(device)

    all_metrics = {}

    m1 = eval_online_adaptation(Z_test, y_test, src_test, gmm, kb)
    all_metrics.update(m1)

    m2 = eval_drift_detection(Z_test, y_test, gmm)
    all_metrics.update(m2)

    m3 = eval_novelty_detection(
        joint_enc, gmm, kb, checkpoint, df_train, df_test, device)
    all_metrics.update(m3)

    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("STREAMING EVALUATION SUMMARY")
    print(f"{'='*70}")

    print(f"\n  ONLINE ADAPTATION ({all_metrics.get('adapt_n_batches',0)} batches):")
    print(f"    Per-flow accuracy: Static={all_metrics.get('adapt_static_flow_acc',0):.4f}"
          f" / Online={all_metrics.get('adapt_online_flow_acc',0):.4f}"
          f" / Dynamic={all_metrics.get('adapt_dynamic_flow_acc',0):.4f}")
    print(f"    Window Top-1:      Static={all_metrics.get('adapt_static_win_top1',0):.4f}"
          f" / Online={all_metrics.get('adapt_online_win_top1',0):.4f}"
          f" / Dynamic={all_metrics.get('adapt_dynamic_win_top1',0):.4f}")
    print(f"    Window Top-2:      Static={all_metrics.get('adapt_static_win_top2',0):.4f}"
          f" / Online={all_metrics.get('adapt_online_win_top2',0):.4f}"
          f" / Dynamic={all_metrics.get('adapt_dynamic_win_top2',0):.4f}")
    print(f"    Window Cosine:     Static={all_metrics.get('adapt_static_win_cosine',0):.4f}"
          f" / Online={all_metrics.get('adapt_online_win_cosine',0):.4f}"
          f" / Dynamic={all_metrics.get('adapt_dynamic_win_cosine',0):.4f}")
    print(f"    LL improvement:    {all_metrics.get('adapt_ll_improvement',0):+.2f}")
    print(f"    Dynamic KB: {all_metrics.get('dkb_n_updates',0)} re-scores, "
          f"{all_metrics.get('dkb_n_new_clusters',0)} new clusters")

    print(f"\n  DRIFT DETECTION:")
    print(f"    Latency: {all_metrics.get('drift_detection_latency',0)} batches")
    print(f"    Precision={all_metrics.get('drift_precision',0):.4f}"
          f" Recall={all_metrics.get('drift_recall',0):.4f}"
          f" F1={all_metrics.get('drift_f1',0):.4f}")

    print(f"\n  NOVELTY (holdout='{all_metrics.get('novelty_holdout','?')}'):")
    print(f"    AUC-ROC:        {all_metrics.get('novelty_auc_holdout',0):.4f}")
    print(f"    Holdout detect: {all_metrics.get('novelty_holdout_rate',0):.4f}")
    print(f"    Known FPR:      {all_metrics.get('novelty_known_fpr',0):.4f}")
    print(f"    Benign FPR:     {all_metrics.get('novelty_benign_fpr',0):.4f}")

    df = pd.DataFrame([all_metrics])
    df.to_csv(os.path.join(EVAL_DIR, 'streaming_metrics.csv'), index=False)
    print(f"\n  Saved: {EVAL_DIR}/")


if __name__ == '__main__':
    main()