"""
Streaming Evaluation Pipeline (v5 — Supervisor Revision)
==========================================================

Three-phase evaluation proving adaptive capability:
  Phase 1: Stable traffic (baseline performance)
  Phase 2: Drift injection (performance degradation)
  Phase 3: Post-adaptation (performance recovery)

Compares: Static GMM vs Online GMM + Dynamic KB

Also includes: drift detection and novelty detection.

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
    accuracy_score, f1_score, roc_auc_score, roc_curve
)

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'

if DATASET == 'cicids2018':
    from src.config.config_cicids import *
elif DATASET == 'unsw_nb15':
    from src.config.config_unsw import *
else:
    from src.config.config import *

from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
from src.streaming.streaming_gmm import StreamingDPGMM
from src.knowledge_base.kb import KnowledgeBase
from src.knowledge_base.dynamic_kb import DynamicKB, compute_batch_feature_stats
from src.mapping.soft_cluster_scorer import SoftClusterScorer
from src.evaluation.ground_truth import GroundTruthMapper

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "streaming_results")
os.makedirs(EVAL_DIR, exist_ok=True)

BATCH_SIZE = 1000
WINDOW_SIZE = 50

# Streaming configuration
ONLINE_EM_LR = 0.3                # Higher LR for visible recovery
UPDATE_FREQUENCY = 'every_batch'   # Update after every batch
DRIFT_THRESHOLD_SIGMA = 2.0       # Alert when LL < mu - 2*sigma
DRIFT_CENTROID_THRESHOLD = 0.5     # KB re-score when centroid shift > 0.5 std


def encode_flows(joint_enc, df, device, checkpoint):
    """Encode DataFrame through trained encoder."""
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
    """Load checkpoint and prepare data."""
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

    n_train = int(len(df) * 0.8)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test = df.iloc[n_train:].reset_index(drop=True)

    print(f"  Train: {len(df_train):,}, Test: {len(df_test):,}")
    print("  Encoding test flows...")
    Z_test, src_test, dst_test = encode_flows(
        joint_enc, df_test, device, checkpoint)
    y_test = df_test['Attack'].values
    print(f"  Z_test: {Z_test.shape}")

    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']

    kb = KnowledgeBase()
    kb.load(KB_PATH)

    return (joint_enc, gmm, kb, checkpoint,
            df_train, df_test, Z_test, y_test, src_test)


# ══════════════════════════════════════════════════════
# EVAL 1: Three-Phase Adaptive Evaluation
# ══════════════════════════════════════════════════════

def eval_three_phase(Z_test, y_test, src_test, gmm, kb):
    """Three-phase: stable -> drift -> recovery."""
    print(f"\n{'='*70}")
    print("EVAL 1: THREE-PHASE ADAPTIVE EVALUATION")
    print(f"{'='*70}")

    gt_mapper = GroundTruthMapper(DATASET)
    true_tactics = gt_mapper.map_labels(y_test)

    print(f"\n  Config: lr={ONLINE_EM_LR}, threshold=mu-{DRIFT_THRESHOLD_SIGMA}sigma")
    print(f"  Input: {len(Z_test)} flows")
    print(f"  Tactic composition:")
    for t in sorted(set(true_tactics)):
        print(f"    {t:20s}: {(true_tactics==t).sum()}")

    # Use ALL available data (no subsampling)
    # Shuffle to mix tactics within each batch (avoids single-tactic batches)
    rng = np.random.RandomState(42)
    shuffle_idx = rng.permutation(len(Z_test))
    Z_all = Z_test[shuffle_idx]
    y_all = y_test[shuffle_idx]
    t_all = true_tactics[shuffle_idx]
    src_all = src_test[shuffle_idx]

    # Phase setup — need at least 3 batches per phase
    total_flows = len(Z_all)
    n_phase = total_flows // 3
    bpp = max(3, n_phase // BATCH_SIZE)  # batches per phase (min 3)
    # Adjust batch size if needed
    actual_batch = min(BATCH_SIZE, n_phase // 3)
    if actual_batch < 100:
        actual_batch = n_phase // 3
    bpp = n_phase // actual_batch

    p1_end = bpp * actual_batch
    p3_end = min(3 * bpp * actual_batch, total_flows)
    total_batches = p3_end // actual_batch

    print(f"  Batch size: {actual_batch}, Batches per phase: {bpp}, "
          f"Total batches: {total_batches}")

    # Create stream with synthetic drift in phases 2-3
    Z_stream = Z_all[:p3_end].copy()
    y_stream = y_all[:p3_end].copy()
    t_stream = t_all[:p3_end].copy()
    src_stream = src_all[:p3_end].copy()

    # Global distribution shift for drift
    # Phase 2: ramp up drift (degradation)
    # Phase 3: HOLD drift constant (allow online EM to recover)
    feat_std = np.std(Z_all, axis=0)
    offset_dir = rng.randn(Z_stream.shape[1])
    offset_dir /= (np.linalg.norm(offset_dir) + 1e-10)

    p2_end = 2 * bpp * actual_batch
    max_drift = 2.5

    for i in range(p1_end, p3_end):
        if i < p2_end:
            # Phase 2: ramp from 0 to max
            prog = (i - p1_end) / (p2_end - p1_end)
            drift = prog * max_drift
        else:
            # Phase 3: hold at max (constant shifted distribution)
            drift = max_drift
        Z_stream[i] += offset_dir * feat_std * drift
        Z_stream[i] *= (1.0 + min(drift / max_drift, 1.0) * 0.3)

    # Parallel systems
    static_gmm = deepcopy(gmm)
    online_gmm = deepcopy(gmm)
    online_gmm.learning_rate = ONLINE_EM_LR  # faster adaptation for recovery

    scorer_obj = SoftClusterScorer(dataset=DATASET)
    dkb = DynamicKB(kb, scorer_obj, online_gmm.model,
                     drift_threshold=DRIFT_CENTROID_THRESHOLD, dataset=DATASET)

    # Original centroids for cumulative shift
    orig_means = {}
    for k in range(len(gmm.model.means_)):
        if gmm.model.weights_[k] > 0.01:
            orig_means[k] = gmm.model.means_[k].copy()

    # Window aggregators
    class WinAgg:
        def __init__(self, ws=50):
            self.ws = ws
            self.bufs = defaultdict(list)
            self.all_wins = []
        def add(self, srcs, dists, trues):
            wins = []
            for i in range(len(srcs)):
                s = srcs[i]
                self.bufs[s].append({'p': dists[i], 't': trues[i]})
                if len(self.bufs[s]) >= self.ws:
                    buf = self.bufs[s]
                    agg = defaultdict(float)
                    for f in buf:
                        for t2, sc in f['p'].items():
                            agg[t2] += sc / len(buf)
                    tot = sum(agg.values())
                    if tot > 0: agg = {t2: sc/tot for t2, sc in agg.items()}
                    tc = Counter([f['t'] for f in buf])
                    tp = tc.most_common(1)[0][0]
                    pp = max(agg, key=agg.get) if agg else 'Unknown'
                    all_t = sorted(set(list(agg.keys()) + list(tc.keys())))
                    pv = np.array([agg.get(t2,0) for t2 in all_t])
                    tv = np.array([tc.get(t2,0)/len(buf) for t2 in all_t])
                    pn, tn = np.linalg.norm(pv), np.linalg.norm(tv)
                    cos = np.dot(pv,tv)/(pn*tn) if pn>0 and tn>0 else 0
                    wins.append({'top1': int(pp==tp), 'cos': cos})
                    self.bufs[s] = []
            self.all_wins.extend(wins)
            return wins

    s_wm = WinAgg(WINDOW_SIZE)
    o_wm = WinAgg(WINDOW_SIZE)

    baseline_lls = []
    drift_det = None
    records = []

    for b in range(total_batches):
        si = b * actual_batch
        ei = min(si + actual_batch, p3_end)
        Zb = Z_stream[si:ei]
        tb = t_stream[si:ei]
        sb = src_stream[si:ei]

        pn = 1 if b < bpp else (2 if b < 2*bpp else 3)
        pname = ['','Stable','Drift','Recovery'][pn]

        # Static
        sc = static_gmm.model.predict(Zb)
        sll = static_gmm.model.score_samples(Zb).mean()
        sd, sp = [], []
        for i in range(len(sc)):
            e = kb.lookup(int(sc[i]))
            sd.append(e['tactic_scores'] if e else {'Unknown':1.0})
            sp.append(e['primary_tactic'] if e else 'Unknown')
        sacc = accuracy_score(tb, sp)
        sf1 = f1_score(tb, sp, average='macro', zero_division=0)
        sw = s_wm.add(sb, sd, tb)

        # Online + DKB
        oc = online_gmm.model.predict(Zb)
        oll = online_gmm.model.score_samples(Zb).mean()
        od, _ = dkb.predict_flows(oc)
        op = np.array([max(d, key=d.get) for d in od])
        oacc = accuracy_score(tb, op)
        of1 = f1_score(tb, op, average='macro', zero_division=0)
        ow = o_wm.add(sb, od, tb)

        # Update online
        online_gmm.update_online_em(Zb)

        # Cumulative centroid check
        for k in range(len(online_gmm.model.means_)):
            if k in orig_means and online_gmm.model.weights_[k] > 0.01:
                std_k = dkb._prev_stds.get(k, np.ones_like(orig_means[k]))
                std_k = np.maximum(std_k, 1e-6)
                shift = np.linalg.norm((online_gmm.model.means_[k] - orig_means[k]) / std_k)
                if shift > DRIFT_CENTROID_THRESHOLD:
                    fs = compute_batch_feature_stats(Zb, oc, online_gmm.model)
                    dkb.check_and_update(online_gmm.model, Zb, oc, fs)
                    break

        # Drift detection
        if pn == 1: baseline_lls.append(sll)
        if len(baseline_lls) >= 2 and drift_det is None:
            mu = np.mean(baseline_lls)
            sd_bl = max(np.std(baseline_lls), abs(mu)*0.05)
            if sll < mu - DRIFT_THRESHOLD_SIGMA * sd_bl:
                drift_det = b

        swt = np.mean([w['top1'] for w in sw]) if sw else float('nan')
        swc = np.mean([w['cos'] for w in sw]) if sw else float('nan')
        owt = np.mean([w['top1'] for w in ow]) if ow else float('nan')
        owc = np.mean([w['cos'] for w in ow]) if ow else float('nan')

        records.append({
            'batch': b+1, 'phase': pname, 'phase_num': pn,
            'static_acc': sacc, 'static_f1': sf1, 'static_ll': sll,
            'static_wt1': swt, 'static_wcos': swc,
            'online_acc': oacc, 'online_f1': of1, 'online_ll': oll,
            'online_wt1': owt, 'online_wcos': owc,
        })

        print(f"  B{b+1:>2d}/{total_batches} [{pname:8s}] "
              f"Acc S={sacc:.3f}/O={oacc:.3f} "
              f"F1 S={sf1:.3f}/O={of1:.3f} "
              f"LL S={sll:.1f}/O={oll:.1f}")

    df = pd.DataFrame(records)

    # Summary
    print(f"\n  {'Phase':12s} {'':8s} {'Acc':>7s} {'M-F1':>7s} "
          f"{'LL':>8s} {'W-Top1':>7s} {'W-Cos':>7s}")
    print(f"  {'─'*58}")

    pm = {}
    for p in [1,2,3]:
        pdf = df[df['phase_num']==p]
        pnm = ['','Stable','Drift','Recovery'][p]
        for m, px in [('Static','static'),('Online','online')]:
            a = pdf[f'{px}_acc'].mean()
            f = pdf[f'{px}_f1'].mean()
            l = pdf[f'{px}_ll'].mean()
            wt = pdf[f'{px}_wt1'].dropna().mean()
            wc = pdf[f'{px}_wcos'].dropna().mean()
            wt = wt if not np.isnan(wt) else 0
            wc = wc if not np.isnan(wc) else 0
            print(f"  {pnm:12s} {m:8s} {a:>7.4f} {f:>7.4f} "
                  f"{l:>8.2f} {wt:>7.4f} {wc:>7.4f}")
            pm[f'p{p}_{px}_acc'] = a
            pm[f'p{p}_{px}_f1'] = f
            pm[f'p{p}_{px}_ll'] = l
            pm[f'p{p}_{px}_wt1'] = wt
            pm[f'p{p}_{px}_wcos'] = wc

    ds = bpp
    if drift_det is not None:
        dd = drift_det - ds
        print(f"\n  Drift detected batch {drift_det} (delay: {dd})")
    else:
        dd = -1
        print(f"\n  Drift NOT detected")

    p1s = pm.get('p1_static_acc',0)
    p3s = pm.get('p3_static_acc',0)
    p3o = pm.get('p3_online_acc',0)
    print(f"  Recovery: baseline={p1s:.4f}, "
          f"static_p3={p3s:.4f} ({(p3s-p1s)*100:+.1f}%), "
          f"online_p3={p3o:.4f} ({(p3o-p1s)*100:+.1f}%)")

    dkbs = dkb.get_stats()
    print(f"  DKB: {dkbs['n_updates']} re-scores, {dkbs['n_new_clusters']} new")

    # Plot
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    #
    # batches = df['batch'].values
    #
    # # Add three-phase background to both plots
    # for ax in axes:
    #     ax.axvspan(0.5, bpp + 0.5, alpha=0.15, color='green')
    #     ax.axvspan(bpp + 0.5, 2 * bpp + 0.5, alpha=0.15, color='red')
    #     ax.axvspan(2 * bpp + 0.5, total_batches + 0.5, alpha=0.15, color='blue')
    #
    # # ---------------------------------------------------------
    # # Plot 1: Per-Flow Accuracy  (previously axes[0,0])
    # # ---------------------------------------------------------
    # ax = axes[0]
    #
    # ax.plot(
    #     batches, df['static_acc'],
    #     'b-o', ms=4, lw=2,
    #     label='Static', alpha=0.8
    # )
    #
    # ax.plot(
    #     batches, df['online_acc'],
    #     'r-s', ms=4, lw=2,
    #     label='Online+DKB', alpha=0.8
    # )
    #
    # ax.set_ylabel('Accuracy')
    # ax.set_xlabel('Batch')
    # ax.set_title('Per-Flow Accuracy')
    # ax.legend(fontsize=9)
    # ax.grid(True, alpha=0.3)
    #
    # # ---------------------------------------------------------
    # # Plot 2: Window Cosine  (previously axes[1,1])
    # # ---------------------------------------------------------
    # ax = axes[1]
    #
    # sc_v = df['static_wcos'].values
    # oc_v = df['online_wcos'].values
    #
    # vs = ~np.isnan(sc_v)
    # vo = ~np.isnan(oc_v)
    #
    # if vs.any():
    #     ax.plot(
    #         batches[vs], sc_v[vs],
    #         'b-o', ms=4, lw=2,
    #         label='Static', alpha=0.8
    #     )
    #
    # if vo.any():
    #     ax.plot(
    #         batches[vo], oc_v[vo],
    #         'r-s', ms=4, lw=2,
    #         label='Online+DKB', alpha=0.8
    #     )
    #
    # ax.set_ylabel('Cosine')
    # ax.set_xlabel('Batch')
    # ax.set_title('Window Cosine')
    # ax.legend(fontsize=9)
    # ax.grid(True, alpha=0.3)
    #
    # for ax in axes:
    #     ym = ax.get_ylim()[1]
    #     ax.text(bpp/2+0.5, ym, 'Stable', ha='center', fontsize=9,
    #             style='italic', va='top')
    #     ax.text(1.5*bpp+0.5, ym, 'Drift', ha='center', fontsize=9,
    #             style='italic', color='red', va='top')
    #     ax.text(2.5*bpp+0.5, ym, 'Recovery', ha='center', fontsize=9,
    #             style='italic', color='blue', va='top')
    #
    # # axes[1,0].set_xlabel('Batch')
    # # plt.suptitle('Three-Phase Streaming Adaptation', fontsize=13)
    # plt.tight_layout()
    # plt.savefig(os.path.join(EVAL_DIR, 'streaming_three_phase.png'),
    #             dpi=150, bbox_inches='tight')
    # plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    batches = df['batch'].values

    for ax in axes.flat:
        ax.axvspan(0.5, bpp + 0.5, alpha=0.15, color='green')
        ax.axvspan(bpp + 0.5, 2 * bpp + 0.5, alpha=0.15, color='red')
        ax.axvspan(2 * bpp + 0.5, total_batches + 0.5, alpha=0.15, color='blue')

    for ax, sm, om, yl, tl in [
        (axes[0, 0], 'static_acc', 'online_acc', 'Accuracy', 'Per-Flow Accuracy'),
        (axes[0, 1], 'static_f1', 'online_f1', 'Macro F1', 'Macro F1-Score'),
        (axes[1, 0], 'static_ll', 'online_ll', 'Log-Likelihood', 'Log-Likelihood')]:
        ax.plot(batches, df[sm], 'b-o', ms=4, lw=2, label='Static', alpha=0.8)
        ax.plot(batches, df[om], 'r-s', ms=4, lw=2, label='Online+DKB', alpha=0.8)
        ax.set_ylabel(yl);
        ax.set_title(tl)
        ax.legend(fontsize=9);
        ax.grid(True, alpha=0.3)

    if len(baseline_lls) >= 2:
        mu = np.mean(baseline_lls)
        sd_bl = max(np.std(baseline_lls), abs(mu) * 0.05)
        axes[1, 0].axhline(y=mu - DRIFT_THRESHOLD_SIGMA * sd_bl,
                           color='red', ls='--', lw=1.5, label='Threshold')
        axes[1, 0].legend(fontsize=9)

    sc_v = df['static_wcos'].values
    oc_v = df['online_wcos'].values
    vs = ~np.isnan(sc_v);
    vo = ~np.isnan(oc_v)
    if vs.any():
        axes[1, 1].plot(batches[vs], sc_v[vs], 'b-o', ms=4, lw=2,
                        label='Static', alpha=0.8)
    if vo.any():
        axes[1, 1].plot(batches[vo], oc_v[vo], 'r-s', ms=4, lw=2,
                        label='Online+DKB', alpha=0.8)
    axes[1, 1].set_ylabel('Cosine');
    axes[1, 1].set_title('Window Cosine')
    axes[1, 1].set_xlabel('Batch');
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ym = ax.get_ylim()[1]
        ax.text(bpp / 2 + 0.5, ym, 'Stable', ha='center', fontsize=9,
                style='italic', va='top')
        ax.text(1.5 * bpp + 0.5, ym, 'Drift', ha='center', fontsize=9,
                style='italic', color='red', va='top')
        ax.text(2.5 * bpp + 0.5, ym, 'Recovery', ha='center', fontsize=9,
                style='italic', color='blue', va='top')

    axes[1, 0].set_xlabel('Batch')
    plt.suptitle('Three-Phase Streaming Adaptation', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'streaming_three_phase.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    pm['detection_delay_batches'] = dd
    pm['detection_delay_flows'] = dd * actual_batch if dd >= 0 else -1
    pm['dkb_rescores'] = dkbs['n_updates']
    pm['dkb_new_clusters'] = dkbs['n_new_clusters']
    return pm


def eval_drift_detection(Z_test, y_test, gmm):
    """Standalone drift detection: LL monitoring with simulated drift."""
    print(f"\n{'='*70}")
    print("EVAL 2: DRIFT DETECTION (LL Monitoring)")
    print(f"{'='*70}")

    gmm_eval = deepcopy(gmm)
    n_normal = int(len(Z_test) * 0.6)
    Z_normal = Z_test[:n_normal]
    Z_drift_base = Z_test[n_normal:]

    n_drift = len(Z_drift_base)
    feature_std = np.std(Z_test, axis=0)
    rng = np.random.RandomState(42)

    # Global distribution shift
    offset_direction = rng.randn(Z_drift_base.shape[1])
    offset_direction = offset_direction / (np.linalg.norm(offset_direction) + 1e-10)

    Z_drift = Z_drift_base.copy()
    for i in range(n_drift):
        progress = i / n_drift
        drift_strength = progress * 3.0
        Z_drift[i] += offset_direction * feature_std * drift_strength
        Z_drift[i] *= (1.0 + progress * 0.3)

    Z_stream = np.vstack([Z_normal, Z_drift])
    n_batches = len(Z_stream) // BATCH_SIZE
    drift_batch = n_normal // BATCH_SIZE

    logliks = []
    batch_labels = []

    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = min(s + BATCH_SIZE, len(Z_stream))
        ll = gmm_eval.model.score_samples(Z_stream[s:e]).mean()
        logliks.append(ll)
        batch_labels.append('drift' if b >= drift_batch else 'normal')

    normal_lls = [logliks[b] for b in range(min(drift_batch, len(logliks)))]
    mu_bl = np.mean(normal_lls)
    std_bl = np.std(normal_lls) if len(normal_lls) > 1 else abs(mu_bl) * 0.1
    threshold = mu_bl - DRIFT_THRESHOLD_SIGMA * std_bl

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
    print(f"  Threshold (μ-{DRIFT_THRESHOLD_SIGMA}σ): {threshold:.2f}")
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
               label=f'Threshold (μ-{DRIFT_THRESHOLD_SIGMA}σ = {threshold:.1f})')
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
        'drift_precision': precision,
        'drift_recall': recall,
        'drift_f1': f1,
    }


# ══════════════════════════════════════════════════════
# EVAL 3: Novelty Detection (hold-out)
# ══════════════════════════════════════════════════════

HOLDOUT_ATTACKS = {
    'bot_iot': 'Theft',
    'cicids2018': 'Bot',
    'unsw_nb15': 'Backdoor',
}

def eval_novelty(joint_enc, gmm, checkpoint, df_train, df_test, device):
    """Hold-out novelty detection."""
    print(f"\n{'='*70}")
    print("EVAL 3: NOVELTY DETECTION")
    print(f"{'='*70}")

    holdout = HOLDOUT_ATTACKS.get(DATASET, 'Theft')
    gt_mapper = GroundTruthMapper(DATASET)

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    y_full = df_full['Attack'].values

    if holdout not in set(y_full):
        available = sorted(set(y_full) - {'Benign'})
        holdout = available[-1]
    print(f"  Holdout: '{holdout}' ({(y_full == holdout).sum()} flows)")

    known_mask = y_full != holdout
    df_known = df_full[known_mask].reset_index(drop=True)
    y_known = df_known['Attack'].values

    train_idx, test_idx = train_test_split(
        np.arange(len(df_known)), train_size=0.8,
        stratify=y_known, random_state=42)
    df_known_train = df_known.iloc[train_idx].reset_index(drop=True)
    df_known_test = df_known.iloc[test_idx].reset_index(drop=True)
    df_holdout = df_full[~known_mask].reset_index(drop=True)

    print(f"  Training GMM on {len(df_known_train)} known flows")
    Z_known, _, _ = encode_flows(joint_enc, df_known_train, device, checkpoint)

    gmm_cfg = checkpoint['gmm_config']
    novelty_gmm = BayesianGaussianMixture(
        n_components=gmm_cfg['max_components'],
        covariance_type=gmm_cfg['covariance_type'],
        weight_concentration_prior_type='dirichlet_process',
        max_iter=300, random_state=42, n_init=3)
    novelty_gmm.fit(Z_known)

    train_lls = novelty_gmm.score_samples(Z_known)
    threshold = np.percentile(train_lls, 5)

    df_test_all = pd.concat([df_known_test, df_holdout], ignore_index=True)
    y_test_all = df_test_all['Attack'].values
    Z_test_all, _, _ = encode_flows(joint_enc, df_test_all, device, checkpoint)

    test_lls = novelty_gmm.score_samples(Z_test_all)
    novelty_flags = test_lls < threshold
    is_holdout = (y_test_all == holdout).astype(int)

    auc = roc_auc_score(is_holdout, -test_lls) if is_holdout.sum() > 0 else 0
    detect_rate = novelty_flags[is_holdout == 1].mean() if is_holdout.sum() > 0 else 0
    true_tactics = gt_mapper.map_labels(y_test_all)
    is_attack = (true_tactics != 'None').astype(int)
    known_fpr = novelty_flags[(is_attack == 1) & (is_holdout == 0)].mean()
    benign_fpr = novelty_flags[is_attack == 0].mean()

    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Detection rate: {detect_rate:.4f}")
    print(f"  Known FPR: {known_fpr:.4f}")
    print(f"  Benign FPR: {benign_fpr:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fpr, tpr, _ = roc_curve(is_holdout, -test_lls)
    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title(f'Novelty ROC: "{holdout}"')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    ll_known = test_lls[is_holdout == 0]
    ll_hold = test_lls[is_holdout == 1]
    axes[1].hist(ll_known, bins=50, alpha=0.6, density=True,
                  color='steelblue', label='Known')
    axes[1].hist(ll_hold, bins=30, alpha=0.6, density=True,
                  color='coral', label=holdout)
    axes[1].axvline(x=threshold, color='red', ls='--', lw=1.5)
    axes[1].set_xlabel('Log-Likelihood'); axes[1].set_title('LL Distribution')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    attack_rates = {}
    for a in sorted(set(y_test_all)):
        m = y_test_all == a
        attack_rates[a] = novelty_flags[m].mean()
    attacks = list(attack_rates.keys())
    rates = [attack_rates[a] for a in attacks]
    colors = ['coral' if a == holdout else 'steelblue' for a in attacks]
    axes[2].barh(attacks, rates, color=colors, alpha=0.7)
    axes[2].set_xlabel('Flag Rate'); axes[2].set_title('Per-Attack Detection')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'novelty_detection.png'), dpi=150)
    plt.close()

    return {
        'novelty_holdout': holdout, 'novelty_auc': auc,
        'novelty_detect_rate': detect_rate,
        'novelty_known_fpr': known_fpr,
    }


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    (joint_enc, gmm, kb, checkpoint,
     df_train, df_test, Z_test, y_test, src_test) = load_model_and_data(device)

    all_metrics = {}

    # Eval 1: Three-phase adaptation (supervisor requirement)
    m1 = eval_three_phase(Z_test, y_test, src_test, gmm, kb)
    all_metrics.update(m1)

    # Eval 2: Drift detection (standalone, already in paper)
    m2 = eval_drift_detection(Z_test, y_test, gmm)
    all_metrics.update(m2)

    # Eval 3: Novelty detection
    m3 = eval_novelty(joint_enc, gmm, checkpoint, df_train, df_test, device)
    all_metrics.update(m3)

    # Summary
    print(f"\n{'='*70}")
    print("STREAMING EVALUATION SUMMARY")
    print(f"{'='*70}")

    print(f"\n  THREE-PHASE ADAPTATION:")
    print(f"  {'Phase':12s} {'':8s} {'Accuracy':>9s} {'Macro-F1':>9s} "
          f"{'AUC':>9s} {'LL':>9s}")
    print(f"  {'─'*55}")
    for p, pname in [(1, 'Stable'), (2, 'Drift'), (3, 'Recovery')]:
        for method, prefix in [('Static', 'static'), ('Online', 'online')]:
            acc = all_metrics.get(f'p{p}_{prefix}_acc', 0)
            f1m = all_metrics.get(f'p{p}_{prefix}_f1', 0)
            auc = all_metrics.get(f'p{p}_{prefix}_auc', 0)
            ll = all_metrics.get(f'p{p}_{prefix}_ll', 0)
            print(f"  {pname:12s} {method:8s} {acc:>9.4f} {f1m:>9.4f} "
                  f"{auc:>9.4f} {ll:>9.2f}")

    dd = all_metrics.get('detection_delay_batches', -1)
    print(f"\n  Detection delay: {dd} batches "
          f"({all_metrics.get('detection_delay_flows', -1)} flows)")
    print(f"  Dynamic KB: {all_metrics.get('dkb_rescores', 0)} re-scores, "
          f"{all_metrics.get('dkb_new_clusters', 0)} new clusters")

    print(f"\n  DRIFT DETECTION (standalone):")
    print(f"    Latency: {all_metrics.get('drift_detection_latency', -1)} batches")
    print(f"    Precision: {all_metrics.get('drift_precision', 0):.4f}")
    print(f"    Recall: {all_metrics.get('drift_recall', 0):.4f}")
    print(f"    F1: {all_metrics.get('drift_f1', 0):.4f}")

    print(f"\n  NOVELTY DETECTION:")
    print(f"    Holdout: {all_metrics.get('novelty_holdout', '?')}")
    print(f"    AUC-ROC: {all_metrics.get('novelty_auc', 0):.4f}")
    print(f"    Detect rate: {all_metrics.get('novelty_detect_rate', 0):.4f}")

    # Save
    df = pd.DataFrame([all_metrics])
    df.to_csv(os.path.join(EVAL_DIR, 'streaming_metrics.csv'), index=False)
    print(f"\n  Saved: {EVAL_DIR}/")


if __name__ == '__main__':
    main()