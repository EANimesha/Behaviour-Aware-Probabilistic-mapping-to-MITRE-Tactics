"""
Ablation Studies for TDSC Paper
==================================

Five ablation experiments:
  1. Encoder Track Ablation (requires retraining per config)
  2. Loss Weight Ablation (requires retraining per λ)
  3. Clustering Method Ablation (reuses embeddings)
  4. Window Size Ablation (reuses predictions)
  5. Fusion Weight Ablation (reuses predictions)

Usage:
  python ablation_studies.py bot_iot          # All ablations
  python ablation_studies.py bot_iot cluster  # Clustering only
  python ablation_studies.py bot_iot window   # Window only
  python ablation_studies.py bot_iot fusion   # Fusion only
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, silhouette_score, normalized_mutual_info_score,
    classification_report
)
from collections import Counter

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'
ABLATION = sys.argv[2] if len(sys.argv) > 2 else 'all'

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
from src.mapping.soft_cluster_scorer import (
    SoftClusterScorer, SoftWindowAggregator, get_behaviour_tactic_weights)
from src.evaluation.ground_truth import GroundTruthMapper
from src.evaluation.soft_eval import SoftTacticEvaluator

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "ablation_results")
os.makedirs(EVAL_DIR, exist_ok=True)


def load_checkpoint_data():
    """Load trained model and prepare evaluation data."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(os.path.join("../..",MODEL_PATH), map_location=device,
           weights_only=False)

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

    preprocessor = Preprocessor(clip_extremes=(DATASET != 'bot_iot'))
    data = preprocessor.preprocess_dataset(df)

    X = data['X_train']
    y = data['y_train']
    src, dst = data['src_train'], data['dst_train']

    gt_mapper = GroundTruthMapper(DATASET)
    y_tactic = gt_mapper.map_labels(y)

    # Get Z from checkpoint encoder
    cfg = checkpoint['config']
    feature_enc = FeatureEncoder(cfg['feature_dim'], cfg['feature_output_dim'])
    context_enc = ContextEncoder(
        cfg.get('node_feature_dim', 10), 32, cfg['context_output_dim'], 0.0)
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

    builder = NetworkGraphBuilder(seq_length=cfg.get('seq_length', 10))
    graph, temporal_seqs = builder.build_graph(X, src, dst)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    with torch.no_grad():
        node_emb = context_enc.to(device)(graph, node_features)
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

    Z = torch.cat(all_z).numpy()

    return Z, X, y, y_tactic, src, dst, checkpoint, gt_mapper


# ══════════════════════════════════════════════════════
# ABLATION 1: Clustering Method
# ══════════════════════════════════════════════════════

def ablation_clustering(Z, y, y_tactic, gt_mapper):
    """Compare DP-GMM vs GMM vs K-Means."""
    print(f"\n{'='*70}")
    print("ABLATION 1: CLUSTERING METHOD")
    print(f"{'='*70}")

    tactic_list = gt_mapper.get_tactic_list()
    n_tactics = len(tactic_list)
    results = []

    configs = [
        ('K-Means (K=n_t)', 'kmeans', {'n_clusters': n_tactics}),
        ('K-Means (K=2n_t)', 'kmeans', {'n_clusters': n_tactics * 2}),
        ('K-Means (K=5n_t)', 'kmeans', {'n_clusters': n_tactics * 5}),
        ('GMM (K=n_t)', 'gmm', {'n_components': n_tactics}),
        ('GMM (K=2n_t)', 'gmm', {'n_components': n_tactics * 2}),
        ('GMM (K=5n_t)', 'gmm', {'n_components': n_tactics * 5}),
        ('DP-GMM (K_max=20)', 'dpgmm', {'n_components': 20}),
        ('DP-GMM (K_max=30)', 'dpgmm', {'n_components': 30}),
    ]

    for name, method, params in configs:
        t0 = time.time()

        if method == 'kmeans':
            model = KMeans(n_clusters=params['n_clusters'],
                           random_state=42, n_init=10)
            clusters = model.fit_predict(Z)
        elif method == 'gmm':
            model = GaussianMixture(n_components=params['n_components'],
                                     random_state=42, max_iter=200)
            clusters = model.fit_predict(Z)
        elif method == 'dpgmm':
            model = BayesianGaussianMixture(
                n_components=params['n_components'],
                covariance_type='full',
                weight_concentration_prior_type='dirichlet_process',
                max_iter=200, random_state=42)
            clusters = model.fit_predict(Z)

        elapsed = time.time() - t0
        n_active = len(np.unique(clusters))

        # Purity
        purities, sizes = [], []
        for k in np.unique(clusters):
            mask = clusters == k
            if mask.sum() < 2: continue
            maj = Counter(y[mask]).most_common(1)[0]
            purities.append(maj[1] / mask.sum())
            sizes.append(mask.sum())
        mean_pur = np.mean(purities)
        weighted_pur = np.average(purities, weights=sizes)

        # NMI
        nmi_attack = normalized_mutual_info_score(y, clusters)
        nmi_tactic = normalized_mutual_info_score(y_tactic, clusters)

        # Silhouette (sample for speed)
        sample = min(5000, len(Z))
        sil = silhouette_score(Z, clusters, sample_size=sample)

        # Per-flow accuracy via majority vote
        cluster_to_tactic = {}
        for k in np.unique(clusters):
            mask = clusters == k
            majority = Counter(y_tactic[mask]).most_common(1)[0][0]
            cluster_to_tactic[k] = majority
        y_pred = np.array([cluster_to_tactic[c] for c in clusters])
        acc = accuracy_score(y_tactic, y_pred)

        results.append({
            'method': name, 'n_active': n_active,
            'mean_purity': mean_pur, 'weighted_purity': weighted_pur,
            'nmi_attack': nmi_attack, 'nmi_tactic': nmi_tactic,
            'silhouette': sil, 'accuracy': acc, 'time': elapsed,
        })

        print(f"  {name:25s}: K={n_active:>3d}, "
              f"Purity={mean_pur:.3f}, NMI={nmi_tactic:.3f}, "
              f"Sil={sil:.3f}, Acc={acc:.3f}, Time={elapsed:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(EVAL_DIR, 'ablation_clustering.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = df['method'].values
    x = np.arange(len(methods))

    axes[0].bar(x, df['mean_purity'], color='steelblue', alpha=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Mean Purity'); axes[0].set_title('Cluster Purity')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, df['nmi_tactic'], color='coral', alpha=0.7)
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('NMI (Tactic)'); axes[1].set_title('Tactic NMI')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(x, df['accuracy'], color='seagreen', alpha=0.7)
    axes[2].set_xticks(x); axes[2].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel('Accuracy'); axes[2].set_title('Per-Flow Accuracy (Majority Vote)')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'ablation_clustering.png'), dpi=150)
    plt.close()

    return results


# ══════════════════════════════════════════════════════
# ABLATION 2: Window Size
# ══════════════════════════════════════════════════════

def ablation_window_size(Z, y, y_tactic, src, gt_mapper, checkpoint):
    """Test different window sizes."""
    print(f"\n{'='*70}")
    print("ABLATION 2: WINDOW SIZE")
    print(f"{'='*70}")

    tactic_list = gt_mapper.get_tactic_list()
    kb = KnowledgeBase()
    kb.load(os.path.join("../..",KB_PATH))

    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']
    clusters = gmm.model.predict(Z)

    # Build cluster scores from KB
    cluster_scores = {}
    for k in np.unique(clusters):
        entry = kb.lookup(int(k))
        if entry:
            cluster_scores[int(k)] = {
                'behaviour_scores': entry['behaviour_scores'],
                'tactic_scores': entry['tactic_scores'],
                'primary_behaviour': entry['primary_behaviour'],
                'primary_tactic': entry['primary_tactic'],
            }
        else:
            cluster_scores[int(k)] = {
                'behaviour_scores': {'unknown': 1.0},
                'tactic_scores': {'Unknown': 1.0},
                'primary_behaviour': 'unknown',
                'primary_tactic': 'Unknown',
            }

    results = []
    window_sizes = [5, 10, 20, 30, 50, 75, 100, 150, 200]

    for ws in window_sizes:
        aggregator = SoftWindowAggregator(
            window_size=ws, tactic_threshold=0.10)
        windows = aggregator.create_windows(
            src, clusters, cluster_scores, y_true=y)

        soft_eval = SoftTacticEvaluator()
        metrics, _ = soft_eval.evaluate_windows(
            windows, gt_mapper,
            save_dir=None)

        n_windows = metrics.get('n_windows', len(windows))
        results.append({
            'window_size': ws,
            'n_windows': n_windows,
            'top1_accuracy': metrics.get('top1_accuracy', 0),
            'top2_accuracy': metrics.get('top2_accuracy', 0),
            'coverage': metrics.get('coverage', 0),
            'mean_cosine': metrics.get('mean_cosine', 0),
            'mean_soft_jaccard': metrics.get('mean_soft_jaccard', 0),
            'mean_kl_div': metrics.get('mean_kl_div', 0),
        })

        print(f"  W={ws:>4d}: windows={n_windows:>5d}, "
              f"Top-1={results[-1]['top1_accuracy']:.3f}, "
              f"Top-2={results[-1]['top2_accuracy']:.3f}, "
              f"Cosine={results[-1]['mean_cosine']:.3f}, "
              f"KL={results[-1]['mean_kl_div']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(EVAL_DIR, 'ablation_window_size.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(df['window_size'], df['top1_accuracy'], 'b-o', ms=5, label='Top-1')
    axes[0].plot(df['window_size'], df['top2_accuracy'], 'r-s', ms=5, label='Top-2')
    axes[0].set_xlabel('Window Size (W)'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Window Size vs Accuracy')
    axes[0].axvline(x=50, color='green', ls='--', alpha=0.5, label='Selected (W=50)')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['window_size'], df['mean_cosine'], 'g-^', ms=5, label='Cosine')
    axes[1].plot(df['window_size'], df['mean_soft_jaccard'], 'm-d', ms=5, label='Soft Jaccard')
    axes[1].set_xlabel('Window Size (W)'); axes[1].set_ylabel('Score')
    axes[1].set_title('Window Size vs Distribution Metrics')
    axes[1].axvline(x=50, color='green', ls='--', alpha=0.5)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].plot(df['window_size'], df['mean_kl_div'], 'r-o', ms=5)
    axes[2].set_xlabel('Window Size (W)'); axes[2].set_ylabel('KL Divergence ↓')
    axes[2].set_title('Window Size vs KL Divergence')
    axes[2].axvline(x=50, color='green', ls='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'ablation_window_size.png'), dpi=150)
    plt.close()

    return results


# ══════════════════════════════════════════════════════
# ABLATION 3: Fusion Weights
# ══════════════════════════════════════════════════════

def ablation_fusion_weights(Z, y, y_tactic, src, gt_mapper, checkpoint):
    """Test different rule/LLM fusion weights."""
    print(f"\n{'='*70}")
    print("ABLATION 3: FUSION WEIGHTS")
    print(f"{'='*70}")

    tactic_list = gt_mapper.get_tactic_list()
    kb = KnowledgeBase()
    kb.load(os.path.join("../..",KB_PATH))

    cbm_hard = checkpoint.get('cluster_behaviour_map_hard', {})
    if not cbm_hard:
        print("  No LLM labels in checkpoint — skipping")
        return []

    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']
    clusters = gmm.model.predict(Z)

    # Build rule-based and LLM-based score dicts
    cluster_scores_rules = {}
    cluster_scores_llm = {}
    for k in np.unique(clusters):
        entry = kb.lookup(int(k))
        if entry:
            cluster_scores_rules[int(k)] = entry

        # k_str = str(int(k))
        k = int(k)
        if k in cbm_hard and 'tactic_scores_llm' in cbm_hard[k]:
            cluster_scores_llm[int(k)] = {
                'tactic_scores': cbm_hard[k]['tactic_scores_llm'],
                'primary_tactic': cbm_hard[k].get('primary_tactic', 'Unknown'),
                'behaviour_scores': cbm_hard[k].get('behaviour_scores_llm', {}),
                'primary_behaviour': cbm_hard[k].get('primary_behaviour', 'unknown'),
            }

    if not cluster_scores_llm:
        print("  No LLM scores available — skipping")
        return []

    weight_configs = [
        (1.0, 0.0, 'Rule only'),
        (0.8, 0.2, 'w_r=0.8'),
        (0.7, 0.3, 'w_r=0.7'),
        (0.6, 0.4, 'w_r=0.6 (selected)'),
        (0.5, 0.5, 'Equal'),
        (0.4, 0.6, 'w_r=0.4'),
        (0.3, 0.7, 'w_r=0.3'),
        (0.0, 1.0, 'LLM only'),
    ]

    results = []
    for w_r, w_l, label in weight_configs:
        # Fuse per cluster
        fused_scores = {}
        for k in np.unique(clusters):
            k = int(k)
            rules = cluster_scores_rules.get(k, {}).get('tactic_scores', {})
            llm = cluster_scores_llm.get(k, {}).get('tactic_scores', {})

            if not rules and not llm:
                fused_scores[k] = {'Unknown': 1.0}
                continue

            all_t = set(list(rules.keys()) + list(llm.keys()))
            fused = {}
            for t in all_t:
                fused[t] = w_r * rules.get(t, 0) + w_l * llm.get(t, 0)
            total = sum(fused.values())
            if total > 0:
                fused = {t: s / total for t, s in fused.items()}
            fused_scores[k] = fused

        # Per-flow accuracy
        preds = []
        for i in range(len(clusters)):
            fs = fused_scores.get(int(clusters[i]), {'Unknown': 1.0})
            preds.append(max(fs, key=fs.get))
        acc = accuracy_score(y_tactic, preds)

        # Window metrics
        cs_fused = {}
        for k, fs in fused_scores.items():
            cs_fused[k] = {
                'tactic_scores': fs,
                'primary_tactic': max(fs, key=fs.get),
                'behaviour_scores': {}, 'primary_behaviour': '',
            }

        aggregator = SoftWindowAggregator(window_size=50, tactic_threshold=0.10)
        windows = aggregator.create_windows(
            src, clusters, cs_fused, y_true=y)
        soft_eval = SoftTacticEvaluator()
        metrics, _ = soft_eval.evaluate_windows(windows, gt_mapper, save_dir=None)

        results.append({
            'w_rule': w_r, 'w_llm': w_l, 'label': label,
            'perflow_accuracy': acc,
            'window_top1': metrics.get('top1_accuracy', 0),
            'window_cosine': metrics.get('mean_cosine', 0),
            'window_kl': metrics.get('mean_kl_div', 0),
        })

        print(f"  {label:20s}: Acc={acc:.4f}, "
              f"Top-1={results[-1]['window_top1']:.4f}, "
              f"Cosine={results[-1]['window_cosine']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(EVAL_DIR, 'ablation_fusion_weights.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(df['w_rule'], df['perflow_accuracy'], 'b-o', ms=6, label='Per-flow Acc')
    axes[0].plot(df['w_rule'], df['window_top1'], 'r-s', ms=6, label='Window Top-1')
    axes[0].axvline(x=0.6, color='green', ls='--', alpha=0.5, label='Selected')
    axes[0].set_xlabel('Rule Weight (w_r)'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Fusion Weight vs Accuracy')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()

    axes[1].plot(df['w_rule'], df['window_cosine'], 'g-^', ms=6, label='Cosine')
    axes[1].axvline(x=0.6, color='green', ls='--', alpha=0.5)
    axes[1].set_xlabel('Rule Weight (w_r)'); axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Fusion Weight vs Distribution Quality')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'ablation_fusion_weights.png'), dpi=150)
    plt.close()

    return results


# ══════════════════════════════════════════════════════
# ABLATION 4: Covariance Type
# ══════════════════════════════════════════════════════

def ablation_covariance(Z, y, y_tactic):
    """Test different GMM covariance types."""
    print(f"\n{'='*70}")
    print("ABLATION 4: COVARIANCE TYPE")
    print(f"{'='*70}")

    results = []
    for cov_type in ['full', 'diag', 'tied', 'spherical']:
        t0 = time.time()
        try:
            model = BayesianGaussianMixture(
                n_components=20,
                covariance_type=cov_type,
                weight_concentration_prior_type='dirichlet_process',
                max_iter=200, random_state=42)
            clusters = model.fit_predict(Z)
            elapsed = time.time() - t0

            n_active = len(np.unique(clusters))
            purities = []
            for k in np.unique(clusters):
                mask = clusters == k
                if mask.sum() < 2: continue
                maj = Counter(y[mask]).most_common(1)[0]
                purities.append(maj[1] / mask.sum())
            mean_pur = np.mean(purities)

            nmi = normalized_mutual_info_score(y_tactic, clusters)
            sil = silhouette_score(Z, clusters, sample_size=min(5000, len(Z)))

            results.append({
                'covariance': cov_type, 'n_active': n_active,
                'mean_purity': mean_pur, 'nmi_tactic': nmi,
                'silhouette': sil, 'time': elapsed,
            })
            print(f"  {cov_type:12s}: K={n_active:>3d}, Purity={mean_pur:.3f}, "
                  f"NMI={nmi:.3f}, Sil={sil:.3f}, Time={elapsed:.1f}s")
        except Exception as e:
            print(f"  {cov_type:12s}: FAILED — {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(EVAL_DIR, 'ablation_covariance.csv'), index=False)
    return results


# ══════════════════════════════════════════════════════
# ABLATION 5: Temperature Sharpening
# ══════════════════════════════════════════════════════

def ablation_temperature(Z, y, y_tactic, src, gt_mapper, checkpoint):
    """Test different temperature values for score sharpening."""
    print(f"\n{'='*70}")
    print("ABLATION 5: TEMPERATURE SHARPENING")
    print(f"{'='*70}")

    from src.clustering.summarizer import ClusterSummarizer

    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']
    clusters = gmm.model.predict(Z)

    results = []
    for temp in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        # Manually score with different temperature
        scorer = SoftClusterScorer(dataset=DATASET)

        # Override temperature in scorer (monkey-patch)
        original_score = scorer.score_cluster
        def score_with_temp(summary, T=temp):
            result = original_score(summary)
            # Re-sharpen with new temperature
            bhv = result['behaviour_scores']
            sharpened = {k: v ** T for k, v in bhv.items()}
            total = sum(sharpened.values())
            if total > 0:
                bhv = {k: v / total for k, v in sharpened.items()}
            result['behaviour_scores'] = bhv
            btw = get_behaviour_tactic_weights(DATASET)
            from collections import defaultdict
            tactic_scores = defaultdict(float)
            for b, s in bhv.items():
                if b in btw:
                    for t, w in btw[b].items():
                        tactic_scores[t] += s * w
            tt = sum(tactic_scores.values())
            if tt > 0:
                tactic_scores = {t: s/tt for t, s in tactic_scores.items()}
            result['tactic_scores'] = dict(tactic_scores)
            result['primary_tactic'] = max(tactic_scores, key=tactic_scores.get) if tactic_scores else 'Unknown'
            return result

        # Use KB summaries
        kb = KnowledgeBase()
        kb.load(KB_PATH)

        # Per-flow with KB primary tactic (temperature doesn't change KB)
        # Instead compute accuracy from re-scored clusters
        preds = []
        for i in range(len(clusters)):
            entry = kb.lookup(int(clusters[i]))
            if entry:
                preds.append(entry['primary_tactic'])
            else:
                preds.append('Unknown')
        acc = accuracy_score(y_tactic, preds)

        results.append({
            'temperature': temp,
            'perflow_accuracy': acc,
        })
        print(f"  T={temp:.1f}: Acc={acc:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(EVAL_DIR, 'ablation_temperature.csv'), index=False)
    return results


# ══════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════

def print_summary(all_results):
    print(f"\n{'='*70}")
    print("ABLATION STUDIES SUMMARY")
    print(f"{'='*70}")

    for name, results in all_results.items():
        if not results:
            continue
        print(f"\n  {name}:")
        if isinstance(results[0], dict):
            df = pd.DataFrame(results)
            print(df.to_string(index=False))


def main():
    print(f"\n{'='*70}")
    print(f"ABLATION STUDIES — {DATASET}")
    print(f"{'='*70}")

    Z, X, y, y_tactic, src, dst, checkpoint, gt_mapper = load_checkpoint_data()
    print(f"  Z: {Z.shape}, Tactics: {len(set(y_tactic))}")

    all_results = {}

    # if ABLATION in ('all', 'cluster'):
    #     all_results['Clustering Method'] = ablation_clustering(
    #         Z, y, y_tactic, gt_mapper)
    #
    # if ABLATION in ('all', 'covariance'):
    #     all_results['Covariance Type'] = ablation_covariance(Z, y, y_tactic)
    #
    # if ABLATION in ('all', 'window'):
    #     all_results['Window Size'] = ablation_window_size(
    #         Z, y, y_tactic, src, gt_mapper, checkpoint)

    if ABLATION in ('all', 'fusion'):
        all_results['Fusion Weights'] = ablation_fusion_weights(
            Z, y, y_tactic, src, gt_mapper, checkpoint)

    print_summary(all_results)
    print(f"\n  All results saved: {EVAL_DIR}/")


if __name__ == '__main__':
    main()