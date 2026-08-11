"""
Cluster Behaviour Analysis
=============================

Uses ground-truth attack labels (NOT for training) to answer:
  1. Which behavioural features distinguish each cluster?
  2. Which features distinguish each attack class?
  3. Which features are useful for predicting MITRE tactics?
  4. What should the cluster summary contain to correctly predict tactics?

Run AFTER training (needs checkpoint + cluster assignments).

Usage:
    python analyse_clusters.py
"""

import os
import sys

DATASET = sys.argv[1] if len(sys.argv) > 1 else "bot_iot"
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler

if DATASET == 'unsw_nb15':
    from src.config.config_unsw import (
        DATA_PATH, SAMPLE_SIZE,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
        VAE_BATCH_SIZE,
        CHECKPOINT_DIR, MODEL_PATH,
    )
elif DATASET == 'cicids2018':
    from src.config.config_cicids import (
        DATA_PATH, SAMPLE_SIZE,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
        VAE_BATCH_SIZE,
        CHECKPOINT_DIR, MODEL_PATH,
    )
else:
    from src.config.config import (
        DATA_PATH, SAMPLE_SIZE,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
        VAE_BATCH_SIZE,
        CHECKPOINT_DIR, MODEL_PATH,
    )
from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
from src.streaming.streaming_gmm import StreamingDPGMM
from src.evaluation.ground_truth import GroundTruthMapper

OUT_DIR = os.path.join(CHECKPOINT_DIR, "cluster_analysis")


# ══════════════════════════════════════════════════════════
# Key raw features to analyse (pre-scaling, interpretable)
# ══════════════════════════════════════════════════════════

RAW_FEATURE_COLS = [
    'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'PROTOCOL',
    'L4_SRC_PORT', 'L4_DST_PORT',
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'MIN_TTL', 'MAX_TTL',
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES',
    'ICMP_TYPE',
]


def load_raw_data_with_clusters():
    """Load raw data, run clustering, return everything needed for analysis."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    # Load raw data (BEFORE preprocessing)
    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    df = df.dropna()

    # Preprocess (to get train/test split indices and scaled features)
    preprocessor = Preprocessor()
    data = preprocessor.preprocess_dataset(df)
    X_train = data['X_train']
    y_train = data['y_train']
    src_train, dst_train = data['src_train'], data['dst_train']

    # Get raw feature values for training data (BEFORE scaling)
    # Re-index into original df
    indices = np.arange(len(df))
    np.random.seed(42)  # Match the split seed
    from sklearn.model_selection import train_test_split
    idx_train, _ = train_test_split(
        indices, test_size=0.2,
        stratify=df['Attack'].values if 'Attack' in df.columns else None,
        random_state=42)

    df_train_raw = df.iloc[idx_train].reset_index(drop=True)

    # Build encoder and extract z
    feature_enc = FeatureEncoder(cfg['feature_dim'], cfg['feature_output_dim'])
    context_enc = ContextEncoder(cfg['node_feature_dim'], 32,
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

    # Build graph
    builder = NetworkGraphBuilder(seq_length=cfg['seq_length'])
    graph, temporal_seqs = builder.build_graph(X_train, src_train, dst_train)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    with torch.no_grad():
        node_emb = context_enc.to(device)(graph, node_features)
        context_emb = builder.get_ip_embeddings(
            node_emb, src_train, dst_train).to(device)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    t_mask = torch.ones(len(X_train), temporal_seqs.shape[1]).to(device)

    # Extract z
    all_z = []
    with torch.no_grad():
        for s in range(0, len(X_tensor), VAE_BATCH_SIZE):
            e = min(s + VAE_BATCH_SIZE, len(X_tensor))
            z = joint_enc.get_embeddings(
                graph, X_tensor[s:e], node_features,
                context_emb[s:e], temporal_seqs[s:e], t_mask[s:e])
            all_z.append(z.cpu())
    Z_train = torch.cat(all_z).numpy()

    # Cluster
    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']
    cluster_assignments = gmm.model.predict(Z_train)

    # Compute graph features per flow
    src_to_dsts = {}
    dst_to_srcs = {}
    for s, d in zip(src_train, dst_train):
        src_to_dsts.setdefault(s, set()).add(d)
        dst_to_srcs.setdefault(d, set()).add(s)

    fan_out = np.array([len(src_to_dsts.get(s, set())) for s in src_train])
    fan_in = np.array([len(dst_to_srcs.get(d, set())) for d in dst_train])

    # Map to tactics
    mapper = GroundTruthMapper(DATASET)
    true_tactics = mapper.map_labels(y_train)

    return {
        'df_raw': df_train_raw,
        'X_scaled': X_train,
        'y_attack': y_train,
        'y_tactic': true_tactics,
        'clusters': cluster_assignments,
        'Z': Z_train,
        'fan_out': fan_out,
        'fan_in': fan_in,
        'src': src_train,
        'dst': dst_train,
        'feature_names': data['feature_names'],
    }


def compute_derived_features(df_raw, fan_out, fan_in):
    """Compute derived features from RAW data (not scaled)."""
    derived = pd.DataFrame(index=df_raw.index)

    # Raw values
    for col in RAW_FEATURE_COLS:
        if col in df_raw.columns:
            derived[col] = df_raw[col].values

    # Derived
    in_pkts = df_raw['IN_PKTS'].values if 'IN_PKTS' in df_raw.columns else np.zeros(len(df_raw))
    out_pkts = df_raw['OUT_PKTS'].values if 'OUT_PKTS' in df_raw.columns else np.zeros(len(df_raw))
    in_bytes = df_raw['IN_BYTES'].values if 'IN_BYTES' in df_raw.columns else np.zeros(len(df_raw))
    out_bytes = df_raw['OUT_BYTES'].values if 'OUT_BYTES' in df_raw.columns else np.zeros(len(df_raw))
    duration = df_raw['FLOW_DURATION_MILLISECONDS'].values if 'FLOW_DURATION_MILLISECONDS' in df_raw.columns else np.ones(len(df_raw))

    total_pkts = in_pkts + out_pkts
    total_bytes = in_bytes + out_bytes
    dur_sec = np.maximum(duration / 1000.0, 0.0001)

    derived['total_packets'] = total_pkts
    derived['total_bytes'] = total_bytes
    derived['packets_per_second'] = total_pkts / dur_sec
    derived['bytes_per_packet'] = np.where(total_pkts > 0, total_bytes / total_pkts, 0)
    derived['bytes_per_second'] = total_bytes / dur_sec

    # Direction ratio
    src_bps = df_raw.get('SRC_TO_DST_SECOND_BYTES', pd.Series(np.zeros(len(df_raw)))).values
    dst_bps = df_raw.get('DST_TO_SRC_SECOND_BYTES', pd.Series(np.zeros(len(df_raw)))).values
    derived['direction_ratio'] = np.where(
        dst_bps > 0.01, src_bps / dst_bps,
        np.where(src_bps > 0.01, 999.0, 1.0))
    derived['direction_ratio'] = derived['direction_ratio'].clip(0, 999)

    # Graph features
    derived['fan_out'] = fan_out
    derived['fan_in'] = fan_in

    return derived


def analyse_clusters(data):
    """Main analysis: per-cluster and per-tactic feature profiles."""
    os.makedirs(OUT_DIR, exist_ok=True)

    df_raw = data['df_raw']
    y_attack = data['y_attack']
    y_tactic = data['y_tactic']
    clusters = data['clusters']
    fan_out = data['fan_out']
    fan_in = data['fan_in']

    # Compute derived features from RAW data
    derived = compute_derived_features(df_raw, fan_out, fan_in)

    # Key features for analysis
    key_features = [
        'FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'OUT_BYTES',
        'IN_PKTS', 'OUT_PKTS', 'total_packets', 'total_bytes',
        'packets_per_second', 'bytes_per_packet', 'bytes_per_second',
        'direction_ratio', 'fan_out', 'fan_in',
        'TCP_FLAGS', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
        'MIN_TTL', 'L4_DST_PORT', 'PROTOCOL',
    ]
    key_features = [f for f in key_features if f in derived.columns]

    # ══════════════════════════════════════════════════════
    # TABLE 1: Per-cluster profile with ground truth
    # ══════════════════════════════════════════════════════
    print("\n" + "="*100)
    print("TABLE 1: PER-CLUSTER PROFILE")
    print("="*100)

    cluster_profiles = []
    for k in sorted(np.unique(clusters)):
        mask = clusters == k
        n = mask.sum()
        if n < 2:
            continue

        # Ground truth
        attack_counts = Counter(y_attack[mask])
        majority_attack = attack_counts.most_common(1)[0]
        tactic_counts = Counter(y_tactic[mask])
        majority_tactic = tactic_counts.most_common(1)[0]
        purity = majority_attack[1] / n

        # Feature means from RAW data
        cluster_derived = derived.loc[mask]
        row = {
            'cluster': k,
            'size': n,
            'majority_attack': majority_attack[0],
            'purity': purity,
            'majority_tactic': majority_tactic[0],
            'tactic_purity': majority_tactic[1] / n,
        }
        for feat in key_features:
            if feat in cluster_derived.columns:
                row[f'{feat}_mean'] = cluster_derived[feat].mean()
                row[f'{feat}_median'] = cluster_derived[feat].median()

        cluster_profiles.append(row)

    df_clusters = pd.DataFrame(cluster_profiles)

    # Print compact table
    print(f"\n{'Cluster':>8} {'Size':>6} {'Attack':>15} {'Pur%':>5} "
          f"{'Tactic':>15} {'Duration_ms':>12} {'PPS':>10} "
          f"{'B/Pkt':>8} {'FanOut':>7} {'FanIn':>7} "
          f"{'OutBytes':>10} {'InPkts':>8} {'TCPFlag':>8}")
    print("-"*140)

    for _, r in df_clusters.iterrows():
        print(f"{r['cluster']:>8.0f} {r['size']:>6.0f} "
              f"{r['majority_attack']:>15s} {r['purity']:>5.1%} "
              f"{r['majority_tactic']:>15s} "
              f"{r.get('FLOW_DURATION_MILLISECONDS_mean', 0):>12.1f} "
              f"{r.get('packets_per_second_mean', 0):>10.1f} "
              f"{r.get('bytes_per_packet_mean', 0):>8.1f} "
              f"{r.get('fan_out_mean', 0):>7.1f} "
              f"{r.get('fan_in_mean', 0):>7.1f} "
              f"{r.get('OUT_BYTES_mean', 0):>10.1f} "
              f"{r.get('IN_PKTS_mean', 0):>8.1f} "
              f"{r.get('TCP_FLAGS_mean', 0):>8.1f}")

    df_clusters.to_csv(os.path.join(OUT_DIR, 'cluster_profiles.csv'), index=False)

    # ══════════════════════════════════════════════════════
    # TABLE 2: Per-attack-type feature profile
    # ══════════════════════════════════════════════════════
    print("\n\n" + "="*100)
    print("TABLE 2: PER-ATTACK-TYPE FEATURE PROFILE (raw values)")
    print("="*100)

    attack_profiles = []
    for attack in sorted(np.unique(y_attack)):
        mask = y_attack == attack
        n = mask.sum()
        attack_derived = derived.loc[mask]

        row = {'attack': attack, 'count': n}
        for feat in key_features:
            if feat in attack_derived.columns:
                row[f'{feat}_mean'] = attack_derived[feat].mean()
                row[f'{feat}_std'] = attack_derived[feat].std()
                row[f'{feat}_median'] = attack_derived[feat].median()

        attack_profiles.append(row)

    df_attacks = pd.DataFrame(attack_profiles)

    print(f"\n{'Attack':>15} {'Count':>7} {'Duration_ms':>12} {'PPS':>10} "
          f"{'B/Pkt':>8} {'FanOut':>7} {'FanIn':>7} "
          f"{'OutBytes':>10} {'InBytes':>10} {'TCPFlag':>8} {'DstPort':>8}")
    print("-"*120)

    for _, r in df_attacks.iterrows():
        print(f"{r['attack']:>15s} {r['count']:>7.0f} "
              f"{r.get('FLOW_DURATION_MILLISECONDS_mean', 0):>12.1f} "
              f"{r.get('packets_per_second_mean', 0):>10.1f} "
              f"{r.get('bytes_per_packet_mean', 0):>8.1f} "
              f"{r.get('fan_out_mean', 0):>7.1f} "
              f"{r.get('fan_in_mean', 0):>7.1f} "
              f"{r.get('OUT_BYTES_mean', 0):>10.1f} "
              f"{r.get('IN_BYTES_mean', 0):>10.1f} "
              f"{r.get('TCP_FLAGS_mean', 0):>8.1f} "
              f"{r.get('L4_DST_PORT_mean', 0):>8.0f}")

    df_attacks.to_csv(os.path.join(OUT_DIR, 'attack_profiles.csv'), index=False)

    # ══════════════════════════════════════════════════════
    # TABLE 3: Per-tactic feature profile (what matters for eval)
    # ══════════════════════════════════════════════════════
    print("\n\n" + "="*100)
    print("TABLE 3: PER-TACTIC FEATURE PROFILE (what distinguishes tactics)")
    print("="*100)

    tactic_profiles = []
    for tactic in sorted(np.unique(y_tactic)):
        mask = y_tactic == tactic
        n = mask.sum()
        tactic_derived = derived.loc[mask]

        row = {'tactic': tactic, 'count': n,
               'attacks': ', '.join(sorted(np.unique(y_attack[mask])))}
        for feat in key_features:
            if feat in tactic_derived.columns:
                row[f'{feat}_mean'] = tactic_derived[feat].mean()
                row[f'{feat}_std'] = tactic_derived[feat].std()

        tactic_profiles.append(row)

    df_tactics = pd.DataFrame(tactic_profiles)

    print(f"\n{'Tactic':>15} {'Count':>7} {'Attacks':>25} "
          f"{'Duration_ms':>12} {'PPS':>10} {'B/Pkt':>8} "
          f"{'FanOut':>7} {'FanIn':>7} {'OutBytes':>10}")
    print("-"*110)

    for _, r in df_tactics.iterrows():
        print(f"{r['tactic']:>15s} {r['count']:>7.0f} "
              f"{r['attacks']:>25s} "
              f"{r.get('FLOW_DURATION_MILLISECONDS_mean', 0):>12.1f} "
              f"{r.get('packets_per_second_mean', 0):>10.1f} "
              f"{r.get('bytes_per_packet_mean', 0):>8.1f} "
              f"{r.get('fan_out_mean', 0):>7.1f} "
              f"{r.get('fan_in_mean', 0):>7.1f} "
              f"{r.get('OUT_BYTES_mean', 0):>10.1f}")

    df_tactics.to_csv(os.path.join(OUT_DIR, 'tactic_profiles.csv'), index=False)

    # ══════════════════════════════════════════════════════
    # TABLE 4: Feature discriminability for tactic prediction
    # ══════════════════════════════════════════════════════
    print("\n\n" + "="*100)
    print("TABLE 4: FEATURE DISCRIMINABILITY (which features separate tactics?)")
    print("="*100)

    # For each feature, compute how well it separates tactics
    # using between-class / within-class variance ratio (Fisher's criterion)
    print(f"\n{'Feature':>30} {'Fisher Ratio':>12} {'Best Separates':>40}")
    print("-"*85)

    fisher_scores = {}
    for feat in key_features:
        if feat not in derived.columns:
            continue

        values = derived[feat].values
        global_mean = values.mean()

        # Between-class variance
        between = 0
        within = 0
        for tactic in np.unique(y_tactic):
            mask = y_tactic == tactic
            class_vals = values[mask]
            n_k = mask.sum()
            if n_k < 2:
                continue
            class_mean = class_vals.mean()
            between += n_k * (class_mean - global_mean) ** 2
            within += class_vals.var() * n_k

        fisher = between / max(within, 1e-8)
        fisher_scores[feat] = fisher

        # Which tactic pair does this feature best separate?
        tactic_means = {}
        for tactic in np.unique(y_tactic):
            mask = y_tactic == tactic
            tactic_means[tactic] = values[mask].mean()

        sorted_tactics = sorted(tactic_means.items(), key=lambda x: x[1])
        best_sep = f"{sorted_tactics[0][0]}({sorted_tactics[0][1]:.1f}) vs {sorted_tactics[-1][0]}({sorted_tactics[-1][1]:.1f})"

        print(f"{feat:>30} {fisher:>12.4f} {best_sep:>40}")

    # Sort by discriminability
    top_features = sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n\nTOP 10 MOST DISCRIMINATIVE FEATURES FOR TACTIC PREDICTION:")
    print("-"*60)
    for feat, score in top_features[:10]:
        print(f"  {feat:>30} Fisher={score:.4f}")

    # ══════════════════════════════════════════════════════
    # PLOT: Feature distributions per tactic
    # ══════════════════════════════════════════════════════
    top_6 = [f for f, _ in top_features[:6]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    tactics = sorted(np.unique(y_tactic))
    colors = plt.cm.tab10(np.linspace(0, 1, len(tactics)))

    for idx, feat in enumerate(top_6):
        ax = axes[idx // 3, idx % 3]
        for t_idx, tactic in enumerate(tactics):
            mask = y_tactic == tactic
            vals = derived.loc[mask, feat].values
            # Clip extremes for visibility
            p99 = np.percentile(vals, 99) if len(vals) > 0 else 1
            vals_clipped = vals[vals <= p99]
            if len(vals_clipped) > 0:
                ax.hist(vals_clipped, bins=50, alpha=0.5,
                        label=f"{tactic} ({mask.sum():,})",
                        color=colors[t_idx], density=True)
        ax.set_title(f"{feat}\n(Fisher={fisher_scores[feat]:.3f})", fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlabel(feat)

    fig.suptitle("Top 6 Discriminative Features by Tactic", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'feature_distributions_by_tactic.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved: {OUT_DIR}/feature_distributions_by_tactic.png")

    # ══════════════════════════════════════════════════════
    # SUMMARY: What should the cluster summary contain?
    # ══════════════════════════════════════════════════════
    print("\n\n" + "="*100)
    print("RECOMMENDATION: Features to include in cluster summaries")
    print("="*100)

    print("""
Based on Fisher discriminability analysis, the cluster summary should
PROMINENTLY include these features (in original scale, from RAW data):

""")
    for i, (feat, score) in enumerate(top_features[:10]):
        # Show per-tactic means
        means = []
        for tactic in sorted(np.unique(y_tactic)):
            mask = y_tactic == tactic
            m = derived.loc[mask, feat].mean()
            means.append(f"{tactic}={m:.1f}")
        means_str = ", ".join(means)
        print(f"  {i+1:2d}. {feat:>30s} (Fisher={score:.4f})")
        print(f"      Tactic means: {means_str}")

    # Save all tables
    print(f"\n\nAll tables saved to: {OUT_DIR}/")
    print("  - cluster_profiles.csv")
    print("  - attack_profiles.csv")
    print("  - tactic_profiles.csv")
    print("  - feature_distributions_by_tactic.png")


if __name__ == '__main__':
    print("Loading data and computing clusters...")
    data = load_raw_data_with_clusters()
    analyse_clusters(data)