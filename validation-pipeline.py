# """
# Validation Pipeline — Window-Based Soft Tactic Scoring
# ========================================================
#
# Uses RAW flow features (not cluster labels) for tactic prediction.
# Tests on training data to validate before inference.
#
# Steps:
#   1. Load raw training data
#   2. Extract raw flow features (TCP_FLAGS, OUT_BYTES, MIN_TTL, etc.)
#   3. Create time windows per source IP
#   4. Compute window-level soft behaviour scores
#   5. Map to soft tactic scores
#   6. Evaluate against ground truth
# """
#
# import os
# import numpy as np
# import sys
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# from src.data.balancer import FastChronologicalBalancer
#
# # from src.config.config import (
# #     DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR,
# # )
# # Dataset selection
# DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'
#
# if DATASET == 'unsw_nb15':
#     from src.config.config_unsw import (
#         DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR,
#         VAE_BATCH_SIZE,MODEL_PATH
#     )
# else:
#     from src.config.config import DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR,VAE_BATCH_SIZE,MODEL_PATH
# from src.data.loader import load_dataset
# from src.mapping.window_predictor import WindowPredictor
# from src.evaluation.ground_truth import GroundTruthMapper
#
# EVAL_DIR = os.path.join(CHECKPOINT_DIR, "validation")
#
#
# def extract_raw_features(df):
#     """Extract raw feature arrays from DataFrame."""
#     features = {}
#     raw_cols = [
#         'TCP_FLAGS', 'OUT_BYTES', 'IN_BYTES', 'IN_PKTS', 'OUT_PKTS',
#         'MIN_TTL', 'L4_DST_PORT', 'FLOW_DURATION_MILLISECONDS',
#         'PROTOCOL', 'L4_SRC_PORT',
#     ]
#     for col in raw_cols:
#         if col in df.columns:
#             features[col] = df[col].values.astype(np.float64)
#
#     # Source and destination IPs
#     if 'IPV4_SRC_ADDR' in df.columns:
#         features['SRC_IP'] = df['IPV4_SRC_ADDR'].values
#     elif 'SRC_IP' in df.columns:
#         features['SRC_IP'] = df['SRC_IP'].values
#
#     if 'IPV4_DST_ADDR' in df.columns:
#         features['DST_IP'] = df['IPV4_DST_ADDR'].values
#     elif 'DST_IP' in df.columns:
#         features['DST_IP'] = df['DST_IP'].values
#
#     return features
#
#
# def run_validation():
#     os.makedirs(EVAL_DIR, exist_ok=True)
#
#     # Load raw data
#     print("Loading raw data...")
#     df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
#
#     if DATASET == 'unsw_nb15':
#         balancer = FastChronologicalBalancer(
#                     attack_col="Attack",
#                     benign_label="Benign",
#                     max_total_samples=50_000,
#                     max_benign_samples=10_000,
#                     max_attack_per_class=6_000,
#                     min_attack_per_class=500,
#                 )
#         df, _ = balancer.balance(df)
#         print(f"  Balanced: {len(df):,} flows")
#
#     # Split same as preprocessor (80/20, stratified, random_state=42)
#     if 'Attack' in df.columns:
#         indices = np.arange(len(df))
#         idx_train, idx_test = train_test_split(
#             indices, test_size=0.2,
#             stratify=df['Attack'].values,
#             random_state=42)
#         df_train = df.iloc[idx_train].reset_index(drop=True)
#         y_train = df_train['Attack'].values
#     else:
#         df_train = df
#         y_train = None
#
#     print(f"  Training flows: {len(df_train):,}")
#
#     # Extract raw features
#     features = extract_raw_features(df_train)
#
#     # Get source IPs
#     if 'SRC_IP' in features:
#         src_ips = features['SRC_IP']
#     elif 'IPV4_SRC_ADDR' in df_train.columns:
#         src_ips = df_train['IPV4_SRC_ADDR'].values
#     else:
#         print("ERROR: No source IP column found")
#         return
#
#     # Ground truth mapper
#     gt_mapper = GroundTruthMapper(DATASET)
#
#     # Test multiple window sizes
#     for ws in [10, 20, 50, 100]:
#         print(f"\n{'='*60}")
#         print(f"WINDOW SIZE = {ws}")
#         print(f"{'='*60}")
#
#         predictor = WindowPredictor(window_size=ws, tactic_threshold=0.15)
#         windows = predictor.create_windows(
#             src_ips=src_ips,
#             flow_features=features,
#             y_true=y_train)
#
#         print(f"  Windows created: {len(windows)}")
#
#         metrics, results_df = predictor.evaluate(
#             windows, gt_mapper,
#             save_dir=os.path.join(EVAL_DIR, f'ws_{ws}'))
#
#         # Show sample predictions
#         if len(results_df) > 0:
#             print(f"\n  Sample predictions:")
#             sample = results_df.head(5)
#             for _, r in sample.iterrows():
#                 print(f"    {r['true_tactic']:>15s} → {r['pred_tactic']:<15s} "
#                       f"| {r['final_tactics']:<35s} | {r['tactic_scores']}")
#
#     print(f"\n{'='*60}")
#     print("VALIDATION COMPLETE")
#     print(f"{'='*60}")
#     print(f"  Results saved to: {EVAL_DIR}/")
#
#
# if __name__ == '__main__':
#     run_validation()

"""
Validation Pipeline — Window-Based Behaviour → Tactic
========================================================

Uses TRAINING data to validate the full chain:
  1. Load trained model + cluster→behaviour mapping
  2. Get cluster assignments for training data
  3. Map each flow to its behaviour (from KB)
  4. Group flows into time-ordered windows per source IP
  5. Collect behaviour SET per window
  6. Rule-based mapping: behaviour set → tactic set
  7. Evaluate against ground truth

This validates whether the pipeline can predict tactics correctly
BEFORE running on unseen inference data.

Usage:
    python validation.py
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
import sys

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from src.data.balancer import FastChronologicalBalancer
from src.evaluation.soft_eval import SoftTacticEvaluator
from src.mapping.window_predictor import BEHAVIOUR_TACTIC_WEIGHTS

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

# from src.config.config import (
#     DATA_PATH, SAMPLE_SIZE,
#     FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
#     Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
#     VAE_BATCH_SIZE,
#     CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
# )
# Dataset selection
DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'

if DATASET == 'unsw_nb15':
    from src.config.config_unsw import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR,
        VAE_BATCH_SIZE,MODEL_PATH
    )
else:
    from src.config.config import DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR,VAE_BATCH_SIZE,MODEL_PATH

from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
from src.streaming.streaming_gmm import StreamingDPGMM
from src.mapping.behaviour_mapping import (
    TimeWindowMapper, map_behaviours_to_tactics, BEHAVIOUR_TO_TACTICS
)
from src.evaluation.ground_truth import GroundTruthMapper
from src.evaluation.metrics import TacticEvaluator

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "validation")


def load_and_cluster():
    """Load model, data, and get cluster assignments for training data."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    # Load data
    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    if DATASET == 'unsw_nb15':
        balancer = FastChronologicalBalancer(
            attack_col="Attack",
            benign_label="Benign",
            max_total_samples=50_000,
            max_benign_samples=10_000,
            max_attack_per_class=6_000,
            min_attack_per_class=500,
        )
        df, _ = balancer.balance(df)
        print(f"  Balanced: {len(df):,} flows")
    preprocessor = Preprocessor()
    data = preprocessor.preprocess_dataset(df)
    X_train = data['X_train']
    src_train, dst_train = data['src_train'], data['dst_train']
    y_train = data['y_train']

    # Rebuild encoder
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
    print("Extracting embeddings...")
    all_z = []
    with torch.no_grad():
        for s in range(0, len(X_tensor), VAE_BATCH_SIZE):
            e = min(s + VAE_BATCH_SIZE, len(X_tensor))
            z = joint_enc.get_embeddings(
                graph, X_tensor[s:e], node_features,
                context_emb[s:e], temporal_seqs[s:e], t_mask[s:e])
            all_z.append(z.cpu())
    Z_train = torch.cat(all_z).numpy()

    # Get cluster assignments
    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']
    cluster_assignments = gmm.model.predict(Z_train)

    # Get cluster→behaviour map from checkpoint
    cluster_behaviour_map = checkpoint.get('cluster_behaviour_map', {})

    return {
        'X_train': X_train,
        'y_train': y_train,
        'src_train': src_train,
        'dst_train': dst_train,
        'clusters': cluster_assignments,
        'cluster_behaviour_map': cluster_behaviour_map,
        'Z': Z_train,
    }


def  run_validation():
    """Run full validation pipeline."""
    os.makedirs(EVAL_DIR, exist_ok=True)

    # ══════════════════════════════════════════════════════
    # Load data and clusters
    # ══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("VALIDATION PIPELINE")
    print("="*60)

    data = load_and_cluster()
    y_train = data['y_train']
    clusters = data['clusters']
    src_train = data['src_train']
    cbm = data['cluster_behaviour_map']

    # Convert string keys to int
    cbm = {int(k): v for k, v in cbm.items()}

    gt_mapper = GroundTruthMapper(DATASET)
    true_tactics = gt_mapper.map_labels(y_train)

    print(f"\n  Training flows: {len(y_train):,}")
    print(f"  Unique clusters: {len(np.unique(clusters))}")
    print(f"  Cluster→behaviour entries: {len(cbm)}")

    # ══════════════════════════════════════════════════════
    # Step 1: Per-flow behaviour + tactic (baseline)
    # ══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STEP 1: Per-Flow Behaviour → Tactic (baseline)")
    print("="*60)

    from src.mapping.behaviour_mapping import get_primary_tactic

    flow_behaviours = []
    flow_tactics = []
    for i in range(len(clusters)):
        k = int(clusters[i])
        entry = cbm.get(k, {})
        bhv = entry.get('primary_behaviour', 'normal_browsing')
        tactic = get_primary_tactic(bhv)
        flow_behaviours.append(bhv)
        flow_tactics.append(tactic)

    flow_tactics = np.array(flow_tactics)
    flow_behaviours = np.array(flow_behaviours)

    # Per-flow accuracy
    flow_correct = (flow_tactics == true_tactics).sum()
    flow_acc = flow_correct / len(true_tactics)
    print(f"\n  Per-flow accuracy (baseline): {flow_acc:.4f} "
          f"({flow_correct:,}/{len(true_tactics):,})")

    # Per-flow evaluation
    evaluator = TacticEvaluator(tactic_labels=gt_mapper.get_tactic_list())
    flow_metrics = evaluator.evaluate(
        true_tactics=true_tactics,
        pred_tactics=flow_tactics,
        cluster_assignments=clusters,
        save_dir=EVAL_DIR,
        prefix="val_perflow_")

    # Behaviour distribution
    print(f"\n  Behaviour distribution:")
    for bhv, count in Counter(flow_behaviours).most_common():
        tactic = get_primary_tactic(bhv)
        print(f"    {bhv:25s} {count:>7,} → {tactic}")


    # new for evaluations:
    # ── Cluster vs Behaviour purity ──
    print(f"\n  Cluster vs Behaviour purity:")
    print(f"  {'Cluster':>8s} {'Size':>6s} {'Behaviour':>25s} {'Purity':>8s}")
    print(f"  {'-' * 55}")
    for k in sorted(np.unique(clusters)):
            mask = clusters == k
            if mask.sum() < 2:
                continue
            bhv_counts = Counter(flow_behaviours[mask])
            majority = bhv_counts.most_common(1)[0]
            purity = majority[1] / mask.sum()
            print(f"  {k:>8d} {mask.sum():>6d} {majority[0]:>25s} {purity:>8.1%}")

    bhv_nmi = normalized_mutual_info_score(flow_behaviours, clusters)
    bhv_ari = adjusted_rand_score(flow_behaviours, clusters)
    print(f"\n    NMI (clusters vs behaviours): {bhv_nmi:.4f}")
    print(f"    ARI (clusters vs behaviours): {bhv_ari:.4f}")

    # ── Cluster vs Attack type purity ──
    if y_train is not None:
            print(f"\n  Cluster vs Attack type purity:")
            print(f"  {'Cluster':>8s} {'Size':>6s} {'Attack':>20s} {'Purity':>8s} "
                  f"{'Behaviour':>25s} {'Tactic':>20s}")
            print(f"  {'-' * 95}")
            for k in sorted(np.unique(clusters)):
                mask = clusters == k
                if mask.sum() < 2:
                    continue
                atk_counts = Counter(y_train[mask])
                majority = atk_counts.most_common(1)[0]
                purity = majority[1] / mask.sum()
                bhv = Counter(flow_behaviours[mask]).most_common(1)[0][0]
                tactic = Counter(flow_tactics[mask]).most_common(1)[0][0]
                match = "✓" if gt_mapper.get_tactic(majority[0]) == tactic else "✗"
                print(f"  {k:>8d} {mask.sum():>6d} {majority[0]:>20s} {purity:>8.1%} "
                      f"{bhv:>25s} {tactic:>20s} {match}")

            atk_nmi = normalized_mutual_info_score(y_train, clusters)
            atk_ari = adjusted_rand_score(y_train, clusters)
            print(f"\n    NMI (clusters vs attacks):    {atk_nmi:.4f}")
            print(f"    ARI (clusters vs attacks):    {atk_ari:.4f}")
            print(f"    NMI (clusters vs tactics):    "
                  f"{normalized_mutual_info_score(true_tactics, clusters):.4f}")

    # ══════════════════════════════════════════════════════
    # Step 2: Window-based behaviour → tactic
    # ══════════════════════════════════════════════════════
    for window_size in [10, 20, 50, 100]:
        print(f"\n{'='*60}")
        print(f"STEP 2: Window-Based (window_size={window_size})")
        print(f"{'='*60}")

        window_mapper = TimeWindowMapper(
            window_size=window_size,
            cluster_to_behaviour=cbm)

        windows = window_mapper.create_windows(
            src_ips=src_train,
            cluster_assignments=clusters,
            y_true=y_train)

        print(f"\n  Total windows: {len(windows)}")

        # Evaluate
        # metrics, results_df = window_mapper.evaluate_windows
        evaluator = SoftTacticEvaluator(behaviour_to_tactic_map=BEHAVIOUR_TACTIC_WEIGHTS)
        metrics, results_df = evaluator.evaluate_windows(windows, gt_mapper, save_dir='checkpoints/soft_eval')
        #
        # print(f"\n  Window-level accuracy:     {metrics['accuracy']:.4f}")
        # print(f"  Window-level top-2 acc:    {metrics['top2_accuracy']:.4f}")

        # Per-tactic breakdown
        print(f"\n  Per-tactic breakdown:")
        print(f"  {'Tactic':>20s} {'Count':>7s} {'Accuracy':>10s} {'Top-2':>10s}")
        print(f"  {'-'*50}")
        for tactic in gt_mapper.get_tactic_list():
            t_count = metrics.get(f'{tactic}_count', 0)
            t_acc = metrics.get(f'{tactic}_accuracy', 0)
            t_top2 = metrics.get(f'{tactic}_top2', 0)
            if t_count > 0:
                print(f"  {tactic:>20s} {t_count:>7d} {t_acc:>10.4f} {t_top2:>10.4f}")

        # Save results
        results_df.to_csv(
            os.path.join(EVAL_DIR, f'val_windows_{window_size}.csv'),
            index=False)

        # # ── Confusion analysis: which mappings go wrong? ──
        # if len(results_df) > 0:
        #     wrong = results_df[~results_df['correct']]
        #     if len(wrong) > 0:
        #         print(f"\n  Wrong predictions ({len(wrong)} windows):")
        #         confusion = wrong.groupby(
        #             ['true_tactic', 'pred_tactic']
        #         ).size().sort_values(ascending=False)
        #         for (true_t, pred_t), count in confusion.head(10).items():
        #             # Show example behaviours
        #             ex = wrong[
        #                 (wrong['true_tactic'] == true_t) &
        #                 (wrong['pred_tactic'] == pred_t)
        #             ].iloc[0]
        #             print(f"    {true_t:>15s} → {pred_t:<15s} "
        #                   f"({count:>4d} windows) | "
        #                   f"behaviours: {ex['behaviours']}")

    # ══════════════════════════════════════════════════════
    # Step 3: Summary plot
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Plot accuracy vs window size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: cluster → behaviour mapping table
    ax1 = axes[0]
    ax1.axis('off')
    table_data = []
    for k in sorted(cbm.keys()):
        entry = cbm[k]
        bhv = entry.get('primary_behaviour', '?')
        conf = entry.get('confidence', 0)
        mask = clusters == k
        n = mask.sum()
        if n > 0 and y_train is not None:
            maj = Counter(y_train[mask]).most_common(1)[0]
            true_label = f"{maj[0]} ({maj[1]/n*100:.0f}%)"
        else:
            true_label = "?"
        tactic = get_primary_tactic(bhv)
        table_data.append([str(k), str(n), bhv, tactic, true_label])

    if table_data:
        table = ax1.table(
            cellText=table_data[:20],  # First 20 rows
            colLabels=['Cluster', 'Size', 'Behaviour', 'Tactic', 'True Label'],
            loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(col=list(range(5)))
    ax1.set_title("Cluster → Behaviour → Tactic Mapping", fontsize=10)

    # Right: per-tactic accuracy bar chart
    ax2 = axes[1]
    tactics = gt_mapper.get_tactic_list()
    per_tactic_acc = []
    for t in tactics:
        mask = true_tactics == t
        if mask.sum() > 0:
            acc = (flow_tactics[mask] == t).sum() / mask.sum()
        else:
            acc = 0
        per_tactic_acc.append(acc)

    colors = ['green' if a > 0.7 else 'orange' if a > 0.3 else 'red'
              for a in per_tactic_acc]
    ax2.barh(tactics, per_tactic_acc, color=colors, alpha=0.7)
    ax2.set_xlabel('Per-flow Accuracy')
    ax2.set_title('Per-Tactic Accuracy')
    ax2.set_xlim(0, 1)
    for i, v in enumerate(per_tactic_acc):
        ax2.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9)

    fig.suptitle("Validation: Behaviour → Tactic Mapping", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, 'validation_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Summary plot saved: {EVAL_DIR}/validation_summary.png")
    print(f"  All results saved to: {EVAL_DIR}/")


if __name__ == '__main__':
    run_validation()