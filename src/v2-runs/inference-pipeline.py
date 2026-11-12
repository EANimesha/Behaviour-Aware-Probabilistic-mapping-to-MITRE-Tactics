# """
# Inference Pipeline — Window-Based Soft Tactic Scoring
# =======================================================
#
# Uses RAW flow features + window aggregation for tactic prediction.
# Independent of cluster indices — robust to distribution shift.
#
# The encoder + GMM still run for novelty detection and visualization,
# but tactic prediction uses the window predictor directly.
# """
#
# import os
# import numpy as np
# import pandas as pd
# import torch
# from collections import Counter
# from sklearn.model_selection import train_test_split
#
# from src.config.config import (
#     DATA_PATH, SAMPLE_SIZE,
#     FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
#     Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
#     VAE_BATCH_SIZE,
#     CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
# )
# from src.data.loader import load_dataset
# from src.data.preprocess import Preprocessor
# from src.data.graph_builder import NetworkGraphBuilder
# from src.encoder.feature import FeatureEncoder
# from src.encoder.context import ContextEncoder
# from src.encoder.behaviour import TransformerBehaviourEncoder
# from src.encoder.jointencoder import JointEncoder
# from src.streaming.streaming_gmm import StreamingDPGMM
# from src.mapping.window_predictor import WindowPredictor
# from src.evaluation.ground_truth import GroundTruthMapper
# from src.evaluation.metrics import TacticEvaluator
#
# WINDOW_SIZE = 10
#
#
# def extract_raw_features(df):
#     """Extract raw feature arrays from DataFrame."""
#     features = {}
#     for col in ['TCP_FLAGS', 'OUT_BYTES', 'IN_BYTES', 'IN_PKTS', 'OUT_PKTS',
#                 'MIN_TTL', 'L4_DST_PORT', 'FLOW_DURATION_MILLISECONDS',
#                 'PROTOCOL', 'L4_SRC_PORT']:
#         if col in df.columns:
#             features[col] = df[col].values.astype(np.float64)
#
#     for src_col in ['IPV4_SRC_ADDR', 'SRC_IP']:
#         if src_col in df.columns:
#             features['SRC_IP'] = df[src_col].values
#             break
#     for dst_col in ['IPV4_DST_ADDR', 'DST_IP']:
#         if dst_col in df.columns:
#             features['DST_IP'] = df[dst_col].values
#             break
#
#     return features
#
#
# def run_inference(test_csv_path=None):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # ═══════════════════════════════════════════════════
#     # Load test data (raw)
#     # ═══════════════════════════════════════════════════
#     print("\n" + "="*60)
#     print("INFERENCE PIPELINE (Window-Based)")
#     print("="*60)
#
#     if test_csv_path:
#         df_test = load_dataset(test_csv_path)
#     else:
#         # Use test split from training data
#         df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
#         indices = np.arange(len(df))
#         if 'Attack' in df.columns:
#             _, idx_test = train_test_split(
#                 indices, test_size=0.2,
#                 stratify=df['Attack'].values,
#                 random_state=42)
#         else:
#             _, idx_test = train_test_split(
#                 indices, test_size=0.2, random_state=42)
#         df_test = df.iloc[idx_test].reset_index(drop=True)
#
#     print(f"  Test flows: {len(df_test):,}")
#
#     y_test = df_test['Attack'].values if 'Attack' in df_test.columns else None
#
#     # ═══════════════════════════════════════════════════
#     # Extract raw features
#     # ═══════════════════════════════════════════════════
#     features = extract_raw_features(df_test)
#     src_ips = features.get('SRC_IP', np.arange(len(df_test)))
#
#     print(f"  Features extracted: {list(features.keys())}")
#
#     # ═══════════════════════════════════════════════════
#     # Window-based tactic prediction
#     # ═══════════════════════════════════════════════════
#     print(f"\n  Window size: {WINDOW_SIZE}")
#
#     predictor = WindowPredictor(
#         window_size=WINDOW_SIZE,
#         tactic_threshold=0.15)
#
#     windows = predictor.create_windows(
#         src_ips=src_ips,
#         flow_features=features,
#         y_true=y_test)
#
#     print(f"  Windows created: {len(windows)}")
#
#     # ═══════════════════════════════════════════════════
#     # Evaluate if ground truth available
#     # ═══════════════════════════════════════════════════
#     eval_dir = os.path.join(CHECKPOINT_DIR, "inference_eval")
#
#     if y_test is not None:
#         print(f"\n{'='*60}")
#         print("EVALUATION")
#         print(f"{'='*60}")
#
#         gt_mapper = GroundTruthMapper('bot_iot')
#         metrics, results_df = predictor.evaluate(
#             windows, gt_mapper, save_dir=eval_dir)
#
#         # Also compute per-flow predictions for standard metrics
#         flow_tactics = np.full(len(df_test), 'Unknown', dtype=object)
#         flow_behaviours = np.full(len(df_test), '', dtype=object)
#
#         for w in windows:
#             for i in w['flow_indices']:
#                 flow_tactics[i] = w['primary_tactic']
#                 top_bhv = max(w['behaviour_scores'],
#                               key=w['behaviour_scores'].get)
#                 flow_behaviours[i] = top_bhv
#
#         true_tactics = gt_mapper.map_labels(y_test)
#
#         print(f"\n  Per-flow evaluation:")
#         evaluator = TacticEvaluator(tactic_labels=gt_mapper.get_tactic_list())
#         evaluator.evaluate(
#             true_tactics=true_tactics,
#             pred_tactics=flow_tactics,
#             cluster_assignments=np.zeros(len(df_test), dtype=int),
#             save_dir=eval_dir,
#             prefix="inference_")
#
#     # ═══════════════════════════════════════════════════
#     # Save detailed results
#     # ═══════════════════════════════════════════════════
#     print(f"\n{'='*60}")
#     print("OUTPUT")
#     print(f"{'='*60}")
#
#     # Window-level results
#     window_rows = []
#     for w in windows:
#         ts = w['tactic_scores']
#         row = {
#             'src_ip': w['src_ip'],
#             'n_flows': w['n_flows'],
#             'primary_tactic': w['primary_tactic'],
#             'final_tactics': ', '.join(w['final_tactics']),
#             'top_behaviour': max(w['behaviour_scores'],
#                                  key=w['behaviour_scores'].get),
#         }
#         # Add tactic scores
#         for t, s in sorted(ts.items(), key=lambda x: -x[1]):
#             row[f'score_{t}'] = round(s, 4)
#
#         if y_test is not None and 'majority_attack' in w:
#             row['true_attack'] = w['majority_attack']
#             row['true_tactic'] = gt_mapper.get_tactic(w['majority_attack'])
#
#         window_rows.append(row)
#
#     df_windows = pd.DataFrame(window_rows)
#     out_path = os.path.join(eval_dir, 'inference_windows.csv')
#     os.makedirs(eval_dir, exist_ok=True)
#     df_windows.to_csv(out_path, index=False)
#     print(f"  Window results: {out_path}")
#
#     # Flow-level results
#     flow_rows = []
#     for w in windows:
#         for i in w['flow_indices']:
#             flow_rows.append({
#                 'flow_idx': i,
#                 'src_ip': w['src_ip'],
#                 'window_tactic': w['primary_tactic'],
#                 'window_tactics': ', '.join(w['final_tactics']),
#             })
#
#     df_flows = pd.DataFrame(flow_rows)
#     flow_path = os.path.join(eval_dir, 'inference_flows.csv')
#     df_flows.to_csv(flow_path, index=False)
#     print(f"  Flow results: {flow_path}")
#
#     # Summary
#     print(f"\n  Tactic distribution (windows):")
#     for tactic, count in Counter(
#             df_windows['primary_tactic']).most_common():
#         print(f"    {tactic:25s} {count:>6d} windows")
#
#     print(f"\n{'='*60}")
#     print("INFERENCE COMPLETE")
#     print(f"{'='*60}")
#
#     return df_windows
#
#
# if __name__ == '__main__':
#     run_inference()

"""
Inference Pipeline
===================

Stage 1: Load test data + preprocess
Stage 2: Graph + temporal
Stage 3: Extract z (frozen encoder)
Stage 4: DP-GMM predict on z
Stage 5: KB lookup (fast) / LLM labeling (slow, for unknown clusters)
Stage 6: Bayesian fusion + output + evaluation
"""

import os
import numpy as np
import pandas as pd
import torch
import logging
from collections import Counter

from src.config.config import (
    DATA_PATH, SAMPLE_SIZE,
    FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
    Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
    VAE_BATCH_SIZE,
    DEFAULT_W1_GMM, DEFAULT_W2_LLM,
    OPENAI_API_KEY, LLM_MODEL, LLM_CONFIDENCE_THRESHOLD,
    CHECKPOINT_DIR, KB_PATH, MODEL_PATH,
)
from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
from src.streaming.streaming_gmm import StreamingDPGMM
from src.clustering.summarizer import ClusterSummarizer
from src.llm.reasoning import LLMLabeler
from src.knowledge_base.kb import KnowledgeBase
from src.fusion.fusion import BayesianFusion
from src.evaluation.ground_truth import GroundTruthMapper
from src.evaluation.metrics import TacticEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(model_path, kb_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

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
    joint_enc.to(device)
    joint_enc.eval()

    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']

    kb = KnowledgeBase()
    kb.load(kb_path)

    active = np.sum(gmm.model.weights_ > gmm_cfg['weight_threshold'])
    print(f"  Loaded: encoder(z={cfg['z_dim']}), "
          f"GMM({active} clusters), KB({len(kb.entries)} entries)")

    return joint_enc, gmm, kb, checkpoint


def extract_embeddings(joint_enc, graph, X_tensor, node_features,
                       context_emb, temporal_seqs, temporal_mask,
                       batch_size, device):
    joint_enc.eval()
    all_z = []
    with torch.no_grad():
        for s in range(0, len(X_tensor), batch_size):
            e = min(s + batch_size, len(X_tensor))
            z = joint_enc.get_embeddings(
                graph=graph, flow_features=X_tensor[s:e],
                node_features=node_features,
                context_embeddings=context_emb[s:e],
                temporal_sequences=temporal_seqs[s:e],
                temporal_mask=temporal_mask[s:e])
            all_z.append(z.cpu())
    return torch.cat(all_z, dim=0).numpy()


def run_inference(test_csv_path=None, model_path=MODEL_PATH, kb_path=KB_PATH):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    joint_enc, gmm, kb, checkpoint = load_checkpoint(model_path, kb_path, device)
    cfg = checkpoint['config']
    saved_scaler = checkpoint['preprocessor_scaler']
    saved_features = checkpoint['feature_names']

    # ── Stage 1: Load + preprocess ──
    print("\n" + "="*60)
    print("STAGE 1: Load Test Data")
    print("="*60)

    preprocessor = Preprocessor()
    if test_csv_path:
        df = load_dataset(test_csv_path)
        preprocessor.scaler = saved_scaler
        preprocessor.fitted = True
        preprocessor.final_feature_names = saved_features
        X_test, src_test, dst_test = preprocessor.transform_new(df)
        y_test = df['Attack'].values if 'Attack' in df.columns else None
    else:
        df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
        data = preprocessor.preprocess_dataset(df)
        X_test = data['X_test']
        src_test, dst_test = data['src_test'], data['dst_test']
        y_test = data['y_test']

    print(f"  Test: {len(X_test)} flows")

    # ── Stage 2: Graph + temporal ──
    builder = NetworkGraphBuilder(seq_length=cfg['seq_length'])
    graph, temporal_seqs = builder.build_graph(X_test, src_test, dst_test)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    with torch.no_grad():
        node_emb = joint_enc.context_encoder(graph, node_features)
        context_emb = builder.get_ip_embeddings(
            node_emb, src_test, dst_test).to(device)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    t_mask = torch.ones(len(X_test), temporal_seqs.shape[1]).to(device)

    # ── Stage 3: Extract z ──
    print("\n  Extracting z embeddings...")
    Z_test = extract_embeddings(
        joint_enc, graph, X_tensor, node_features,
        context_emb, temporal_seqs, t_mask, VAE_BATCH_SIZE, device)
    print(f"  Z_test: {Z_test.shape}")

    # ── Stage 4: DP-GMM predict ──
    print("\n" + "="*60)
    print("STAGE 4: DP-GMM Predict")
    print("="*60)

    assignments, soft_probs, log_liks, novelty_flags = gmm.predict(Z_test)
    p_gmm = soft_probs.max(axis=1)
    print(f"  Clusters: {len(np.unique(assignments))}")
    print(f"  P(GMM): [{p_gmm.min():.4f}, {p_gmm.max():.4f}]")

    # ── Stage 5: KB lookup / LLM for unknown clusters ──
    print("\n" + "="*60)
    print("STAGE 5: KB Lookup / LLM Labeling")
    print("="*60)

    # Summarize test clusters (for any that need LLM labeling)
    summarizer = ClusterSummarizer(
        feature_names=saved_features, scaler=saved_scaler)
    test_summaries = summarizer.summarize_all_clusters(
        assignments, Z_test, X_test,
        src_ips=src_test, dst_ips=dst_test)

    labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)

    # Check which clusters need LLM labeling
    unknown_clusters = set()
    for k in np.unique(assignments):
        if kb.lookup(int(k)) is None:
            unknown_clusters.add(int(k))

    if unknown_clusters:
        print(f"  {len(unknown_clusters)} unknown clusters need LLM labeling")
        for k in unknown_clusters:
            summary = test_summaries.get(k, {})
            if not summary:
                continue
            weight = float(gmm.model.weights_[k])
            result = labeler.label_cluster(summary, weight)
            kb.add_entry(
                cluster_idx=k,
                tactic=result['tactic'],
                tactic_id=result['tactic_id'],
                technique_id=result.get('technique_id', 'Unknown'),
                technique_name=result.get('technique_name', 'Unknown'),
                behaviour_label=result.get('behaviour_label', ''),
                behaviour_description=result.get('behaviour_description', ''),
                tactics=result.get('tactics', []),
                p_gmm=weight,
                p_llm=result['p_llm'],
                reasoning=result['reasoning'],
                summary_text=summary.get('text', ''),
                fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM])
    else:
        print(f"  All clusters found in KB")

    # Assign per-flow predictions from KB
    tactics, tactic_ids, p_llm_values = [], [], []
    behaviour_labels, reasonings = [], []

    for i in range(len(X_test)):
        k = int(assignments[i])
        entry = kb.lookup(k)
        if entry:
            tactics.append(entry['tactic'])
            tactic_ids.append(entry.get('tactic_id', 'Unknown'))
            p_llm_values.append(entry['p_llm'])
            behaviour_labels.append(entry.get('behaviour_label', ''))
            reasonings.append(entry.get('reasoning', ''))
        else:
            tactics.append('Unknown')
            tactic_ids.append('Unknown')
            p_llm_values.append(0.5)
            behaviour_labels.append('')
            reasonings.append('')

    p_llm_values = np.array(p_llm_values)
    print(f"  Tactic distribution:")
    for tactic, count in Counter(tactics).most_common():
        print(f"    {tactic:30s} {count:6,} ({count/len(tactics)*100:.1f}%)")

    # ── Stage 6: Fusion + Output + Evaluation ──
    print("\n" + "="*60)
    print("STAGE 6: Fusion + Output + Evaluation")
    print("="*60)

    fusion = BayesianFusion(DEFAULT_W1_GMM, DEFAULT_W2_LLM)
    p_final = []
    for i in range(len(X_test)):
        k = int(assignments[i])
        w1, w2 = kb.get_fusion_weights(k)
        p_final.append(fusion.fuse(float(p_gmm[i]), float(p_llm_values[i]), w1, w2))
    p_final = np.array(p_final)

    results = pd.DataFrame({
        'src_ip': src_test, 'dst_ip': dst_test,
        'cluster': assignments,
        'behaviour': behaviour_labels,
        'tactic': tactics, 'tactic_id': tactic_ids,
        'p_gmm': p_gmm, 'p_llm': p_llm_values,
        'p_final': p_final,
        'reasoning': reasonings,
    })

    if y_test is not None:
        results['true_attack'] = y_test
        gt_mapper = GroundTruthMapper('bot_iot')
        true_tactics = gt_mapper.map_labels(y_test)
        results['true_tactic'] = true_tactics
        gt_mapper.print_mapping()

        eval_dir = os.path.join(CHECKPOINT_DIR, "evaluation")
        evaluator = TacticEvaluator(tactic_labels=gt_mapper.get_tactic_list())
        evaluator.evaluate(
            true_tactics=true_tactics,
            pred_tactics=np.array(tactics),
            cluster_assignments=assignments,
            Z=Z_test,
            save_dir=eval_dir, prefix="inference_")

    out_path = os.path.join(CHECKPOINT_DIR, "inference_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Results saved: {out_path}")

    cols = ['src_ip', 'dst_ip', 'behaviour', 'tactic', 'p_final']
    if y_test is not None:
        cols.append('true_tactic')
    print(f"\n  Samples:\n{results[cols].head(10).to_string(index=False)}")

    return results


if __name__ == '__main__':
    run_inference()