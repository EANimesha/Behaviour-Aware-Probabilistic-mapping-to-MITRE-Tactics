"""
Cyber Threat Detection System — Inference Pipeline
====================================================

Loads trained checkpoint and Knowledge Base from the training pipeline,
then classifies new network flows into MITRE ATT&CK tactics.

Stage 1: Load test data + preprocess (using saved scaler)
Stage 2: Graph construction (DGL graph + temporal sequences)
Stage 3: Three-track encoder (frozen) + Trained VAE -> Z_test
Stage 4: DP-GMM predict cluster + Cluster Summarization
Stage 5: LLM Semantic Labeling Knowledge Base lookup
Stage 6: Bayesian Fusion: P_final = w1 * P(GMM) + w2 * P(LLM)
Stage 7: Output Prediction with Reasoning
"""

import os
import numpy as np
import pandas as pd
import torch
import logging
import json
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
    """Load trained model checkpoint and Knowledge Base.

    Returns:
        joint_enc: trained JointEncoder (VAE) in eval mode
        gmm: StreamingDPGMM with fitted model
        kb: populated KnowledgeBase
        checkpoint: full checkpoint dict (scaler, config, etc.)
    """
    print("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    # ── Reconstruct encoder architecture ──
    feature_enc = FeatureEncoder(
        cfg['feature_dim'], output_dim=cfg['feature_output_dim']
    )
    context_enc = ContextEncoder(
        cfg['node_feature_dim'], hidden_dim=32,
        output_dim=cfg['context_output_dim'], dropout=0.0  # No dropout at inference
    )
    behaviour_enc = TransformerBehaviourEncoder(
        cfg['feature_dim'], hidden_dim=64,
        output_dim=cfg['behaviour_output_dim'],
        num_layers=2, num_heads=4, dropout=0.0
    )
    joint_enc = JointEncoder(
        feature_encoder=feature_enc,
        context_encoder=context_enc,
        behaviour_encoder=behaviour_enc,
        feature_output_dim=cfg['feature_output_dim'],
        context_output_dim=cfg['context_output_dim'],
        behaviour_output_dim=cfg['behaviour_output_dim'],
        z_dim=cfg['z_dim'],
    )

    # Load trained weights
    joint_enc.load_state_dict(checkpoint['joint_encoder_state'])
    joint_enc.to(device)
    joint_enc.eval()
    print(f"  Encoder loaded (z_dim={cfg['z_dim']}, "
          f"feature_dim={cfg['feature_dim']})")

    # ── Reconstruct DP-GMM ──
    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        max_components=gmm_cfg['max_components'],
        covariance_type=gmm_cfg['covariance_type'],
        weight_threshold=gmm_cfg['weight_threshold'],
        novelty_threshold=gmm_cfg['novelty_threshold'],
    )
    gmm.model = checkpoint['gmm_model']
    active = np.sum(gmm.model.weights_ > gmm_cfg['weight_threshold'])
    print(f"  DP-GMM loaded ({active} active clusters)")

    # ── Load Knowledge Base ──
    kb = KnowledgeBase()
    kb.load(kb_path)
    print(f"  Knowledge Base loaded ({len(kb.entries)} entries)")

    return joint_enc, gmm, kb, checkpoint


def extract_embeddings(joint_enc, graph, X_tensor, node_features,
                       context_embeddings, temporal_seqs, temporal_mask,
                       batch_size, device):
    """Extract latent embeddings Z_test using trained encoder (mu, no noise)."""
    joint_enc.eval()
    n_samples = len(X_tensor)
    all_z = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            z = joint_enc.get_embeddings(
                graph=graph,
                flow_features=X_tensor[start:end],
                node_features=node_features,
                context_embeddings=context_embeddings[start:end],
                temporal_sequences=temporal_seqs[start:end],
                temporal_mask=temporal_mask[start:end],
            )
            all_z.append(z.cpu())

    Z = torch.cat(all_z, dim=0).numpy()
    return Z


def run_inference(test_csv_path=None, model_path=MODEL_PATH, kb_path=KB_PATH):
    """Run the full 7-stage inference pipeline.

    Args:
        test_csv_path: path to test CSV (if None, uses test split from DATA_PATH)
        model_path: path to saved model checkpoint
        kb_path: path to saved Knowledge Base JSON

    Returns:
        results: DataFrame with per-flow predictions
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ═══════════════════════════════════════════════════════════
    # Load trained components
    # ═══════════════════════════════════════════════════════════
    joint_enc, gmm, kb, checkpoint = load_checkpoint(
        model_path, kb_path, device
    )

    cfg = checkpoint['config']
    saved_scaler = checkpoint['preprocessor_scaler']
    saved_feature_names = checkpoint['feature_names']

    # ═══════════════════════════════════════════════════════════
    # Stage 1: Load Test Data and Preprocess
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 1: Load Test Data and Preprocess")
    print("="*60)

    preprocessor = Preprocessor()

    if test_csv_path:
        # Load separate test file
        df_test = load_dataset(test_csv_path, sample_size=None)
        # Use saved scaler for transform
        preprocessor.scaler = saved_scaler
        preprocessor.fitted = True
        preprocessor.final_feature_names = saved_feature_names
        X_test, src_test, dst_test = preprocessor.transform_new(df_test)
        y_test = df_test['Attack'].values if 'Attack' in df_test.columns else None
    else:
        # Use test split from training data
        df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
        data = preprocessor.preprocess_dataset(df)
        X_test = data['X_test']
        src_test = data['src_test']
        dst_test = data['dst_test']
        y_test = data['y_test']

    feature_dim = X_test.shape[1]
    print(f"  Test samples: {len(X_test)}, Features: {feature_dim}")

    # ═══════════════════════════════════════════════════════════
    # Stage 2: Graph Construction
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 2: Graph Construction")
    print("="*60)

    builder = NetworkGraphBuilder(seq_length=cfg['seq_length'])
    test_graph, test_temporal_seqs = builder.build_graph(
        X_test, src_test, dst_test
    )

    test_graph = test_graph.to(device)
    test_temporal_seqs = test_temporal_seqs.to(device)
    test_node_features = builder.node_features.to(device)

    print(f"  Graph: {test_graph.num_nodes()} nodes, "
          f"{test_graph.num_edges()} edges")
    print(f"  Temporal sequences: {test_temporal_seqs.shape}")

    # ═══════════════════════════════════════════════════════════
    # Stage 3: Three-Track Encoder (Frozen) + Trained VAE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 3: Three-Track Encoder (Frozen) + Trained VAE -> Z_test")
    print("="*60)

    # Compute context embeddings using the trained context encoder
    context_enc = joint_enc.context_encoder
    with torch.no_grad():
        test_node_embeddings = context_enc(test_graph, test_node_features)
        test_context_embeddings = builder.get_ip_embeddings(
            test_node_embeddings, src_test, dst_test
        ).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_temporal_mask = torch.ones(
        len(X_test), test_temporal_seqs.shape[1]
    ).to(device)

    # Extract embeddings (uses mu, no sampling noise)
    Z_test = extract_embeddings(
        joint_enc, test_graph, X_test_tensor, test_node_features,
        test_context_embeddings, test_temporal_seqs, test_temporal_mask,
        batch_size=VAE_BATCH_SIZE, device=device,
    )
    print(f"  Z_test shape: {Z_test.shape}")

    # ═══════════════════════════════════════════════════════════
    # Stage 4: DP-GMM Predict Cluster + Cluster Summarization
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 4: DP-GMM Predict Cluster + Cluster Summarization")
    print("="*60)

    # Predict clusters and get P(GMM)
    assignments, soft_probs, log_liks, novelty_flags = gmm.predict(Z_test)

    # P(GMM) = max cluster membership probability per flow
    p_gmm_values = soft_probs.max(axis=1)

    print(f"  Cluster assignments: {len(np.unique(assignments))} unique clusters")
    print(f"  P(GMM) range: [{p_gmm_values.min():.4f}, {p_gmm_values.max():.4f}]")
    print(f"  Novelty flags: {novelty_flags.sum()} flows flagged")

    # Cluster Summarization Module (for any flows needing LLM validation)
    summarizer = ClusterSummarizer(
        feature_names=saved_feature_names,
        scaler=saved_scaler,
    )
    test_summaries = summarizer.summarize_all_clusters(
        assignments, Z_test, X_test,
        src_ips=src_test, dst_ips=dst_test,
    )

    # ═══════════════════════════════════════════════════════════
    # Stage 5: LLM Semantic Labeling Knowledge Base
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 5: LLM Semantic Labeling Knowledge Base Lookup")
    print("="*60)

    # For each flow: look up KB[cluster_id] -> tactic, P(LLM), reasoning
    # If cluster not in KB (new cluster) or low confidence -> selective LLM call
    labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)

    tactics = []
    tactic_ids = []
    p_llm_values = []
    reasonings = []
    llm_call_count = 0

    for i in range(len(X_test)):
        cluster_k = int(assignments[i])
        gmm_conf = float(p_gmm_values[i])
        is_novel = bool(novelty_flags[i])

        # Look up KB
        kb_entry = kb.lookup(cluster_k)

        if kb_entry is not None and gmm_conf >= LLM_CONFIDENCE_THRESHOLD: #and not is_novel:
            # ── Fast path: KB hit with high confidence ──
            tactics.append(kb_entry['tactic'])
            tactic_ids.append(kb_entry.get('tactic_id', 'Unknown'))
            p_llm_values.append(kb_entry['p_llm'])
            reasonings.append(kb_entry.get('reasoning', ''))
        else:
            # ── Slow path: LLM validation needed ──
            # Use cluster summary if available, otherwise build one
            if cluster_k in test_summaries:
                summary = test_summaries[cluster_k]
            else:
                # Single-flow fallback summary
                summary = summarizer.summarize_cluster(
                    cluster_idx=cluster_k,
                    cluster_flows=Z_test[i:i+1],
                    cluster_raw_features=X_test[i:i+1],
                    all_flows=X_test,
                    src_ips=np.array([src_test[i]]),
                    dst_ips=np.array([dst_test[i]]),
                )

            cluster_weight = float(gmm.model.weights_[cluster_k])
            label_result = labeler.label_cluster(summary, cluster_weight)

            tactics.append(label_result['tactic'])
            tactic_ids.append(label_result.get('tactic_id', 'Unknown'))
            p_llm_values.append(label_result['p_llm'])
            reasonings.append(label_result.get('reasoning', ''))
            llm_call_count += 1

            # Update KB with new entry if cluster was unknown
            if kb_entry is None:
                kb.add_entry(
                    cluster_idx=cluster_k,
                    tactic=label_result['tactic'],
                    tactic_id=label_result.get('tactic_id', 'Unknown'),
                    technique_id=label_result.get('technique_id', 'Unknown'),
                    technique_name=label_result.get('technique_name', 'Unknown'),
                    p_gmm=cluster_weight,
                    p_llm=label_result['p_llm'],
                    reasoning=label_result.get('reasoning', ''),
                    summary_text=summary.get('text', ''),
                    fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM],
                )

    p_llm_values = np.array(p_llm_values)

    print(f"  KB lookups (fast path): {len(X_test) - llm_call_count}")
    print(f"  LLM calls (slow path): {llm_call_count}")

    # ═══════════════════════════════════════════════════════════
    # Stage 6: Bayesian Fusion
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 6: Bayesian Fusion")
    print("="*60)

    fusion = BayesianFusion(default_w1=DEFAULT_W1_GMM, default_w2=DEFAULT_W2_LLM)

    p_final_values = []
    final_w1s = []
    final_w2s = []

    for i in range(len(X_test)):
        cluster_k = int(assignments[i])

        # Get per-cluster fusion weights from KB
        w1, w2 = kb.get_fusion_weights(cluster_k)

        # P_final = w1 * P(GMM) + w2 * P(LLM)
        p_final = fusion.fuse(
            p_gmm=float(p_gmm_values[i]),
            p_llm=float(p_llm_values[i]),
            w1=w1, w2=w2
        )

        p_final_values.append(p_final)
        final_w1s.append(w1)
        final_w2s.append(w2)

    p_final_values = np.array(p_final_values)

    print(f"  P_final range: [{p_final_values.min():.4f}, "
          f"{p_final_values.max():.4f}]")
    print(f"  P_final mean:  {p_final_values.mean():.4f}")

    # ═══════════════════════════════════════════════════════════
    # Stage 7: Output Prediction with Reasoning
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 7: Output Prediction with Reasoning")
    print("="*60)

    # Build results DataFrame
    results = pd.DataFrame({
        'src_ip': src_test,
        'dst_ip': dst_test,
        'cluster': assignments,
        'tactic': tactics,
        'tactic_id': tactic_ids,
        'p_gmm': p_gmm_values,
        'p_llm': p_llm_values,
        'p_final': p_final_values,
        'w1_gmm': final_w1s,
        'w2_llm': final_w2s,
        'novelty': novelty_flags,
        'reasoning': reasonings,
    })

    # Add ground truth if available
    if y_test is not None:
        results['true_attack'] = y_test

        # Map attack labels to ground truth tactics
        mapper = GroundTruthMapper('bot_iot')
        true_tactics = mapper.map_labels(y_test)
        results['true_tactic'] = true_tactics

        mapper.print_mapping()

    # ── Print summary statistics ──
    print(f"\n  Total flows classified: {len(results)}")
    print(f"\n  Tactic distribution:")
    for tactic, count in Counter(tactics).most_common():
        pct = count / len(tactics) * 100
        avg_conf = results[results['tactic'] == tactic]['p_final'].mean()
        print(f"    {tactic:30s}  {count:6,} flows ({pct:5.1f}%)  "
              f"avg_conf={avg_conf:.3f}")

    if novelty_flags.sum() > 0:
        print(f"\n  Novel/zero-day candidates: {novelty_flags.sum()} flows")

    # ── Tactic-level evaluation (if ground truth available) ──
    if y_test is not None:
        eval_dir = os.path.join(CHECKPOINT_DIR, "evaluation")
        evaluator = TacticEvaluator(tactic_labels=mapper.get_tactic_list())

        eval_metrics = evaluator.evaluate(
            true_tactics=true_tactics,
            pred_tactics=np.array(tactics),
            cluster_assignments=assignments,
            Z=Z_test,
            save_dir=eval_dir,
            prefix="inference_",
        )

    # ── Save results ──
    output_path = os.path.join(CHECKPOINT_DIR, "inference_results.csv")
    results.to_csv(output_path, index=False)
    print(f"\n  Results saved: {output_path}")
    #
    # # ── Sample predictions ──
    # print(f"\n  Sample predictions (first 5):")
    # sample_cols = ['src_ip', 'dst_ip', 'tactic', 'p_gmm', 'p_llm',
    #                'p_final', 'novelty']
    # if y_test is not None:
    #     sample_cols.extend(['true_attack', 'true_tactic'])
    # print(results[sample_cols].head().to_string(index=False))

    print("\n" + "="*60)
    print("INFERENCE PIPELINE COMPLETE")
    print("="*60)

    return results


if __name__ == '__main__':
    results = run_inference()