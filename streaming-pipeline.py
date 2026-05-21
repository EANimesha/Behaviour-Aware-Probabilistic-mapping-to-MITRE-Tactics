"""
Cyber Threat Detection System — Streaming Pipeline
=====================================================

Processes a continuous flow stream in FIFO windows of W flows each.
Adapts the DP-GMM model online, detects drift, re-labels clusters
when needed, and adjusts fusion weights dynamically.

Stage 1: Window Buffer (FIFO) — buffer W flows, preprocess
Stage 2: Graph Construction — DGL graph + temporal sequences per window
Stage 3: Three-Track Encoder + VAE — encode all W flows -> Z_window (W, 64D)
Stage 4: Update P(GMM) Model (Streaming DP-GMM Fit) — online EM + batch predict
Stage 5: LLM Semantic Labeling KB — consistency check, re-label if new/drifted
Stage 6: Drift Detection + Adaptive Fusion — adjust w1/w2 per cluster
Stage 7: Output Prediction with Reasoning — tactic, confidence, explanation
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import logging
from collections import Counter

from src.config.config import (
    DATA_PATH, SAMPLE_SIZE,
    FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
    Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
    VAE_BATCH_SIZE, VAE_BETA,
    DEFAULT_W1_GMM, DEFAULT_W2_LLM,
    OPENAI_API_KEY, LLM_MODEL, LLM_CONFIDENCE_THRESHOLD,
    CHECKPOINT_DIR, KB_PATH, MODEL_PATH,
    WINDOW_SIZE, STREAMING_LR,
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# Model Loading (shared with inference)
# ══════════════════════════════════════════════════════════

def load_checkpoint(model_path, kb_path, device='cpu'):
    """Load trained model checkpoint and Knowledge Base."""
    print("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    feature_enc = FeatureEncoder(
        cfg['feature_dim'], output_dim=cfg['feature_output_dim']
    )
    context_enc = ContextEncoder(
        cfg['node_feature_dim'], hidden_dim=32,
        output_dim=cfg['context_output_dim'], dropout=0.0
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
        beta=VAE_BETA,
    )

    joint_enc.load_state_dict(checkpoint['joint_encoder_state'])
    joint_enc.to(device)
    joint_enc.eval()

    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        max_components=gmm_cfg['max_components'],
        covariance_type=gmm_cfg['covariance_type'],
        weight_threshold=gmm_cfg['weight_threshold'],
        novelty_threshold=gmm_cfg['novelty_threshold'],
        learning_rate=STREAMING_LR,
    )
    gmm.model = checkpoint['gmm_model']

    kb = KnowledgeBase()
    kb.load(kb_path)

    active = np.sum(gmm.model.weights_ > gmm_cfg['weight_threshold'])
    print(f"  Encoder loaded (z_dim={cfg['z_dim']})")
    print(f"  DP-GMM loaded ({active} active clusters)")
    print(f"  KB loaded ({len(kb.entries)} entries)")

    return joint_enc, gmm, kb, checkpoint


def extract_embeddings(joint_enc, graph, X_tensor, node_features,
                       context_embeddings, temporal_seqs, temporal_mask,
                       batch_size, device):
    """Extract Z using trained encoder (mu, no noise)."""
    joint_enc.eval()
    n = len(X_tensor)
    all_z = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            z = joint_enc.get_embeddings(
                graph=graph,
                flow_features=X_tensor[s:e],
                node_features=node_features,
                context_embeddings=context_embeddings[s:e],
                temporal_sequences=temporal_seqs[s:e],
                temporal_mask=temporal_mask[s:e],
            )
            all_z.append(z.cpu())
    return torch.cat(all_z, dim=0).numpy()


# ══════════════════════════════════════════════════════════
# Window Buffer (Stage 1)
# ══════════════════════════════════════════════════════════

class WindowBuffer:
    """FIFO buffer that accumulates flows until window_size is reached.

    Simulates a continuous flow stream by chunking a DataFrame into
    windows of size W. In production, this would read from a network
    tap, Kafka topic, or PCAP stream.
    """

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.buffer = []

    def fill_from_dataframe(self, df):
        """Yield windows of rows from a DataFrame (simulation mode).

        Args:
            df: full DataFrame of flows to stream through

        Yields:
            window_df: DataFrame of window_size rows
            window_idx: window number (0-based)
        """
        n = len(df)
        n_windows = (n + self.window_size - 1) // self.window_size

        for i in range(n_windows):
            start = i * self.window_size
            end = min(start + self.window_size, n)
            window_df = df.iloc[start:end].copy().reset_index(drop=True)
            yield window_df, i

    def fill_from_buffer(self, flow_records):
        """Add individual flow records; yield when buffer is full.

        Args:
            flow_records: iterable of flow dicts

        Yields:
            window_df: DataFrame when buffer reaches window_size
        """
        for record in flow_records:
            self.buffer.append(record)
            if len(self.buffer) >= self.window_size:
                window_df = pd.DataFrame(self.buffer[:self.window_size])
                self.buffer = self.buffer[self.window_size:]
                yield window_df


# ══════════════════════════════════════════════════════════
# Streaming Pipeline Processor
# ══════════════════════════════════════════════════════════

class StreamingProcessor:
    """Runs the 7-stage streaming pipeline per window.

    Holds all stateful components (encoder, GMM, KB, fusion)
    and processes windows sequentially.
    """

    def __init__(self, joint_enc, gmm, kb, checkpoint, device='cpu',
                 drift_threshold=0.15):
        self.joint_enc = joint_enc
        self.gmm = gmm
        self.kb = kb
        self.checkpoint = checkpoint
        self.device = device
        self.drift_threshold = drift_threshold

        self.cfg = checkpoint['config']
        self.saved_scaler = checkpoint['preprocessor_scaler']
        self.saved_feature_names = checkpoint['feature_names']

        # Preprocessor with saved scaler
        self.preprocessor = Preprocessor()
        self.preprocessor.scaler = self.saved_scaler
        self.preprocessor.fitted = True
        self.preprocessor.final_feature_names = self.saved_feature_names

        # Supporting modules
        self.summarizer = ClusterSummarizer(feature_names=self.saved_feature_names)
        self.labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)
        self.fusion = BayesianFusion(
            default_w1=DEFAULT_W1_GMM, default_w2=DEFAULT_W2_LLM
        )
        self.builder = NetworkGraphBuilder(seq_length=self.cfg['seq_length'])

        # Statistics
        self.windows_processed = 0
        self.total_flows_processed = 0
        self.total_drift_events = 0
        self.total_new_clusters = 0
        self.total_llm_calls = 0

    def process_window(self, window_df, window_idx=0):
        """Process a single window through all 7 stages.

        Args:
            window_df: DataFrame of raw flows (window_size rows)
            window_idx: window number for logging

        Returns:
            results: DataFrame with per-flow predictions for this window
        """
        W = len(window_df)
        t_start = time.time()

        print(f"\n{'─'*60}")
        print(f"WINDOW {window_idx} ({W} flows)")
        print(f"{'─'*60}")

        # ── Stage 1: Preprocess ──
        X_window, src_ips, dst_ips = self.preprocessor.transform_new(window_df)
        y_true = (window_df['Attack'].values
                  if 'Attack' in window_df.columns else None)

        # ── Stage 2: Graph Construction ──
        graph, temporal_seqs = self.builder.build_graph(
            X_window, src_ips, dst_ips
        )
        graph = graph.to(self.device)
        temporal_seqs = temporal_seqs.to(self.device)
        node_features = self.builder.node_features.to(self.device)

        # ── Stage 3: Three-Track Encoder + VAE ──
        context_enc = self.joint_enc.context_encoder
        with torch.no_grad():
            node_emb = context_enc(graph, node_features)
            context_emb = self.builder.get_ip_embeddings(
                node_emb, src_ips, dst_ips
            ).to(self.device)

        X_tensor = torch.tensor(X_window, dtype=torch.float32).to(self.device)
        t_mask = torch.ones(W, temporal_seqs.shape[1]).to(self.device)

        Z_window = extract_embeddings(
            self.joint_enc, graph, X_tensor, node_features,
            context_emb, temporal_seqs, t_mask,
            batch_size=VAE_BATCH_SIZE, device=self.device,
        )
        print(f"  Stage 3: Z_window {Z_window.shape}")

        # ── Stage 4: Update P(GMM) Model (Streaming DP-GMM) ──
        (assignments, soft_probs, log_liks,
         novelty_flags, new_clusters) = self.gmm.update_online_em(Z_window)

        p_gmm_values = soft_probs.max(axis=1)

        print(f"  Stage 4: {len(np.unique(assignments))} clusters, "
              f"{novelty_flags.sum()} novel, "
              f"{len(new_clusters)} new clusters")

        # ── Stage 5: LLM Semantic Labeling KB ──
        #   - Retrieve KB
        #   - Consistency check
        #   - If new cluster or KB mismatch → summarize → LLM label → update KB
        #   - Update cluster semantic mapping
        tactics, tactic_ids, p_llm_values, reasonings = (
            self._stage5_kb_labeling(
                assignments, Z_window, X_window,
                src_ips, dst_ips, p_gmm_values,
                novelty_flags, new_clusters,
            )
        )

        # ── Stage 6: Drift Detection + Adaptive Fusion ──
        drift_detected, drift_magnitude, drift_details = (
            self.gmm.detect_drift(threshold=self.drift_threshold)
        )

        if drift_detected:
            self.total_drift_events += 1
            print(f"  Stage 6: DRIFT DETECTED (magnitude={drift_magnitude:.4f})")
            self._adapt_fusion_weights_for_drift(assignments)
        else:
            print(f"  Stage 6: No drift (magnitude={drift_magnitude:.4f})")

        # Adaptive Fusion: P_final = w1 * P(GMM) + w2 * P(LLM)
        p_final_values, w1_values, w2_values = self._stage6_fusion(
            assignments, p_gmm_values, p_llm_values,
        )

        # ── Stage 7: Output Prediction with Reasoning ──
        results = pd.DataFrame({
            'window': window_idx,
            'src_ip': src_ips,
            'dst_ip': dst_ips,
            'cluster': assignments,
            'tactic': tactics,
            'tactic_id': tactic_ids,
            'p_gmm': p_gmm_values,
            'p_llm': p_llm_values,
            'p_final': p_final_values,
            'w1_gmm': w1_values,
            'w2_llm': w2_values,
            'novelty': novelty_flags,
            'reasoning': reasonings,
        })
        if y_true is not None:
            results['true_attack'] = y_true

        # Update counters
        self.windows_processed += 1
        self.total_flows_processed += W
        self.total_new_clusters += len(new_clusters)

        elapsed = time.time() - t_start
        print(f"  Stage 7: {W} predictions in {elapsed:.2f}s "
              f"({W/elapsed:.0f} flows/sec)")

        # Print tactic summary for this window
        for tactic, count in Counter(tactics).most_common(5):
            avg_conf = np.mean([
                p_final_values[i] for i in range(len(tactics))
                if tactics[i] == tactic
            ])
            print(f"    {tactic:25s} {count:5d} flows  "
                  f"avg_conf={avg_conf:.3f}")

        return results

    # ──────────────────────────────────────────────────────
    # Stage 5 internals
    # ──────────────────────────────────────────────────────

    def _stage5_kb_labeling(self, assignments, Z_window, X_window,
                            src_ips, dst_ips, p_gmm_values,
                            novelty_flags, new_clusters):
        """Stage 5: KB lookup with consistency check and selective re-labeling.

        Flow:
          1. Retrieve KB for each cluster
          2. Consistency check against current window statistics
          3. IF new cluster OR KB mismatch:
               → Summarize cluster
               → LLM labeling
               → Update KB
          4. Update cluster semantic mapping [P(tactic|k)]
        """
        # Identify clusters that need (re-)labeling
        clusters_to_label = set(new_clusters)  # New clusters always need labeling
        window_clusters = np.unique(assignments)

        for k in window_clusters:
            k = int(k)
            entry = self.kb.lookup(k)
            if entry is None:
                # Cluster not in KB
                clusters_to_label.add(k)
            else:
                # Consistency check
                is_consistent, reason = self.kb.consistency_check(k, None)
                if not is_consistent:
                    clusters_to_label.add(k)
                    logger.info(f"KB inconsistency for cluster {k}: {reason}")

        # Summarize and label clusters that need it
        if clusters_to_label:
            print(f"  Stage 5: Re-labeling {len(clusters_to_label)} clusters: "
                  f"{sorted(clusters_to_label)}")

            # Build summaries for clusters to re-label
            for k in clusters_to_label:
                k = int(k)
                mask = assignments == k
                if mask.sum() < 2:
                    continue

                # Summarize
                summary = self.summarizer.summarize_cluster(
                    cluster_idx=k,
                    cluster_flows=Z_window[mask],
                    cluster_raw_features=X_window[mask],
                    all_flows=X_window,
                    src_ips=src_ips[mask],
                    dst_ips=dst_ips[mask],
                )

                # LLM label
                cluster_weight = float(self.gmm.model.weights_[k])
                label_result = self.labeler.label_cluster(summary, cluster_weight)
                self.total_llm_calls += 1

                # Update KB
                self.kb.add_entry(
                    cluster_idx=k,
                    tactic=label_result['tactic'],
                    tactic_id=label_result.get('tactic_id', 'Unknown'),
                    p_gmm=cluster_weight,
                    p_llm=label_result['p_llm'],
                    reasoning=label_result.get('reasoning', ''),
                    summary_text=summary.get('text', ''),
                    fusion_weights=self.kb.get_fusion_weights(k),
                )
        else:
            print(f"  Stage 5: All {len(window_clusters)} clusters "
                  f"consistent in KB")

        # Now assign labels from KB for every flow
        tactics = []
        tactic_ids = []
        p_llm_values = []
        reasonings = []

        for i in range(len(assignments)):
            k = int(assignments[i])
            entry = self.kb.lookup(k)
            if entry is not None:
                tactics.append(entry['tactic'])
                tactic_ids.append(entry.get('tactic_id', 'Unknown'))
                p_llm_values.append(entry['p_llm'])
                reasonings.append(entry.get('reasoning', ''))
            else:
                tactics.append('Unknown')
                tactic_ids.append('Unknown')
                p_llm_values.append(0.5)
                reasonings.append('Cluster not in KB')

        p_llm_values = np.array(p_llm_values)
        return tactics, tactic_ids, p_llm_values, reasonings

    # ──────────────────────────────────────────────────────
    # Stage 6 internals
    # ──────────────────────────────────────────────────────

    def _adapt_fusion_weights_for_drift(self, assignments):
        """When drift detected: increase w1 (trust GMM more, LLM may be stale).

        Applied to all active clusters in the current window.
        """
        affected_clusters = np.unique(assignments)
        for k in affected_clusters:
            k = int(k)
            current_w1, current_w2 = self.kb.get_fusion_weights(k)
            new_w1, new_w2 = self.fusion.adapt_weights_for_drift(
                current_w1, current_w2, drift_detected=True
            )
            self.kb.update_fusion_weights(k, new_w1, new_w2)

    def _stage6_fusion(self, assignments, p_gmm_values, p_llm_values):
        """Compute P_final per flow using per-cluster adaptive weights."""
        p_final_values = []
        w1_values = []
        w2_values = []

        for i in range(len(assignments)):
            k = int(assignments[i])
            w1, w2 = self.kb.get_fusion_weights(k)
            p_final = self.fusion.fuse(
                p_gmm=float(p_gmm_values[i]),
                p_llm=float(p_llm_values[i]),
                w1=w1, w2=w2,
            )
            p_final_values.append(p_final)
            w1_values.append(w1)
            w2_values.append(w2)

        return np.array(p_final_values), w1_values, w2_values

    # ──────────────────────────────────────────────────────
    # Summary / persistence
    # ──────────────────────────────────────────────────────

    def print_summary(self):
        """Print cumulative streaming statistics."""
        print(f"\n{'='*60}")
        print("STREAMING PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"  Windows processed:    {self.windows_processed}")
        print(f"  Total flows:          {self.total_flows_processed:,}")
        print(f"  Drift events:         {self.total_drift_events}")
        print(f"  New clusters created: {self.total_new_clusters}")
        print(f"  LLM re-labeling calls:{self.total_llm_calls}")
        print(f"  Active clusters:      "
              f"{len(self.gmm.get_active_cluster_indices())}")
        print(f"\n  Knowledge Base:")
        print(f"  {self.kb}")

    def save_state(self):
        """Persist updated KB and GMM state after streaming."""
        updated_kb_path = os.path.join(CHECKPOINT_DIR, "knowledge_base_streaming.json")
        self.kb.save(updated_kb_path)
        print(f"  Updated KB saved: {updated_kb_path}")

        # Save updated GMM checkpoint
        updated_model_path = os.path.join(CHECKPOINT_DIR, "model_streaming.pt")
        checkpoint = self.checkpoint.copy()
        checkpoint['gmm_model'] = self.gmm.model
        torch.save(checkpoint, updated_model_path)
        print(f"  Updated model saved: {updated_model_path}")


# ══════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════

def run_streaming(stream_csv_path=None, model_path=MODEL_PATH,
                  kb_path=KB_PATH, window_size=WINDOW_SIZE,
                  max_windows=None):
    """Run the streaming pipeline.

    Args:
        stream_csv_path: path to CSV to stream through (simulated)
                         if None, uses test split from DATA_PATH
        model_path: path to saved model checkpoint
        kb_path: path to saved Knowledge Base
        window_size: number of flows per window
        max_windows: stop after N windows (None = process all)

    Returns:
        all_results: DataFrame of all predictions across all windows
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load trained components ──
    joint_enc, gmm, kb, checkpoint = load_checkpoint(
        model_path, kb_path, device
    )

    # ── Initialise streaming processor ──
    processor = StreamingProcessor(
        joint_enc=joint_enc,
        gmm=gmm,
        kb=kb,
        checkpoint=checkpoint,
        device=device,
    )

    # ── Load stream data ──
    if stream_csv_path:
        df_stream = load_dataset(stream_csv_path, sample_size=None)
    else:
        # Use test split from training data as simulated stream
        df_full = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
        preprocessor_tmp = Preprocessor()
        data_tmp = preprocessor_tmp.preprocess_dataset(df_full)
        # Get original test indices to reconstruct the raw df
        # For simulation: just use the last portion of the full df
        n_train = len(data_tmp['X_train'])
        df_stream = df_full.iloc[n_train:].reset_index(drop=True)
        print(f"  Simulated stream: {len(df_stream)} flows from test split")

    # ── Stage 1: Window Buffer ──
    buffer = WindowBuffer(window_size=window_size)
    all_results = []

    print(f"\n{'='*60}")
    print(f"STREAMING PIPELINE — window_size={window_size}")
    print(f"{'='*60}")

    for window_df, window_idx in buffer.fill_from_dataframe(df_stream):
        if max_windows is not None and window_idx >= max_windows:
            print(f"\nReached max_windows={max_windows}, stopping.")
            break

        # Process this window through all 7 stages
        results = processor.process_window(window_df, window_idx)
        all_results.append(results)

    # ── Combine all results ──
    all_results = pd.concat(all_results, ignore_index=True)

    # ── Final summary ──
    processor.print_summary()

    # ── Save updated state ──
    processor.save_state()

    # ── Save results ──
    output_path = os.path.join(CHECKPOINT_DIR, "streaming_results.csv")
    all_results.to_csv(output_path, index=False)
    print(f"  Results saved: {output_path}")

    # ── Evaluation against ground truth ──
    if 'true_attack' in all_results.columns:
        print(f"\n  Ground truth evaluation:")
        print(f"    Total flows: {len(all_results):,}")
        print(f"    Unique predicted tactics: "
              f"{all_results['tactic'].nunique()}")
        print(f"    Unique true attacks: "
              f"{all_results['true_attack'].nunique()}")

    print(f"\n{'='*60}")
    print("STREAMING PIPELINE COMPLETE")
    print(f"{'='*60}")

    return all_results


if __name__ == '__main__':
    results = run_streaming()