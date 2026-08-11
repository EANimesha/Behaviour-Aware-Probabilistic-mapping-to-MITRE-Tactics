"""
Cyber Threat Detection System — Training Pipeline
===================================================

Stage 1: Load & preprocess network flow data
Stage 2: Graph construction + temporal sequences
Stage 3: Three-track encoder + VAE (TRAINED)
Stage 4: Fit DP-GMM (unsupervised) + Cluster Summarization
Stage 5: LLM Semantic Labeling -> Knowledge Base
Stage 6: Bayesian Fusion Setup
Stage 7: Save Model Checkpoint for Inference
"""

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.config import (
    DATA_PATH, SAMPLE_SIZE,
    FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
    Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
    VAE_EPOCHS, VAE_BATCH_SIZE, VAE_LR, #VAE_BETA, VAE_BETA_WARMUP,
    GMM_MAX_COMPONENTS, GMM_COVARIANCE_TYPE,
    DEFAULT_W1_GMM, DEFAULT_W2_LLM,
    OPENAI_API_KEY, LLM_MODEL,
    CHECKPOINT_DIR, KB_PATH, MODEL_PATH,
)
from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.behaviour_scorer import BehaviourScorer
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
# from src.mapping.behaviour_mapping import BehaviourMapper
from src.streaming.streaming_gmm import StreamingDPGMM
from src.clustering.summarizer import ClusterSummarizer
from src.llm.reasoning import LLMLabeler
from src.knowledge_base.kb import KnowledgeBase
from src.fusion.fusion import BayesianFusion
from src.utils.training_logger import TrainingLogger, compute_grad_norms


# def train_ae(joint_enc, graph, X_train_tensor, node_features,
#              context_embeddings, temporal_seqs, temporal_mask,
#              epochs, batch_size, lr, device):
#     """Stage 3: Train the AE with reconstruction loss only.
#
#     No KL divergence, no beta warmup — just MSE reconstruction.
#     """
#     print("\n" + "=" * 60)
#     print("STAGE 3: Training Autoencoder (Three-Track Encoder)")
#     print("=" * 60)
#
#     joint_enc.to(device)
#     joint_enc.train()
#
#     optimizer = torch.optim.Adam(joint_enc.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#
#     n_samples = len(X_train_tensor)
#     n_batches = (n_samples + batch_size - 1) // batch_size
#
#     for epoch in range(epochs):
#         perm = torch.randperm(n_samples)
#         epoch_loss = 0.0
#
#         for batch_idx in range(n_batches):
#             start = batch_idx * batch_size
#             end = min(start + batch_size, n_samples)
#             batch_perm = perm[start:end]
#
#             batch_features = X_train_tensor[batch_perm]
#             batch_context = context_embeddings[batch_perm]
#             batch_temporal = temporal_seqs[batch_perm]
#             batch_mask = temporal_mask[batch_perm]
#
#             # Forward pass (AE: returns z, h_fusion, h_recon)
#             z, h_fusion, h_recon = joint_enc(
#                 graph=graph,
#                 flow_features=batch_features,
#                 node_features=node_features,
#                 context_embeddings=batch_context,
#                 temporal_sequences=batch_temporal,
#                 temporal_mask=batch_mask,
#             )
#
#             # Compute loss (MSE only)
#             loss = joint_enc.compute_loss(h_fusion, h_recon)
#
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(joint_enc.parameters(), max_norm=1.0)
#             optimizer.step()
#
#             epoch_loss += loss.item()
#
#         scheduler.step()
#         avg_loss = epoch_loss / n_batches
#
#         if (epoch + 1) % 5 == 0 or epoch == 0:
#             print(f"  Epoch {epoch + 1:3d}/{epochs}  |  "
#                   f"Recon MSE: {avg_loss:.6f}  |  "
#                   f"LR: {scheduler.get_last_lr()[0]:.6f}")
#
#     print(f"  AE training complete. Final MSE: {avg_loss:.6f}")
#     return joint_enc

def train_ae(joint_enc, graph, X_train_tensor, node_features,
             context_embeddings, temporal_seqs, temporal_mask,
             behaviour_targets, epochs, batch_size, lr, device, y_labels= None):
    """Train behaviour-aware AE: L = L_recon + lambda * L_behaviour."""
    print("\n" + "=" * 60)
    print("STAGE 4: Training Behaviour-Aware Autoencoder")
    print("=" * 60)

    logger = TrainingLogger(save_dir=os.path.join(CHECKPOINT_DIR, 'training_curves'))
    joint_enc.to(device)
    joint_enc.train()
    optimizer = torch.optim.Adam(joint_enc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(X_train_tensor)
    n_batches = (n + batch_size - 1) // batch_size

    for epoch in range(epochs):
        perm = torch.randperm(n)
        ep_total, ep_recon, ep_bhv = 0.0, 0.0, 0.0

        for b in range(n_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, n)
            bp = perm[s:e]

            z, h_fusion, h_recon, pred_bhv = joint_enc(
                graph=graph, flow_features=X_train_tensor[bp],
                node_features=node_features,
                context_embeddings=context_embeddings[bp],
                temporal_sequences=temporal_seqs[bp],
                temporal_mask=temporal_mask[bp])

            loss, recon_l, bhv_l = joint_enc.compute_loss(
                h_fusion, h_recon, pred_bhv, behaviour_targets[bp])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(joint_enc.parameters(), 1.0)
            optimizer.step()

            ep_total += loss.item()
            ep_recon += recon_l.item()
            ep_bhv += bhv_l.item()

        scheduler.step()
        at, ar, ab = ep_total / n_batches, ep_recon / n_batches, ep_bhv / n_batches

    #     if (epoch + 1) % 5 == 0 or epoch == 0:
    #         print(f"  Epoch {epoch + 1:3d}/{epochs} | "
    #               f"Total: {at:.6f} | Recon: {ar:.6f} | "
    #               f"Behaviour: {ab:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    #
    # print(f"  Done. total={at:.6f}, recon={ar:.6f}, behaviour={ab:.6f}")
    # return joint_enc
        current_lr = scheduler.get_last_lr()[0]
        gnorms = compute_grad_norms(joint_enc)
        logger.log_epoch(epoch + 1, at, ar, ab, current_lr, gnorms)

        sil_interval = max(1, epochs // 6)
        if y_labels is not None and ((epoch + 1) % sil_interval == 0 or epoch == 0):
            joint_enc.eval()
            with torch.no_grad():
                all_z = []
                for si in range(0, n, batch_size):
                    ei = min(si + batch_size, n)
                    zc = joint_enc.get_embeddings(
                        graph, X_train_tensor[si:ei], node_features,
                        context_embeddings[si:ei], temporal_seqs[si:ei],
                        temporal_mask[si:ei])
                    all_z.append(zc.cpu())
                Z_check = torch.cat(all_z).numpy()
            sil = logger.log_silhouette(epoch + 1, Z_check, y_labels)
            joint_enc.train()
            sil_str = f" | Sil: {sil:.4f}" if sil else ""
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Total: {at:.6f} | "
                  f"Recon: {ar:.6f} | Bhv: {ab:.6f}{sil_str}")
        elif (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Total: {at:.6f} | "
                  f"Recon: {ar:.6f} | Bhv: {ab:.6f}")

    # logger.plot_all()
    print(f"  Done. total={at:.6f}")
    return joint_enc

def extract_embeddings(joint_enc, graph, X_tensor, node_features,
                       context_embeddings, temporal_seqs, temporal_mask,
                       batch_size, device):
    """Extract latent embeddings Z using trained encoder (mu, no noise)."""
    joint_enc.eval()
    joint_enc.to(device)

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
    print(f"  Extracted embeddings: {Z.shape}")
    return Z


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ═══════════════════════════════════════════════════════════
    # Stage 1: Load & Preprocess Data
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 1: Load & Preprocess Data")
    print("="*60)

    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    preprocessor = Preprocessor()
    data = preprocessor.preprocess_dataset(df)

    X_train = data['X_train']
    src_train, dst_train = data['src_train'], data['dst_train']
    y_train = data['y_train']  # Only for evaluation, not training
    feature_dim = X_train.shape[1]

    # print(f"  Features: {feature_dim}, Training samples: {len(X_train)}")
    train_indices = data.get('train_indices', None)

    # Get raw DataFrame for training flows (before preprocessing)
    if train_indices is not None:
        df_train_raw = df.iloc[train_indices].reset_index(drop=True)
    else:
        # Fallback: use first N rows
        df_train_raw = df.iloc[:len(X_train)].reset_index(drop=True)

    print(f"  Features: {feature_dim}, Samples: {len(X_train)}")
    print(f"  Raw DataFrame: {len(df_train_raw)} rows")

    # ═══════════════════════════════════════════════════════════
    # Stage 2: Graph Construction + Temporal Sequences
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 2: Graph Construction + Temporal Sequences")
    print("="*60)

    builder = NetworkGraphBuilder(seq_length=SEQ_LENGTH)
    graph, temporal_seqs = builder.build_graph(X_train, src_train, dst_train)

    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    # Pre-compute context embeddings (GraphSAGE node -> flow-level src||dst)
    context_enc = ContextEncoder(
        NODE_FEATURE_DIM, hidden_dim=32,
        output_dim=CONTEXT_OUTPUT_DIM, dropout=0.1
    ).to(device)

    with torch.no_grad():
        node_embeddings = context_enc(graph, node_features)
        context_embeddings = builder.get_ip_embeddings(
            node_embeddings, src_train, dst_train
        ).to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    temporal_mask = torch.ones(
        len(X_train), temporal_seqs.shape[1]
    ).to(device)

    print(f"  Graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    print(f"  Temporal sequences: {temporal_seqs.shape}")
    print(f"  Context embeddings: {context_embeddings.shape}")

    # ═══════════════════════════════════════════════════════════
    # Stage 3: Three-Track Encoder + VAE (TRAINING)
    # ═══════════════════════════════════════════════════════════
    feature_enc = FeatureEncoder(feature_dim, output_dim=FEATURE_OUTPUT_DIM)
    behaviour_enc = TransformerBehaviourEncoder(
        feature_dim, hidden_dim=64,
        output_dim=BEHAVIOUR_OUTPUT_DIM,
        num_layers=2, num_heads=4, dropout=0.1
    )

    # Type 1
    # joint_enc = JointEncoder(
    #     feature_encoder=feature_enc,
    #     context_encoder=context_enc,
    #     behaviour_encoder=behaviour_enc,
    #     feature_output_dim=FEATURE_OUTPUT_DIM,
    #     context_output_dim=CONTEXT_OUTPUT_DIM,
    #     behaviour_output_dim=BEHAVIOUR_OUTPUT_DIM,
    #     z_dim=Z_DIM,
    #     beta=VAE_BETA,
    # )

    # # Train the VAE
    # joint_enc = train_vae(
    #     joint_enc, graph, X_train_tensor, node_features,
    #     context_embeddings, temporal_seqs, temporal_mask,
    #     epochs=VAE_EPOCHS, batch_size=VAE_BATCH_SIZE,
    #     lr=VAE_LR, beta_max=VAE_BETA, beta_warmup=VAE_BETA_WARMUP,
    #     device=device,
    # )

    #type 2 with simple AE instead VAE
    # joint_enc = JointEncoder(
    #     feature_encoder=feature_enc,
    #     context_encoder=context_enc,
    #     behaviour_encoder=behaviour_enc,
    #     feature_output_dim=FEATURE_OUTPUT_DIM,
    #     context_output_dim=CONTEXT_OUTPUT_DIM,
    #     behaviour_output_dim=BEHAVIOUR_OUTPUT_DIM,
    #     z_dim=Z_DIM,
    # )
    # # Train the AE
    # joint_enc = train_ae(
    #     joint_enc, graph, X_train_tensor, node_features,
    #     context_embeddings, temporal_seqs, temporal_mask,
    #     epochs=VAE_EPOCHS, batch_size=VAE_BATCH_SIZE,
    #     lr=VAE_LR, device=device,
    # )

    # ═══════════════════════════════════════════════════════
    # Stage 3: Compute Behaviour Scores (heuristic targets)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 3: Compute Behaviour Scores (training targets)")
    print("=" * 60)

    scorer = BehaviourScorer(feature_names=data['feature_names'])
    fan_out, fan_in, n_peers = scorer.compute_graph_features(src_train, dst_train)
    behaviour_scores = scorer.compute(
        X_train, fan_out=fan_out, fan_in=fan_in, n_peers=n_peers)
    behaviour_targets = torch.tensor(
        behaviour_scores, dtype=torch.float32).to(device)

    # ═══════════════════════════════════════════════════════
    # Stage 4: Train Behaviour-Aware AE
    # ═══════════════════════════════════════════════════════
    feature_enc = FeatureEncoder(feature_dim, output_dim=FEATURE_OUTPUT_DIM)
    behaviour_enc = TransformerBehaviourEncoder(
        feature_dim, hidden_dim=64, output_dim=BEHAVIOUR_OUTPUT_DIM,
        num_layers=2, num_heads=4, dropout=0.1)

    joint_enc = JointEncoder(
        feature_encoder=feature_enc, context_encoder=context_enc,
        behaviour_encoder=behaviour_enc,
        feature_output_dim=FEATURE_OUTPUT_DIM,
        context_output_dim=CONTEXT_OUTPUT_DIM,
        behaviour_output_dim=BEHAVIOUR_OUTPUT_DIM,
        z_dim=Z_DIM, n_behaviours=scorer.n_scores, lambda_bhv=1.0)

    joint_enc = train_ae(
        joint_enc, graph, X_train_tensor, node_features,
        context_embeddings, temporal_seqs, temporal_mask,
        behaviour_targets,
        epochs=VAE_EPOCHS, batch_size=VAE_BATCH_SIZE,
        lr=VAE_LR, device=device, y_labels=y_train)
    ###########

    # # Extract final embeddings (using mu, no noise)
    # Z_train = extract_embeddings(
    #     joint_enc, graph, X_train_tensor, node_features,
    #     context_embeddings, temporal_seqs, temporal_mask,
    #     batch_size=VAE_BATCH_SIZE, device=device,
    # )
    #
    # # ═══════════════════════════════════════════════════════════
    # # Stage 4: Fit DP-GMM (Unsupervised) + Cluster Summarization
    # # ═══════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("STAGE 4: Fit DP-GMM + Cluster Summarization")
    # print("=" * 60)
    #
    # gmm = StreamingDPGMM(
    #     max_components=GMM_MAX_COMPONENTS,
    #     covariance_type=GMM_COVARIANCE_TYPE,
    # )
    # cluster_assignments, log_liks = gmm.fit(Z_train)
    #
    # active_clusters = np.unique(cluster_assignments)
    # print(f"  Active clusters: {len(active_clusters)}")
    # print(f"  Log-likelihood range: [{log_liks.min():.2f}, {log_liks.max():.2f}]")
    #
    # # Cluster sizes
    # for k in active_clusters:
    #     count = (cluster_assignments == k).sum()
    #     weight = gmm.model.weights_[k]
    #     print(f"    Cluster {k}: {count:,} flows (weight={weight:.4f})")
    #
    # # Cluster Summarization Module
    # summarizer = ClusterSummarizer(
    #     feature_names=data["feature_names"],
    #     scaler=data["scaler"],
    # )
    # cluster_summaries = summarizer.summarize_all_clusters(
    #     cluster_assignments, Z_train, X_train,
    #     src_ips=src_train, dst_ips=dst_train,
    # )
    #
    # # ═══════════════════════════════════════════════════════════
    # # Stage 5: LLM Semantic Labeling -> Knowledge Base
    # # ═══════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("STAGE 5: LLM Semantic Labeling -> Knowledge Base")
    # print("=" * 60)
    #
    # labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)
    #
    # # Prepare cluster weights for the labeler
    # cluster_weights = {
    #     k: float(gmm.model.weights_[k]) for k in active_clusters
    # }
    #
    # # Label all clusters
    # labels = labeler.label_all_clusters(cluster_summaries, cluster_weights)
    #
    # # Populate Knowledge Base
    # kb = KnowledgeBase()
    # for k in active_clusters:
    #     if k not in labels:
    #         continue
    #     label = labels[k]
    #     summary = cluster_summaries.get(k, {})
    #
    #     # kb.add_entry(
    #     #     cluster_idx=int(k),
    #     #     tactic=label['tactic'],
    #     #     tactic_id=label['tactic_id'],
    #     #     # technique_id=label.get('technique_id', 'Unknown'),
    #     #     # technique_name=label.get('technique_name', 'Unknown'),
    #     #     p_gmm=cluster_weights.get(k, 0.0),
    #     #     p_llm=label['p_llm'],
    #     #     reasoning=label['reasoning'],
    #     #     summary_text=summary.get('text', ''),
    #     #     fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM],
    #     # )
    #     kb.add_entry(
    #         cluster_idx=int(k),
    #         tactic=label["primary_tactic"],
    #         tactic_id=label["primary_tactic_id"],
    #         # Soft probabilistic mapping
    #         tactic_probabilities=label.get("tactic_probabilities", []),
    #         # Probabilities
    #         p_gmm=cluster_weights.get(k, 0.0),
    #         p_llm=label["p_llm"],
    #         reasoning=label["reasoning"],
    #         summary_text=summary.get("text", ""),
    #         fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM],
    #     )
    #
    # print(f"\n{kb}")
    #
    # # ═══════════════════════════════════════════════════════════
    # # Stage 6: Bayesian Fusion Setup
    # # ═══════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("STAGE 6: Bayesian Fusion Setup")
    # print("=" * 60)
    #
    # fusion = BayesianFusion(default_w1=DEFAULT_W1_GMM, default_w2=DEFAULT_W2_LLM)
    # print(f"  Default fusion weights: w1(GMM)={DEFAULT_W1_GMM}, "
    #       f"w2(LLM)={DEFAULT_W2_LLM}")
    # print(f"  Formula: P_final = w1 * P(GMM) + w2 * P(LLM)")
    # print(f"  Ready for inference: New flow -> Encode -> Cluster -> "
    #       f"Label -> Confidence")

    # ═══════════════════════════════════════════════════════
    # Stage 5: Extract z + DP-GMM Clustering on z
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 5: Extract Embeddings + DP-GMM Clustering")
    print("=" * 60)

    Z_train = extract_embeddings(
        joint_enc, graph, X_train_tensor, node_features,
        context_embeddings, temporal_seqs, temporal_mask,
        batch_size=VAE_BATCH_SIZE, device=device)
    print(f"  Z_train: {Z_train.shape}")

    gmm = StreamingDPGMM(
        max_components=GMM_MAX_COMPONENTS,
        covariance_type=GMM_COVARIANCE_TYPE)
    cluster_assignments, log_liks = gmm.fit(Z_train)

    active_clusters = np.unique(cluster_assignments)
    print(f"  Active clusters: {len(active_clusters)}")
    for k in active_clusters:
        count = (cluster_assignments == k).sum()
        print(f"    Cluster {k}: {count:,} flows "
              f"(weight={gmm.model.weights_[k]:.4f})")

    # # ═══════════════════════════════════════════════════════
    # # Stage 6: Map Clusters → Behaviours → Tactics
    # # ═══════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("STAGE 6: Cluster → Behaviour → Tactic Mapping")
    # print("=" * 60)
    #
    # mapper = BehaviourMapper(behaviour_threshold=0.55)
    # cluster_map = mapper.map_clusters(cluster_assignments, behaviour_scores)
    # mapper.print_mapping(cluster_map)
    #
    # # Per-flow predictions
    # pred_tactics, pred_behaviours = mapper.get_flow_predictions(
    #     cluster_assignments, cluster_map)
    #
    # # Purity check against ground truth
    # if y_train is not None:
    #     print(f"\n  Per-cluster purity (ground truth):")
    #     for k in active_clusters:
    #         mask = cluster_assignments == k
    #         maj = Counter(y_train[mask]).most_common(1)[0]
    #         cm = cluster_map.get(int(k), {})
    #         bhvs = ', '.join(cm.get('behaviours', []))
    #         tactic = cm.get('primary_tactic', '?')
    #         print(f"    Cluster {k:2d}: {mask.sum():>6,} | "
    #               f"[{bhvs}] -> {tactic:20s} | "
    #               f"True: {maj[0]:15s} ({maj[1] / mask.sum() * 100:.0f}%)")
    #
    # # ═══════════════════════════════════════════════════════
    # # Stage 7: Summarize + Build Knowledge Base
    # # ═══════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("STAGE 7: Cluster Summarization + Knowledge Base")
    # print("=" * 60)
    #
    # summarizer = ClusterSummarizer(
    #     feature_names=data['feature_names'],
    #     scaler=data['scaler'])
    # cluster_summaries = summarizer.summarize_all_clusters(
    #     cluster_assignments, Z_train, X_train,
    #     src_ips=src_train, dst_ips=dst_train)
    #
    # kb = KnowledgeBase()
    # for k in active_clusters:
    #     cm = cluster_map.get(int(k), {})
    #     summary = cluster_summaries.get(k, {})
    #     tactics = cm.get('tactics', [])
    #
    #     primary_tactic_entry = tactics[0] if tactics else {
    #         'tactic': 'Unknown', 'tactic_id': 'Unknown',
    #         'technique_id': 'Unknown', 'technique_name': 'Unknown'}
    #
    #     kb.add_entry(
    #         cluster_idx=int(k),
    #         tactic=primary_tactic_entry.get('tactic', 'Unknown'),
    #         tactic_id=primary_tactic_entry.get('tactic_id', 'Unknown'),
    #         technique_id=primary_tactic_entry.get('technique_id', 'Unknown'),
    #         technique_name=primary_tactic_entry.get('technique_name', 'Unknown'),
    #         behaviour_label=cm.get('primary_behaviour', ''),
    #         behaviour_description=', '.join(cm.get('behaviours', [])),
    #         tactics=tactics,
    #         p_gmm=float(gmm.model.weights_[k]),
    #         p_llm=0.75,  # placeholder — LLM validation can adjust later
    #         reasoning=f"Mapped via predefined behaviour rules: "
    #                   f"{cm.get('primary_behaviour', '')}",
    #         summary_text=summary.get('text', ''),
    #         fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM])
    #
    # print(f"\n{kb}")




    # # ═══════════════════════════════════════════════════════
    # # Stage 6: Cluster Summarization
    # # ═══════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("STAGE 6: Cluster Summarization")
    # print("=" * 60)
    #
    # summarizer = ClusterSummarizer(
    #     feature_names=data['feature_names'],
    #     scaler=data['scaler'])
    # cluster_summaries = summarizer.summarize_all_clusters(
    #     cluster_assignments, Z_train, X_train,
    #     src_ips=src_train, dst_ips=dst_train)
    #
    # # ═══════════════════════════════════════════════════════
    # # Stage 7: LLM Cluster Labeling (behaviour → tactic)
    # # ═══════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("STAGE 7: LLM Cluster Labeling")
    # print("=" * 60)
    #
    # labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)
    # cluster_weights = {k: float(gmm.model.weights_[k]) for k in active_clusters}
    # labels = labeler.label_all_clusters(cluster_summaries, cluster_weights)
    #
    #     # Build Knowledge Base
    # kb = KnowledgeBase()
    # for k in active_clusters:
    #     if k not in labels:
    #             continue
    #     label = labels[k]
    #     summary = cluster_summaries.get(k, {})
    #
    #     kb.add_entry(
    #             cluster_idx=int(k),
    #             tactic=label['tactic'],
    #             tactic_id=label['tactic_id'],
    #             technique_id=label.get('technique_id', 'Unknown'),
    #             technique_name=label.get('technique_name', 'Unknown'),
    #             behaviour_label=label.get('behaviour_label', ''),
    #             behaviour_description=label.get('behaviour_description', ''),
    #             tactics=label.get('tactics', []),
    #             p_gmm=cluster_weights.get(k, 0.0),
    #             p_llm=label['p_llm'],
    #             reasoning=label['reasoning'],
    #             summary_text=summary.get('text', ''),
    #             fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM])
    #
    # print(f"\n{kb}")
    #
    # # ═══════════════════════════════════════════════════════════
    # # Stage 7: Save Model Checkpoint
    # # ═══════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("STAGE 7: Save Model for Inference")
    # print("=" * 60)
    #
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    #
    # # Save model checkpoint
    # checkpoint = {
    #     'joint_encoder_state': joint_enc.state_dict(),
    #     'gmm_model': gmm.model,
    #     'gmm_config': {
    #         'max_components': gmm.max_components,
    #         'covariance_type': gmm.covariance_type,
    #         'weight_threshold': gmm.weight_threshold,
    #         'novelty_threshold': gmm.novelty_threshold,
    #         'll_stats': getattr(gmm, '_ll_stats', {}),
    #     },
    #     'preprocessor_scaler': data['scaler'],
    #     'feature_names': data['feature_names'],
    #     'ip_to_idx': builder.ip_to_idx,
    #     'config': {
    #         'feature_dim': feature_dim,
    #         'feature_output_dim': FEATURE_OUTPUT_DIM,
    #         'context_output_dim': CONTEXT_OUTPUT_DIM,
    #         'behaviour_output_dim': BEHAVIOUR_OUTPUT_DIM,
    #         'z_dim': Z_DIM,
    #         'seq_length': SEQ_LENGTH,
    #         'node_feature_dim': NODE_FEATURE_DIM,
    #     },
    # }
    # torch.save(checkpoint, MODEL_PATH)
    # print(f"  Model checkpoint saved: {MODEL_PATH}")
    #
    # # Save Knowledge Base
    # kb.save(KB_PATH)
    # print(f"  Knowledge Base saved: {KB_PATH}")
    #
    # print("\n" + "=" * 60)
    # print("TRAINING PIPELINE COMPLETE")
    # print("=" * 60)
    # print(f"  Trained VAE encoder ({Z_DIM}d latent space)")
    # print(f"  DP-GMM with {len(active_clusters)} clusters")
    # print(f"  Knowledge Base with {len(kb.entries)} labeled clusters")
    # print(f"  Ready for: New flow -> Encode -> Cluster -> Label -> Confidence")

    # Stage 6: Benign detection (size + feature based)
    print("\n" + "=" * 60)
    print("STAGE 6: Benign Detection")
    print("=" * 60)
    total_flows = len(X_train)
    benign_clusters = set()
    feature_names_list = data['feature_names']
    fn_idx = {n: i for i, n in enumerate(feature_names_list)}

    for k in active_clusters:
            mask = cluster_assignments == k
            proportion = mask.sum() / total_flows
            cluster_scaled = X_train[mask]
            global_mean = X_train.mean(axis=0)
            deviation = np.abs(cluster_scaled.mean(axis=0) - global_mean).mean()

            # Get raw-scale key features
            cluster_mean = cluster_scaled.mean(axis=0)
            min_ttl = None
            tcp_flags = None
            if 'MIN_TTL' in fn_idx:
                min_ttl = (cluster_mean[fn_idx['MIN_TTL']] *
                           data['scaler'].scale_[fn_idx['MIN_TTL']] +
                           data['scaler'].mean_[fn_idx['MIN_TTL']])
            if 'TCP_FLAGS' in fn_idx:
                tcp_flags = (cluster_mean[fn_idx['TCP_FLAGS']] *
                             data['scaler'].scale_[fn_idx['TCP_FLAGS']] +
                             data['scaler'].mean_[fn_idx['TCP_FLAGS']])

            is_benign = False
            reason = ""

            # Rule 1: Any cluster >10% of data with normal TCP flags → benign
            if proportion > 0.10 and tcp_flags is not None and tcp_flags > 8:
                is_benign = True
                reason = f"large ({proportion:.1%}) + normal TCP={tcp_flags:.1f}"

            # Rule 2: Proportion >5% + low deviation + normal TCP
            elif (proportion > 0.05 and deviation < 0.5 and
                  tcp_flags is not None and tcp_flags > 8):
                is_benign = True
                reason = f"moderate ({proportion:.1%}) + low dev={deviation:.3f}"

            # Rule 3: Low MIN_TTL (<30) + normal TCP (>10) + not tiny
            elif (min_ttl is not None and min_ttl < 30 and
                  tcp_flags is not None and tcp_flags > 10 and
                  mask.sum() > 100):
                is_benign = True
                reason = f"low TTL={min_ttl:.1f} + normal TCP={tcp_flags:.1f}"

            if is_benign:
                benign_clusters.add(int(k))
                print(f"  Cluster {k}: {mask.sum():,} ({proportion:.1%}) "
                      f"— BENIGN ({reason})")

    print(f"  Total benign clusters: {len(benign_clusters)}")

    # Stage 7: Summarize + LLM behaviour labeling
    print("\n" + "=" * 60)
    print("STAGE 7: Cluster Summarization + LLM Behaviour Labeling")
    print("=" * 60)
    summarizer = ClusterSummarizer(
            feature_names=data['feature_names'], scaler=data['scaler'])
    cluster_summaries = summarizer.summarize_all_clusters(
        cluster_assignments, Z_train, X_train,
        src_ips=src_train, dst_ips=dst_train,
        df_raw=df_train_raw)

    labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)
    cluster_weights = {k: float(gmm.model.weights_[k]) for k in active_clusters}

    # Label non-benign clusters with LLM; benign clusters get auto-label
    cluster_behaviour_map = {}

    for k in active_clusters:
            k = int(k)
            if k in benign_clusters:
                cluster_behaviour_map[k] = {
                    'behaviours': [{'label': 'normal_browsing', 'confidence': 0.95}],
                    'primary_behaviour': 'normal_browsing',
                    'confidence': 0.95,
                    'reasoning': 'Auto-detected: large cluster with low feature deviation',
                }
                print(f"  Cluster {k}: [normal_browsing] (auto-detected benign)")
            else:
                summary = cluster_summaries.get(k, {})
                result = labeler.label_cluster(summary, cluster_weights.get(k, 0))
                cluster_behaviour_map[k] = result
                bhvs = ', '.join(b['label'] for b in result['behaviours'])
                print(f"  Cluster {k}: [{bhvs}] conf={result['confidence']:.2f}")

    # Build KB
    kb = KnowledgeBase()
    for k in active_clusters:
            k = int(k)
            bhv_entry = cluster_behaviour_map.get(k, {})
            summary = cluster_summaries.get(k, {})
            primary_bhv = bhv_entry.get('primary_behaviour', 'normal_browsing')

            # Get primary tactic from rule mapping
            from src.mapping.behaviour_mapping import get_primary_tactic
            primary_tactic = get_primary_tactic(primary_bhv)

            kb.add_entry(
                cluster_idx=k,
                tactic=primary_tactic,
                tactic_id='',
                behaviour_label=primary_bhv,
                behaviour_description=', '.join(
                    b['label'] for b in bhv_entry.get('behaviours', [])),
                p_gmm=cluster_weights.get(k, 0),
                p_llm=bhv_entry.get('confidence', 0.5),
                reasoning=bhv_entry.get('reasoning', ''),
                summary_text=summary.get('text', ''),
                fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM])

    print(f"\n  KB: {len(kb.entries)} entries")

    # Purity check
    if y_train is not None:
            print(f"\n  Per-cluster ground truth:")
            for k in active_clusters:
                mask = cluster_assignments == k
                maj = Counter(y_train[mask]).most_common(1)[0]
                bhv = cluster_behaviour_map.get(int(k), {}).get('primary_behaviour', '?')
                print(f"    Cluster {k:2d}: {mask.sum():>6,} | "
                      f"[{bhv:25s}] | True: {maj[0]:15s} ({maj[1] / mask.sum() * 100:.0f}%)")

    # Stage 8: Save
    print("\n" + "=" * 60)
    print("STAGE 8: Save Checkpoint")
    print("=" * 60)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint = {
            'joint_encoder_state': joint_enc.state_dict(),
            'gmm_model': gmm.model,
            'gmm_config': {
                'max_components': gmm.max_components,
                'covariance_type': gmm.covariance_type,
                'weight_threshold': gmm.weight_threshold,
                'novelty_threshold': gmm.novelty_threshold,
                'll_stats': getattr(gmm, '_ll_stats', {}),
            },
            'preprocessor_scaler': data['scaler'],
            'feature_names': data['feature_names'],
            'ip_to_idx': builder.ip_to_idx,
            'cluster_behaviour_map': cluster_behaviour_map,
            'config': {
                'feature_dim': feature_dim,
                'feature_output_dim': FEATURE_OUTPUT_DIM,
                'context_output_dim': CONTEXT_OUTPUT_DIM,
                'behaviour_output_dim': BEHAVIOUR_OUTPUT_DIM,
                'z_dim': Z_DIM,
                'seq_length': SEQ_LENGTH,
                'node_feature_dim': NODE_FEATURE_DIM,
                'n_behaviours': scorer.n_scores,
                'lambda_bhv': 1.0,
            },
        }
    torch.save(checkpoint, MODEL_PATH)
    kb.save(KB_PATH)
    print(f"  Saved: {MODEL_PATH}, {KB_PATH}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()