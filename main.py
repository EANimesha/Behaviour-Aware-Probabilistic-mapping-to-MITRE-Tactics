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
from src.streaming.streaming_gmm import StreamingDPGMM
from src.clustering.summarizer import ClusterSummarizer
from src.llm.reasoning import LLMLabeler
from src.knowledge_base.kb import KnowledgeBase
from src.fusion.fusion import BayesianFusion


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
             behaviour_targets, epochs, batch_size, lr, device):
    """Stage 3: Train behaviour-aware AE.

    Combined loss: L = L_recon + lambda * L_behaviour
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Training Behaviour-Aware Autoencoder")
    print("=" * 60)

    joint_enc.to(device)
    joint_enc.train()

    optimizer = torch.optim.Adam(joint_enc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_samples = len(X_train_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_bhv = 0.0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            batch_perm = perm[start:end]

            batch_features = X_train_tensor[batch_perm]
            batch_context = context_embeddings[batch_perm]
            batch_temporal = temporal_seqs[batch_perm]
            batch_mask = temporal_mask[batch_perm]
            batch_bhv = behaviour_targets[batch_perm]

            # Forward: returns z, h_fusion, h_recon, pred_behaviour
            z, h_fusion, h_recon, pred_bhv = joint_enc(
                graph=graph,
                flow_features=batch_features,
                node_features=node_features,
                context_embeddings=batch_context,
                temporal_sequences=batch_temporal,
                temporal_mask=batch_mask,
            )

            # Combined loss
            loss, recon_loss, bhv_loss = joint_enc.compute_loss(
                h_fusion, h_recon, pred_bhv, batch_bhv
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(joint_enc.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_bhv += bhv_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_bhv = epoch_bhv / n_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  |  "
                  f"Total: {avg_loss:.6f}  |  "
                  f"Recon: {avg_recon:.6f}  |  "
                  f"Behaviour: {avg_bhv:.6f}  |  "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"  Training complete. Final: total={avg_loss:.6f}, "
          f"recon={avg_recon:.6f}, behaviour={avg_bhv:.6f}")
    return joint_enc

def train_vae(joint_enc, graph, X_train_tensor, node_features,
              context_embeddings, temporal_seqs, temporal_mask,
              epochs, batch_size, lr, beta_max, beta_warmup, device):
    """Stage 3: Train the VAE with reconstruction + KL loss.

    Uses mini-batch training with beta warmup to avoid posterior collapse.
    """
    print("\n" + "="*60)
    print("STAGE 3: Training VAE (Three-Track Encoder)")
    print("="*60)

    joint_enc.to(device)
    joint_enc.train()

    optimizer = torch.optim.Adam(joint_enc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_samples = len(X_train_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Create index dataset for mini-batching
    indices = torch.arange(n_samples)

    for epoch in range(epochs):
        # Beta warmup: linearly increase from 0 to beta_max
        if epoch < beta_warmup:
            current_beta = beta_max * (epoch / beta_warmup)
        else:
            current_beta = beta_max
        joint_enc.beta = current_beta

        # Shuffle indices each epoch
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            batch_perm = perm[start:end]

            # Get batch data
            batch_features = X_train_tensor[batch_perm]
            batch_context = context_embeddings[batch_perm]
            batch_temporal = temporal_seqs[batch_perm]
            batch_mask = temporal_mask[batch_perm]

            # Forward pass
            z, mu, log_var, h_fusion, h_recon = joint_enc(
                graph=graph,
                flow_features=batch_features,
                node_features=node_features,
                context_embeddings=batch_context,
                temporal_sequences=batch_temporal,
                temporal_mask=batch_mask,
            )

            # Compute loss
            loss, recon_loss, kl_loss = joint_enc.compute_loss(
                h_fusion, h_recon, mu, log_var
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(joint_enc.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  |  "
                  f"Loss: {avg_loss:.4f}  |  "
                  f"Recon: {avg_recon:.4f}  |  "
                  f"KL: {avg_kl:.4f}  |  "
                  f"Beta: {current_beta:.3f}  |  "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"  VAE training complete. Final loss: {avg_loss:.4f}")
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

    print(f"  Features: {feature_dim}, Training samples: {len(X_train)}")

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

    # ── Compute behaviour target scores (unsupervised) ──
    print("\n  Computing behaviour target scores...")
    scorer = BehaviourScorer(feature_names=data['feature_names'])
    fan_out, fan_in, n_peers = scorer.compute_graph_features(src_train, dst_train)
    behaviour_scores = scorer.compute(
        X_train, fan_out=fan_out, fan_in=fan_in, n_peers=n_peers
    )
    behaviour_targets = torch.tensor(
        behaviour_scores, dtype=torch.float32
    ).to(device)

    joint_enc = JointEncoder(
        feature_encoder=feature_enc,
        context_encoder=context_enc,
        behaviour_encoder=behaviour_enc,
        feature_output_dim=FEATURE_OUTPUT_DIM,
        context_output_dim=CONTEXT_OUTPUT_DIM,
        behaviour_output_dim=BEHAVIOUR_OUTPUT_DIM,
        z_dim=Z_DIM,
        n_behaviours=scorer.n_scores,
        lambda_bhv=1.0,
    )

    # Train the behaviour-aware AE
    joint_enc = train_ae(
        joint_enc, graph, X_train_tensor, node_features,
        context_embeddings, temporal_seqs, temporal_mask,
        behaviour_targets,
        epochs=VAE_EPOCHS, batch_size=VAE_BATCH_SIZE,
        lr=VAE_LR, device=device,
    )

    ###########

    # Extract final embeddings (using mu, no noise)
    Z_train = extract_embeddings(
        joint_enc, graph, X_train_tensor, node_features,
        context_embeddings, temporal_seqs, temporal_mask,
        batch_size=VAE_BATCH_SIZE, device=device,
    )

    # ═══════════════════════════════════════════════════════════
    # Stage 4: Fit DP-GMM (Unsupervised) + Cluster Summarization
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 4: Fit DP-GMM + Cluster Summarization")
    print("=" * 60)

    gmm = StreamingDPGMM(
        max_components=GMM_MAX_COMPONENTS,
        covariance_type=GMM_COVARIANCE_TYPE,
    )
    cluster_assignments, log_liks = gmm.fit(Z_train)

    active_clusters = np.unique(cluster_assignments)
    print(f"  Active clusters: {len(active_clusters)}")
    print(f"  Log-likelihood range: [{log_liks.min():.2f}, {log_liks.max():.2f}]")

    # Cluster sizes
    for k in active_clusters:
        count = (cluster_assignments == k).sum()
        weight = gmm.model.weights_[k]
        print(f"    Cluster {k}: {count:,} flows (weight={weight:.4f})")

    # Cluster Summarization Module
    summarizer = ClusterSummarizer(
        feature_names=data["feature_names"],
        scaler=data["scaler"],
    )
    cluster_summaries = summarizer.summarize_all_clusters(
        cluster_assignments, Z_train, X_train,
        src_ips=src_train, dst_ips=dst_train,
    )

    # ═══════════════════════════════════════════════════════════
    # Stage 5: LLM Semantic Labeling -> Knowledge Base
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 5: LLM Semantic Labeling -> Knowledge Base")
    print("=" * 60)

    labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)

    # Prepare cluster weights for the labeler
    cluster_weights = {
        k: float(gmm.model.weights_[k]) for k in active_clusters
    }

    # Label all clusters
    labels = labeler.label_all_clusters(cluster_summaries, cluster_weights)

    # Populate Knowledge Base
    kb = KnowledgeBase()
    for k in active_clusters:
        if k not in labels:
            continue
        label = labels[k]
        summary = cluster_summaries.get(k, {})

        # kb.add_entry(
        #     cluster_idx=int(k),
        #     tactic=label['tactic'],
        #     tactic_id=label['tactic_id'],
        #     # technique_id=label.get('technique_id', 'Unknown'),
        #     # technique_name=label.get('technique_name', 'Unknown'),
        #     p_gmm=cluster_weights.get(k, 0.0),
        #     p_llm=label['p_llm'],
        #     reasoning=label['reasoning'],
        #     summary_text=summary.get('text', ''),
        #     fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM],
        # )
        kb.add_entry(
            cluster_idx=int(k),
            tactic=label["primary_tactic"],
            tactic_id=label["primary_tactic_id"],
            # Soft probabilistic mapping
            tactic_probabilities=label.get("tactic_probabilities", []),
            # Probabilities
            p_gmm=cluster_weights.get(k, 0.0),
            p_llm=label["p_llm"],
            reasoning=label["reasoning"],
            summary_text=summary.get("text", ""),
            fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM],
        )

    print(f"\n{kb}")

    # ═══════════════════════════════════════════════════════════
    # Stage 6: Bayesian Fusion Setup
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 6: Bayesian Fusion Setup")
    print("=" * 60)

    fusion = BayesianFusion(default_w1=DEFAULT_W1_GMM, default_w2=DEFAULT_W2_LLM)
    print(f"  Default fusion weights: w1(GMM)={DEFAULT_W1_GMM}, "
          f"w2(LLM)={DEFAULT_W2_LLM}")
    print(f"  Formula: P_final = w1 * P(GMM) + w2 * P(LLM)")
    print(f"  Ready for inference: New flow -> Encode -> Cluster -> "
          f"Label -> Confidence")

    # ═══════════════════════════════════════════════════════════
    # Stage 7: Save Model Checkpoint
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 7: Save Model for Inference")
    print("=" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save model checkpoint
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
        'config': {
            'feature_dim': feature_dim,
            'feature_output_dim': FEATURE_OUTPUT_DIM,
            'context_output_dim': CONTEXT_OUTPUT_DIM,
            'behaviour_output_dim': BEHAVIOUR_OUTPUT_DIM,
            'z_dim': Z_DIM,
            'seq_length': SEQ_LENGTH,
            'node_feature_dim': NODE_FEATURE_DIM,
        },
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"  Model checkpoint saved: {MODEL_PATH}")

    # Save Knowledge Base
    kb.save(KB_PATH)
    print(f"  Knowledge Base saved: {KB_PATH}")

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Trained VAE encoder ({Z_DIM}d latent space)")
    print(f"  DP-GMM with {len(active_clusters)} clusters")
    print(f"  Knowledge Base with {len(kb.entries)} labeled clusters")
    print(f"  Ready for: New flow -> Encode -> Cluster -> Label -> Confidence")


if __name__ == '__main__':
    main()