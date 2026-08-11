"""
Encoder Track Ablation — Joint Training
==========================================

Each configuration trains encoders + AE jointly (same as main.py).
NOT frozen random projections.

Configs:
  A) Feature only (MLP + AE)
  B) Feature + Context (MLP + GraphSAGE + AE)
  C) Feature + Temporal (MLP + Transformer + AE)
  D) All Three (MLP + GraphSAGE + Transformer + AE) — proposed

Usage:
  python ablation_encoder.py bot_iot
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score, normalized_mutual_info_score
from sklearn.mixture import BayesianGaussianMixture
from collections import Counter

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
from src.encoder.behaviour_scorer import BehaviourScorer
from src.evaluation.ground_truth import GroundTruthMapper

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "ablation_results")
os.makedirs(EVAL_DIR, exist_ok=True)

AE_EPOCHS = 30
AE_BATCH = 256
AE_LR = 1e-3


class AblationJointModel(nn.Module):
    """Joint encoder + AE for ablation — trains everything end-to-end."""

    def __init__(self, feature_dim, node_feat_dim, tracks,
                 feat_out=32, ctx_out=32, bhv_out=32, z_dim=64,
                 n_behaviours=5, lambda_bhv=1.0):
        super().__init__()
        self.tracks = tracks
        self.lambda_bhv = lambda_bhv

        # Track encoders (only create what's needed)
        self.feature_enc = FeatureEncoder(feature_dim, feat_out)
        self.norm_f = nn.LayerNorm(feat_out)
        fusion_dim = feat_out

        self.has_context = 'context' in tracks
        self.has_temporal = 'temporal' in tracks

        if self.has_context:
            self.context_enc = ContextEncoder(node_feat_dim, 32, ctx_out, 0.1)
            ctx_concat = ctx_out * 2  # src || dst
            self.norm_c = nn.LayerNorm(ctx_concat)
            fusion_dim += ctx_concat

        if self.has_temporal:
            self.temporal_enc = TransformerBehaviourEncoder(
                feature_dim, 64, bhv_out, 2, 4, 0.1)
            self.norm_b = nn.LayerNorm(bhv_out)
            fusion_dim += bhv_out

        self.fusion_dim = fusion_dim

        # Autoencoder
        hidden = max(128, fusion_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, fusion_dim))
        self.bhv_head = nn.Sequential(
            nn.Linear(z_dim, 32), nn.ReLU(),
            nn.Linear(32, n_behaviours), nn.Sigmoid())

    def forward(self, flow_features, graph=None, node_features=None,
                context_emb=None, temporal_seqs=None, temporal_mask=None,
                bhv_targets=None):
        """Full forward: encode tracks → fuse → AE → loss."""

        # Feature track (always present)
        h_f = self.norm_f(self.feature_enc(flow_features))
        components = [h_f]

        # Context track
        if self.has_context and context_emb is not None:
            h_c = self.norm_c(context_emb)
            components.append(h_c)

        # Temporal track
        if self.has_temporal and temporal_seqs is not None:
            h_b = self.norm_b(self.temporal_enc(temporal_seqs, temporal_mask))
            components.append(h_b)

        # Fuse
        h_fusion = torch.cat(components, dim=1)

        # AE
        z = self.encoder(h_fusion)
        h_recon = self.decoder(z)
        bhv_pred = self.bhv_head(z)

        # Losses
        loss_recon = nn.MSELoss()(h_recon, h_fusion.detach())
        if bhv_targets is not None:
            loss_bhv = nn.MSELoss()(bhv_pred, bhv_targets)
            loss = loss_recon + self.lambda_bhv * loss_bhv
        else:
            loss_bhv = torch.tensor(0.0)
            loss = loss_recon

        return z, loss, loss_recon, loss_bhv


def prepare_data():
    """Load and preprocess dataset."""
    print(f"  Loading {DATASET}...")
    df = load_dataset(os.path.join("../..",DATA_PATH), sample_size=SAMPLE_SIZE)

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

    gt_mapper = GroundTruthMapper(DATASET)
    y_tactic = gt_mapper.map_labels(data['y_train'])

    return data, y_tactic, gt_mapper


def build_graph_data(X, src, dst, device):
    """Build graph and temporal sequences once (shared across configs)."""
    builder = NetworkGraphBuilder(seq_length=SEQ_LENGTH)
    graph, temporal_seqs = builder.build_graph(X, src, dst)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)
    return graph, temporal_seqs, node_features, builder


def train_config(config_name, tracks, data, graph, temporal_seqs,
                  node_features, builder, bhv_targets, device):
    """Train one configuration end-to-end and evaluate."""
    print(f"\n  {'─'*60}")
    print(f"  CONFIG: {config_name} — tracks: {tracks}")
    print(f"  {'─'*60}")

    X = data['X_train']
    y = data['y_train']
    src, dst = data['src_train'], data['dst_train']
    feature_dim = X.shape[1]
    node_feat_dim = node_features.shape[1]

    gt_mapper = GroundTruthMapper(DATASET)
    y_tactic = gt_mapper.map_labels(y)

    # Create model
    model = AblationJointModel(
        feature_dim=feature_dim,
        node_feat_dim=node_feat_dim,
        tracks=tracks,
        feat_out=FEATURE_OUTPUT_DIM,
        ctx_out=CONTEXT_OUTPUT_DIM,
        bhv_out=BEHAVIOUR_OUTPUT_DIM,
        z_dim=Z_DIM,
        n_behaviours=5,
        lambda_bhv=1.0
    ).to(device)

    print(f"    Fusion dim: {model.fusion_dim}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=AE_EPOCHS)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    bhv_gpu = bhv_targets.to(device)
    t_mask = torch.ones(len(X), temporal_seqs.shape[1]).to(device)

    # Pre-compute context embeddings (re-computed each epoch for
    # context encoder training, but expensive — do it per epoch)
    has_context = 'context' in tracks
    has_temporal = 'temporal' in tracks

    # Training loop
    t0 = time.time()
    model.train()

    for epoch in range(AE_EPOCHS):
        epoch_loss, epoch_rec, epoch_bhv = 0, 0, 0
        n_batches = 0
        perm = torch.randperm(len(X_tensor))

        # Recompute context embeddings with current encoder weights
        if has_context:
            with torch.no_grad():
                node_emb = model.context_enc(graph, node_features)
                context_emb = builder.get_ip_embeddings(
                    node_emb, src, dst).to(device)
        else:
            context_emb = None

        for i in range(0, len(X_tensor), AE_BATCH):
            idx = perm[i:i + AE_BATCH]
            batch_x = X_tensor[idx]
            batch_bhv = bhv_gpu[idx]
            batch_ctx = context_emb[idx] if context_emb is not None else None
            batch_seq = temporal_seqs[idx] if has_temporal else None
            batch_mask = t_mask[idx] if has_temporal else None

            z, loss, l_rec, l_bhv = model(
                flow_features=batch_x,
                graph=graph,
                node_features=node_features,
                context_emb=batch_ctx,
                temporal_seqs=batch_seq,
                temporal_mask=batch_mask,
                bhv_targets=batch_bhv)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rec += l_rec.item()
            epoch_bhv += l_bhv.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{AE_EPOCHS}: "
                  f"Total={epoch_loss/n_batches:.4f}, "
                  f"Rec={epoch_rec/n_batches:.4f}, "
                  f"Bhv={epoch_bhv/n_batches:.4f}")

    train_time = time.time() - t0

    # Extract Z
    model.eval()
    if has_context:
        with torch.no_grad():
            node_emb = model.context_enc(graph, node_features)
            context_emb = builder.get_ip_embeddings(
                node_emb, src, dst).to(device)

    all_z = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), AE_BATCH):
            batch_x = X_tensor[i:i + AE_BATCH]
            batch_ctx = context_emb[i:i + AE_BATCH] if context_emb is not None else None
            batch_seq = temporal_seqs[i:i + AE_BATCH] if has_temporal else None
            batch_mask = t_mask[i:i + AE_BATCH] if has_temporal else None

            z, _, _, _ = model(
                flow_features=batch_x,
                context_emb=batch_ctx,
                temporal_seqs=batch_seq,
                temporal_mask=batch_mask)
            all_z.append(z.cpu())

    Z = torch.cat(all_z).numpy()
    print(f"    Z: {Z.shape}, Time: {train_time:.1f}s")

    # Fit DP-GMM
    t0 = time.time()
    gmm = BayesianGaussianMixture(
        n_components=GMM_MAX_COMPONENTS,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        max_iter=200, random_state=42)
    clusters = gmm.fit_predict(Z)
    gmm_time = time.time() - t0
    n_active = len(np.unique(clusters))

    # Evaluate
    purities, sizes = [], []
    for k in np.unique(clusters):
        mask = clusters == k
        if mask.sum() < 2: continue
        maj = Counter(y[mask]).most_common(1)[0]
        purities.append(maj[1] / mask.sum())
        sizes.append(mask.sum())
    mean_purity = np.mean(purities) if purities else 0
    weighted_purity = np.average(purities, weights=sizes) if purities else 0

    nmi_attack = normalized_mutual_info_score(y, clusters)
    nmi_tactic = normalized_mutual_info_score(y_tactic, clusters)

    sample = min(5000, len(Z))
    sil_tactic = silhouette_score(Z, y_tactic, sample_size=sample)
    sil_cluster = silhouette_score(Z, clusters, sample_size=sample)

    # Per-flow accuracy via majority vote
    cluster_to_tactic = {}
    for k in np.unique(clusters):
        mask = clusters == k
        cluster_to_tactic[k] = Counter(y_tactic[mask]).most_common(1)[0][0]
    y_pred = np.array([cluster_to_tactic[c] for c in clusters])
    flow_acc = accuracy_score(y_tactic, y_pred)

    # Top-2
    top2_correct = 0
    for k in np.unique(clusters):
        mask = clusters == k
        tc = Counter(y_tactic[mask])
        top2 = [t for t, _ in tc.most_common(2)]
        for j in np.where(mask)[0]:
            if y_tactic[j] in top2:
                top2_correct += 1
    top2_acc = top2_correct / len(clusters)

    result = {
        'config': config_name,
        'tracks': '+'.join(tracks),
        'fusion_dim': model.fusion_dim,
        'n_clusters': n_active,
        'mean_purity': mean_purity,
        'weighted_purity': weighted_purity,
        'nmi_attack': nmi_attack,
        'nmi_tactic': nmi_tactic,
        'sil_tactic': sil_tactic,
        'sil_cluster': sil_cluster,
        'flow_accuracy': flow_acc,
        'top2_accuracy': top2_acc,
        'train_time': train_time,
        'gmm_time': gmm_time,
    }

    print(f"\n    Results:")
    print(f"      Clusters: {n_active}, Purity: {mean_purity:.4f}")
    print(f"      NMI (attack/tactic): {nmi_attack:.4f}/{nmi_tactic:.4f}")
    print(f"      Silhouette (tactic/cluster): {sil_tactic:.4f}/{sil_cluster:.4f}")
    print(f"      Accuracy: {flow_acc:.4f}, Top-2: {top2_acc:.4f}")

    return result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"ENCODER TRACK ABLATION (Joint Training) — {DATASET}")
    print(f"{'='*70}")

    data, y_tactic, gt_mapper = prepare_data()
    X = data['X_train']
    src, dst = data['src_train'], data['dst_train']
    print(f"  Data: {len(X)} flows, {X.shape[1]} features")

    print("  Building graph and temporal structures...")
    graph, temporal_seqs, node_features, builder = build_graph_data(
        X, src, dst, device)

    print("  Computing behaviour targets...")
    scorer = BehaviourScorer(feature_names=[])
    fan_out, fan_in, n_peers = scorer.compute_graph_features(src, dst)
    bhv_targets = torch.tensor(
        scorer.compute(X, fan_out, fan_in, n_peers),
        dtype=torch.float32)
    print(f"  Behaviour targets: {bhv_targets.shape}")

    configs = [
        ('Feature Only', ['feature']),
        ('Feature + Context', ['feature', 'context']),
        ('Feature + Temporal', ['feature', 'temporal']),
        ('All Three (Proposed)', ['feature', 'context', 'temporal']),
    ]

    results = []
    for config_name, tracks in configs:
        result = train_config(
            config_name, tracks, data, graph, temporal_seqs,
            node_features, builder, bhv_targets, device)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("ENCODER TRACK ABLATION SUMMARY")
    print(f"{'='*70}")

    df = pd.DataFrame(results)
    print(f"\n{'Config':25s} {'Dim':>4s} {'K':>3s} {'Purity':>7s} "
          f"{'NMI_t':>6s} {'Sil_t':>6s} {'Acc':>6s} {'Top-2':>6s}")
    print('─' * 70)
    for _, row in df.iterrows():
        print(f"{row['config']:25s} {row['fusion_dim']:>4d} "
              f"{row['n_clusters']:>3d} {row['mean_purity']:>7.4f} "
              f"{row['nmi_tactic']:>6.4f} {row['sil_tactic']:>6.4f} "
              f"{row['flow_accuracy']:>6.4f} {row['top2_accuracy']:>6.4f}")

    df.to_csv(os.path.join(EVAL_DIR, 'ablation_encoder_tracks.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(len(configs))
    labels = ['Feature\nOnly', 'Feature+\nContext', 'Feature+\nTemporal',
              'All Three\n(Proposed)']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

    for ax, metric, title in [
        (axes[0], 'mean_purity', 'Cluster Purity'),
        (axes[1], 'nmi_tactic', 'NMI (Tactic)'),
        (axes[2], 'sil_tactic', 'Silhouette (Tactic)'),
        (axes[3], 'flow_accuracy', 'Per-Flow Accuracy')]:

        vals = df[metric].values
        bars = ax.bar(x, vals, color=colors, alpha=0.8,
                       edgecolor='white', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        best_idx = np.argmax(vals)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2.5)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

    plt.suptitle(f'Encoder Track Ablation — Joint Training ({DATASET})',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'ablation_encoder_tracks.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Improvement analysis
    base = df[df['config'] == 'Feature Only'].iloc[0]
    full = df[df['config'] == 'All Three (Proposed)'].iloc[0]
    print(f"\n  Improvement (Feature Only → All Three):")
    for m in ['mean_purity', 'nmi_tactic', 'sil_tactic', 'flow_accuracy']:
        print(f"    {m:15s}: {base[m]:.4f} → {full[m]:.4f} "
              f"({(full[m]-base[m])*100:+.2f}%)")

    print(f"\n  Saved: {EVAL_DIR}/")


if __name__ == '__main__':
    main()