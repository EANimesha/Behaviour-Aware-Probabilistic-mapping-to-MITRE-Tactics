"""
Embedding Quality Diagnostics
==============================

Evaluates the quality of fused embeddings from the multi-modal
encoder pipeline (MLP + GraphSAGE + Transformer → VAE fusion)
BEFORE using them for DP-GMM clustering.

Run after training (Stage 3 complete, before Stage 4).

Produces 8 diagnostic plots:
  1. t-SNE projection colored by ground truth attack type
  2. UMAP projection colored by ground truth attack type
  3. VAE reconstruction error distribution per attack type
  4. Per-track contribution: cosine similarity of each track to fused z
  5. Latent dimension activity: variance per z-dimension (detects collapse)
  6. KL divergence per dimension (detects posterior collapse)
  7. Pairwise distance distributions: intra-class vs inter-class
  8. Hopkins statistic + silhouette score (cluster tendency)

Usage:
    python evaluate_embeddings.py
    # or import and call:
    from evaluate_embeddings import run_diagnostics
    run_diagnostics(model_path="checkpoints/model_checkpoint.pt")
"""

import os
import numpy as np
import torch
import matplotlib
from sklearn.decomposition import PCA

from src.streaming.streaming_gmm import StreamingDPGMM

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import warnings
import umap
import sys
warnings.filterwarnings('ignore')

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'
if DATASET == 'cicids2018':
    from src.config.config_cicids import (
    DATA_PATH, SAMPLE_SIZE,
    FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
    Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
    VAE_BATCH_SIZE, # VAE_BETA,
    CHECKPOINT_DIR, MODEL_PATH,
    )
elif DATASET == 'unsw_nb15':
    from src.config.config_unsw import (
    DATA_PATH, SAMPLE_SIZE,
    FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
    Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
    VAE_BATCH_SIZE, # VAE_BETA,
    CHECKPOINT_DIR, MODEL_PATH,
    )
else:
    from src.config.config import (
    DATA_PATH, SAMPLE_SIZE,
    FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
    Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
    VAE_BATCH_SIZE, # VAE_BETA,
    CHECKPOINT_DIR, MODEL_PATH,
    )
from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder

# ── Output directory ──
DIAG_DIR = os.path.join(CHECKPOINT_DIR, "diagnostics")

def load_and_encode(model_path=MODEL_PATH, max_samples=50000):
    """Load checkpoint, rebuild encoder, extract embeddings + intermediates.

    Returns all the tensors needed for diagnostics:
      Z, h_fusion, h_recon, h_f, h_c, h_b,
      y_labels, X_train
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    # Rebuild encoder
    feature_enc = FeatureEncoder(cfg['feature_dim'], output_dim=cfg['feature_output_dim'])
    context_enc = ContextEncoder(cfg['node_feature_dim'], hidden_dim=32,
                                 output_dim=cfg['context_output_dim'], dropout=0.0)
    behaviour_enc = TransformerBehaviourEncoder(cfg['feature_dim'], hidden_dim=64,
                                                output_dim=cfg['behaviour_output_dim'],
                                                num_layers=2, num_heads=4, dropout=0.0)
    joint_enc = JointEncoder(
        feature_encoder=feature_enc, context_encoder=context_enc,
        behaviour_encoder=behaviour_enc,
        feature_output_dim=cfg['feature_output_dim'],
        context_output_dim=cfg['context_output_dim'],
        behaviour_output_dim=cfg['behaviour_output_dim'],
        z_dim=cfg['z_dim'],
        n_behaviours=5,
        lambda_bhv=1.0,
    )
    joint_enc.load_state_dict(checkpoint['joint_encoder_state'])
    joint_enc.to(device)
    joint_enc.eval()

    # Load data
    print("Loading data...")
    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    preprocessor = Preprocessor()
    data = preprocessor.preprocess_dataset(df)

    X_train = data['X_train']
    y_labels = data['y_train']
    src_train, dst_train = data['src_train'], data['dst_train']

    # Subsample if too large
    if max_samples and len(X_train) > max_samples:
        idx = np.random.RandomState(42).choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
        y_labels = y_labels[idx] if y_labels is not None else None
        src_train = src_train[idx]
        dst_train = dst_train[idx]
        print(f"  Subsampled to {max_samples} for diagnostics")

    # Build graph + sequences
    print("Building graph...")
    builder = NetworkGraphBuilder(seq_length=cfg['seq_length'])
    graph, temporal_seqs = builder.build_graph(X_train, src_train, dst_train)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    with torch.no_grad():
        node_emb = context_enc.to(device)(graph, node_features)
        context_emb = builder.get_ip_embeddings(node_emb, src_train, dst_train).to(device)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    t_mask = torch.ones(len(X_train), temporal_seqs.shape[1]).to(device)

    # ── Extract ALL intermediates in batches ──
    print("Extracting embeddings + intermediates...")
    all_z, all_fusion, all_recon = [], [], []
    all_hf, all_hc, all_hb = [], [], []

    bs = VAE_BATCH_SIZE
    n = len(X_tensor)

    all_pred_bhv = []

    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            bf = X_tensor[s:e]
            bc = context_emb[s:e]
            bt = temporal_seqs[s:e]
            bm = t_mask[s:e]

            # Per-track outputs
            h_f = joint_enc.feature_encoder(bf)
            h_b = joint_enc.behaviour_encoder(bt, bm)
            h_c = bc  # already encoded

            # Fusion + AE
            h_fusion = torch.cat([h_f, h_c, h_b], dim=1)
            z = joint_enc.encoder(h_fusion)
            h_recon = joint_enc.decoder(z)

            # Behaviour head prediction
            if hasattr(joint_enc, 'behaviour_head'):
                pred_bhv = joint_enc.behaviour_head(z)
                all_pred_bhv.append(pred_bhv.cpu())

            all_z.append(z.cpu())
            all_fusion.append(h_fusion.cpu())
            all_recon.append(h_recon.cpu())
            all_hf.append(h_f.cpu())
            all_hc.append(h_c.cpu())
            all_hb.append(h_b.cpu())

    # Compute true behaviour scores from raw features
    true_bhv = None
    try:
        from src.encoder.behaviour_scorer import BehaviourScorer
        scorer = BehaviourScorer(feature_names=data['feature_names'])
        fan_out, fan_in, n_peers = scorer.compute_graph_features(
            src_train, dst_train)
        true_bhv = scorer.compute(
            X_train, fan_out=fan_out, fan_in=fan_in, n_peers=n_peers)
        print(f"  Behaviour scores computed: {true_bhv.shape}")
    except Exception as e:
        print(f"  Could not compute behaviour scores: {e}")

    results = {
        'Z': torch.cat(all_z).numpy(),
        'h_fusion': torch.cat(all_fusion).numpy(),
        'h_recon': torch.cat(all_recon).numpy(),
        'h_f': torch.cat(all_hf).numpy(),
        'h_c': torch.cat(all_hc).numpy(),
        'h_b': torch.cat(all_hb).numpy(),
        'pred_bhv': (torch.cat(all_pred_bhv).numpy()
                     if all_pred_bhv else None),
        'true_bhv': true_bhv,
        'y_labels': y_labels,
        'X_train': X_train,
        'checkpoint': checkpoint,
    }
    print(f"  Z: {results['Z'].shape}, labels: {len(np.unique(y_labels))}")
    return results

# ══════════════════════════════════════════════════════════
# Diagnostic plots
# ══════════════════════════════════════════════════════════

def plot_tsne(Z, labels, save_path, max_points=10000, bhv_labels=None):
    """Plot 1: t-SNE — attack types + behaviours side by side."""
    print("  Computing t-SNE...")
    if len(Z) > max_points:
        idx = np.random.RandomState(42).choice(len(Z), max_points, replace=False)
        Z_sub, labels_sub = Z[idx], labels[idx]
        bhv_sub = bhv_labels[idx] if bhv_labels is not None else None
    else:
        Z_sub, labels_sub = Z, labels
        bhv_sub = bhv_labels

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    Z_2d = tsne.fit_transform(Z_sub)

    has_bhv = bhv_sub is not None
    n_rows = 2 if has_bhv else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 7 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Left: Attack type labels
    ax = axes[0]
    unique_labels = sorted(np.unique(labels_sub))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))
    for i, label in enumerate(unique_labels):
        mask = labels_sub == label
        count = mask.sum()
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=[colors[i % 10]],
                   s=8, alpha=0.5, label=f"{label} ({count:,})")
    ax.legend(loc='upper right', fontsize=10, markerscale=3)
    ax.set_title("t-SNE by Attack Type", fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)

    # Right: Behaviour labels
    if has_bhv:
        ax = axes[1]
        unique_bhvs = sorted(np.unique(bhv_sub))
        bhv_colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_bhvs), 1)))
        for i, bhv in enumerate(unique_bhvs):
            mask = bhv_sub == bhv
            count = mask.sum()
            ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=[bhv_colors[i]],
                       s=8, alpha=0.5, label=f"{bhv} ({count:,})")
        ax.legend(loc='upper right', fontsize=10, markerscale=3)
        ax.set_title("t-SNE by Behaviour Label", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Latent Space Visualization (z)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {save_path}")


def plot_umap(Z, labels, save_path, max_points=10000):
    """Plot 2: UMAP 2D projection colored by attack type."""
    print("  Computing UMAP...")
    if len(Z) > max_points:
        idx = np.random.RandomState(42).choice(len(Z), max_points, replace=False)
        Z_sub, labels_sub = Z[idx], labels[idx]
    else:
        Z_sub, labels_sub = Z, labels

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    Z_2d = reducer.fit_transform(Z_sub)

    unique_labels = sorted(np.unique(labels_sub))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels_sub == label
        count = mask.sum()
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=[colors[i % 10]],
                   s=8, alpha=0.5, label=f"{label} ({count:,})")

    ax.legend(loc='upper right', fontsize=8, markerscale=3)
    ax.set_title("UMAP of VAE latent space (z = μ)", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {save_path}")


def plot_reconstruction_error(h_fusion, h_recon, labels, save_path,
                              pred_bhv=None, true_bhv=None):
    """Plot 3: Reconstruction + Behaviour loss by attack type.

    Shows both components of the training loss:
      L = L_reconstruction + λ * L_behaviour
    """
    print("  Computing reconstruction errors...")
    recon_errors = np.mean((h_fusion - h_recon) ** 2, axis=1)

    has_bhv = (pred_bhv is not None and true_bhv is not None)
    if has_bhv:
        bhv_errors = np.mean((pred_bhv - true_bhv) ** 2, axis=1)
        total_errors = recon_errors + bhv_errors
        bhv_names = ['scan', 'flood', 'exfil', 'beacon', 'lateral']

    unique_labels = sorted(np.unique(labels))
    n_plots = 3 if has_bhv else 2
    fig, axes = plt.subplots(2, n_plots, figsize=(6 * n_plots, 10))

    # ── Row 1: Boxplots ──

    # Reconstruction MSE boxplot
    ax = axes[0, 0]
    data_by_class = [recon_errors[labels == l] for l in unique_labels]
    bp = ax.boxplot(data_by_class, labels=unique_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'],
                            plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Reconstruction MSE by attack type", fontsize=11)
    ax.set_ylabel("MSE")
    ax.tick_params(axis='x', rotation=45)

    if has_bhv:
        # Behaviour MSE boxplot
        ax = axes[0, 1]
        data_by_class = [bhv_errors[labels == l] for l in unique_labels]
        bp = ax.boxplot(data_by_class, labels=unique_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'],
                                plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title("Behaviour Prediction MSE by attack type", fontsize=11)
        ax.set_ylabel("MSE")
        ax.tick_params(axis='x', rotation=45)

        # Total loss boxplot
        ax = axes[0, 2]
        data_by_class = [total_errors[labels == l] for l in unique_labels]
        bp = ax.boxplot(data_by_class, labels=unique_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'],
                                plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title("Total Loss (Recon + Behaviour) by attack type", fontsize=11)
        ax.set_ylabel("MSE")
        ax.tick_params(axis='x', rotation=45)

    # ── Row 2: Histograms / Per-behaviour breakdown ──

    # Reconstruction histogram
    ax = axes[1, 0]
    for i, label in enumerate(unique_labels):
        err = recon_errors[labels == label]
        ax.hist(err, bins=50, alpha=0.4, label=label, density=True)
    ax.set_title("Reconstruction error distribution", fontsize=11)
    ax.set_xlabel("MSE")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.set_xlim(0, np.percentile(recon_errors, 99))

    if has_bhv:
        # Per-behaviour-dimension MSE
        ax = axes[1, 1]
        n_bhv = pred_bhv.shape[1]
        per_dim_errors = {}
        for label in unique_labels:
            mask = labels == label
            dim_errs = []
            for d in range(n_bhv):
                dim_errs.append(
                    np.mean((pred_bhv[mask, d] - true_bhv[mask, d]) ** 2))
            per_dim_errors[label] = dim_errs

        x = np.arange(n_bhv)
        width = 0.8 / len(unique_labels)
        for i, label in enumerate(unique_labels):
            ax.bar(x + i * width, per_dim_errors[label], width,
                   label=label, alpha=0.7)
        dim_labels = bhv_names[:n_bhv] if len(bhv_names) >= n_bhv else [
            f'dim_{d}' for d in range(n_bhv)]
        ax.set_xticks(x + width * len(unique_labels) / 2)
        ax.set_xticklabels(dim_labels)
        ax.set_title("Per-Behaviour Prediction MSE", fontsize=11)
        ax.set_ylabel("MSE")
        ax.legend(fontsize=7)

        # Behaviour prediction scatter (pred vs true, first 2 dims)
        ax = axes[1, 2]
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(true_bhv[mask, 0], pred_bhv[mask, 0],
                       alpha=0.3, s=5, label=label, color=colors[i])
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='perfect')
        ax.set_xlabel(f"True {bhv_names[0]} score")
        ax.set_ylabel(f"Predicted {bhv_names[0]} score")
        ax.set_title(f"Behaviour Prediction: {bhv_names[0]}", fontsize=11)
        ax.legend(fontsize=7)

    fig.suptitle("Training Loss Components by Attack Type", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {save_path}")

    # Print summary
    print(f"    Reconstruction MSE: mean={recon_errors.mean():.4f}")
    if has_bhv:
        print(f"    Behaviour MSE:     mean={bhv_errors.mean():.4f}")
        print(f"    Total Loss:        mean={total_errors.mean():.4f}")
    for label in unique_labels:
        r = recon_errors[labels == label].mean()
        line = f"      {label:20s}: recon={r:.4f}"
        if has_bhv:
            b = bhv_errors[labels == label].mean()
            line += f", bhv={b:.4f}, total={r + b:.4f}"
        print(line)

def plot_track_contributions(h_f, h_c, h_b, Z_mu, labels, save_path):
    """Plot 4: Cosine similarity of each track to the final z embedding.

    Shows whether all three tracks are contributing meaningfully or
    if one track dominates the latent space.
    """
    print("  Computing per-track contributions...")
    from numpy.linalg import norm

    def cosine_sim(A, B):
        dot = np.sum(A * B, axis=1)
        nA = norm(A, axis=1) + 1e-8
        nB = norm(B, axis=1) + 1e-8
        return dot / (nA * nB)

    # We can't directly compare tracks to z (different dims),
    # so compare track norms and their relative contribution to h_fusion
    h_fusion = np.concatenate([h_f, h_c, h_b], axis=1)

    # L2 norm of each track per sample
    norm_f = norm(h_f, axis=1)
    norm_c = norm(h_c, axis=1)
    norm_b = norm(h_b, axis=1)
    total_norm = norm_f + norm_c + norm_b + 1e-8

    # Relative contribution
    contrib_f = norm_f / total_norm
    contrib_c = norm_c / total_norm
    contrib_b = norm_b / total_norm

    unique_labels = sorted(np.unique(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: global contribution distribution
    axes[0].hist(contrib_f, bins=50, alpha=0.6, label='Feature (MLP)', color='#4472C4')
    axes[0].hist(contrib_c, bins=50, alpha=0.6, label='Context (GraphSAGE)', color='#548235')
    axes[0].hist(contrib_b, bins=50, alpha=0.6, label='Behaviour (Transformer)', color='#C65911')
    axes[0].set_title("Per-track norm contribution to fusion", fontsize=12)
    axes[0].set_xlabel("Fraction of total L2 norm")
    axes[0].legend()

    # Middle: per-class mean contribution (stacked bar)
    means_f, means_c, means_b = [], [], []
    for label in unique_labels:
        mask = labels == label
        means_f.append(contrib_f[mask].mean())
        means_c.append(contrib_c[mask].mean())
        means_b.append(contrib_b[mask].mean())

    x = np.arange(len(unique_labels))
    axes[1].bar(x, means_f, 0.6, label='Feature', color='#4472C4')
    axes[1].bar(x, means_c, 0.6, bottom=means_f, label='Context', color='#548235')
    axes[1].bar(x, means_b, 0.6,
                bottom=np.array(means_f) + np.array(means_c),
                label='Behaviour', color='#C65911')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(unique_labels, rotation=45, ha='right', fontsize=8)
    axes[1].set_title("Track contribution by attack type", fontsize=12)
    axes[1].set_ylabel("Fraction")
    axes[1].legend(fontsize=8)

    # Right: per-track variance (are they all producing diverse outputs?)
    var_f = np.var(h_f, axis=0)
    var_c = np.var(h_c, axis=0)
    var_b = np.var(h_b, axis=0)
    axes[2].bar(['Feature\n(MLP)', 'Context\n(GraphSAGE)', 'Behaviour\n(Transformer)'],
                [var_f.mean(), var_c.mean(), var_b.mean()],
                color=['#4472C4', '#548235', '#C65911'], alpha=0.7)
    axes[2].set_title("Mean activation variance per track", fontsize=12)
    axes[2].set_ylabel("Mean variance across dimensions")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {save_path}")

    # Print summary
    print(f"    Track contributions (global mean):")
    print(f"      Feature (MLP):        {contrib_f.mean():.3f}")
    print(f"      Context (GraphSAGE):  {contrib_c.mean():.3f}")
    print(f"      Behaviour (Transf.):  {contrib_b.mean():.3f}")


def plot_latent_dimensions(Z_mu, Z_logvar, save_path):
    """Plot 5+6: Latent dimension activity and KL per dimension.

    Detects:
      - Posterior collapse: dimensions where KL ≈ 0 (z just matches prior)
      - Dead dimensions: dimensions with near-zero variance
      - Informative dimensions: high variance + high KL
    """
    print("  Analyzing latent dimensions...")
    z_dim = Z_mu.shape[1]

    # Variance of mu per dimension (across all samples)
    mu_var = np.var(Z_mu, axis=0)

    # Mean of sigma^2 per dimension
    mean_sigma2 = np.mean(np.exp(Z_logvar), axis=0)

    # KL per dimension: 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
    kl_per_dim = 0.5 * np.mean(
        Z_mu**2 + np.exp(Z_logvar) - Z_logvar - 1, axis=0
    )

    # Active dimensions: KL > 0.1
    active_dims = np.sum(kl_per_dim > 0.1)
    collapsed_dims = np.sum(kl_per_dim < 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: mu variance per dimension
    sorted_var = np.sort(mu_var)[::-1]
    axes[0, 0].bar(range(z_dim), sorted_var, color='#4472C4', alpha=0.7)
    axes[0, 0].axhline(y=0.01, color='red', linestyle='--', alpha=0.5,
                        label='Collapse threshold')
    axes[0, 0].set_title(f"μ variance per dimension (sorted)", fontsize=12)
    axes[0, 0].set_xlabel("Dimension (sorted by variance)")
    axes[0, 0].set_ylabel("Var(μ)")
    axes[0, 0].legend()

    # Top-right: KL per dimension
    sorted_kl = np.sort(kl_per_dim)[::-1]
    colors_kl = ['#548235' if k > 0.1 else '#C65911' if k > 0.01 else '#999'
                 for k in sorted_kl]
    axes[0, 1].bar(range(z_dim), sorted_kl, color=colors_kl, alpha=0.7)
    axes[0, 1].axhline(y=0.1, color='green', linestyle='--', alpha=0.5,
                        label='Active threshold (0.1)')
    axes[0, 1].axhline(y=0.01, color='red', linestyle='--', alpha=0.5,
                        label='Collapse threshold (0.01)')
    axes[0, 1].set_title(f"KL divergence per dimension (sorted)\n"
                          f"{active_dims} active, {collapsed_dims} collapsed",
                          fontsize=12)
    axes[0, 1].set_xlabel("Dimension (sorted by KL)")
    axes[0, 1].set_ylabel("KL(q||p)")
    axes[0, 1].legend(fontsize=8)

    # Bottom-left: mu variance vs KL scatter (identifies dimension roles)
    axes[1, 0].scatter(mu_var, kl_per_dim, c='#4472C4', alpha=0.6, s=30)
    for i in range(z_dim):
        if kl_per_dim[i] > np.percentile(kl_per_dim, 90) or mu_var[i] > np.percentile(mu_var, 90):
            axes[1, 0].annotate(f'd{i}', (mu_var[i], kl_per_dim[i]), fontsize=7)
    axes[1, 0].set_xlabel("Var(μ)")
    axes[1, 0].set_ylabel("KL divergence")
    axes[1, 0].set_title("Dimension role map\n(bottom-left = collapsed, top-right = informative)",
                          fontsize=11)
    axes[1, 0].grid(True, alpha=0.2)

    # Bottom-right: mean sigma^2 per dimension (learned uncertainty)
    axes[1, 1].bar(range(z_dim), np.sort(mean_sigma2)[::-1],
                    color='#C65911', alpha=0.7)
    axes[1, 1].axhline(y=1.0, color='blue', linestyle='--', alpha=0.5,
                        label='Prior σ²=1')
    axes[1, 1].set_title("Learned σ² per dimension (sorted)", fontsize=12)
    axes[1, 1].set_xlabel("Dimension")
    axes[1, 1].set_ylabel("Mean σ²")
    axes[1, 1].legend()

    fig.suptitle(f"Latent Space Health: {z_dim}D, {active_dims} active dims, "
                 f"{collapsed_dims} collapsed", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {save_path}")

    print(f"    Latent space: {z_dim} dimensions")
    print(f"      Active (KL > 0.1):    {active_dims}")
    print(f"      Weak (0.01 < KL < 0.1): {z_dim - active_dims - collapsed_dims}")
    print(f"      Collapsed (KL < 0.01): {collapsed_dims}")
    print(f"      Total KL: {kl_per_dim.sum():.2f}")


def plot_distance_distributions(Z, labels, save_path, max_pairs=50000):
    """Plot 7: Intra-class vs inter-class pairwise distance distributions.

    If embeddings are good, intra-class distances should be smaller
    than inter-class distances with minimal overlap.
    """
    print("  Computing pairwise distances...")
    unique_labels = np.unique(labels)

    # Subsample for speed
    if len(Z) > 5000:
        idx = np.random.RandomState(42).choice(len(Z), 5000, replace=False)
        Z_sub, labels_sub = Z[idx], labels[idx]
    else:
        Z_sub, labels_sub = Z, labels

    intra_dists = []
    inter_dists = []

    for label in unique_labels:
        mask = labels_sub == label
        Z_class = Z_sub[mask]
        Z_other = Z_sub[~mask]

        if len(Z_class) < 2:
            continue

        # Intra-class: pairwise within this class (sample pairs)
        n_pairs = min(max_pairs // len(unique_labels), len(Z_class) * (len(Z_class)-1) // 2)
        if len(Z_class) <= 200:
            d = cdist(Z_class, Z_class, metric='euclidean')
            triu = d[np.triu_indices_from(d, k=1)]
            intra_dists.extend(triu.tolist())
        else:
            idx1 = np.random.choice(len(Z_class), n_pairs)
            idx2 = np.random.choice(len(Z_class), n_pairs)
            dists = np.linalg.norm(Z_class[idx1] - Z_class[idx2], axis=1)
            intra_dists.extend(dists[dists > 0].tolist())

        # Inter-class: pairs between this class and others
        if len(Z_other) > 0:
            idx1 = np.random.choice(len(Z_class), min(n_pairs, len(Z_class)))
            idx2 = np.random.choice(len(Z_other), min(n_pairs, len(Z_other)))
            dists = np.linalg.norm(Z_class[idx1] - Z_other[idx2], axis=1)
            inter_dists.extend(dists.tolist())

    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(intra_dists, bins=100, alpha=0.6, density=True,
            label=f'Intra-class (n={len(intra_dists):,})', color='#4472C4')
    ax.hist(inter_dists, bins=100, alpha=0.6, density=True,
            label=f'Inter-class (n={len(inter_dists):,})', color='#C65911')
    ax.axvline(np.median(intra_dists), color='#4472C4', linestyle='--', alpha=0.7)
    ax.axvline(np.median(inter_dists), color='#C65911', linestyle='--', alpha=0.7)

    # Overlap ratio
    overlap_threshold = (np.median(intra_dists) + np.median(inter_dists)) / 2
    intra_above = np.mean(intra_dists > overlap_threshold)
    inter_below = np.mean(inter_dists < overlap_threshold)
    overlap_pct = (intra_above + inter_below) / 2 * 100

    ax.set_title(f"Pairwise distance distributions\n"
                 f"Separation quality: {100-overlap_pct:.0f}% "
                 f"(higher = better separated)", fontsize=12)
    ax.set_xlabel("Euclidean distance in latent space")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {save_path}")
    print(f"    Intra-class median: {np.median(intra_dists):.3f}")
    print(f"    Inter-class median: {np.median(inter_dists):.3f}")
    print(f"    Separation ratio:   {np.median(inter_dists)/np.median(intra_dists):.2f}x")
    print(f"    Overlap:            {overlap_pct:.1f}%")


def plot_cluster_tendency(Z, labels, save_path, max_points=5000):
    """Plot 8: Hopkins statistic + silhouette score.

    Hopkins: measures whether data has clustering tendency (>0.7 = good)
    Silhouette: measures how well labels match embedding structure (-1 to 1)
    """
    print("  Computing cluster tendency metrics...")

    if len(Z) > max_points:
        idx = np.random.RandomState(42).choice(len(Z), max_points, replace=False)
        Z_sub, labels_sub = Z[idx], labels[idx]
    else:
        Z_sub, labels_sub = Z, labels

    # Hopkins statistic
    n = len(Z_sub)
    d = Z_sub.shape[1]
    m = min(100, n // 10)

    np.random.seed(42)
    random_indices = np.random.choice(n, m, replace=False)
    nn_model = NearestNeighbors(n_neighbors=2)
    nn_model.fit(Z_sub)

    # Distance from random sample points to their nearest neighbor in data
    u_distances, _ = nn_model.kneighbors(Z_sub[random_indices], n_neighbors=2)
    u_dist = u_distances[:, 1]

    # Distance from random uniform points to their nearest neighbor in data
    Z_min = Z_sub.min(axis=0)
    Z_max = Z_sub.max(axis=0)
    random_points = np.random.uniform(Z_min, Z_max, size=(m, d))
    w_distances, _ = nn_model.kneighbors(random_points, n_neighbors=1)
    w_dist = w_distances[:, 0]

    hopkins = np.sum(w_dist) / (np.sum(u_dist) + np.sum(w_dist))

    # Silhouette score
    sil_score = silhouette_score(Z_sub, labels_sub, sample_size=min(5000, len(Z_sub)))

    # Per-class silhouette
    from sklearn.metrics import silhouette_samples
    sil_samples = silhouette_samples(Z_sub, labels_sub)
    unique_labels = sorted(np.unique(labels_sub))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Hopkins gauge
    theta = np.linspace(0, np.pi, 100)
    axes[0].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    axes[0].fill_between(np.cos(theta[:33]), np.sin(theta[:33]), alpha=0.15, color='red')
    axes[0].fill_between(np.cos(theta[33:66]), np.sin(theta[33:66]), alpha=0.15, color='orange')
    axes[0].fill_between(np.cos(theta[66:]), np.sin(theta[66:]), alpha=0.15, color='green')
    angle = np.pi * (1 - hopkins)
    axes[0].annotate('', xy=(0.8*np.cos(angle), 0.8*np.sin(angle)),
                      xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->', color='black', lw=2))
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_ylim(-0.1, 1.3)
    axes[0].set_aspect('equal')
    axes[0].text(0, -0.05, f"Hopkins = {hopkins:.3f}", ha='center', fontsize=14, fontweight='bold')
    axes[0].text(-0.9, 0.05, "Random\n(<0.5)", ha='center', fontsize=8, color='red')
    axes[0].text(0, 0.05, "Weak\n(0.5-0.7)", ha='center', fontsize=8, color='orange')
    axes[0].text(0.9, 0.05, "Strong\n(>0.7)", ha='center', fontsize=8, color='green')
    axes[0].set_title("Hopkins statistic (clustering tendency)", fontsize=12)
    axes[0].axis('off')

    # Right: per-class silhouette
    y_lower = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels_sub == label
        sil_class = np.sort(sil_samples[mask])
        size = len(sil_class)
        y_upper = y_lower + size
        axes[1].fill_betweenx(np.arange(y_lower, y_upper), 0, sil_class,
                                facecolor=colors[i], alpha=0.6)
        axes[1].text(-0.05, y_lower + 0.5 * size, label, fontsize=7, va='center')
        y_lower = y_upper + 5

    axes[1].axvline(x=sil_score, color='red', linestyle='--',
                     label=f'Mean = {sil_score:.3f}')
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_title(f"Silhouette plot (score = {sil_score:.3f})", fontsize=12)
    axes[1].set_xlabel("Silhouette coefficient")
    axes[1].set_yticks([])
    axes[1].legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {save_path}")
    print(f"    Hopkins statistic: {hopkins:.3f} "
          f"({'good' if hopkins > 0.7 else 'weak' if hopkins > 0.5 else 'poor'})")
    print(f"    Silhouette score:  {sil_score:.3f} "
          f"({'good' if sil_score > 0.3 else 'moderate' if sil_score > 0.1 else 'poor'})")


# ══════════════════════════════════════════════════════════
# Main entry
# ══════════════════════════════════════════════════════════

def plot_gmm_clusters_pca(model_path, Z, save_path):
    # ── Reconstruct DP-GMM ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']
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
    # Predict clusters and get P(GMM)
    assignments, soft_probs, log_liks, novelty_flags = gmm.predict(Z)
    p_gmm_values = soft_probs.max(axis=1)
    print(f"  Cluster assignments: {len(np.unique(assignments))} unique clusters")
    print(f"  P(GMM) range: [{p_gmm_values.min():.4f}, {p_gmm_values.max():.4f}]")
    print(f"  Novelty flags: {novelty_flags.sum()} flows flagged")

    assignments_np = np.asarray(assignments)

    pca = PCA(n_components=2, random_state=42)
    all_z_np = np.asarray(Z)
    z_2d = pca.fit_transform(all_z_np)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=assignments_np,
        s=10,
        alpha=0.6,
        cmap="tab20"
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    plt.title("DP-GMM Cluster Assignments on Embeddings using PCA")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster ID")

    if novelty_flags is not None:
        novelty_flags_np = np.asarray(novelty_flags).astype(bool)

        plt.scatter(
            z_2d[novelty_flags_np, 0],
            z_2d[novelty_flags_np, 1],
            s=40,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="Novelty"
        )

        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # plt.show()


def plot_silhouette_comparison(dataset, data, attack_labels, save_path):
    """Plot 9: Compare silhouette scores at multiple semantic levels.

    Shows how well the z-space separates:
      1. Attack types (DDoS, DoS, Recon, Theft, Benign) — finest granularity
      2. Tactics (Impact, Recon, Exfil, None) — merges DDoS+DoS
      3. Behaviours (from cluster→behaviour mapping) — what the system predicts
      4. GMM clusters — raw cluster separation quality
    """
    from sklearn.metrics import silhouette_score, silhouette_samples
    print("  Computing multi-level silhouette scores...")

    Z = data['Z']
    n = len(Z)
    sample_size = min(5000, n)

    # Level 1: Attack type labels (existing)
    attack_sil = silhouette_score(
        Z, attack_labels, sample_size=sample_size)

    # Level 2: Tactic labels (merged)
    try:
        from src.evaluation.ground_truth import GroundTruthMapper
        mapper = GroundTruthMapper(dataset)
        tactic_labels = mapper.map_labels(attack_labels)
        tactic_sil = silhouette_score(
            Z, tactic_labels, sample_size=sample_size)
    except Exception:
        tactic_labels = attack_labels
        tactic_sil = attack_sil

    # Level 3: Behaviour labels (from checkpoint if available)
    bhv_labels = None
    bhv_sil = None
    try:
        import torch
        from src.config.config import MODEL_PATH
        checkpoint = torch.load(MODEL_PATH, map_location='cpu',
                                weights_only=False)
        cbm = checkpoint.get('cluster_behaviour_map', {})
        cbm = {int(k): v for k, v in cbm.items()}
        if cbm:
            from src.streaming.streaming_gmm import StreamingDPGMM
            gmm_cfg = checkpoint['gmm_config']
            gmm = StreamingDPGMM(
                gmm_cfg['max_components'], gmm_cfg['covariance_type'],
                gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
            gmm.model = checkpoint['gmm_model']
            cluster_ids = gmm.model.predict(Z)
            bhv_labels = np.array([
                cbm.get(int(c), {}).get('primary_behaviour', 'unknown')
                for c in cluster_ids])
            if len(np.unique(bhv_labels)) > 1:
                bhv_sil = silhouette_score(
                    Z, bhv_labels, sample_size=sample_size)
    except Exception as e:
        print(f"    Could not compute behaviour silhouette: {e}")

    # Level 4: GMM cluster labels
    cluster_sil = None
    try:
        if 'cluster_ids' not in dir():
            import torch
            from src.config.config import MODEL_PATH
            checkpoint = torch.load(MODEL_PATH, map_location='cpu',
                                    weights_only=False)
            gmm_cfg = checkpoint['gmm_config']
            from src.streaming.streaming_gmm import StreamingDPGMM
            gmm = StreamingDPGMM(
                gmm_cfg['max_components'], gmm_cfg['covariance_type'],
                gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
            gmm.model = checkpoint['gmm_model']
            cluster_ids = gmm.model.predict(Z)
        if len(np.unique(cluster_ids)) > 1:
            cluster_sil = silhouette_score(
                Z, cluster_ids, sample_size=sample_size)
    except Exception as e:
        print(f"    Could not compute cluster silhouette: {e}")

    # ── Plot comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar chart comparison
    ax = axes[0]
    levels = []
    scores = []
    colors_bar = []

    levels.append(f'Attack Types\n(n={len(np.unique(attack_labels))})')
    scores.append(attack_sil)
    colors_bar.append('#e74c3c')

    levels.append(f'MITRE Tactics\n(n={len(np.unique(tactic_labels))})')
    scores.append(tactic_sil)
    colors_bar.append('#3498db')

    if bhv_sil is not None:
        levels.append(f'Behaviours\n(n={len(np.unique(bhv_labels))})')
        scores.append(bhv_sil)
        colors_bar.append('#2ecc71')

    if cluster_sil is not None:
        levels.append(f'GMM Clusters\n(n={len(np.unique(cluster_ids))})')
        scores.append(cluster_sil)
        colors_bar.append('#9b59b6')

    bars = ax.bar(levels, scores, color=colors_bar, alpha=0.7, edgecolor='black')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score at Different Semantic Levels', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(min(min(scores) - 0.05, -0.05), max(scores) + 0.08)

    # Right: Per-behaviour silhouette (what the system predicts)
    ax = axes[1]
    if bhv_labels is not None and bhv_sil is not None:
        sil_samples_bhv = silhouette_samples(
            Z[:sample_size], bhv_labels[:sample_size])
        unique_bhvs = sorted(np.unique(bhv_labels[:sample_size]))
        colors_bhv = plt.cm.Set2(np.linspace(0, 1, len(unique_bhvs)))

        y_lower = 0
        for i, bhv in enumerate(unique_bhvs):
            mask = bhv_labels[:sample_size] == bhv
            sil_class = np.sort(sil_samples_bhv[mask])
            y_upper = y_lower + len(sil_class)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_class,
                             facecolor=colors_bhv[i], alpha=0.6)
            mean_sil = sil_class.mean()
            ax.text(-0.05, y_lower + len(sil_class) / 2,
                    f'{bhv}\n({mean_sil:.3f})',
                    fontsize=7, va='center', ha='right')
            y_lower = y_upper + 10

        ax.axvline(x=bhv_sil, color='red', linestyle='--',
                   label=f'Mean = {bhv_sil:.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'Per-Behaviour Silhouette (score = {bhv_sil:.3f})',
                     fontsize=12)
        ax.set_xlabel('Silhouette coefficient')
        ax.set_yticks([])
        ax.legend(loc='lower right')

    fig.suptitle('Multi-Level Embedding Quality', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {save_path}")

    # Print comparison
    print(f"\n    Silhouette Comparison:")
    print(f"      Attack types:  {attack_sil:.4f}")
    print(f"      Tactics:       {tactic_sil:.4f}")
    if bhv_sil is not None:
        print(f"      Behaviours:    {bhv_sil:.4f}")
    if cluster_sil is not None:
        print(f"      GMM Clusters:  {cluster_sil:.4f}")
    print(f"\n    Tactic silhouette {'>' if tactic_sil > attack_sil else '<'} "
          f"Attack silhouette → z-space organizes by "
          f"{'tactic/behaviour' if tactic_sil > attack_sil else 'attack type'}")


def main(model_path=MODEL_PATH, max_samples=50000):
    """Run all 8 diagnostic plots."""
    os.makedirs(DIAG_DIR, exist_ok=True)

    print("="*60)
    print("EMBEDDING QUALITY DIAGNOSTICS")
    print("="*60)

    # Load data and extract embeddings
    data = load_and_encode(model_path, max_samples)

    # Z = data['Z_mu']
    Z = data['Z']
    # fused = data['h_fusion']
    labels = data['y_labels']

    if labels is None:
        print("WARNING: No ground truth labels available. "
              "Skipping label-dependent plots.")
        return

    print(f"\nLabel distribution:")
    for attack, count in Counter(labels).most_common():
        print(f"  {attack:20s} {count:>8,}")

        # ── Load behaviour labels from checkpoint ──
    bhv_labels = None
    checkpoint = data.get('checkpoint', {})
    try:
        cbm = checkpoint.get('cluster_behaviour_map', {})
        cbm = {int(k): v for k, v in cbm.items()}
        if cbm:
            from src.streaming.streaming_gmm import StreamingDPGMM
            gmm_cfg = checkpoint['gmm_config']
            gmm_temp = StreamingDPGMM(
                gmm_cfg['max_components'], gmm_cfg['covariance_type'],
                gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
            gmm_temp.model = checkpoint['gmm_model']
            cluster_ids = gmm_temp.model.predict(Z)
            bhv_labels = np.array([
                cbm.get(int(c), {}).get('primary_behaviour', 'unknown')
                for c in cluster_ids])
            print(f"  Behaviour labels loaded: {len(np.unique(bhv_labels))} unique")
    except Exception as e:
        print(f"  Could not load behaviour labels: {e}")


    # # plot_cluster_tendency(fused, labels,
    # #                       os.path.join(DIAG_DIR, "07_cluster_tendency.png"))
    # #
    # # # ── Plot 1: t-SNE ──
    # # print("\n[1/8] t-SNE projection")
    # # plot_tsne(fused, labels, os.path.join(DIAG_DIR, "01_tsne_fused.png"))
    #
    # plot_gmm_clusters_pca(model_path, Z,  os.path.join(DIAG_DIR, "06_cluster_umap.png"))
    #
    # # ── Plot 1: t-SNE ──
    # print("\n[1/8] t-SNE projection")
    # plot_tsne(Z, labels, os.path.join(DIAG_DIR, "01_tsne_beh_scored.png"),bhv_labels=bhv_labels)
    #
    # # ── Plot 2: UMAP ──
    # print("\n[2/8] UMAP projection")
    # plot_umap(Z, labels, os.path.join(DIAG_DIR, "02_umap.png"))
    #
    # # ── Plot 3: Reconstruction error ──
    # # print("\n[3/8] VAE reconstruction error")
    # # plot_reconstruction_error(data['h_fusion'], data['h_recon'], labels,
    # #                            os.path.join(DIAG_DIR, "03_recon_error.png"))
    # # ── Plot 3: Reconstruction + Behaviour error ──
    # print("\n[3/8] Reconstruction + Behaviour loss")
    # plot_reconstruction_error(data['h_fusion'], data['h_recon'], labels,
    #                           os.path.join(DIAG_DIR, "03_recon_bhv_error.png"),
    #                           pred_bhv=data.get('pred_bhv'),
    #                           true_bhv=data.get('true_bhv'))
    #
    # # ── Plot 4: Track contributions ──
    # print("\n[4/8] Per-track contributions")
    # plot_track_contributions(data['h_f'], data['h_c'], data['h_b'],
    #                           Z, labels,
    #                           os.path.join(DIAG_DIR, "04_track_contrib.png"))
    #
    # # # ── Plot 5+6: Latent dimensions ──
    # # print("\n[5/8] Latent dimension analysis")
    # # plot_latent_dimensions(Z, data['Z_logvar'],
    # #                         os.path.join(DIAG_DIR, "05_latent_dims.png"))
    #
    # # # ── Plot 7: Distance distributions ──
    # # print("\n[6/8] Pairwise distance distributions")
    # # plot_distance_distributions(Z, labels,
    # #                              os.path.join(DIAG_DIR, "06_distances.png"))

    # ── Plot 8: Cluster tendency ──
    print("\n[7/8] Cluster tendency (Hopkins + Silhouette)")
    plot_cluster_tendency(Z, labels,
                           os.path.join(DIAG_DIR, "07_cluster_tendency.png"))

    # ── Plot 9: Multi-level silhouette comparison ──
    print("\n[9/9] Multi-level silhouette comparison")
    plot_silhouette_comparison(
        DATASET,
        data, labels,
        os.path.join(DIAG_DIR, "09_silhouette_comparison.png"))

    # ── Summary dashboard ──
    print("\n[8/8] Summary dashboard")
    create_summary_dashboard(data, labels)

    print(f"\n{'='*60}")
    print(f"All diagnostics saved to: {DIAG_DIR}/")
    print(f"{'='*60}")


def create_summary_dashboard(data, labels):
    """Plot 8: Single-page summary with key metrics."""
    Z = data['Z_mu']
    Z_logvar = data['Z_logvar']
    h_fusion = data['h_fusion']
    h_recon = data['h_recon']
    h_f, h_c, h_b = data['h_f'], data['h_c'], data['h_b']

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (0,0) Quick t-SNE
    ax1 = fig.add_subplot(gs[0, 0])
    if len(Z) > 5000:
        idx = np.random.RandomState(42).choice(len(Z), 5000, replace=False)
        Z_s, l_s = Z[idx], labels[idx]
    else:
        Z_s, l_s = Z, labels
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
    Z_2d = tsne.fit_transform(Z_s)
    unique = sorted(np.unique(l_s))
    for i, lab in enumerate(unique):
        m = l_s == lab
        ax1.scatter(Z_2d[m, 0], Z_2d[m, 1], s=4, alpha=0.4, label=lab)
    ax1.set_title("t-SNE overview", fontsize=10)
    ax1.legend(fontsize=6, markerscale=3, loc='best')

    # (0,1) Recon error by class
    ax2 = fig.add_subplot(gs[0, 1])
    recon_err = np.mean((h_fusion - h_recon)**2, axis=1)
    data_by_class = [recon_err[labels == l] for l in unique]
    ax2.boxplot(data_by_class, labels=unique)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_title("Reconstruction MSE", fontsize=10)

    # (0,2) Track contributions
    ax3 = fig.add_subplot(gs[0, 2])
    from numpy.linalg import norm as np_norm
    nf = np_norm(h_f, axis=1).mean()
    nc = np_norm(h_c, axis=1).mean()
    nb = np_norm(h_b, axis=1).mean()
    total = nf + nc + nb
    ax3.bar(['Feature\nMLP', 'Context\nGraphSAGE', 'Behaviour\nTransformer'],
            [nf/total, nc/total, nb/total],
            color=['#4472C4', '#548235', '#C65911'])
    ax3.set_title("Track contribution (norm fraction)", fontsize=10)
    ax3.set_ylim(0, 0.6)

    # (1,0) KL per dim
    ax4 = fig.add_subplot(gs[1, 0])
    kl = 0.5 * np.mean(Z**2 + np.exp(Z_logvar) - Z_logvar - 1, axis=0)
    ax4.bar(range(len(kl)), np.sort(kl)[::-1], color='#4472C4', alpha=0.7)
    ax4.axhline(0.1, color='red', linestyle='--', alpha=0.5)
    active = np.sum(kl > 0.1)
    ax4.set_title(f"KL per dim ({active}/{len(kl)} active)", fontsize=10)

    # (1,1) Key metrics text
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    sil = silhouette_score(Z_s, l_s, sample_size=min(3000, len(Z_s)))
    metrics = [
        f"Z dimensions:     {Z.shape[1]}",
        f"Active dims:      {active}/{Z.shape[1]}",
        f"Total KL:         {kl.sum():.2f}",
        f"Recon MSE:        {recon_err.mean():.4f}",
        f"Silhouette:       {sil:.3f}",
        f"Track balance:    F={nf/total:.2f} C={nc/total:.2f} B={nb/total:.2f}",
        f"Samples:          {len(Z):,}",
        f"Attack types:     {len(unique)}",
    ]
    ax5.text(0.1, 0.95, "Key Metrics", fontsize=12, fontweight='bold',
             transform=ax5.transAxes, va='top')
    for i, line in enumerate(metrics):
        color = 'green' if i == 4 and sil > 0.2 else 'red' if i == 4 and sil < 0.1 else 'black'
        ax5.text(0.1, 0.82 - i*0.1, line, fontsize=10, fontfamily='monospace',
                 transform=ax5.transAxes, va='top', color=color)

    # (1,2) Distance separation
    ax6 = fig.add_subplot(gs[1, 2])
    intra, inter = [], []
    for lab in unique[:5]:  # top 5 classes only
        m = l_s == lab
        Zc = Z_s[m][:100]
        Zo = Z_s[~m][:100]
        if len(Zc) > 1:
            d = cdist(Zc, Zc)
            intra.extend(d[np.triu_indices_from(d, k=1)].tolist())
        if len(Zc) > 0 and len(Zo) > 0:
            inter.extend(cdist(Zc[:50], Zo[:50]).ravel().tolist())
    ax6.hist(intra, bins=50, alpha=0.5, density=True, label='Intra', color='#4472C4')
    ax6.hist(inter, bins=50, alpha=0.5, density=True, label='Inter', color='#C65911')
    ax6.legend(fontsize=8)
    ax6.set_title("Distance separation", fontsize=10)

    fig.suptitle("Embedding Quality Dashboard", fontsize=14, fontweight='bold')
    save_path = os.path.join(DIAG_DIR, "08_dashboard.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {save_path}")


if __name__ == '__main__':
    main(model_path='checkpoints/model_checkpoint.pt')