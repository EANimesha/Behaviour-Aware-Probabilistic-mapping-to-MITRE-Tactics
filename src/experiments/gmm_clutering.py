# ========================================================================
#  APPLY DP-GMM CLUSTERING
# ========================================================================
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture


def plot_bayesian_gmm_clusters(Z_train_np, y_train):
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z_train_np)

    # Encode attack types as colors
    attack_to_color = {attack: i for i, attack in enumerate(np.unique(y_train))}
    colors = np.array([attack_to_color[a] for a in y_train])

    dpgmm = BayesianGaussianMixture(
        n_components=20,
        covariance_type='diag',
        weight_concentration_prior=1e-2,
        n_init=10,
        random_state=42,
        verbose=0
    )

    print("  Fitting DP-GMM...")
    cluster_labels = dpgmm.fit_predict(Z_train_np)
    weights = dpgmm.weights_

    # Get actual number of clusters
    n_clusters = np.sum(weights > 0.01)

    print(f"  ✓ Clusters found: {n_clusters}")
    print(f"\n  Cluster assignments:")
    for i in range(n_clusters):
        size = np.sum(cluster_labels == i)
        pct = 100 * size / len(Z_train_np)
        print(f"    Cluster {i}: {size:5d} flows ({pct:5.1f}%) - weight: {weights[i]:.4f}")

    # Silhouette score
    sil_score = silhouette_score(Z_train_np, cluster_labels)
    print(f"\n  Silhouette score: {sil_score:.4f}")
    if sil_score > 0.7:
        print("    → Excellent clustering!")
    elif sil_score > 0.5:
        print("    → Good clustering!")
    elif sil_score > 0.0:
        print("    → Acceptable clustering")
    else:
        print("    → Poor clustering (overlapping)")

    # Visualize clusters
    print("\n  Creating cluster visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("DP-GMM Clustering Results", fontsize=14, fontweight='bold')

    # Cluster assignments
    sc1 = axes[0].scatter(Z_pca[:, 0], Z_pca[:, 1], c=cluster_labels,
                          alpha=0.6, s=20, cmap='tab20')
    axes[0].set_title(f"DP-GMM Clusters ({n_clusters} clusters)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[0], label="Cluster ID")

    # Ground truth
    sc2 = axes[1].scatter(Z_pca[:, 0], Z_pca[:, 1], c=colors,
                          alpha=0.6, s=20, cmap='tab10')
    axes[1].set_title("Ground Truth (Attack Type)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/clustering_results.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: clustering_results.png")
    plt.close()