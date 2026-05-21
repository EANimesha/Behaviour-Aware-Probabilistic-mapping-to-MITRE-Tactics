import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_embeddings(Z_train_np, y_train):
    print("  Running PCA...")
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z_train_np)
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    print("  Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30)
    Z_tsne = tsne.fit_transform(Z_train_np)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Embedding Visualization (Colored by Attack Type)", fontsize=14, fontweight='bold')

    # Encode attack types as colors
    attack_to_color = {attack: i for i, attack in enumerate(np.unique(y_train))}
    colors = np.array([attack_to_color[a] for a in y_train])

    # PCA plot
    sc1 = axes[0].scatter(Z_pca[:, 0], Z_pca[:, 1], c=colors, alpha=0.6, s=20, cmap='tab10')
    axes[0].set_title(f"PCA (Explained Var: {pca.explained_variance_ratio_.sum():.2%})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, alpha=0.3)

    # t-SNE plot
    sc2 = axes[1].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=colors, alpha=0.6, s=20, cmap='tab10')
    axes[1].set_title("t-SNE")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.tab10(i), label=attack)
                       for i, attack in enumerate(np.unique(y_train))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(np.unique(y_train)),
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('embeddings_visualization.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: embeddings_visualization.png")
    plt.close()