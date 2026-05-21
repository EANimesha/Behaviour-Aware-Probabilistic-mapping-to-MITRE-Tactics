"""
Tactic-Level Evaluation Metrics
=================================

Evaluates predicted tactics against ground truth at the TACTIC level.
Technique predictions are included in output but not in metrics.

Metrics computed:
  - Per-tactic precision, recall, F1
  - Macro/weighted F1
  - Confusion matrix
  - Cluster purity (at tactic level)
  - Normalized Mutual Information (NMI)
  - Adjusted Rand Index (ARI)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TacticEvaluator:
    """Evaluate predicted tactics against ground truth.

    All metrics are computed at the TACTIC level:
      - DDoS and DoS both map to "Impact" in ground truth
      - A cluster containing both is 100% pure at tactic level
    """

    def __init__(self, tactic_labels=None):
        """
        Args:
            tactic_labels: ordered list of tactic names for confusion matrix.
                           If None, derived from data.
        """
        self.tactic_labels = tactic_labels

    def evaluate(self, true_tactics, pred_tactics, cluster_assignments=None,
                 Z=None, save_dir=None, prefix=""):
        """Run full tactic-level evaluation.

        Args:
            true_tactics: (N,) array of ground truth tactic strings
            pred_tactics: (N,) array of predicted tactic strings
            cluster_assignments: (N,) optional, for cluster-level metrics
            Z: (N, z_dim) optional, for silhouette score
            save_dir: directory to save plots (None = no plots)
            prefix: filename prefix for plots

        Returns:
            metrics: dict with all computed metrics
        """
        true_tactics = np.array(true_tactics)
        pred_tactics = np.array(pred_tactics)

        metrics = {}

        # ── Classification metrics ──
        all_labels = sorted(set(list(true_tactics) + list(pred_tactics)))
        if self.tactic_labels:
            all_labels = self.tactic_labels

        report = classification_report(
            true_tactics, pred_tactics,
            labels=all_labels, output_dict=True, zero_division=0
        )
        report_str = classification_report(
            true_tactics, pred_tactics,
            labels=all_labels, zero_division=0
        )

        metrics['classification_report'] = report
        metrics['macro_f1'] = f1_score(
            true_tactics, pred_tactics,
            labels=all_labels, average='macro', zero_division=0
        )
        metrics['weighted_f1'] = f1_score(
            true_tactics, pred_tactics,
            labels=all_labels, average='weighted', zero_division=0
        )
        metrics['accuracy'] = np.mean(true_tactics == pred_tactics)

        # Print report
        print(f"\n{'='*60}")
        print(f"TACTIC-LEVEL EVALUATION {prefix}")
        print(f"{'='*60}")
        print(report_str)
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")

        # ── Confusion matrix ──
        cm = confusion_matrix(true_tactics, pred_tactics, labels=all_labels)
        metrics['confusion_matrix'] = cm
        metrics['labels'] = all_labels

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._plot_confusion_matrix(
                cm, all_labels,
                os.path.join(save_dir, f"{prefix}confusion_matrix.png")
            )

        # ── Cluster-level metrics (if cluster assignments provided) ──
        if cluster_assignments is not None:
            cluster_metrics = self._evaluate_clusters(
                true_tactics, pred_tactics, cluster_assignments
            )
            metrics.update(cluster_metrics)

            if save_dir:
                self._plot_cluster_purity(
                    true_tactics, cluster_assignments,
                    os.path.join(save_dir, f"{prefix}cluster_purity.png")
                )

        # ── Embedding-level metrics (if Z provided) ──
        if Z is not None and len(np.unique(true_tactics)) > 1:
            # Silhouette at tactic level
            try:
                sil = silhouette_score(
                    Z, true_tactics,
                    sample_size=min(5000, len(Z))
                )
                metrics['silhouette_tactic'] = sil
                print(f"  Silhouette (tactic-level): {sil:.4f}")
            except Exception as e:
                logger.warning(f"Silhouette failed: {e}")

        return metrics

    def _evaluate_clusters(self, true_tactics, pred_tactics, cluster_assignments):
        """Compute cluster-level metrics."""
        metrics = {}

        # NMI: how much info clusters share with ground truth tactics
        nmi = normalized_mutual_info_score(true_tactics, cluster_assignments)
        metrics['nmi'] = nmi

        # ARI: agreement between cluster partition and tactic partition
        ari = adjusted_rand_score(true_tactics, cluster_assignments)
        metrics['ari'] = ari

        print(f"\n  Cluster-level metrics:")
        print(f"    NMI (clusters vs tactics): {nmi:.4f}")
        print(f"    ARI (clusters vs tactics): {ari:.4f}")

        # Per-cluster purity at tactic level
        unique_clusters = np.unique(cluster_assignments)
        purities = []

        print(f"\n  Per-cluster breakdown (tactic level):")
        print(f"  {'Cluster':>8} {'Size':>8} {'Predicted':>20} "
              f"{'Majority True':>20} {'Purity':>8}")
        print(f"  {'-'*68}")

        for k in sorted(unique_clusters):
            mask = cluster_assignments == k
            cluster_true = true_tactics[mask]
            cluster_pred = pred_tactics[mask]

            if len(cluster_true) == 0:
                continue

            majority_true = Counter(cluster_true).most_common(1)[0]
            purity = majority_true[1] / len(cluster_true)
            purities.append(purity)

            pred_tactic = Counter(cluster_pred).most_common(1)[0][0]

            match = "✓" if pred_tactic == majority_true[0] else "✗"
            print(f"  {k:>8} {mask.sum():>8,} {pred_tactic:>20} "
                  f"{majority_true[0]:>20} {purity:>7.1%} {match}")

        metrics['mean_purity'] = np.mean(purities)
        metrics['weighted_purity'] = np.average(
            purities,
            weights=[np.sum(cluster_assignments == k)
                     for k in sorted(unique_clusters)
                     if np.sum(cluster_assignments == k) > 0]
        )

        print(f"\n    Mean cluster purity:     {metrics['mean_purity']:.4f}")
        print(f"    Weighted cluster purity: {metrics['weighted_purity']:.4f}")

        return metrics

    def _plot_confusion_matrix(self, cm, labels, save_path):
        """Plot confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Normalize per row (per true class)
        cm_norm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Proportion')

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted Tactic", fontsize=11)
        ax.set_ylabel("True Tactic", fontsize=11)
        ax.set_title("Confusion Matrix (tactic level, row-normalized)", fontsize=12)

        # Annotate cells with count and percentage
        for i in range(len(labels)):
            for j in range(len(labels)):
                count = cm[i, j]
                pct = cm_norm[i, j]
                if count > 0:
                    color = 'white' if pct > 0.5 else 'black'
                    ax.text(j, i, f"{count:,}\n({pct:.0%})",
                            ha='center', va='center', fontsize=8, color=color)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Confusion matrix saved: {save_path}")

    def _plot_cluster_purity(self, true_tactics, cluster_assignments, save_path):
        """Plot stacked bar chart showing tactic composition per cluster."""
        unique_clusters = sorted(np.unique(cluster_assignments))
        unique_tactics = sorted(np.unique(true_tactics))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_tactics), 10)))

        fig, ax = plt.subplots(figsize=(max(8, len(unique_clusters) * 0.8), 6))

        bottom = np.zeros(len(unique_clusters))
        for t_idx, tactic in enumerate(unique_tactics):
            counts = []
            for k in unique_clusters:
                mask = cluster_assignments == k
                cluster_true = true_tactics[mask]
                counts.append(np.sum(cluster_true == tactic))
            counts = np.array(counts, dtype=float)

            # Normalize to proportions
            totals = np.array([np.sum(cluster_assignments == k)
                               for k in unique_clusters], dtype=float)
            totals[totals == 0] = 1
            proportions = counts / totals

            ax.bar(range(len(unique_clusters)), proportions, bottom=bottom,
                   label=tactic, color=colors[t_idx % 10], alpha=0.8)
            bottom += proportions

        ax.set_xticks(range(len(unique_clusters)))
        ax.set_xticklabels([str(k) for k in unique_clusters], fontsize=8)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Proportion")
        ax.set_title("Tactic composition per cluster", fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Cluster purity plot saved: {save_path}")