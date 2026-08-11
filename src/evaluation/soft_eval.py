"""
Soft Multi-Tactic Window Evaluation
======================================

Evaluates window-level predictions where both predicted and true
labels can contain multiple tactics with proportions.

Metrics:
  1. Jaccard Similarity — set overlap: |pred ∩ true| / |pred ∪ true|
  2. Soft Jaccard — weighted: Σmin(pred, true) / Σmax(pred, true)
  3. Cosine Similarity — distribution similarity
  4. Top-1 Accuracy — primary predicted == primary true
  5. Coverage — true primary tactic appears in predicted set
  6. Per-tactic Precision/Recall — multi-label binary evaluation
  7. Hamming Loss — fraction of tactics incorrectly predicted
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All possible tactics
ALL_TACTICS = [
    'Impact', 'Reconnaissance', 'Exfiltration',
    'None', 'Command and Control', 'Discovery',
    'Lateral Movement', 'Initial Access', 'Execution',
    'Persistence',
]


def compute_ground_truth_distribution(attacks, gt_mapper):
    """Compute tactic distribution from ground-truth attack labels.

    Args:
        attacks: list of attack labels for flows in the window
        gt_mapper: GroundTruthMapper instance

    Returns:
        dist: dict {tactic: proportion}
        primary: str (majority tactic)
        tactic_set: set of tactics present
    """
    tactic_counts = Counter()
    for attack in attacks:
        tactic = gt_mapper.get_tactic(attack)
        tactic_counts[tactic] += 1

    total = sum(tactic_counts.values())
    dist = {t: c / total for t, c in tactic_counts.items()}

    primary = tactic_counts.most_common(1)[0][0]
    tactic_set = set(tactic_counts.keys())

    return dist, primary, tactic_set


def compute_predicted_distribution(behaviour_counts, behaviour_to_tactic_map):
    """Compute tactic distribution from predicted behaviour counts.

    Args:
        behaviour_counts: dict {behaviour: count} in the window
        behaviour_to_tactic_map: dict {behaviour: {tactic: weight}}

    Returns:
        dist: dict {tactic: proportion}
        primary: str
        tactic_set: set of tactics with proportion > threshold
    """
    tactic_scores = defaultdict(float)
    total_flows = sum(behaviour_counts.values())

    for bhv, count in behaviour_counts.items():
        proportion = count / max(total_flows, 1)
        if bhv in behaviour_to_tactic_map:
            for tactic, weight in behaviour_to_tactic_map[bhv].items():
                tactic_scores[tactic] += proportion * weight

    # Normalize
    total_score = sum(tactic_scores.values())
    if total_score > 0:
        dist = {t: s / total_score for t, s in tactic_scores.items()}
    else:
        dist = {'None': 1.0}

    primary = max(dist, key=dist.get)
    tactic_set = {t for t, s in dist.items() if s >= 0.10}

    return dist, primary, tactic_set


# ══════════════════════════════════════════════════════
# Similarity Metrics
# ══════════════════════════════════════════════════════

def jaccard_similarity(set_a, set_b):
    """Binary Jaccard: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / max(union, 1)


def soft_jaccard(dist_pred, dist_true, tactics=None):
    """Soft Jaccard: Σmin(pred, true) / Σmax(pred, true).

    Treats distributions as fuzzy sets and computes overlap.
    """
    if tactics is None:
        tactics = set(list(dist_pred.keys()) + list(dist_true.keys()))

    numerator = 0.0
    denominator = 0.0
    for t in tactics:
        p = dist_pred.get(t, 0.0)
        g = dist_true.get(t, 0.0)
        numerator += min(p, g)
        denominator += max(p, g)

    return numerator / max(denominator, 1e-8)


def cosine_similarity(dist_pred, dist_true, tactics=None):
    """Cosine similarity between tactic distribution vectors."""
    if tactics is None:
        tactics = sorted(set(list(dist_pred.keys()) + list(dist_true.keys())))

    vec_p = np.array([dist_pred.get(t, 0.0) for t in tactics])
    vec_t = np.array([dist_true.get(t, 0.0) for t in tactics])

    dot = np.dot(vec_p, vec_t)
    norm_p = np.linalg.norm(vec_p)
    norm_t = np.linalg.norm(vec_t)

    if norm_p < 1e-8 or norm_t < 1e-8:
        return 0.0
    return float(dot / (norm_p * norm_t))


def kl_divergence(dist_pred, dist_true, tactics=None, epsilon=1e-8):
    """KL divergence: D_KL(true || pred). Lower is better."""
    if tactics is None:
        tactics = sorted(set(list(dist_pred.keys()) + list(dist_true.keys())))

    kl = 0.0
    for t in tactics:
        p = dist_true.get(t, 0.0)
        q = dist_pred.get(t, 0.0)
        if p > epsilon:
            kl += p * np.log(p / max(q, epsilon))
    return float(kl)


# ══════════════════════════════════════════════════════
# Multi-Label Per-Tactic Metrics
# ══════════════════════════════════════════════════════

def compute_multilabel_metrics(pred_sets, true_sets, tactics=None):
    """Compute per-tactic precision, recall, F1 for multi-label.

    Each window has a SET of predicted tactics and a SET of true tactics.
    """
    if tactics is None:
        tactics = ALL_TACTICS

    metrics = {}
    for tactic in tactics:
        tp = sum(1 for p, t in zip(pred_sets, true_sets)
                 if tactic in p and tactic in t)
        fp = sum(1 for p, t in zip(pred_sets, true_sets)
                 if tactic in p and tactic not in t)
        fn = sum(1 for p, t in zip(pred_sets, true_sets)
                 if tactic not in p and tactic in t)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        if tp + fp + fn > 0:
            metrics[tactic] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn,
            }

    return metrics


# ══════════════════════════════════════════════════════
# Main Evaluator
# ══════════════════════════════════════════════════════

class SoftTacticEvaluator:
    """Evaluates window predictions using soft multi-tactic metrics."""

    def __init__(self, behaviour_to_tactic_map=None):
        """
        Args:
            behaviour_to_tactic_map: dict {behaviour: {tactic: weight}}
        """
        self.bhv_to_tactic = behaviour_to_tactic_map or {}

    def evaluate_windows(self, windows, gt_mapper, save_dir=None):
        """Full evaluation of window predictions.

        Each window must have:
          - 'behaviour_counts': dict {behaviour: flow_count}
             OR 'tactic_scores': dict {tactic: score} (pre-computed)
          - ground truth via 'true_attacks' list or 'majority_attack'

        Returns:
            metrics: dict of aggregate metrics
            df: DataFrame of per-window results
        """
        rows = []
        jaccards, soft_jaccards, cosines, kls = [], [], [], []
        top1_correct, coverage_correct = 0, 0
        pred_sets_all, true_sets_all = [], []

        for w in windows:
            # ── Predicted distribution ──
            if 'pred_tactic_dist' in w:
                pred_dist = w['pred_tactic_dist']
                pred_primary = w.get('primary_tactic',
                                     max(pred_dist, key=pred_dist.get))
                pred_set = {t for t, s in pred_dist.items() if s >= 0.10}
            elif 'tactic_scores' in w:
                pred_dist = w['tactic_scores']
                pred_primary = w.get('primary_tactic',
                                     max(pred_dist, key=pred_dist.get))
                pred_set = set(w.get('final_tactics', [pred_primary]))
            elif 'behaviour_counts' in w:
                pred_dist, pred_primary, pred_set = compute_predicted_distribution(
                    w['behaviour_counts'], self.bhv_to_tactic)
            else:
                continue

            # ── True distribution ──
            if 'true_attacks' in w:
                true_dist, true_primary, true_set = (
                    compute_ground_truth_distribution(
                        w['true_attacks'], gt_mapper))
            elif 'majority_attack' in w:
                true_tactic = gt_mapper.get_tactic(w['majority_attack'])
                true_dist = {true_tactic: 1.0}
                true_primary = true_tactic
                true_set = {true_tactic}
            else:
                continue

            # ── Compute metrics ──
            j = jaccard_similarity(pred_set, true_set)
            sj = soft_jaccard(pred_dist, true_dist, ALL_TACTICS)
            cs = cosine_similarity(pred_dist, true_dist, ALL_TACTICS)
            kl = kl_divergence(pred_dist, true_dist, ALL_TACTICS)

            is_top1 = pred_primary == true_primary
            is_covered = true_primary in pred_set

            jaccards.append(j)
            soft_jaccards.append(sj)
            cosines.append(cs)
            kls.append(kl)
            if is_top1: top1_correct += 1
            if is_covered: coverage_correct += 1
            pred_sets_all.append(pred_set)
            true_sets_all.append(true_set)

            # Format distributions for output
            pred_str = ', '.join(f"{t}:{s:.2f}" for t, s
                                 in sorted(pred_dist.items(),
                                           key=lambda x: -x[1])
                                 if s > 0.01)
            true_str = ', '.join(f"{t}:{s:.2f}" for t, s
                                 in sorted(true_dist.items(),
                                           key=lambda x: -x[1])
                                 if s > 0.01)

            rows.append({
                'src_ip': w.get('src_ip', ''),
                'n_flows': w.get('n_flows', 0),
                'pred_primary': pred_primary,
                'pred_tactics': ', '.join(sorted(pred_set)),
                'pred_dist': pred_str,
                'true_primary': true_primary,
                'true_tactics': ', '.join(sorted(true_set)),
                'true_dist': true_str,
                'jaccard': j,
                'soft_jaccard': sj,
                'cosine': cs,
                'kl_div': kl,
                'top1_correct': is_top1,
                'coverage': is_covered,
            })

        df = pd.DataFrame(rows)
        n = len(rows)

        # Aggregate metrics
        metrics = {
            'n_windows': n,
            'top1_accuracy': top1_correct / max(n, 1),
            'coverage': coverage_correct / max(n, 1),
            'mean_jaccard': np.mean(jaccards) if jaccards else 0,
            'mean_soft_jaccard': np.mean(soft_jaccards) if soft_jaccards else 0,
            'mean_cosine': np.mean(cosines) if cosines else 0,
            'mean_kl_divergence': np.mean(kls) if kls else 0,
        }

        # Per-tactic multi-label metrics
        ml_metrics = compute_multilabel_metrics(
            pred_sets_all, true_sets_all, ALL_TACTICS)

        # ── Print results ──
        print(f"\n{'='*60}")
        print("SOFT MULTI-TACTIC EVALUATION")
        print(f"{'='*60}")
        print(f"  Windows evaluated: {n}")
        print(f"\n  Distribution Metrics:")
        print(f"    Top-1 Accuracy:     {metrics['top1_accuracy']:.4f}")
        print(f"    Coverage (top-k):   {metrics['coverage']:.4f}")
        print(f"    Mean Jaccard:       {metrics['mean_jaccard']:.4f}")
        print(f"    Mean Soft Jaccard:  {metrics['mean_soft_jaccard']:.4f}")
        print(f"    Mean Cosine Sim:    {metrics['mean_cosine']:.4f}")
        print(f"    Mean KL Divergence: {metrics['mean_kl_divergence']:.4f}")

        print(f"\n  Multi-Label Per-Tactic Metrics:")
        print(f"  {'Tactic':>20s} {'Prec':>8s} {'Recall':>8s} "
              f"{'F1':>8s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
        print(f"  {'-'*60}")
        for tactic in ALL_TACTICS:
            if tactic in ml_metrics:
                m = ml_metrics[tactic]
                print(f"  {tactic:>20s} {m['precision']:>8.4f} "
                      f"{m['recall']:>8.4f} {m['f1']:>8.4f} "
                      f"{m['tp']:>6d} {m['fp']:>6d} {m['fn']:>6d}")
                metrics[f'{tactic}_precision'] = m['precision']
                metrics[f'{tactic}_recall'] = m['recall']
                metrics[f'{tactic}_f1'] = m['f1']

        # ── Plots ──
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            df.to_csv(os.path.join(save_dir, 'soft_eval_windows.csv'),
                      index=False)
            self._plot_results(df, ml_metrics, metrics, save_dir)

        return metrics, df

    def _plot_results(self, df, ml_metrics, metrics, save_dir):
        """Generate evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Distribution of similarity scores
        ax = axes[0, 0]
        for name, col, color in [
            ('Jaccard', 'jaccard', 'blue'),
            ('Soft Jaccard', 'soft_jaccard', 'green'),
            ('Cosine', 'cosine', 'orange'),
        ]:
            ax.hist(df[col], bins=30, alpha=0.5, label=name, color=color)
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Similarity Metrics')
        ax.legend()

        # 2. Per-tactic F1
        ax = axes[0, 1]
        tactics = [t for t in ALL_TACTICS if t in ml_metrics]
        f1s = [ml_metrics[t]['f1'] for t in tactics]
        colors = ['green' if f > 0.5 else 'orange' if f > 0.2 else 'red'
                  for f in f1s]
        ax.barh(tactics, f1s, color=colors, alpha=0.7)
        ax.set_xlabel('F1 Score')
        ax.set_title('Multi-Label Per-Tactic F1')
        ax.set_xlim(0, 1)
        for i, v in enumerate(f1s):
            ax.text(v + 0.02, i, f'{v:.2f}', va='center')

        # 3. Confusion: pred_primary vs true_primary
        ax = axes[1, 0]
        if len(df) > 0:
            conf = df.groupby(['true_primary', 'pred_primary']).size().unstack(
                fill_value=0)
            conf.plot(kind='bar', stacked=True, ax=ax, alpha=0.7)
            ax.set_title('Primary Tactic: True vs Predicted')
            ax.set_xlabel('True Tactic')
            ax.set_ylabel('Count')
            ax.legend(title='Predicted', fontsize=7)
            ax.tick_params(axis='x', rotation=45)

        # 4. Summary metrics
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = (
            f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}\n"
            f"Coverage:       {metrics['coverage']:.4f}\n"
            f"Mean Jaccard:   {metrics['mean_jaccard']:.4f}\n"
            f"Soft Jaccard:   {metrics['mean_soft_jaccard']:.4f}\n"
            f"Cosine Sim:     {metrics['mean_cosine']:.4f}\n"
            f"KL Divergence:  {metrics['mean_kl_divergence']:.4f}\n"
            f"\nWindows: {metrics['n_windows']}"
        )
        ax.text(0.1, 0.5, summary_text, fontsize=12,
                family='monospace', va='center',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Summary')

        fig.suptitle('Soft Multi-Tactic Evaluation', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'soft_eval_summary.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Plots saved: {save_dir}/soft_eval_summary.png")