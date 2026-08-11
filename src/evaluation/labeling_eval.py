"""
Cluster Behaviour Labeling Evaluation
========================================

Compares cluster-level labeling strategies:
  - Mock Hard:  rule-based decision tree → one label per cluster
  - Mock Soft:  rule-based soft scorer → distribution per cluster
  - LLM Hard:   LLM assigns one label per cluster
  - LLM Soft:   LLM assigns label, converted to soft distribution

Metrics:
  - Hard Agreement: % clusters where primary predicted tactic == GT majority tactic
  - Soft Jaccard:   Σmin(pred, true) / Σmax(pred, true) per cluster, averaged
  - KL Divergence:  KL(true || pred) per cluster, averaged
"""

import numpy as np
from collections import Counter, defaultdict
import logging

from src.llm.reasoning import LLMLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_TACTICS = ['Impact', 'Reconnaissance', 'Exfiltration',
               'None', 'Command and Control', 'Discovery']


def compute_cluster_gt_distribution(cluster_assignments, y_train, gt_mapper):
    """Compute ground truth tactic distribution per cluster.

    Returns:
        gt_dists: dict {cluster_idx: {tactic: proportion}}
        gt_primaries: dict {cluster_idx: primary_tactic}
    """
    gt_dists = {}
    gt_primaries = {}

    for k in np.unique(cluster_assignments):
        mask = cluster_assignments == k
        attacks = y_train[mask]
        tactic_counts = Counter()
        for attack in attacks:
            tactic = gt_mapper.get_tactic(attack)
            tactic_counts[tactic] += 1
        total = sum(tactic_counts.values())
        gt_dists[int(k)] = {t: c / total for t, c in tactic_counts.items()}
        gt_primaries[int(k)] = tactic_counts.most_common(1)[0][0]

    return gt_dists, gt_primaries


def soft_jaccard(dist_a, dist_b, tactics=None):
    if tactics is None:
        tactics = set(list(dist_a.keys()) + list(dist_b.keys()))
    num = sum(min(dist_a.get(t, 0), dist_b.get(t, 0)) for t in tactics)
    den = sum(max(dist_a.get(t, 0), dist_b.get(t, 0)) for t in tactics)
    return num / max(den, 1e-8)


def kl_divergence(true_dist, pred_dist, tactics=None, eps=1e-8):
    if tactics is None:
        tactics = set(list(true_dist.keys()) + list(pred_dist.keys()))
    kl = 0.0
    for t in tactics:
        p = true_dist.get(t, 0)
        q = pred_dist.get(t, 0)
        if p > eps:
            kl += p * np.log(p / max(q, eps))
    return kl
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

def evaluate_labeling_strategy(name, cluster_tactic_dists, cluster_primaries,
                                gt_dists, gt_primaries):
    """Evaluate one labeling strategy against ground truth.

    Args:
        name: strategy name for display
        cluster_tactic_dists: dict {cluster_idx: {tactic: score}}
        cluster_primaries: dict {cluster_idx: primary_tactic}
        gt_dists: ground truth distributions
        gt_primaries: ground truth primaries

    Returns:
        dict with hard_agreement, soft_jaccard, kl_divergence
    """
    agreements = []
    jaccards = []
    kls = []
    cosines = []

    for k in gt_dists:
        if k not in cluster_tactic_dists:
            continue

        pred_dist = cluster_tactic_dists[k]
        true_dist = gt_dists[k]
        pred_primary = cluster_primaries.get(k, 'Unknown')
        true_primary = gt_primaries[k]

        # Hard agreement
        agreements.append(1 if pred_primary == true_primary else 0)

        # Soft Jaccard
        jaccards.append(soft_jaccard(pred_dist, true_dist, ALL_TACTICS))

        # KL divergence
        kls.append(kl_divergence(true_dist, pred_dist, ALL_TACTICS))

        cosines.append(cosine_similarity(pred_dist, true_dist, ALL_TACTICS))

    n = len(agreements)
    return {
        'name': name,
        'hard_agreement': sum(agreements) / max(n, 1),
        'soft_jaccard': np.mean(jaccards) if jaccards else 0,
        'kl_divergence': np.mean(kls) if kls else 0,
        'cosines': np.mean(cosines) if kls else 0,
        'n_clusters': n,
    }


def evaluate_all_strategies(cluster_summaries, cluster_assignments,
                             y_train, gt_mapper, checkpoint=None):
    """Evaluate all 4 labeling strategies and print comparison table.

    Args:
        cluster_summaries: from ClusterSummarizer
        cluster_assignments: (N,) cluster indices
        y_train: (N,) attack labels
        gt_mapper: GroundTruthMapper
        checkpoint: loaded checkpoint (for LLM labels if available)

    Returns:
        results: list of dicts with metrics per strategy
    """
    # Ground truth distributions
    gt_dists, gt_primaries = compute_cluster_gt_distribution(
        cluster_assignments, y_train, gt_mapper)

    results = []

    # ── Strategy 1: Mock Hard ──
    try:
        mock_hard_dists = {}
        mock_hard_primaries = {}
        labeler = LLMLabeler()
        for k, summary in cluster_summaries.items():
            k = int(k)
            result = labeler._mock_label(summary)
            primary_bhv = result.get('primary_behaviour', 'normal_browsing')
            from src.mapping.behaviour_mapping import get_primary_tactic
            primary_tactic = get_primary_tactic(primary_bhv)
            mock_hard_primaries[k] = primary_tactic
            # Hard = all weight on primary
            mock_hard_dists[k] = {primary_tactic: 1.0}

        results.append(evaluate_labeling_strategy(
            'Mock — Hard', mock_hard_dists, mock_hard_primaries,
            gt_dists, gt_primaries))
    except Exception as e:
        logger.warning(f"Mock Hard evaluation failed: {e}")

    # ── Strategy 2: Mock Soft ──
    try:
        from src.mapping.soft_cluster_scorer import SoftClusterScorer
        soft_scorer = SoftClusterScorer()
        cluster_scores = soft_scorer.score_all_clusters(cluster_summaries)

        mock_soft_dists = {}
        mock_soft_primaries = {}
        for k, cs in cluster_scores.items():
            k = int(k)
            mock_soft_dists[k] = cs['tactic_scores']
            mock_soft_primaries[k] = cs['primary_tactic']

        results.append(evaluate_labeling_strategy(
            'Mock — Soft', mock_soft_dists, mock_soft_primaries,
            gt_dists, gt_primaries))
    except Exception as e:
        logger.warning(f"Mock Soft evaluation failed: {e}")

    # ── Strategy 3: LLM Hard ──
    if checkpoint and 'cluster_behaviour_map_hard' in checkpoint:
        try:
            cbm = checkpoint['cluster_behaviour_map_hard']
            llm_hard_dists = {}
            llm_hard_primaries = {}
            for k_str, entry in cbm.items():
                k = int(k_str)
                primary_bhv = entry.get('primary_behaviour', 'normal_browsing')
                from src.mapping.behaviour_mapping import get_primary_tactic
                primary_tactic = get_primary_tactic(primary_bhv)
                llm_hard_primaries[k] = primary_tactic
                llm_hard_dists[k] = {primary_tactic: 1.0}

            results.append(evaluate_labeling_strategy(
                'LLM — Hard', llm_hard_dists, llm_hard_primaries,
                gt_dists, gt_primaries))
        except Exception as e:
            logger.warning(f"LLM Hard evaluation failed: {e}")

    # ── Strategy 4: LLM Soft ──
    # Uses LLM soft behaviour scores (from single LLM call)
    if checkpoint and 'cluster_behaviour_map_hard' in checkpoint:
        try:
            cbm = checkpoint['cluster_behaviour_map_hard']
            llm_soft_dists = {}
            llm_soft_primaries = {}
            for k_str, entry in cbm.items():
                k = int(k_str)
                if 'tactic_scores_llm' in entry:
                    llm_soft_dists[k] = entry['tactic_scores_llm']
                    llm_soft_primaries[k] = max(
                        entry['tactic_scores_llm'],
                        key=entry['tactic_scores_llm'].get)
                else:
                    primary_bhv = entry.get('primary_behaviour',
                                            'normal_browsing')
                    from src.mapping.behaviour_mapping import get_primary_tactic
                    primary_tactic = get_primary_tactic(primary_bhv)
                    llm_soft_primaries[k] = primary_tactic
                    llm_soft_dists[k] = {primary_tactic: 1.0}

            results.append(evaluate_labeling_strategy(
                'LLM — Soft', llm_soft_dists, llm_soft_primaries,
                gt_dists, gt_primaries))
        except Exception as e:
            logger.warning(f"LLM Soft evaluation failed: {e}")

    # ── Print comparison table ──
    print(f"\n{'='*65}")
    print("CLUSTER BEHAVIOUR LABELING EVALUATION")
    print(f"{'='*65}")
    print(f"  {'Strategy':<20s} {'Hard Agree':>12s} {'Soft Jaccard':>13s} {'KL Div':>10s} "
          f"{'Cosines':>10s}")
    print(f"  {'-'*57}")
    for r in results:
        print(f"  {r['name']:<20s} {r['hard_agreement']:>12.4f} "
              f"{r['soft_jaccard']:>13.4f} {r['kl_divergence']:>10.4f} {r['cosines']:>10.4f}")
    print(f"  {'-'*57}")
    print(f"  (Evaluated on {gt_dists and len(gt_dists) or 0} clusters)")

    return results