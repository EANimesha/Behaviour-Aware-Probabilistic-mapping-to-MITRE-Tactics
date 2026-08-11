# """
# Bayesian Fusion Module
# =======================
#
# Stage 6 of the training pipeline (setup) and core of the inference pipeline.
#
# Fusion method: Weighted Average (Bayesian)
#     P_final = w1 * P(GMM) + w2 * P(LLM)
#     where w1 + w2 = 1
#
# During training: configure default fusion weights per cluster.
# During inference: apply fusion to combine GMM and LLM predictions.
# During streaming: adapt weights based on drift detection.
# """
#
# import numpy as np
# import logging
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class BayesianFusion:
#     """Fuse P(GMM) and P(LLM) for final tactic prediction."""
#
#     def __init__(self, default_w1=0.6, default_w2=0.4):
#         """
#         Args:
#             default_w1: weight for P(GMM)
#             default_w2: weight for P(LLM)
#         """
#         assert abs(default_w1 + default_w2 - 1.0) < 1e-6, "Weights must sum to 1"
#         self.default_w1 = default_w1
#         self.default_w2 = default_w2
#
#     def fuse(self, p_gmm, p_llm, w1=None, w2=None):
#         """Compute fused probability.
#
#         Args:
#             p_gmm: float, GMM confidence / probability of cluster membership
#             p_llm: float, LLM confidence in tactic assignment
#             w1: optional override for GMM weight
#             w2: optional override for LLM weight
#
#         Returns:
#             p_final: fused probability
#         """
#         w1 = w1 if w1 is not None else self.default_w1
#         w2 = w2 if w2 is not None else self.default_w2
#         p_final = w1 * p_gmm + w2 * p_llm
#         return float(np.clip(p_final, 0.0, 1.0))
#
#     def fuse_with_kb(self, p_gmm, cluster_idx, knowledge_base):
#         """Fuse using per-cluster weights and P_LLM from the Knowledge Base.
#
#         Args:
#             p_gmm: float, GMM confidence for this prediction
#             cluster_idx: int, cluster index to lookup in KB
#             knowledge_base: KnowledgeBase instance
#
#         Returns:
#             result dict with tactic, p_final, p_gmm, p_llm, weights
#         """
#         entry = knowledge_base.lookup(cluster_idx)
#         if entry is None:
#             return {
#                 'tactic': 'Unknown',
#                 'p_final': float(p_gmm),
#                 'p_gmm': float(p_gmm),
#                 'p_llm': 0.5,
#                 'w1': self.default_w1,
#                 'w2': self.default_w2,
#                 'reasoning': 'No KB entry for this cluster',
#             }
#
#         w1, w2 = entry['fusion_weights']
#         p_llm = entry['p_llm']
#         p_final = self.fuse(p_gmm, p_llm, w1, w2)
#
#         return {
#             'tactic': entry['tactic'],
#             'tactic_id': entry.get('tactic_id', 'Unknown'),
#             'p_final': p_final,
#             'p_gmm': float(p_gmm),
#             'p_llm': p_llm,
#             'w1': w1,
#             'w2': w2,
#             'reasoning': entry.get('reasoning', ''),
#         }
#
#     def adapt_weights_for_drift(self, w1, w2, drift_detected):
#         """Adapt fusion weights when drift is detected.
#
#         When drift is detected, trust the statistical model more
#         because the LLM labels may be stale.
#
#         Args:
#             w1, w2: current weights
#             drift_detected: bool
#
#         Returns:
#             new_w1, new_w2
#         """
#         if drift_detected:
#             # Shift trust toward GMM (fresh statistics)
#             new_w1 = min(w1 + 0.1, 0.8)
#             new_w2 = 1.0 - new_w1
#             logger.info(f"Drift detected: adapting weights w1={new_w1:.2f}, "
#                          f"w2={new_w2:.2f}")
#             return new_w1, new_w2
#         return w1, w2

"""
Bayesian Fusion: GMM Posteriors × KB Soft Tactic Scores
=========================================================

Combines two independent signals:
  1. GMM posterior P(cluster=k | flow) — geometric membership
  2. KB tactic scores P(tactic | cluster=k) — semantic meaning

Formula:
  P(tactic | flow) = Σ_k P(cluster=k | flow) × P(tactic | cluster=k)

Window aggregation:
  P(tactic | window) = mean of P(tactic | flow) across flows in window
"""

import numpy as np
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianFusion:
    """Fuses GMM cluster posteriors with KB tactic distributions."""

    def __init__(self, kb, tactic_threshold=0.10):
        """
        Args:
            kb: KnowledgeBase with soft tactic scores per cluster
            tactic_threshold: minimum score to include in final_tactics
        """
        self.kb = kb
        self.tactic_threshold = tactic_threshold

        # Pre-load KB tactic distributions for all clusters
        self._cluster_tactic_dists = {}
        self._cluster_bhv_dists = {}
        for k, entry in kb.entries.items():
            self._cluster_tactic_dists[int(k)] = entry.get(
                'tactic_scores', {})
            self._cluster_bhv_dists[int(k)] = entry.get(
                'behaviour_scores', {})

        logger.info(f"BayesianFusion: {len(self._cluster_tactic_dists)} "
                     f"clusters loaded from KB")

    def fuse_flow(self, gmm_posterior):
        """Compute fused tactic distribution for a single flow.

        Args:
            gmm_posterior: (K,) array of P(cluster=k | flow) from GMM

        Returns:
            tactic_dist: dict {tactic: probability}
            bhv_dist: dict {behaviour: probability}
            primary_tactic: str
        """
        tactic_dist = defaultdict(float)
        bhv_dist = defaultdict(float)

        for k, p_k in enumerate(gmm_posterior):
            if p_k < 1e-6:
                continue

            # Tactic fusion
            kb_tactics = self._cluster_tactic_dists.get(k, {})
            for tactic, p_tactic in kb_tactics.items():
                tactic_dist[tactic] += p_k * p_tactic

            # Behaviour fusion
            kb_bhvs = self._cluster_bhv_dists.get(k, {})
            for bhv, p_bhv in kb_bhvs.items():
                bhv_dist[bhv] += p_k * p_bhv

        # Normalize
        tactic_dist = dict(tactic_dist)
        bhv_dist = dict(bhv_dist)
        t_total = sum(tactic_dist.values())
        if t_total > 0:
            tactic_dist = {t: s / t_total for t, s in tactic_dist.items()}
        b_total = sum(bhv_dist.values())
        if b_total > 0:
            bhv_dist = {b: s / b_total for b, s in bhv_dist.items()}

        primary = max(tactic_dist, key=tactic_dist.get) if tactic_dist else 'Unknown'

        return tactic_dist, bhv_dist, primary

    def fuse_all_flows(self, gmm_posteriors):
        """Compute fused tactic distributions for all flows.

        Args:
            gmm_posteriors: (N, K) array from gmm.predict_proba(Z)

        Returns:
            flow_tactic_dists: list of dicts
            flow_primaries: (N,) array of primary tactics
            flow_bhv_dists: list of dicts
        """
        N = len(gmm_posteriors)
        flow_tactic_dists = []
        flow_bhv_dists = []
        flow_primaries = []

        for i in range(N):
            tactic_dist, bhv_dist, primary = self.fuse_flow(
                gmm_posteriors[i])
            flow_tactic_dists.append(tactic_dist)
            flow_bhv_dists.append(bhv_dist)
            flow_primaries.append(primary)

        return flow_tactic_dists, np.array(flow_primaries), flow_bhv_dists

    def create_windows(self, src_ips, gmm_posteriors, window_size=50,
                       y_true=None):
        """Create time windows with Bayesian-fused tactic predictions.

        Args:
            src_ips: (N,) source IP per flow
            gmm_posteriors: (N, K) from gmm.predict_proba(Z)
            window_size: flows per window
            y_true: optional ground truth labels

        Returns:
            windows: list of window dicts
        """
        N = len(src_ips)

        # Fuse all flows first
        flow_tactic_dists, flow_primaries, flow_bhv_dists = \
            self.fuse_all_flows(gmm_posteriors)

        # Group by source IP
        src_flows = defaultdict(list)
        for i in range(N):
            src_flows[src_ips[i]].append(i)

        windows = []
        for src_ip, indices in src_flows.items():
            for w_start in range(0, len(indices), window_size):
                w_end = min(w_start + window_size, len(indices))
                w_idx = indices[w_start:w_end]
                n = len(w_idx)

                # Aggregate fused tactic distributions across window
                agg_tactic = defaultdict(float)
                agg_bhv = defaultdict(float)

                for i in w_idx:
                    for t, s in flow_tactic_dists[i].items():
                        agg_tactic[t] += s / n
                    for b, s in flow_bhv_dists[i].items():
                        agg_bhv[b] += s / n

                # Normalize
                t_total = sum(agg_tactic.values())
                if t_total > 0:
                    agg_tactic = {t: s / t_total
                                  for t, s in agg_tactic.items()}
                b_total = sum(agg_bhv.values())
                if b_total > 0:
                    agg_bhv = {b: s / b_total for b, s in agg_bhv.items()}

                primary = max(agg_tactic, key=agg_tactic.get) \
                    if agg_tactic else 'Unknown'
                final = sorted(
                    [t for t, s in agg_tactic.items()
                     if s >= self.tactic_threshold],
                    key=lambda t: agg_tactic.get(t, 0), reverse=True)

                window = {
                    'src_ip': src_ip,
                    'flow_indices': w_idx,
                    'n_flows': n,
                    'tactic_scores': dict(agg_tactic),
                    'behaviour_scores': dict(agg_bhv),
                    'primary_tactic': primary,
                    'final_tactics': final,
                }

                if y_true is not None:
                    w_attacks = [y_true[i] for i in w_idx]
                    majority = Counter(w_attacks).most_common(1)[0]
                    window['majority_attack'] = majority[0]
                    window['attack_purity'] = majority[1] / n
                    window['true_attacks'] = w_attacks

                windows.append(window)

        return windows