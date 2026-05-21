"""
Bayesian Fusion Module
=======================

Stage 6 of the training pipeline (setup) and core of the inference pipeline.

Fusion method: Weighted Average (Bayesian)
    P_final = w1 * P(GMM) + w2 * P(LLM)
    where w1 + w2 = 1

During training: configure default fusion weights per cluster.
During inference: apply fusion to combine GMM and LLM predictions.
During streaming: adapt weights based on drift detection.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianFusion:
    """Fuse P(GMM) and P(LLM) for final tactic prediction."""

    def __init__(self, default_w1=0.6, default_w2=0.4):
        """
        Args:
            default_w1: weight for P(GMM)
            default_w2: weight for P(LLM)
        """
        assert abs(default_w1 + default_w2 - 1.0) < 1e-6, "Weights must sum to 1"
        self.default_w1 = default_w1
        self.default_w2 = default_w2

    def fuse(self, p_gmm, p_llm, w1=None, w2=None):
        """Compute fused probability.

        Args:
            p_gmm: float, GMM confidence / probability of cluster membership
            p_llm: float, LLM confidence in tactic assignment
            w1: optional override for GMM weight
            w2: optional override for LLM weight

        Returns:
            p_final: fused probability
        """
        w1 = w1 if w1 is not None else self.default_w1
        w2 = w2 if w2 is not None else self.default_w2
        p_final = w1 * p_gmm + w2 * p_llm
        return float(np.clip(p_final, 0.0, 1.0))

    def fuse_with_kb(self, p_gmm, cluster_idx, knowledge_base):
        """Fuse using per-cluster weights and P_LLM from the Knowledge Base.

        Args:
            p_gmm: float, GMM confidence for this prediction
            cluster_idx: int, cluster index to lookup in KB
            knowledge_base: KnowledgeBase instance

        Returns:
            result dict with tactic, p_final, p_gmm, p_llm, weights
        """
        entry = knowledge_base.lookup(cluster_idx)
        if entry is None:
            return {
                'tactic': 'Unknown',
                'p_final': float(p_gmm),
                'p_gmm': float(p_gmm),
                'p_llm': 0.5,
                'w1': self.default_w1,
                'w2': self.default_w2,
                'reasoning': 'No KB entry for this cluster',
            }

        w1, w2 = entry['fusion_weights']
        p_llm = entry['p_llm']
        p_final = self.fuse(p_gmm, p_llm, w1, w2)

        return {
            'tactic': entry['tactic'],
            'tactic_id': entry.get('tactic_id', 'Unknown'),
            'p_final': p_final,
            'p_gmm': float(p_gmm),
            'p_llm': p_llm,
            'w1': w1,
            'w2': w2,
            'reasoning': entry.get('reasoning', ''),
        }

    def adapt_weights_for_drift(self, w1, w2, drift_detected):
        """Adapt fusion weights when drift is detected.

        When drift is detected, trust the statistical model more
        because the LLM labels may be stale.

        Args:
            w1, w2: current weights
            drift_detected: bool

        Returns:
            new_w1, new_w2
        """
        if drift_detected:
            # Shift trust toward GMM (fresh statistics)
            new_w1 = min(w1 + 0.1, 0.8)
            new_w2 = 1.0 - new_w1
            logger.info(f"Drift detected: adapting weights w1={new_w1:.2f}, "
                         f"w2={new_w2:.2f}")
            return new_w1, new_w2
        return w1, w2
