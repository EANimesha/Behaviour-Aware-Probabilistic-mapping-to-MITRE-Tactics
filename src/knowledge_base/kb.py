"""
Knowledge Base with Soft Behaviour + Tactic Distributions
============================================================

Stores per cluster:
  - behaviour_scores: soft distribution {behaviour: score}
  - tactic_scores: soft distribution {tactic: score}
  - primary_behaviour / primary_tactic: highest scoring
  - cluster_features: raw feature values for interpretability
  - cluster_size, cluster_proportion, p_gmm
"""

import json
import os
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self):
        self.entries: Dict[int, Dict[str, Any]] = {}
        self.metadata = {
            'n_clusters': 0,
            'n_labeled': 0,
            'dataset': '',
        }

    def add_entry(self, cluster_idx, behaviour_scores, tactic_scores,
                  primary_behaviour, primary_tactic,
                  cluster_features=None, cluster_size=0,
                  cluster_proportion=0.0, p_gmm=0.0):
        """Add or update a cluster entry with soft scores.

        Args:
            cluster_idx: cluster ID
            behaviour_scores: dict {behaviour: score} soft distribution
            tactic_scores: dict {tactic: score} soft distribution
            primary_behaviour: str highest scoring behaviour
            primary_tactic: str highest scoring tactic
            cluster_features: dict of raw feature values
            cluster_size: number of flows in cluster
            cluster_proportion: fraction of total flows
            p_gmm: GMM prior probability
        """
        self.entries[cluster_idx] = {
            'behaviour_scores': behaviour_scores,
            'primary_behaviour': primary_behaviour,
            'tactic_scores': tactic_scores,
            'primary_tactic': primary_tactic,
            'cluster_size': int(cluster_size),
            'cluster_proportion': round(float(cluster_proportion), 4),
            'cluster_features': cluster_features or {},
            'p_gmm': round(float(p_gmm), 6),
        }
        self.metadata['n_clusters'] = len(self.entries)
        self.metadata['n_labeled'] = sum(
            1 for e in self.entries.values()
            if e['primary_tactic'] != 'Unknown')

        # Log top-3 behaviours
        top3 = sorted(behaviour_scores.items(), key=lambda x: -x[1])[:3]
        bhv_str = ', '.join(f"{b}:{s:.2f}" for b, s in top3)
        logger.info(f"KB: Cluster {cluster_idx} ({cluster_size:,}) "
                     f"[{bhv_str}] -> {primary_tactic}")

    def add_from_soft_scorer(self, cluster_idx, scorer_result,
                              cluster_size=0, cluster_proportion=0.0,
                              p_gmm=0.0, cluster_features=None):
        """Add entry directly from SoftClusterScorer output.

        Args:
            scorer_result: dict from SoftClusterScorer.score_cluster()
        """
        self.add_entry(
            cluster_idx=cluster_idx,
            behaviour_scores=scorer_result['behaviour_scores'],
            tactic_scores=scorer_result['tactic_scores'],
            primary_behaviour=scorer_result['primary_behaviour'],
            primary_tactic=scorer_result['primary_tactic'],
            cluster_features=cluster_features,
            cluster_size=cluster_size,
            cluster_proportion=cluster_proportion,
            p_gmm=p_gmm)

    # ── Lookup methods ──

    def lookup(self, cluster_idx):
        return self.entries.get(cluster_idx, None)

    def get_tactic(self, cluster_idx):
        entry = self.lookup(cluster_idx)
        return entry['primary_tactic'] if entry else 'Unknown'

    def get_behaviour(self, cluster_idx):
        entry = self.lookup(cluster_idx)
        return entry['primary_behaviour'] if entry else 'unknown'

    def get_behaviour_scores(self, cluster_idx):
        entry = self.lookup(cluster_idx)
        if entry:
            return entry.get('behaviour_scores', {})
        return {}

    def get_tactic_scores(self, cluster_idx):
        entry = self.lookup(cluster_idx)
        if entry:
            return entry.get('tactic_scores', {})
        return {}

    # ── Backward compatibility ──

    def get_behaviour_label(self, cluster_idx):
        return self.get_behaviour(cluster_idx)

    def get_p_llm(self, cluster_idx):
        entry = self.lookup(cluster_idx)
        if entry:
            scores = entry.get('behaviour_scores', {})
            return max(scores.values()) if scores else 0.5
        return 0.5

    def get_all_tactics(self, cluster_idx):
        entry = self.lookup(cluster_idx)
        if entry:
            ts = entry.get('tactic_scores', {})
            return [{'tactic': t, 'confidence': s}
                    for t, s in sorted(ts.items(), key=lambda x: -x[1])
                    if s > 0.05]
        return []

    # ── Save / Load ──

    def save(self, filepath):
        data = {
            'metadata': self.metadata,
            'entries': {str(k): v for k, v in self.entries.items()}
        }
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"KB saved: {filepath} ({len(self.entries)} clusters)")

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.metadata = data.get('metadata', {})
        self.entries = {int(k): v for k, v in data['entries'].items()}
        logger.info(f"KB loaded: {filepath} ({len(self.entries)} clusters)")

    def __repr__(self):
        lines = [f"KnowledgeBase({len(self.entries)} clusters)"]
        for k, v in sorted(self.entries.items()):
            bhv = v.get('primary_behaviour', '?')
            tactic = v.get('primary_tactic', '?')
            size = v.get('cluster_size', 0)
            # Top behaviour scores
            bs = v.get('behaviour_scores', {})
            top = sorted(bs.items(), key=lambda x: -x[1])[:3]
            score_str = ', '.join(f"{b}:{s:.2f}" for b, s in top)
            lines.append(f"  Cluster {k:3d} ({size:>6,}): "
                          f"[{score_str}] -> {tactic}")
        return "\n".join(lines)

# """
# Knowledge Base (KB) for Cluster-Tactic Mapping
# ================================================
#
# Stores complete knowledge per cluster k:
#     cluster_kb[k] = {
#         "tactic": "DoS",               # From LLM
#         "tactic_id": "TA0040",         # MITRE ID
#         "p_gmm": 0.20,                 # From DP-GMM (cluster weight w_k)
#         "p_llm": 0.92,                 # From LLM reasoning confidence
#         "reasoning": "high volume ...", # Why this tactic
#         "fusion_weights": [0.6, 0.4],  # w1 (GMM), w2 (LLM) for Bayesian fusion
#         "summary_text": "...",          # Full cluster summary
#     }
#
# Used during inference for fast lookup and during streaming for
# consistency checking and drift-triggered re-labeling.
# """
#
# import json
# import os
# import logging
# from typing import Dict, Optional, Any
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class KnowledgeBase:
#     """Neuro-symbolic knowledge base for cluster-tactic associations.
#
#     Persisted to disk as JSON for reuse across training/inference/streaming.
#     """
#
#     def __init__(self):
#         self.entries: Dict[int, Dict[str, Any]] = {}
#         self.metadata = {
#             'n_clusters': 0,
#             'n_labeled': 0,
#             'default_fusion_weights': [0.6, 0.4],  # w1=GMM, w2=LLM
#         }
#
#     def add_entry(self, cluster_idx, tactic, tactic_id, p_gmm, p_llm,
#                   reasoning, technique_id='Unknown', technique_name='Unknown',
#                   summary_text="", fusion_weights=None):
#         """Add or update a cluster entry in the KB.
#
#         Args:
#             cluster_idx: DP-GMM cluster index
#             tactic: MITRE ATT&CK tactic name
#             tactic_id: MITRE tactic ID (e.g. "TA0040")
#             technique_id: MITRE technique ID (e.g. "T1499")
#             technique_name: MITRE technique name
#             p_gmm: cluster weight from DP-GMM (prior probability)
#             p_llm: LLM confidence in tactic assignment
#             reasoning: LLM explanation for the labeling
#             summary_text: full cluster summary from summarizer
#             fusion_weights: [w1, w2] for Bayesian fusion (default: [0.6, 0.4])
#         """
#         self.entries[cluster_idx] = {
#             'tactic': tactic,
#             'tactic_id': tactic_id,
#             'technique_id': technique_id,
#             'technique_name': technique_name,
#             'p_gmm': float(p_gmm),
#             'p_llm': float(p_llm),
#             'reasoning': reasoning,
#             'summary_text': summary_text,
#             'fusion_weights': fusion_weights or self.metadata['default_fusion_weights'],
#         }
#         self.metadata['n_clusters'] = len(self.entries)
#         self.metadata['n_labeled'] = sum(
#             1 for e in self.entries.values() if e['tactic'] != 'Unknown'
#         )
#
#         logger.info(f"KB: Cluster {cluster_idx} -> {tactic} ({tactic_id}), "
#                      f"P_GMM={p_gmm:.3f}, P_LLM={p_llm:.3f}")
#
#     def lookup(self, cluster_idx):
#         """Retrieve KB entry for a cluster.
#
#         Returns:
#             entry dict or None if not found
#         """
#         return self.entries.get(cluster_idx, None)
#
#     def get_tactic(self, cluster_idx):
#         """Get tactic label for a cluster."""
#         entry = self.lookup(cluster_idx)
#         return entry['tactic'] if entry else 'Unknown'
#
#     def get_p_llm(self, cluster_idx):
#         """Get LLM confidence for a cluster."""
#         entry = self.lookup(cluster_idx)
#         return entry['p_llm'] if entry else 0.5
#
#     def get_fusion_weights(self, cluster_idx):
#         """Get fusion weights [w1_gmm, w2_llm] for a cluster."""
#         entry = self.lookup(cluster_idx)
#         if entry and 'fusion_weights' in entry:
#             return entry['fusion_weights']
#         return self.metadata['default_fusion_weights']
#
#     def get_all_tactics(self):
#         """Get unique set of all assigned tactics."""
#         return list(set(e['tactic'] for e in self.entries.values()))
#
#     def consistency_check(self, cluster_idx, new_summary_stats):
#         """Check if a cluster's current KB entry is still consistent.
#
#         Used during streaming to detect if a cluster has drifted
#         and needs re-labeling.
#
#         Args:
#             cluster_idx: cluster to check
#             new_summary_stats: updated cluster statistics
#
#         Returns:
#             is_consistent: bool
#             reason: str explaining inconsistency
#         """
#         entry = self.lookup(cluster_idx)
#         if entry is None:
#             return False, "No existing entry"
#
#         # Simple heuristic: check if cluster size changed dramatically
#         # In production, would compare feature distributions
#         return True, "OK"
#
#     def update_fusion_weights(self, cluster_idx, w1_gmm, w2_llm):
#         """Update fusion weights for a specific cluster.
#
#         Called during streaming when drift is detected.
#         """
#         if cluster_idx in self.entries:
#             self.entries[cluster_idx]['fusion_weights'] = [
#                 float(w1_gmm), float(w2_llm)
#             ]
#             logger.info(f"KB: Updated fusion weights for cluster {cluster_idx}: "
#                          f"w1={w1_gmm:.2f}, w2={w2_llm:.2f}")
#
#     def save(self, filepath):
#         """Save KB to JSON file."""
#         data = {
#             'metadata': self.metadata,
#             'entries': {str(k): v for k, v in self.entries.items()},
#         }
#         os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
#         with open(filepath, 'w') as f:
#             json.dump(data, f, indent=2)
#         logger.info(f"KB saved to {filepath} ({len(self.entries)} entries)")
#
#     def load(self, filepath):
#         """Load KB from JSON file."""
#         with open(filepath, 'r') as f:
#             data = json.load(f)
#         self.metadata = data['metadata']
#         self.entries = {int(k): v for k, v in data['entries'].items()}
#         logger.info(f"KB loaded from {filepath} ({len(self.entries)} entries)")
#
#     def __repr__(self):
#         lines = [f"KnowledgeBase({len(self.entries)} clusters)"]
#         for k, v in sorted(self.entries.items()):
#             lines.append(f"  Cluster {k}: {v['tactic']} "
#                           f"(P_GMM={v['p_gmm']:.3f}, P_LLM={v['p_llm']:.3f})")
#         return "\n".join(lines)