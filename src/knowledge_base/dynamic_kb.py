"""
Dynamic Knowledge Base
========================

Wraps the static KB with adaptive update capabilities:
  1. Detects cluster drift (centroid shift after online EM)
  2. Re-scores drifted clusters using SoftClusterScorer
  3. Creates new entries for novel clusters from accumulated novel flows
  4. Tracks update history for audit/analysis

Usage:
    dkb = DynamicKB(kb, scorer, gmm, drift_threshold=0.5)

    for batch in stream:
        # Predict with current KB
        predictions = dkb.predict_flows(clusters)

        # After online EM update
        dkb.check_and_update(gmm, Z_batch, clusters, raw_features)
"""

import numpy as np
import logging
from collections import defaultdict
from copy import deepcopy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicKB:
    """Adaptive Knowledge Base that evolves with online GMM updates."""

    def __init__(self, kb, scorer, gmm_model,
                 drift_threshold=0.5,
                 novelty_buffer_size=50,
                 dataset='bot_iot'):
        """
        Args:
            kb: static KnowledgeBase (baseline)
            scorer: SoftClusterScorer instance
            gmm_model: fitted GMM model
            drift_threshold: centroid shift (in std units) to trigger re-score
            novelty_buffer_size: min novel flows before creating new entry
            dataset: for scorer calibration
        """
        self.kb = deepcopy(kb)
        self.scorer = scorer
        self.dataset = dataset
        self.drift_threshold = drift_threshold
        self.novelty_buffer_size = novelty_buffer_size

        # Track cluster centroids from GMM
        self._prev_means = {}
        self._prev_stds = {}
        if hasattr(gmm_model, 'means_'):
            for k in range(len(gmm_model.means_)):
                if gmm_model.weights_[k] > 0.01:
                    self._prev_means[k] = gmm_model.means_[k].copy()
                    if hasattr(gmm_model, 'covariances_'):
                        cov = gmm_model.covariances_[k]
                        if cov.ndim == 1:
                            self._prev_stds[k] = np.sqrt(cov)
                        else:
                            self._prev_stds[k] = np.sqrt(np.diag(cov))
                    else:
                        self._prev_stds[k] = np.ones(len(gmm_model.means_[k]))

        # Novelty buffer: accumulate flows that don't fit any cluster
        self.novelty_buffer = []
        self.novelty_labels = []

        # Update history
        self.update_log = []
        self.n_updates = 0
        self.n_new_clusters = 0

    def predict_flow(self, cluster_idx):
        """Get tactic prediction for a flow from current KB."""
        entry = self.kb.lookup(int(cluster_idx))
        if entry:
            return {
                'tactic_scores': entry['tactic_scores'],
                'primary_tactic': entry['primary_tactic'],
                'primary_behaviour': entry['primary_behaviour'],
            }
        return {
            'tactic_scores': {'Unknown': 1.0},
            'primary_tactic': 'Unknown',
            'primary_behaviour': 'unknown',
        }

    def predict_flows(self, clusters):
        """Get tactic distributions for all flows."""
        tactic_dists = []
        primaries = []
        for c in clusters:
            pred = self.predict_flow(c)
            tactic_dists.append(pred['tactic_scores'])
            primaries.append(pred['primary_tactic'])
        return tactic_dists, np.array(primaries)

    def check_and_update(self, gmm_model, Z_batch, clusters,
                         feature_stats=None, y_batch=None):
        """Check for cluster drift and update KB if needed.

        Call this AFTER online EM update.

        Args:
            gmm_model: GMM model (after online EM update)
            Z_batch: (N, z_dim) embeddings of current batch
            clusters: (N,) cluster assignments
            feature_stats: dict per cluster of raw feature summaries
                           (from batch, for re-scoring)
            y_batch: optional attack labels (for logging only)
        """
        updated_clusters = []
        new_clusters = []

        # 1. Check centroid drift for each active cluster
        for k in range(len(gmm_model.means_)):
            if gmm_model.weights_[k] < 0.01:
                continue

            new_mean = gmm_model.means_[k]

            if k in self._prev_means:
                old_mean = self._prev_means[k]
                std = self._prev_stds.get(k, np.ones_like(old_mean))
                std = np.maximum(std, 1e-6)

                # Normalised shift (in std units)
                shift = np.linalg.norm((new_mean - old_mean) / std)

                if shift > self.drift_threshold:
                    # Cluster has drifted — re-score
                    mask = clusters == k
                    n_flows = mask.sum()

                    if n_flows > 10 and feature_stats and k in feature_stats:
                        # Re-score with current feature stats
                        summary = feature_stats[k]
                        result = self.scorer.score_cluster(summary)

                        # Update KB entry
                        self.kb.entries[str(k)] = {
                            'behaviour_scores': result['behaviour_scores'],
                            'tactic_scores': result['tactic_scores'],
                            'primary_behaviour': result['primary_behaviour'],
                            'primary_tactic': result['primary_tactic'],
                            'cluster_size': n_flows,
                            'cluster_proportion': 0.0,
                            'p_gmm': float(gmm_model.weights_[k]),
                            'updated': True,
                        }

                        updated_clusters.append(k)
                        self.n_updates += 1

                        self.update_log.append({
                            'type': 'drift_update',
                            'cluster': k,
                            'shift': float(shift),
                            'new_behaviour': result['primary_behaviour'],
                            'new_tactic': result['primary_tactic'],
                            'n_flows': n_flows,
                        })

                        logger.info(
                            f"DynamicKB: Cluster {k} drifted (shift={shift:.3f})"
                            f" → re-scored: {result['primary_behaviour']}"
                            f" → {result['primary_tactic']}")

            # Update tracked centroids
            self._prev_means[k] = new_mean.copy()
            if hasattr(gmm_model, 'covariances_'):
                cov = gmm_model.covariances_[k]
                if cov.ndim == 1:
                    self._prev_stds[k] = np.sqrt(cov)
                else:
                    self._prev_stds[k] = np.sqrt(np.diag(cov))

        # 2. Check for novel clusters (not in KB)
        for k in np.unique(clusters):
            k = int(k)
            if str(k) not in self.kb.entries and gmm_model.weights_[k] > 0.01:
                mask = clusters == k
                n_flows = mask.sum()

                if n_flows > self.novelty_buffer_size:
                    if feature_stats and k in feature_stats:
                        result = self.scorer.score_cluster(feature_stats[k])
                        self.kb.entries[str(k)] = {
                            'behaviour_scores': result['behaviour_scores'],
                            'tactic_scores': result['tactic_scores'],
                            'primary_behaviour': result['primary_behaviour'],
                            'primary_tactic': result['primary_tactic'],
                            'cluster_size': n_flows,
                            'cluster_proportion': 0.0,
                            'p_gmm': float(gmm_model.weights_[k]),
                            'is_new': True,
                        }

                        new_clusters.append(k)
                        self.n_new_clusters += 1

                        self.update_log.append({
                            'type': 'new_cluster',
                            'cluster': k,
                            'behaviour': result['primary_behaviour'],
                            'tactic': result['primary_tactic'],
                            'n_flows': n_flows,
                        })

                        logger.info(
                            f"DynamicKB: New cluster {k} ({n_flows} flows)"
                            f" → {result['primary_behaviour']}"
                            f" → {result['primary_tactic']}")

        return updated_clusters, new_clusters

    def get_stats(self):
        """Return update statistics."""
        return {
            'n_updates': self.n_updates,
            'n_new_clusters': self.n_new_clusters,
            'kb_size': len(self.kb.entries),
            'update_log': self.update_log,
        }


def compute_batch_feature_stats(Z_batch, clusters, gmm_model):
    """Compute simple feature summaries per cluster from embeddings.

    Since we don't have raw features in streaming, use embedding
    statistics as proxy for re-scoring.
    """
    stats = {}
    for k in np.unique(clusters):
        k = int(k)
        mask = clusters == k
        if mask.sum() < 5:
            continue

        z_cluster = Z_batch[mask]
        z_mean = z_cluster.mean(axis=0)
        z_std = z_cluster.std(axis=0)

        # Create a minimal summary compatible with SoftClusterScorer
        # Use GMM parameters as proxy features
        if k < len(gmm_model.means_):
            gmm_mean = gmm_model.means_[k]
        else:
            gmm_mean = z_mean

        # Map embedding dimensions to approximate feature values
        # This is a rough proxy — the scorer will use whatever it gets
        stats[k] = {
            'stats': {
                'behaviour': {
                    'avg_tcp_flags': float(np.abs(gmm_mean[0]) * 10),
                    'avg_out_bytes': float(np.abs(gmm_mean[1]) * 1000),
                    'avg_in_bytes': float(np.abs(gmm_mean[2]) * 1000),
                    'avg_min_ttl': float(np.abs(gmm_mean[3]) * 30),
                    'avg_duration_ms': float(np.abs(gmm_mean[4]) * 100000),
                    'avg_dst_port': float(np.abs(gmm_mean[5]) * 1000),
                    'packets_per_second': float(np.abs(gmm_mean[6]) * 5000),
                    'throughput_ratio': float(np.abs(gmm_mean[7])),
                    'bytes_per_packet': float(np.abs(gmm_mean[8]) * 100),
                    'avg_protocol': 6.0,
                    'avg_longest_pkt_bytes': float(np.abs(gmm_mean[9]) * 500)
                    if len(gmm_mean) > 9 else 0,
                    'avg_shortest_pkt_bytes': float(np.abs(gmm_mean[10]) * 50)
                    if len(gmm_mean) > 10 else 0,
                    'traffic_direction': '',
                },
                'graph_properties': {
                    'avg_fan_out': float(np.abs(gmm_mean[11]) * 5 + 1)
                    if len(gmm_mean) > 11 else 1.0,
                    'avg_fan_in': float(np.abs(gmm_mean[12]) * 5 + 1)
                    if len(gmm_mean) > 12 else 1.0,
                    'n_unique_destinations': int(mask.sum() * 0.1),
                },
                'proportion': float(mask.sum()) / len(Z_batch),
            }
        }

    return stats