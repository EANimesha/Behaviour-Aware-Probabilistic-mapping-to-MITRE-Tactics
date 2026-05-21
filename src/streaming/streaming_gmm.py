"""
Streaming Dirichlet Process Gaussian Mixture Model
====================================================

Unsupervised clustering of latent embeddings with:
  - Batch fitting on training data
  - Online EM updates for streaming windows
  - New cluster creation when no match found
  - Novelty detection for zero-day flagging
  - Parameter refinement via partial_fit-style updates
"""

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import logging
from copy import deepcopy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingDPGMM:
    """
    DP-GMM supporting both batch and streaming operation.

    Batch phase:  fit() on training embeddings
    Streaming:    update_online_em() per window of new flows
    """

    def __init__(self, max_components=20, covariance_type='diag',
                 weight_threshold=0.01, novelty_threshold=None,
                 learning_rate=0.01, drift_detector=None):
        self.max_components = max_components
        self.covariance_type = covariance_type
        self.weight_threshold = weight_threshold
        self.novelty_threshold = novelty_threshold
        self.learning_rate = learning_rate
        self.drift_detector = drift_detector

        self.model = None
        self.n_samples_seen = 0
        self.cluster_samples = {}
        self.cluster_to_tactic = {}

        # Track previous weights for drift detection
        self._prev_weights = None
        self._prev_means = None

    # ──────────────────────────────────────────────────────
    # Batch phase
    # ──────────────────────────────────────────────────────

    def fit(self, Z_batch, novelty_n_sigma=3.0):
        """Batch phase: fit DP-GMM on initial dataset.

        Args:
            Z_batch: (N, z_dim) embeddings
            novelty_n_sigma: number of standard deviations below the mean
                             training log-likelihood to set as novelty threshold.
                             Default 3.0 means only samples > 3σ worse than
                             the training average are flagged as novel.

        Returns:
            cluster_assignments: (N,)
            log_likelihoods: (N,)
        """
        logger.info(f"Fitting DP-GMM on {len(Z_batch)} samples")

        self.model = BayesianGaussianMixture(
            n_components=self.max_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior_type='dirichlet_process',
            n_init=10,
            random_state=42,
        )

        self.model.fit(Z_batch)
        self.n_samples_seen = len(Z_batch)

        cluster_assignments = self.model.predict(Z_batch)
        log_likelihoods = self.model.score_samples(Z_batch)

        # ── Calibrate novelty threshold from training data ──
        # Use mean - n_sigma * std of training log-likelihoods.
        # Robust to train/test distribution shift because it accounts
        # for the natural spread in log-likelihoods.
        ll_mean = float(np.mean(log_likelihoods))
        ll_std = float(np.std(log_likelihoods))
        self.novelty_threshold = ll_mean - novelty_n_sigma * ll_std

        # Store stats for debugging/checkpoint
        self._ll_stats = {
            'mean': ll_mean,
            'std': ll_std,
            'min': float(log_likelihoods.min()),
            'max': float(log_likelihoods.max()),
            'median': float(np.median(log_likelihoods)),
            'n_sigma': novelty_n_sigma,
            'threshold': self.novelty_threshold,
        }

        train_novel = np.sum(log_likelihoods < self.novelty_threshold)
        logger.info(f"Training log-likelihood: mean={ll_mean:.2f}, "
                     f"std={ll_std:.2f}, "
                     f"range=[{log_likelihoods.min():.2f}, "
                     f"{log_likelihoods.max():.2f}]")
        logger.info(f"Novelty threshold: {self.novelty_threshold:.2f} "
                     f"(mean - {novelty_n_sigma}σ), "
                     f"flagged {train_novel}/{len(Z_batch)} training samples "
                     f"({train_novel/len(Z_batch)*100:.2f}%)")

        # Track cluster samples
        for k in range(self.model.n_components):
            mask = cluster_assignments == k
            if mask.sum() > 0:
                self.cluster_samples[k] = {
                    'samples': Z_batch[mask],
                    'size': int(mask.sum()),
                    'weight': float(self.model.weights_[k]),
                }

        # Snapshot for drift detection
        self._prev_weights = self.model.weights_.copy()
        self._prev_means = self.model.means_.copy()

        active = np.sum(self.model.weights_ > self.weight_threshold)
        logger.info(f"Fitted: {active} active clusters out of {self.max_components}")

        return cluster_assignments, log_likelihoods

    # ──────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────

    def predict(self, Z_new):
        """Predict clusters for new samples.

        Returns:
            assignments, soft_probs, log_likelihoods, novelty_flags
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        assignments = self.model.predict(Z_new)
        soft_probs = self.model.predict_proba(Z_new)
        log_likelihoods = self.model.score_samples(Z_new)

        if self.novelty_threshold is not None:
            novelty_flags = log_likelihoods < self.novelty_threshold
        else:
            # Not calibrated — no novelty detection
            novelty_flags = np.zeros(len(Z_new), dtype=bool)

        return assignments, soft_probs, log_likelihoods, novelty_flags

    # ──────────────────────────────────────────────────────
    # Streaming: Online EM Update
    # ──────────────────────────────────────────────────────

    def update_online_em(self, Z_window):
        """Online EM update for a streaming window.

        Implements incremental EM:
          E-step: compute responsibilities gamma(k) for each sample
          M-step: update means, covariances, weights using learning_rate
          New cluster: if novel samples exceed threshold, create new cluster

        Args:
            Z_window: (W, z_dim) window of new embeddings

        Returns:
            assignments: (W,) cluster assignments
            soft_probs: (W, K) soft cluster probabilities
            log_likelihoods: (W,) per-sample log-likelihoods
            novelty_flags: (W,) boolean novelty indicators
            new_clusters: list of newly created cluster indices
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        W = len(Z_window)
        if W == 0:
            logger.warning("Empty window passed to update_online_em, skipping")
            return (np.array([]), np.array([]).reshape(0, 0),
                    np.array([]), np.array([], dtype=bool), [])
        eta = self.learning_rate
        new_clusters = []

        # Snapshot before update (for drift detection)
        self._prev_weights = self.model.weights_.copy()
        self._prev_means = self.model.means_.copy()

        # ── E-step: compute responsibilities ──
        soft_probs = self.model.predict_proba(Z_window)   # (W, K)
        log_likelihoods = self.model.score_samples(Z_window)

        if self.novelty_threshold is not None:
            novelty_flags = log_likelihoods < self.novelty_threshold
        else:
            novelty_flags = np.zeros(W, dtype=bool)

        # ── Handle novel samples: create new cluster if needed ──
        n_novel = novelty_flags.sum()
        if n_novel > max(5, W * 0.05):
            # Enough novel samples to warrant a new cluster
            new_k = self._create_new_cluster(Z_window[novelty_flags])
            if new_k is not None:
                new_clusters.append(new_k)
                logger.info(f"Created new cluster {new_k} from "
                            f"{n_novel} novel samples")
                # Recompute responsibilities with updated model
                soft_probs = self.model.predict_proba(Z_window)
                log_likelihoods = self.model.score_samples(Z_window)
                if self.novelty_threshold is not None:
                    novelty_flags = log_likelihoods < self.novelty_threshold
                else:
                    novelty_flags = np.zeros(W, dtype=bool)

        # ── M-step: incremental parameter update ──
        K = self.model.n_components
        z_dim = Z_window.shape[1]

        for k in range(K):
            gamma_k = soft_probs[:, k]           # (W,) responsibilities
            N_k = gamma_k.sum()

            if N_k < 1e-6:
                continue

            # Weighted mean of the window for cluster k
            window_mean_k = (gamma_k[:, None] * Z_window).sum(axis=0) / N_k

            # Weighted covariance of the window for cluster k
            diff = Z_window - window_mean_k
            if self.covariance_type == 'diag':
                window_cov_k = (
                    gamma_k[:, None] * (diff ** 2)
                ).sum(axis=0) / N_k
            else:
                window_cov_k = (
                    gamma_k[:, None, None] * (diff[:, :, None] * diff[:, None, :])
                ).sum(axis=0) / N_k

            # Incremental update: theta_new = (1 - eta) * theta_old + eta * theta_window
            self.model.means_[k] = (
                (1 - eta) * self.model.means_[k] + eta * window_mean_k
            )

            if self.covariance_type == 'diag':
                self.model.covariances_[k] = (
                    (1 - eta) * self.model.covariances_[k] + eta * window_cov_k
                )
                # Update precisions for consistency
                self.model.precisions_cholesky_[k] = (
                    1.0 / np.sqrt(self.model.covariances_[k] + 1e-8)
                )
            else:
                self.model.covariances_[k] = (
                    (1 - eta) * self.model.covariances_[k] + eta * window_cov_k
                )
                # Update precisions_cholesky via Cholesky of precision matrix
                try:
                    precision = np.linalg.inv(self.model.covariances_[k])
                    self.model.precisions_cholesky_[k] = np.linalg.cholesky(precision)
                except np.linalg.LinAlgError:
                    pass  # Keep old precisions if inversion fails

        # Update weights with smoothing
        window_weights = soft_probs.sum(axis=0) / W
        self.model.weights_ = (
            (1 - eta) * self.model.weights_ + eta * window_weights
        )
        # Renormalize
        weight_sum = self.model.weights_.sum()
        if weight_sum > 0:
            self.model.weights_ /= weight_sum
        else:
            self.model.weights_ = np.ones(K) / K

        # ── Final predictions with updated model ──
        assignments = self.model.predict(Z_window)

        # Update tracking
        self.n_samples_seen += W
        for k in range(K):
            mask = assignments == k
            if mask.sum() > 0:
                self.cluster_samples[k] = {
                    'samples': Z_window[mask],
                    'size': int(mask.sum()),
                    'weight': float(self.model.weights_[k]),
                }

        logger.info(f"Online EM update: {W} samples processed "
                    f"(total: {self.n_samples_seen}), "
                    f"{len(new_clusters)} new clusters")

        return assignments, soft_probs, log_likelihoods, novelty_flags, new_clusters

    def _create_new_cluster(self, Z_novel):
        """Create a new cluster from novel samples.

        Finds the least-active cluster slot and reinitializes it
        with the novel samples' statistics.

        Args:
            Z_novel: (N, z_dim) novel sample embeddings

        Returns:
            new_k: index of the new cluster, or None if no slot available
        """
        if len(Z_novel) < 2:
            return None

        # Find least active cluster (lowest weight)
        min_k = np.argmin(self.model.weights_)
        if self.model.weights_[min_k] > 0.05:
            logger.warning("No low-weight cluster slot to repurpose")
            return None

        # Initialize with novel samples
        new_mean = Z_novel.mean(axis=0)
        new_weight = len(Z_novel) / max(self.n_samples_seen, 1)

        self.model.means_[min_k] = new_mean
        self.model.weights_[min_k] = new_weight

        if self.covariance_type == 'diag':
            new_cov = Z_novel.var(axis=0) + 1e-6
            self.model.covariances_[min_k] = new_cov
            self.model.precisions_cholesky_[min_k] = 1.0 / np.sqrt(new_cov)
        else:
            new_cov = np.cov(Z_novel.T) + np.eye(Z_novel.shape[1]) * 1e-6
            self.model.covariances_[min_k] = new_cov
            try:
                precision = np.linalg.inv(new_cov)
                self.model.precisions_cholesky_[min_k] = np.linalg.cholesky(precision)
            except np.linalg.LinAlgError:
                self.model.precisions_cholesky_[min_k] = np.eye(Z_novel.shape[1])

        # Renormalize weights
        weight_sum = self.model.weights_.sum()
        if weight_sum > 0:
            self.model.weights_ /= weight_sum

        logger.info(f"New cluster {min_k}: mean_norm={np.linalg.norm(new_mean):.3f}, "
                    f"weight={self.model.weights_[min_k]:.4f}")

        return min_k

    # ──────────────────────────────────────────────────────
    # Drift detection
    # ──────────────────────────────────────────────────────

    def detect_drift(self, threshold=0.15):
        """Detect if the model parameters have drifted significantly.

        Compares current weights and means to previous snapshot.

        Args:
            threshold: drift magnitude threshold

        Returns:
            drift_detected: bool
            drift_magnitude: float (L2 distance of weight change)
            details: dict with per-cluster drift info
        """
        if self._prev_weights is None:
            return False, 0.0, {}

        # Weight drift: L2 distance between weight vectors
        weight_drift = np.linalg.norm(
            self.model.weights_ - self._prev_weights
        )

        # Mean drift: average L2 shift across active clusters
        active_mask = self.model.weights_ > self.weight_threshold
        if active_mask.sum() > 0:
            mean_shifts = np.linalg.norm(
                self.model.means_[active_mask] - self._prev_means[active_mask],
                axis=1
            )
            avg_mean_drift = float(mean_shifts.mean())
        else:
            avg_mean_drift = 0.0

        drift_detected = weight_drift > threshold or avg_mean_drift > threshold

        details = {
            'weight_drift': float(weight_drift),
            'avg_mean_drift': avg_mean_drift,
            'threshold': threshold,
            'active_clusters': int(active_mask.sum()),
        }

        if drift_detected:
            logger.info(f"DRIFT DETECTED: weight_drift={weight_drift:.4f}, "
                        f"mean_drift={avg_mean_drift:.4f}")

        return drift_detected, float(weight_drift), details

    # ──────────────────────────────────────────────────────
    # Cluster statistics
    # ──────────────────────────────────────────────────────

    def get_cluster_statistics(self, cluster_idx):
        """Get statistics for a specific cluster."""
        if cluster_idx not in self.cluster_samples:
            return None

        cluster_data = self.cluster_samples[cluster_idx]
        samples = cluster_data['samples']

        return {
            'cluster_idx': cluster_idx,
            'size': cluster_data['size'],
            'weight': cluster_data['weight'],
            'mean': samples.mean(axis=0),
            'std': samples.std(axis=0),
            'tactic_label': self.cluster_to_tactic.get(cluster_idx, 'Unknown'),
        }

    def get_all_cluster_statistics(self):
        """Get statistics for all active clusters."""
        return [
            self.get_cluster_statistics(k)
            for k in self.cluster_samples
            if self.get_cluster_statistics(k) is not None
        ]

    def get_active_cluster_indices(self):
        """Return indices of clusters with weight above threshold."""
        if self.model is None:
            return []
        return np.where(self.model.weights_ > self.weight_threshold)[0].tolist()