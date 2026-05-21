"""
Interpretable Behavioural Features
=====================================

Computes ~20 objective behavioural measurements per flow from raw
features + graph properties. NO heuristic scores, NO embedded domain
assumptions about what "scanning" or "flooding" looks like.

Each dimension is a measurable network property:
  - Volumetric:    bytes, packets
  - Temporal:      duration, rate
  - Directional:   throughput asymmetry
  - Packet:        size profile
  - Connectivity:  fan-out, fan-in (from IP graph)
  - Protocol:      TCP/UDP/ICMP fractions
  - Port:          well-known / ephemeral presence
  - TCP:           flag behaviour
  - TTL:           hop distance

The DP-GMM clusters in this space to discover behavioural patterns.
The LLM then interprets what each cluster's pattern means.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Feature names for the behavioural feature vector (in order)
BEHAVIOURAL_FEATURE_NAMES = [
    # Volumetric (3)
    'bhv_bytes_out',
    'bhv_pkts_in',
    'bhv_pkts_out',
    # Temporal (2)
    'bhv_duration',
    'bhv_byte_rate',
    # Directional (2)
    'bhv_send_rate',
    'bhv_recv_rate',
    # Packet profile (3)
    'bhv_largest_pkt',
    'bhv_smallest_pkt',
    'bhv_small_pkt_ratio',
    # TCP behaviour (3)
    'bhv_tcp_flags',
    'bhv_client_flags',
    'bhv_server_flags',
    # TTL (1)
    'bhv_min_ttl',
    # Connectivity (3) — from graph
    'bhv_fan_out',
    'bhv_fan_in',
    'bhv_unique_pairs',
    # Protocol (3)
    'bhv_proto_tcp',
    'bhv_proto_udp',
    'bhv_icmp_active',
    # Port (2)
    'bhv_dst_well_known',
    'bhv_src_ephemeral',
]

N_BEHAVIOURAL_FEATURES = len(BEHAVIOURAL_FEATURE_NAMES)


class BehaviouralFeatureExtractor:
    """Extract interpretable behavioural features from raw scaled data.

    All features are objective measurements — no heuristic scores,
    no embedded assumptions about attack types.

    Usage:
        extractor = BehaviouralFeatureExtractor(feature_names)
        fan_out, fan_in, n_pairs = extractor.compute_graph_features(src_ips, dst_ips)
        B = extractor.extract(X_scaled, fan_out, fan_in, n_pairs)
        # B is (N, 22) — ready for DP-GMM clustering
    """

    def __init__(self, feature_names):
        """
        Args:
            feature_names: list of column names from preprocessor
        """
        self.feature_names = feature_names
        self._idx = {name: i for i, name in enumerate(feature_names)}
        self.scaler = StandardScaler()
        self.fitted = False

    def _get(self, X, name):
        """Get a feature column by name, zeros if missing."""
        if name in self._idx:
            return X[:, self._idx[name]]
        return np.zeros(len(X), dtype=np.float32)

    def extract(self, X_scaled, fan_out, fan_in, n_pairs):
        """Extract behavioural features.

        Args:
            X_scaled: (N, feature_dim) preprocessed scaled features
            fan_out: (N,) per-flow source fan-out
            fan_in: (N,) per-flow destination fan-in
            n_pairs: (N,) per-flow unique src-dst pairs for that source

        Returns:
            B: (N, N_BEHAVIOURAL_FEATURES) behavioural feature matrix (scaled)
        """
        N = len(X_scaled)

        features = []

        # ── Volumetric (3) ──
        features.append(self._get(X_scaled, 'OUT_BYTES'))
        features.append(self._get(X_scaled, 'IN_PKTS'))
        features.append(self._get(X_scaled, 'OUT_PKTS'))

        # ── Temporal (2) ──
        features.append(self._get(X_scaled, 'FLOW_DURATION_MILLISECONDS'))
        # Byte rate = throughput
        features.append(self._get(X_scaled, 'SRC_TO_DST_AVG_THROUGHPUT'))

        # ── Directional (2) ──
        features.append(self._get(X_scaled, 'SRC_TO_DST_SECOND_BYTES'))
        features.append(self._get(X_scaled, 'DST_TO_SRC_SECOND_BYTES'))

        # ── Packet profile (3) ──
        features.append(self._get(X_scaled, 'LONGEST_FLOW_PKT'))
        features.append(self._get(X_scaled, 'SHORTEST_FLOW_PKT'))
        # Small packet ratio
        small_pkts = self._get(X_scaled, 'NUM_PKTS_UP_TO_128_BYTES')
        features.append(small_pkts)

        # ── TCP behaviour (3) ──
        features.append(self._get(X_scaled, 'TCP_FLAGS'))
        features.append(self._get(X_scaled, 'CLIENT_TCP_FLAGS'))
        features.append(self._get(X_scaled, 'SERVER_TCP_FLAGS'))

        # ── TTL (1) ──
        features.append(self._get(X_scaled, 'MIN_TTL'))

        # ── Connectivity from graph (3) ──
        features.append(np.log1p(fan_out).astype(np.float32))
        features.append(np.log1p(fan_in).astype(np.float32))
        features.append(np.log1p(n_pairs).astype(np.float32))

        # ── Protocol (3) ──
        features.append(self._get(X_scaled, 'PROTO_TCP'))
        features.append(self._get(X_scaled, 'PROTO_UDP'))
        features.append(self._get(X_scaled, 'ICMP_ACTIVE'))

        # ── Port (2) ──
        features.append(self._get(X_scaled, 'DST_PORT_WELL_KNOWN'))
        features.append(self._get(X_scaled, 'SRC_PORT_EPHEMERAL'))

        # Stack into (N, 22) matrix
        B = np.column_stack(features).astype(np.float32)

        assert B.shape[1] == N_BEHAVIOURAL_FEATURES, (
            f"Expected {N_BEHAVIOURAL_FEATURES} features, got {B.shape[1]}"
        )

        logger.info(f"Behavioural features: {B.shape} "
                     f"({N_BEHAVIOURAL_FEATURES} dimensions)")

        return B

    def fit_scale(self, B):
        """Fit scaler on training behavioural features and transform.

        Ensures all dimensions are on comparable scale for DP-GMM.
        """
        self.scaler.fit(B)
        self.fitted = True
        return self.scaler.transform(B).astype(np.float32)

    def transform_scale(self, B):
        """Transform using fitted scaler (for inference/streaming)."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scale first.")
        return self.scaler.transform(B).astype(np.float32)

    def compute_graph_features(self, src_ips, dst_ips):
        """Compute per-flow graph features from IP arrays.

        Returns:
            fan_out: (N,) unique destinations per source IP
            fan_in: (N,) unique sources per destination IP
            n_pairs: (N,) unique src-dst pairs for that source
        """
        # Fan-out: how many unique dsts each src talks to
        src_to_dsts = {}
        for s, d in zip(src_ips, dst_ips):
            if s not in src_to_dsts:
                src_to_dsts[s] = set()
            src_to_dsts[s].add(d)

        # Fan-in: how many unique srcs each dst receives from
        dst_to_srcs = {}
        for s, d in zip(src_ips, dst_ips):
            if d not in dst_to_srcs:
                dst_to_srcs[d] = set()
            dst_to_srcs[d].add(s)

        fan_out = np.array([len(src_to_dsts.get(s, set()))
                            for s in src_ips], dtype=np.float32)
        fan_in = np.array([len(dst_to_srcs.get(d, set()))
                           for d in dst_ips], dtype=np.float32)
        n_pairs = fan_out.copy()  # unique pairs from source perspective

        return fan_out, fan_in, n_pairs

    def get_feature_names(self):
        """Return ordered list of behavioural feature names."""
        return BEHAVIOURAL_FEATURE_NAMES.copy()