"""
Behaviour Score Computation
=============================

Computes 5 tactic-relevant behaviour scores per flow from raw features
and graph properties. These are UNSUPERVISED — no attack labels used.

Scores (each in [0, 1]):
  1. scan_score:    short duration + small packets + high fan-out + well-known dst ports
  2. flood_score:   high packet rate + high throughput + short duration + many sources
  3. exfil_score:   outbound-heavy traffic + large bytes + long duration
  4. beacon_score:  balanced bidirectional + steady rate + long duration + few peers
  5. lateral_score: distributed connectivity + ephemeral ports + moderate spread

These become the target for the auxiliary prediction head in the encoder,
forcing the latent space to organize around attack behaviours.
"""

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_BEHAVIOUR_DIMS = 5
BEHAVIOUR_NAMES = ['scan', 'flood', 'exfil', 'beacon', 'lateral']


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


class BehaviourScorer:
    """Compute behaviour indicator scores per flow.

    Uses scaled features (post-StandardScaler) so positive z-scores
    mean "above average" and negative means "below average".
    Feature indices are resolved by name from the feature_names list.
    """

    def __init__(self, feature_names):
        """
        Args:
            feature_names: list of feature column names from preprocessor
        """
        self.feature_names = feature_names
        self._idx = {name: i for i, name in enumerate(feature_names)}
        self.n_scores = N_BEHAVIOUR_DIMS

    def _get(self, X, name, default=0.0):
        """Get a feature column by name, or return default if missing."""
        if name in self._idx:
            return X[:, self._idx[name]]
        return np.full(len(X), default)

    def _get_onehot_score(self, X, names):
        """Get the max z-score across a list of one-hot feature names.
        Higher value = more prevalent in this flow relative to dataset.
        """
        vals = []
        for name in names:
            if name in self._idx:
                vals.append(X[:, self._idx[name]])
        if vals:
            return np.max(vals, axis=0)
        return np.zeros(len(X))

    def compute(self, X_scaled, fan_out=None, fan_in=None, n_peers=None):
        """Compute behaviour scores for all flows.

        Args:
            X_scaled: (N, feature_dim) scaled feature matrix
            fan_out: (N,) per-flow fan-out of the source IP (optional)
            fan_in: (N,) per-flow fan-in of the destination IP (optional)
            n_peers: (N,) per-flow number of unique communication partners (optional)

        Returns:
            scores: (N, 5) behaviour score matrix in [0, 1]
        """
        N = len(X_scaled)

        # Extract key features (all in z-score space: 0=mean, 1=1std above)
        duration = self._get(X_scaled, 'FLOW_DURATION_MILLISECONDS')
        out_bytes = self._get(X_scaled, 'OUT_BYTES')
        in_pkts = self._get(X_scaled, 'IN_PKTS')
        out_pkts = self._get(X_scaled, 'OUT_PKTS')
        src_bps = self._get(X_scaled, 'SRC_TO_DST_SECOND_BYTES')
        dst_bps = self._get(X_scaled, 'DST_TO_SRC_SECOND_BYTES')
        src_thru = self._get(X_scaled, 'SRC_TO_DST_AVG_THROUGHPUT')
        dst_thru = self._get(X_scaled, 'DST_TO_SRC_AVG_THROUGHPUT')
        longest_pkt = self._get(X_scaled, 'LONGEST_FLOW_PKT')
        shortest_pkt = self._get(X_scaled, 'SHORTEST_FLOW_PKT')
        small_pkts = self._get(X_scaled, 'NUM_PKTS_UP_TO_128_BYTES')
        tcp_flags = self._get(X_scaled, 'TCP_FLAGS')

        # One-hot derived
        dst_well_known = self._get_onehot_score(X_scaled, [
            'DST_PORT_WELL_KNOWN'
        ])
        src_ephemeral = self._get_onehot_score(X_scaled, [
            'SRC_PORT_EPHEMERAL'
        ])
        proto_tcp = self._get_onehot_score(X_scaled, ['PROTO_TCP'])
        proto_udp = self._get_onehot_score(X_scaled, ['PROTO_UDP'])
        proto_icmp = self._get_onehot_score(X_scaled, ['PROTO_ICMP'])
        icmp_active = self._get(X_scaled, 'ICMP_ACTIVE')

        # Graph features (if provided, otherwise neutral)
        if fan_out is None:
            fan_out = np.zeros(N)
        if fan_in is None:
            fan_in = np.zeros(N)
        if n_peers is None:
            n_peers = np.zeros(N)

        # Normalize graph features to z-score-like scale
        fan_out_z = (fan_out - fan_out.mean()) / (fan_out.std() + 1e-8)
        fan_in_z = (fan_in - fan_in.mean()) / (fan_in.std() + 1e-8)
        n_peers_z = (n_peers - n_peers.mean()) / (n_peers.std() + 1e-8)

        # ── 1. Scan score ──
        # Scanning: short flows, small packets, many destinations, well-known ports
        scan_raw = (
            -duration * 0.8 +          # short duration (negative z = short)
            -out_bytes * 0.5 +          # small bytes
            small_pkts * 0.5 +          # many small packets
            fan_out_z * 1.0 +           # high fan-out
            dst_well_known * 0.6 +      # targeting well-known ports
            icmp_active * 0.4           # ICMP probing
        )
        scan_score = sigmoid(scan_raw)

        # ── 2. Flood score ──
        # DoS/DDoS: high packet rate, high throughput, short duration
        flood_raw = (
            in_pkts * 0.7 +             # high inbound packets
            out_pkts * 0.7 +            # high outbound packets
            src_thru * 0.8 +            # high throughput
            dst_thru * 0.8 +            # high return throughput
            -duration * 0.5 +           # short flows
            tcp_flags * 0.4 +           # unusual TCP flags (SYN floods)
            fan_in_z * 0.6              # many-to-one pattern
        )
        flood_score = sigmoid(flood_raw)

        # ── 3. Exfiltration score ──
        # Data theft: outbound heavy, large bytes, moderate-long duration
        direction_ratio = src_bps - dst_bps  # positive = outbound heavy
        exfil_raw = (
            direction_ratio * 1.0 +     # outbound-heavy traffic
            out_bytes * 0.8 +           # large outbound bytes
            longest_pkt * 0.6 +         # large payload packets
            duration * 0.4 +            # longer duration
            -fan_out_z * 0.3            # focused targets (low fan-out)
        )
        exfil_score = sigmoid(exfil_raw)

        # ── 4. Beacon score ──
        # C2: balanced bidirectional, steady, long-lived, few peers
        balance = -np.abs(src_bps - dst_bps)  # higher when balanced
        beacon_raw = (
            balance * 0.8 +             # balanced traffic
            duration * 0.8 +            # long duration
            -fan_out_z * 0.6 +          # few destinations (1-to-1)
            -fan_in_z * 0.4 +           # few sources
            proto_tcp * 0.3 +           # TCP dominant
            -tcp_flags * 0.3            # stable flags (no SYN floods)
        )
        beacon_score = sigmoid(beacon_raw)

        # ── 5. Lateral movement score ──
        # Spreading: distributed, ephemeral ports, internal-like
        lateral_raw = (
            n_peers_z * 0.8 +           # many unique peers
            src_ephemeral * 0.5 +       # ephemeral source ports
            proto_tcp * 0.4 +           # TCP dominant
            fan_out_z * 0.4 +           # moderate fan-out
            fan_in_z * 0.4 +            # moderate fan-in
            -duration * 0.2             # short-medium flows
        )
        lateral_score = sigmoid(lateral_raw)

        # Stack into (N, 5) matrix
        scores = np.column_stack([
            scan_score,
            flood_score,
            exfil_score,
            beacon_score,
            lateral_score,
        ]).astype(np.float32)

        # Log distribution
        logger.info(f"Behaviour scores computed for {N} flows:")
        for i, name in enumerate(BEHAVIOUR_NAMES):
            s = scores[:, i]
            logger.info(f"  {name:>10s}: mean={s.mean():.3f}, "
                         f"std={s.std():.3f}, "
                         f"range=[{s.min():.3f}, {s.max():.3f}]")

        return scores

    def compute_graph_features(self, src_ips, dst_ips):
        """Compute per-flow graph features from IP arrays.

        Args:
            src_ips: (N,) source IP strings
            dst_ips: (N,) destination IP strings

        Returns:
            fan_out: (N,) per-flow fan-out of source IP
            fan_in: (N,) per-flow fan-in of destination IP
            n_peers: (N,) per-flow unique communication partners
        """
        from collections import Counter

        # Fan-out: how many unique dsts does each src talk to
        src_to_dsts = {}
        for s, d in zip(src_ips, dst_ips):
            if s not in src_to_dsts:
                src_to_dsts[s] = set()
            src_to_dsts[s].add(d)

        # Fan-in: how many unique srcs does each dst receive from
        dst_to_srcs = {}
        for s, d in zip(src_ips, dst_ips):
            if d not in dst_to_srcs:
                dst_to_srcs[d] = set()
            dst_to_srcs[d].add(s)

        # Per-flow assignment
        fan_out = np.array([len(src_to_dsts.get(s, set())) for s in src_ips],
                           dtype=np.float32)
        fan_in = np.array([len(dst_to_srcs.get(d, set())) for d in dst_ips],
                          dtype=np.float32)
        # n_peers = fan_out + fan_in (total unique partners)
        n_peers = fan_out + fan_in

        return fan_out, fan_in, n_peers
# """
# Percentile-Based Behaviour Score Computation
# ================================================
#
# Computes 5 tactic-relevant behaviour scores per flow using
# PERCENTILE RANKS instead of absolute thresholds.
#
# Self-calibrating: "unusual" means top/bottom percentile of THIS
# dataset, not a hardcoded value. Works across datasets without
# dataset-specific tuning.
#
# Scores (each in [0, 1]):
#   1. scan_score:    many destinations + tiny payloads (relative)
#   2. flood_score:   high asymmetry + unusual flags + low fan_in
#   3. exfil_score:   extreme outbound volume (relative)
#   4. beacon_score:  balanced direction + long duration + few peers
#   5. lateral_score: high fan-out + moderate payload + spreading
# """
#
# import numpy as np
# import torch
# import logging
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# N_BEHAVIOUR_DIMS = 5
# BEHAVIOUR_NAMES = ['scan', 'flood', 'exfil', 'beacon', 'lateral']
#
#
# def sigmoid(x):
#     """Numerically stable sigmoid."""
#     return np.where(x >= 0,
#                     1 / (1 + np.exp(-np.clip(x, -20, 20))),
#                     np.exp(np.clip(x, -20, 20)) / (1 + np.exp(np.clip(x, -20, 20))))
#
#
# def percentile_rank(values):
#     """Compute percentile rank (0-1) for each value in the array.
#     Self-calibrating: adapts to any dataset distribution.
#     """
#     n = len(values)
#     if n == 0:
#         return values
#     sorted_idx = np.argsort(values)
#     ranks = np.empty_like(sorted_idx, dtype=np.float64)
#     ranks[sorted_idx] = np.linspace(0, 1, n)
#     return ranks.astype(np.float32)
#
#
# def pct_high(pct, center=0.85, sharpness=15.0):
#     """Score high when percentile is above center.
#     pct=0.95 → ~0.82,  pct=0.50 → ~0.00
#     """
#     return sigmoid((pct - center) * sharpness)
#
#
# def pct_low(pct, center=0.15, sharpness=15.0):
#     """Score high when percentile is below center.
#     pct=0.05 → ~0.82,  pct=0.50 → ~0.00
#     """
#     return sigmoid((center - pct) * sharpness)
#
#
# def pct_mid(pct, center=0.5, width=0.15):
#     """Score high when percentile is near center.
#     pct=0.50 → ~1.0,  pct=0.10 → ~0.00
#     """
#     return np.exp(-((pct - center) / width) ** 2)
#
#
# class BehaviourScorer:
#     """Compute behaviour scores using percentile-based self-calibration.
#
#     Instead of absolute thresholds (TCP_FLAGS < 4 → flood):
#       → "TCP_FLAGS in bottom 5% of THIS dataset" → flood
#
#     Works across BoT-IoT, UNSW-NB15, or any other dataset
#     without threshold recalibration.
#     """
#
#     def __init__(self, feature_names):
#         self.feature_names = feature_names
#         self._idx = {name: i for i, name in enumerate(feature_names)}
#         self.n_scores = N_BEHAVIOUR_DIMS
#
#     def _get(self, X, name, default=0.0):
#         if name in self._idx:
#             return X[:, self._idx[name]]
#         return np.full(len(X), default)
#
#     def _get_onehot_score(self, X, names):
#         vals = []
#         for name in names:
#             if name in self._idx:
#                 vals.append(X[:, self._idx[name]])
#         if vals:
#             return np.max(vals, axis=0)
#         return np.zeros(len(X))
#
#     def compute(self, X_scaled, fan_out=None, fan_in=None, n_peers=None):
#         """Compute percentile-based behaviour scores.
#
#         Args:
#             X_scaled: (N, feature_dim) scaled feature matrix
#             fan_out, fan_in, n_peers: (N,) graph features
#
#         Returns:
#             scores: (N, 5) behaviour score matrix in [0, 1]
#         """
#         N = len(X_scaled)
#
#         # Extract raw z-score features
#         duration = self._get(X_scaled, 'FLOW_DURATION_MILLISECONDS')
#         out_bytes = self._get(X_scaled, 'OUT_BYTES')
#         in_bytes = self._get(X_scaled, 'IN_BYTES') if 'IN_BYTES' in self._idx else (
#             self._get(X_scaled, 'DST_TO_SRC_SECOND_BYTES'))
#         in_pkts = self._get(X_scaled, 'IN_PKTS')
#         out_pkts = self._get(X_scaled, 'OUT_PKTS')
#         src_bps = self._get(X_scaled, 'SRC_TO_DST_SECOND_BYTES')
#         dst_bps = self._get(X_scaled, 'DST_TO_SRC_SECOND_BYTES')
#         src_thru = self._get(X_scaled, 'SRC_TO_DST_AVG_THROUGHPUT')
#         longest_pkt = self._get(X_scaled, 'LONGEST_FLOW_PKT')
#         shortest_pkt = self._get(X_scaled, 'SHORTEST_FLOW_PKT')
#         small_pkts = self._get(X_scaled, 'NUM_PKTS_UP_TO_128_BYTES')
#         tcp_flags = self._get(X_scaled, 'TCP_FLAGS')
#         min_ttl = self._get(X_scaled, 'MIN_TTL')
#
#         dst_well_known = self._get_onehot_score(X_scaled, ['DST_PORT_WELL_KNOWN'])
#         src_ephemeral = self._get_onehot_score(X_scaled, ['SRC_PORT_EPHEMERAL'])
#         icmp_active = self._get(X_scaled, 'ICMP_ACTIVE')
#
#         if fan_out is None: fan_out = np.zeros(N)
#         if fan_in is None: fan_in = np.zeros(N)
#         if n_peers is None: n_peers = np.zeros(N)
#
#         # ═══════════════════════════════════════════
#         # Convert ALL features to percentile ranks
#         # This is the self-calibrating step
#         # ═══════════════════════════════════════════
#         p_duration = percentile_rank(duration)
#         p_out_bytes = percentile_rank(out_bytes)
#         p_in_pkts = percentile_rank(in_pkts)
#         p_out_pkts = percentile_rank(out_pkts)
#         p_src_bps = percentile_rank(src_bps)
#         p_dst_bps = percentile_rank(dst_bps)
#         p_src_thru = percentile_rank(src_thru)
#         p_longest_pkt = percentile_rank(longest_pkt)
#         p_shortest_pkt = percentile_rank(shortest_pkt)
#         p_small_pkts = percentile_rank(small_pkts)
#         p_tcp_flags = percentile_rank(tcp_flags)
#         p_min_ttl = percentile_rank(min_ttl)
#         p_fan_out = percentile_rank(fan_out)
#         p_fan_in = percentile_rank(fan_in)
#         p_n_peers = percentile_rank(n_peers)
#
#         # Direction ratio percentile
#         dir_ratio = src_bps - dst_bps
#         p_dir_ratio = percentile_rank(dir_ratio)
#
#         # Total packets
#         total_pkts = in_pkts + out_pkts
#         p_total_pkts = percentile_rank(total_pkts)
#
#         # ═══════════════════════════════════════════
#         # 1. SCAN score
#         # Universal: many destinations + tiny payloads
#         # High fan_out + low out_bytes + low B/pkt + well-known ports
#         # ═══════════════════════════════════════════
#         scan_score = (
#             pct_high(p_fan_out, 0.80) * 0.30 +
#             pct_low(p_out_bytes, 0.20) * 0.25 +
#             pct_low(p_longest_pkt, 0.20) * 0.15 +
#             pct_high(p_small_pkts, 0.80) * 0.10 +
#             dst_well_known * 0.10 +
#             icmp_active * 0.10
#         )
#
#         # ═══════════════════════════════════════════
#         # 2. FLOOD score
#         # Universal: high rate + asymmetric + unusual flags
#         # Works for SYN floods (BoT-IoT) AND volumetric (UNSW)
#         # Key: low fan_in = attack-like in UNSW (Fisher=4.23)
#         # ═══════════════════════════════════════════
#         flood_score = (
#             pct_high(p_total_pkts, 0.85) * 0.20 +
#             pct_high(p_src_thru, 0.85) * 0.15 +
#             pct_high(p_dir_ratio, 0.85) * 0.15 +
#             pct_low(p_tcp_flags, 0.15) * 0.15 +
#             pct_low(p_fan_in, 0.15) * 0.15 +
#             pct_low(p_duration, 0.20) * 0.10 +
#             pct_high(p_min_ttl, 0.85) * 0.10
#         )
#
#         # ═══════════════════════════════════════════
#         # 3. EXFIL score
#         # Universal: extreme outbound volume
#         # High out_bytes + high direction_ratio + long duration
#         # ═══════════════════════════════════════════
#         exfil_score = (
#             pct_high(p_out_bytes, 0.90) * 0.35 +
#             pct_high(p_dir_ratio, 0.85) * 0.25 +
#             pct_high(p_longest_pkt, 0.85) * 0.15 +
#             pct_high(p_duration, 0.70) * 0.10 +
#             pct_low(p_fan_out, 0.30) * 0.15
#         )
#
#         # ═══════════════════════════════════════════
#         # 4. BEACON score
#         # Universal: balanced direction + long duration + few peers
#         # ═══════════════════════════════════════════
#         beacon_score = (
#             pct_mid(p_dir_ratio, 0.50, 0.15) * 0.30 +
#             pct_high(p_duration, 0.80) * 0.25 +
#             pct_low(p_fan_out, 0.20) * 0.20 +
#             pct_low(p_fan_in, 0.30) * 0.10 +
#             pct_low(p_total_pkts, 0.30) * 0.15
#         )
#
#         # ═══════════════════════════════════════════
#         # 5. LATERAL score
#         # Universal: spreading to many hosts with payload
#         # High fan_out + moderate out_bytes + many peers
#         # ═══════════════════════════════════════════
#         lateral_score = (
#             pct_high(p_n_peers, 0.85) * 0.25 +
#             pct_high(p_fan_out, 0.80) * 0.25 +
#             pct_mid(p_out_bytes, 0.60, 0.20) * 0.15 +
#             src_ephemeral * 0.10 +
#             pct_high(p_min_ttl, 0.80) * 0.10 +
#             pct_low(p_duration, 0.30) * 0.15
#         )
#
#         # Stack
#         scores = np.column_stack([
#             scan_score, flood_score, exfil_score,
#             beacon_score, lateral_score,
#         ]).astype(np.float32)
#
#         # Log
#         logger.info(f"Behaviour scores (percentile-based) for {N} flows:")
#         for i, name in enumerate(BEHAVIOUR_NAMES):
#             s = scores[:, i]
#             logger.info(f"  {name:>10s}: mean={s.mean():.3f}, "
#                          f"std={s.std():.3f}, "
#                          f"range=[{s.min():.3f}, {s.max():.3f}]")
#
#         return scores
#
#     def compute_graph_features(self, src_ips, dst_ips):
#         """Compute per-flow graph features from IP arrays."""
#         src_to_dsts = {}
#         for s, d in zip(src_ips, dst_ips):
#             if s not in src_to_dsts:
#                 src_to_dsts[s] = set()
#             src_to_dsts[s].add(d)
#
#         dst_to_srcs = {}
#         for s, d in zip(src_ips, dst_ips):
#             if d not in dst_to_srcs:
#                 dst_to_srcs[d] = set()
#             dst_to_srcs[d].add(s)
#
#         fan_out = np.array([len(src_to_dsts.get(s, set())) for s in src_ips],
#                            dtype=np.float32)
#         fan_in = np.array([len(dst_to_srcs.get(d, set())) for d in dst_ips],
#                           dtype=np.float32)
#         n_peers = fan_out + fan_in
#
#         return fan_out, fan_in, n_peers