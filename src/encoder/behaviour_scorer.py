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