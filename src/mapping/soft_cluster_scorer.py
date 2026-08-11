# """
# Soft Cluster-Level Behaviour Scoring
# ========================================
#
# Computes a behaviour DISTRIBUTION per cluster (not a hard label).
# Uses the cluster's centroid features from the summarizer stats.
#
# Each flow inherits its cluster's soft behaviour distribution.
# Windows aggregate these distributions across flows.
# """
#
# import numpy as np
# from collections import Counter, defaultdict
# import logging
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Behaviour → Tactic weights
# BEHAVIOUR_TACTIC_WEIGHTS = {
#     'syn_flooding':        {'Impact': 1.0},
#     'volumetric_flood':    {'Impact': 1.0},
#     'port_scanning':       {'Reconnaissance': 1.0},
#     'service_probing':     {'Reconnaissance': 0.6, 'Discovery': 0.4},
#     'data_transfer':       {'Exfiltration': 1.0},
#     'normal_activity':     {'None': 1.0},
#     'connection_attempts': {'Reconnaissance': 0.5, 'Impact': 0.5},
#     'periodic_comm':       {'Command and Control': 1.0},
# }
#
#
# class SoftClusterScorer:
#     """Scores each cluster with a behaviour probability distribution.
#
#     Instead of: cluster → one behaviour → one tactic
#     Now:        cluster → {behaviour: score} → {tactic: score}
#     """
#
#     def score_cluster(self, cluster_summary):
#         """Compute soft behaviour scores for a cluster.
#
#         Args:
#             cluster_summary: dict with 'stats' containing 'behaviour'
#                 and 'graph_properties' from ClusterSummarizer
#
#         Returns:
#             bhv_scores: dict {behaviour: score}
#             tactic_scores: dict {tactic: score}
#             primary_behaviour: str
#             primary_tactic: str
#         """
#         stats = cluster_summary.get('stats', {})
#         bhv = stats.get('behaviour', {})
#         graph = stats.get('graph_properties', {})
#         proportion = stats.get('proportion', 0.0)
#
#         tcp_flags = bhv.get('avg_tcp_flags', 18.0)
#         out_bytes = bhv.get('avg_out_bytes', 0)
#         min_ttl = bhv.get('avg_min_ttl', 30)
#         fan_out = graph.get('avg_fan_out', 1)
#         fan_in = graph.get('avg_fan_in', 1)
#         n_dst = graph.get('n_unique_destinations', 1)
#         pps = bhv.get('packets_per_second', 0)
#         dir_ratio = bhv.get('throughput_ratio', 1.0)
#         duration = bhv.get('avg_duration_ms', 0)
#         direction = bhv.get('traffic_direction', '')
#
#         is_syn_dominant = tcp_flags < 4
#
#         scores = {}
#
#         # ── SYN flooding ──
#         s = 0.0
#         if tcp_flags < 4: s += 0.5
#         if tcp_flags < 2: s += 0.2
#         if out_bytes < 10: s += 0.15
#         if min_ttl > 50: s += 0.1
#         if dir_ratio > 100: s += 0.05
#         scores['syn_flooding'] = min(s, 1.0)
#
#         # ── Volumetric flooding ──
#         s = 0.0
#         if tcp_flags >= 4 and min_ttl > 50: s += 0.3
#         if dir_ratio > 100: s += 0.2
#         if pps > 5000 and out_bytes < 1000: s += 0.2
#         if fan_out < 5: s += 0.1
#         scores['volumetric_flood'] = min(s, 1.0)
#
#         # ── Port scanning ── (suppressed when SYN dominant)
#         s = 0.0
#         if not is_syn_dominant:
#             if tcp_flags > 15 and out_bytes < 200: s += 0.3
#             if 20 < min_ttl < 50: s += 0.2
#             if pps > 5000: s += 0.15
#             if n_dst > 5: s += 0.15
#             if fan_out > 5: s += 0.1
#         scores['port_scanning'] = min(s, 1.0)
#
#         # ── Service probing ──
#         s = 0.0
#         if not is_syn_dominant:
#             if tcp_flags > 15 and 100 < out_bytes < 5000: s += 0.3
#             if 20 < min_ttl < 50: s += 0.2
#             if 2 < fan_out < 10: s += 0.2
#         scores['service_probing'] = min(s, 1.0)
#
#         # ── Data transfer ──
#         s = 0.0
#         if not is_syn_dominant:
#             if out_bytes > 200000: s += 0.5
#             elif out_bytes > 50000: s += 0.3
#             if min_ttl > 30: s += 0.15
#             if dir_ratio > 3: s += 0.15
#             if fan_out > 10: s += 0.1
#         scores['data_transfer'] = min(s, 1.0)
#
#         # ── Normal activity ──
#         s = 0.0
#         if not is_syn_dominant:
#             if min_ttl < 25: s += 0.3
#             if proportion > 0.05: s += 0.2
#             if 100 < out_bytes < 100000: s += 0.15
#             if tcp_flags > 15: s += 0.1
#             if 0.3 < dir_ratio < 3: s += 0.1
#         # Override: data_transfer + low TTL = normal, not exfil
#         if scores.get('data_transfer', 0) > 0.3 and min_ttl < 25:
#             s += 0.3
#             scores['data_transfer'] *= 0.3
#         scores['normal_activity'] = min(s, 1.0)
#
#         # ── Connection attempts ──
#         s = 0.0
#         if tcp_flags < 4 and fan_out <= 3: s += 0.5
#         if out_bytes < 10: s += 0.2
#         scores['connection_attempts'] = min(s, 1.0)
#
#         # ── Periodic communication ──
#         s = 0.0
#         if not is_syn_dominant:
#             if duration > 10000: s += 0.3
#             if direction == 'BALANCED': s += 0.2
#             if fan_out <= 2: s += 0.2
#             if tcp_flags > 10: s += 0.1
#         scores['periodic_comm'] = min(s, 1.0)
#
#         # Normalize to sum=1
#         total = sum(scores.values())
#         if total > 0:
#             bhv_scores = {k: v / total for k, v in scores.items()}
#         else:
#             bhv_scores = {'normal_activity': 1.0}
#
#         # Map to tactic scores
#         tactic_scores = self._map_to_tactics(bhv_scores)
#
#         primary_bhv = max(bhv_scores, key=bhv_scores.get)
#         primary_tactic = max(tactic_scores, key=tactic_scores.get)
#
#         return {
#             'behaviour_scores': bhv_scores,
#             'tactic_scores': tactic_scores,
#             'primary_behaviour': primary_bhv,
#             'primary_tactic': primary_tactic,
#         }
#
#     def score_all_clusters(self, cluster_summaries):
#         """Score all clusters with soft behaviour distributions.
#
#         Returns:
#             cluster_scores: dict {cluster_idx: {behaviour_scores, tactic_scores, ...}}
#         """
#         cluster_scores = {}
#         for k, summary in cluster_summaries.items():
#             result = self.score_cluster(summary)
#             cluster_scores[int(k)] = result
#             bhv_str = ', '.join(f"{b}:{s:.2f}"
#                                 for b, s in sorted(result['behaviour_scores'].items(),
#                                                     key=lambda x: -x[1])[:3])
#             logger.info(f"Cluster {k}: [{bhv_str}] → {result['primary_tactic']}")
#         return cluster_scores
#
#     def _map_to_tactics(self, bhv_scores):
#         tactic_scores = defaultdict(float)
#         for bhv, score in bhv_scores.items():
#             if bhv in BEHAVIOUR_TACTIC_WEIGHTS:
#                 for tactic, weight in BEHAVIOUR_TACTIC_WEIGHTS[bhv].items():
#                     tactic_scores[tactic] += score * weight
#
#         # Normalize
#         total = sum(tactic_scores.values())
#         if total > 0:
#             tactic_scores = {t: s / total for t, s in tactic_scores.items()}
#
#         # Ensure all main tactics present
#         for t in ['Impact', 'Reconnaissance', 'Exfiltration', 'None',
#                   'Command and Control', 'Discovery']:
#             if t not in tactic_scores:
#                 tactic_scores[t] = 0.0
#
#         return dict(tactic_scores)
#
#
# class SoftWindowAggregator:
#     """Aggregates soft cluster-level behaviours into window-level tactic predictions.
#
#     Each flow inherits its cluster's soft behaviour distribution.
#     The window averages these distributions across all flows.
#     """
#
#     def __init__(self, window_size=20, tactic_threshold=0.10):
#         self.window_size = window_size
#         self.tactic_threshold = tactic_threshold
#
#     def create_windows(self, src_ips, cluster_assignments,
#                        cluster_scores, y_true=None):
#         """Create time windows with soft tactic predictions.
#
#         Args:
#             src_ips: (N,) source IP per flow
#             cluster_assignments: (N,) cluster index per flow
#             cluster_scores: dict from SoftClusterScorer.score_all_clusters()
#             y_true: (N,) optional ground truth labels
#         """
#         src_flows = defaultdict(list)
#         for i, src in enumerate(src_ips):
#             src_flows[src].append(i)
#
#         windows = []
#         for src_ip, indices in src_flows.items():
#             for w_start in range(0, len(indices), self.window_size):
#                 w_end = min(w_start + self.window_size, len(indices))
#                 w_idx = indices[w_start:w_end]
#                 n = len(w_idx)
#
#                 # Aggregate soft behaviour scores across flows in window
#                 agg_bhv = defaultdict(float)
#                 agg_tactic = defaultdict(float)
#
#                 for i in w_idx:
#                     k = int(cluster_assignments[i])
#                     cs = cluster_scores.get(k, {})
#                     for bhv, score in cs.get('behaviour_scores', {}).items():
#                         agg_bhv[bhv] += score / n
#                     for tac, score in cs.get('tactic_scores', {}).items():
#                         agg_tactic[tac] += score / n
#
#                 # Normalize
#                 bhv_total = sum(agg_bhv.values())
#                 if bhv_total > 0:
#                     agg_bhv = {k: v / bhv_total for k, v in agg_bhv.items()}
#                 tac_total = sum(agg_tactic.values())
#                 if tac_total > 0:
#                     agg_tactic = {k: v / tac_total for k, v in agg_tactic.items()}
#
#                 primary_tactic = max(agg_tactic, key=agg_tactic.get) if agg_tactic else 'None'
#                 final_tactics = sorted(
#                     [t for t, s in agg_tactic.items() if s >= self.tactic_threshold],
#                     key=lambda t: agg_tactic.get(t, 0), reverse=True)
#
#                 window = {
#                     'src_ip': src_ip,
#                     'flow_indices': w_idx,
#                     'n_flows': n,
#                     'behaviour_scores': dict(agg_bhv),
#                     'tactic_scores': dict(agg_tactic),
#                     'primary_tactic': primary_tactic,
#                     'final_tactics': final_tactics,
#                     'clusters': [int(cluster_assignments[i]) for i in w_idx],
#                 }
#
#                 if y_true is not None:
#                     w_attacks = [y_true[i] for i in w_idx]
#                     majority = Counter(w_attacks).most_common(1)[0]
#                     window['majority_attack'] = majority[0]
#                     window['attack_purity'] = majority[1] / n
#                     window['true_attacks'] = w_attacks
#
#                 windows.append(window)
#
#         return windows
"""
Soft Cluster-Level Behaviour Scoring
========================================

Computes a behaviour DISTRIBUTION per cluster (not a hard label).
Uses the cluster's centroid features from the summarizer stats.

Each flow inherits its cluster's soft behaviour distribution.
Windows aggregate these distributions across flows.
"""

import numpy as np
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Behaviour → Tactic weights
BEHAVIOUR_TACTIC_WEIGHTS_BOTIOT = {
    'syn_flooding': {'Impact': 1.0},
    'volumetric_flood': {'Impact': 1.0},
    'port_scanning': {'Reconnaissance': 1.0},
    'service_probing': {'Reconnaissance': 0.6, 'Discovery': 0.4},
    'data_transfer': {'Exfiltration': 1.0},
    'normal_activity': {'None': 1.0},
    'connection_attempts': {'Reconnaissance': 0.5, 'Impact': 0.5},
    'periodic_comm': {'Command and Control': 1.0},
    'fuzzing': {'Initial Access': 1.0},
    'exploitation': {'Execution': 0.6, 'Initial Access': 0.4},
    'backdoor_comm': {'Persistence': 1.0},
    'worm_spreading': {'Lateral Movement': 1.0},
}

BEHAVIOUR_TACTIC_WEIGHTS_UNSW = {
    'syn_flooding': {'Impact': 1.0},
    'volumetric_flood': {'Impact': 1.0},
    'port_scanning': {'Reconnaissance': 1.0},
    'service_probing': {'Reconnaissance': 0.4, 'Discovery': 0.6},
    'data_transfer': {'Exfiltration': 0.5, 'Lateral Movement': 0.5},
    'normal_activity': {'None': 1.0},
    'connection_attempts': {'Reconnaissance': 0.5, 'Impact': 0.5},
    'periodic_comm': {'Command and Control': 1.0},
    'fuzzing': {'Initial Access': 1.0},
    'exploitation': {'Execution': 0.6, 'Initial Access': 0.4},
    'backdoor_comm': {'Persistence': 1.0},
    'worm_spreading': {'Lateral Movement': 1.0},
}

BEHAVIOUR_TACTIC_WEIGHTS_CICIDS = {
    'syn_flooding': {'Impact': 1.0},
    'volumetric_flood': {'Impact': 1.0},
    'port_scanning': {'Reconnaissance': 1.0},
    'service_probing': {'Reconnaissance': 0.6, 'Discovery': 0.4},
    'data_transfer': {'Exfiltration': 0.3, 'Lateral Movement': 0.7},
    'normal_activity': {'None': 1.0},
    'connection_attempts': {'Credential Access': 0.7, 'Reconnaissance': 0.3},
    'periodic_comm': {'Command and Control': 1.0},
    'fuzzing': {'Initial Access': 1.0},
    'exploitation': {'Execution': 0.6, 'Initial Access': 0.4},
    'backdoor_comm': {'Persistence': 1.0},
    'worm_spreading': {'Lateral Movement': 1.0},
}

# Default (backward compatible)
BEHAVIOUR_TACTIC_WEIGHTS = BEHAVIOUR_TACTIC_WEIGHTS_BOTIOT


def get_behaviour_tactic_weights(dataset='bot_iot'):
    """Get dataset-specific behaviour-to-tactic weights."""
    if dataset == 'unsw_nb15':
        return BEHAVIOUR_TACTIC_WEIGHTS_UNSW
    elif dataset == 'cicids2018':
        return BEHAVIOUR_TACTIC_WEIGHTS_CICIDS
    else:
        return BEHAVIOUR_TACTIC_WEIGHTS_BOTIOT


class SoftClusterScorer:
    """Scores each cluster with a behaviour probability distribution.

    Dataset-aware: uses different scoring rules based on Fisher analysis.
    BoT-IoT: TCP_FLAGS is top discriminator (Fisher=1.82)
    UNSW-NB15: fan_in (Fisher=4.23) and MIN_TTL (Fisher=2.96) are top
    """

    def __init__(self, dataset='bot_iot'):
        self.dataset = dataset

    def score_cluster(self, cluster_summary):
        stats = cluster_summary.get('stats', {})
        bhv = stats.get('behaviour', {})
        graph = stats.get('graph_properties', {})
        proportion = stats.get('proportion', 0.0)

        tcp_flags = bhv.get('avg_tcp_flags', 18.0)
        out_bytes = bhv.get('avg_out_bytes', 0)
        min_ttl = bhv.get('avg_min_ttl', 30)
        fan_out = graph.get('avg_fan_out', 1)
        fan_in = graph.get('avg_fan_in', 1)
        n_dst = graph.get('n_unique_destinations', 1)
        pps = bhv.get('packets_per_second', 0)
        dir_ratio = bhv.get('throughput_ratio', 1.0)
        duration = bhv.get('avg_duration_ms', 0)
        direction = bhv.get('traffic_direction', '')
        in_bytes = bhv.get('avg_in_bytes', 0)
        longest_pkt = bhv.get('avg_longest_pkt_bytes', 0)
        shortest_pkt = bhv.get('avg_shortest_pkt_bytes', 0)
        protocol = bhv.get('avg_protocol', 6)
        bpp = bhv.get('bytes_per_packet', 0)
        dst_port = bhv.get('avg_dst_port', 0)

        if self.dataset == 'unsw_nb15':
            scores = self._score_unsw(
                tcp_flags, out_bytes, in_bytes, min_ttl, fan_out, fan_in,
                n_dst, pps, dir_ratio, duration, proportion, protocol,
                longest_pkt, shortest_pkt, bpp)
        elif self.dataset == 'cicids2018':
            scores = self._score_cicids(
                tcp_flags, out_bytes, in_bytes, min_ttl, fan_out, fan_in,
                n_dst, pps, dir_ratio, duration, proportion, bpp, dst_port)
        else:
            # bot_iot
            scores = self._score_botiot(
                tcp_flags, out_bytes, in_bytes, min_ttl, fan_out, fan_in,
                n_dst, pps, dir_ratio, duration, direction, proportion)

        # Temperature sharpening
        TEMPERATURE = 2.0
        sharpened = {k: v ** TEMPERATURE for k, v in scores.items()}

        total = sum(sharpened.values())
        if total > 0:
            bhv_scores = {k: v / total for k, v in sharpened.items()}
        else:
            bhv_scores = {'normal_activity': 1.0}

        tactic_scores = self._map_to_tactics(bhv_scores)
        primary_bhv = max(bhv_scores, key=bhv_scores.get)
        primary_tactic = max(tactic_scores, key=tactic_scores.get)

        return {
            'behaviour_scores': bhv_scores,
            'tactic_scores': tactic_scores,
            'primary_behaviour': primary_bhv,
            'primary_tactic': primary_tactic,
        }

    # ══════════════════════════════════════════════════════
    # UNSW-NB15 Scoring (Fisher: fan_in=4.23, MIN_TTL=2.96)
    # ══════════════════════════════════════════════════════

    def _score_unsw(self, tcp_flags, out_bytes, in_bytes, min_ttl,
                     fan_out, fan_in, n_dst, pps, dir_ratio, duration,
                     proportion, protocol, longest_pkt, shortest_pkt, bpp):
        """UNSW-NB15 scoring rules based on Fisher discriminability.

        Top discriminators:
          fan_in=4.23:  Benign=10, ALL attacks=4
          MIN_TTL=2.96: Benign=32, attacks=135-253
          dir_ratio=0.60: Benign=8, attacks=109-861
          PROTOCOL=0.23: Discovery/Persistence=105, others<30
        """
        scores = {}
        is_benign_like = fan_in >= 8 and min_ttl < 50
        is_attack_like = fan_in < 6 and min_ttl > 100
        is_non_tcp = protocol > 50

        # ── Normal activity (Benign) ──
        # fan_in=10, MIN_TTL=32, dir_ratio=8, TCP_FLAGS=20
        s = 0.0
        if fan_in >= 8: s += 0.35
        if min_ttl < 50: s += 0.25
        if dir_ratio < 15: s += 0.15
        if tcp_flags > 15: s += 0.1
        if proportion > 0.05: s += 0.1
        if is_attack_like: s *= 0.1
        scores['normal_activity'] = min(s, 1.0)

        # ── Port scanning (Reconnaissance) ──
        # OUT_BYTES=164, B/Pkt=83, MIN_TTL=253, TCP_FLAGS=9.8, DstPort=92
        s = 0.0
        if is_attack_like:
            if out_bytes < 300: s += 0.3
            if bpp < 100: s += 0.2
            if min_ttl > 200: s += 0.15
            if 5 < tcp_flags < 15: s += 0.15
            if dir_ratio > 100: s += 0.1
        scores['port_scanning'] = min(s, 1.0)

        # ── Service probing (Discovery ← Analysis) ──
        # PROTOCOL=105, dir_ratio=837, TCP_FLAGS=4.4, OUT_BYTES=214
        s = 0.0
        if is_attack_like:
            if is_non_tcp: s += 0.35
            if tcp_flags < 6: s += 0.2
            if dir_ratio > 500: s += 0.15
            if out_bytes < 500: s += 0.15
        scores['service_probing'] = min(s, 1.0)

        # ── Exploitation (Execution ← Exploits, Shellcode) ──
        # TCP_FLAGS=21, LONGEST_PKT=749, B/Pkt=276, IN_BYTES=36K
        s = 0.0
        if is_attack_like:
            if tcp_flags > 15: s += 0.25
            if longest_pkt > 500: s += 0.2
            if bpp > 200: s += 0.2
            if in_bytes > 10000: s += 0.15
            if not is_non_tcp: s += 0.1
        scores['exploitation'] = min(s, 1.0)

        # ── Volumetric flooding (Impact ← DoS, Generic) ──
        # Generic: TCP_FLAGS=3.6, dir_ratio=740, OUT_BYTES=11K
        # DoS: TCP_FLAGS=12.5, OUT_BYTES=55K, IN_BYTES=36K
        s = 0.0
        if is_attack_like:
            if dir_ratio > 100 and not is_non_tcp: s += 0.2
            if tcp_flags < 15 and out_bytes > 5000: s += 0.2
            if 3000 < out_bytes < 60000: s += 0.15
            if tcp_flags < 15 and not is_non_tcp: s += 0.15
        scores['volumetric_flood'] = min(s, 1.0)

        # ── SYN flooding (Impact — less common in UNSW) ──
        s = 0.0
        if tcp_flags < 4 and is_attack_like:
            s += 0.4
            if dir_ratio > 500: s += 0.2
            if out_bytes < 1000: s += 0.15
        scores['syn_flooding'] = min(s, 1.0)

        # ── Fuzzing (Initial Access ← Fuzzers) ──
        # SHORTEST_PKT=121, OUT_BYTES=553, dir_ratio=326, TCP_FLAGS=12.7
        s = 0.0
        if is_attack_like:
            if shortest_pkt > 90: s += 0.25
            if 100 < out_bytes < 2000: s += 0.2
            if 100 < dir_ratio < 500: s += 0.2
            if 10 < tcp_flags < 18: s += 0.15
            if not is_non_tcp: s += 0.1
        scores['fuzzing'] = min(s, 1.0)

        # ── Backdoor comm (Persistence ← Backdoor) ──
        # PROTOCOL=101, TCP_FLAGS=2.4, dir_ratio=861, SHORTEST_PKT=131
        s = 0.0
        if is_attack_like:
            if is_non_tcp: s += 0.3
            if tcp_flags < 4: s += 0.25
            if dir_ratio > 500: s += 0.2
            if shortest_pkt > 100: s += 0.1
        scores['backdoor_comm'] = min(s, 1.0)

        # ── Worm spreading (Lateral Movement ← Worms) ──
        # OUT_BYTES=60K, B/Pkt=289, duration=696K, TCP_FLAGS=16.4
        s = 0.0
        if is_attack_like:
            if out_bytes > 40000: s += 0.3
            if bpp > 250: s += 0.2
            if duration > 500000: s += 0.2
            if tcp_flags > 10 and not is_non_tcp: s += 0.15
        scores['worm_spreading'] = min(s, 1.0)

        # ── Data transfer (Exfiltration — not primary in UNSW) ──
        s = 0.0
        if is_attack_like:
            if out_bytes > 100000: s += 0.4
            if dir_ratio > 10 and bpp > 200: s += 0.2
        scores['data_transfer'] = min(s, 1.0)

        # ── Connection attempts ──
        s = 0.0
        if tcp_flags < 4 and fan_in < 6 and out_bytes < 100: s += 0.5
        scores['connection_attempts'] = min(s, 1.0)

        # ── Periodic communication ──
        s = 0.0
        if is_attack_like and not is_non_tcp:
            if 0.5 < dir_ratio < 5: s += 0.3
            if fan_out <= 2: s += 0.2
            if pps < 1000: s += 0.15
        scores['periodic_comm'] = min(s, 1.0)

        return scores

    # ══════════════════════════════════════════════════════
    # CIC-IDS2018 Scoring (Fisher: fan_out=2.31, duration=1.33)
    # ══════════════════════════════════════════════════════

    def _score_cicids(self, tcp_flags, out_bytes, in_bytes, min_ttl,
                       fan_out, fan_in, n_dst, pps, dir_ratio, duration,
                       proportion, bpp, dst_port):
        """CIC-IDS2018 scoring rules based on Fisher discriminability.

        Key separations by port (Fisher=0.69):
          port≈8080 + flags=219 + fan_out>30    → Bot (C&C)
          port≈22 + fan_out=1                   → SSH-BruteForce (Credential Access)
          port≈21 + duration>4M + fan_out=1     → FTP-BruteForce (Credential Access)
          port≈80 + duration>4M + flags<35      → DoS (Impact)
          port≈80 + flags>200                   → DDoS (Impact)
          fan_out>50 + fan_in>100               → Infiltration (Lateral Movement)
          fan_in>100 + moderate fan_out         → Benign (None)
        """
        scores = {}

        is_long_duration = duration > 2_000_000
        is_very_long = duration > 4_000_000
        is_high_fanout = fan_out > 30
        is_high_fanin = fan_in > 100
        has_high_flags = tcp_flags > 200
        is_low_pps = pps < 100
        is_port_ssh = 20 < dst_port < 25
        is_port_ftp = 19 < dst_port < 23
        is_port_http = 75 < dst_port < 85
        is_port_bot = dst_port > 5000

        # ── SYN flooding / DDoS (Impact) ──
        # DDoS-HOIC: flags=219, port=80, fan_out=2
        # DDoS-LOIC-HTTP: flags=203, port=80
        # DDoS-LOIC-UDP: IN_BYTES=5.8M, port=80
        s = 0.0
        if has_high_flags and is_port_http and fan_out <= 5:
            s += 0.6   # High flags + port 80 = DDoS
        elif has_high_flags and fan_out <= 5:
            s += 0.3   # High flags but non-80 port
        if in_bytes > 100000: s += 0.15  # LOIC-UDP floods inbound
        if is_very_long and has_high_flags: s += 0.1
        # Suppress if port=8080 and high fan_out (that's Bot)
        if is_port_bot and is_high_fanout: s *= 0.1
        scores['syn_flooding'] = min(s, 1.0)

        # ── Volumetric / Slow DoS (Impact) ──
        # Slowloris: duration=1.6M, port=80, flags=23
        # SlowHTTPTest: duration=4.3M, port=21, flags=22
        # GoldenEye: duration=4.1M, port=80, flags=31
        # Hulk: duration=4.1M, port=80, flags=27
        s = 0.0
        if is_very_long and not has_high_flags and fan_out <= 2:
            if is_port_http: s += 0.6     # Slow DoS to port 80
            elif is_port_ftp: s += 0.3    # Could be FTP-Brute too
            else: s += 0.2
        if 20 < tcp_flags < 35 and is_port_http and fan_out <= 2: s += 0.15
        if is_long_duration and is_low_pps and fan_out <= 2 and is_port_http: s += 0.1
        # Suppress if SSH/FTP port (that's brute force)
        if is_port_ssh and pps > 100000: s *= 0.2
        scores['volumetric_flood'] = min(s, 1.0)

        # ── Connection attempts / Brute Force (Credential Access) ──
        # FTP-BruteForce: port=21, duration=4.3M, PPS=0, flags=22
        # SSH-Bruteforce: port=22, PPS=378K, flags=27
        # Brute Force-Web: port≈100, flags=213
        s = 0.0
        if is_port_ssh and fan_out <= 5:
            s += 0.6   # SSH port = brute force
            if pps > 100000: s += 0.2  # Very high PPS confirms
        elif is_port_ftp and is_very_long and fan_out <= 2:
            s += 0.5   # FTP port + long duration
            if is_low_pps: s += 0.15
        elif has_high_flags and is_long_duration and fan_out <= 3:
            if not is_port_http and not is_port_bot:
                s += 0.4   # Web brute force (non-80 port)
        scores['connection_attempts'] = min(s, 1.0)

        # ── Periodic communication / Bot (C&C) ──
        # Bot: fan_out=60-92, TCP_FLAGS=219, port=8080, MIN_TTL=128
        s = 0.0
        if is_high_fanout and has_high_flags:
            s += 0.5   # High fan-out + high flags = Bot
            if is_port_bot: s += 0.25  # Port >5000 confirms
            if min_ttl > 100: s += 0.1
            if fan_in <= 15: s += 0.1
        elif is_port_bot and has_high_flags:
            s += 0.4   # Port alone strong signal
        scores['periodic_comm'] = min(s, 1.0)

        # ── Data transfer / Infiltration (Lateral Movement) ──
        # Infiltration: fan_out=94, fan_in=268-443, MIN_TTL=41, port=2231
        s = 0.0
        if is_high_fanout and is_high_fanin:
            s += 0.6   # Very high fan-out AND fan-in
        if fan_in > 200 and fan_out > 30: s += 0.2
        if min_ttl < 50 and is_high_fanout: s += 0.1
        if not is_long_duration and is_high_fanout: s += 0.1
        # Suppress if Bot-like (high flags + port 8080)
        if has_high_flags and is_port_bot: s *= 0.2
        scores['data_transfer'] = min(s, 1.0)

        # ── Normal activity (Benign) ──
        # Benign: fan_in=217-443, fan_out=17-40, port=1371, TCP_FLAGS=65-172
        s = 0.0
        if is_high_fanin and not is_high_fanout:
            s += 0.3
        if not is_long_duration and not has_high_flags: s += 0.2
        if 50 < tcp_flags < 180: s += 0.15
        if proportion > 0.03: s += 0.1
        if dst_port > 500 and not is_port_bot: s += 0.1
        # Suppress if clearly attack-like
        if is_very_long and fan_out <= 2: s *= 0.1
        if has_high_flags and is_high_fanout: s *= 0.1
        scores['normal_activity'] = min(s, 1.0)

        # ── Port scanning ──
        s = 0.0
        if n_dst > 10 and out_bytes < 500 and not is_high_fanin:
            s += 0.4
        if fan_out > 5 and bpp < 100 and not is_high_fanin: s += 0.2
        scores['port_scanning'] = min(s, 1.0)

        # ── Service probing ──
        s = 0.0
        if 3 < fan_out < 30 and out_bytes < 2000 and not is_high_fanin:
            s += 0.3
        if not is_long_duration and not has_high_flags: s += 0.15
        scores['service_probing'] = min(s, 1.0)

        return scores

    # ══════════════════════════════════════════════════════
    # BoT-IoT Scoring (Fisher: TCP_FLAGS=1.82, dir_ratio=1.69)
    # ══════════════════════════════════════════════════════

    # def _score_botiot(self, tcp_flags, out_bytes, min_ttl, fan_out, fan_in,
    #                    n_dst, pps, dir_ratio, duration, direction, proportion):
    #     """BoT-IoT scoring rules (original, unchanged)."""
    #     is_syn_dominant = tcp_flags < 4
    #     scores = {}
    #
    #     # SYN flooding
    #     s = 0.0
    #     if tcp_flags < 4: s += 0.5
    #     if tcp_flags < 2: s += 0.2
    #     if out_bytes < 10: s += 0.15
    #     if min_ttl > 50: s += 0.1
    #     if dir_ratio > 100: s += 0.05
    #     scores['syn_flooding'] = min(s, 1.0)
    #
    #     # Volumetric flooding
    #     s = 0.0
    #     if tcp_flags >= 4 and min_ttl > 50: s += 0.3
    #     if dir_ratio > 100: s += 0.2
    #     if pps > 5000 and out_bytes < 1000: s += 0.2
    #     if fan_out < 5: s += 0.1
    #     scores['volumetric_flood'] = min(s, 1.0)
    #
    #     # Port scanning
    #     s = 0.0
    #     if not is_syn_dominant:
    #         if tcp_flags > 15 and out_bytes < 200: s += 0.35
    #         if pps > 5000: s += 0.2
    #         if n_dst > 5: s += 0.15
    #         if fan_out > 3: s += 0.15
    #         if min_ttl > 20: s += 0.1
    #     scores['port_scanning'] = min(s, 1.0)
    #
    #     # Service probing
    #     s = 0.0
    #     if not is_syn_dominant:
    #         if tcp_flags > 15 and out_bytes < 5000: s += 0.35
    #         if n_dst > 3: s += 0.2
    #         if pps > 1000: s += 0.15
    #         if fan_out > 2: s += 0.15
    #     scores['service_probing'] = min(s, 1.0)
    #
    #     # Data transfer
    #     s = 0.0
    #     if not is_syn_dominant:
    #         if out_bytes > 200000: s += 0.5
    #         elif out_bytes > 50000: s += 0.35
    #         elif out_bytes > 10000 and min_ttl > 30: s += 0.2
    #         if fan_out > 10: s += 0.15
    #         if dir_ratio > 3 and min_ttl > 30: s += 0.15
    #         if tcp_flags > 10: s += 0.1
    #     scores['data_transfer'] = min(s, 1.0)
    #
    #     # Normal activity
    #     s = 0.0
    #     if not is_syn_dominant:
    #         if min_ttl < 25: s += 0.3
    #         if proportion > 0.05: s += 0.2
    #         if 100 < out_bytes < 100000: s += 0.15
    #         if tcp_flags > 15: s += 0.1
    #         if 0.3 < dir_ratio < 3: s += 0.1
    #     if scores.get('data_transfer', 0) > 0.3 and min_ttl < 25:
    #         s += 0.3
    #         scores['data_transfer'] *= 0.3
    #     scores['normal_activity'] = min(s, 1.0)
    #
    #     # Connection attempts
    #     s = 0.0
    #     if tcp_flags < 4 and fan_out <= 3: s += 0.5
    #     if out_bytes < 10: s += 0.2
    #     scores['connection_attempts'] = min(s, 1.0)
    #
    #     # Periodic communication
    #     s = 0.0
    #     if not is_syn_dominant:
    #         if 0.5 < dir_ratio < 2.0: s += 0.3
    #         if fan_out <= 2 and n_dst <= 3: s += 0.25
    #         if pps < 100: s += 0.2
    #         if tcp_flags > 10: s += 0.1
    #         if out_bytes < 1000: s += 0.1
    #     scores['periodic_comm'] = min(s, 1.0)
    #
    #     return scores
    #
    # def _score_botiot(self, tcp_flags, out_bytes, in_bytes, min_ttl,
    #                   fan_out, fan_in, n_dst, pps, dir_ratio, duration,
    #                   direction, proportion):
    #     """BoT-IoT scoring rules."""
    #     is_syn_dominant = tcp_flags < 4
    #     scores = {}
    #
    #     # SYN flooding
    #     s = 0.0
    #     if tcp_flags < 4: s += 0.5
    #     if tcp_flags < 2: s += 0.2
    #     if out_bytes < 10: s += 0.15
    #     if min_ttl > 50: s += 0.1
    #     if dir_ratio > 100: s += 0.05
    #     scores['syn_flooding'] = min(s, 1.0)
    #
    #     # Volumetric flooding
    #     s = 0.0
    #     if tcp_flags >= 4 and min_ttl > 50: s += 0.3
    #     if dir_ratio > 100: s += 0.2
    #     if pps > 5000 and out_bytes < 1000: s += 0.2
    #     if fan_out < 5: s += 0.1
    #     scores['volumetric_flood'] = min(s, 1.0)
    #
    #     # Port scanning
    #     s = 0.0
    #     if not is_syn_dominant:
    #         if tcp_flags > 15 and out_bytes < 200: s += 0.35
    #         if pps > 5000: s += 0.2
    #         if n_dst > 5: s += 0.15
    #         if fan_out > 3: s += 0.15
    #         if min_ttl > 20: s += 0.1
    #     scores['port_scanning'] = min(s, 1.0)
    #
    #     # Service probing
    #     s = 0.0
    #     if not is_syn_dominant:
    #         if tcp_flags > 15 and out_bytes < 5000: s += 0.35
    #         if n_dst > 3: s += 0.2
    #         if pps > 1000: s += 0.15
    #         if fan_out > 2: s += 0.15
    #     scores['service_probing'] = min(s, 1.0)
    #
    #     # Data transfer (Exfiltration)
    #     # BoT-IoT Theft: data stolen INBOUND (high in_bytes, not out_bytes)
    #     s = 0.0
    #     if not is_syn_dominant:
    #         # Outbound exfiltration
    #         if out_bytes > 200000:
    #             s += 0.5
    #         elif out_bytes > 50000:
    #             s += 0.35
    #         elif out_bytes > 10000 and min_ttl > 30:
    #             s += 0.2
    #         # Inbound theft (Theft attack type)
    #         if in_bytes > 200000:
    #             s += 0.5
    #         elif in_bytes > 50000:
    #             s += 0.35
    #         elif in_bytes > 10000 and min_ttl > 30:
    #             s += 0.2
    #         if fan_out > 10: s += 0.15
    #         if dir_ratio > 3 and min_ttl > 30: s += 0.15
    #         if tcp_flags > 10: s += 0.1
    #     scores['data_transfer'] = min(s, 1.0)
    #
    #     # Normal activity
    #     s = 0.0
    #     if not is_syn_dominant:
    #         if min_ttl < 25: s += 0.3
    #         if proportion > 0.05: s += 0.2
    #         if 100 < out_bytes < 100000: s += 0.15
    #         if tcp_flags > 15: s += 0.1
    #         if 0.3 < dir_ratio < 3: s += 0.1
    #     if scores.get('data_transfer', 0) > 0.3 and min_ttl < 25:
    #         s += 0.3
    #         scores['data_transfer'] *= 0.3
    #     scores['normal_activity'] = min(s, 1.0)
    #
    #     # Connection attempts
    #     s = 0.0
    #     if tcp_flags < 4 and fan_out <= 3: s += 0.5
    #     if out_bytes < 10: s += 0.2
    #     scores['connection_attempts'] = min(s, 1.0)
    #
    #     # Periodic communication
    #     s = 0.0
    #     if not is_syn_dominant:
    #         if 0.5 < dir_ratio < 2.0: s += 0.3
    #         if fan_out <= 2 and n_dst <= 3: s += 0.25
    #         if pps < 100: s += 0.2
    #         if tcp_flags > 10: s += 0.1
    #         if out_bytes < 1000: s += 0.1
    #     scores['periodic_comm'] = min(s, 1.0)
    #
    #     return scores
    def _score_botiot(self, tcp_flags, out_bytes, in_bytes, min_ttl,
                      fan_out, fan_in, n_dst, pps, dir_ratio, duration,
                      direction, proportion):
        """BoT-IoT scoring rules."""
        is_syn_dominant = tcp_flags < 4
        scores = {}

        # SYN flooding
        s = 0.0
        if tcp_flags < 4: s += 0.5
        if tcp_flags < 2: s += 0.2
        if out_bytes < 10: s += 0.15
        if min_ttl > 50: s += 0.1
        if dir_ratio > 100: s += 0.05
        scores['syn_flooding'] = min(s, 1.0)

        # Volumetric flooding
        s = 0.0
        if tcp_flags >= 4 and min_ttl > 50: s += 0.3
        if dir_ratio > 100: s += 0.2
        if pps > 5000 and out_bytes < 1000: s += 0.2
        if fan_out < 5: s += 0.1
        scores['volumetric_flood'] = min(s, 1.0)

        # Port scanning
        s = 0.0
        if not is_syn_dominant:
            if tcp_flags > 15 and out_bytes < 200: s += 0.35
            if pps > 5000: s += 0.2
            if n_dst > 5: s += 0.15
            if fan_out > 3: s += 0.15
            if min_ttl > 20: s += 0.1
        scores['port_scanning'] = min(s, 1.0)

        # Service probing
        s = 0.0
        if not is_syn_dominant:
            if tcp_flags > 15 and out_bytes < 5000: s += 0.35
            if n_dst > 3: s += 0.2
            if pps > 1000: s += 0.15
            if fan_out > 2: s += 0.15
        scores['service_probing'] = min(s, 1.0)

        # Data transfer (Exfiltration)
        # BoT-IoT Theft: data stolen INBOUND (high in_bytes, not out_bytes)
        s = 0.0
        if not is_syn_dominant:
            # Outbound exfiltration
            if out_bytes > 200000:
                s += 0.5
            elif out_bytes > 50000:
                s += 0.35
            elif out_bytes > 10000 and min_ttl > 30:
                s += 0.2
            # Inbound theft (Theft attack type)
            if in_bytes > 200000:
                s += 0.5
            elif in_bytes > 50000:
                s += 0.35
            elif in_bytes > 10000 and min_ttl > 30:
                s += 0.2
            if fan_out > 10: s += 0.15
            if dir_ratio > 3 and min_ttl > 30: s += 0.15
            if tcp_flags > 10: s += 0.1
            # Suppress if benign-like (low TTL = benign in BoT-IoT)
            if min_ttl < 20: s *= 0.3
        scores['data_transfer'] = min(s, 1.0)

        # Normal activity (Benign)
        # Key insight: MIN_TTL is the #1 benign separator in BoT-IoT
        #   Benign=16, Recon=27, Exfil=39, Impact=61
        # Benign traffic is DIVERSE: includes SYN connections, large
        # downloads, service requests — so multiple paths must score
        s = 0.0

        # Path 1: Classic benign — low TTL + normal TCP
        if min_ttl < 20:
            s += 0.35  # Low TTL is strongest benign signal
        elif min_ttl < 25:
            s += 0.20

        # Path 2: Large cluster proportion (benign is typically majority)
        if proportion > 0.05:
            s += 0.20
        elif proportion > 0.02:
            s += 0.10

        # Path 3: Moderate bytes (not zero like SYN flood, not extreme like exfil)
        if 50 < out_bytes < 100000 and not is_syn_dominant: s += 0.15

        # Path 4: Normal TCP flags (not SYN-only, not extreme)
        if tcp_flags > 15 and tcp_flags < 100: s += 0.10

        # Path 5: Balanced direction
        if 0.3 < dir_ratio < 3: s += 0.10

        # Path 6: SYN-dominant BUT benign
        # Legitimate SYN connections have low TTL and small clusters
        if is_syn_dominant and min_ttl < 20: s += 0.30

        # Path 7: High-bytes benign (downloads) — low TTL separates from exfil
        if out_bytes > 10000 and min_ttl < 20: s += 0.15

        scores['normal_activity'] = min(s, 1.0)

        # ── Benign suppression: proportional, not hard cutoff ──
        # Only suppress when normal_activity DOMINATES other scores
        benign_score = scores['normal_activity']
        max_attack = max(scores.get('syn_flooding', 0),
                         scores.get('data_transfer', 0),
                         scores.get('service_probing', 0),
                         scores.get('port_scanning', 0),
                         scores.get('periodic_comm', 0))

        if min_ttl < 20 and benign_score > max_attack and benign_score > 0.5:
            # Strong benign signal AND it's the dominant behaviour
            suppress = 0.5  # reduce attack scores by 50%
            for bhv in ['syn_flooding', 'volumetric_flood',
                        'service_probing', 'port_scanning']:
                if bhv in scores:
                    scores[bhv] *= suppress
            # Lighter suppression for data_transfer and periodic_comm
            # (these can co-exist with benign-like TTL in edge cases)
            if 'data_transfer' in scores:
                scores['data_transfer'] *= 0.6
            if 'periodic_comm' in scores:
                scores['periodic_comm'] *= 0.6

        # Connection attempts
        s = 0.0
        if tcp_flags < 4 and fan_out <= 3: s += 0.5
        if out_bytes < 10: s += 0.2
        scores['connection_attempts'] = min(s, 1.0)

        # Periodic communication
        s = 0.0
        if not is_syn_dominant:
            if 0.5 < dir_ratio < 2.0: s += 0.3
            if fan_out <= 2 and n_dst <= 3: s += 0.25
            if pps < 100: s += 0.2
            if tcp_flags > 10: s += 0.1
            if out_bytes < 1000: s += 0.1
            # Suppress if low TTL (benign in BoT-IoT, not C&C)
            if min_ttl < 20: s *= 0.3
        scores['periodic_comm'] = min(s, 1.0)

        return scores
    def score_all_clusters(self, cluster_summaries):
        """Score all clusters with soft behaviour distributions.

        Returns:
            cluster_scores: dict {cluster_idx: {behaviour_scores, tactic_scores, ...}}
        """
        cluster_scores = {}
        for k, summary in cluster_summaries.items():
            result = self.score_cluster(summary)
            cluster_scores[int(k)] = result
            bhv_str = ', '.join(f"{b}:{s:.2f}"
                                for b, s in sorted(result['behaviour_scores'].items(),
                                                    key=lambda x: -x[1])[:3])
            logger.info(f"Cluster {k}: [{bhv_str}] → {result['primary_tactic']}")
        return cluster_scores

    def _map_to_tactics(self, bhv_scores):
        tactic_scores = defaultdict(float)
        for bhv, score in bhv_scores.items():
            if bhv in BEHAVIOUR_TACTIC_WEIGHTS:
                for tactic, weight in BEHAVIOUR_TACTIC_WEIGHTS[bhv].items():
                    tactic_scores[tactic] += score * weight

        # Normalize
        total = sum(tactic_scores.values())
        if total > 0:
            tactic_scores = {t: s / total for t, s in tactic_scores.items()}

        # Ensure all main tactics present
        for t in ['Impact', 'Reconnaissance', 'Exfiltration', 'None',
                  'Command and Control', 'Discovery', 'Credential Access',
                  'Initial Access', 'Execution', 'Lateral Movement',
                  'Persistence']:
            if t not in tactic_scores:
                tactic_scores[t] = 0.0

        return dict(tactic_scores)


class SoftWindowAggregator:
    """Aggregates soft cluster-level behaviours into window-level tactic predictions.

    Each flow inherits its cluster's soft behaviour distribution.
    The window averages these distributions across all flows.
    """

    def __init__(self, window_size=20, tactic_threshold=0.10):
        self.window_size = window_size
        self.tactic_threshold = tactic_threshold

    def create_windows(self, src_ips, cluster_assignments,
                       cluster_scores, y_true=None):
        """Create time windows with soft tactic predictions.

        Args:
            src_ips: (N,) source IP per flow
            cluster_assignments: (N,) cluster index per flow
            cluster_scores: dict from SoftClusterScorer.score_all_clusters()
            y_true: (N,) optional ground truth labels
        """
        src_flows = defaultdict(list)
        for i, src in enumerate(src_ips):
            src_flows[src].append(i)

        windows = []
        for src_ip, indices in src_flows.items():
            for w_start in range(0, len(indices), self.window_size):
                w_end = min(w_start + self.window_size, len(indices))
                w_idx = indices[w_start:w_end]
                n = len(w_idx)

                # Aggregate soft behaviour scores across flows in window
                agg_bhv = defaultdict(float)
                agg_tactic = defaultdict(float)

                for i in w_idx:
                    k = int(cluster_assignments[i])
                    cs = cluster_scores.get(k, {})
                    for bhv, score in cs.get('behaviour_scores', {}).items():
                        agg_bhv[bhv] += score / n
                    for tac, score in cs.get('tactic_scores', {}).items():
                        agg_tactic[tac] += score / n

                # Normalize
                bhv_total = sum(agg_bhv.values())
                if bhv_total > 0:
                    agg_bhv = {k: v / bhv_total for k, v in agg_bhv.items()}
                tac_total = sum(agg_tactic.values())
                if tac_total > 0:
                    agg_tactic = {k: v / tac_total for k, v in agg_tactic.items()}

                primary_tactic = max(agg_tactic, key=agg_tactic.get) if agg_tactic else 'None'
                final_tactics = sorted(
                    [t for t, s in agg_tactic.items() if s >= self.tactic_threshold],
                    key=lambda t: agg_tactic.get(t, 0), reverse=True)

                window = {
                    'src_ip': src_ip,
                    'flow_indices': w_idx,
                    'n_flows': n,
                    'behaviour_scores': dict(agg_bhv),
                    'tactic_scores': dict(agg_tactic),
                    'primary_tactic': primary_tactic,
                    'final_tactics': final_tactics,
                    'clusters': [int(cluster_assignments[i]) for i in w_idx],
                }

                if y_true is not None:
                    w_attacks = [y_true[i] for i in w_idx]
                    majority = Counter(w_attacks).most_common(1)[0]
                    window['majority_attack'] = majority[0]
                    window['attack_purity'] = majority[1] / n
                    window['true_attacks'] = w_attacks

                windows.append(window)

        return windows