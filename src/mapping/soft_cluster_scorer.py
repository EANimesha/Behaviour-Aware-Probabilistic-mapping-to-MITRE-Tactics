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
# NF-CIC-IDS2018-v3: one behaviour per CIC tactic (fixed before any test run)
BEHAVIOUR_TACTIC_WEIGHTS_CIC_V3 = {
    'normal_activity': {'None': 1.0},
    'volumetric_flood': {'Impact': 1.0},
    'brute_force': {'Credential Access': 1.0},
    'bot_beaconing': {'Command and Control': 1.0},
    'internal_spreading': {'Lateral Movement': 1.0},
    'web_exploitation': {'Initial Access': 1.0},
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
    if dataset ==  ('unsw_nb15', 'unsw_nb15_v3'):
        return BEHAVIOUR_TACTIC_WEIGHTS_UNSW
    elif dataset == 'cicids2018_v3':
        return BEHAVIOUR_TACTIC_WEIGHTS_CIC_V3
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
        elif self.dataset == 'bot_iot_v3':
            scores = self._score_botiot_v3(
                bhv, fan_out, out_bytes, min_ttl, tcp_flags, pps,
                dir_ratio, duration, dst_port)
        elif self.dataset == 'unsw_nb15_v3':
            scores = self._score_unsw_v3(
                bhv, fan_out, out_bytes, min_ttl, tcp_flags, pps,
                dir_ratio, duration, dst_port)
        elif self.dataset == 'cicids2018':
            scores = self._score_cicids(
                tcp_flags, out_bytes, in_bytes, min_ttl, fan_out, fan_in,
                n_dst, pps, dir_ratio, duration, proportion, bpp, dst_port)
        elif self.dataset == 'cicids2018_v3':
            scores = self._score_cicids_v3(
                bhv, graph)
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
    # ══════════════════════════════════════════════════════
    # NF-BoT-IoT-v3 Scoring (Fisher on the TRAINING split only)
    # ══════════════════════════════════════════════════════
    # Each rule: (behaviour, feature, op, threshold, weight)
    #   weight    = the feature's Fisher ratio (feature vs tactic, train split)
    #   threshold = geometric mean of the target tactic's mean and the
    #               nearest other tactic's mean (train split, analyse_clusters)
    # Behaviour score = sum of weights of satisfied rules / sum of all its weights.
    # Regenerate with derive_v3_rules() if the training split changes.
    V3_RULES = [
        # Impact (DoS/DDoS): slow-rate, long, SYN-ish, tiny window, high TTL
        ('syn_flooding', 'min_ttl',    '>', 36.8,  1.0495),
        ('syn_flooding', 'iat_max',    '>', 2925,  0.8099),
        ('syn_flooding', 'tcp_flags',  '<', 9.6,   0.7627),
        ('syn_flooding', 'iat_avg',    '>', 1186,  0.7507),
        ('syn_flooding', 'tcp_win_in', '<', 881,   0.6978),
        ('syn_flooding', 'pps',        '<', 2553,  0.3991),
        ('syn_flooding', 'duration',   '>', 12714, 0.2864),
        ('syn_flooding', 'iat_std',    '>', 667,   0.2360),
        # Reconnaissance: very short probes, low TTL, high destination ports
        ('port_scanning', 'min_ttl',  '<', 9.1,   1.0495),
        ('port_scanning', 'iat_max',  '<', 61,    0.8099),
        ('port_scanning', 'iat_avg',  '<', 22,    0.7507),
        ('port_scanning', 'dst_port', '>', 11432, 0.4354),
        ('port_scanning', 'duration', '<', 308,   0.2864),
        ('port_scanning', 'iat_std',  '<', 13,    0.2360),
        # Exfiltration (Theft): wide fan-out, big window, outbound-heavy
        ('data_transfer', 'tcp_win_in', '>', 11475,  0.6978),
        ('data_transfer', 'fan_out',    '>', 30.5,   0.6119),
        ('data_transfer', 'out_bytes',  '>', 100000, 0.2),   # not in Fisher top 10: fixed low weight
        ('data_transfer', 'dir_ratio',  '>', 3.0,    0.2),   # not in Fisher top 10: fixed low weight
        # Benign: sits BETWEEN the attack extremes on the top features
        ('normal_activity', 'min_ttl',    'in', (9.1, 36.8),  1.0495),
        ('normal_activity', 'iat_max',    'in', (61, 2925),   0.8099),
        ('normal_activity', 'tcp_flags',  '>',  9.6,          0.7627),
        ('normal_activity', 'iat_avg',    'in', (22, 1186),   0.7507),
        ('normal_activity', 'tcp_win_in', 'in', (881, 11475), 0.6978),
        ('normal_activity', 'fan_out',    '<',  30.5,         0.6119),
        ('normal_activity', 'duration',   'in', (308, 12714), 0.2864),
    ]

    # NF-UNSW-NB15-v3: same derivation, from ITS training split, per ATTACK TYPE
    # (attack_means_v3.py): a tactic covering two attack types (Impact = DoS +
    # Generic, Execution = Exploits + Shellcode) has a mean matching neither, so
    # each type gets its own profile 'behaviour#Attack'; a behaviour's score is the
    # best of its profiles. Behaviour per tactic:
    #   Discovery=service_probing, Execution=exploitation, Impact=volumetric_flood,
    #   Initial Access=fuzzing, Lateral Movement=worm_spreading, None=normal_activity,
    #   Persistence=backdoor_comm, Reconnaissance=port_scanning
    # threshold = geometric mean of the type's mean and the nearest other type's mean
    #   on each side; types within 5% of each other count as tied (one group)
    # weight = Fisher ratio (analyse_clusters, top-10 features). Excluded:
    #   - fan_in (Fisher 39.7): every attack has fan_in = 4.0, it counts the attacker
    #     hosts, so it identifies the attackers rather than a behaviour.
    #   - MIN_TTL for attack-vs-attack rules: its Fisher comes from benign (~31) vs
    #     attack (150-254); between attack types it is a bimodal testbed artifact
    #     (Generic clusters sit at TTL 253 with a type mean of 231) and, at weight
    #     2.18, would override every other feature. So attack profiles use it only
    #     as a gate (V3_GATE_UNSW): benign/attack boundary, same threshold as
    #     normal_activity's TTL rule.
    V3_RULES_UNSW = [
        ('service_probing', 'tcp_flags', '>', 23.65, 0.1568),
        ('service_probing', 'longest_pkt', 'in', (392.5, 480.4), 0.1154),
        ('service_probing', 'dst_port', '<', 265.7, 0.1078),
        ('service_probing', 'iat_in_avg', 'in', (104.9, 119.3), 0.0858),
        ('service_probing', 'pps_flow', '<', 195.5, 0.0856),
        ('service_probing', 'iat_in_std', 'in', (177.5, 206.2), 0.065),
        ('service_probing', 'dir_ratio_flow', 'in', (1.342, 2.225), 0.0397),
        ('service_probing', 'bpp_flow', 'in', (97.71, 115.4), 0.0367),
        ('exploitation#Exploits', 'tcp_flags', 'in', (18.65, 23.65), 0.1568),
        ('exploitation#Exploits', 'longest_pkt', '>', 576.6, 0.1154),
        ('exploitation#Exploits', 'dst_port', 'in', (2415.0, 2672.0), 0.1078),
        ('exploitation#Exploits', 'iat_in_avg', 'in', (87.1, 95.21), 0.0858),
        ('exploitation#Exploits', 'pps_flow', 'in', (195.5, 2096.0), 0.0856),
        ('exploitation#Exploits', 'iat_in_std', 'in', (140.6, 159.9), 0.065),
        ('exploitation#Exploits', 'dir_ratio_flow', 'in', (6.625, 13.48), 0.0397),
        ('exploitation#Exploits', 'bpp_flow', 'in', (238.1, 288.8), 0.0367),
        ('exploitation#Shellcode', 'tcp_flags', 'in', (12.59, 14.3), 0.1568),
        ('exploitation#Shellcode', 'longest_pkt', 'in', (217.4, 312.4), 0.1154),
        ('exploitation#Shellcode', 'dst_port', '>', 14570.0, 0.1078),
        ('exploitation#Shellcode', 'iat_in_avg', 'in', (31.63, 61.6), 0.0858),
        ('exploitation#Shellcode', 'pps_flow', 'in', (5975.0, 6644.0), 0.0856),
        ('exploitation#Shellcode', 'iat_in_std', 'in', (53.33, 98.77), 0.065),
        ('exploitation#Shellcode', 'dir_ratio_flow', '<', 0.6325, 0.0397),
        ('exploitation#Shellcode', 'bpp_flow', 'in', (115.4, 138.5), 0.0367),
        ('volumetric_flood#DoS', 'tcp_flags', 'in', (15.99, 18.65), 0.1568),
        ('volumetric_flood#DoS', 'longest_pkt', 'in', (480.4, 576.6), 0.1154),
        ('volumetric_flood#DoS', 'dst_port', 'in', (2672.0, 5633.0), 0.1078),
        ('volumetric_flood#DoS', 'iat_in_avg', 'in', (61.6, 81.3), 0.0858),
        ('volumetric_flood#DoS', 'pps_flow', 'in', (3094.0, 3632.0), 0.0856),
        ('volumetric_flood#DoS', 'iat_in_std', 'in', (98.77, 126.6), 0.065),
        ('volumetric_flood#DoS', 'dir_ratio_flow', 'in', (4.266, 5.444), 0.0397),
        ('volumetric_flood#DoS', 'bpp_flow', 'in', (185.5, 207.6), 0.0367),
        ('volumetric_flood#Generic', 'tcp_flags', '<', 8.117, 0.1568),
        ('volumetric_flood#Generic', 'longest_pkt', 'in', (217.4, 312.4), 0.1154),
        ('volumetric_flood#Generic', 'dst_port', 'in', (1161.0, 1921.0), 0.1078),
        ('volumetric_flood#Generic', 'iat_in_avg', 'in', (14.38, 31.63), 0.0858),
        ('volumetric_flood#Generic', 'pps_flow', '>', 10210.0, 0.0856),
        ('volumetric_flood#Generic', 'iat_in_std', 'in', (32.3, 53.33), 0.065),
        ('volumetric_flood#Generic', 'dir_ratio_flow', 'in', (2.225, 3.399), 0.0397),
        ('volumetric_flood#Generic', 'bpp_flow', 'in', (138.5, 163.3), 0.0367),
        ('fuzzing', 'tcp_flags', 'in', (12.59, 14.3), 0.1568),
        ('fuzzing', 'longest_pkt', 'in', (312.4, 392.5), 0.1154),
        ('fuzzing', 'dst_port', 'in', (265.7, 1161.0), 0.1078),
        ('fuzzing', 'iat_in_avg', 'in', (61.6, 81.3), 0.0858),
        ('fuzzing', 'pps_flow', 'in', (5062.0, 5975.0), 0.0856),
        ('fuzzing', 'iat_in_std', 'in', (98.77, 140.6), 0.065),
        ('fuzzing', 'dir_ratio_flow', 'in', (0.9798, 1.342), 0.0397),
        ('fuzzing', 'bpp_flow', 'in', (163.3, 185.5), 0.0367),
        ('worm_spreading', 'tcp_flags', 'in', (15.99, 18.65), 0.1568),
        ('worm_spreading', 'longest_pkt', 'in', (576.6, 700.3), 0.1154),
        ('worm_spreading', 'dst_port', 'in', (1161.0, 1921.0), 0.1078),
        ('worm_spreading', 'iat_in_avg', 'in', (95.21, 104.9), 0.0858),
        ('worm_spreading', 'pps_flow', 'in', (2096.0, 3094.0), 0.0856),
        ('worm_spreading', 'iat_in_std', 'in', (159.9, 177.5), 0.065),
        ('worm_spreading', 'dir_ratio_flow', '>', 13.48, 0.0397),
        ('worm_spreading', 'bpp_flow', '>', 288.8, 0.0367),
        ('normal_activity', 'min_ttl', '<', 43.98, 2.1765),
        ('normal_activity', 'tcp_flags', 'in', (18.65, 23.65), 0.1568),
        ('normal_activity', 'longest_pkt', '>', 670.0, 0.1154),
        ('normal_activity', 'dst_port', 'in', (5633.0, 14570.0), 0.1078),
        ('normal_activity', 'iat_in_avg', '<', 14.38, 0.0858),
        ('normal_activity', 'pps_flow', 'in', (3632.0, 4132.0), 0.0856),
        ('normal_activity', 'iat_in_std', '<', 32.3, 0.065),
        ('normal_activity', 'dir_ratio_flow', 'in', (5.444, 6.625), 0.0397),
        ('normal_activity', 'bpp_flow', 'in', (207.6, 238.1), 0.0367),
        ('backdoor_comm', 'tcp_flags', 'in', (14.3, 15.99), 0.1568),
        ('backdoor_comm', 'longest_pkt', 'in', (392.5, 480.4), 0.1154),
        ('backdoor_comm', 'dst_port', 'in', (2415.0, 2672.0), 0.1078),
        ('backdoor_comm', 'iat_in_avg', '>', 119.3, 0.0858),
        ('backdoor_comm', 'pps_flow', 'in', (4132.0, 5062.0), 0.0856),
        ('backdoor_comm', 'iat_in_std', '>', 206.2, 0.065),
        ('backdoor_comm', 'dir_ratio_flow', 'in', (3.399, 4.266), 0.0397),
        ('backdoor_comm', 'bpp_flow', 'in', (185.5, 207.6), 0.0367),
        ('port_scanning', 'tcp_flags', 'in', (8.117, 12.59), 0.1568),
        ('port_scanning', 'longest_pkt', '<', 217.4, 0.1154),
        ('port_scanning', 'dst_port', 'in', (1921.0, 2415.0), 0.1078),
        ('port_scanning', 'iat_in_avg', 'in', (81.3, 87.1), 0.0858),
        ('port_scanning', 'pps_flow', 'in', (6644.0, 10210.0), 0.0856),
        ('port_scanning', 'iat_in_std', 'in', (121.9, 140.6), 0.065),
        ('port_scanning', 'dir_ratio_flow', 'in', (0.6325, 0.9798), 0.0397),
        ('port_scanning', 'bpp_flow', '<', 97.71, 0.0367),
    ]

    V3_RULES_CICIDS =[
    ('bot_beaconing', 'shortest_pkt', '<', 44.9, 2.3901),
    ('bot_beaconing', 'bpp_flow', 'in', (72.58, 93.62), 1.0765),
    ('bot_beaconing', 'iat_avg', 'in', (0.8158, 3.273), 0.8105),
    ('bot_beaconing', 'fan_out', 'in', (7.14, 19.42), 0.5278),
    ('bot_beaconing', 'iat_max', 'in', (5.142, 13.2), 0.3495),
    ('bot_beaconing', 'tcp_win_in', 'in', (7273.0, 9579.0), 0.317),
    ('bot_beaconing', 'protocol', '<', 6.782, 0.2961),
    ('bot_beaconing', 'iat_std', 'in', (1.939, 5.098), 0.2768),
    ('bot_beaconing', 'duration', 'in', (3.44, 16.16), 0.2602),
    ('brute_force#Brute_Force_-Web', 'shortest_pkt', 'in', (84.82, 226.4), 2.3901),
    ('brute_force#Brute_Force_-Web', 'bpp_flow', 'in', (156.6, 173.6), 1.0765),
    ('brute_force#Brute_Force_-Web', 'iat_avg', 'in', (2716.0, 3798.0), 0.8105),
    ('brute_force#Brute_Force_-Web', 'fan_out', 'in', (2.613, 7.14), 0.5278),
    ('brute_force#Brute_Force_-Web', 'iat_max', 'in', (5724.0, 8274.0), 0.3495),
    ('brute_force#Brute_Force_-Web', 'tcp_win_in', 'in', (9579.0, 11860.0), 0.317),
    ('brute_force#Brute_Force_-Web', 'protocol', 'in', (6.782, 9.181), 0.2961),
    ('brute_force#Brute_Force_-Web', 'iat_std', 'in', (2048.0, 2915.0), 0.2768),
    ('brute_force#Brute_Force_-Web', 'duration', 'in', (14830.0, 18370.0), 0.2602),
    ('brute_force#FTP-BruteForce', 'shortest_pkt', '<', 44.9, 2.3901),
    ('brute_force#FTP-BruteForce', 'bpp_flow', '<', 54.77, 1.0765),
    ('brute_force#FTP-BruteForce', 'iat_avg', '<', 0.1489, 0.8105),
    ('brute_force#FTP-BruteForce', 'fan_out', '<', 1.095, 0.5278),
    ('brute_force#FTP-BruteForce', 'iat_max', '<', 5.142, 0.3495),
    ('brute_force#FTP-BruteForce', 'tcp_win_in', 'in', (25640.0, 41970.0), 0.317),
    ('brute_force#FTP-BruteForce', 'protocol', '<', 6.782, 0.2961),
    ('brute_force#FTP-BruteForce', 'iat_std', '<', 1.939, 0.2768),
    ('brute_force#FTP-BruteForce', 'duration', '<', 3.44, 0.2602),
    ('brute_force#SSH-Bruteforce', 'shortest_pkt', 'in', (44.9, 53.33), 2.3901),
    ('brute_force#SSH-Bruteforce', 'bpp_flow', 'in', (142.0, 156.6), 1.0765),
    ('brute_force#SSH-Bruteforce', 'iat_avg', 'in', (8.571, 96.48), 0.8105),
    ('brute_force#SSH-Bruteforce', 'fan_out', '<', 1.095, 0.5278),
    ('brute_force#SSH-Bruteforce', 'iat_max', 'in', (41.7, 273.6), 0.3495),
    ('brute_force#SSH-Bruteforce', 'tcp_win_in', 'in', (25640.0, 41970.0), 0.317),
    ('brute_force#SSH-Bruteforce', 'protocol', '<', 6.782, 0.2961),
    ('brute_force#SSH-Bruteforce', 'iat_std', 'in', (23.33, 148.6), 0.2768),
    ('brute_force#SSH-Bruteforce', 'duration', 'in', (89.79, 1393.0), 0.2602),
    ('internal_spreading', 'shortest_pkt', 'in', (53.33, 57.83), 2.3901),
    ('internal_spreading', 'bpp_flow', 'in', (93.62, 111.5), 1.0765),
    ('internal_spreading', 'iat_avg', 'in', (96.48, 678.8), 0.8105),
    ('internal_spreading', 'fan_out', '>', 27.61, 0.5278),
    ('internal_spreading', 'iat_max', 'in', (1280.0, 3221.0), 0.3495),
    ('internal_spreading', 'tcp_win_in', 'in', (2637.0, 5213.0), 0.317),
    ('internal_spreading', 'protocol', 'in', (9.181, 12.95), 0.2961),
    ('internal_spreading', 'iat_std', 'in', (148.6, 1142.0), 0.2768),
    ('internal_spreading', 'duration', 'in', (1393.0, 8338.0), 0.2602),
    ('normal_activity', 'shortest_pkt', 'in', (53.33, 57.83), 2.3901),
    ('normal_activity', 'bpp_flow', 'in', (129.4, 142.0), 1.0765),
    ('normal_activity', 'iat_avg', 'in', (947.0, 1330.0), 0.8105),
    ('normal_activity', 'fan_out', 'in', (19.42, 27.61), 0.5278),
    ('normal_activity', 'iat_max', 'in', (4761.0, 5724.0), 0.3495),
    ('normal_activity', 'tcp_win_in', 'in', (5213.0, 7273.0), 0.317),
    ('normal_activity', 'protocol', 'in', (9.181, 12.95), 0.2961),
    ('normal_activity', 'iat_std', 'in', (1647.0, 1838.0), 0.2768),
    ('normal_activity', 'duration', 'in', (13430.0, 14830.0), 0.2602),
    ('volumetric_flood#DDOS_attack-HOIC', 'shortest_pkt', '<', 44.9, 2.3901),
    ('volumetric_flood#DDOS_attack-HOIC', 'bpp_flow', 'in', (156.6, 173.6), 1.0765),
    ('volumetric_flood#DDOS_attack-HOIC', 'iat_avg', 'in', (3.273, 8.571), 0.8105),
    ('volumetric_flood#DDOS_attack-HOIC', 'fan_out', 'in', (1.553, 2.613), 0.5278),
    ('volumetric_flood#DDOS_attack-HOIC', 'iat_max', 'in', (13.2, 41.7), 0.3495),
    ('volumetric_flood#DDOS_attack-HOIC', 'tcp_win_in', '>', 41970.0, 0.317),
    ('volumetric_flood#DDOS_attack-HOIC', 'protocol', '<', 6.782, 0.2961),
    ('volumetric_flood#DDOS_attack-HOIC', 'iat_std', 'in', (5.098, 11.72), 0.2768),
    ('volumetric_flood#DDOS_attack-HOIC', 'duration', 'in', (16.16, 89.79), 0.2602),
    ('volumetric_flood#DDOS_attack-LOIC-UDP', 'shortest_pkt', 'in', (57.83, 84.82), 2.3901),
    ('volumetric_flood#DDOS_attack-LOIC-UDP', 'bpp_flow', 'in', (54.77, 72.58), 1.0765),
    ('volumetric_flood#DDOS_attack-LOIC-UDP', 'iat_avg', 'in', (0.1489, 0.8158), 0.8105),
    ('volumetric_flood#DDOS_attack-LOIC-UDP', 'fan_out', 'in', (1.553, 2.613), 0.5278),
    ('volumetric_flood#DDOS_attack-LOIC-UDP', 'iat_max', 'in', (273.6, 1280.0), 0.3495),
    ('volumetric_flood#DDOS_attack-LOIC-UDP', 'tcp_win_in', '<', 462.8, 0.317),
    ('volumetric_flood#DDOS_attack-LOIC-UDP', 'protocol', '>', 16.37, 0.2961),
    ('volumetric_flood#DDOS_attack-LOIC-UDP', 'iat_std', 'in', (11.72, 23.33), 0.2768),
    ('volumetric_flood#DDOS_attack-LOIC-UDP', 'duration', '>', 100800.0, 0.2602),
    ('volumetric_flood#DDoS_attacks-LOIC-HTTP', 'shortest_pkt', '<', 44.9, 2.3901),
    ('volumetric_flood#DDoS_attacks-LOIC-HTTP', 'bpp_flow', 'in', (142.0, 156.6), 1.0765),
    ('volumetric_flood#DDoS_attacks-LOIC-HTTP', 'iat_avg', 'in', (3798.0, 8596.0), 0.8105),
    ('volumetric_flood#DDoS_attacks-LOIC-HTTP', 'fan_out', 'in', (1.553, 2.613), 0.5278),
    ('volumetric_flood#DDoS_attacks-LOIC-HTTP', 'iat_max', 'in', (13920.0, 25630.0), 0.3495),
    ('volumetric_flood#DDoS_attacks-LOIC-HTTP', 'tcp_win_in', 'in', (7273.0, 9579.0), 0.317),
    ('volumetric_flood#DDoS_attacks-LOIC-HTTP', 'protocol', '<', 6.782, 0.2961),
    ('volumetric_flood#DDoS_attacks-LOIC-HTTP', 'iat_std', 'in', (5725.0, 9569.0), 0.2768),
    ('volumetric_flood#DDoS_attacks-LOIC-HTTP', 'duration', 'in', (18370.0, 40070.0), 0.2602),
    ('volumetric_flood#DoS_attacks-GoldenEye', 'shortest_pkt', 'in', (44.9, 53.33), 2.3901),
    ('volumetric_flood#DoS_attacks-GoldenEye', 'bpp_flow', 'in', (192.3, 332.1), 1.0765),
    ('volumetric_flood#DoS_attacks-GoldenEye', 'iat_avg', 'in', (1988.0, 2716.0), 0.8105),
    ('volumetric_flood#DoS_attacks-GoldenEye', 'fan_out', '<', 1.095, 0.5278),
    ('volumetric_flood#DoS_attacks-GoldenEye', 'iat_max', 'in', (8274.0, 13920.0), 0.3495),
    ('volumetric_flood#DoS_attacks-GoldenEye', 'tcp_win_in', 'in', (17540.0, 25640.0), 0.317),
    ('volumetric_flood#DoS_attacks-GoldenEye', 'protocol', '<', 6.782, 0.2961),
    ('volumetric_flood#DoS_attacks-GoldenEye', 'iat_std', 'in', (2915.0, 5725.0), 0.2768),
    ('volumetric_flood#DoS_attacks-GoldenEye', 'duration', 'in', (8338.0, 13430.0), 0.2602),
    ('volumetric_flood#DoS_attacks-Hulk', 'shortest_pkt', 'in', (44.9, 53.33), 2.3901),
    ('volumetric_flood#DoS_attacks-Hulk', 'bpp_flow', 'in', (173.6, 192.3), 1.0765),
    ('volumetric_flood#DoS_attacks-Hulk', 'iat_avg', 'in', (678.8, 947.0), 0.8105),
    ('volumetric_flood#DoS_attacks-Hulk', 'fan_out', '<', 1.095, 0.5278),
    ('volumetric_flood#DoS_attacks-Hulk', 'iat_max', 'in', (3221.0, 4761.0), 0.3495),
    ('volumetric_flood#DoS_attacks-Hulk', 'tcp_win_in', 'in', (25640.0, 41970.0), 0.317),
    ('volumetric_flood#DoS_attacks-Hulk', 'protocol', '<', 6.782, 0.2961),
    ('volumetric_flood#DoS_attacks-Hulk', 'iat_std', 'in', (1142.0, 1647.0), 0.2768),
    ('volumetric_flood#DoS_attacks-Hulk', 'duration', 'in', (82100.0, 100800.0), 0.2602),
    ('volumetric_flood#DoS_attacks-SlowHTTPTest', 'shortest_pkt', '<', 44.9, 2.3901),
    ('volumetric_flood#DoS_attacks-SlowHTTPTest', 'bpp_flow', '<', 54.77, 1.0765),
    ('volumetric_flood#DoS_attacks-SlowHTTPTest', 'iat_avg', '<', 0.1489, 0.8105),
    ('volumetric_flood#DoS_attacks-SlowHTTPTest', 'fan_out', '<', 1.095, 0.5278),
    ('volumetric_flood#DoS_attacks-SlowHTTPTest', 'iat_max', '<', 5.142, 0.3495),
    ('volumetric_flood#DoS_attacks-SlowHTTPTest', 'tcp_win_in', 'in', (25640.0, 41970.0), 0.317),
    ('volumetric_flood#DoS_attacks-SlowHTTPTest', 'protocol', '<', 6.782, 0.2961),
    ('volumetric_flood#DoS_attacks-SlowHTTPTest', 'iat_std', '<', 1.939, 0.2768),
    ('volumetric_flood#DoS_attacks-SlowHTTPTest', 'duration', '<', 3.44, 0.2602),
    ('volumetric_flood#DoS_attacks-Slowloris', 'shortest_pkt', 'in', (44.9, 53.33), 2.3901),
    ('volumetric_flood#DoS_attacks-Slowloris', 'bpp_flow', 'in', (111.5, 129.4), 1.0765),
    ('volumetric_flood#DoS_attacks-Slowloris', 'iat_avg', 'in', (1330.0, 1988.0), 0.8105),
    ('volumetric_flood#DoS_attacks-Slowloris', 'fan_out', '<', 1.095, 0.5278),
    ('volumetric_flood#DoS_attacks-Slowloris', 'iat_max', 'in', (5724.0, 8274.0), 0.3495),
    ('volumetric_flood#DoS_attacks-Slowloris', 'tcp_win_in', 'in', (11860.0, 17540.0), 0.317),
    ('volumetric_flood#DoS_attacks-Slowloris', 'protocol', '<', 6.782, 0.2961),
    ('volumetric_flood#DoS_attacks-Slowloris', 'iat_std', 'in', (1838.0, 2048.0), 0.2768),
    ('volumetric_flood#DoS_attacks-Slowloris', 'duration', 'in', (14830.0, 18370.0), 0.2602),
    ('web_exploitation#Brute_Force_-XSS', 'shortest_pkt', '>', 441.4, 2.3901),
    ('web_exploitation#Brute_Force_-XSS', 'bpp_flow', '>', 332.1, 1.0765),
    ('web_exploitation#Brute_Force_-XSS', 'iat_avg', '>', 8596.0, 0.8105),
    ('web_exploitation#Brute_Force_-XSS', 'fan_out', 'in', (1.095, 1.553), 0.5278),
    ('web_exploitation#Brute_Force_-XSS', 'iat_max', '>', 25630.0, 0.3495),
    ('web_exploitation#Brute_Force_-XSS', 'tcp_win_in', 'in', (462.8, 1250.0), 0.317),
    ('web_exploitation#Brute_Force_-XSS', 'protocol', 'in', (15.24, 16.37), 0.2961),
    ('web_exploitation#Brute_Force_-XSS', 'iat_std', '>', 9569.0, 0.2768),
    ('web_exploitation#Brute_Force_-XSS', 'duration', 'in', (40070.0, 82100.0), 0.2602),
    ('web_exploitation#SQL_Injection', 'shortest_pkt', 'in', (226.4, 441.4), 2.3901),
    ('web_exploitation#SQL_Injection', 'bpp_flow', '>', 332.1, 1.0765),
    ('web_exploitation#SQL_Injection', 'iat_avg', '>', 8596.0, 0.8105),
    ('web_exploitation#SQL_Injection', 'fan_out', 'in', (1.095, 1.553), 0.5278),
    ('web_exploitation#SQL_Injection', 'iat_max', '>', 25630.0, 0.3495),
    ('web_exploitation#SQL_Injection', 'tcp_win_in', 'in', (1250.0, 2637.0), 0.317),
    ('web_exploitation#SQL_Injection', 'protocol', 'in', (12.95, 15.24), 0.2961),
    ('web_exploitation#SQL_Injection', 'iat_std', '>', 9569.0, 0.2768),
    ('web_exploitation#SQL_Injection', 'duration', 'in', (40070.0, 82100.0), 0.2602),
]

    def _v3_rules(self):
        return self.V3_RULES_UNSW if self.dataset == 'unsw_nb15_v3' else self.V3_RULES

    def _score_botiot_v3(self, bhv, fan_out, out_bytes, min_ttl, tcp_flags,
                         pps, dir_ratio, duration, dst_port):
        """NF-BoT-IoT-v3 rules, data-driven from V3_RULES."""
        vals = {
            'min_ttl': min_ttl, 'tcp_flags': tcp_flags, 'fan_out': fan_out,
            'out_bytes': out_bytes, 'pps': pps, 'dir_ratio': dir_ratio,
            'duration': duration, 'dst_port': dst_port,
            'iat_max': bhv.get('avg_iat_max'), 'iat_avg': bhv.get('avg_iat_avg'),
            'iat_std': bhv.get('avg_iat_std'), 'tcp_win_in': bhv.get('avg_tcp_win_in'),
        }
        missing = sorted(k for k, v in vals.items() if v is None)
        if missing:
            logger.warning(f"bot_iot_v3 scorer: missing {missing} "
                           f"(summarizer needs df_raw with NF-v3 columns); rules skipped")

        hit, total = defaultdict(float), defaultdict(float)
        for bhv_name, feat, op, thr, w in self.V3_RULES:
            v = vals[feat]
            if v is None:
                continue
            total[bhv_name] += w
            if op == '>':
                ok = v > thr
            elif op == '<':
                ok = v < thr
            else:
                ok = thr[0] <= v <= thr[1]
            if ok:
                hit[bhv_name] += w
        return {b: (hit[b] / total[b] if total[b] > 0 else 0.0) for b in total}

    def _score_unsw_v3(self, bhv, fan_out, out_bytes, min_ttl, tcp_flags,
                         pps, dir_ratio, duration, dst_port):
        """UNSW-v3 rules, data-driven from V3_RULES."""
        vals = {
            'min_ttl': min_ttl, 'tcp_flags': tcp_flags, 'fan_out': fan_out,
            'out_bytes': out_bytes, 'pps': pps, 'dir_ratio': dir_ratio,
            'duration': duration, 'dst_port': dst_port,
            'iat_max': bhv.get('avg_iat_max'), 'iat_avg': bhv.get('avg_iat_avg'),
            'iat_std': bhv.get('avg_iat_std'), 'tcp_win_in': bhv.get('avg_tcp_win_in'),
            # used by V3_RULES_UNSW
            'longest_pkt': bhv.get('avg_longest_pkt_bytes'),
            'iat_in_avg': bhv.get('avg_iat_in_avg'), 'iat_in_std': bhv.get('avg_iat_in_std'),
            'pps_flow': bhv.get('avg_pps_flow'), 'bpp_flow': bhv.get('avg_bpp_flow'),
            'dir_ratio_flow': bhv.get('avg_dir_ratio_flow'),
        }
        missing = sorted(k for k, v in vals.items() if v is None and k in self.V3_RULES_UNSW)
        if missing:
            logger.warning(f"{self.dataset} scorer: missing {missing} "
                           f"(summarizer needs df_raw with NF-v3 columns); rules skipped")

        hit, total = defaultdict(float), defaultdict(float)
        for bhv_name, feat, op, thr, w in self._v3_rules():
            v = vals[feat]
            if v is None:
                continue
            total[bhv_name] += w
            if op == '>':
                ok = v > thr
            elif op == '<':
                ok = v < thr
            else:
                ok = thr[0] <= v <= thr[1]
            if ok:
                hit[bhv_name] += w
        scores = {}
        for name in total:  # 'behaviour#Attack' profiles -> best per behaviour
            b = name.split('#')[0]
            scores[b] = max(scores.get(b, 0.0), hit[name] / total[name] if total[name] > 0 else 0.0)
        return scores

    def _score_cicids_v3(self, bhv, graph):
        """NF-CIC-IDS2018-v3 expert rules (shared 12-behaviour vocabulary).

        Every threshold comes from the TRAIN split only (analyse_clusters.py:
        per-attack means / per-tactic medians, fisher_cicids2018_v3.csv), written
        before any CIC test result of this rule set. Cluster inputs are MEDIANS
        where available (robust to heavy tails); port = median dst port.
        Port/flag rules are testbed signatures (dataset-specific, stated in paper).

        Training evidence (attack type: median/mean values):
          Bot:        port 8080, TCP flags 219, TTL 128, IAT avg 2 ms, 11 ms flows
          SSH-BF:     port 22, fan-out 1, flags 27, ~365 ms
          FTP-BF:     port 21, 1 ms, 50 B/pkt, flags 22  (== SlowHTTPTest: inseparable)
          HOIC:       port 80, flags 219, fan-out 2, 22 ms
          LOIC-HTTP:  port 80, flags 223, fan-out 2, ~20 s, IAT avg ~5 s
          LOIC-UDP:   UDP, inbound ~7 MB, ~120 s
          GoldenEye:  port 80, flags 30, 204 B/pkt;  Hulk: port 80, flags 27, 181 B/pkt
          Slowloris:  port 80, flags 22, 124 B/pkt (slow, held-open)
          XSS/SQLi:   shortest pkt 441-528, 542 B/pkt, ~80 s, TCP window 0
          Infiltr.:   fan-out 28-30, TTL 0, TCP window 1024, flags 22, 1 ms (internal scan)
          Benign:     DNS (UDP port 53, TTL 0, fan-out 16-100) or Windows clients
                      (TTL 128, flags ~218, ports 443/445/3389)
        """
        def b(key, default=None):
            v = bhv.get('med_' + key)
            return v if v is not None else bhv.get('avg_' + key, default)
        port = bhv.get('median_dst_port', bhv.get('avg_dst_port', 0)) or 0
        udp = bhv.get('share_udp', 0.0) or 0.0
        flags = bhv.get('avg_tcp_flags', 0.0) or 0.0
        dur = b('duration_ms', 0.0) or 0.0
        ttl = b('min_ttl', 64.0)
        shortest = b('shortest_pkt_bytes', 40.0) or 0.0
        bpp = bhv.get('med_bpp_flow', bhv.get('avg_bpp_flow', 0.0)) or 0.0
        win = b('tcp_win_in', 0.0) or 0.0
        iat = b('iat_avg', 0.0) or 0.0
        in_bytes = bhv.get('avg_in_bytes', 0.0) or 0.0
        fan_out = graph.get('med_fan_out', graph.get('avg_fan_out', 1.0)) or 0.0

        is_http = 75 <= port <= 85
        is_ftp_ssh = 20 <= port <= 23
        high_flags = flags > 200          # Bot/HOIC/LOIC 219-223 vs DoS/BF 22-30
        scores = {}

        # Bot -> C2: port 8080 + flags 219 + 2 ms beacons + TTL 128
        s = 0.0
        if 8000 <= port <= 8100: s += 0.5
        if high_flags: s += 0.2
        if iat < 10 and dur < 100: s += 0.15
        if ttl is not None and ttl >= 120: s += 0.15
        if is_http: s *= 0.2
        scores['periodic_comm'] = min(s, 1.0)

        # FTP/SSH brute force -> Credential Access
        s = 0.0
        if is_ftp_ssh: s += 0.5
        if fan_out <= 2: s += 0.2
        if flags < 35: s += 0.15
        if dur < 2000: s += 0.15
        if not is_ftp_ssh: s *= 0.2        # FTP/SSH brute force always hits 21/22
        scores['connection_attempts'] = min(s, 1.0)

        # Slow DoS (Slowloris) -> Impact: port 80, low flags, small B/pkt
        s = 0.0
        if is_http: s += 0.4
        if flags < 35: s += 0.2
        if 0 < bpp < 140: s += 0.2      # Slowloris 124 vs GoldenEye/Hulk 181-204
        if fan_out <= 2: s += 0.1
        if udp > 0.5 or not is_http: s *= 0.2   # every Slowloris flow targets port 80
        if fan_out > 3: s *= 0.3           # DoS sources hit 1-2 targets; benign web clients ~16
        scores['syn_flooding'] = min(s, 1.0)

        # HTTP / UDP floods (HOIC, LOIC-HTTP, LOIC-UDP, GoldenEye, Hulk) -> Impact
        s = 0.0
        if is_http: s += 0.4
        if high_flags and fan_out <= 3: s += 0.3      # HOIC / LOIC-HTTP
        if flags < 35 and bpp >= 140: s += 0.3        # GoldenEye / Hulk
        if udp > 0.5 and in_bytes > 100_000: s += 0.6  # LOIC-UDP
        elif not is_http: s *= 0.2                     # TCP floods all target port 80
        if fan_out > 3: s *= 0.3           # DoS sources hit 1-2 targets; benign web clients ~16
        scores['volumetric_flood'] = min(s, 1.0)

        # XSS / SQL injection -> Initial Access: large packets, long flows
        s = 0.0
        if shortest > 200: s += 0.5        # 441-528 vs <= 70 for every other type
        if bpp > 300: s += 0.3             # 542 vs <= 204
        if dur > 30_000: s += 0.1
        if win < 1000: s += 0.1
        scores['fuzzing'] = min(s, 1.0)

        # Infiltration -> Lateral Movement: internal TCP scan from a compromised host
        s = 0.0
        if fan_out > 20: s += 0.3          # 28-30 vs 1-2 for external attacks
        if udp < 0.5 and 1000 <= win <= 1100: s += 0.3   # probe window 1024
        if dur < 10: s += 0.2
        if ttl is not None and ttl < 10: s += 0.2
        if udp > 0.5: s *= 0.3             # DNS-heavy clusters are benign-like
        if fan_out <= 20: s *= 0.3         # a scan from one host needs many targets
        scores['worm_spreading'] = min(s, 1.0)

        # Benign -> None: DNS, or Windows clients on non-attack ports
        s = 0.0
        if udp > 0.5 and 50 <= port <= 56: s += 0.5          # DNS
        if ttl is not None and ttl >= 100 and not is_http and not (8000 <= port <= 8100):
            s += 0.3                                          # Windows hosts, TTL 128
        if not is_http and not is_ftp_ssh and not (8000 <= port <= 8100): s += 0.2
        if is_http and fan_out > 3: s += 0.3   # web browsing: many servers per client
        if shortest > 200 or (udp > 0.5 and in_bytes > 100_000): s *= 0.2
        scores['normal_activity'] = min(s, 1.0)
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
        weights = get_behaviour_tactic_weights(self.dataset)
        for bhv, score in bhv_scores.items():
            if bhv in weights:
                for tactic, weight in weights[bhv].items():
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