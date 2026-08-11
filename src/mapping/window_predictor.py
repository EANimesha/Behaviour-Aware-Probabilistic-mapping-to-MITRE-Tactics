"""
Window-Level Tactic Predictor (Multi-Dataset)
===============================================

Supports both NF-BoT-IoT-v2 and NF-UNSW-NB15-v2 datasets.

Behaviours:
  syn_flooding        → Impact
  volumetric_flood    → Impact
  port_scanning       → Reconnaissance
  service_probing     → Reconnaissance, Discovery
  data_transfer       → Exfiltration
  normal_activity     → None
  connection_attempts → Reconnaissance, Initial Access
  periodic_comm       → Command and Control
  fuzzing             → Initial Access          (UNSW)
  exploitation        → Execution               (UNSW)
  backdoor_comm       → Persistence             (UNSW)
  worm_spreading      → Lateral Movement        (UNSW)
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BEHAVIOUR_TACTIC_WEIGHTS = {
    'syn_flooding':        {'Impact': 1.0},
    'volumetric_flood':    {'Impact': 1.0},
    'port_scanning':       {'Reconnaissance': 1.0},
    'service_probing':     {'Reconnaissance': 0.6, 'Discovery': 0.4},
    'data_transfer':       {'Exfiltration': 1.0},
    'normal_activity':     {'None': 1.0},
    'connection_attempts': {'Reconnaissance': 0.5, 'Initial Access': 0.5},
    'periodic_comm':       {'Command and Control': 1.0},
    'fuzzing':             {'Initial Access': 1.0},
    'exploitation':        {'Execution': 1.0},
    'backdoor_comm':       {'Persistence': 1.0},
    'worm_spreading':      {'Lateral Movement': 1.0},
}


class WindowPredictor:
    """Predicts tactics from time-window features using soft scoring."""

    def __init__(self, window_size=20, tactic_threshold=0.15):
        self.window_size = window_size
        self.tactic_threshold = tactic_threshold

    def create_windows(self, src_ips, flow_features, y_true=None):
        N = len(src_ips)
        src_flows = defaultdict(list)
        for i in range(N):
            src_flows[src_ips[i]].append(i)

        windows = []
        for src_ip, indices in src_flows.items():
            for w_start in range(0, len(indices), self.window_size):
                w_end = min(w_start + self.window_size, len(indices))
                w_idx = indices[w_start:w_end]

                wf = self._compute_window_features(w_idx, flow_features)
                bhv_scores = self._score_behaviours(wf)
                tactic_scores = self._map_to_tactics(bhv_scores)

                primary = max(tactic_scores, key=tactic_scores.get)
                final = sorted(
                    [t for t, s in tactic_scores.items()
                     if s >= self.tactic_threshold],
                    key=lambda t: tactic_scores[t], reverse=True)

                window = {
                    'src_ip': src_ip,
                    'flow_indices': w_idx,
                    'n_flows': len(w_idx),
                    'window_features': wf,
                    'behaviour_scores': bhv_scores,
                    'tactic_scores': tactic_scores,
                    'primary_tactic': primary,
                    'final_tactics': final,
                }

                if y_true is not None:
                    w_attacks = [y_true[i] for i in w_idx]
                    majority = Counter(w_attacks).most_common(1)[0]
                    window['majority_attack'] = majority[0]
                    window['attack_purity'] = majority[1] / len(w_idx)
                    window['true_attacks'] = w_attacks

                windows.append(window)

        return windows

    def _compute_window_features(self, indices, ff):
        idx = indices
        n = len(idx)

        def get(name):
            if name in ff:
                return np.array([ff[name][i] for i in idx], dtype=np.float64)
            return np.zeros(n)

        tcp = get('TCP_FLAGS')
        out_bytes = get('OUT_BYTES')
        in_bytes = get('IN_BYTES')
        in_pkts = get('IN_PKTS')
        out_pkts = get('OUT_PKTS')
        min_ttl = get('MIN_TTL')
        dst_port = get('L4_DST_PORT')
        src_port = get('L4_SRC_PORT')
        duration = get('FLOW_DURATION_MILLISECONDS')
        protocol = get('PROTOCOL')

        total_pkts = in_pkts + out_pkts
        total_bytes = in_bytes + out_bytes

        if 'DST_IP' in ff:
            fan_out = len(set(ff['DST_IP'][i] for i in idx))
        else:
            fan_out = min(n, 10)

        if 'SRC_IP' in ff:
            fan_in_count = len(set(ff['SRC_IP'][i] for i in idx))
        else:
            fan_in_count = 1

        return {
            'n_flows': n,
            'fan_out': fan_out,
            'fan_in': fan_in_count,
            'tcp_flags_mean': float(tcp.mean()),
            'syn_only_fraction': float((tcp < 4).sum() / max(n, 1)),
            'normal_tcp_fraction': float((tcp > 15).sum() / max(n, 1)),
            'total_out_bytes': float(out_bytes.sum()),
            'mean_out_bytes': float(out_bytes.mean()),
            'total_in_bytes': float(in_bytes.sum()),
            'mean_in_bytes': float(in_bytes.mean()),
            'zero_payload_fraction': float((out_bytes < 10).sum() / max(n, 1)),
            'mean_total_pkts': float(total_pkts.mean()),
            'total_pkts': float(total_pkts.sum()),
            'mean_bytes_per_pkt': float(
                total_bytes.sum() / max(total_pkts.sum(), 1)),
            'mean_min_ttl': float(min_ttl.mean()),
            'mean_dst_port': float(dst_port.mean()),
            'mean_src_port': float(src_port.mean()),
            'low_port_fraction': float((dst_port < 1024).sum() / max(n, 1)),
            'high_port_fraction': float((dst_port > 10000).sum() / max(n, 1)),
            'mean_duration_ms': float(duration.mean()),
            'direction_ratio': float(
                out_bytes.sum() / max(in_bytes.sum(), 0.01)),
            'tcp_fraction': float((protocol == 6).sum() / max(n, 1)),
            'udp_fraction': float((protocol == 17).sum() / max(n, 1)),
            'icmp_fraction': float((protocol == 1).sum() / max(n, 1)),
            # Diversity metrics (useful for UNSW attack types)
            'port_diversity': float(len(set(dst_port)) / max(n, 1)),
            'protocol_diversity': float(len(set(protocol)) / max(n, 1)),
            'pkt_size_std': float(np.std(total_bytes) if n > 1 else 0),
        }

    def _score_behaviours(self, wf):
        """Score each behaviour with mutual exclusion logic."""
        scores = {}
        is_syn_dominant = wf['syn_only_fraction'] > 0.3
        is_normal_tcp = wf['normal_tcp_fraction'] > 0.5

        # ── SYN flooding ──
        s = 0.0
        s += wf['syn_only_fraction'] * 0.6
        s += wf['zero_payload_fraction'] * 0.15
        s += wf['low_port_fraction'] * 0.1
        if wf['mean_min_ttl'] > 50: s += 0.15
        scores['syn_flooding'] = min(s, 1.0)

        # ── Volumetric flooding ──
        s = 0.0
        if wf['mean_total_pkts'] > 100: s += 0.3
        if is_normal_tcp and wf['mean_min_ttl'] > 50: s += 0.3
        if wf['direction_ratio'] > 10: s += 0.2
        if wf['low_port_fraction'] > 0.5: s += 0.2
        scores['volumetric_flood'] = min(s, 1.0)

        # ── Port scanning (suppressed when SYN-dominant) ──
        s = 0.0
        if not is_syn_dominant:
            if wf['fan_out'] > 5: s += 0.3
            elif wf['fan_out'] > 2: s += 0.15
            if wf['mean_out_bytes'] < 200 and is_normal_tcp: s += 0.25
            if 20 < wf['mean_min_ttl'] < 50: s += 0.2
            if wf['n_flows'] > 10: s += 0.1
            if wf['high_port_fraction'] > 0.3: s += 0.15
        scores['port_scanning'] = min(s, 1.0)

        # ── Service probing ──
        s = 0.0
        if not is_syn_dominant:
            if is_normal_tcp and wf['mean_out_bytes'] < 5000: s += 0.35
            if 20 < wf['mean_min_ttl'] < 50: s += 0.25
            if 2 < wf['fan_out'] < 10: s += 0.2
            if wf['port_diversity'] > 0.3: s += 0.1
        scores['service_probing'] = min(s, 1.0)

        # ── Data transfer ──
        s = 0.0
        if wf['total_out_bytes'] > 500000: s += 0.45
        elif wf['total_out_bytes'] > 100000: s += 0.3
        if wf['mean_bytes_per_pkt'] > 200: s += 0.2
        if wf['direction_ratio'] > 3 and wf['mean_min_ttl'] > 30: s += 0.2
        if is_normal_tcp: s += 0.1
        if is_syn_dominant: s *= 0.1
        scores['data_transfer'] = min(s, 1.0)

        # ── Normal activity ──
        s = 0.0
        if wf['mean_min_ttl'] < 25: s += 0.35
        if wf['high_port_fraction'] > 0.5: s += 0.2
        if is_normal_tcp and wf['syn_only_fraction'] < 0.1: s += 0.15
        if 100 < wf['mean_out_bytes'] < 100000: s += 0.15
        if 0.3 < wf['direction_ratio'] < 3: s += 0.15
        if is_syn_dominant: s *= 0.2
        scores['normal_activity'] = min(s, 1.0)

        # ── Connection attempts ──
        s = 0.0
        if wf['syn_only_fraction'] > 0.3 and wf['fan_out'] <= 3: s += 0.5
        if wf['zero_payload_fraction'] > 0.5: s += 0.2
        if wf['n_flows'] < 10: s += 0.15
        scores['connection_attempts'] = min(s, 1.0)

        # ── Periodic communication (C2) ──
        s = 0.0
        if wf['mean_duration_ms'] > 10000: s += 0.3
        if 0.3 < wf['direction_ratio'] < 3: s += 0.2
        if wf['fan_out'] <= 2: s += 0.25
        if is_normal_tcp: s += 0.15
        if wf['mean_out_bytes'] < 1000: s += 0.1
        if is_syn_dominant: s *= 0.2
        scores['periodic_comm'] = min(s, 1.0)

        # ── Fuzzing (UNSW: Initial Access) ──
        # High port diversity + many small varied payloads + targeting services
        s = 0.0
        if not is_syn_dominant:
            if wf['port_diversity'] > 0.5: s += 0.25
            if wf['pkt_size_std'] > 100: s += 0.2  # varied packet sizes
            if wf['low_port_fraction'] > 0.3: s += 0.2
            if is_normal_tcp and wf['mean_out_bytes'] > 50: s += 0.15
            if wf['fan_out'] <= 3: s += 0.1  # focused on few targets
        scores['fuzzing'] = min(s, 1.0)

        # ── Exploitation (UNSW: Execution) ──
        # Moderate payload to specific services + unusual flags
        s = 0.0
        if not is_syn_dominant:
            if 200 < wf['mean_out_bytes'] < 50000: s += 0.25
            if wf['low_port_fraction'] > 0.5: s += 0.2
            if wf['fan_out'] <= 3: s += 0.2  # targeted
            if is_normal_tcp: s += 0.15
            if wf['mean_duration_ms'] < 5000: s += 0.1
        scores['exploitation'] = min(s, 1.0)

        # ── Backdoor communication (UNSW: Persistence) ──
        # Long sessions, specific ports, bidirectional, few peers
        s = 0.0
        if not is_syn_dominant:
            if wf['mean_duration_ms'] > 5000: s += 0.25
            if wf['fan_out'] <= 2: s += 0.25
            if 0.3 < wf['direction_ratio'] < 5: s += 0.2
            if is_normal_tcp: s += 0.15
            if wf['high_port_fraction'] > 0.3: s += 0.1  # non-standard ports
        scores['backdoor_comm'] = min(s, 1.0)

        # ── Worm spreading (UNSW: Lateral Movement) ──
        # High fan-out, spreading across hosts, moderate payloads
        s = 0.0
        if not is_syn_dominant:
            if wf['fan_out'] > 5: s += 0.3
            if wf['mean_out_bytes'] > 200: s += 0.15
            if is_normal_tcp: s += 0.15
            if wf['protocol_diversity'] > 0.2: s += 0.1
            if wf['n_flows'] > 10: s += 0.15
            if wf['low_port_fraction'] > 0.3: s += 0.1
        scores['worm_spreading'] = min(s, 1.0)

        return scores

    def _map_to_tactics(self, bhv_scores):
        tactic_scores = defaultdict(float)
        for bhv, bhv_score in bhv_scores.items():
            if bhv in BEHAVIOUR_TACTIC_WEIGHTS:
                for tactic, weight in BEHAVIOUR_TACTIC_WEIGHTS[bhv].items():
                    tactic_scores[tactic] += bhv_score * weight

        max_score = max(tactic_scores.values()) if tactic_scores else 1.0
        if max_score > 0:
            tactic_scores = {t: round(s / max_score, 4)
                             for t, s in tactic_scores.items()}

        for t in ['Impact', 'Reconnaissance', 'Exfiltration', 'None',
                  'Command and Control', 'Discovery', 'Initial Access',
                  'Execution', 'Persistence', 'Lateral Movement']:
            if t not in tactic_scores:
                tactic_scores[t] = 0.0

        return dict(tactic_scores)

    def evaluate(self, windows, gt_mapper, save_dir=None):
        rows = []
        correct, correct_top2, total = 0, 0, 0

        for w in windows:
            if 'majority_attack' not in w:
                continue
            true_tactic = gt_mapper.get_tactic(w['majority_attack'])
            pred = w['primary_tactic']
            final_set = set(w['final_tactics'])

            is_correct = pred == true_tactic
            is_top2 = true_tactic in final_set
            if is_correct: correct += 1
            if is_top2: correct_top2 += 1
            total += 1

            ts = w['tactic_scores']
            score_str = ', '.join(f"{t}:{s:.2f}" for t, s
                                  in sorted(ts.items(), key=lambda x: -x[1])
                                  if s > 0.01)
            rows.append({
                'src_ip': w['src_ip'], 'n_flows': w['n_flows'],
                'pred_tactic': pred,
                'final_tactics': ', '.join(w['final_tactics']),
                'tactic_scores': score_str,
                'true_attack': w['majority_attack'],
                'true_tactic': true_tactic,
                'correct': is_correct, 'correct_top2': is_top2,
                'purity': w.get('attack_purity', 0),
            })

        df = pd.DataFrame(rows)
        metrics = {
            'total_windows': total,
            'accuracy': correct / max(total, 1),
            'top2_accuracy': correct_top2 / max(total, 1),
        }

        if len(df) > 0:
            for tactic in sorted(df['true_tactic'].unique()):
                mask = df['true_tactic'] == tactic
                t_total = mask.sum()
                metrics[f'{tactic}_accuracy'] = (
                    df.loc[mask, 'correct'].sum() / max(t_total, 1))
                metrics[f'{tactic}_top2'] = (
                    df.loc[mask, 'correct_top2'].sum() / max(t_total, 1))
                metrics[f'{tactic}_count'] = int(t_total)

        print(f"\n  Window accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Window top-2 acc:  {metrics['top2_accuracy']:.4f}")
        print(f"\n  {'Tactic':>20s} {'Count':>7s} {'Acc':>8s} {'Top-2':>8s}")
        print(f"  {'-'*45}")
        for tactic in sorted(df['true_tactic'].unique()):
            c = metrics.get(f'{tactic}_count', 0)
            a = metrics.get(f'{tactic}_accuracy', 0)
            t = metrics.get(f'{tactic}_top2', 0)
            if c > 0:
                print(f"  {tactic:>20s} {c:>7d} {a:>8.4f} {t:>8.4f}")

        wrong = df[~df['correct']]
        if len(wrong) > 0:
            print(f"\n  Errors ({len(wrong)} windows):")
            conf = wrong.groupby(['true_tactic', 'pred_tactic']).size()
            for (tt, pt), cnt in conf.sort_values(ascending=False).head(10).items():
                print(f"    {tt:>15s} -> {pt:<15s} ({cnt:>4d})")

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            df.to_csv(os.path.join(save_dir, 'window_predictions.csv'), index=False)

        return metrics, df