"""
Behaviour → Tactic Rule-Based Mapping
========================================

Two levels:
  1. Cluster → Behaviour(s): from LLM labeling (stored in KB)
  2. Behaviour set → Tactic(s): from predefined rules (this module)

Time-window aggregation:
  - Group flows by source IP + time window
  - Collect behaviour set per window
  - Map behaviour set → tactic set
"""

import numpy as np
import pandas as pd
import logging
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# Behaviour → Tactic rules (many-to-many)
# ══════════════════════════════════════════════════════════

BEHAVIOUR_TO_TACTICS = {
    'syn_flooding': [
        {'tactic': 'Impact', 'tactic_id': 'TA0040',
         'technique_id': 'T1499', 'technique_name': 'Endpoint Denial of Service'},
    ],
    'volumetric_flooding': [
        {'tactic': 'Impact', 'tactic_id': 'TA0040',
         'technique_id': 'T1499', 'technique_name': 'Endpoint Denial of Service'},
    ],
    'port_scanning': [
        {'tactic': 'Reconnaissance', 'tactic_id': 'TA0043',
         'technique_id': 'T1046', 'technique_name': 'Network Service Scanning'},
    ],
    'service_probing': [
        {'tactic': 'Reconnaissance', 'tactic_id': 'TA0043',
         'technique_id': 'T1046', 'technique_name': 'Network Service Scanning'},
        {'tactic': 'Discovery', 'tactic_id': 'TA0007',
         'technique_id': 'T1046', 'technique_name': 'Network Service Discovery'},
    ],
    'data_transfer': [
        {'tactic': 'Exfiltration', 'tactic_id': 'TA0010',
         'technique_id': 'T1041', 'technique_name': 'Exfiltration Over Network'},
        {'tactic': 'Lateral Movement', 'tactic_id': 'TA0008',
         'technique_id': 'T1570', 'technique_name': 'Lateral Tool Transfer'},
    ],
    'periodic_communication': [
        {'tactic': 'Command and Control', 'tactic_id': 'TA0011',
         'technique_id': 'T1071', 'technique_name': 'Application Layer Protocol'},
    ],
    'normal_browsing': [
        {'tactic': 'None', 'tactic_id': 'None',
         'technique_id': 'None', 'technique_name': 'None'},
    ],
    'connection_attempts': [
        {'tactic': 'Credential Access', 'tactic_id': 'TA0006',
         'technique_id': 'T1110', 'technique_name': 'Brute Force'},
        {'tactic': 'Reconnaissance', 'tactic_id': 'TA0043',
         'technique_id': 'T1046', 'technique_name': 'Network Service Scanning'},
    ],
    # UNSW-NB15 and CIC-IDS2018 additional behaviours
    'fuzzing': [
        {'tactic': 'Initial Access', 'tactic_id': 'TA0001',
         'technique_id': 'T1190', 'technique_name': 'Exploit Public-Facing Application'},
    ],
    'exploitation': [
        {'tactic': 'Execution', 'tactic_id': 'TA0002',
         'technique_id': 'T1203', 'technique_name': 'Exploitation for Client Execution'},
        {'tactic': 'Initial Access', 'tactic_id': 'TA0001',
         'technique_id': 'T1190', 'technique_name': 'Exploit Public-Facing Application'},
    ],
    'backdoor_comm': [
        {'tactic': 'Persistence', 'tactic_id': 'TA0003',
         'technique_id': 'T1505', 'technique_name': 'Server Software Component'},
    ],
    'worm_spreading': [
        {'tactic': 'Lateral Movement', 'tactic_id': 'TA0008',
         'technique_id': 'T1570', 'technique_name': 'Lateral Tool Transfer'},
    ],
}


def get_primary_tactic(behaviour_label):
    """Get primary tactic for a single behaviour."""
    tactics = BEHAVIOUR_TO_TACTICS.get(behaviour_label, [])
    if tactics:
        return tactics[0]['tactic']
    return 'Unknown'


def map_behaviours_to_tactics(behaviour_set):
    """Map a SET of behaviours to a SET of tactics.

    Args:
        behaviour_set: set of behaviour label strings

    Returns:
        tactics: list of unique tactic dicts
        primary_tactic: str (highest-priority tactic)
    """
    all_tactics = []
    seen = set()

    # Priority order for primary tactic selection
    priority = ['Impact', 'Exfiltration', 'Command and Control',
                'Credential Access', 'Initial Access', 'Execution',
                'Lateral Movement', 'Persistence',
                'Reconnaissance', 'Discovery', 'None']

    for bhv in behaviour_set:
        for tactic_entry in BEHAVIOUR_TO_TACTICS.get(bhv, []):
            tactic_name = tactic_entry['tactic']
            if tactic_name not in seen and tactic_name != 'None':
                entry = dict(tactic_entry)
                entry['source_behaviour'] = bhv
                all_tactics.append(entry)
                seen.add(tactic_name)

    # If only normal_browsing, add None
    if not all_tactics:
        all_tactics.append({
            'tactic': 'None', 'tactic_id': 'None',
            'technique_id': 'None', 'technique_name': 'None',
            'source_behaviour': 'normal_browsing'})
        seen.add('None')

    # Primary tactic by priority
    primary = 'None'
    for p in priority:
        if p in seen:
            primary = p
            break

    return all_tactics, primary


# ══════════════════════════════════════════════════════════
# Time-window aggregation
# ══════════════════════════════════════════════════════════

class TimeWindowMapper:
    """Groups flows into time windows and maps behaviour sets to tactics.

    Windows are defined per source IP with a fixed number of flows.
    """

    def __init__(self, window_size=50, cluster_to_behaviour=None):
        """
        Args:
            window_size: number of flows per window
            cluster_to_behaviour: dict {cluster_idx: {
                'behaviours': [{'label': ..., 'confidence': ...}],
                'primary_behaviour': str,
            }}
        """
        self.window_size = window_size
        self.cluster_to_behaviour = cluster_to_behaviour or {}

    def get_flow_behaviour(self, cluster_idx):
        """Get behaviour label for a flow based on its cluster."""
        entry = self.cluster_to_behaviour.get(int(cluster_idx), {})
        return entry.get('primary_behaviour', 'normal_browsing')

    def get_flow_behaviours_all(self, cluster_idx):
        """Get all behaviours for a flow's cluster."""
        entry = self.cluster_to_behaviour.get(int(cluster_idx), {})
        bhvs = entry.get('behaviours', [])
        return [b['label'] for b in bhvs]

    def create_windows(self, src_ips, cluster_assignments, y_true=None):
        """Create time-ordered windows per source IP.

        Uses MAJORITY VOTE: the behaviour with the most flows in the
        window determines the primary tactic, not a fixed priority order.
        """
        # Group flow indices by source IP (preserving order)
        src_flows = defaultdict(list)
        for i, src in enumerate(src_ips):
            src_flows[src].append(i)

        windows = []

        for src_ip, indices in src_flows.items():
            for w_start in range(0, len(indices), self.window_size):
                w_end = min(w_start + self.window_size, len(indices))
                w_indices = indices[w_start:w_end]

                w_clusters = [int(cluster_assignments[i]) for i in w_indices]

                # Count flows per behaviour in this window
                behaviour_counts = Counter()
                behaviour_set = set()
                for cidx in w_clusters:
                    primary_bhv = self.get_flow_behaviour(cidx)
                    behaviour_counts[primary_bhv] += 1
                    for bhv in self.get_flow_behaviours_all(cidx):
                        behaviour_set.add(bhv)

                # Primary behaviour = the one with MOST flows (majority vote)
                majority_behaviour = behaviour_counts.most_common(1)[0][0]

                # Map ALL behaviours to tactics
                tactics, _ = map_behaviours_to_tactics(behaviour_set)

                # Primary tactic from MAJORITY behaviour (not fixed priority)
                _, primary_tactic = map_behaviours_to_tactics({majority_behaviour})

                window = {
                    'src_ip': src_ip,
                    'flow_indices': w_indices,
                    'n_flows': len(w_indices),
                    'clusters': w_clusters,
                    'unique_clusters': list(set(w_clusters)),
                    'behaviours': behaviour_set,
                    'behaviour_list': sorted(behaviour_set),
                    'behaviour_counts': dict(behaviour_counts),
                    'majority_behaviour': majority_behaviour,
                    'tactics': tactics,
                    'primary_tactic': primary_tactic,
                    'tactic_set': set(t['tactic'] for t in tactics),
                }

                if y_true is not None:
                    w_attacks = [y_true[i] for i in w_indices]
                    majority = Counter(w_attacks).most_common(1)[0]
                    window['true_attacks'] = w_attacks
                    window['majority_attack'] = majority[0]
                    window['attack_purity'] = majority[1] / len(w_indices)

                windows.append(window)

        return windows

    def evaluate_windows(self, windows, gt_mapper):
        """Evaluate window-level tactic predictions against ground truth.

        Args:
            windows: list from create_windows (with y_true)
            gt_mapper: GroundTruthMapper instance

        Returns:
            metrics: dict with accuracy, per-tactic metrics
            results_df: DataFrame with per-window results
        """
        rows = []
        correct = 0
        correct_top2 = 0
        total = 0

        for w in windows:
            if 'majority_attack' not in w:
                continue

            true_tactic = gt_mapper.get_tactic(w['majority_attack'])
            pred_tactic = w['primary_tactic']
            pred_set = w['tactic_set']

            is_correct = (pred_tactic == true_tactic)
            is_top2 = (true_tactic in pred_set)

            if is_correct:
                correct += 1
            if is_top2:
                correct_top2 += 1
            total += 1

            rows.append({
                'src_ip': w['src_ip'],
                'n_flows': w['n_flows'],
                'behaviours': ', '.join(w['behaviour_list']),
                'pred_tactic': pred_tactic,
                'pred_set': ', '.join(sorted(pred_set)),
                'true_attack': w['majority_attack'],
                'true_tactic': true_tactic,
                'correct': is_correct,
                'correct_top2': is_top2,
                'attack_purity': w.get('attack_purity', 0),
            })

        df = pd.DataFrame(rows)

        metrics = {
            'total_windows': total,
            'accuracy': correct / max(total, 1),
            'top2_accuracy': correct_top2 / max(total, 1),
        }

        # Per-tactic breakdown
        if len(df) > 0:
            for tactic in sorted(df['true_tactic'].unique()):
                mask = df['true_tactic'] == tactic
                t_total = mask.sum()
                t_correct = df.loc[mask, 'correct'].sum()
                t_top2 = df.loc[mask, 'correct_top2'].sum()
                metrics[f'{tactic}_accuracy'] = t_correct / max(t_total, 1)
                metrics[f'{tactic}_top2'] = t_top2 / max(t_total, 1)
                metrics[f'{tactic}_count'] = int(t_total)

        return metrics, df

# """
# Behaviour → Tactic Rule-Based Mapping
# ========================================
#
# Two levels:
#   1. Cluster → Behaviour(s): from LLM labeling (stored in KB)
#   2. Behaviour set → Tactic(s): from predefined rules (this module)
#
# Time-window aggregation:
#   - Group flows by source IP + time window
#   - Collect behaviour set per window
#   - Map behaviour set → tactic set
# """
#
# import numpy as np
# import pandas as pd
# import logging
# from collections import Counter, defaultdict
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# # ══════════════════════════════════════════════════════════
# # Behaviour → Tactic rules (many-to-many)
# # ══════════════════════════════════════════════════════════
#
# BEHAVIOUR_TO_TACTICS = {
#     'syn_flooding': [
#         {'tactic': 'Impact', 'tactic_id': 'TA0040',
#          'technique_id': 'T1499', 'technique_name': 'Endpoint Denial of Service'},
#     ],
#     'volumetric_flooding': [
#         {'tactic': 'Impact', 'tactic_id': 'TA0040',
#          'technique_id': 'T1499', 'technique_name': 'Endpoint Denial of Service'},
#     ],
#     'port_scanning': [
#         {'tactic': 'Reconnaissance', 'tactic_id': 'TA0043',
#          'technique_id': 'T1046', 'technique_name': 'Network Service Scanning'},
#     ],
#     'service_probing': [
#         {'tactic': 'Reconnaissance', 'tactic_id': 'TA0043',
#          'technique_id': 'T1046', 'technique_name': 'Network Service Scanning'},
#         {'tactic': 'Discovery', 'tactic_id': 'TA0007',
#          'technique_id': 'T1046', 'technique_name': 'Network Service Discovery'},
#     ],
#     'data_transfer': [
#         {'tactic': 'Exfiltration', 'tactic_id': 'TA0010',
#          'technique_id': 'T1041', 'technique_name': 'Exfiltration Over Network'},
#     ],
#     'periodic_communication': [
#         {'tactic': 'Command and Control', 'tactic_id': 'TA0011',
#          'technique_id': 'T1071', 'technique_name': 'Application Layer Protocol'},
#     ],
#     'normal_browsing': [
#         {'tactic': 'None', 'tactic_id': 'None',
#          'technique_id': 'None', 'technique_name': 'None'},
#     ],
#     'connection_attempts': [
#         {'tactic': 'Reconnaissance', 'tactic_id': 'TA0043',
#          'technique_id': 'T1046', 'technique_name': 'Network Service Scanning'},
#     ],
# }
#
#
# def get_primary_tactic(behaviour_label):
#     """Get primary tactic for a single behaviour."""
#     tactics = BEHAVIOUR_TO_TACTICS.get(behaviour_label, [])
#     if tactics:
#         return tactics[0]['tactic']
#     return 'Unknown'
#
#
# def map_behaviours_to_tactics(behaviour_set):
#     """Map a SET of behaviours to a SET of tactics.
#
#     Args:
#         behaviour_set: set of behaviour label strings
#
#     Returns:
#         tactics: list of unique tactic dicts
#         primary_tactic: str (highest-priority tactic)
#     """
#     all_tactics = []
#     seen = set()
#
#     # Priority order for primary tactic selection
#     priority = ['Impact', 'Exfiltration', 'Command and Control',
#                 'Reconnaissance', 'Discovery', 'Lateral Movement', 'None']
#
#     for bhv in behaviour_set:
#         for tactic_entry in BEHAVIOUR_TO_TACTICS.get(bhv, []):
#             tactic_name = tactic_entry['tactic']
#             if tactic_name not in seen and tactic_name != 'None':
#                 entry = dict(tactic_entry)
#                 entry['source_behaviour'] = bhv
#                 all_tactics.append(entry)
#                 seen.add(tactic_name)
#
#     # If only normal_browsing, add None
#     if not all_tactics:
#         all_tactics.append({
#             'tactic': 'None', 'tactic_id': 'None',
#             'technique_id': 'None', 'technique_name': 'None',
#             'source_behaviour': 'normal_browsing'})
#         seen.add('None')
#
#     # Primary tactic by priority
#     primary = 'None'
#     for p in priority:
#         if p in seen:
#             primary = p
#             break
#
#     return all_tactics, primary
#
#
# # ══════════════════════════════════════════════════════════
# # Time-window aggregation
# # ══════════════════════════════════════════════════════════
#
# class TimeWindowMapper:
#     """Groups flows into time windows and maps behaviour sets to tactics.
#
#     Windows are defined per source IP with a fixed number of flows.
#     """
#
#     def __init__(self, window_size=50, cluster_to_behaviour=None):
#         """
#         Args:
#             window_size: number of flows per window
#             cluster_to_behaviour: dict {cluster_idx: {
#                 'behaviours': [{'label': ..., 'confidence': ...}],
#                 'primary_behaviour': str,
#             }}
#         """
#         self.window_size = window_size
#         self.cluster_to_behaviour = cluster_to_behaviour or {}
#
#     def get_flow_behaviour(self, cluster_idx):
#         """Get behaviour label for a flow based on its cluster."""
#         entry = self.cluster_to_behaviour.get(int(cluster_idx), {})
#         return entry.get('primary_behaviour', 'normal_browsing')
#
#     def get_flow_behaviours_all(self, cluster_idx):
#         """Get all behaviours for a flow's cluster."""
#         entry = self.cluster_to_behaviour.get(int(cluster_idx), {})
#         bhvs = entry.get('behaviours', [])
#         return [b['label'] for b in bhvs]
#
#     def create_windows(self, src_ips, cluster_assignments, y_true=None):
#         """Create time-ordered windows per source IP.
#
#         Uses MAJORITY VOTE: the behaviour with the most flows in the
#         window determines the primary tactic, not a fixed priority order.
#         """
#         # Group flow indices by source IP (preserving order)
#         src_flows = defaultdict(list)
#         for i, src in enumerate(src_ips):
#             src_flows[src].append(i)
#
#         windows = []
#
#         for src_ip, indices in src_flows.items():
#             for w_start in range(0, len(indices), self.window_size):
#                 w_end = min(w_start + self.window_size, len(indices))
#                 w_indices = indices[w_start:w_end]
#
#                 w_clusters = [int(cluster_assignments[i]) for i in w_indices]
#
#                 # Count flows per behaviour in this window
#                 behaviour_counts = Counter()
#                 behaviour_set = set()
#                 for cidx in w_clusters:
#                     primary_bhv = self.get_flow_behaviour(cidx)
#                     behaviour_counts[primary_bhv] += 1
#                     for bhv in self.get_flow_behaviours_all(cidx):
#                         behaviour_set.add(bhv)
#
#                 # Primary behaviour = the one with MOST flows (majority vote)
#                 majority_behaviour = behaviour_counts.most_common(1)[0][0]
#
#                 # Map ALL behaviours to tactics
#                 tactics, _ = map_behaviours_to_tactics(behaviour_set)
#
#                 # Primary tactic from MAJORITY behaviour (not fixed priority)
#                 _, primary_tactic = map_behaviours_to_tactics({majority_behaviour})
#
#                 window = {
#                     'src_ip': src_ip,
#                     'flow_indices': w_indices,
#                     'n_flows': len(w_indices),
#                     'clusters': w_clusters,
#                     'unique_clusters': list(set(w_clusters)),
#                     'behaviours': behaviour_set,
#                     'behaviour_list': sorted(behaviour_set),
#                     'behaviour_counts': dict(behaviour_counts),
#                     'majority_behaviour': majority_behaviour,
#                     'tactics': tactics,
#                     'primary_tactic': primary_tactic,
#                     'tactic_set': set(t['tactic'] for t in tactics),
#                 }
#
#                 if y_true is not None:
#                     w_attacks = [y_true[i] for i in w_indices]
#                     majority = Counter(w_attacks).most_common(1)[0]
#                     window['true_attacks'] = w_attacks
#                     window['majority_attack'] = majority[0]
#                     window['attack_purity'] = majority[1] / len(w_indices)
#
#                 windows.append(window)
#
#         return windows
#
#     def evaluate_windows(self, windows, gt_mapper):
#         """Evaluate window-level tactic predictions against ground truth.
#
#         Args:
#             windows: list from create_windows (with y_true)
#             gt_mapper: GroundTruthMapper instance
#
#         Returns:
#             metrics: dict with accuracy, per-tactic metrics
#             results_df: DataFrame with per-window results
#         """
#         rows = []
#         correct = 0
#         correct_top2 = 0
#         total = 0
#
#         for w in windows:
#             if 'majority_attack' not in w:
#                 continue
#
#             true_tactic = gt_mapper.get_tactic(w['majority_attack'])
#             pred_tactic = w['primary_tactic']
#             pred_set = w['tactic_set']
#
#             is_correct = (pred_tactic == true_tactic)
#             is_top2 = (true_tactic in pred_set)
#
#             if is_correct:
#                 correct += 1
#             if is_top2:
#                 correct_top2 += 1
#             total += 1
#
#             rows.append({
#                 'src_ip': w['src_ip'],
#                 'n_flows': w['n_flows'],
#                 'behaviours': ', '.join(w['behaviour_list']),
#                 'pred_tactic': pred_tactic,
#                 'pred_set': ', '.join(sorted(pred_set)),
#                 'true_attack': w['majority_attack'],
#                 'true_tactic': true_tactic,
#                 'correct': is_correct,
#                 'correct_top2': is_top2,
#                 'attack_purity': w.get('attack_purity', 0),
#             })
#
#         df = pd.DataFrame(rows)
#
#         metrics = {
#             'total_windows': total,
#             'accuracy': correct / max(total, 1),
#             'top2_accuracy': correct_top2 / max(total, 1),
#         }
#
#         # Per-tactic breakdown
#         if len(df) > 0:
#             for tactic in sorted(df['true_tactic'].unique()):
#                 mask = df['true_tactic'] == tactic
#                 t_total = mask.sum()
#                 t_correct = df.loc[mask, 'correct'].sum()
#                 t_top2 = df.loc[mask, 'correct_top2'].sum()
#                 metrics[f'{tactic}_accuracy'] = t_correct / max(t_total, 1)
#                 metrics[f'{tactic}_top2'] = t_top2 / max(t_total, 1)
#                 metrics[f'{tactic}_count'] = int(t_total)
#
#         return metrics, df