"""
Ground Truth Mapping: Attack Types -> MITRE ATT&CK Tactics & Techniques
=========================================================================

Maps dataset-specific attack labels to standardised MITRE ATT&CK
tactic/technique for evaluation.

Predict at technique level (actionable alerts),
evaluate at tactic level (robust metrics).
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# NF-BoT-IoT-v2 mapping
# ══════════════════════════════════════════════════════════

BOT_IOT_MAPPING = {
    'DDoS': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1499',
        'technique_name': 'Endpoint Denial of Service',
    },
    'DoS': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1499',
        'technique_name': 'Endpoint Denial of Service',
    },
    'Reconnaissance': {
        'tactic': 'Reconnaissance',
        'tactic_id': 'TA0043',
        'technique_id': 'T1046/T1016',
        'technique_name': 'Network Service Scanning / System Network Discovery',
    },
    'Theft': {
        'tactic': 'Exfiltration',
        'tactic_id': 'TA0010',
        'technique_id': 'T1041',
        'technique_name': 'Exfiltration Over Network',
    },
    'Benign': {
        'tactic': 'None',
        'tactic_id': 'None',
        'technique_id': 'None',
        'technique_name': 'None',
    },
}


# ══════════════════════════════════════════════════════════
# NF-UNSW-NB15-v2 mapping
# ══════════════════════════════════════════════════════════

UNSW_NB15_MAPPING = {
    'Reconnaissance': {
        'tactic': 'Reconnaissance',
        'tactic_id': 'TA0043',
        'technique_id': 'T1046',
        'technique_name': 'Network Service Scanning',
    },
    'Fuzzers': {
        'tactic': 'Initial Access',
        'tactic_id': 'TA0001',
        'technique_id': 'T1190',
        'technique_name': 'Exploit Public-Facing Application',
    },
    'Exploits': {
        'tactic': 'Execution',
        'tactic_id': 'TA0002',
        'technique_id': 'T1203',
        'technique_name': 'Exploitation for Client Execution',
    },
    'Generic': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1499',
        'technique_name': 'Endpoint Denial of Service',
    },
    'DoS': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1499',
        'technique_name': 'Endpoint Denial of Service',
    },
    'Analysis': {
        'tactic': 'Discovery',
        'tactic_id': 'TA0007',
        'technique_id': 'T1046',
        'technique_name': 'Network Service Scanning',
    },
    'Backdoor': {
        'tactic': 'Persistence',
        'tactic_id': 'TA0003',
        'technique_id': 'T1105',
        'technique_name': 'Ingress Tool Transfer',
    },
    'Backdoors': {
        'tactic': 'Persistence',
        'tactic_id': 'TA0003',
        'technique_id': 'T1105',
        'technique_name': 'Ingress Tool Transfer',
    },
    'Shellcode': {
        'tactic': 'Execution',
        'tactic_id': 'TA0002',
        'technique_id': 'T1059',
        'technique_name': 'Command and Scripting Interpreter',
    },
    'Worms': {
        'tactic': 'Lateral Movement',
        'tactic_id': 'TA0008',
        'technique_id': 'T1570',
        'technique_name': 'Lateral Tool Transfer',
    },
    'Benign': {
        'tactic': 'None',
        'tactic_id': 'None',
        'technique_id': 'None',
        'technique_name': 'None',
    },
}


# ══════════════════════════════════════════════════════
# NF-CSE-CIC-IDS2018-v2 mapping
# ══════════════════════════════════════════════════════

CICIDS2018_MAPPING = {
    'Benign': {
        'tactic': 'None',
        'tactic_id': '',
        'technique_id': '',
        'technique_name': 'Normal Traffic',
    },
    # DDoS attacks → Impact
    'DDoS attacks-LOIC-HTTP': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1498',
        'technique_name': 'Network Denial of Service',
    },
    'DDOS attack-LOIC-UDP': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1498',
        'technique_name': 'Network Denial of Service',
    },
    'DDOS attack-HOIC': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1498',
        'technique_name': 'Network Denial of Service',
    },
    # DoS attacks → Impact
    'DoS attacks-Hulk': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1499',
        'technique_name': 'Endpoint Denial of Service',
    },
    'DoS attacks-SlowHTTPTest': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1499',
        'technique_name': 'Endpoint Denial of Service',
    },
    'DoS attacks-Slowloris': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1499',
        'technique_name': 'Endpoint Denial of Service',
    },
    'DoS attacks-GoldenEye': {
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'technique_id': 'T1499',
        'technique_name': 'Endpoint Denial of Service',
    },
    # Bot → Command and Control
    'Bot': {
        'tactic': 'Command and Control',
        'tactic_id': 'TA0011',
        'technique_id': 'T1071',
        'technique_name': 'Application Layer Protocol',
    },
    # Infiltration → Lateral Movement
    'Infilteration': {
        'tactic': 'Lateral Movement',
        'tactic_id': 'TA0008',
        'technique_id': 'T1570',
        'technique_name': 'Lateral Tool Transfer',
    },
    # Brute Force → Credential Access
    'FTP-BruteForce': {
        'tactic': 'Credential Access',
        'tactic_id': 'TA0006',
        'technique_id': 'T1110',
        'technique_name': 'Brute Force',
    },
    'SSH-Bruteforce': {
        'tactic': 'Credential Access',
        'tactic_id': 'TA0006',
        'technique_id': 'T1110',
        'technique_name': 'Brute Force',
    },
    # Web attacks → Initial Access
    'Brute Force -Web': {
        'tactic': 'Credential Access',
        'tactic_id': 'TA0006',
        'technique_id': 'T1110',
        'technique_name': 'Brute Force - Web',
    },
    'Brute Force -XSS': {
        'tactic': 'Initial Access',
        'tactic_id': 'TA0001',
        'technique_id': 'T1189',
        'technique_name': 'Drive-by Compromise',
    },
    'SQL Injection': {
        'tactic': 'Initial Access',
        'tactic_id': 'TA0001',
        'technique_id': 'T1190',
        'technique_name': 'Exploit Public-Facing Application',
    },
}


class GroundTruthMapper:
    """Maps dataset attack labels to MITRE ATT&CK tactics for evaluation."""

    def __init__(self, dataset='bot_iot'):
        """
        Args:
            dataset: 'bot_iot', 'unsw_nb15', or 'cicids2018'
        """
        if dataset == 'bot_iot':
            self.mapping = BOT_IOT_MAPPING
        elif dataset == 'unsw_nb15':
            self.mapping = UNSW_NB15_MAPPING
        elif dataset == 'cicids2018':
            self.mapping = CICIDS2018_MAPPING
        else:
            raise ValueError(f"Unknown dataset: {dataset}. "
                             f"Use 'bot_iot', 'unsw_nb15', or 'cicids2018'.")

        self.dataset = dataset
        self.unique_tactics = sorted(set(
            v['tactic'] for v in self.mapping.values()
        ))
        logger.info(f"GroundTruthMapper({dataset}): "
                    f"{len(self.mapping)} attack types -> "
                    f"{len(self.unique_tactics)} tactics "
                    f"{self.unique_tactics}")

    def map_single(self, attack_label):
        """Map a single attack label to its full MITRE entry."""
        entry = self.mapping.get(attack_label)
        if entry is None:
            logger.warning(f"Unknown attack label: '{attack_label}'")
            return {
                'tactic': 'Unknown',
                'tactic_id': 'Unknown',
                'technique_id': 'Unknown',
                'technique_name': 'Unknown',
            }
        return entry

    def get_tactic(self, attack_label):
        """Map a single attack label to its tactic."""
        return self.map_single(attack_label)['tactic']

    def map_labels(self, attack_labels):
        """Map an array of attack labels to tactic strings.

        Args:
            attack_labels: array-like of attack type strings

        Returns:
            tactics: numpy array of tactic strings
        """
        return np.array([self.get_tactic(label) for label in attack_labels])

    def map_labels_full(self, attack_labels):
        """Map an array of attack labels to full MITRE entries.

        Returns:
            list of dicts with tactic, tactic_id, technique_id, technique_name
        """
        return [self.map_single(label) for label in attack_labels]

    def get_tactic_list(self):
        """Return sorted list of unique tactics (for confusion matrix labels)."""
        return self.unique_tactics

    def print_mapping(self):
        """Print the full mapping table."""
        print(f"\n{'='*70}")
        print(f"Ground Truth Mapping: {self.dataset}")
        print(f"{'='*70}")
        print(f"{'Attack Type':<20} {'Tactic':<20} {'Technique ID':<15} "
              f"{'Technique Name'}")
        print(f"{'-'*70}")
        for attack, entry in sorted(self.mapping.items()):
            print(f"{attack:<20} {entry['tactic']:<20} "
                  f"{entry['technique_id']:<15} {entry['technique_name']}")