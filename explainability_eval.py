"""
Explainability Evaluation via LLM-as-Judge
=============================================

Evaluates explanation quality using LLM scoring since
human experiments are not feasible.

Three evaluations:
  1. Availability: Structured explanation chain exists for each cluster
  2. Consistency: Rule-based vs LLM agreement on explanations
  3. Quality: LLM-as-judge scores on 5 dimensions

Usage:
  python explainability_eval.py bot_iot
  python explainability_eval.py cicids2018
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'

if DATASET == 'cicids2018':
    from src.config.config_cicids import CHECKPOINT_DIR, KB_PATH, OPENAI_API_KEY
elif DATASET == 'unsw_nb15':
    from src.config.config_unsw import CHECKPOINT_DIR, KB_PATH, OPENAI_API_KEY
else:
    from src.config.config import CHECKPOINT_DIR, KB_PATH, OPENAI_API_KEY

from src.knowledge_base.kb import KnowledgeBase

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "explainability_results")
os.makedirs(EVAL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# Build explanation text from KB entry
# ══════════════════════════════════════════════════════

def build_explanation_text(cluster_idx, entry):
    """Construct a structured explanation from a KB entry."""
    bhv_scores = entry.get('behaviour_scores', {})
    tactic_scores = entry.get('tactic_scores', {})
    features = entry.get('cluster_features', {})
    primary_bhv = entry.get('primary_behaviour', 'unknown')
    primary_tactic = entry.get('primary_tactic', 'Unknown')
    size = entry.get('cluster_size', 0)
    proportion = entry.get('cluster_proportion', 0)

    # Top 3 behaviours
    top_bhv = sorted(bhv_scores.items(), key=lambda x: -x[1])[:3]
    bhv_str = ', '.join(f"{b} ({s:.2f})" for b, s in top_bhv)

    # Top 3 tactics
    top_tac = sorted(tactic_scores.items(), key=lambda x: -x[1])[:3]
    tac_str = ', '.join(f"{t} ({s:.2f})" for t, s in top_tac)

    # Key features
    feat_lines = []
    for key in ['avg_tcp_flags', 'avg_out_bytes', 'avg_in_bytes',
                 'avg_min_ttl', 'avg_duration_ms', 'avg_dst_port',
                 'packets_per_second', 'bytes_per_packet']:
        if key in features:
            feat_lines.append(f"  {key}: {features[key]}")

    # Graph features
    graph = entry.get('graph_properties', {})
    for key in ['avg_fan_out', 'avg_fan_in', 'n_unique_destinations']:
        if key in graph:
            feat_lines.append(f"  {key}: {graph[key]}")

    explanation = (
        f"CLUSTER {cluster_idx} EXPLANATION\n"
        f"{'='*50}\n"
        f"Cluster size: {size} flows ({proportion*100:.1f}% of traffic)\n\n"
        f"OBSERVED FEATURES:\n"
        + '\n'.join(feat_lines) + '\n\n'
        f"BEHAVIOUR ANALYSIS:\n"
        f"  Behaviour distribution: {bhv_str}\n"
        f"  Primary behaviour: {primary_bhv} "
        f"(confidence: {bhv_scores.get(primary_bhv, 0):.2f})\n\n"
        f"TACTIC ATTRIBUTION:\n"
        f"  Tactic distribution: {tac_str}\n"
        f"  Primary tactic: {primary_tactic} "
        f"(score: {tactic_scores.get(primary_tactic, 0):.2f})\n\n"
        f"REASONING CHAIN:\n"
        f"  Features → {primary_bhv} → {primary_tactic}\n"
        f"  The cluster exhibits {primary_bhv} characteristics based on\n"
        f"  the observed feature profile, which maps to the MITRE ATT&CK\n"
        f"  tactic '{primary_tactic}'."
    )

    return explanation


# ══════════════════════════════════════════════════════
# EVAL 1: Explanation Availability
# ══════════════════════════════════════════════════════

def eval_availability(kb):
    """Check that every cluster has a complete explanation chain."""
    print(f"\n{'='*70}")
    print("EVAL 1: EXPLANATION AVAILABILITY")
    print(f"{'='*70}")

    results = []
    required_fields = [
        'behaviour_scores', 'tactic_scores', 'primary_behaviour',
        'primary_tactic', 'cluster_size'
    ]

    for k, entry in kb.entries.items():
        has_all = all(field in entry for field in required_fields)
        has_bhv_dist = len(entry.get('behaviour_scores', {})) > 0
        has_tac_dist = len(entry.get('tactic_scores', {})) > 0
        has_features = 'cluster_features' in entry or 'cluster_size' in entry
        bhv_sums_to_1 = abs(
            sum(entry.get('behaviour_scores', {}).values()) - 1.0) < 0.1

        results.append({
            'cluster': k,
            'has_all_fields': has_all,
            'has_behaviour_dist': has_bhv_dist,
            'has_tactic_dist': has_tac_dist,
            'has_features': has_features,
            'bhv_normalised': bhv_sums_to_1,
            'primary_behaviour': entry.get('primary_behaviour', 'missing'),
            'primary_tactic': entry.get('primary_tactic', 'missing'),
        })

    df = pd.DataFrame(results)
    completeness = df['has_all_fields'].mean()
    bhv_coverage = df['has_behaviour_dist'].mean()
    tac_coverage = df['has_tactic_dist'].mean()
    normalised = df['bhv_normalised'].mean()

    print(f"  Clusters in KB: {len(kb.entries)}")
    print(f"  Complete explanations: {completeness*100:.0f}%")
    print(f"  Behaviour distributions: {bhv_coverage*100:.0f}%")
    print(f"  Tactic distributions: {tac_coverage*100:.0f}%")
    print(f"  Normalised (sum≈1): {normalised*100:.0f}%")

    # Print sample explanation
    sample_k = list(kb.entries.keys())[0]
    sample_text = build_explanation_text(sample_k, kb.entries[sample_k])
    print(f"\n  Sample explanation:\n{'─'*50}")
    print(sample_text)

    return {
        'avail_completeness': completeness,
        'avail_bhv_coverage': bhv_coverage,
        'avail_tac_coverage': tac_coverage,
        'avail_normalised': normalised,
        'avail_n_clusters': len(kb.entries),
    }


# ══════════════════════════════════════════════════════
# EVAL 2: Explanation Consistency
# ══════════════════════════════════════════════════════

def eval_consistency(kb, checkpoint):
    """Compare rule-based vs LLM explanations for consistency."""
    print(f"\n{'='*70}")
    print("EVAL 2: EXPLANATION CONSISTENCY")
    print(f"{'='*70}")

    cbm_hard = checkpoint.get('cluster_behaviour_map_hard', {})
    if not cbm_hard:
        print("  No LLM labels in checkpoint — skipping")
        return {}

    agreements = []
    bhv_cosines = []
    tac_cosines = []

    for k_str, entry in cbm_hard.items():
        k = int(k_str)
        kb_entry = kb.lookup(k)
        if not kb_entry:
            continue

        # Rule-based (from KB)
        rule_bhv = kb_entry.get('primary_behaviour', '')
        rule_tactic = kb_entry.get('primary_tactic', '')
        rule_bhv_dist = kb_entry.get('behaviour_scores', {})
        rule_tac_dist = kb_entry.get('tactic_scores', {})

        # LLM-based (from checkpoint)
        llm_bhv = entry.get('primary_behaviour', '')
        llm_tactic = entry.get('primary_tactic', '')
        llm_bhv_dist = entry.get('behaviour_scores_llm', {})
        llm_tac_dist = entry.get('tactic_scores_llm', {})

        # Hard agreement
        bhv_agree = 1 if rule_bhv == llm_bhv else 0
        tac_agree = 1 if rule_tactic == llm_tactic else 0

        # Cosine similarity of distributions
        all_bhvs = set(list(rule_bhv_dist.keys()) + list(llm_bhv_dist.keys()))
        if all_bhvs:
            r_vec = np.array([rule_bhv_dist.get(b, 0) for b in all_bhvs])
            l_vec = np.array([llm_bhv_dist.get(b, 0) for b in all_bhvs])
            r_norm = np.linalg.norm(r_vec)
            l_norm = np.linalg.norm(l_vec)
            if r_norm > 0 and l_norm > 0:
                bhv_cosines.append(np.dot(r_vec, l_vec) / (r_norm * l_norm))

        all_tacs = set(list(rule_tac_dist.keys()) + list(llm_tac_dist.keys()))
        if all_tacs:
            r_vec = np.array([rule_tac_dist.get(t, 0) for t in all_tacs])
            l_vec = np.array([llm_tac_dist.get(t, 0) for t in all_tacs])
            r_norm = np.linalg.norm(r_vec)
            l_norm = np.linalg.norm(l_vec)
            if r_norm > 0 and l_norm > 0:
                tac_cosines.append(np.dot(r_vec, l_vec) / (r_norm * l_norm))

        agreements.append({
            'cluster': k,
            'rule_bhv': rule_bhv, 'llm_bhv': llm_bhv,
            'bhv_agree': bhv_agree,
            'rule_tactic': rule_tactic, 'llm_tactic': llm_tactic,
            'tac_agree': tac_agree,
        })

    if not agreements:
        print("  No comparable clusters found")
        return {}

    df = pd.DataFrame(agreements)
    bhv_agreement = df['bhv_agree'].mean()
    tac_agreement = df['tac_agree'].mean()
    mean_bhv_cos = np.mean(bhv_cosines) if bhv_cosines else 0
    mean_tac_cos = np.mean(tac_cosines) if tac_cosines else 0

    # High-confidence agreement (both methods confident)
    high_conf = [a for a in agreements
                 if a['bhv_agree'] == 1]

    print(f"  Clusters compared: {len(agreements)}")
    print(f"  Behaviour hard agreement: {bhv_agreement:.4f}")
    print(f"  Tactic hard agreement:    {tac_agreement:.4f}")
    print(f"  Behaviour cosine sim:     {mean_bhv_cos:.4f}")
    print(f"  Tactic cosine sim:        {mean_tac_cos:.4f}")
    print(f"  High-confidence agree:    {len(high_conf)}/{len(agreements)}")

    # Show disagreements
    disagree = df[df['bhv_agree'] == 0]
    if len(disagree) > 0:
        print(f"\n  Disagreements ({len(disagree)}):")
        for _, row in disagree.iterrows():
            print(f"    Cluster {row['cluster']}: "
                  f"Rule={row['rule_bhv']}→{row['rule_tactic']} | "
                  f"LLM={row['llm_bhv']}→{row['llm_tactic']}")

    return {
        'consist_bhv_agreement': bhv_agreement,
        'consist_tac_agreement': tac_agreement,
        'consist_bhv_cosine': mean_bhv_cos,
        'consist_tac_cosine': mean_tac_cos,
        'consist_n_compared': len(agreements),
    }


# ══════════════════════════════════════════════════════
# EVAL 3: LLM-as-Judge Semantic Quality
# ══════════════════════════════════════════════════════

JUDGE_PROMPT = """You are an expert cybersecurity analyst evaluating the quality of automated network intrusion detection explanations.

Below is an explanation generated by an unsupervised NIDS system that clusters network flows and maps them to MITRE ATT&CK tactics. The explanation includes observed network features, behaviour analysis, and tactic attribution.

Rate the explanation on 5 dimensions using a 1-5 scale:

1. READABILITY (1-5): Is the explanation clear and understandable for a SOC analyst?
   1=incomprehensible, 3=understandable with effort, 5=immediately clear

2. COMPLETENESS (1-5): Does it cite sufficient network features to justify the conclusion?
   1=no features cited, 3=some features, 5=all relevant features with values

3. CORRECTNESS (1-5): Does the tactic logically follow from the cited features?
   1=contradictory, 3=plausible but weak, 5=fully justified reasoning

4. ACTIONABILITY (1-5): Can an analyst make a security decision from this explanation?
   1=useless, 3=needs investigation, 5=directly actionable

5. SPECIFICITY (1-5): Does it distinguish this pattern from other attack types?
   1=generic/could be anything, 3=somewhat specific, 5=uniquely identifies the pattern

=== EXPLANATION TO EVALUATE ===

{explanation}

=== END EXPLANATION ===

Respond ONLY in JSON format:
{{
    "readability": <1-5>,
    "completeness": <1-5>,
    "correctness": <1-5>,
    "actionability": <1-5>,
    "specificity": <1-5>,
    "overall": <1-5>,
    "justification": "<1-2 sentence justification>"
}}"""


def eval_quality_llm(kb):
    """Evaluate explanation quality using LLM-as-judge."""
    print(f"\n{'='*70}")
    print("EVAL 3: LLM-AS-JUDGE QUALITY SCORING")
    print(f"{'='*70}")

    api_key = OPENAI_API_KEY or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  No OpenAI API key — skipping LLM evaluation")
        print("  Set OPENAI_API_KEY environment variable to enable")
        return {}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        print("  OpenAI package not installed — skipping")
        return {}

    # Select representative clusters (diverse tactics)
    tactics_seen = set()
    selected = []
    for k, entry in kb.entries.items():
        tactic = entry.get('primary_tactic', 'Unknown')
        if tactic not in tactics_seen and tactic != 'Unknown':
            selected.append((k, entry))
            tactics_seen.add(tactic)
        if len(selected) >= 8:
            break

    # Also add a few more for robustness
    for k, entry in kb.entries.items():
        if k not in [s[0] for s in selected]:
            selected.append((k, entry))
        if len(selected) >= 12:
            break

    print(f"  Evaluating {len(selected)} clusters with LLM judge")
    print(f"  Tactics covered: {sorted(tactics_seen)}")

    results = []
    for k, entry in selected:
        explanation = build_explanation_text(k, entry)

        prompt = JUDGE_PROMPT.replace('{explanation}', explanation)

        try:
            response = client.chat.completions.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                max_tokens=300)

            text = response.choices[0].message.content.strip()
            # Clean JSON
            text = text.replace('```json', '').replace('```', '').strip()
            scores = json.loads(text)

            results.append({
                'cluster': k,
                'tactic': entry.get('primary_tactic', '?'),
                'behaviour': entry.get('primary_behaviour', '?'),
                'readability': scores.get('readability', 0),
                'completeness': scores.get('completeness', 0),
                'correctness': scores.get('correctness', 0),
                'actionability': scores.get('actionability', 0),
                'specificity': scores.get('specificity', 0),
                'overall': scores.get('overall', 0),
                'justification': scores.get('justification', ''),
            })

            print(f"    Cluster {k} ({entry.get('primary_tactic', '?')}): "
                  f"R={scores.get('readability',0)} "
                  f"C={scores.get('completeness',0)} "
                  f"Cr={scores.get('correctness',0)} "
                  f"A={scores.get('actionability',0)} "
                  f"S={scores.get('specificity',0)} "
                  f"O={scores.get('overall',0)}")

            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"    Cluster {k}: LLM error — {e}")
            continue

    if not results:
        print("  No results obtained")
        return {}

    df = pd.DataFrame(results)

    # Aggregate scores
    dims = ['readability', 'completeness', 'correctness',
            'actionability', 'specificity', 'overall']
    print(f"\n  {'Dimension':15s} {'Mean':>6s} {'Std':>6s} {'Min':>5s} {'Max':>5s}")
    print(f"  {'─'*40}")
    metrics = {}
    for dim in dims:
        vals = df[dim].values
        print(f"  {dim:15s} {vals.mean():>6.2f} {vals.std():>6.2f} "
              f"{vals.min():>5.0f} {vals.max():>5.0f}")
        metrics[f'quality_{dim}_mean'] = float(vals.mean())
        metrics[f'quality_{dim}_std'] = float(vals.std())

    # Per-tactic scores
    print(f"\n  Per-tactic quality:")
    for tactic in sorted(df['tactic'].unique()):
        t_df = df[df['tactic'] == tactic]
        print(f"    {tactic:25s}: overall={t_df['overall'].mean():.2f} "
              f"(n={len(t_df)})")

    # Save detailed results
    df.to_csv(os.path.join(EVAL_DIR, 'llm_judge_scores.csv'), index=False)

    # Print justifications
    print(f"\n  LLM Justifications:")
    for _, row in df.iterrows():
        print(f"    Cluster {row['cluster']} ({row['tactic']}): "
              f"{row['justification']}")

    metrics['quality_n_evaluated'] = len(results)
    return metrics


# ══════════════════════════════════════════════════════
# EVAL 4: Feature-Behaviour Alignment (automated)
# ══════════════════════════════════════════════════════

# Expected feature signatures per tactic (from Fisher analysis)
EXPECTED_FEATURES = {
    'bot_iot': {
        'Impact': {'avg_tcp_flags': '<4', 'avg_out_bytes': '<10'},
        'Reconnaissance': {'avg_tcp_flags': '>15', 'avg_out_bytes': '<200'},
        'Exfiltration': {'avg_out_bytes': '>50000'},
        'None': {'avg_min_ttl': '<25'},
    },
    'cicids2018': {
        'Impact': {'avg_duration_ms': '>2000000'},
        'Command and Control': {'avg_tcp_flags': '>200'},
        'Credential Access': {'avg_dst_port': '<25'},
        'None': {'avg_duration_ms': '<500000'},
    },
}


def eval_feature_alignment(kb):
    """Check if cited features align with expected attack signatures."""
    print(f"\n{'=' * 70}")
    print("EVAL 4: FEATURE-BEHAVIOUR ALIGNMENT")
    print(f"{'=' * 70}")

    expected = EXPECTED_FEATURES.get(DATASET, {})
    if not expected:
        print(f"  No expected features defined for {DATASET}")
        return {}

    # Track results per tactic
    tactic_stats = {}
    aligned = 0
    total = 0

    for k, entry in kb.entries.items():
        tactic = entry.get('primary_tactic', 'Unknown')
        if tactic not in expected:
            continue

        features = entry.get('cluster_features', {})
        if not features:
            continue

        checks = expected[tactic]
        cluster_aligned = True
        failed_checks = []

        # ─────────────────────────────────────────────────────────
        # Detailed feature check
        # ─────────────────────────────────────────────────────────
        cited_features_str = []
        for feat, condition in checks.items():
            val = features.get(feat, None)
            if val is None:
                continue

            cited_features_str.append(f"{feat}={val:.2f}" if isinstance(val, float) else f"{feat}={val}")

            # Check condition
            if condition.startswith('<'):
                threshold = float(condition[1:])
                if val >= threshold:
                    cluster_aligned = False
                    failed_checks.append(f"{feat}={val} NOT< {threshold}")
            elif condition.startswith('>'):
                threshold = float(condition[1:])
                if val <= threshold:
                    cluster_aligned = False
                    failed_checks.append(f"{feat}={val} NOT> {threshold}")

        total += 1
        if cluster_aligned:
            aligned += 1
            status = '✓'
        else:
            status = '✗'

        # Track per-tactic stats
        if tactic not in tactic_stats:
            tactic_stats[tactic] = {
                'total': 0,
                'aligned': 0,
                'expected_features': ', '.join([f"{k} {v}" for k, v in checks.items()]),
                'clusters': []
            }

        tactic_stats[tactic]['total'] += 1
        if cluster_aligned:
            tactic_stats[tactic]['aligned'] += 1

        tactic_stats[tactic]['clusters'].append({
            'id': k,
            'cited': ', '.join(cited_features_str),
            'status': status,
            'failed': failed_checks
        })

        # Print per-cluster details
        print(f"\n  Cluster {k}: {tactic:25s} → {status}")
        print(f"    Expected: {tactic_stats[tactic]['expected_features']}")
        print(f"    Cited:    {', '.join(cited_features_str) if cited_features_str else 'N/A'}")
        if failed_checks:
            print(f"    Failed:   {'; '.join(failed_checks)}")

    # ═════════════════════════════════════════════════════════════════════════
    # Summary per tactic (FOR TABLE FILLING)
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TACTIC SUMMARY (for Table)")
    print(f"{'=' * 70}")

    for tactic in sorted(tactic_stats.keys()):
        stats = tactic_stats[tactic]
        rate = stats['aligned'] / max(stats['total'], 1) * 100
        print(f"\n{tactic}:")
        print(f"  Expected Features: {stats['expected_features']}")
        print(f"  Clusters Evaluated: {stats['total']}")
        print(f"  Clusters Aligned: {stats['aligned']}/{stats['total']} ({rate:.0f}%)")

        # Show example cluster
        if stats['clusters']:
            example = stats['clusters'][0]
            print(f"  Example (Cluster {example['id']}): {example['cited']}")

    # ═════════════════════════════════════════════════════════════════════════
    # Overall stats
    # ═════════════════════════════════════════════════════════════════════════
    alignment_rate = aligned / max(total, 1)
    print(f"\n{'=' * 70}")
    print(f"Overall Alignment: {aligned}/{total} = {alignment_rate:.4f} ({alignment_rate * 100:.1f}%)")
    print(f"{'=' * 70}\n")

    return {
        'align_rate': alignment_rate,
        'align_correct': aligned,
        'align_total': total,
        'tactic_stats': tactic_stats,  # For programmatic access
    }

# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    import torch

    print(f"\n{'='*70}")
    print(f"EXPLAINABILITY EVALUATION — {DATASET}")
    print(f"{'='*70}")

    kb = KnowledgeBase()
    kb.load('/home/nimesha/Downloads/PythonProject1_updated/PythonProject1/checkpoints/knowledge_base.json')
    print(f"  KB loaded: {len(kb.entries)} clusters")

    # Load checkpoint for consistency eval
    model_path = '/home/nimesha/Downloads/PythonProject1_updated/PythonProject1/checkpoints/model_checkpoint.pt'
    checkpoint = {}
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu',
                                weights_only=False)

    all_metrics = {}

    # Eval 1: Availability
    m1 = eval_availability(kb)
    all_metrics.update(m1)

    # Eval 2: Consistency
    m2 = eval_consistency(kb, checkpoint)
    all_metrics.update(m2)

    # Eval 3: LLM-as-Judge Quality
    m3 = eval_quality_llm(kb)
    all_metrics.update(m3)

    # Eval 4: Feature Alignment
    m4 = eval_feature_alignment(kb)
    all_metrics.update(m4)

    # Summary
    print(f"\n{'='*70}")
    print("EXPLAINABILITY SUMMARY")
    print(f"{'='*70}")

    print(f"\n  AVAILABILITY:")
    print(f"    Completeness: {all_metrics.get('avail_completeness', 0)*100:.0f}%")

    print(f"\n  CONSISTENCY (Rule vs LLM):")
    print(f"    Behaviour agreement: {all_metrics.get('consist_bhv_agreement', 0):.4f}")
    print(f"    Tactic agreement:    {all_metrics.get('consist_tac_agreement', 0):.4f}")

    print(f"\n  LLM-AS-JUDGE QUALITY:")
    for dim in ['readability', 'completeness', 'correctness',
                'actionability', 'specificity', 'overall']:
        val = all_metrics.get(f'quality_{dim}_mean', 0)
        print(f"    {dim:15s}: {val:.2f}/5")

    print(f"\n  FEATURE ALIGNMENT:")
    print(f"    Rate: {all_metrics.get('align_rate', 0):.4f}")

    # Save
    df = pd.DataFrame([all_metrics])
    df.to_csv(os.path.join(EVAL_DIR, 'explainability_metrics.csv'), index=False)
    print(f"\n  Saved: {EVAL_DIR}/")


if __name__ == '__main__':
    main()