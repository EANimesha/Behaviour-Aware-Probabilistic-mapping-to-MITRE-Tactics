"""
Explanation evaluation for NF-v3 (reviewer: self-evaluation bias, low
rule-LLM agreement, sensitivity, repeated runs, hallucination).

Five analyses on the main checkpoint (the one with LLM labels):
  1. complementarity  rule vs LLM vs fused tactic, checked against the
                      majority TRAINING-label tactic of every LLM-scored cluster
  2. hallucination    every number in an LLM reasoning text is checked against
                      the facts the LLM was given (cluster facts + reference
                      profiles); unsupported numbers are listed for audit
  3. sensitivity      repeated LLM runs x prompt variants (paraphrase, no
                      reference profiles, reasoning effort; temperature if the
                      model supports it): label stability, Fleiss' kappa
  4. rule explanation template explanation from the fired v1 rules
  5. blinded judges   non-OpenAI judges (Claude + optional Gemini) see the
                      cluster facts and the LLM and rule explanations in random
                      order (A/B); scores, preference, inter-judge agreement

The cluster facts are rebuilt exactly as training Stage 7 built them
(same training rows, summarizer, fan-out override, reference profiles).
Only aggregated cluster statistics are ever sent to an API.

Usage (repo root):
  V3_DATASET=bot_iot OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \
      python explainability_v3_extra.py [all|1|2|3|4|5 ...]
  env: N_REPEATS (5), LLM_TEMPERATURE_OK=1 (add temperature variants),
       GEMINI_API_KEY (+ GEMINI_MODEL) for a second judge,
       JUDGE_CLAUDE_MODEL (default claude-opus-4-8)
"""
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

import inference_v3 as inf
from src.clustering.summarizer import ClusterSummarizer
from src.llm.reasoning import (LLMLabeler, build_cluster_facts,
                               build_reference_profiles, LLM_MAX_COMPLETION_TOKENS)
from src.mapping.soft_cluster_scorer import (SoftClusterScorer,
                                             get_behaviour_tactic_weights)

SCORER_DATASET = "bot_iot_v3"
N_REPEATS = int(os.getenv('N_REPEATS', 3))
OUT = os.path.join(inf.CHECKPOINT_DIR, 'explainability_v3')
os.makedirs(OUT, exist_ok=True)
RNG = random.Random(7)


# ══════════════════════════════════════════════════════════
# Rebuild training clusters + the exact facts the LLM saw
# ══════════════════════════════════════════════════════════

def rebuild(device):
    joint_enc, gmm, kb, ckpt = inf.load_checkpoint(device)
    data, _ = inf.load_test_data(ckpt)          # identical preprocessing to training
    prep = ckpt['preprocessing']
    idx_train, idx_split = np.asarray(prep['idx_train']), np.asarray(prep['idx_train_split'])
    X_all, src_all, dst_all = data['X_all'], data['src_all'], data['dst_all']
    X_train, src_train, dst_train = X_all[idx_train], src_all[idx_train], dst_all[idx_train]
    y_train = data['y_all'][idx_train]
    df_raw = data['df_raw'].iloc[idx_train].reset_index(drop=True)

    # Training Stage 2 encoding (eval mode, train-period graph, causal sequences)
    builder = inf.NetworkGraphBuilder(seq_length=ckpt['config']['seq_length'])
    graph = builder.build_graph(data['G_all'][idx_split], src_all[idx_split], dst_all[idx_split])
    seqs, mask = builder.build_causal_sequences(X_all, src_all, idx_train, history_positions=idx_split)
    with torch.no_grad():
        joint_enc.context_encoder.eval()
        node_emb = joint_enc.context_encoder(graph.to(device), builder.node_features.to(device))
        ctx = builder.get_ip_embeddings(node_emb, src_train, dst_train)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    Z = torch.cat([joint_enc.get_embeddings(None, Xt[s:s + 2048], None, ctx[s:s + 2048],
                                            seqs[s:s + 2048], mask[s:s + 2048]).cpu()
                   for s in range(0, len(Xt), 2048)]).numpy()
    if 'train_clusters' in ckpt:
        clusters = np.asarray(ckpt['train_clusters'])
        print(f"  training clusters: saved ({(gmm.model.predict(Z) == clusters).mean():.1%} "
              f"re-predicted identically)")
    else:
        clusters = gmm.model.predict(Z)
        print("  training clusters: re-predicted (checkpoint has no train_clusters)")

    summarizer = ClusterSummarizer(feature_names=data['feature_names'],
                                   scaler=data['scaler'], dataset='bot_iot')
    summaries = summarizer.summarize_all_clusters(clusters, Z, X_train, src_ips=src_train,
                                                  dst_ips=dst_train, df_raw=df_raw)
    ip = pd.DataFrame({'s': src_train, 'd': dst_train})
    fan_out = ip['s'].map(ip.groupby('s')['d'].nunique()).to_numpy()
    for k in summaries:
        m = clusters == int(k)
        gp = summaries[k].setdefault('stats', {}).setdefault('graph_properties', {})
        gp['avg_fan_out_cluster'] = gp.get('avg_fan_out')
        gp['avg_fan_out'] = float(fan_out[m].mean())

    gt = inf.GroundTruthMapper(inf.DATASET)
    tactics = np.asarray(gt.map_labels(y_train))
    ref = build_reference_profiles(df_raw, tactics, fan_out=fan_out)
    weights = {int(k): float(gmm.model.weights_[int(k)]) for k in summaries}
    facts = {int(k): build_cluster_facts(summaries[k], weights[int(k)]) for k in summaries}
    truth = {}
    for k in summaries:
        m = clusters == int(k)
        c = Counter(tactics[m])
        truth[int(k)] = {'tactic': c.most_common(1)[0][0], 'purity': c.most_common(1)[0][1] / m.sum(),
                         'n_train': int(m.sum())}
    return dict(kb=kb, ckpt=ckpt, summaries={int(k): v for k, v in summaries.items()},
                weights=weights, facts=facts, ref=ref, truth=truth)


def llm_clusters(ckpt):
    """Clusters the LLM actually scored (no Stage 6 auto-benign, no mock)."""
    out = {}
    for k, e in ckpt.get('cluster_behaviour_map_hard', {}).items():
        r = str(e.get('reasoning', ''))
        if r == 'Auto-detected benign' or 'Mock' in r or 'behaviour_scores_llm' not in e:
            continue
        out[int(k)] = e
    return out


def to_tactics(bhv_scores):
    w = get_behaviour_tactic_weights(SCORER_DATASET)
    t = defaultdict(float)
    for b, s in bhv_scores.items():
        for tac, wt in w.get(b, {}).items():
            t[tac] += s * wt
    tot = sum(t.values()) or 1.0
    return {k: v / tot for k, v in t.items()}


def _top(d):
    return max(d, key=d.get) if d else 'Unknown'


# ══════════════════════════════════════════════════════════
# 1. Complementarity
# ══════════════════════════════════════════════════════════

def complementarity(R):
    print(f"\n{'=' * 70}\n1. COMPLEMENTARITY (vs majority training-label tactic)\n{'=' * 70}")
    rows = []
    for k, e in llm_clusters(R['ckpt']).items():
        kbe = R['kb'].lookup(k) or {}
        rule_t, llm_t = kbe.get('primary_tactic', 'Unknown'), e.get('primary_tactic', 'Unknown')
        fused = inf.fuse({k: {'behaviour_scores': kbe.get('behaviour_scores', {}),
                              'tactic_scores': kbe.get('tactic_scores', {})}},
                         {k: {'behaviour_scores': e['behaviour_scores_llm'],
                              'tactic_scores': e['tactic_scores_llm']}},
                         float(os.getenv('FUSION_W_RULES', 0.5)),
                         1 - float(os.getenv('FUSION_W_RULES', 0.5)))[k]['primary_tactic']
        tr = R['truth'][k]
        rows.append({'cluster': k, 'n_train': tr['n_train'], 'true_tactic': tr['tactic'],
                     'purity': tr['purity'], 'rule': rule_t, 'llm': llm_t, 'fused': fused,
                     'agree': rule_t == llm_t, 'rule_ok': rule_t == tr['tactic'],
                     'llm_ok': llm_t == tr['tactic'], 'fused_ok': fused == tr['tactic']})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'complementarity.csv'), index=False)
    print(df.to_string(index=False))
    dis = df[~df.agree]
    summary = {
        'n_llm_clusters': len(df), 'tactic_agreement': df.agree.mean(),
        'rule_acc': df.rule_ok.mean(), 'llm_acc': df.llm_ok.mean(), 'fused_acc': df.fused_ok.mean(),
        'disagree_n': len(dis), 'disagree_rule_right': int(dis.rule_ok.sum()),
        'disagree_llm_right': int(dis.llm_ok.sum()),
        'disagree_both_wrong': int((~dis.rule_ok & ~dis.llm_ok).sum()),
        'oracle_either_acc': (df.rule_ok | df.llm_ok).mean(),
    }
    for c in ('rule_ok', 'llm_ok', 'fused_ok'):     # flow-weighted
        summary[f'{c}_flow_weighted'] = np.average(df[c], weights=df.n_train)
    print(json.dumps(summary, indent=2, default=float))
    return summary


# ══════════════════════════════════════════════════════════
# 2. Hallucination (numeric claims vs facts given)
# ══════════════════════════════════════════════════════════

NUM = re.compile(r'(?<![\w.])-?\d[\d,]*\.?\d*%?')


def _nums(text):
    out = []
    for tok in NUM.findall(text or ''):
        pct = tok.endswith('%')
        try:
            v = float(tok.rstrip('%').replace(',', ''))
        except ValueError:
            continue
        out.append((tok, v, pct))
    return out


def _supported(v, pct, pool):
    cands = [v] + ([v / 100] if pct else [v / 100, v * 100])
    for c in cands:
        for p in pool:
            if abs(c - p) <= max(0.051 * abs(p), 0.011):   # 5% or rounding
                return True
    return False


def hallucination_rate(texts, R):
    """texts: list of (cluster, source, text). Ignores 0/1 and cluster ids."""
    rows = []
    for k, src, text in texts:
        pool = [v for _, v, _ in _nums(R['facts'][k] + '\n' + (R['ref'] or ''))]
        for tok, v, pct in _nums(text):
            if v in (0, 1) or v == k:
                continue
            rows.append({'cluster': k, 'source': src, 'claim': tok,
                         'supported': _supported(v, pct, pool), 'text': text})
    return pd.DataFrame(rows)


def hallucination(R, extra_texts=()):
    print(f"\n{'=' * 70}\n2. HALLUCINATION (numbers in LLM reasoning vs facts given)\n{'=' * 70}")
    texts = [(k, 'training_run', e.get('reasoning', '')) for k, e in llm_clusters(R['ckpt']).items()]
    texts += list(extra_texts)
    df = hallucination_rate(texts, R)
    df.to_csv(os.path.join(OUT, 'hallucination_claims.csv'), index=False)
    if df.empty:
        print("  no numeric claims found")
        return {}
    by = df.groupby('source').supported.agg(['count', 'mean'])
    print(by.to_string())
    print("  unsupported claims (audit these by hand):")
    print(df[~df.supported][['cluster', 'source', 'claim']].head(30).to_string(index=False))
    return {'halluc_numeric_claims': len(df),
            'halluc_unsupported_rate': 1 - df.supported.mean(),
            'halluc_unsupported_rate_training_run':
                1 - df[df.source == 'training_run'].supported.mean()}


# ══════════════════════════════════════════════════════════
# 3. Sensitivity: repeats x prompt variants
# ══════════════════════════════════════════════════════════

PARAPHRASE_HEAD = ("Rate, for EVERY behaviour listed, how well it describes the cluster of "
                   "network flows summarised below. Use 0.0 for no fit and 1.0 for a perfect "
                   "fit; the scores should add up to roughly 1.0. Use only the definitions and "
                   "the facts provided.")


def _variant_prompt(labeler, summary, weight, variant):
    prompt = labeler._build_soft_prompt(summary, weight)
    if variant == 'paraphrase':
        head, rest = prompt.split('BEHAVIOUR DEFINITIONS:', 1)
        prompt = PARAPHRASE_HEAD + '\n\nBEHAVIOUR DEFINITIONS:' + rest
        a = prompt.index('):\n', prompt.index('CLUSTER FACTS')) + 3
        b = prompt.index('\n\nNotes on the features')
        lines = prompt[a:b].split('\n')
        RNG.shuffle(lines)                                  # fact order shuffled
        prompt = prompt[:a] + '\n'.join(lines) + prompt[b:]
    return prompt


def _ask(labeler, prompt, extra):
    msgs = [{"role": "system", "content": "You are a network traffic behaviour analyst. "
             "Score how strongly each behaviour pattern matches the given cluster. "
             "Return scores as JSON."}, {"role": "user", "content": prompt}]
    for _ in range(3):
        try:
            r = labeler.client.chat.completions.create(
                model=labeler.model, messages=msgs, response_format={"type": "json_object"},
                max_completion_tokens=LLM_MAX_COMPLETION_TOKENS, **extra)
            res = labeler._parse_soft_response((r.choices[0].message.content or '').strip())
            if res:
                return res
        except Exception as e:
            print(f"    API error ({extra}): {e}")
            if 'temperature' in str(e) or 'reasoning_effort' in str(e):
                return 'unsupported'
    return None


def fleiss_kappa(table):
    """table: subjects x categories counts (equal raters per subject)."""
    table = np.asarray(table, dtype=float)
    n = table.sum(1)[0]
    p = table.sum(0) / table.sum()
    P = ((table ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (p ** 2).sum()
    return float((P.mean() - Pe) / (1 - Pe)) if Pe < 1 else 1.0


def sensitivity(R):
    print(f"\n{'=' * 70}\n3. SENSITIVITY ({N_REPEATS} repeats per variant)\n{'=' * 70}")
    from src.config.config_v3 import LLM_MODEL
    labeler = LLMLabeler(model=LLM_MODEL)
    if labeler.client is None:
        print("  OPENAI_API_KEY not set: skipped")
        return {}, []
    variants = [('original', 'original', {}), ('paraphrase', 'paraphrase', {}),
                ('no_reference', 'no_reference', {}),
                ('effort_low', 'original', {'reasoning_effort': 'low'}),
                ('effort_high', 'original', {'reasoning_effort': 'high'})]
    if os.getenv('LLM_TEMPERATURE_OK') == '1':
        variants += [('temp_0', 'original', {'temperature': 0.0}),
                     ('temp_1', 'original', {'temperature': 1.0})]
    stored = llm_clusters(R['ckpt'])
    rows, texts = [], []
    for name, prompt_kind, extra in variants:
        for k in stored:
            labeler.reference_profiles = None if prompt_kind == 'no_reference' else R['ref']
            prompt = _variant_prompt(labeler, R['summaries'][k], R['weights'][k], prompt_kind)
            for rep in range(N_REPEATS):
                res = _ask(labeler, prompt, extra)
                if res == 'unsupported':
                    print(f"  {name}: not supported by {labeler.model}, variant skipped")
                    break
                if not res:
                    continue
                bt = res['behaviour_scores']
                rows.append({'variant': name, 'cluster': k, 'repeat': rep,
                             'behaviour': _top(bt), 'tactic': _top(to_tactics(bt)),
                             'stored_behaviour': stored[k]['primary_behaviour'],
                             'stored_tactic': stored[k]['primary_tactic'],
                             'scores': json.dumps(bt)})
                texts.append((k, f'rerun_{name}', res.get('reasoning', '')))
                time.sleep(0.5)
            else:
                continue
            break
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'sensitivity_runs.csv'), index=False)
    if df.empty:
        return {}, texts
    res = []
    for name, g in df.groupby('variant', sort=False):
        stab_b = g.groupby('cluster').behaviour.agg(lambda s: s.value_counts().iloc[0] / len(s)).mean()
        stab_t = g.groupby('cluster').tactic.agg(lambda s: s.value_counts().iloc[0] / len(s)).mean()
        cats = sorted(df.behaviour.unique())
        counts = g.groupby('cluster').behaviour.value_counts().unstack(fill_value=0).reindex(
            columns=cats, fill_value=0)
        full = counts[counts.sum(1) == counts.sum(1).max()]
        res.append({'variant': name, 'runs': len(g),
                    'behaviour_stability': stab_b, 'tactic_stability': stab_t,
                    'fleiss_kappa_behaviour': fleiss_kappa(full.values) if len(full) > 1 else np.nan,
                    'agree_with_stored_behaviour': (g.behaviour == g.stored_behaviour).mean(),
                    'agree_with_stored_tactic': (g.tactic == g.stored_tactic).mean()})
    summ = pd.DataFrame(res)
    summ.to_csv(os.path.join(OUT, 'sensitivity_summary.csv'), index=False)
    print(summ.round(3).to_string(index=False))
    return {'sens_' + k: v for k, v in summ.set_index('variant').stack().items()
            if isinstance(v, float)}, texts


# ══════════════════════════════════════════════════════════
# 4. Rule-based explanation
# ══════════════════════════════════════════════════════════

def rule_explanation(R, k):
    s = R['summaries'][k]['stats']
    b, g = s.get('behaviour', {}), s.get('graph_properties', {})
    vals = {'min_ttl': b.get('avg_min_ttl'), 'tcp_flags': b.get('avg_tcp_flags'),
            'fan_out': g.get('avg_fan_out'), 'out_bytes': b.get('avg_out_bytes'),
            'pps': b.get('packets_per_second'), 'dir_ratio': b.get('throughput_ratio'),
            'duration': b.get('avg_duration_ms'), 'dst_port': b.get('avg_dst_port'),
            'iat_max': b.get('avg_iat_max'), 'iat_avg': b.get('avg_iat_avg'),
            'iat_std': b.get('avg_iat_std'), 'tcp_win_in': b.get('avg_tcp_win_in')}
    kbe = R['kb'].lookup(k) or {}
    bhv = kbe.get('primary_behaviour', 'unknown').split('#')[0]
    lines = []
    for name, feat, op, thr, w in SoftClusterScorer(dataset=SCORER_DATASET)._v3_rules():
        if name.split('#')[0] != bhv or vals.get(feat) is None:
            continue
        v = vals[feat]
        ok = (v > thr) if op == '>' else (v < thr) if op == '<' else (thr[0] <= v <= thr[1])
        cond = f"in [{thr[0]}, {thr[1]}]" if op == 'in' else f"{op} {thr}"
        lines.append(f"  - {feat} = {v:,.1f} ({'meets' if ok else 'does not meet'} {cond}; weight {w:.2f})")
    return (f"Behaviour: {bhv}. Tactic: {kbe.get('primary_tactic', 'Unknown')}.\n"
            f"Evidence (rules derived from the training split):\n" + '\n'.join(lines))


def llm_explanation(e):
    return (f"Behaviour: {e['primary_behaviour'].split('#')[0]}. Tactic: {e['primary_tactic']}.\n"
            f"Evidence: {e.get('reasoning', '').strip()}")


# ══════════════════════════════════════════════════════════
# 5. Blinded judges (no OpenAI model)
# ══════════════════════════════════════════════════════════

JUDGE = """You are a senior network security analyst. Below are the measured
facts for one cluster of network flows and two candidate explanations of what
the cluster is. Judge each explanation ONLY against the facts.

CLUSTER FACTS:
{facts}

EXPLANATION A:
{a}

EXPLANATION B:
{b}

For EACH explanation give integer scores 1-5:
  correctness   - every claim is supported by the facts (5 = all supported)
  completeness  - uses the facts that matter for the conclusion
  specificity   - concrete values rather than generic statements
  actionability - an analyst could act on it
and count unsupported_claims (claims contradicted by or absent from the facts).
Then say which explanation you prefer overall.

Respond ONLY with JSON:
{{"A": {{"correctness": n, "completeness": n, "specificity": n, "actionability": n,
        "unsupported_claims": n}},
  "B": {{"correctness": n, "completeness": n, "specificity": n, "actionability": n,
        "unsupported_claims": n}},
  "preference": "A" | "B" | "tie", "justification": "<one sentence>"}}"""

DIMS = ['correctness', 'completeness', 'specificity', 'actionability', 'unsupported_claims']


def _json(text):
    text = text.strip().strip('`')
    text = text[text.index('{'): text.rindex('}') + 1]
    return json.loads(text)


def _claude(prompt):
    import anthropic
    r = anthropic.Anthropic().messages.create(
        model=os.getenv('JUDGE_CLAUDE_MODEL', 'claude-opus-4-8'), max_tokens=800,
        messages=[{'role': 'user', 'content': prompt}])
    return _json(r.content[0].text)


import os
import requests
import json

def _gemini(prompt):
    """Call Gemini via REST API using plain requests."""

    # model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    model = "gemini-3.5-flash"

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    body = {'contents': [{'parts': [{'text': prompt}]}]}
    for attempt in range(5):  # retry slow / overloaded (503) requests
        try:
            r = requests.post(url, headers={'x-goog-api-key': os.environ['GEMINI_API_KEY']},
                              json=body, timeout=(15, int(os.getenv('GEMINI_TIMEOUT', 300))))
            if r.status_code in (429, 500, 503) and attempt < 4:
                time.sleep(15 * (attempt + 1));
                continue
            break
        except requests.exceptions.RequestException:
            if attempt == 4:
                raise
            time.sleep(15 * (attempt + 1))
    if not r.ok:  # message without the key (it is in a header, not the URL)
        raise RuntimeError(f"Gemini {r.status_code} for model {model}: {r.text[:300]}")
    parts = r.json()['candidates'][0]['content']['parts']
    return _json(''.join(p.get('text', '') for p in parts))


def judges(R):
    print(f"\n{'=' * 70}\n5. BLINDED JUDGES (LLM vs rule explanation)\n{'=' * 70}")
    js = []
    if os.getenv('ANTHROPIC_API_KEY'):
        js.append(('Claude', _claude))
    if os.getenv('GEMINI_API_KEY'):
        js.append(('Gemini', _gemini))
    if not js:
        print("  no judge keys (ANTHROPIC_API_KEY / GEMINI_API_KEY): skipped")
        return {}
    rows = []
    for k, e in llm_clusters(R['ckpt']).items():
        expl = {'llm': llm_explanation(e), 'rule': rule_explanation(R, k)}
        order = ['llm', 'rule']
        RNG.shuffle(order)                                  # blind: random A/B
        prompt = JUDGE.format(facts=R['facts'][k], a=expl[order[0]], b=expl[order[1]])
        for jname, fn in js:
            try:
                out = fn(prompt)
            except Exception as ex:
                print(f"    {jname} C{k}: error {ex}")
                continue
            pref = {'A': order[0], 'B': order[1]}.get(out.get('preference'), 'tie')
            for slot, src in zip('AB', order):
                rows.append({'cluster': k, 'judge': jname, 'source': src, 'position': slot,
                             'preferred': pref == src, 'tie': pref == 'tie',
                             **{d: out.get(slot, {}).get(d) for d in DIMS}})
            print(f"    {jname} C{k}: prefers {pref}")
            time.sleep(0.5)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'judge_scores.csv'), index=False)
    if df.empty:
        return {}
    summ = df.groupby(['judge', 'source'])[DIMS + ['preferred']].mean()
    summ.to_csv(os.path.join(OUT, 'judge_summary.csv'))
    print(summ.round(2).to_string())
    out = {f'judge_{j}_{s}_{d}': v for (j, s), r in summ.iterrows() for d, v in r.items()}
    if len(js) == 2:                                        # inter-judge agreement
        p = df.pivot_table(index=['cluster', 'source'], columns='judge', values=DIMS[:4])
        for d in DIMS[:4]:
            a, b = p[(d, js[0][0])].dropna(), p[(d, js[1][0])].dropna()
            idx = a.index.intersection(b.index)
            if len(idx) > 2:
                out[f'interjudge_kappa_{d}'] = cohen_kappa_score(
                    a[idx].astype(int), b[idx].astype(int), weights='quadratic')
                out[f'interjudge_spearman_{d}'] = spearmanr(a[idx], b[idx]).correlation
        print({k: round(v, 3) for k, v in out.items() if k.startswith('interjudge')})
    return out


def main():
    which = set(sys.argv[1:]) or {'all'}
    run = lambda n: 'all' in which or n in which
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    R = rebuild(device)
    print(f"  LLM-scored clusters: {sorted(llm_clusters(R['ckpt']))}")
    with open(os.path.join(OUT, 'llm_inputs_facts.txt'), 'w') as f:   # what was sent
        for k, v in sorted(R['facts'].items()):
            f.write(f"=== cluster {k} ===\n{v}\n\n")
        f.write(f"=== reference profiles ===\n{R['ref']}\n")
    metrics, texts = {}, []
    if run('1'):
        metrics.update(complementarity(R))
    if run('3'):
        m, texts = sensitivity(R)
        metrics.update(m)
    if run('2'):
        metrics.update(hallucination(R, texts))
    if run('4'):
        with open(os.path.join(OUT, 'rule_explanations.txt'), 'w') as f:
            for k in llm_clusters(R['ckpt']):
                f.write(f"=== cluster {k} ===\n{rule_explanation(R, k)}\n\n")
        print(f"\n  4. rule explanations -> {OUT}/rule_explanations.txt")
    if run('5'):
        metrics.update(judges(R))
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT, 'explainability_v3_metrics.csv'), index=False)
    print(f"\n  Saved: {OUT}/")


if __name__ == '__main__':
    main()
