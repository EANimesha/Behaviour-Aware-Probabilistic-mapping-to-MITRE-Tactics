# """Collect leave-one-attack-out novelty results into one table.
# Usage: python novelty_loo_summary.py checkpoints_botv3/streaming_results"""
# import glob, os, sys
# import pandas as pd
# base = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints_botv3/streaming_results'
# rows = []
# for d in sorted(glob.glob(os.path.join(base, 'novelty_*'))):
#     f = os.path.join(d, 'novelty_metrics.csv')
#     if os.path.exists(f):
#         rows.append(pd.read_csv(f).iloc[0])
# df = pd.DataFrame(rows)
# cols = ['novelty_holdout', 'novelty_auc', 'novelty_auc_vs_benign',
#         'novelty_auc_vs_known_attacks', 'novelty_detect_rate',
#         'novelty_benign_flag_rate', 'novelty_known_attack_flag_rate',
#         'novelty_flag_precision', 'gated_frac_gated', 'gated_holdout_gated_rate',
#         'gated_auc', 'gated_auc_vs_benign', 'gated_auc_vs_known_attacks',
#         'gated_detect_rate', 'gated_benign_flag_rate', 'gated_known_attack_flag_rate',
#         'gated_flag_precision']
# df = df[[c for c in cols if c in df]]
# print(df.round(3).to_string(index=False))
# df.to_csv(os.path.join(base, 'novelty_loo_summary.csv'), index=False)








# """
# Per-cluster size / behaviour / tactic / purity table from the FINAL model.
#
# Uses the saved checkpoint and knowledge base only (training split, no test data):
#   - training cluster assignments saved in the checkpoint (ckpt['train_clusters'])
#   - rule labels from the KB, LLM labels from the checkpoint, fused at FUSION_W_RULES
#   - attack-level purity (dominant attack) and tactic-level purity (dominant tactic)
#
# Usage (from the project folder):
#   FUSION_W_RULES=0.7 V3_DATASET=bot_iot python cluster_purity_table_v3.py
# Outputs: <CHECKPOINT_DIR>/paper_results/cluster_purity.csv and cluster_purity_table.tex
# """
# import os
# from collections import Counter
# import numpy as np
# import pandas as pd
# import torch
#
# import inference_v3 as inf
#
# W = float(os.getenv('FUSION_W_RULES', 0.7))
# OUT = os.path.join(inf.CHECKPOINT_DIR, 'paper_results')
# os.makedirs(OUT, exist_ok=True)
#
#
# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     _, gmm, kb, ckpt = inf.load_checkpoint(device)
#     data, _ = inf.load_test_data(ckpt)
#     idx_train = np.asarray(ckpt['preprocessing']['idx_train'])
#     y_train = np.asarray(data['y_all'][idx_train])
#     if 'train_clusters' in ckpt:
#         clusters = np.asarray(ckpt['train_clusters'])
#     else:
#         raise SystemExit("Checkpoint has no train_clusters; cannot build the training-split table.")
#     assert len(clusters) == len(y_train), "cluster/label length mismatch"
#
#     gt = inf.GroundTruthMapper(inf.DATASET)
#     tac = np.asarray(gt.map_labels(y_train))
#     rules = inf.rules_scores(kb, clusters)
#     llm = inf.llm_scores(ckpt)
#     if not llm:
#         raise SystemExit("No real LLM labels in this checkpoint (Mock or SKIP_LLM).")
#     fused = inf.fuse(rules, llm, W, 1 - W)
#
#     rows = []
#     for k in sorted(np.unique(clusters)):
#         m = clusters == k
#         a, an = Counter(y_train[m]).most_common(1)[0]
#         t, tn = Counter(tac[m]).most_common(1)[0]
#         f = fused[int(k)]
#         rows.append({'cluster': int(k), 'size': int(m.sum()),
#                      'behaviour': f['primary_behaviour'], 'tactic': f['primary_tactic'],
#                      'rule_tactic': rules[int(k)]['primary_tactic'],
#                      'llm_tactic': llm[int(k)]['primary_tactic'] if int(k) in llm else '',
#                      'dominant_attack': a, 'attack_purity': an / m.sum(),
#                      'dominant_tactic': t, 'tactic_purity': tn / m.sum(),
#                      'correct': f['primary_tactic'] == gt.get_tactic(a)})
#     df = pd.DataFrame(rows)
#     df.to_csv(os.path.join(OUT, 'cluster_purity.csv'), index=False)
#     print(df.to_string(index=False))
#     mean_p = df.attack_purity.mean(); w_p = np.average(df.attack_purity, weights=df['size'])
#     print(f"\n  attack purity: mean {mean_p:.4f}, weighted {w_p:.4f} | "
#           f"tactic purity: mean {df.tactic_purity.mean():.4f}, "
#           f"weighted {np.average(df.tactic_purity, weights=df['size']):.4f} | "
#           f"correct {df.correct.sum()}/{len(df)}")
#
#     esc = lambda s: str(s).replace('_', '\\_')
#     L = [r'\begin{tabular}{r r l l l r c}', r'\toprule',
#          r'\textbf{Cl.} & \textbf{Size} & \textbf{Behaviour} & \textbf{Tactic} & '
#          r'\textbf{Dominant attack} & \textbf{Purity} & \\', r'\midrule']
#     CK, XX = '$\\checkmark$', '$\\times$'
#     for r in rows:
#         mark = CK if r['correct'] else XX
#         L.append(f"{r['cluster']} & {r['size']:,} & {esc(r['behaviour'])} & {esc(r['tactic'])} & "
#                  f"{esc(r['dominant_attack'])} & {100 * r['attack_purity']:.0f}\\% & "
#                  f"{mark} \\\\")
#     L += [r'\midrule',
#           f"\\multicolumn{{5}}{{l}}{{\\textbf{{Mean purity}}}} & \\multicolumn{{2}}{{l}}{{\\textbf{{{100*mean_p:.2f}\\%}}}} \\\\",
#           f"\\multicolumn{{5}}{{l}}{{\\textbf{{Weighted purity}}}} & \\multicolumn{{2}}{{l}}{{\\textbf{{{100*w_p:.2f}\\%}}}} \\\\",
#           r'\bottomrule', r'\end{tabular}']
#     open(os.path.join(OUT, 'cluster_purity_table.tex'), 'w').write('\n'.join(L) + '\n')
#     print(f"\n  Saved: {OUT}/cluster_purity.csv, cluster_purity_table.tex")
#
#
# if __name__ == '__main__':
#     main()

"""
Explanation chains from the FINAL model (training split only; no test data).

Produces three outputs in <CHECKPOINT_DIR>/paper_results/:
  1. explain_examples_table.tex   : regenerated Table "Example Explanation Chains"
       (cluster size, key raw features, fused behaviour distribution, primary
        behaviour -> tactic, LLM reasoning excerpt)
  2. heuristic_semantic_matrix.csv/.tex : mean of the 5 heuristic behaviour
       targets (scan, flood, exfil, beacon, lateral) over training flows, grouped
       by the 8 semantic behaviour labels assigned to their cluster
  3. flow_trace.txt               : one original training flow traced end to end
       (raw features -> 5 heuristic scores -> cluster -> rule/LLM/fused
        behaviour distribution -> behaviour-to-tactic weights -> tactic)

Usage (project folder):
  FUSION_W_RULES=0.7 V3_DATASET=bot_iot python explain_chain_v3.py
  Optional: EXAMPLE_CLUSTERS=14,8,3,19   TRACE_CLUSTER=3
"""



# import os
# from collections import Counter
# import numpy as np
# import pandas as pd
# import torch
#
# import inference_v3 as inf
# from src.encoder.behaviour_scorer import BehaviourScorer, BEHAVIOUR_NAMES
# from src.mapping.soft_cluster_scorer import get_behaviour_tactic_weights
#
# W = float(os.getenv('FUSION_W_RULES', 0.7))
# OUT = os.path.join(inf.CHECKPOINT_DIR, 'paper_results')
# os.makedirs(OUT, exist_ok=True)
# SEMANTIC = ['syn_flooding', 'volumetric_flood', 'port_scanning', 'service_probing',
#             'data_transfer', 'periodic_comm', 'normal_activity', 'connection_attempts']
# KEY_FEATS = ['TCP_FLAGS', 'IN_BYTES', 'OUT_BYTES', 'MIN_TTL', 'FLOW_DURATION_MILLISECONDS',
#              'IN_PKTS', 'OUT_PKTS', 'L4_DST_PORT']
# SHORT = {'syn_flooding': 'syn\\_flood', 'volumetric_flood': 'vol\\_flood',
#          'port_scanning': 'port\\_scan', 'service_probing': 'svc\\_probe',
#          'data_transfer': 'data\\_transfer', 'periodic_comm': 'periodic',
#          'normal_activity': 'normal', 'connection_attempts': 'conn\\_attempts'}
#
#
# def esc(s):
#     return str(s).replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
#
#
# def top(d, n=3):
#     return sorted(((k, v) for k, v in d.items() if v > 0), key=lambda x: -x[1])[:n]
#
#
# def main():
#     _, gmm, kb, ckpt = inf.load_checkpoint('cpu')
#     data, _ = inf.load_test_data(ckpt)
#     idx = np.asarray(ckpt['preprocessing']['idx_train'])
#     if 'train_clusters' not in ckpt:
#         raise SystemExit("Checkpoint has no train_clusters.")
#     cl = np.asarray(ckpt['train_clusters'])
#     y = np.asarray(data['y_all'][idx])
#     X = data['X_all'][idx]
#     raw = data['df_raw'].iloc[idx].reset_index(drop=True)
#     src, dst = data['src_all'][idx], data['dst_all'][idx]
#     gt = inf.GroundTruthMapper(inf.DATASET)
#
#     # 5 heuristic targets, exactly as in training Stage 3
#     sc = BehaviourScorer(feature_names=data['feature_names'])
#     fo, fi, npr = sc.compute_graph_features(src, dst)
#     H = sc.compute(X, fan_out=fo, fan_in=fi, n_peers=npr, tcp_flags_raw=raw['TCP_FLAGS'].values)
#
#     rules = inf.rules_scores(kb, cl)
#     llm = inf.llm_scores(ckpt)
#     if not llm:
#         raise SystemExit("No real LLM labels in this checkpoint.")
#     fused = inf.fuse(rules, llm, W, 1 - W)
#     b2t = get_behaviour_tactic_weights('bot_iot')
#     hard = ckpt['cluster_behaviour_map_hard']
#
#     # ---- 2. heuristic -> semantic matrix ----
#     prim = np.array([fused[int(k)]['primary_behaviour'].split('#')[0] for k in cl])
#     rows = []
#     for s in SEMANTIC:
#         m = prim == s
#         r = {'semantic': s, 'n_flows': int(m.sum()),
#              'n_clusters': int(len(np.unique(cl[m])))}
#         for j, b in enumerate(BEHAVIOUR_NAMES):
#             r[b] = float(H[m, j].mean()) if m.any() else np.nan
#         rows.append(r)
#     M = pd.DataFrame(rows)
#     M.to_csv(os.path.join(OUT, 'heuristic_semantic_matrix.csv'), index=False)
#     print(M.round(3).to_string(index=False))
#     L = ['\\begin{tabular}{l r ' + 'c' * len(BEHAVIOUR_NAMES) + '}', '\\toprule',
#          'Semantic label & Flows & ' + ' & '.join(BEHAVIOUR_NAMES) + ' \\\\', '\\midrule']
#     for r in rows:
#         vals = ['--' if np.isnan(r[b]) else '%.2f' % r[b] for b in BEHAVIOUR_NAMES]
#         L.append('%s & %s & %s \\\\' % (esc(r['semantic']), format(r['n_flows'], ','), ' & '.join(vals)))
#     L += ['\\bottomrule', '\\end{tabular}']
#     open(os.path.join(OUT, 'heuristic_semantic_matrix.tex'), 'w').write('\n'.join(L) + '\n')
#
#     # ---- 1. example explanation chains ----
#     feats = [f for f in KEY_FEATS if f in raw.columns]
#     env = os.getenv('EXAMPLE_CLUSTERS', '14,1,3,19')
#     if env:
#         ex = [int(x) for x in env.split(',')]
#     else:   # largest LLM-scored cluster per fused tactic
#         best = {}
#         for k in llm:
#             t = fused[k]['primary_tactic']; n = int((cl == k).sum())
#             if t not in best or n > best[t][1]:
#                 best[t] = (k, n)
#         ex = [v[0] for v in sorted(best.values(), key=lambda x: -x[1])]
#     T = ['\\begin{tabular}{p{1.2cm} p{4.2cm} p{4cm} p{2.3cm} p{3.8cm}}', '\\toprule',
#          '\\textbf{Cluster} & \\textbf{Key Features (median)} & \\textbf{Behaviour Distribution (fused)} '
#          '& \\textbf{Primary} & \\textbf{LLM Reasoning (excerpt)} \\\\', '\\midrule']
#     for i, k in enumerate(ex):
#         m = cl == k
#         fv = ', '.join('%s=%s' % (esc(f.replace('FLOW_DURATION_MILLISECONDS', 'DUR_MS')), format(float(np.median(raw.loc[m, f])), ',.0f')) for f in feats[:5])
#         bd = ', '.join('%s: %.2f' % (SHORT.get(b, esc(b)), v) for b, v in top(fused[k]['behaviour_scores']))
#         pr = '%s $\\to$ %s' % (esc(fused[k]['primary_behaviour'].split('#')[0]), esc(fused[k]['primary_tactic']))
#         e = hard.get(k) or hard.get(str(k)) or {}
#         txt = ' '.join(str(e.get('reasoning', '')).split())
#         cut = txt.find('. ') + 1 if 0 < txt.find('. ') < 220 else 0
#         rs = esc(txt[:cut] if cut else txt[:200].rsplit(' ', 1)[0] + ' ...')
#         T.append('%d (%s) & %s & %s & %s & %s \\\\' % (k, format(int(m.sum()), ','), fv, bd, pr, rs))
#         if i < len(ex) - 1:
#             T.append('\\midrule')
#     T += ['\\bottomrule', '\\end{tabular}']
#     open(os.path.join(OUT, 'explain_examples_table.tex'), 'w').write('\n'.join(T) + '\n')
#
#     # ---- 3. one flow traced end to end ----
#     k = int(os.getenv('TRACE_CLUSTER', ex[0]))
#     m = np.where(cl == k)[0]
#     dom = Counter(y[m]).most_common(1)[0][0]
#     cand = m[y[m] == dom]
#     med = np.median(X[cand], axis=0)
#     i = cand[np.argmin(((X[cand] - med) ** 2).sum(1))]      # most typical flow of the cluster
#     lines = ['FLOW TRACE (training split, cluster %d, w_rules=%.1f)' % (k, W), '',
#              '1. Original flow (raw NetFlow fields):']
#     for f in feats:
#         lines.append('   %-28s %s' % (f, raw.loc[i, f]))
#     lines += ['   true attack: %s  (tactic %s)' % (y[i], gt.get_tactic(y[i])), '',
#               '2. Heuristic behaviour targets (encoder training signal):']
#     for j, b in enumerate(BEHAVIOUR_NAMES):
#         lines.append('   %-8s %.3f' % (b, H[i, j]))
#     lines += ['', '3. Encoder + DP-GMM -> cluster %d (%s training flows)' % (k, format(len(m), ',')), '',
#               '4. Cluster-level behaviour distribution:',
#               '   rules : ' + ', '.join('%s %.2f' % x for x in top(rules[k]['behaviour_scores'], 4)),
#               '   LLM   : ' + (', '.join('%s %.2f' % x for x in top(llm[k]['behaviour_scores'], 4)) if k in llm else 'n/a (Stage 6 benign)'),
#               '   fused : ' + ', '.join('%s %.2f' % x for x in top(fused[k]['behaviour_scores'], 4)), '',
#               '5. Behaviour-to-tactic weights used:']
#     for b, _ in top(fused[k]['behaviour_scores'], 4):
#         lines.append('   %-20s -> %s' % (b.split('#')[0], b2t.get(b.split('#')[0], {})))
#     lines += ['', '6. Tactic distribution: ' + ', '.join('%s %.2f' % x for x in top(fused[k]['tactic_scores'], 4)),
#               '   PRIMARY: %s -> %s' % (fused[k]['primary_behaviour'].split('#')[0], fused[k]['primary_tactic']),
#               '', '7. LLM reasoning: ' + ' '.join(str((hard.get(k) or hard.get(str(k)) or {}).get('reasoning', '')).split())]
#     open(os.path.join(OUT, 'flow_trace.txt'), 'w').write('\n'.join(lines) + '\n')
#     print('\n'.join(lines))
#     print('\n  Saved to', OUT)
#
#
# if __name__ == '__main__':
#     main()




"""Report actual date ranges of train / gap / test and check overlap.
Usage: python check_split_dates.py <NF-BoT-IoT-v3 sample csv> [features.json]
Uses the same Preprocessor.chronological_split as training (test 0.15, gap 0.05)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
from src.data.preprocess import Preprocessor

T = 'FLOW_START_MILLISECONDS'
df = pd.read_csv('/data/NF-BoT-IoT-v3-sample.csv').sort_values(T, kind='stable').reset_index(drop=True)
pp = Preprocessor('/home/nimesha/Downloads/PythonProject1_updated/PythonProject1/data/features_NF-BoT-IoT-v3.json')
tr, gap, te = pp.chronological_split(df, 0.15, 0.05, 'Attack')
ts = pd.to_datetime(df[T], unit='ms')
fmt = lambda s: f"{s.min()} -> {s.max()}"
rows = []
for cls in sorted(df['Attack'].unique()):
    m = (df['Attack'] == cls).values
    a, b = ts[tr][m[tr]], ts[te][m[te]]
    rows.append((cls, len(a), fmt(a), len(b), fmt(b), a.max() < b.min()))
print(pd.DataFrame(rows, columns=['class', 'n_train', 'train range', 'n_test', 'test range', 'train<test']).to_string(index=False))
a, b = ts[tr], ts[te]
print(f"\nALL train: {fmt(a)}\nALL gap  : {fmt(ts[gap])}\nALL test : {fmt(b)}")
late = (a > b.min()).sum()
print(f"Train flows later than the earliest test flow: {late:,} / {len(a):,} ({100*late/len(a):.1f}%)")
print(f"Test flows earlier than the latest train flow: {(b < a.max()).sum():,} / {len(b):,}")
print("Global (all train before all test):", a.max() < b.min())
