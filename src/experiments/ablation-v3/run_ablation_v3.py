"""
Encoder ablation on the v3 chronological split.

Every variant goes through the same frozen pipeline: DP-GMM (same K cap),
frozen v1 rules, Stage 6 benign detection, causal test inference.
LLM is on by default (as in the final model); set SKIP_LLM=1 to turn it off.
Main metric: per-flow tactic macro-F1 on the held-out test split.

Two studies:
  --study encoder  (default) encoder tracks / behaviour loss / raw features
  --study cluster  K-Means / GMM / DP-GMM on the full encoder

Usage:
  V3_DATASET=bot_iot python run_ablation_v3.py
  V3_DATASET=bot_iot python run_ablation_v3.py --study cluster
  python run_ablation_v3.py --variants full raw --seeds 42   # subset
  python run_ablation_v3.py --collect-only                   # just rebuild table
"""
import argparse, os, subprocess, sys
import pandas as pd

VARIANTS = ['full', 'feature', 'feature_context', 'feature_temporal',
            'no_bhv_loss', 'raw']
NAMES = {'full': 'Full (F+C+T, $\\lambda$=1)', 'feature': 'Feature only',
         'feature_context': 'Feature + Context', 'feature_temporal': 'Feature + Temporal',
         'no_bhv_loss': 'Full, no behaviour loss ($\\lambda$=0)',
         'raw': 'Raw features + DP-GMM'}
# name -> (CLUSTER_METHOD, CLUSTER_K); 'mnt' = m x number of tactics
CLUSTER_VARIANTS = {
    'kmeans_1nt': ('kmeans', '1nt'), 'kmeans_2nt': ('kmeans', '2nt'),
    'kmeans_5nt': ('kmeans', '5nt'), 'gmm_1nt': ('gmm', '1nt'),
    'gmm_2nt': ('gmm', '2nt'), 'gmm_5nt': ('gmm', '5nt'),
    'gmm_20': ('gmm', '20'), 'dpgmm_20': ('dpgmm', '20'), 'dpgmm_30': ('dpgmm', '30'),
}
NAMES.update({
    'kmeans_1nt': 'K-Means ($K=n_t$)', 'kmeans_2nt': 'K-Means ($K=2n_t$)',
    'kmeans_5nt': 'K-Means ($K=5n_t$)', 'gmm_1nt': 'GMM ($K=n_t$)',
    'gmm_2nt': 'GMM ($K=2n_t$)', 'gmm_5nt': 'GMM ($K=5n_t$)', 'gmm_20': 'GMM ($K=20$)',
    'dpgmm_20': 'DP-GMM ($K_{max}=20$)', 'dpgmm_30': 'DP-GMM ($K_{max}=30$)'})
METRICS = ['perflow_rules_macro_f1', 'perflow_rules_accuracy',
           'window50_rules_top1_accuracy', 'mean_purity',
           'weighted_purity', 'nmi_tactic', 'n_clusters']

ap = argparse.ArgumentParser()
ap.add_argument('--study', choices=['encoder', 'cluster'], default='encoder')
ap.add_argument('--variants', nargs='+', default=None)
ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
ap.add_argument('--collect-only', action='store_true')
args = ap.parse_args()

dataset = os.getenv('V3_DATASET', 'bot_iot')
if args.study == 'cluster':
    VARIANTS = list(CLUSTER_VARIANTS)
    root = f'checkpoints_ablation_cluster_{dataset}'
else:
    root = f'checkpoints_ablation_{dataset}'
args.variants = args.variants or VARIANTS
root = '/home/nimesha/Downloads/PythonProject1_updated/PythonProject1/checkpoints_ablation_cluster_bot_iot'


def run_dir(v, s):
    return os.path.join(root, f'{v}_s{s}')



if not args.collect_only:
    for v in args.variants:
        for s in args.seeds:
            d = run_dir(v, s)
            out = os.path.join(d, 'inference_results', 'inference_metrics.csv')
            if os.path.exists(out):
                print(f'skip {v} seed {s} (done)'); continue
            env = dict(os.environ, V3_DATASET=dataset, SEED=str(s),
                       SKIP_LLM=os.getenv('SKIP_LLM', '0'), CHECKPOINT_DIR=d)
            if args.study == 'cluster':
                env.update(ABLATION='full', CLUSTER_METHOD=CLUSTER_VARIANTS[v][0],
                           CLUSTER_K=CLUSTER_VARIANTS[v][1])
            else:
                env.update(ABLATION=v)
            os.makedirs(d, exist_ok=True)
            for script in ('training_v3.py', 'inference_v3.py'):
                print(f'\n### {v} seed {s}: {script}')
                with open(os.path.join(d, script.replace('.py', '.log')), 'w') as log:
                    r = subprocess.run([sys.executable, script], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
                if r.returncode:
                    print(f'  FAILED, see {d}/{script[:-3]}.log'); break

rows = []
for v in args.variants:
    for s in args.seeds:
        f = os.path.join(run_dir(v, s), 'inference_results', 'inference_metrics.csv')
        if os.path.exists(f):
            m = pd.read_csv(f).iloc[0]
            rows.append({'variant': v, 'seed': s,
                         **{k: m[k] for k in METRICS if k in m}})
if not rows:
    sys.exit('no results yet')
raw = pd.DataFrame(rows)
raw.to_csv(os.path.join(root, 'ablation_runs.csv'), index=False)
cols = [k for k in METRICS if k in raw]
agg = raw.groupby('variant')[cols].agg(['mean', 'std'])
agg = agg.reindex([v for v in VARIANTS if v in agg.index])
agg.to_csv(os.path.join(root, 'ablation_summary.csv'))
print('\n', agg.round(4))

hdr = ['Tactic macro-F1', 'Per-flow acc.', 'W50 Top-1', 'Mean purity', 'Weighted purity',
       'NMI (tactic)', 'Active $K$']
lines = ['\\begin{table}[t]', '\\centering',
         f'\\caption{{{args.study.capitalize()} ablation on NF-{dataset}-v3 (chronological split, '
         f'rules-only, mean $\\pm$ std over {len(args.seeds)} seeds).}}',
         f'\\label{{tab:ablation_{args.study}_v3}}', '\\small',
         '\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule',
         'Variant & ' + ' & '.join(hdr[METRICS.index(c)] for c in cols) + ' \\\\',
         '\\midrule']
for v in agg.index:
    cells = [f"{agg.loc[v, (c, 'mean')]:.3f} $\\pm$ {agg.loc[v, (c, 'std')]:.3f}"
             for c in cols]
    lines.append(NAMES[v] + ' & ' + ' & '.join(cells) + ' \\\\')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
tex = os.path.join(root, 'ablation_table.tex')
open(tex, 'w').write('\n'.join(lines) + '\n')
print(f'\nSaved {root}/ablation_runs.csv, ablation_summary.csv, ablation_table.tex')
