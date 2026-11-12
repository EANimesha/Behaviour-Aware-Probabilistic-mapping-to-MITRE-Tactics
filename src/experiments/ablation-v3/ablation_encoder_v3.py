"""
Encoder ablation on the v3 chronological split.

Every variant goes through the same frozen pipeline: DP-GMM (same K cap),
frozen v1 rules, Stage 6 benign detection, causal test inference.
LLM is off (SKIP_LLM=1), so all variants share the same semantic backend.
Main metric: per-flow tactic macro-F1 on the held-out test split.

Usage:
  V3_DATASET=bot_iot python run_ablation_v3.py
  python run_ablation_v3.py --variants full raw --seeds 42   # subset
  python run_ablation_v3.py --collect-only                   # just rebuild table
"""
import argparse, os, subprocess, sys
import pandas as pd

VARIANTS = ['full-llm','full', 'feature', 'feature_context', 'feature_temporal',
            'no_bhv_loss', 'raw']
NAMES = {'full-llm':'Full with llm','full': 'Full (F+C+T, $\\lambda$=1)', 'feature': 'Feature only',
         'feature_context': 'Feature + Context', 'feature_temporal': 'Feature + Temporal',
         'no_bhv_loss': 'Full, no behaviour loss ($\\lambda$=0)',
         'raw': 'Raw features + DP-GMM'}
METRICS = ['perflow_rules_macro_f1', 'mean_purity', 'weighted_purity', 'nmi_tactic']

ap = argparse.ArgumentParser()
ap.add_argument('--variants', nargs='+', default=VARIANTS)
ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
ap.add_argument('--collect-only', action='store_true')
args = ap.parse_args()

dataset = os.getenv('V3_DATASET', 'bot_iot')
root = '/home/nimesha/Downloads/PythonProject1_updated/PythonProject1/checkpoints_ablation_bot_iot'


def run_dir(v, s):
    return os.path.join(root, f'{v}_s{s}')


if not args.collect_only:
    for v in args.variants:
        for s in args.seeds:
            d = run_dir(v, s)
            out = os.path.join(d, 'inference_results', 'inference_metrics.csv')
            if os.path.exists(out):
                print(f'skip {v} seed {s} (done)'); continue
            # env = dict(os.environ, V3_DATASET=dataset, ABLATION=v, SEED=str(s),
            #            SKIP_LLM='1', CHECKPOINT_DIR=d)
            env = dict(os.environ, V3_DATASET=dataset, ABLATION=v, SEED=str(s),
                       SKIP_LLM='0', CHECKPOINT_DIR=d, OPENAI_API_KEY='')
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

hdr = ['Tactic macro-F1', 'Mean purity', 'Weighted purity', 'NMI (tactic)']
lines = ['\\begin{table}[t]', '\\centering',
         f'\\caption{{Encoder ablation on NF-{dataset}-v3 (chronological split, '
         f'rules-only, mean $\\pm$ std over {len(args.seeds)} seeds).}}',
         '\\label{tab:ablation_v3}', '\\small',
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
