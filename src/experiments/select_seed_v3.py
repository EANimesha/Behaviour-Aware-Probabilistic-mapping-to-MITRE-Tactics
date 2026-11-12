"""
Seed (model) selection on the VALIDATION split only.

For each seed's checkpoint, run inference on the chronological gap rows first:
    SEED=42 CHECKPOINT_DIR=ckpt_seed42 EVAL_SPLIT=val FUSION_W_RULES=0.5 python inference_v3.py
    (same for 43, 44)
which writes <ckpt>/inference_val/inference_metrics.csv. Test results are not read.

Then:
    python select_seed_v3.py --dirs ckpt_seed42 ckpt_seed43 ckpt_seed44
Selects the seed with the best VALIDATION fused macro-F1 (ties: accuracy) and
writes seed_selection_val.csv and seed_selection_val.tex.
"""
import argparse
import os

import pandas as pd

COLS = [('perflow_fused_accuracy', 'Flow acc.'),
        ('perflow_fused_macro_f1', 'Macro-F1'),
        ('perflow_rules_macro_f1', 'Rules F1'),
        ('perflow_llm_macro_f1', 'LLM F1')]
root = '/home/nimesha/Downloads/PythonProject1_updated/PythonProject1/checkpoints_ablation_bot_iot'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--out', default='.')
    args = ap.parse_args()
    rows = []
    for d in args.dirs:
        f = os.path.join(d, 'inference_val', 'inference_metrics.csv')
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}: run inference_v3.py with EVAL_SPLIT=val first")
        m = pd.read_csv(f).iloc[0].to_dict()
        try:
            import torch
            seed = torch.load(os.path.join(d, '../../../Results1/model_checkpoint.pt'), map_location='cpu',
                              weights_only=False)['config'].get('seed', d)
        except Exception:
            seed = d
        row = {'seed': seed, 'dir': d}
        for k, _ in COLS:
            row[k] = m.get(k, float('nan'))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(
        ['perflow_fused_macro_f1', 'perflow_fused_accuracy'], ascending=False)
    best = df.iloc[0]
    print(df.to_string(index=False))
    print(f"\nSelected on VALIDATION: seed {best['seed']} ({best['dir']})")
    df.to_csv(os.path.join(args.out, 'seed_selection_val.csv'), index=False)

    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize",
             r"\caption{Model (seed) selection on the validation split "
             r"(chronological gap between train and test; test data not used). "
             r"The selected model is marked in bold.}",
             r"\label{tab:seed_selection_val}",
             r"\begin{tabular}{l " + "c " * len(COLS) + "}", r"\toprule",
             r"\textbf{Seed} & " + " & ".join(rf"\textbf{{{n}}}" for _, n in COLS) + r" \\",
             r"\midrule"]
    for _, r in df.sort_values('seed', key=lambda s: s.astype(str)).iterrows():
        vals = [f"{r[k]:.3f}" for k, _ in COLS]
        sel = r['dir'] == best['dir']
        if sel:
            vals = [rf"\textbf{{{v}}}" for v in vals]
        lines.append((rf"\textbf{{{r['seed']}}}" if sel else str(r['seed'])) +
                     " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(args.out, 'seed_selection_val.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print("Saved seed_selection_val.csv / .tex")


if __name__ == '__main__':
    main()
