"""
Fusion-weight and window-size sweeps for NF-v3 (replaces Tables 19/20 of
ablation.py, which were computed on v2 training data).

  - Uses the main checkpoint WITH LLM labels (trained by training_v3.py,
    which now saves 'train_clusters').
  - Validation set = the 5% GAP of the chronological split (between train
    and test). Training never sees these flows and the rules are fitted
    without them; they are encoded causally exactly like the test flows,
    deduplicated against train and balanced with the test-split balancer.
  - Fusion weight: chosen on VALIDATION (per-flow tactic macro-F1); every
    weight is then reported on test. Test never picks anything.
  - Window size: reported on both splits as a sensitivity curve; W=50 is
    fixed in advance, not chosen from this curve.
  - Cluster semantics (rules, LLM) are the frozen training ones; test flows
    use the same causal encoding as inference_v3.py.

Usage:
  V3_DATASET=bot_iot python sweep_fusion_window_v3.py
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import inference_v3 as inf

FUSION_W = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
WINDOWS = [5, 10, 20, 30, 50, 75, 100, 200]
W_FIXED = 50
OUT = os.path.join(inf.CHECKPOINT_DIR, 'sweep_results')
os.makedirs(OUT, exist_ok=True)


def perflow(scores, clusters, true_t, tactic_list):
    pred = np.array([scores.get(int(c), {}).get('primary_tactic', 'Unknown')
                     for c in clusters])
    return ((pred == true_t).mean(),
            f1_score(true_t, pred, labels=tactic_list, average='macro', zero_division=0))


def window(scores, clusters, src, y, gt_mapper, ws):
    agg = inf.SoftWindowAggregator(window_size=ws, tactic_threshold=0.10)
    wins = agg.create_windows(src, clusters, scores, y_true=y)
    m, _ = inf.SoftTacticEvaluator().evaluate_windows(wins, gt_mapper, save_dir=None)
    return {'n_windows': m.get('n_windows', len(wins)),
            'top1': m.get('top1_accuracy', np.nan),
            'cosine': m.get('mean_cosine', np.nan),
            'soft_jaccard': m.get('mean_soft_jaccard', np.nan)}


def validation_positions(ckpt, data):
    """Gap rows of the chronological split, deduplicated against train and
    balanced like the test split."""
    prep = ckpt['preprocessing']
    pre = inf.Preprocessor(prep.get('feature_config_path', inf.FEATURES_PATH))
    df = data['df_raw']
    idx_tr, idx_gap, _ = pre.chronological_split(df, prep['test_size'], prep['gap_size'])
    assert np.array_equal(idx_tr, np.asarray(prep['idx_train_split'])), \
        "re-split differs from training"
    idx_gap, n_dup = pre.remove_cross_split_duplicates(df, idx_tr, idx_gap)
    if prep.get('use_balancer'):
        bal = inf.FastChronologicalBalancer(
            max_total_samples=30_000, max_benign_samples=7_500,
            max_attack_per_class=3_000, allow_oversample=False)
        idx_gap, _ = bal.select(df.iloc[idx_gap])
    idx_gap = np.sort(np.asarray(idx_gap))
    assert not np.intersect1d(idx_gap, prep['idx_test']).size
    assert not np.intersect1d(idx_gap, idx_tr).size
    print(f"  Validation (gap) flows: {len(idx_gap):,} ({n_dup} duplicates of train removed)")
    print(f"  Validation classes: {dict(pd.Series(df['Attack'].values[idx_gap]).value_counts())}")
    return idx_gap


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    joint_enc, gmm, kb, ckpt = inf.load_checkpoint(device)
    llm = inf.llm_scores(ckpt)
    if not llm:
        raise SystemExit("No LLM labels in checkpoint: retrain without SKIP_LLM=1.")

    data, idx_test = inf.load_test_data(ckpt)
    idx_val = validation_positions(ckpt, data)
    gt_mapper = inf.GroundTruthMapper(inf.DATASET)
    tactic_list = gt_mapper.get_tactic_list()

    splits = {}
    for split, idx in (('val', idx_val), ('test', idx_test)):
        Z, _ = inf.encode_test(joint_enc, ckpt, data, idx, device)
        splits[split] = (gmm.model.predict(Z), data['src_all'][idx], data['y_all'][idx])

    # ── Fusion weights ──
    rows = []
    for split, (cl, src, y) in splits.items():
        true_t = np.asarray(gt_mapper.map_labels(y))
        rules = inf.rules_scores(kb, cl)
        for w in FUSION_W:
            sc = inf.fuse(rules, llm, w, 1.0 - w)
            acc, f1 = perflow(sc, cl, true_t, tactic_list)
            rows.append({'split': split, 'w_rules': w, 'w_llm': round(1 - w, 1),
                         'perflow_acc': acc, 'macro_f1': f1,
                         **{f'w{W_FIXED}_{k}': v for k, v in
                            window(sc, cl, src, y, gt_mapper, W_FIXED).items()}})
            print(f"  {split:5s} w_rules={w:.1f}: acc={acc:.4f} F1={f1:.4f}")
    fus = pd.DataFrame(rows)
    va = fus[fus.split == 'val']
    w_sel = float(va.loc[va.macro_f1.idxmax(), 'w_rules'])
    fus['selected'] = fus.w_rules == w_sel
    fus.to_csv(os.path.join(OUT, 'fusion_sweep.csv'), index=False)
    print(f"\n  Selected on VALIDATION: w_rules={w_sel}")

    # ── Window size (fused scores at the selected weight) ──
    rows = []
    for split, (cl, src, y) in splits.items():
        sc = inf.fuse(inf.rules_scores(kb, cl), llm, w_sel, 1.0 - w_sel)
        for ws in WINDOWS:
            rows.append({'split': split, 'window': ws,
                         **window(sc, cl, src, y, gt_mapper, ws)})
            print(f"  {split:5s} W={ws:>3d}: {rows[-1]}")
    win = pd.DataFrame(rows)
    win.to_csv(os.path.join(OUT, 'window_sweep.csv'), index=False)

    # ── LaTeX (test numbers) ──
    t = fus[fus.split == 'test']
    lines = ['\\begin{tabular}{ccccc}', '\\toprule',
             '$w_{rule}$ & $w_{LLM}$ & Macro-F1 & Per-flow acc. & '
             f'W{W_FIXED} Top-1 \\\\', '\\midrule']
    for _, r in t.iterrows():
        cells = [f"{r.w_rules:.1f}", f"{r.w_llm:.1f}", f"{r.macro_f1:.4f}",
                 f"{r.perflow_acc:.4f}", f"{r[f'w{W_FIXED}_top1']:.4f}"]
        if r.selected:
            cells = [f'\\textbf{{{c}}}' for c in cells]
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    t = win[win.split == 'test']
    lines += ['\\begin{tabular}{rrccc}', '\\toprule',
              '$W$ & Windows & Top-1 & Cosine & Soft Jac. \\\\', '\\midrule']
    for _, r in t.iterrows():
        cells = [f"{int(r.window)}", f"{int(r.n_windows)}", f"{r.top1:.4f}",
                 f"{r.cosine:.4f}", f"{r.soft_jaccard:.4f}"]
        if r.window == W_FIXED:
            cells = [f'\\textbf{{{c}}}' for c in cells]
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(OUT, 'sweep_tables.tex'), 'w').write('\n'.join(lines) + '\n')
    print(f"\n  Saved: {OUT}/fusion_sweep.csv, window_sweep.csv, sweep_tables.tex")


if __name__ == '__main__':
    main()
