"""
Inference — NF-BoT-IoT-v3 (leakage-safe)
==========================================

Evaluates a checkpoint from training_bot_iot_v3.py on its held-out test rows.

Leakage controls (same as training):
  - Test rows = the exact positions saved in the checkpoint (per-class
    chronological split + gap + duplicate removal), not a new random split.
  - Test graph context is causal: the test period is cut into N_WINDOWS time
    windows; each window's IP graph is built only from flows BEFORE it.
    Node features use the training graph's node scaler.
  - Test sequences = previous flows from the same source IP (all earlier
    traffic, i.e. what a deployed IDS would have seen).
  - The feature scaler is the one fitted on the training period.

Scoring sources (all evaluated per flow and per window):
  1. Rules  : KB (SoftClusterScorer + Stage 6 benign rules)
  2. LLM    : cluster_behaviour_map_hard (*_llm scores)
  3. Fusion : w_rules * rules + w_llm * llm (behaviour and tactic level)

Usage:
  python inference_bot_iot_v3.py
  FUSION_W_RULES=0.7 python inference_bot_iot_v3.py
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import sys
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import (
    silhouette_score, normalized_mutual_info_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot_v3'

if DATASET == 'unsw_nb15_v3':
    from src.config.config_unsw_v3 import (
        DATA_PATH, FEATURES_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
        NODE_FEATURE_DIM, SEQ_LENGTH, VAE_BATCH_SIZE,
    )
if DATASET == 'cicids2018_v3':
    from src.config.config_cicids_v3 import (
        DATA_PATH, FEATURES_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
        NODE_FEATURE_DIM, SEQ_LENGTH, VAE_BATCH_SIZE,
    )
else:
    from src.config.config_v3 import (
        DATA_PATH, FEATURES_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
        NODE_FEATURE_DIM, SEQ_LENGTH, VAE_BATCH_SIZE,
    )

from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.balancer import FastChronologicalBalancer
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
from src.streaming.streaming_gmm import StreamingDPGMM
from src.knowledge_base.kb import KnowledgeBase
from src.mapping.soft_cluster_scorer import SoftWindowAggregator
from src.evaluation.ground_truth import GroundTruthMapper
from src.evaluation.soft_eval import SoftTacticEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GT_DATASET = {
    'bot_iot_v3': 'bot_iot',
    'unsw_nb15_v3': 'unsw_nb15',
    'cicids2018_v3':'cicids2018'
}.get(DATASET, DATASET)         # ground-truth tactic mapping (same labels in v3)
N_WINDOWS = 10                 # causal test-graph windows
WINDOW_SIZES = [20, 50]        # flows per source-IP window
W_RULES = float(os.getenv('FUSION_W_RULES', 0.7))
W_LLM = 1.0 - W_RULES

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "inference_results")
os.makedirs(EVAL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════

def load_checkpoint(device='cpu'):
    """Rebuild encoder + GMM + KB exactly as saved by training."""
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = ckpt['config']

    feature_enc = FeatureEncoder(cfg['feature_dim'], output_dim=cfg['feature_output_dim'])
    context_enc = ContextEncoder(
        cfg.get('node_feature_dim', NODE_FEATURE_DIM), 32, cfg['context_output_dim'], 0.0)
    behaviour_enc = TransformerBehaviourEncoder(
        cfg['feature_dim'], 64, cfg['behaviour_output_dim'], 2, 4, 0.0)
    joint_enc = JointEncoder(
        feature_enc, context_enc, behaviour_enc,
        cfg['feature_output_dim'], cfg['context_output_dim'],
        cfg['behaviour_output_dim'], cfg['z_dim'],
        n_behaviours=cfg.get('n_behaviours', 5),
        lambda_bhv=cfg.get('lambda_bhv', 1.0))
    joint_enc.load_state_dict(ckpt['joint_encoder_state'])
    joint_enc.to(device).eval()

    gcfg = ckpt['gmm_config']
    gmm = StreamingDPGMM(gcfg['max_components'], gcfg['covariance_type'],
                         gcfg['weight_threshold'], gcfg['novelty_threshold'])
    gmm.model = ckpt['gmm_model']

    kb = KnowledgeBase()
    kb.load(KB_PATH)

    active = int(np.sum(gmm.model.weights_ > gcfg['weight_threshold']))
    print(f"  Loaded: encoder(z={cfg['z_dim']}), GMM({active} clusters), "
          f"KB({len(kb.entries)} entries)")
    return joint_enc, gmm, kb, ckpt


def load_test_data(ckpt):
    """Re-create the training timeline and take the SAVED test positions."""
    prep = ckpt['preprocessing']
    df = load_dataset(prep.get('data_path', DATA_PATH), sample_size=SAMPLE_SIZE)
    preprocessor = Preprocessor(prep.get('feature_config_path', FEATURES_PATH))

    # Same split + balancing as training (the saved positions are what matter;
    # balancers are passed only so preprocessing runs identically)
    if prep.get('use_balancer'):
        train_bal = FastChronologicalBalancer(allow_oversample=True)
        test_bal = FastChronologicalBalancer(
            max_total_samples=30_000, max_benign_samples=7_500,
            max_attack_per_class=3_000, allow_oversample=False)
    else:
        train_bal = test_bal = None
    data = preprocessor.preprocess_dataset(
        df, test_size=prep['test_size'], gap_size=prep['gap_size'],
        train_balancer=train_bal, test_balancer=test_bal)

    # Use the training scaler, not a refit (identical if the data is identical)
    saved = ckpt['preprocessor_scaler']
    if not np.allclose(saved.mean_, data['scaler'].mean_):
        raise RuntimeError("Scaler differs from training: DATA_PATH or the "
                           "features JSON is not the one used for training.")
    if list(data['feature_names']) != list(ckpt['feature_names']):
        raise RuntimeError("Feature names differ from training.")

    idx_test = np.sort(np.asarray(prep['idx_test']))
    idx_train_split = np.asarray(prep['idx_train_split'])
    if idx_test.max() >= len(data['X_all']):
        raise RuntimeError("Saved test positions exceed the timeline: wrong DATA_PATH?")
    overlap = np.intersect1d(idx_test, idx_train_split)
    assert len(overlap) == 0, "test rows overlap the train period"

    print(f"  Timeline: {len(data['X_all']):,} flows | test: {len(idx_test):,} "
          f"(saved positions) | duplicates removed in training: "
          f"{prep.get('n_duplicates_removed', 'n/a')}")
    return data, idx_test


# ══════════════════════════════════════════════════════════
# Causal encoding of the test flows
# ══════════════════════════════════════════════════════════

def causal_context(joint_enc, ckpt, data, idx_test, device):
    """Graph context per test flow, from flows strictly before its window."""
    G_all, src_all, dst_all = data['G_all'], data['src_all'], data['dst_all']
    builder = NetworkGraphBuilder(seq_length=SEQ_LENGTH)
    builder.node_mean = np.asarray(ckpt['node_scaler']['mean'])
    builder.node_std = np.asarray(ckpt['node_scaler']['std'])

    out, n_unseen = [], 0
    for pos in np.array_split(idx_test, N_WINDOWS):
        if len(pos) == 0:
            continue
        past = np.arange(int(pos[0]))  # every flow before this window starts
        emb = None
        if len(past) > 0:
            graph = builder.build_graph(
                G_all[past], src_all[past], dst_all[past], fit_node_scaler=False)
            with torch.no_grad():
                node_emb = joint_enc.context_encoder(
                    graph.to(device), builder.node_features.to(device))
            emb = builder.get_ip_embeddings(node_emb, src_all[pos], dst_all[pos])
            emb = np.asarray(emb.cpu() if hasattr(emb, 'cpu') else emb)
            n_unseen += int(sum(ip not in builder.ip_to_idx for ip in src_all[pos]))
        out.append((pos, emb))

    dim = next(e.shape[1] for _, e in out if e is not None)
    ctx = np.zeros((len(idx_test), dim), dtype=np.float32)
    where = {p: i for i, p in enumerate(idx_test)}
    for pos, emb in out:
        if emb is not None:
            ctx[[where[p] for p in pos]] = emb
    print(f"  Causal graph: {N_WINDOWS} windows, {n_unseen:,} test flows from "
          f"source IPs not seen before their window (zero context)")
    return torch.tensor(ctx, dtype=torch.float32)


def encode_test(joint_enc, ckpt, data, idx_test, device):
    X_all = data['X_all']
    builder = NetworkGraphBuilder(seq_length=ckpt['config'].get('seq_length', SEQ_LENGTH))
    seqs, mask = builder.build_causal_sequences(
        X_all, data['src_all'], idx_test, history_positions=None)
    ctx = causal_context(joint_enc, ckpt, data, idx_test, device)
    X = torch.tensor(X_all[idx_test], dtype=torch.float32)

    Z, recon = [], []
    with torch.no_grad():
        for s in range(0, len(X), VAE_BATCH_SIZE):
            e = min(s + VAE_BATCH_SIZE, len(X))
            z, h_fusion, h_recon, _ = joint_enc(
                None, X[s:e].to(device), None, ctx[s:e].to(device),
                seqs[s:e].to(device), mask[s:e].to(device))
            Z.append(z.cpu().numpy())
            recon.append(((h_fusion - h_recon) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(Z), np.concatenate(recon)


# ══════════════════════════════════════════════════════════
# Scoring sources
# ══════════════════════════════════════════════════════════

def _primary(d):
    return max(d, key=d.get) if d else 'Unknown'


def rules_scores(kb, clusters):
    out = {}
    for k in np.unique(clusters):
        e = kb.lookup(int(k))
        if e:
            out[int(k)] = {'behaviour_scores': e['behaviour_scores'],
                           'tactic_scores': e['tactic_scores'],
                           'primary_behaviour': e['primary_behaviour'],
                           'primary_tactic': e['primary_tactic']}
        else:
            out[int(k)] = {'behaviour_scores': {'unknown': 1.0},
                           'tactic_scores': {'Unknown': 1.0},
                           'primary_behaviour': 'unknown', 'primary_tactic': 'Unknown'}
    return out


def llm_scores(ckpt):
    """LLM scores saved by training. Mock-tree fallbacks are excluded."""
    out, n_mock = {}, 0
    for k, e in ckpt.get('cluster_behaviour_map_hard', {}).items():
        if 'behaviour_scores_llm' not in e or 'tactic_scores_llm' not in e:
            continue
        if str(e.get('reasoning', '')).startswith('Mock'):
            n_mock += 1
            continue
        out[int(k)] = {'behaviour_scores': e['behaviour_scores_llm'],
                       'tactic_scores': e['tactic_scores_llm'],
                       'primary_behaviour': _primary(e['behaviour_scores_llm']),
                       'primary_tactic': _primary(e['tactic_scores_llm'])}
    if n_mock:
        print(f"  WARNING: {n_mock} clusters have mock (non-LLM) labels; excluded from LLM/fusion")
    # Auto-benign entries come from Stage 6, not the LLM: no real LLM run -> none
    if all(ckpt['cluster_behaviour_map_hard'][k].get('reasoning') == 'Auto-detected benign'
           for k in out):
        return {}
    return out


def fuse(rules, llm, w_rules, w_llm):
    fused = {}
    for k, r in rules.items():
        l = llm.get(k)
        if not l:
            fused[k] = r
            continue
        entry = {}
        for key in ('behaviour_scores', 'tactic_scores'):
            keys = set(r[key]) | set(l[key])
            d = {x: w_rules * r[key].get(x, 0) + w_llm * l[key].get(x, 0) for x in keys}
            tot = sum(d.values()) or 1.0
            entry[key] = {x: v / tot for x, v in d.items()}
        entry['primary_behaviour'] = _primary(entry['behaviour_scores'])
        entry['primary_tactic'] = _primary(entry['tactic_scores'])
        fused[k] = entry
    return fused


# ══════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════

def plot_confusion(true_t, pred_t, labels, title, path):
    cm = confusion_matrix(true_t, pred_t, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.0%})',
                    ha='center', va='center', fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def evaluate_source(name, scores, clusters, src_test, y_test, true_t,
                    tactic_list, gt_mapper, metrics):
    print(f"\n  === {name.upper()} ===")
    pred = np.array([scores.get(int(c), {}).get('primary_tactic', 'Unknown')
                     for c in clusters])
    acc = (pred == true_t).mean()
    rep = classification_report(true_t, pred, labels=tactic_list,
                                output_dict=True, zero_division=0)
    print(f"  Per-flow accuracy: {acc:.4f}")
    print(classification_report(true_t, pred, labels=tactic_list,
                                zero_division=0, digits=4))
    metrics[f'perflow_{name}_accuracy'] = acc
    metrics[f'perflow_{name}_macro_f1'] = rep['macro avg']['f1-score']
    metrics[f'perflow_{name}_weighted_f1'] = rep['weighted avg']['f1-score']
    for t in tactic_list:
        if t in rep:
            for m in ('f1-score', 'precision', 'recall'):
                metrics[f'perflow_{name}_{t}_{m.split("-")[0]}'] = rep[t][m]
    plot_confusion(true_t, pred, tactic_list, f'Per-flow tactic confusion ({name})',
                   os.path.join(EVAL_DIR, f'confusion_{name}.png'))

    for ws in WINDOW_SIZES:
        print(f"\n  --- {name} window size = {ws} ---")
        agg = SoftWindowAggregator(window_size=ws, tactic_threshold=0.10)
        windows = agg.create_windows(src_test, clusters, scores, y_true=y_test)
        wm, _ = SoftTacticEvaluator().evaluate_windows(
            windows, gt_mapper, save_dir=os.path.join(EVAL_DIR, f'{name}_ws_{ws}'))
        for k, v in wm.items():
            metrics[f'window{ws}_{name}_{k}'] = v
    return rep


def run_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'=' * 70}\nINFERENCE — NF-BoT-IoT-v3 (leakage-safe)\n{'=' * 70}")

    joint_enc, gmm, kb, ckpt = load_checkpoint(device)
    data, idx_test = load_test_data(ckpt)
    y_test = data['y_all'][idx_test]
    src_test = data['src_all'][idx_test]

    t1 = time.perf_counter()
    Z, recon_errors = encode_test(joint_enc, ckpt, data, idx_test, device)
    t_encode = time.perf_counter() - t1
    print(f"  Embeddings: {Z.shape} in {t_encode:.2f}s")

    t2 = time.perf_counter()
    clusters = gmm.model.predict(Z)
    # t_cluster = time.perf_counter() - t2
    # # Attribution: KB lookup + fusion + per-flow tactic (no LLM call at inference)
    # t3 = time.perf_counter()
    # _r = rules_scores(kb, clusters)
    # _sc = fuse(_r, llm_scores(ckpt), W_RULES, W_LLM)
    # _pred = [_sc[int(c)]['primary_tactic'] for c in clusters]
    # t_attr = time.perf_counter() - t3
    # _n = len(idx_test); _k = 1000.0 / max(_n, 1)
    # timing = {'time_n_flows': _n,
    #           'time_encode_per1k_s': t_encode * _k,
    #           'time_cluster_per1k_s': t_cluster * _k,
    #           'time_attribution_per1k_s': t_attr * _k,
    #           'time_end_to_end_per1k_s': (t_encode + t_cluster + t_attr) * _k,
    #           'time_per_flow_ms': (t_encode + t_cluster + t_attr) * 1000.0 / max(_n, 1)}
    # print("  Timing (per 1000 flows): encode %.4fs | cluster %.4fs | attribution %.4fs | "
    #       "end-to-end %.4fs | per flow %.4f ms" % (
    #           timing['time_encode_per1k_s'], timing['time_cluster_per1k_s'],
    #           timing['time_attribution_per1k_s'], timing['time_end_to_end_per1k_s'],
    #           timing['time_per_flow_ms']))
    gmm_scores = gmm.model.score_samples(Z)
    novelty_mask = gmm_scores < np.percentile(gmm_scores, 5)

    gt_mapper = GroundTruthMapper(GT_DATASET)
    true_t = np.asarray(gt_mapper.map_labels(y_test))
    tactic_list = gt_mapper.get_tactic_list()
    metrics = {}

    # ── Section 1: clustering quality ──
    print(f"\n{'=' * 70}\nSECTION 1: CLUSTERING QUALITY (test)\n{'=' * 70}")
    n_s = min(5000, len(Z))
    purities, sizes = [], []
    for k in np.unique(clusters):
        m = clusters == k
        purities.append(Counter(y_test[m]).most_common(1)[0][1] / m.sum())
        sizes.append(m.sum())
    metrics.update({
        'mean_purity': float(np.mean(purities)),
        'weighted_purity': float(np.average(purities, weights=sizes)),
        'nmi_attack': normalized_mutual_info_score(y_test, clusters),
        'nmi_tactic': normalized_mutual_info_score(true_t, clusters),
        'sil_attack': silhouette_score(Z, y_test, sample_size=n_s, random_state=42),
        'sil_tactic': silhouette_score(Z, true_t, sample_size=n_s, random_state=42),
    })
    if len(np.unique(clusters)) > 1:
        metrics['sil_cluster'] = silhouette_score(Z, clusters, sample_size=n_s, random_state=42)
    for k in ('mean_purity', 'weighted_purity', 'nmi_tactic', 'sil_tactic'):
        print(f"  {k:<16s} {metrics[k]:.4f}")

    # ── Sections 2-4: rules, LLM, fusion (per flow + window) ──
    rules = rules_scores(kb, clusters)
    llm = llm_scores(ckpt)
    sources = [('rules', rules)]
    if llm:
        sources += [('llm', {k: llm.get(k, rules[k]) for k in rules}),
                    ('fused', fuse(rules, llm, W_RULES, W_LLM))]
        print(f"  LLM scores for {len(llm)} clusters; fusion w_rules={W_RULES}, w_llm={W_LLM:.2f}")
        missing = sorted(set(rules) - set(llm))
        if missing:
            print(f"  Clusters without LLM scores use rules in 'llm'/'fused': {missing}")
    else:
        print("  No LLM scores in checkpoint: LLM and fusion skipped")

    # Per-cluster diagnostic on the TEST flows: what each source predicts
    print(f"\n{'=' * 70}\nPER-CLUSTER (test): true mix vs predicted tactic\n{'=' * 70}")
    print(f"  {'k':>3s} {'n':>6s}  {'true tactic mix':<45s} " +
          ' '.join(f'{n:>15s}' for n, _ in sources))
    rows = []
    for k in np.unique(clusters):
        m = clusters == k
        mix = Counter(true_t[m]).most_common(3)
        mix_s = ', '.join(f"{t} {c / m.sum():.0%}" for t, c in mix)
        preds = [sc.get(int(k), {}).get('primary_tactic', 'Unknown') for _, sc in sources]
        print(f"  {int(k):>3d} {m.sum():>6,}  {mix_s:<45s} " +
              ' '.join(f'{p:>15s}' for p in preds))
        rows.append({'cluster': int(k), 'n_test': int(m.sum()), 'true_mix': mix_s,
                     **{f'pred_{n}': p for (n, _), p in zip(sources, preds)}})
    pd.DataFrame(rows).to_csv(os.path.join(EVAL_DIR, 'per_cluster_test.csv'), index=False)

    print(f"\n{'=' * 70}\nSECTIONS 2-4: PER-FLOW + WINDOW (rules / LLM / fused)\n{'=' * 70}")
    reports = {}
    for name, sc in sources:
        reports[name] = evaluate_source(name, sc, clusters, src_test, y_test, true_t,
                                        tactic_list, gt_mapper, metrics)

    if len(reports) > 1:
        names = list(reports)
        print(f"\n  {'Tactic':>16s} " + ' '.join(f'{n + " F1":>10s}' for n in names))
        for t in tactic_list:
            print(f"  {t:>16s} " + ' '.join(
                f"{reports[n].get(t, {}).get('f1-score', 0):>10.4f}" for n in names))

    # ── Section 5: anomaly detection ──
    print(f"\n{'=' * 70}\nSECTION 5: ANOMALY DETECTION\n{'=' * 70}")
    binary = (true_t != 'None').astype(int)
    if len(np.unique(binary)) > 1:
        neg = -gmm_scores
        neg_n = (neg - neg.min()) / (np.ptp(neg) + 1e-10)
        rec_n = (recon_errors - recon_errors.min()) / (np.ptp(recon_errors) + 1e-10)
        for name, s in [('recon_error', recon_errors), ('gmm_neg_loglik', neg),
                        ('combined', (neg_n + rec_n) / 2)]:
            metrics[f'auc_{name}'] = roc_auc_score(binary, s)
            metrics[f'ap_{name}'] = average_precision_score(binary, s)
            print(f"  {name:>15s}: AUC-ROC={metrics[f'auc_{name}']:.4f}, "
                  f"AUC-PR={metrics[f'ap_{name}']:.4f}")
    print(f"  Novelty (lowest 5% log-lik): {novelty_mask.sum()} flows "
          f"{dict(Counter(y_test[novelty_mask]))}")

    # ── Summary ──
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"  {'':>8s} {'flow acc':>9s} {'macro F1':>9s} " +
          ' '.join(f'{"W" + str(w) + " top1":>9s}' for w in WINDOW_SIZES))
    for name, _ in sources:
        print(f"  {name:>8s} {metrics[f'perflow_{name}_accuracy']:>9.4f} "
              f"{metrics[f'perflow_{name}_macro_f1']:>9.4f} " +
              ' '.join(f"{metrics.get(f'window{w}_{name}_top1_accuracy', float('nan')):>9.4f}"
                       for w in WINDOW_SIZES))

    pd.DataFrame([metrics]).to_csv(os.path.join(EVAL_DIR, 'inference_metrics.csv'), index=False)
    print(f"\n  Metrics saved: {EVAL_DIR}/inference_metrics.csv")


if __name__ == '__main__':
    t0 = time.perf_counter()
    run_inference()
    print(f"Total inference time: {time.perf_counter() - t0:.2f}s")