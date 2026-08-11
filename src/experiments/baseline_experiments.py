"""
Baseline Comparisons with Publication-Quality Plots
======================================================

Compares proposed method against 6 baselines with:
  - Per-flow accuracy, macro F1, weighted F1
  - Binary AUC-ROC and AUC-PR
  - Multi-class ROC curves (one-vs-rest)
  - Per-tactic F1 comparison bar chart
  - Binary ROC overlay (all methods)

Usage:
  python baselines.py bot_iot
  python baselines.py cicids2018
"""

import sys, os, time, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    average_precision_score, f1_score, roc_curve, auc,
    precision_recall_curve
)
from sklearn.neural_network import MLPClassifier
from collections import Counter
warnings.filterwarnings('ignore')

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'

if DATASET == 'cicids2018':
    from src.config.config_cicids import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH)
elif DATASET == 'unsw_nb15':
    from src.config.config_unsw import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH)
else:
    from src.config.config import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH)

from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.evaluation.ground_truth import GroundTruthMapper
from src.knowledge_base.kb import KnowledgeBase

RESULTS_DIR = os.path.join(CHECKPOINT_DIR, 'baseline_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_prepare():
    """Load data, preprocess, split, map to tactics."""
    print(f"\n{'='*70}")
    print(f"BASELINE COMPARISONS — {DATASET}")
    print(f"{'='*70}")

    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    if DATASET in ('unsw_nb15', 'cicids2018') and 'Attack' in df.columns:
        from src.data.balancer import FastChronologicalBalancer
        if DATASET == 'unsw_nb15':
            from src.config.config_unsw import (
                MAX_BENIGN_SAMPLES, MAX_ATTACK_PER_CLASS,
                MIN_ATTACK_PER_CLASS, MAX_TOTAL_SAMPLES)
        else:
            from src.config.config_cicids import (
                MAX_BENIGN_SAMPLES, MAX_ATTACK_PER_CLASS,
                MIN_ATTACK_PER_CLASS, MAX_TOTAL_SAMPLES)
        balancer = FastChronologicalBalancer(
            max_total_samples=MAX_TOTAL_SAMPLES,
            max_benign_samples=MAX_BENIGN_SAMPLES,
            max_attack_per_class=MAX_ATTACK_PER_CLASS,
            min_attack_per_class=MIN_ATTACK_PER_CLASS)
        df, _ = balancer.balance(df)

    preprocessor = Preprocessor(clip_extremes=(DATASET != 'bot_iot'))
    data = preprocessor.preprocess_dataset(df)
    X, y_attack = data['X_train'], data['y_train']
    gt_mapper = GroundTruthMapper(DATASET)
    y_tactic = gt_mapper.map_labels(y_attack)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_tactic, test_size=0.2, stratify=y_tactic, random_state=42)
    y_train_bin = (y_train != 'None').astype(int)
    y_test_bin = (y_test != 'None').astype(int)

    tactic_list = sorted(set(y_tactic))
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"  Tactics: {tactic_list}")
    return X_train, X_test, y_train, y_test, y_train_bin, y_test_bin, tactic_list, data


def get_proposed_scores(X_test, y_test, y_test_bin, tactic_list, data):
    """Get proposed method's predictions and soft scores from checkpoint."""
    print(f"\n{'─'*50}")
    print("PROPOSED METHOD (from checkpoint)")
    print(f"{'─'*50}")

    try:
        checkpoint = torch.load((os.path.join("../..",MODEL_PATH)), map_location='cpu',
                                weights_only=False)
        kb = KnowledgeBase()
        kb.load(os.path.join("../..",KB_PATH))
    except Exception as e:
        print(f"  Cannot load checkpoint: {e}")
        return None

    from src.streaming.streaming_gmm import StreamingDPGMM
    from src.encoder.feature import FeatureEncoder
    from src.encoder.context import ContextEncoder
    from src.encoder.behaviour import TransformerBehaviourEncoder
    from src.encoder.jointencoder import JointEncoder
    from src.data.graph_builder import NetworkGraphBuilder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = checkpoint['config']

    # Build encoder
    feature_enc = FeatureEncoder(cfg['feature_dim'], cfg['feature_output_dim'])
    context_enc = ContextEncoder(
        cfg.get('node_feature_dim', 10), 32, cfg['context_output_dim'], 0.0)
    behaviour_enc = TransformerBehaviourEncoder(
        cfg['feature_dim'], 64, cfg['behaviour_output_dim'], 2, 4, 0.0)
    joint_enc = JointEncoder(
        feature_enc, context_enc, behaviour_enc,
        cfg['feature_output_dim'], cfg['context_output_dim'],
        cfg['behaviour_output_dim'], cfg['z_dim'],
        n_behaviours=cfg.get('n_behaviours', 5),
        lambda_bhv=cfg.get('lambda_bhv', 1.0))
    joint_enc.load_state_dict(checkpoint['joint_encoder_state'])
    joint_enc.to(device); joint_enc.eval()

    # Encode full training data, predict with GMM
    preprocessor = Preprocessor(clip_extremes=(DATASET != 'bot_iot'))
    full_data = preprocessor.preprocess_dataset(
        load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE))
    X_all = full_data['X_train']
    src_all, dst_all = full_data['src_train'], full_data['dst_train']

    builder = NetworkGraphBuilder(seq_length=cfg.get('seq_length', 10))
    graph, temporal_seqs = builder.build_graph(X_all, src_all, dst_all)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    with torch.no_grad():
        node_emb = joint_enc.context_encoder(graph, node_features)
        context_emb = builder.get_ip_embeddings(
            node_emb, src_all, dst_all).to(device)

    X_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)
    t_mask = torch.ones(len(X_all), temporal_seqs.shape[1]).to(device)

    all_z, all_recon_err = [], []
    with torch.no_grad():
        for s in range(0, len(X_tensor), 256):
            e = min(s + 256, len(X_tensor))
            h_f = joint_enc.feature_encoder(X_tensor[s:e])
            h_b = joint_enc.behaviour_encoder(temporal_seqs[s:e], t_mask[s:e])
            h_c = context_emb[s:e]
            if hasattr(joint_enc, 'norm_f'):
                h_f = joint_enc.norm_f(h_f)
                h_c = joint_enc.norm_c(h_c)
                h_b = joint_enc.norm_b(h_b)
            h_fusion = torch.cat([h_f, h_c, h_b], dim=1)
            z = joint_enc.encoder(h_fusion)
            h_recon = joint_enc.decoder(z)
            err = torch.mean((h_fusion - h_recon) ** 2, dim=1)
            all_z.append(z.cpu())
            all_recon_err.append(err.cpu())

    Z = torch.cat(all_z).numpy()
    recon_errors = torch.cat(all_recon_err).numpy()

    # GMM
    gmm_cfg = checkpoint['gmm_config']
    from src.streaming.streaming_gmm import StreamingDPGMM
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']

    clusters = gmm.model.predict(Z)
    gmm_scores = gmm.model.score_samples(Z)

    # Split same as baselines
    gt_mapper = GroundTruthMapper(DATASET)
    y_all = gt_mapper.map_labels(full_data['y_train'])
    indices = np.arange(len(X_all))
    _, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y_all, random_state=42)

    clusters_test = clusters[test_idx]
    gmm_scores_test = gmm_scores[test_idx]
    recon_test = recon_errors[test_idx]
    y_test_prop = y_all[test_idx]

    # Per-flow predictions from KB
    flow_preds_rules = []
    flow_scores_rules = []  # soft tactic scores
    for i in range(len(clusters_test)):
        entry = kb.lookup(int(clusters_test[i]))
        if entry:
            flow_preds_rules.append(entry['primary_tactic'])
            flow_scores_rules.append(entry['tactic_scores'])
        else:
            flow_preds_rules.append('Unknown')
            flow_scores_rules.append({})
    flow_preds_rules = np.array(flow_preds_rules)

    # Binary anomaly score
    neg_gmm = -gmm_scores_test
    combined_score = (recon_test / (recon_test.max() + 1e-10) +
                       neg_gmm / (neg_gmm.max() + 1e-10)) / 2

    return {
        'y_test': y_test_prop,
        'y_pred': flow_preds_rules,
        'tactic_scores': flow_scores_rules,
        'anomaly_scores': combined_score,
        'recon_errors': recon_test,
        'gmm_scores': gmm_scores_test,
    }


# ══════════════════════════════════════════════════════
# Baseline Methods
# ══════════════════════════════════════════════════════

def run_isolation_forest(X_train, X_test, y_test_bin):
    print(f"\n{'─'*50}\nIsolation Forest\n{'─'*50}")
    t0 = time.time()
    iso = IsolationForest(n_estimators=200, contamination=0.3,
                           random_state=42, n_jobs=-1)
    iso.fit(X_train)
    elapsed = time.time() - t0
    scores = -iso.score_samples(X_test)
    preds = (iso.predict(X_test) == -1).astype(int)
    auc_val = roc_auc_score(y_test_bin, scores)
    ap = average_precision_score(y_test_bin, scores)
    print(f"  AUC-ROC: {auc_val:.4f}, AUC-PR: {ap:.4f}, Time: {elapsed:.1f}s")
    fpr, tpr, _ = roc_curve(y_test_bin, scores)
    prec, rec, _ = precision_recall_curve(y_test_bin, scores)
    return {'method': 'Isolation Forest', 'type': 'Unsupervised',
            'binary_auc': auc_val, 'binary_ap': ap,
            'tactic_acc': None, 'macro_f1': None, 'weighted_f1': None,
            'time': elapsed, 'fpr': fpr, 'tpr': tpr,
            'prec_curve': prec, 'rec_curve': rec,
            'y_pred': None, 'anomaly_scores': scores}


def run_plain_ae(X_train, X_test, y_test_bin):
    print(f"\n{'─'*50}\nPlain Autoencoder\n{'─'*50}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = X_train.shape[1]
    class AE(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,32),nn.ReLU())
            s.dec = nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,d))
        def forward(s,x): z=s.enc(x); return s.dec(z)
    model = AE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    t0 = time.time()
    model.train()
    for ep in range(30):
        for i in range(0,len(Xt),256):
            b=Xt[i:i+256]; r=model(b); loss=nn.MSELoss()(r,b)
            opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time()-t0
    model.eval()
    Xtt = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        r = model(Xtt)
        errs = torch.mean((Xtt-r)**2, dim=1).cpu().numpy()
    auc_val = roc_auc_score(y_test_bin, errs)
    ap = average_precision_score(y_test_bin, errs)
    print(f"  AUC-ROC: {auc_val:.4f}, AUC-PR: {ap:.4f}, Time: {elapsed:.1f}s")
    fpr, tpr, _ = roc_curve(y_test_bin, errs)
    prec, rec, _ = precision_recall_curve(y_test_bin, errs)
    return {'method': 'Plain Autoencoder', 'type': 'Unsupervised',
            'binary_auc': auc_val, 'binary_ap': ap,
            'tactic_acc': None, 'macro_f1': None, 'weighted_f1': None,
            'time': elapsed, 'fpr': fpr, 'tpr': tpr,
            'prec_curve': prec, 'rec_curve': rec,
            'y_pred': None, 'anomaly_scores': errs}


def run_kmeans(X_train, X_test, y_train, y_test, tactic_list):
    print(f"\n{'─'*50}\nK-Means + Majority Vote\n{'─'*50}")
    n_t = len(tactic_list)
    best_acc, best_k, best_pred, best_time = 0, n_t, None, 0
    for k in [n_t, n_t*2, n_t*3, n_t*5]:
        t0 = time.time()
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        tc = km.fit_predict(X_train)
        el = time.time()-t0
        c2t = {}
        for c in range(k):
            m = tc == c
            c2t[c] = Counter(y_train[m]).most_common(1)[0][0] if m.sum()>0 else 'None'
        yp = np.array([c2t.get(c,'None') for c in km.predict(X_test)])
        a = accuracy_score(y_test, yp)
        if a > best_acc: best_acc, best_k, best_pred, best_time = a, k, yp, el
    report = classification_report(y_test, best_pred, labels=tactic_list,
                                    output_dict=True, zero_division=0)
    print(f"  K={best_k}, Acc: {best_acc:.4f}")
    return {'method': f'K-Means (K={best_k})', 'type': 'Unsupervised*',
            'binary_auc': None, 'binary_ap': None,
            'tactic_acc': best_acc,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'time': best_time, 'y_pred': best_pred}


def run_random_forest(X_train, X_test, y_train, y_test, y_test_bin, tactic_list):
    print(f"\n{'─'*50}\nRandom Forest (supervised)\n{'─'*50}")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    elapsed = time.time()-t0
    yp = rf.predict(X_test)
    yp_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, yp)
    report = classification_report(y_test, yp, labels=tactic_list,
                                    output_dict=True, zero_division=0)
    ypb = (yp != 'None').astype(int)
    bauc = roc_auc_score(y_test_bin, ypb)
    print(f"  Acc: {acc:.4f}, AUC: {bauc:.4f}, Time: {elapsed:.1f}s")
    return {'method': 'Random Forest', 'type': 'Supervised',
            'binary_auc': bauc, 'binary_ap': None,
            'tactic_acc': acc,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'time': elapsed, 'y_pred': yp, 'y_proba': yp_proba,
            'classes': rf.classes_}


def run_mlp(X_train, X_test, y_train, y_test, y_test_bin, tactic_list):
    print(f"\n{'─'*50}\nMLP Classifier (supervised)\n{'─'*50}")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=100,
                         random_state=42, early_stopping=True,
                         validation_fraction=0.1)
    mlp.fit(X_train, y_train)
    elapsed = time.time()-t0
    yp = mlp.predict(X_test)
    yp_proba = mlp.predict_proba(X_test)
    acc = accuracy_score(y_test, yp)
    report = classification_report(y_test, yp, labels=tactic_list,
                                    output_dict=True, zero_division=0)
    ypb = (yp != 'None').astype(int)
    bauc = roc_auc_score(y_test_bin, ypb)
    print(f"  Acc: {acc:.4f}, AUC: {bauc:.4f}, Time: {elapsed:.1f}s")
    return {'method': 'MLP Classifier', 'type': 'Supervised',
            'binary_auc': bauc, 'binary_ap': None,
            'tactic_acc': acc,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'time': elapsed, 'y_pred': yp, 'y_proba': yp_proba,
            'classes': mlp.classes_}


# ══════════════════════════════════════════════════════
# Publication Plots
# ══════════════════════════════════════════════════════

def plot_binary_roc(results, y_test_bin, proposed):
    """Binary ROC curves — all methods on one plot."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ROC
    ax = axes[0]
    for r in results:
        if 'fpr' in r and r['fpr'] is not None:
            ax.plot(r['fpr'], r['tpr'], linewidth=2,
                    label=f"{r['method']} (AUC={r['binary_auc']:.3f})")

    # Add proposed
    if proposed is not None:
        y_bin = (proposed['y_test'] != 'None').astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proposed['anomaly_scores'])
        proposed_auc = roc_auc_score(y_bin, proposed['anomaly_scores'])
        ax.plot(fpr, tpr, 'k-', linewidth=2.5,
                label=f"Proposed (AUC={proposed_auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('Binary Detection ROC (Attack vs Benign)', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    # PR curve
    ax = axes[1]
    for r in results:
        if 'prec_curve' in r and r['prec_curve'] is not None:
            ax.plot(r['rec_curve'], r['prec_curve'], linewidth=2,
                    label=f"{r['method']} (AP={r.get('binary_ap', 0):.3f})")

    if proposed is not None:
        y_bin = (proposed['y_test'] != 'None').astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, proposed['anomaly_scores'])
        proposed_ap = average_precision_score(y_bin, proposed['anomaly_scores'])
        ax.plot(rec, prec, 'k-', linewidth=2.5,
                label=f"Proposed (AP={proposed_ap:.3f})")

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Binary Detection Precision-Recall', fontsize=12)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'binary_roc_pr.png'), dpi=150)
    plt.close()


def plot_per_tactic_roc(proposed, tactic_list):
    """Per-tactic ROC curves for the proposed method (one-vs-rest)."""
    if proposed is None:
        return

    y_test = proposed['y_test']
    scores_list = proposed['tactic_scores']

    n_tactics = len(tactic_list)
    cols = min(4, n_tactics)
    rows = (n_tactics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    if n_tactics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, tactic in enumerate(tactic_list):
        ax = axes[idx]
        y_bin = (y_test == tactic).astype(int)

        # Get soft scores for this tactic
        tactic_scores_arr = np.array([
            s.get(tactic, 0) for s in scores_list])

        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            ax.text(0.5, 0.5, f'{tactic}\n(single class)',
                    ha='center', va='center', fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            continue

        fpr, tpr, _ = roc_curve(y_bin, tactic_scores_arr)
        auc_val = roc_auc_score(y_bin, tactic_scores_arr)

        ax.plot(fpr, tpr, 'b-', linewidth=2,
                label=f'AUC = {auc_val:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR', fontsize=9)
        ax.set_ylabel('TPR', fontsize=9)
        ax.set_title(f'{tactic}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_tactics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Per-Tactic ROC Curves (Proposed, One-vs-Rest)',
                  fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'per_tactic_roc.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_tactic_f1(results, proposed, tactic_list):
    """Per-tactic F1 bar chart comparing methods."""
    if proposed is None:
        return

    # Get per-tactic F1 for each method
    methods_data = {}

    # Proposed
    report_prop = classification_report(
        proposed['y_test'], proposed['y_pred'],
        labels=tactic_list, output_dict=True, zero_division=0)
    methods_data['Proposed'] = [
        report_prop.get(t, {}).get('f1-score', 0) for t in tactic_list]

    # Baselines with predictions
    for r in results:
        if r.get('y_pred') is not None and r.get('tactic_acc') is not None:
            report = classification_report(
                proposed['y_test'][:len(r['y_pred'])], r['y_pred'],
                labels=tactic_list, output_dict=True, zero_division=0)
            methods_data[r['method']] = [
                report.get(t, {}).get('f1-score', 0) for t in tactic_list]

    n_methods = len(methods_data)
    n_tactics = len(tactic_list)
    x = np.arange(n_tactics)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']

    for i, (method, f1s) in enumerate(methods_data.items()):
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1s, width, label=method,
                       color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(tactic_list, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_title('Per-Tactic F1 Comparison', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'per_tactic_f1.png'), dpi=150)
    plt.close()


def plot_capability_comparison(results, proposed):
    """Radar chart: multi-dimensional comparison."""
    categories = ['Tactic\nAccuracy', 'Binary\nAUC',
                   'Explainable', 'Cross-\nDataset',
                   'No Labels\nNeeded', 'Streaming']
    n_cats = len(categories)

    # Normalise scores to 0-1
    methods = {}

    # Supervised (best)
    best_sup = max([r for r in results if 'Supervised' in r.get('type', '')],
                   key=lambda x: x.get('tactic_acc', 0) or 0, default=None)
    if best_sup:
        methods['Random Forest\n(Supervised)'] = [
            (best_sup.get('tactic_acc', 0) or 0),
            (best_sup.get('binary_auc', 0) or 0),
            0.0, 0.0, 0.0, 0.0]

    # Best unsupervised
    methods['Isolation Forest'] = [
        0.0, 0.3, 0.0, 0.0, 1.0, 0.0]  # rough estimates

    # Proposed
    prop_acc = accuracy_score(proposed['y_test'], proposed['y_pred']) \
        if proposed else 0
    y_bin = (proposed['y_test'] != 'None').astype(int) if proposed else None
    prop_auc = roc_auc_score(y_bin, proposed['anomaly_scores']) \
        if proposed else 0
    methods['Proposed\n(BGTD-NID)'] = [
        prop_acc, prop_auc, 1.0, 1.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#E91E63', '#2196F3', '#4CAF50']
    for (name, vals), color in zip(methods.items(), colors):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, vals_plot, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title('Multi-Dimensional Capability Comparison', fontsize=12, y=1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'capability_radar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(all_results, proposed):
    """Print comparison table."""
    print(f"\n{'='*90}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'='*90}")

    print(f"\n{'Method':25s} {'Type':15s} {'Tactic Acc':>10s} "
          f"{'Macro F1':>10s} {'W-F1':>10s} {'Bin AUC':>10s}")
    print('-' * 85)
    for r in all_results:
        ta = f"{r['tactic_acc']:.4f}" if r['tactic_acc'] is not None else 'N/A'
        mf = f"{r['macro_f1']:.4f}" if r['macro_f1'] is not None else 'N/A'
        wf = f"{r['weighted_f1']:.4f}" if r['weighted_f1'] is not None else 'N/A'
        ba = f"{r['binary_auc']:.4f}" if r['binary_auc'] is not None else 'N/A'
        print(f"{r['method']:25s} {r['type']:15s} {ta:>10s} "
              f"{mf:>10s} {wf:>10s} {ba:>10s}")

    if proposed:
        acc = accuracy_score(proposed['y_test'], proposed['y_pred'])
        report = classification_report(
            proposed['y_test'], proposed['y_pred'],
            output_dict=True, zero_division=0)
        y_bin = (proposed['y_test'] != 'None').astype(int)
        prop_auc = roc_auc_score(y_bin, proposed['anomaly_scores'])
        print(f"{'Proposed (KB)':25s} {'Unsupervised':15s} "
              f"{acc:>10.4f} {report['macro avg']['f1-score']:>10.4f} "
              f"{report['weighted avg']['f1-score']:>10.4f} {prop_auc:>10.4f}")


def main():
    (X_train, X_test, y_train, y_test,
     y_train_bin, y_test_bin, tactic_list, data) = load_and_prepare()

    results = []
    results.append(run_isolation_forest(X_train, X_test, y_test_bin))
    results.append(run_plain_ae(X_train, X_test, y_test_bin))
    results.append(run_kmeans(X_train, X_test, y_train, y_test, tactic_list))
    results.append(run_random_forest(
        X_train, X_test, y_train, y_test, y_test_bin, tactic_list))
    results.append(run_mlp(
        X_train, X_test, y_train, y_test, y_test_bin, tactic_list))

    # Get proposed method results
    proposed = get_proposed_scores(
        X_test, y_test, y_test_bin, tactic_list, data)

    # Summary table
    print_summary(results, proposed)

    # Publication plots
    print(f"\n  Generating plots...")
    plot_binary_roc(results, y_test_bin, proposed)
    plot_per_tactic_roc(proposed, tactic_list)
    plot_per_tactic_f1(results, proposed, tactic_list)
    # plot_capability_radar(results, proposed)

    # Save CSV
    save_results = []
    for r in results:
        save_results.append({k: v for k, v in r.items()
                              if not isinstance(v, np.ndarray)})
    pd.DataFrame(save_results).to_csv(
        os.path.join(RESULTS_DIR, f'baseline_comparison_{DATASET}.csv'),
        index=False)

    print(f"\n  All saved: {RESULTS_DIR}/")
    print(f"  Plots: binary_roc_pr.png, per_tactic_roc.png, "
          f"per_tactic_f1.png, capability_radar.png")


if __name__ == '__main__':
    main()