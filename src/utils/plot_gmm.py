"""
Generate a REAL DP-GMM convergence plot from actual training data.

Fits the DP-GMM incrementally (max_iter=1,2,3,...) and records the
true variational lower bound (ELBO) at each iteration. This produces
the genuine concave convergence curve, not a synthetic straight line.

Usage:
  python plot_convergence.py bot_iot
  # NF-v3 (run from the project root; V3 embeddings rebuilt exactly as training):
  python -m src.utils.plot_gmm bot_iot_v3
  python -m src.utils.plot_gmm unsw_nb15_v3
"""

import sys, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import warnings
warnings.filterwarnings('ignore')

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'
IS_V3 = DATASET.endswith('_v3')
if DATASET == 'bot_iot_v3':  # config_v3 picks paths/checkpoints
    from src.config.config_v3 import *
elif DATASET == 'cicids2018':
    from src.config.config_cicids import *
elif DATASET == 'unsw_nb15':
    from src.config.config_unsw import *
else:
    from src.config.config import *

from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder

# v2 scripts ran from src/utils ("../.."); v3 runs from the project root
ROOT = "." if IS_V3 else os.path.join("..", "..")
EVAL_DIR = os.path.join(ROOT, CHECKPOINT_DIR, "convergence_results")
TITLES = {'bot_iot': 'NF-BoT-IoT-v2', 'unsw_nb15': 'NF-UNSW-NB15-v2',
          'cicids2018': 'NF-CSE-CIC-IDS2018-v2', 'bot_iot_v3': 'NF-BoT-IoT-v3',
          'unsw_nb15_v3': 'NF-UNSW-NB15-v3'}
os.makedirs(EVAL_DIR, exist_ok=True)

def get_embeddings_v3(device='cpu'):
    """Training embeddings of an NF-v3 checkpoint, rebuilt exactly as training:
    seeded per-class chronological split + balancers, train-period graph,
    causal sequences with their real mask, context encoder in train mode
    (dropout 0.1, as in training Stage 2). These are the Z the DP-GMM was fit on."""
    import random
    from src.data.balancer import FastChronologicalBalancer
    ckpt = torch.load(os.path.join("../..",MODEL_PATH), map_location=device, weights_only=False)
    cfg, prep = ckpt['config'], ckpt['preprocessing']

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    df = load_dataset(prep.get('data_path', DATA_PATH), sample_size=SAMPLE_SIZE)
    pre = Preprocessor(prep.get('feature_config_path', FEATURES_PATH))
    train_bal = test_bal = None
    if prep.get('use_balancer'):
        train_bal = FastChronologicalBalancer(allow_oversample=True)
        test_bal = FastChronologicalBalancer(
            max_total_samples=30_000, max_benign_samples=7_500,
            max_attack_per_class=3_000, allow_oversample=False)
    d = pre.preprocess_dataset(df, test_size=prep['test_size'], gap_size=prep['gap_size'],
                               train_balancer=train_bal, test_balancer=test_bal)
    if not np.array_equal(d['idx_train'], prep['idx_train']):
        raise RuntimeError("Training rows differ from the checkpoint: wrong DATA_PATH/FEATURES_PATH?")
    if not np.allclose(d['scaler'].mean_, ckpt['preprocessor_scaler'].mean_):
        raise RuntimeError("Scaler differs from training")

    idx_train, idx_split = d['idx_train'], d['idx_train_split']
    b = NetworkGraphBuilder(seq_length=cfg['seq_length'])
    g = b.build_graph(d['G_all'][idx_split], d['src_all'][idx_split], d['dst_all'][idx_split])
    nf = b.node_features
    ts, tm = b.build_causal_sequences(d['X_all'], d['src_all'], idx_train,
                                      history_positions=idx_split)

    ce = ContextEncoder(cfg['node_feature_dim'], 32, cfg['context_output_dim'], 0.1)
    fe = FeatureEncoder(cfg['feature_dim'], output_dim=cfg['feature_output_dim'])
    be = TransformerBehaviourEncoder(cfg['feature_dim'], 64, cfg['behaviour_output_dim'], 2, 4, 0.1)
    je = JointEncoder(fe, ce, be, cfg['feature_output_dim'],
        cfg['context_output_dim'], cfg['behaviour_output_dim'], cfg['z_dim'],
        n_behaviours=cfg.get('n_behaviours', 5), lambda_bhv=cfg.get('lambda_bhv', 1.0))
    je.load_state_dict(ckpt['joint_encoder_state'])
    with torch.no_grad():
        je.context_encoder.train()     # as training: dropout active for context
        cemb = b.get_ip_embeddings(je.context_encoder(g, nf), d['src_train'], d['dst_train'])
    je.eval()

    Xt = torch.tensor(d['X_train'], dtype=torch.float32)
    allz = []
    with torch.no_grad():
        for s in range(0, len(Xt), VAE_BATCH_SIZE):
            e = min(s + VAE_BATCH_SIZE, len(Xt))
            z, _ = je.encode(g, Xt[s:e], nf, cemb[s:e], ts[s:e], tm[s:e])
            allz.append(z)
    Z = torch.cat(allz).numpy()
    counts = np.bincount(ckpt['gmm_model'].predict(Z))
    print(f"  Check: saved GMM on rebuilt Z uses {int((counts > 0).sum())} clusters "
          f"(largest {counts.max() / len(Z):.1%}) - should match the training log")
    return Z, ckpt['gmm_config']


def get_embeddings(device='cpu'):
    if IS_V3:
        return get_embeddings_v3(device)
    ckpt = torch.load(os.path.join("../..",MODEL_PATH), map_location=device, weights_only=False)
    cfg = ckpt['config']
    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    if DATASET in ('unsw_nb15', 'cicids2018') and 'Attack' in df.columns:
        from src.data.balancer import FastChronologicalBalancer
        if DATASET == 'unsw_nb15':
            from src.config.config_unsw import (MAX_BENIGN_SAMPLES,
                MAX_ATTACK_PER_CLASS, MIN_ATTACK_PER_CLASS, MAX_TOTAL_SAMPLES)
        else:
            from src.config.config_cicids import (MAX_BENIGN_SAMPLES,
                MAX_ATTACK_PER_CLASS, MIN_ATTACK_PER_CLASS, MAX_TOTAL_SAMPLES)
        bal = FastChronologicalBalancer(
            max_total_samples=MAX_TOTAL_SAMPLES, max_benign_samples=MAX_BENIGN_SAMPLES,
            max_attack_per_class=MAX_ATTACK_PER_CLASS, min_attack_per_class=MIN_ATTACK_PER_CLASS)
        df, _ = bal.balance(df)
    n_train = int(len(df)*0.8)
    df_train = df.iloc[:n_train].reset_index(drop=True)

    pre = Preprocessor(clip_extremes=(DATASET != 'bot_iot'))
    if 'preprocessor_scaler' in ckpt:
        pre.scaler = ckpt['preprocessor_scaler']; pre.fitted = True
        pre.final_feature_names = ckpt.get('feature_names', [])
        X, src, dst = pre.transform_new(df_train)
    else:
        d = pre.preprocess_dataset(df_train)
        X, src, dst = d['X_train'], d['src_train'], d['dst_train']

    fe = FeatureEncoder(cfg['feature_dim'], cfg['feature_output_dim'])
    ce = ContextEncoder(cfg.get('node_feature_dim', NODE_FEATURE_DIM), 32,
                        cfg['context_output_dim'], 0.0)
    be = TransformerBehaviourEncoder(cfg['feature_dim'], 64,
                                     cfg['behaviour_output_dim'], 2, 4, 0.0)
    je = JointEncoder(fe, ce, be, cfg['feature_output_dim'],
        cfg['context_output_dim'], cfg['behaviour_output_dim'], cfg['z_dim'],
        n_behaviours=cfg.get('n_behaviours', 5), lambda_bhv=cfg.get('lambda_bhv', 1.0))
    je.load_state_dict(ckpt['joint_encoder_state']); je.eval()

    b = NetworkGraphBuilder(seq_length=cfg.get('seq_length', SEQ_LENGTH))
    g, ts = b.build_graph(X, src, dst); nf = b.node_features
    with torch.no_grad():
        ne = je.context_encoder(g, nf)
        cemb = b.get_ip_embeddings(ne, src, dst)
    Xt = torch.tensor(X, dtype=torch.float32); tm = torch.ones(len(X), ts.shape[1])
    allz = []
    with torch.no_grad():
        for s in range(0, len(Xt), VAE_BATCH_SIZE):
            e = min(s+VAE_BATCH_SIZE, len(Xt))
            hf = je.feature_encoder(Xt[s:e]); hb = je.behaviour_encoder(ts[s:e], tm[s:e])
            hc = cemb[s:e]
            if hasattr(je, 'norm_f'):
                hf = je.norm_f(hf); hc = je.norm_c(hc); hb = je.norm_b(hb)
            allz.append(je.encoder(torch.cat([hf, hc, hb], 1)))
    return torch.cat(allz).numpy(), ckpt['gmm_config']


def main():
    print(f"\n{'='*60}\nDP-GMM CONVERGENCE — {DATASET}\n{'='*60}")
    Z, gmm_cfg = get_embeddings()
    print(f"  Embeddings: {Z.shape}")

    max_iters = 100
    lower_bounds = []

    # Fit with increasing max_iter, recording the ELBO at each.
    # warm_start would be ideal but sklearn re-inits; instead we fit
    # to each max_iter and read the converged lower_bound_.
    print("  Fitting incrementally to trace convergence...")
    for it in range(1, max_iters + 1):
        gmm = BayesianGaussianMixture(
            n_components=gmm_cfg['max_components'],
            covariance_type=gmm_cfg['covariance_type'],
            weight_concentration_prior_type='dirichlet_process',
            max_iter=it, n_init=1, random_state=42,
            init_params='kmeans', tol=1e-10)  # tiny tol so it runs full it
        gmm.fit(Z)
        lower_bounds.append(gmm.lower_bound_)
        if it % 10 == 0:
            print(f"    iter {it}: ELBO={gmm.lower_bound_:.4f}, "
                  f"n_iter={gmm.n_iter_}, converged={gmm.converged_}")
        # Stop if actually converged
        if it > 5 and abs(lower_bounds[-1] - lower_bounds[-2]) < 1e-4:
            print(f"    Converged at iteration {it}")
            break

    lower_bounds = np.array(lower_bounds)
    iters = np.arange(1, len(lower_bounds) + 1)
    conv_iter = len(lower_bounds)
    conv_val = lower_bounds[-1]

    # Plot the REAL curve (concave, plateauing)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(iters, lower_bounds, 'b-o', ms=4, lw=2, label='Variational Lower Bound (ELBO)')
    ax.axhline(y=conv_val, color='red', ls='--', lw=1.5,
               label=f'Converged: {conv_val:.2f}')
    ax.annotate(f'Convergence at iteration {conv_iter}',
                xy=(conv_iter, conv_val),
                xytext=(conv_iter*0.55, conv_val - (lower_bounds.max()-lower_bounds.min())*0.25),
                color='red', fontsize=13, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_xlabel('EM Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Variational Lower Bound', fontsize=14, fontweight='bold')
    ax.set_title(f'DP-GMM Convergence on {TITLES.get(DATASET, DATASET)}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(EVAL_DIR, 'dpgmm_convergence.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved: {out}")
    print(f"  Final ELBO: {conv_val:.4f} at iteration {conv_iter}")
    print(f"  Curve shape: rise={lower_bounds[0]:.2f} -> {conv_val:.2f}")


if __name__ == '__main__':
    main()