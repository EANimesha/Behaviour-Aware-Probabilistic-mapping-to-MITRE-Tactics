"""
Training Pipeline — Behaviour-Aware
=====================================

Stage 1: Load & preprocess (leakage-safe: time-sorted, per-class
         chronological split with gap, duplicate removal, train-only scaling)
Stage 2: Graph (train-period flows only) + causal temporal sequences
Stage 3: Compute behaviour scores (AE training targets)
Stage 4: Train behaviour-aware AE
Stage 5: Extract z + DP-GMM clustering on z
Stage 6: Size-based benign detection
Stage 7: Cluster summarization + LLM behaviour labeling
Stage 8: Save checkpoint + KB (cluster → behaviour mapping)
"""

import os
import numpy as np
import torch
from collections import Counter, defaultdict
import time

from src.config.config_cicids_v3 import (
    DATA_PATH, FEATURES_PATH, SAMPLE_SIZE,
    FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
    Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
    VAE_EPOCHS, VAE_BATCH_SIZE, VAE_LR,
    GMM_MAX_COMPONENTS, GMM_COVARIANCE_TYPE,
    DEFAULT_W1_GMM, DEFAULT_W2_LLM,
    OPENAI_API_KEY, LLM_MODEL,
    CHECKPOINT_DIR, KB_PATH, MODEL_PATH,
)
from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.balancer import FastChronologicalBalancer
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
from src.encoder.behaviour_scorer import BehaviourScorer
from src.streaming.streaming_gmm import StreamingDPGMM
from src.clustering.summarizer import ClusterSummarizer, flow_ratio_means
from src.llm.reasoning import LLMLabeler
from src.knowledge_base.kb import KnowledgeBase
# from src.utils.plot_gmm import plot_dpgmm_convergence
from src.utils.training_logger import TrainingLogger, compute_grad_norms

# ── Leakage-safe preprocessing settings (move to src/config/config.py if you prefer) ──
# Written by feature_analysis_nf_v3.ipynb (train split only)
TEST_SIZE = 0.15   # latest 15% of every class -> test
GAP_SIZE = 0.05    # next-latest 5% of every class -> dropped (no session spans the cut)
USE_BALANCER = True  # caps large classes / tops up rare ones in TRAIN only

# V3_DATASET (config_v3 / env): bot_iot or unsw_nb15
GT_DATASET = 'cicids2018'                 # ground-truth tactic mapping
SCORER_DATASET = "cicids2018_v3"     # rule set in SoftClusterScorer
# SKIP_LLM=1: no LLM calls (first run, only to get clusters for analyse_clusters)
SKIP_LLM = os.getenv('SKIP_LLM') == '1'


def train_ae(joint_enc, graph, X_train_tensor, node_features,
             context_embeddings, temporal_seqs, temporal_mask,
             behaviour_targets, epochs, batch_size, lr, device,
             y_labels=None):
    """Train behaviour-aware AE with logging for paper figures."""
    print("\n" + "="*60)
    print("STAGE 4: Training Behaviour-Aware Autoencoder")
    print("="*60)

    logger = TrainingLogger(save_dir=os.path.join(CHECKPOINT_DIR, 'training_curves'))
    joint_enc.to(device); joint_enc.train()
    optimizer = torch.optim.Adam(joint_enc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = len(X_train_tensor)
    n_batches = (n + batch_size - 1) // batch_size

    for epoch in range(epochs):
        perm = torch.randperm(n)
        ep_t, ep_r, ep_b = 0., 0., 0.
        for b in range(n_batches):
            s, e = b*batch_size, min((b+1)*batch_size, n)
            bp = perm[s:e]
            z, hf, hr, pb = joint_enc(
                graph=graph, flow_features=X_train_tensor[bp],
                node_features=node_features,
                context_embeddings=context_embeddings[bp],
                temporal_sequences=temporal_seqs[bp],
                temporal_mask=temporal_mask[bp])
            loss, rl, bl = joint_enc.compute_loss(hf, hr, pb, behaviour_targets[bp])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(joint_enc.parameters(), 1.0)
            optimizer.step()
            ep_t += loss.item(); ep_r += rl.item(); ep_b += bl.item()
        scheduler.step()
        at, ar, ab = ep_t/n_batches, ep_r/n_batches, ep_b/n_batches
        current_lr = scheduler.get_last_lr()[0]
        gnorms = compute_grad_norms(joint_enc)
        logger.log_epoch(epoch + 1, at, ar, ab, current_lr, gnorms)

        sil_interval = max(1, epochs // 6)
        if y_labels is not None and ((epoch+1) % sil_interval == 0 or epoch == 0):
            joint_enc.eval()
            with torch.no_grad():
                all_z = []
                for si in range(0, n, batch_size):
                    ei = min(si + batch_size, n)
                    zc = joint_enc.get_embeddings(
                        graph, X_train_tensor[si:ei], node_features,
                        context_embeddings[si:ei], temporal_seqs[si:ei],
                        temporal_mask[si:ei])
                    all_z.append(zc.cpu())
                Z_check = torch.cat(all_z).numpy()
            sil = logger.log_silhouette(epoch + 1, Z_check, y_labels)
            joint_enc.train()
            sil_str = f" | Sil: {sil:.4f}" if sil else ""
            print(f"  Epoch {epoch+1:3d}/{epochs} | Total: {at:.6f} | "
                  f"Recon: {ar:.6f} | Bhv: {ab:.6f}{sil_str}")
        elif (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Total: {at:.6f} | "
                  f"Recon: {ar:.6f} | Bhv: {ab:.6f}")

    logger.plot_all()
    print(f"  Done. total={at:.6f}")
    return joint_enc


def report_ip_overlap(data):
    """Share of test flows whose source IP also appears in the train period."""
    train_ips = set(data['src_all'][data['idx_train_split']])
    seen = np.array([ip in train_ips for ip in data['src_test']])
    print("  Source-IP overlap (test flows whose src IP was seen in train):")
    for label in np.unique(data['y_test']):
        m = data['y_test'] == label
        print(f"    {label}: {seen[m].mean():.1%} of {m.sum():,} flows")
    print(f"    Overall: {seen.mean():.1%}")


def extract_embeddings(joint_enc, graph, X_tensor, node_features,
                       context_emb, temporal_seqs, temporal_mask,
                       batch_size, device):
    joint_enc.eval()
    all_z = []
    with torch.no_grad():
        for s in range(0, len(X_tensor), batch_size):
            e = min(s + batch_size, len(X_tensor))
            z = joint_enc.get_embeddings(
                graph, X_tensor[s:e], node_features,
                context_emb[s:e], temporal_seqs[s:e], temporal_mask[s:e])
            all_z.append(z.cpu())
    return torch.cat(all_z).numpy()


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | dataset: {'cicids-2018'} | checkpoints: {CHECKPOINT_DIR}"
          f"{' | SKIP_LLM' if SKIP_LLM else ''}")
    t0 = time.perf_counter()

    # Stage 1
    print("\n" + "="*60)
    print("STAGE 1: Load & Preprocess")
    print("="*60)
    # DATA_PATH must point to the NF-BoT-IoT-v3 sample. Sampling here must keep
    # time order: use the pre-made chronological sample (SAMPLE_SIZE=None).
    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    preprocessor = Preprocessor(FEATURES_PATH)
    train_balancer = FastChronologicalBalancer(allow_oversample=True) if USE_BALANCER else None
    test_balancer = FastChronologicalBalancer(
        max_total_samples=30_000, max_benign_samples=7_500,
        max_attack_per_class=3_000, allow_oversample=False) if USE_BALANCER else None
    data = preprocessor.preprocess_dataset(
        df, test_size=TEST_SIZE, gap_size=GAP_SIZE,
        train_balancer=train_balancer, test_balancer=test_balancer)

    # Full timeline + which rows are trained on (time positions)
    X_all, G_all = data['X_all'], data['G_all']
    src_all, dst_all = data['src_all'], data['dst_all']
    idx_train, idx_train_split = data['idx_train'], data['idx_train_split']

    X_train = data['X_train']
    src_train, dst_train = data['src_train'], data['dst_train']
    y_train = data['y_train']
    feature_dim = X_train.shape[1]

    # Raw rows of the training flows (same order as X_train)
    df_train_raw = data['df_raw'].iloc[idx_train].reset_index(drop=True)

    print(f"  Features: {feature_dim}, Samples: {len(X_train)}")
    print(f"  Raw DataFrame: {len(df_train_raw)} rows")
    report_ip_overlap(data)
    data_time = time.perf_counter() - t0
    print(f"Data Loading Time: {data_time:.4f}s")

    # Stage 2
    print("\n" + "="*60)
    print("STAGE 2: Graph + Temporal")
    print("="*60)
    t1 = time.perf_counter()
    builder = NetworkGraphBuilder(seq_length=SEQ_LENGTH)
    # Graph from train-period flows only (no test edges), using raw graph columns
    graph = builder.build_graph(
        G_all[idx_train_split], src_all[idx_train_split], dst_all[idx_train_split])
    graph = graph.to(device)
    node_features = builder.node_features.to(device)

    # Causal sequences: previous SEQ_LENGTH-1 flows of the same src IP, built on
    # the full (unbalanced) timeline, history restricted to the train period
    temporal_seqs, t_mask = builder.build_causal_sequences(
        X_all, src_all, idx_train, history_positions=idx_train_split)
    temporal_seqs = temporal_seqs.to(device)
    t_mask = t_mask.to(device)  # 1 = real flow, 0 = padding

    context_enc = ContextEncoder(
        NODE_FEATURE_DIM, 32, CONTEXT_OUTPUT_DIM, 0.1).to(device)
    with torch.no_grad():
        node_emb = context_enc(graph, node_features)
        context_emb = builder.get_ip_embeddings(
            node_emb, src_train, dst_train).to(device)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    graph_temporal_time = time.perf_counter() - t1
    print(f"Graph Construction Time: {graph_temporal_time:.4f}s")

    # Stage 3
    print("\n" + "="*60)
    print("STAGE 3: Compute Behaviour Scores")
    print("="*60)
    t2 = time.perf_counter()
    scorer = BehaviourScorer(feature_names=data['feature_names'])
    fan_out, fan_in, n_peers = scorer.compute_graph_features(src_train, dst_train)
    bhv_scores = scorer.compute(X_train, fan_out=fan_out, fan_in=fan_in, n_peers=n_peers,
                                tcp_flags_raw=df_train_raw['TCP_FLAGS'].values)
    bhv_targets = torch.tensor(bhv_scores, dtype=torch.float32).to(device)
    behaviour_score_time = time.perf_counter() - t2
    print(f"Compute Behaviour Scores Time: {behaviour_score_time:.4f}s")

    # Stage 4
    t3 = time.perf_counter()
    feature_enc = FeatureEncoder(feature_dim, output_dim=FEATURE_OUTPUT_DIM)
    behaviour_enc = TransformerBehaviourEncoder(
        feature_dim, 64, BEHAVIOUR_OUTPUT_DIM, 2, 4, 0.1)
    joint_enc = JointEncoder(
        feature_enc, context_enc, behaviour_enc,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, n_behaviours=scorer.n_scores, lambda_bhv=1.0)

    joint_enc = train_ae(
        joint_enc, graph, X_tensor, node_features,
        context_emb, temporal_seqs, t_mask, bhv_targets,
        VAE_EPOCHS, VAE_BATCH_SIZE, VAE_LR, device,
        y_labels=y_train)

    ve_train_time = time.perf_counter() - t3
    print(f"VE train Time: {ve_train_time:.4f}s")

    # Stage 5
    print("\n" + "="*60)
    print("STAGE 5: Extract z + DP-GMM Clustering")
    print("="*60)
    Z_train = extract_embeddings(
        joint_enc, graph, X_tensor, node_features,
        context_emb, temporal_seqs, t_mask, VAE_BATCH_SIZE, device)
    print(f"  Z_train: {Z_train.shape}")

    t4 = time.perf_counter()
    gmm = StreamingDPGMM(GMM_MAX_COMPONENTS, GMM_COVARIANCE_TYPE)
    cluster_assignments, log_liks = gmm.fit(Z_train)
    active = np.unique(cluster_assignments)
    print(f"  Active clusters: {len(active)}")
    gmm_train_time = time.perf_counter() - t4
    print(f"GMM train Time: {gmm_train_time:.4f}s")
    # plot_dpgmm_convergence(Z_train, gmm.model, os.path.join(CHECKPOINT_DIR, 'dpgmm_convergence.png'))

    for k in active:
        c = (cluster_assignments == k).sum()
        print(f"    Cluster {k}: {c:,} flows (w={gmm.model.weights_[k]:.4f})")

    # Stage 6: Benign detection (size + feature based)
    print("\n" + "="*60)
    print("STAGE 6: Benign Detection")
    print("="*60)
    t5 = time.perf_counter()
    total_flows = len(X_train)
    benign_clusters = set()
    feature_names_list = data['feature_names']
    fn_idx = {n: i for i, n in enumerate(feature_names_list)}
    from src.mapping.soft_cluster_scorer import SoftClusterScorer
    benign_rule_scorer = SoftClusterScorer(dataset=SCORER_DATASET)
    BENIGN_RULE_MIN = 0.6  # share of benign rule weight a cluster must satisfy
    TTL_OTHER_MIN = 0.5    # share of flows with TTL not in {0, 64}
    WIN1024_MAX = 0.35     # max share of flows with the Nmap probe window (1024)
    FAN_OUT_MAX = 30.5     # Exfiltration threshold from V3_RULES (train split)
    # fan-out per source IP over the training flows (unique destinations)
    import pandas as pd
    _ip = pd.DataFrame({'s': np.asarray(src_train), 'd': np.asarray(dst_train)})
    flow_fan_out = _ip['s'].map(_ip.groupby('s')['d'].nunique()).to_numpy()

    # benign_total = (y_train == 'Benign').sum()
    # print(f"\n  Benign flows in training: {benign_total:,} ({benign_total / len(y_train) * 100:.1f}%)")
    for k in active:
        # n = (y_train[cluster_assignments == k] == 'Benign').sum()
        # if n > 0:
        #     print(f"    Cluster {k:2d}: {n:,} benign ({n / benign_total * 100:.1f}% of all benign)")
        mask = cluster_assignments == k
        proportion = mask.sum() / total_flows
        cluster_scaled = X_train[mask]
        global_mean = X_train.mean(axis=0)
        deviation = np.abs(cluster_scaled.mean(axis=0) - global_mean).mean()

        # Get raw-scale key features
        cluster_mean = cluster_scaled.mean(axis=0)
        min_ttl = None
        tcp_flags = None
        if 'MIN_TTL' in fn_idx:
            min_ttl = (cluster_mean[fn_idx['MIN_TTL']] *
                       data['scaler'].scale_[fn_idx['MIN_TTL']] +
                       data['scaler'].mean_[fn_idx['MIN_TTL']])
        if 'TCP_FLAGS' in fn_idx:
            tcp_flags = (cluster_mean[fn_idx['TCP_FLAGS']] *
                         data['scaler'].scale_[fn_idx['TCP_FLAGS']] +
                         data['scaler'].mean_[fn_idx['TCP_FLAGS']])

        is_benign = False
        reason = ""

        # v3 rule: same Fisher-weighted benign rules as SoftClusterScorer
        # (bot_iot_v3), on RAW cluster means (the scaled+log inverse is not raw).
        raw = df_train_raw[mask]
        def _m(col):
            return float(raw[col].mean()) if col in raw.columns else None
        bhv_raw = {'avg_iat_max': _m('SRC_TO_DST_IAT_MAX'),
                   'avg_iat_avg': _m('SRC_TO_DST_IAT_AVG'),
                   'avg_iat_std': _m('SRC_TO_DST_IAT_STDDEV'),
                   'avg_tcp_win_in': _m('TCP_WIN_MAX_IN'),
                   'avg_longest_pkt_bytes': _m('LONGEST_FLOW_PKT'),
                   'avg_iat_in_avg': _m('DST_TO_SRC_IAT_AVG'),
                   'avg_iat_in_std': _m('DST_TO_SRC_IAT_STDDEV'),
                   **flow_ratio_means(raw)}
        v3_scores = benign_rule_scorer._score_botiot_v3(
            bhv_raw, fan_out=None, out_bytes=None,
            min_ttl=_m('MIN_TTL'), tcp_flags=_m('TCP_FLAGS'), pps=None,
            dir_ratio=None, duration=_m('FLOW_DURATION_MILLISECONDS'),
            dst_port=_m('L4_DST_PORT'))
        benign_s = v3_scores.get('normal_activity', 0.0)
        best_attack = max((v for b_, v in v3_scores.items() if b_ != 'normal_activity'),
                          default=0.0)
        min_ttl, tcp_flags = _m('MIN_TTL'), _m('TCP_FLAGS')

        # TTL fingerprint: BoT-IoT attackers send with initial TTL 64 (Linux),
        # so a cluster whose flows mostly have a TTL other than 0 or 64 comes
        # from other hosts. Uses the share of flows, not the mean.
        ttl_raw = raw['MIN_TTL'] if 'MIN_TTL' in raw.columns else None
        ttl_other = (float((~ttl_raw.isin([0, 64])).mean())
                     if ttl_raw is not None else 0.0)

        win1024 = (float((raw['TCP_WIN_MAX_IN'] == 1024).mean())
                   if 'TCP_WIN_MAX_IN' in raw.columns else 1.0)
        syn_only = (float((raw['TCP_FLAGS'] == 2).mean())
                    if 'TCP_FLAGS' in raw.columns else 0.0)
        fan_out_c = float(flow_fan_out[mask].mean())

        if benign_s >= BENIGN_RULE_MIN and benign_s > best_attack and mask.sum() > 100:
            is_benign = True
            reason = f"v3 benign score={benign_s:.2f} > attack={best_attack:.2f}"
        elif 'cicids-2018' != 'bot_iot':
            pass  # TTL/window fingerprints below are BoT-IoT specific
        elif ttl_other >= TTL_OTHER_MIN:
            is_benign = True
            reason = f"TTL fingerprint: {ttl_other:.0%} of flows TTL not 0/64"
        elif (ttl_raw is not None and float((ttl_raw == 0).mean()) >= 0.5
              and win1024 < WIN1024_MAX
              and syn_only < 0.5              # SYN-only = DoS/DDoS probes
              and fan_out_c < FAN_OUT_MAX):   # wide fan-out = Theft
            # TTL-0 flows are both benign and scan probes; Nmap SYN probes
            # carry a TCP window of exactly 1024, benign TTL-0 flows do not
            is_benign = True
            reason = f"TTL 0 without scan window (win1024={win1024:.0%})"

        # ── DIAGNOSTIC ──
        n_benign = (y_train[mask] == 'Benign').sum()  # check your label string
        benign_pct = n_benign / mask.sum() * 100
        if min_ttl is not None and tcp_flags is not None:
            print(f"  C{k:2d}: n={mask.sum():>6,} benign={benign_pct:5.1f}% | "
                  f"TTL={min_ttl:7.2f} TCP={tcp_flags:6.2f} | "
                  f"benign_s={benign_s:.2f} attack_s={best_attack:.2f} "
                  f"ttl_other={ttl_other:.2f} win1024={win1024:.2f} "
                  f"syn={syn_only:.2f} fan_out={fan_out_c:.1f} | "
                  f"prop={proportion:.3f}")
        if is_benign:
            benign_clusters.add(int(k))
            print(f"  Cluster {k}: {mask.sum():,} ({proportion:.1%}) "
                  f"— BENIGN ({reason})")

    print(f"  Total benign clusters: {len(benign_clusters)}")
    print("\n  Rule attribution:")
    for k in sorted(benign_clusters):
        mask = cluster_assignments == k
        n_benign = (y_train[mask] == 'Benign').sum()
        print(f"    C{k:2d}: {n_benign:>5,}/{mask.sum():>6,} benign "
              f"({n_benign / mask.sum() * 100:5.1f}%)")

    # Stage 7: Summarize + LLM behaviour labeling
    print("\n" + "="*60)
    print("STAGE 7: Cluster Summarization + LLM Behaviour Labeling")
    print("="*60)
    summarizer = ClusterSummarizer(
        feature_names=data["feature_names"], scaler=data["scaler"], dataset="bot_iot")
    cluster_summaries = summarizer.summarize_all_clusters(
        cluster_assignments, Z_train, X_train,
        src_ips=src_train, dst_ips=dst_train,
        df_raw=df_train_raw)
    # Same fan-out as the Fisher analysis and Stage 6: unique destinations of
    # each flow's source IP over ALL training flows (the summarizer's value
    # only counts destinations inside the cluster)
    for k in cluster_summaries:
        m = cluster_assignments == int(k)
        if m.any():
            gp = cluster_summaries[k].setdefault('stats', {}).setdefault('graph_properties', {})
            gp['avg_fan_out_cluster'] = gp.get('avg_fan_out')
            gp['avg_fan_out'] = float(flow_fan_out[m].mean())

    labeler = LLMLabeler(api_key=None if SKIP_LLM else OPENAI_API_KEY, model=LLM_MODEL)

    from src.llm.reasoning import build_reference_profiles
    from src.evaluation.ground_truth import GroundTruthMapper
    train_tactics = GroundTruthMapper(GT_DATASET).map_labels(y_train)
    labeler.reference_profiles = build_reference_profiles(
            df_train_raw, train_tactics, fan_out=flow_fan_out)
    print("  LLM reference profiles (training split):\n" + labeler.reference_profiles)

    cluster_weights = {k: float(gmm.model.weights_[k]) for k in active}

    # ══════════════════════════════════════════════════════
    # Cluster Behaviour Labeling (single LLM call per cluster)
    # ══════════════════════════════════════════════════════
    from src.mapping.behaviour_mapping import get_primary_tactic
    from src.mapping.soft_cluster_scorer import (
        SoftClusterScorer, get_behaviour_tactic_weights)
    BEHAVIOUR_TACTIC_WEIGHTS = get_behaviour_tactic_weights(SCORER_DATASET)

    cluster_behaviour_map_hard = {}
    # The LLM is called whenever a client exists; its scores are saved
    # SEPARATELY in cluster_behaviour_map_hard (*_llm keys) for the weighted
    # fusion at inference. LABEL_MODE only changes the training-time KB:
    #   'rules' (default, use this with inference fusion), or 'llm' /
    #   'combined' for a quick training-time check only - otherwise the
    #   LLM would be counted twice at inference.
    LABEL_MODE = os.getenv('LABEL_MODE', 'rules')
    has_llm = labeler.client is not None
    if LABEL_MODE in ('llm', 'combined') and not has_llm:
        print(f"  LABEL_MODE={LABEL_MODE} but no LLM client: falling back to 'rules'")
        LABEL_MODE = 'rules'
    llm_soft = {}  # cluster -> LLM behaviour scores (scorer names)

    for k in active:
        k = int(k)

        if k in benign_clusters:
            # Auto-label benign clusters
            cluster_behaviour_map_hard[k] = {
                'primary_behaviour': 'normal_activity',
                'confidence': 0.95,
                'reasoning': 'Auto-detected benign',
                'behaviour_scores_llm': {'normal_activity': 0.95},
                'tactic_scores_llm': {'None': 0.95},
                'primary_tactic': 'None',
            }
            print(f"  Cluster {k}: [normal_activity] (auto-detected benign)")

        elif has_llm:
            # ONE LLM call → soft behaviour scores → derive everything
            summary = cluster_summaries.get(k, {})
            t6 = time.perf_counter()
            soft_result = labeler.label_cluster_soft(
                summary, cluster_weights.get(k, 0))
            llm_api_time = time.perf_counter() - t6
            print(llm_api_time)

            if soft_result and soft_result.get('behaviour_scores'):
                bhv_scores = soft_result['behaviour_scores']
                llm_soft[k] = bhv_scores
                primary_bhv = max(bhv_scores, key=bhv_scores.get)
                confidence = bhv_scores[primary_bhv]

                # Map behaviour scores → tactic scores
                tactic_scores = defaultdict(float)
                for bhv, score in bhv_scores.items():
                    if bhv in BEHAVIOUR_TACTIC_WEIGHTS:
                        for tac, w in BEHAVIOUR_TACTIC_WEIGHTS[bhv].items():
                            tactic_scores[tac] += score * w
                total_ts = sum(tactic_scores.values())
                if total_ts > 0:
                    tactic_scores = {t: s/total_ts
                                     for t, s in tactic_scores.items()}

                primary_tactic = max(tactic_scores, key=tactic_scores.get)

                cluster_behaviour_map_hard[k] = {
                    'primary_behaviour': primary_bhv,
                    'confidence': confidence,
                    'reasoning': soft_result.get('reasoning', ''),
                    'behaviour_scores_llm': bhv_scores,
                    'tactic_scores_llm': dict(tactic_scores),
                    'primary_tactic': primary_tactic,
                }
                top3 = sorted(bhv_scores.items(), key=lambda x: -x[1])[:3]
                score_str = ', '.join(f"{b}:{s:.2f}" for b, s in top3)
                print(f"  Cluster {k}: [{score_str}] → {primary_tactic} (LLM)")
            else:
                # LLM call failed — fallback to mock
                result = labeler._mock_label(cluster_summaries.get(k, {}))
                primary_tactic = get_primary_tactic(result['primary_behaviour'])
                result['primary_tactic'] = primary_tactic
                result['behaviour_scores_llm'] = {
                    result['primary_behaviour']: result['confidence']}
                result['tactic_scores_llm'] = {primary_tactic: result['confidence']}
                cluster_behaviour_map_hard[k] = result
                print(f"  Cluster {k}: [{result['primary_behaviour']}] "
                      f"→ {primary_tactic} (LLM failed, mock fallback)")

        else:
            # No LLM — use mock decision tree for hard labels
            summary = cluster_summaries.get(k, {})
            result = labeler._mock_label(summary)
            primary_bhv = result['primary_behaviour']
            conf = result['confidence']
            primary_tactic = get_primary_tactic(primary_bhv)

            # Derive soft distribution from mock confidence
            other_tactics = ['Impact', 'Reconnaissance', 'Exfiltration',
                             'None', 'Command and Control', 'Discovery']
            tactic_dist = {primary_tactic: conf}
            remaining = 1.0 - conf
            non_primary = [t for t in other_tactics if t != primary_tactic]
            for t in non_primary:
                tactic_dist[t] = remaining / max(len(non_primary), 1)

            result['primary_tactic'] = primary_tactic
            result['behaviour_scores_llm'] = {primary_bhv: conf}
            result['tactic_scores_llm'] = tactic_dist
            cluster_behaviour_map_hard[k] = result
            print(f"  Cluster {k}: [{primary_bhv}] conf={conf:.2f} "
                  f"→ {primary_tactic} (mock)")

    print(f"\n  Labeling: {'LLM' if has_llm else 'Mock'} mode, "
          f"{len(cluster_behaviour_map_hard)} clusters labeled")
    total_labeling_time = time.perf_counter() - t5
    print(f"LLM/Mock Labeling + Summarization Time: {total_labeling_time:.4f}s")

    # Build KB with rule-based soft cluster scoring (separate from LLM)
    soft_scorer = SoftClusterScorer(dataset=SCORER_DATASET)
    cluster_scores = soft_scorer.score_all_clusters(cluster_summaries)
    # Combine with the LLM's soft scores (benign override below still wins)
    for k, llm_bhv in (llm_soft.items() if LABEL_MODE != 'rules' else []):
        rule_bhv = cluster_scores.get(k, {}).get('behaviour_scores', {})
        if LABEL_MODE == 'llm':
            bhv = dict(llm_bhv)
        else:  # combined
            keys = set(rule_bhv) | set(llm_bhv)
            bhv = {b: 0.5 * rule_bhv.get(b, 0.0) + 0.5 * llm_bhv.get(b, 0.0) for b in keys}
        tot = sum(bhv.values()) or 1.0
        bhv = {b: v / tot for b, v in bhv.items()}
        tac = soft_scorer._map_to_tactics(bhv)
        cluster_scores[k] = {
            'behaviour_scores': bhv,
            'tactic_scores': tac,
            'primary_behaviour': max(bhv, key=bhv.get),
            'primary_tactic': max(tac, key=tac.get),
        }
    print(f"  Cluster labelling mode: {LABEL_MODE} "
          f"({len(llm_soft)} clusters with LLM scores)")
    # Stage 6 benign decision overrides the soft scorer (it uses the TTL
    # fingerprint, which the scorer's mean-based rules cannot see)
    for k in benign_clusters:
        cluster_scores[int(k)] = {
            'behaviour_scores': {'normal_activity': 1.0},
            'tactic_scores': soft_scorer._map_to_tactics({'normal_activity': 1.0}),
            'primary_behaviour': 'normal_activity',
            'primary_tactic': 'None',
        }

    cluster_behaviour_map_soft = {}
    kb = KnowledgeBase()
    for k in active:
        k = int(k)
        cs = cluster_scores.get(k, {})
        mask = cluster_assignments == k
        size = int(mask.sum())

        # Extract raw features for KB storage
        summary = cluster_summaries.get(k, {})
        bhv_stats = summary.get('stats', {}).get('behaviour', {})
        features = {
            'tcp_flags': bhv_stats.get('avg_tcp_flags', 0),
            'out_bytes': bhv_stats.get('avg_out_bytes', 0),
            'min_ttl': bhv_stats.get('avg_min_ttl', 0),
            'packets_per_second': bhv_stats.get('packets_per_second', 0),
            'direction_ratio': bhv_stats.get('throughput_ratio', 0),
        }
        graph_props = summary.get('stats', {}).get('graph_properties', {})
        features['fan_out'] = graph_props.get('avg_fan_out', 0)
        features['fan_in'] = graph_props.get('avg_fan_in', 0)

        kb.add_from_soft_scorer(
            cluster_idx=k,
            scorer_result=cs,
            cluster_size=size,
            cluster_proportion=size / len(cluster_assignments),
            p_gmm=cluster_weights.get(k, 0),
            cluster_features=features)

        cluster_behaviour_map_soft[k] = {
            'primary_behaviour': cs.get('primary_behaviour', 'unknown'),
            'behaviours': [{'label': cs.get('primary_behaviour', 'unknown')}],
            'behaviour_scores': cs.get('behaviour_scores', {}),
            'tactic_scores': cs.get('tactic_scores', {}),
            'confidence': max(cs.get('behaviour_scores', {}).values(), default=0.5),
        }

    print(f"\n  KB: {len(kb.entries)} entries")
    print(kb)

    # Purity check
    if y_train is not None:
        from src.evaluation.ground_truth import GroundTruthMapper
        gt_mapper = GroundTruthMapper(GT_DATASET)
        print(f"\n  Per-cluster ground truth:")
        purities = []
        sizes = []
        for k in active:
            mask = cluster_assignments == k
            size = mask.sum()
            maj = Counter(y_train[mask]).most_common(1)[0]
            purity = maj[1] / size
            purities.append(purity)
            sizes.append(size)
            bhv = cluster_behaviour_map_soft.get(int(k), {}).get('primary_behaviour', '?')
            match = "✓" if kb.get_tactic(int(k)) == gt_mapper.get_tactic(maj[0]) else "✗"
            print(f"    Cluster {k:2d}: {size:>6,} | "
                  f"[{bhv:25s}] | True: {maj[0]:15s} ({purity*100:.0f}%) {match}")

        mean_purity = sum(purities) / len(purities)
        weighted_purity = sum(p * s for p, s in zip(purities, sizes)) / sum(sizes)
        print(f"\n    Mean cluster purity:     {mean_purity:.4f}")
        print(f"    Weighted cluster purity: {weighted_purity:.4f}")

    # Stage 8: Save
    print("\n" + "="*60)
    print("STAGE 8: Save Checkpoint")
    print("="*60)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint = {
        'joint_encoder_state': joint_enc.state_dict(),
        'gmm_model': gmm.model,
        'gmm_config': {
            'max_components': gmm.max_components,
            'covariance_type': gmm.covariance_type,
            'weight_threshold': gmm.weight_threshold,
            'novelty_threshold': gmm.novelty_threshold,
            'll_stats': getattr(gmm, '_ll_stats', {}),
        },
        'preprocessor_scaler': data['scaler'],
        'feature_names': data['feature_names'],
        'ip_to_idx': builder.ip_to_idx,
        # Training statistics reused at test time (no test statistics)
        'node_scaler': {'mean': builder.node_mean, 'std': builder.node_std},
        'behaviour_scorer_stats': scorer.stats,
        # Leakage-safe split: evaluation must use exactly these test rows and
        # build test context from past flows only
        'preprocessing': {
            'data_path': DATA_PATH,
            'feature_config_path': FEATURES_PATH,
            'test_size': TEST_SIZE,
            'gap_size': GAP_SIZE,
            'use_balancer': USE_BALANCER,
            'idx_train': data['idx_train'],
            'idx_train_split': data['idx_train_split'],
            'idx_test': data['idx_test'],
            'n_duplicates_removed': data['n_duplicates_removed'],
        },
        'cluster_behaviour_map': cluster_behaviour_map_soft,
        'cluster_behaviour_map_hard': cluster_behaviour_map_hard,
        'cluster_behaviour_map_soft': cluster_behaviour_map_soft,
        'config': {
            'feature_dim': feature_dim,
            'feature_output_dim': FEATURE_OUTPUT_DIM,
            'context_output_dim': CONTEXT_OUTPUT_DIM,
            'behaviour_output_dim': BEHAVIOUR_OUTPUT_DIM,
            'z_dim': Z_DIM,
            'seq_length': SEQ_LENGTH,
            'node_feature_dim': NODE_FEATURE_DIM,
            'n_behaviours': scorer.n_scores,
            'lambda_bhv': 1.0,
        },
    }
    torch.save(checkpoint, MODEL_PATH)
    kb.save(KB_PATH)
    print(f"  Saved: {MODEL_PATH}, {KB_PATH}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    t = time.perf_counter()
    main()
    time_taken = time.perf_counter() - t
    print("Total time taken: ", time_taken)
