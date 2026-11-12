"""
Training Pipeline — UNSW-NB15 (Behaviour-Aware)
=====================================

Stage 1: Load & preprocess
Stage 2: Graph + temporal sequences
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

from src.config.config_unsw import (
    DATA_PATH, SAMPLE_SIZE,
    FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
    Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
    VAE_EPOCHS, VAE_BATCH_SIZE, VAE_LR,
    GMM_MAX_COMPONENTS, GMM_COVARIANCE_TYPE,
    DEFAULT_W1_GMM, DEFAULT_W2_LLM,
    OPENAI_API_KEY, LLM_MODEL,
    CHECKPOINT_DIR, KB_PATH, MODEL_PATH,
    BENIGN_RATIO, MIN_SAMPLES_PER_CLASS, MAX_TOTAL_SAMPLES,
    MAX_BENIGN_SAMPLES, MAX_ATTACK_PER_CLASS, MIN_ATTACK_PER_CLASS,
)
from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
from src.encoder.behaviour_scorer import BehaviourScorer
from src.streaming.streaming_gmm import StreamingDPGMM
from src.clustering.summarizer import ClusterSummarizer
from src.llm.reasoning import LLMLabeler
from src.knowledge_base.kb import KnowledgeBase
from src.utils.training_logger import TrainingLogger, compute_grad_norms
from src.data.balancer import FastChronologicalBalancer


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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Stage 1
    print("\n" + "="*60)
    print("STAGE 1: Load & Preprocess")
    print("="*60)
    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)

    # Balance UNSW-NB15 (96% benign → 2:1 ratio)
    if 'Attack' in df.columns:
        balancer = FastChronologicalBalancer(
            attack_col='Attack',
            benign_label='Benign',
            max_total_samples=MAX_TOTAL_SAMPLES,
            max_benign_samples=MAX_BENIGN_SAMPLES,
            max_attack_per_class=MAX_ATTACK_PER_CLASS,
            min_attack_per_class=MIN_ATTACK_PER_CLASS,
        )
        df, balance_stats = balancer.balance(df)
        print(f"  Balanced: {len(df):,} flows")
        for attack, count in df['Attack'].value_counts().items():
            print(f"    {attack:20s}: {count:>7,}")

    preprocessor = Preprocessor(clip_extremes=True)
    data = preprocessor.preprocess_dataset(df)
    X_train = data['X_train']
    src_train, dst_train = data['src_train'], data['dst_train']
    y_train = data['y_train']
    feature_dim = X_train.shape[1]
    train_indices = data.get('train_indices', None)

    # Get raw DataFrame for training flows (before preprocessing)
    if train_indices is not None:
        df_train_raw = df.iloc[train_indices].reset_index(drop=True)
    else:
        # Fallback: use first N rows
        df_train_raw = df.iloc[:len(X_train)].reset_index(drop=True)

    print(f"  Features: {feature_dim}, Samples: {len(X_train)}")
    print(f"  Raw DataFrame: {len(df_train_raw)} rows")

    # Stage 2
    print("\n" + "="*60)
    print("STAGE 2: Graph + Temporal")
    print("="*60)
    builder = NetworkGraphBuilder(seq_length=SEQ_LENGTH)
    graph, temporal_seqs = builder.build_graph(X_train, src_train, dst_train)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    context_enc = ContextEncoder(
        NODE_FEATURE_DIM, 32, CONTEXT_OUTPUT_DIM, 0.1).to(device)
    with torch.no_grad():
        node_emb = context_enc(graph, node_features)
        context_emb = builder.get_ip_embeddings(
            node_emb, src_train, dst_train).to(device)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    t_mask = torch.ones(len(X_train), temporal_seqs.shape[1]).to(device)

    # Stage 3
    print("\n" + "="*60)
    print("STAGE 3: Compute Behaviour Scores")
    print("="*60)
    scorer = BehaviourScorer(feature_names=data['feature_names'])
    fan_out, fan_in, n_peers = scorer.compute_graph_features(src_train, dst_train)
    bhv_scores = scorer.compute(X_train, fan_out=fan_out, fan_in=fan_in, n_peers=n_peers)
    bhv_targets = torch.tensor(bhv_scores, dtype=torch.float32).to(device)

    # Stage 4
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

    # Stage 5
    print("\n" + "="*60)
    print("STAGE 5: Extract z + DP-GMM Clustering")
    print("="*60)
    Z_train = extract_embeddings(
        joint_enc, graph, X_tensor, node_features,
        context_emb, temporal_seqs, t_mask, VAE_BATCH_SIZE, device)
    print(f"  Z_train: {Z_train.shape}")

    gmm = StreamingDPGMM(GMM_MAX_COMPONENTS, GMM_COVARIANCE_TYPE)
    cluster_assignments, log_liks = gmm.fit(Z_train)
    active = np.unique(cluster_assignments)
    print(f"  Active clusters: {len(active)}")

    for k in active:
        c = (cluster_assignments == k).sum()
        print(f"    Cluster {k}: {c:,} flows (w={gmm.model.weights_[k]:.4f})")

    # Stage 6: Benign detection (size + feature based)
    print("\n" + "="*60)
    print("STAGE 6: Benign Detection")
    print("="*60)
    total_flows = len(X_train)
    benign_clusters = set()
    feature_names_list = data['feature_names']
    fn_idx = {n: i for i, n in enumerate(feature_names_list)}

    for k in active:
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

        # Rule 1: Any cluster >10% of data with normal TCP flags → benign
        if proportion > 0.10 and tcp_flags is not None and tcp_flags > 8:
            is_benign = True
            reason = f"large ({proportion:.1%}) + normal TCP={tcp_flags:.1f}"

        # Rule 2: Proportion >5% + low deviation + normal TCP
        elif (proportion > 0.05 and deviation < 0.5 and
              tcp_flags is not None and tcp_flags > 8):
            is_benign = True
            reason = f"moderate ({proportion:.1%}) + low dev={deviation:.3f}"

        # Rule 3: Low MIN_TTL (<30) + normal TCP (>10) + not tiny
        elif (min_ttl is not None and min_ttl < 30 and
              tcp_flags is not None and tcp_flags > 10 and
              mask.sum() > 100):
            is_benign = True
            reason = f"low TTL={min_ttl:.1f} + normal TCP={tcp_flags:.1f}"

        if is_benign:
            benign_clusters.add(int(k))
            print(f"  Cluster {k}: {mask.sum():,} ({proportion:.1%}) "
                  f"— BENIGN ({reason})")

    print(f"  Total benign clusters: {len(benign_clusters)}")

    # Stage 7: Summarize + LLM behaviour labeling
    print("\n" + "="*60)
    print("STAGE 7: Cluster Summarization + LLM Behaviour Labeling")
    print("="*60)
    summarizer = ClusterSummarizer(
        feature_names=data['feature_names'], scaler=data['scaler'])
    cluster_summaries = summarizer.summarize_all_clusters(
        cluster_assignments, Z_train, X_train,
        src_ips=src_train, dst_ips=dst_train,
        df_raw=df_train_raw)

    labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)
    cluster_weights = {k: float(gmm.model.weights_[k]) for k in active}

    # ══════════════════════════════════════════════════════
    # Cluster Behaviour Labeling (single LLM call per cluster)
    # ══════════════════════════════════════════════════════
    from src.mapping.behaviour_mapping import get_primary_tactic
    from src.mapping.soft_cluster_scorer import (
        SoftClusterScorer, BEHAVIOUR_TACTIC_WEIGHTS)

    cluster_behaviour_map_hard = {}
    has_llm = labeler.client is not None

    for k in active:
        k = int(k)

        if k in benign_clusters:
            # Auto-label benign clusters
            cluster_behaviour_map_hard[k] = {
                'primary_behaviour': 'normal_browsing',
                'confidence': 0.95,
                'reasoning': 'Auto-detected benign',
                'behaviour_scores_llm': {'normal_browsing': 0.95},
                'tactic_scores_llm': {'None': 0.95},
                'primary_tactic': 'None',
            }
            print(f"  Cluster {k}: [normal_browsing] (auto-detected benign)")

        elif has_llm:
            # ONE LLM call → soft behaviour scores → derive everything
            summary = cluster_summaries.get(k, {})
            soft_result = labeler.label_cluster_soft(
                summary, cluster_weights.get(k, 0))

            if soft_result and soft_result.get('behaviour_scores'):
                bhv_scores = soft_result['behaviour_scores']
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

    # Build KB with rule-based soft cluster scoring (separate from LLM)
    soft_scorer = SoftClusterScorer(dataset="unsw_nb15")
    cluster_scores = soft_scorer.score_all_clusters(cluster_summaries)

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
        gt_mapper = GroundTruthMapper('unsw_nb15')
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
    main()
# """
# Cyber Threat Detection System — Training Pipeline (NF-UNSW-NB15-v2)
# =====================================================================
#
# Same architecture as NF-BoT-IoT-v2 pipeline (same 43 NetFlow features).
# Key differences handled here:
#   - 9 attack types + benign (vs BoT-IoT's 4 attack types)
#   - Heavy class imbalance: ~87% benign, ~13% attack
#   - Stratified sampling to preserve minority attack classes
#   - GMM max components raised to 25
#
# Stage 1: Load & preprocess network flow data
# Stage 2: Graph construction + temporal sequences
# Stage 3: Three-track encoder + VAE (TRAINED)
# Stage 4: Fit DP-GMM (unsupervised) + Cluster Summarization
# Stage 5: LLM Semantic Labeling -> Knowledge Base
# Stage 6: Bayesian Fusion Setup
# Stage 7: Save Model Checkpoint for Inference
# """
#
# import os
# import numpy as np
# import torch
# from collections import Counter
#
# from src.config.config_unsw import (
#     DATASET_NAME, DATA_PATH, SAMPLE_SIZE, ATTACK_TYPES, ATTACK_TO_MITRE,
#     BENIGN_RATIO, MIN_SAMPLES_PER_CLASS, MAX_TOTAL_SAMPLES,
#     FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
#     Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM,
#     VAE_EPOCHS, VAE_BATCH_SIZE, VAE_LR, VAE_BETA, VAE_BETA_WARMUP,
#     GMM_MAX_COMPONENTS, GMM_COVARIANCE_TYPE,
#     DEFAULT_W1_GMM, DEFAULT_W2_LLM,
#     OPENAI_API_KEY, LLM_MODEL,
#     CHECKPOINT_DIR, KB_PATH, MODEL_PATH,
# )
# from src.data.loader import load_dataset
# from src.data.preprocess import Preprocessor
# from src.data.balancer import ChronologicalBalancer
# from src.data.graph_builder import NetworkGraphBuilder
# from src.encoder.feature import FeatureEncoder
# from src.encoder.context import ContextEncoder
# from src.encoder.behaviour import TransformerBehaviourEncoder
# from src.encoder.jointencoder import JointEncoder
# from src.streaming.streaming_gmm import StreamingDPGMM
# from src.clustering.summarizer import ClusterSummarizer
# from src.llm.reasoning import LLMLabeler
# from src.knowledge_base.kb import KnowledgeBase
# from src.fusion.fusion import BayesianFusion
#
#
# def train_vae(joint_enc, graph, X_train_tensor, node_features,
#               context_embeddings, temporal_seqs, temporal_mask,
#               epochs, batch_size, lr, beta_max, beta_warmup, device):
#     """Stage 3: Train the VAE with reconstruction + KL loss."""
#     print("\n" + "="*60)
#     print("STAGE 3: Training VAE (Three-Track Encoder)")
#     print("="*60)
#
#     joint_enc.to(device)
#     joint_enc.train()
#
#     optimizer = torch.optim.Adam(joint_enc.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#
#     n_samples = len(X_train_tensor)
#     n_batches = (n_samples + batch_size - 1) // batch_size
#
#     for epoch in range(epochs):
#         if epoch < beta_warmup:
#             current_beta = beta_max * (epoch / beta_warmup)
#         else:
#             current_beta = beta_max
#         joint_enc.beta = current_beta
#
#         perm = torch.randperm(n_samples)
#         epoch_loss = 0.0
#         epoch_recon = 0.0
#         epoch_kl = 0.0
#
#         for batch_idx in range(n_batches):
#             start = batch_idx * batch_size
#             end = min(start + batch_size, n_samples)
#             batch_perm = perm[start:end]
#
#             batch_features = X_train_tensor[batch_perm]
#             batch_context = context_embeddings[batch_perm]
#             batch_temporal = temporal_seqs[batch_perm]
#             batch_mask = temporal_mask[batch_perm]
#
#             z, mu, log_var, h_fusion, h_recon = joint_enc(
#                 graph=graph,
#                 flow_features=batch_features,
#                 node_features=node_features,
#                 context_embeddings=batch_context,
#                 temporal_sequences=batch_temporal,
#                 temporal_mask=batch_mask,
#             )
#
#             loss, recon_loss, kl_loss = joint_enc.compute_loss(
#                 h_fusion, h_recon, mu, log_var
#             )
#
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(joint_enc.parameters(), max_norm=1.0)
#             optimizer.step()
#
#             epoch_loss += loss.item()
#             epoch_recon += recon_loss.item()
#             epoch_kl += kl_loss.item()
#
#         scheduler.step()
#
#         avg_loss = epoch_loss / n_batches
#         avg_recon = epoch_recon / n_batches
#         avg_kl = epoch_kl / n_batches
#
#         if (epoch + 1) % 5 == 0 or epoch == 0:
#             print(f"  Epoch {epoch+1:3d}/{epochs}  |  "
#                   f"Loss: {avg_loss:.4f}  |  "
#                   f"Recon: {avg_recon:.4f}  |  "
#                   f"KL: {avg_kl:.4f}  |  "
#                   f"Beta: {current_beta:.3f}  |  "
#                   f"LR: {scheduler.get_last_lr()[0]:.6f}")
#
#     print(f"  VAE training complete. Final loss: {avg_loss:.4f}")
#     return joint_enc
#
#
# def extract_embeddings(joint_enc, graph, X_tensor, node_features,
#                        context_embeddings, temporal_seqs, temporal_mask,
#                        batch_size, device):
#     """Extract latent embeddings Z using trained encoder (mu, no noise)."""
#     joint_enc.eval()
#     joint_enc.to(device)
#
#     n_samples = len(X_tensor)
#     all_z = []
#
#     with torch.no_grad():
#         for start in range(0, n_samples, batch_size):
#             end = min(start + batch_size, n_samples)
#             z = joint_enc.get_embeddings(
#                 graph=graph,
#                 flow_features=X_tensor[start:end],
#                 node_features=node_features,
#                 context_embeddings=context_embeddings[start:end],
#                 temporal_sequences=temporal_seqs[start:end],
#                 temporal_mask=temporal_mask[start:end],
#             )
#             all_z.append(z.cpu())
#
#     Z = torch.cat(all_z, dim=0).numpy()
#     print(f"  Extracted embeddings: {Z.shape}")
#     return Z
#
#
# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Device: {device}")
#     print(f"Dataset: {DATASET_NAME}")
#
#     # ═══════════════════════════════════════════════════════════
#     # Stage 1: Load & Preprocess Data
#     # ═══════════════════════════════════════════════════════════
#     print("\n" + "="*60)
#     print(f"STAGE 1: Load & Preprocess Data ({DATASET_NAME})")
#     print("="*60)
#
#     df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
#
#     # ── Dataset-specific: show attack distribution ──
#     if 'Attack' in df.columns:
#         print(f"\n  Original attack distribution:")
#         for attack, count in df['Attack'].value_counts().items():
#             pct = count / len(df) * 100
#             print(f"    {attack:20s} {count:>10,} ({pct:5.1f}%)")
#
#     # ── Chronological balancing ──
#     # Keep all attacks, stride-sample benign to 2:1 ratio,
#     # boost rare classes (Worms: 164 -> 200)
#     balancer = ChronologicalBalancer(
#         benign_ratio=BENIGN_RATIO,
#         min_samples_per_class=MIN_SAMPLES_PER_CLASS,
#         max_total_samples=MAX_TOTAL_SAMPLES,
#         benign_label='Benign',
#     )
#     df, balance_stats = balancer.balance_with_minority_boost(df)
#
#     print(f"\n  After balancing:")
#     print(f"    {balance_stats['balanced_total']:,} flows "
#           f"(was {balance_stats['original_total']:,}, "
#           f"{balance_stats['reduction_factor']:.1f}x reduction)")
#     for attack, count in df['Attack'].value_counts().items():
#         pct = count / len(df) * 100
#         print(f"    {attack:20s} {count:>10,} ({pct:5.1f}%)")
#
#     preprocessor = Preprocessor()
#     data = preprocessor.preprocess_dataset(df)
#
#     X_train = data['X_train']
#     src_train, dst_train = data['src_train'], data['dst_train']
#     y_train = data['y_train']
#     feature_dim = X_train.shape[1]
#
#     print(f"\n  Features: {feature_dim}, Training samples: {len(X_train)}")
#
#     if y_train is not None:
#         print(f"  Training attack distribution:")
#         for attack, count in Counter(y_train).most_common():
#             pct = count / len(y_train) * 100
#             print(f"    {attack:20s} {count:>10,} ({pct:5.1f}%)")
#
#     # ═══════════════════════════════════════════════════════════
#     # Stage 2: Graph Construction + Temporal Sequences
#     # ═══════════════════════════════════════════════════════════
#     print("\n" + "="*60)
#     print("STAGE 2: Graph Construction + Temporal Sequences")
#     print("="*60)
#
#     builder = NetworkGraphBuilder(seq_length=SEQ_LENGTH)
#     graph, temporal_seqs = builder.build_graph(X_train, src_train, dst_train)
#
#     graph = graph.to(device)
#     temporal_seqs = temporal_seqs.to(device)
#     node_features = builder.node_features.to(device)
#
#     context_enc = ContextEncoder(
#         NODE_FEATURE_DIM, hidden_dim=32,
#         output_dim=CONTEXT_OUTPUT_DIM, dropout=0.1
#     ).to(device)
#
#     with torch.no_grad():
#         node_embeddings = context_enc(graph, node_features)
#         context_embeddings = builder.get_ip_embeddings(
#             node_embeddings, src_train, dst_train
#         ).to(device)
#
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#     temporal_mask = torch.ones(
#         len(X_train), temporal_seqs.shape[1]
#     ).to(device)
#
#     print(f"  Graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
#     print(f"  Temporal sequences: {temporal_seqs.shape}")
#     print(f"  Context embeddings: {context_embeddings.shape}")
#
#     # ═══════════════════════════════════════════════════════════
#     # Stage 3: Three-Track Encoder + VAE (TRAINING)
#     # ═══════════════════════════════════════════════════════════
#     feature_enc = FeatureEncoder(feature_dim, output_dim=FEATURE_OUTPUT_DIM)
#     behaviour_enc = TransformerBehaviourEncoder(
#         feature_dim, hidden_dim=64,
#         output_dim=BEHAVIOUR_OUTPUT_DIM,
#         num_layers=2, num_heads=4, dropout=0.1
#     )
#
#     joint_enc = JointEncoder(
#         feature_encoder=feature_enc,
#         context_encoder=context_enc,
#         behaviour_encoder=behaviour_enc,
#         feature_output_dim=FEATURE_OUTPUT_DIM,
#         context_output_dim=CONTEXT_OUTPUT_DIM,
#         behaviour_output_dim=BEHAVIOUR_OUTPUT_DIM,
#         z_dim=Z_DIM,
#         beta=VAE_BETA,
#     )
#
#     joint_enc = train_vae(
#         joint_enc, graph, X_train_tensor, node_features,
#         context_embeddings, temporal_seqs, temporal_mask,
#         epochs=VAE_EPOCHS, batch_size=VAE_BATCH_SIZE,
#         lr=VAE_LR, beta_max=VAE_BETA, beta_warmup=VAE_BETA_WARMUP,
#         device=device,
#     )
#
#     Z_train = extract_embeddings(
#         joint_enc, graph, X_train_tensor, node_features,
#         context_embeddings, temporal_seqs, temporal_mask,
#         batch_size=VAE_BATCH_SIZE, device=device,
#     )
#
#     # ═══════════════════════════════════════════════════════════
#     # Stage 4: Fit DP-GMM (Unsupervised) + Cluster Summarization
#     # ═══════════════════════════════════════════════════════════
#     print("\n" + "="*60)
#     print("STAGE 4: Fit DP-GMM + Cluster Summarization")
#     print("="*60)
#
#     gmm = StreamingDPGMM(
#         max_components=GMM_MAX_COMPONENTS,
#         covariance_type=GMM_COVARIANCE_TYPE,
#     )
#     cluster_assignments, log_liks = gmm.fit(Z_train)
#
#     active_clusters = np.unique(cluster_assignments)
#     print(f"  Active clusters: {len(active_clusters)}")
#     print(f"  Log-likelihood range: [{log_liks.min():.2f}, {log_liks.max():.2f}]")
#
#     for k in active_clusters:
#         count = (cluster_assignments == k).sum()
#         weight = gmm.model.weights_[k]
#         print(f"    Cluster {k}: {count:,} flows (weight={weight:.4f})")
#
#     # ── Cluster purity against ground truth (evaluation only) ──
#     if y_train is not None:
#         print(f"\n  Per-cluster purity (ground truth):")
#         for k in active_clusters:
#             mask = cluster_assignments == k
#             cluster_labels = y_train[mask]
#             majority = Counter(cluster_labels).most_common(1)[0]
#             purity = majority[1] / mask.sum()
#             print(f"    Cluster {k:2d}: {mask.sum():>7,} flows | "
#                   f"Majority: {majority[0]:15s} ({purity*100:.1f}%)")
#
#     # Cluster Summarization Module
#     summarizer = ClusterSummarizer()
#     cluster_summaries = summarizer.summarize_all_clusters(
#         cluster_assignments, Z_train, X_train,
#         src_ips=src_train, dst_ips=dst_train,
#     )
#
#     # ═══════════════════════════════════════════════════════════
#     # Stage 5: LLM Semantic Labeling -> Knowledge Base
#     # ═══════════════════════════════════════════════════════════
#     print("\n" + "="*60)
#     print("STAGE 5: LLM Semantic Labeling -> Knowledge Base")
#     print("="*60)
#
#     labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)
#
#     cluster_weights = {
#         k: float(gmm.model.weights_[k]) for k in active_clusters
#     }
#
#     labels = labeler.label_all_clusters(cluster_summaries, cluster_weights)
#
#     kb = KnowledgeBase()
#     for k in active_clusters:
#         if k not in labels:
#             continue
#         label = labels[k]
#         summary = cluster_summaries.get(k, {})
#
#         kb.add_entry(
#             cluster_idx=int(k),
#             tactic=label['tactic'],
#             tactic_id=label['tactic_id'],
#             p_gmm=cluster_weights.get(k, 0.0),
#             p_llm=label['p_llm'],
#             reasoning=label['reasoning'],
#             summary_text=summary.get('text', ''),
#             fusion_weights=[DEFAULT_W1_GMM, DEFAULT_W2_LLM],
#         )
#
#     print(f"\n{kb}")
#
#     # ═══════════════════════════════════════════════════════════
#     # Stage 6: Bayesian Fusion Setup
#     # ═══════════════════════════════════════════════════════════
#     print("\n" + "="*60)
#     print("STAGE 6: Bayesian Fusion Setup")
#     print("="*60)
#
#     fusion = BayesianFusion(default_w1=DEFAULT_W1_GMM, default_w2=DEFAULT_W2_LLM)
#     print(f"  Default fusion weights: w1(GMM)={DEFAULT_W1_GMM}, "
#           f"w2(LLM)={DEFAULT_W2_LLM}")
#     print(f"  Formula: P_final = w1 * P(GMM) + w2 * P(LLM)")
#
#     # ═══════════════════════════════════════════════════════════
#     # Stage 7: Save Model Checkpoint
#     # ═══════════════════════════════════════════════════════════
#     print("\n" + "="*60)
#     print("STAGE 7: Save Model for Inference")
#     print("="*60)
#
#     os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#
#     checkpoint = {
#         'dataset_name': DATASET_NAME,
#         'joint_encoder_state': joint_enc.state_dict(),
#         'gmm_model': gmm.model,
#         'gmm_config': {
#             'max_components': gmm.max_components,
#             'covariance_type': gmm.covariance_type,
#             'weight_threshold': gmm.weight_threshold,
#             'novelty_threshold': gmm.novelty_threshold,
#         },
#         'preprocessor_scaler': data['scaler'],
#         'feature_names': data['feature_names'],
#         'ip_to_idx': builder.ip_to_idx,
#         'config': {
#             'feature_dim': feature_dim,
#             'feature_output_dim': FEATURE_OUTPUT_DIM,
#             'context_output_dim': CONTEXT_OUTPUT_DIM,
#             'behaviour_output_dim': BEHAVIOUR_OUTPUT_DIM,
#             'z_dim': Z_DIM,
#             'seq_length': SEQ_LENGTH,
#             'node_feature_dim': NODE_FEATURE_DIM,
#         },
#         'attack_types': ATTACK_TYPES,
#         'attack_to_mitre': ATTACK_TO_MITRE,
#     }
#     torch.save(checkpoint, MODEL_PATH)
#     print(f"  Model checkpoint saved: {MODEL_PATH}")
#
#     kb.save(KB_PATH)
#     print(f"  Knowledge Base saved: {KB_PATH}")
#
#     print("\n" + "="*60)
#     print(f"TRAINING PIPELINE COMPLETE ({DATASET_NAME})")
#     print("="*60)
#     print(f"  Trained VAE encoder ({Z_DIM}d latent space)")
#     print(f"  DP-GMM with {len(active_clusters)} clusters")
#     print(f"  Knowledge Base with {len(kb.entries)} labeled clusters")
#     print(f"  Attack types: {ATTACK_TYPES}")
#     print(f"  Ready for: New flow -> Encode -> Cluster -> Label -> Confidence")
#
#
# if __name__ == '__main__':
#     main()