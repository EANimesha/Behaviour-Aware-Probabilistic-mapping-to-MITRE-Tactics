"""
Training Pipeline — Behaviour-Aware
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
import time

from src.config.config import (
    DATA_PATH, SAMPLE_SIZE,
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
from src.utils.plot_gmm import plot_dpgmm_convergence
from src.utils.training_logger import TrainingLogger, compute_grad_norms


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
    t0 = time.perf_counter()

    # Stage 1
    print("\n" + "="*60)
    print("STAGE 1: Load & Preprocess")
    print("="*60)
    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    preprocessor = Preprocessor(clip_extremes=False)
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
    data_time = time.perf_counter() - t0
    print(f"Data Loading Time: {data_time:.4f}s")

    # Stage 2
    print("\n" + "="*60)
    print("STAGE 2: Graph + Temporal")
    print("="*60)
    t1 = time.perf_counter()
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
    graph_temporal_time = time.perf_counter() - t1
    print(f"Graph Construction Time: {graph_temporal_time:.4f}s")

    # Stage 3
    print("\n" + "="*60)
    print("STAGE 3: Compute Behaviour Scores")
    print("="*60)
    t2 = time.perf_counter()
    scorer = BehaviourScorer(feature_names=data['feature_names'])
    fan_out, fan_in, n_peers = scorer.compute_graph_features(src_train, dst_train)
    bhv_scores = scorer.compute(X_train, fan_out=fan_out, fan_in=fan_in, n_peers=n_peers)
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
    plot_dpgmm_convergence(Z_train, gmm.model, os.path.join(CHECKPOINT_DIR, 'dpgmm_convergence.png'))

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

        # ── DIAGNOSTIC ──
        n_benign = (y_train[mask] == 'Benign').sum()  # check your label string
        benign_pct = n_benign / mask.sum() * 100
        raw_ttl_scaled = cluster_mean[fn_idx['MIN_TTL']]
        raw_tcp_scaled = cluster_mean[fn_idx['TCP_FLAGS']]
        print(f"  C{k:2d}: n={mask.sum():>6,} benign={benign_pct:5.1f}% | "
                  f"TTL raw={min_ttl:7.2f} scaled={raw_ttl_scaled:6.2f} | "
                  f"TCP raw={tcp_flags:6.2f} scaled={raw_tcp_scaled:6.2f} | "
                  f"prop={proportion:.3f} dev={deviation:.3f}")
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

    labeler = LLMLabeler(api_key=OPENAI_API_KEY, model=LLM_MODEL)
    cluster_weights = {k: float(gmm.model.weights_[k]) for k in active}

    # ══════════════════════════════════════════════════════
    # Cluster Behaviour Labeling (single LLM call per cluster)
    # ══════════════════════════════════════════════════════
    from src.mapping.behaviour_mapping import get_primary_tactic
    from src.mapping.soft_cluster_scorer import (
        SoftClusterScorer, BEHAVIOUR_TACTIC_WEIGHTS)

    cluster_behaviour_map_hard = {}
    # has_llm = labeler.client is not None
    has_llm = False

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
            t6 = time.perf_counter()
            soft_result = labeler.label_cluster_soft(
                summary, cluster_weights.get(k, 0))
            llm_api_time = time.perf_counter() - t6
            print(llm_api_time)

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
    total_labeling_time = time.perf_counter() - t5
    print(f"LLM/Mock Labeling + Summarization Time: {total_labeling_time:.4f}s")

    # Build KB with rule-based soft cluster scoring (separate from LLM)
    soft_scorer = SoftClusterScorer()
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
        gt_mapper = GroundTruthMapper('bot_iot')
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
    t = time.perf_counter()
    main()
    time_taken = time.perf_counter() - t
    print("Total time taken: ", time_taken)
