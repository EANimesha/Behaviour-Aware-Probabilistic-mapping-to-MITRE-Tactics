# """
# Comprehensive Validation Pipeline — TDSC Paper Metrics
# =========================================================
#
# Computes ALL metrics needed for the paper:
#   1. Clustering quality (purity, NMI, ARI, silhouette at 3 levels)
#   2. Per-flow tactic prediction (precision/recall/F1/confusion)
#   3. Window-level soft tactic prediction (top-1, top-2)
#   4. Soft multi-tactic evaluation (Jaccard, cosine, KL)
#   5. Anomaly detection (AUC-ROC using reconstruction error)
#   6. Multi-label per-tactic P/R/F1
#
# Usage:
#     python validation.py              # bot_iot
#     python validation.py unsw_nb15    # unsw
# """
#
# import sys
# import os
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from collections import Counter
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     silhouette_score, normalized_mutual_info_score,
#     adjusted_rand_score, classification_report,
#     confusion_matrix, roc_auc_score, average_precision_score,
#     precision_recall_curve, roc_curve
# )
#
# DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'
#
# if DATASET == 'unsw_nb15':
#     from src.config.config_unsw import (
#         DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH,
#         FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
#         Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
#     )
# else:
#     from src.config.config import (
#         DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH,
#         FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
#         Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
#     )
#
# from src.data.loader import load_dataset
# from src.data.preprocess import Preprocessor
# from src.data.graph_builder import NetworkGraphBuilder
# from src.encoder.feature import FeatureEncoder
# from src.encoder.context import ContextEncoder
# from src.encoder.behaviour import TransformerBehaviourEncoder
# from src.encoder.jointencoder import JointEncoder
# from src.streaming.streaming_gmm import StreamingDPGMM
# from src.clustering.summarizer import ClusterSummarizer
# from src.mapping.soft_cluster_scorer import SoftClusterScorer, SoftWindowAggregator
# from src.evaluation.ground_truth import GroundTruthMapper
# from src.evaluation.soft_eval import SoftTacticEvaluator
#
# EVAL_DIR = os.path.join(CHECKPOINT_DIR, "paper_results")
#
#
# def load_and_encode():
#     """Load data, encode, cluster — return everything needed."""
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
#     cfg = checkpoint['config']
#
#     df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
#
#     if DATASET == 'unsw_nb15' and 'Attack' in df.columns:
#         from src.data.balancer import ChronologicalBalancer
#         from src.config.config_unsw import (
#             BENIGN_RATIO, MIN_SAMPLES_PER_CLASS, MAX_TOTAL_SAMPLES)
#         balancer = ChronologicalBalancer(
#             BENIGN_RATIO, MIN_SAMPLES_PER_CLASS,
#             MAX_TOTAL_SAMPLES, 'Benign')
#         df, _ = balancer.balance_with_minority_boost(df)
#
#     preprocessor = Preprocessor()
#     data = preprocessor.preprocess_dataset(df)
#     X_train = data['X_train']
#     y_train = data['y_train']
#     src_train, dst_train = data['src_train'], data['dst_train']
#
#     # Raw DataFrame for training split
#     train_indices = data.get('train_indices', np.arange(len(X_train)))
#     df_train_raw = df.iloc[train_indices].reset_index(drop=True)
#
#     # Build encoder
#     feature_enc = FeatureEncoder(cfg['feature_dim'], cfg['feature_output_dim'])
#     context_enc = ContextEncoder(cfg['node_feature_dim'], 32,
#                                   cfg['context_output_dim'], 0.0)
#     behaviour_enc = TransformerBehaviourEncoder(
#         cfg['feature_dim'], 64, cfg['behaviour_output_dim'], 2, 4, 0.0)
#     joint_enc = JointEncoder(
#         feature_enc, context_enc, behaviour_enc,
#         cfg['feature_output_dim'], cfg['context_output_dim'],
#         cfg['behaviour_output_dim'], cfg['z_dim'],
#         n_behaviours=cfg.get('n_behaviours', 5),
#         lambda_bhv=cfg.get('lambda_bhv', 1.0))
#     joint_enc.load_state_dict(checkpoint['joint_encoder_state'])
#     joint_enc.to(device); joint_enc.eval()
#
#     # Graph + temporal
#     builder = NetworkGraphBuilder(seq_length=cfg['seq_length'])
#     graph, temporal_seqs = builder.build_graph(X_train, src_train, dst_train)
#     graph = graph.to(device)
#     temporal_seqs = temporal_seqs.to(device)
#     node_features = builder.node_features.to(device)
#
#     with torch.no_grad():
#         node_emb = context_enc.to(device)(graph, node_features)
#         context_emb = builder.get_ip_embeddings(
#             node_emb, src_train, dst_train).to(device)
#
#     X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#     t_mask = torch.ones(len(X_train), temporal_seqs.shape[1]).to(device)
#
#     # Extract z + reconstruction
#     print("Extracting embeddings...")
#     all_z, all_fusion, all_recon = [], [], []
#     with torch.no_grad():
#         for s in range(0, len(X_tensor), VAE_BATCH_SIZE):
#             e = min(s + VAE_BATCH_SIZE, len(X_tensor))
#             h_f = joint_enc.feature_encoder(X_tensor[s:e])
#             h_b = joint_enc.behaviour_encoder(temporal_seqs[s:e], t_mask[s:e])
#             h_c = context_emb[s:e]
#             if hasattr(joint_enc, 'norm_f'):
#                 h_f = joint_enc.norm_f(h_f)
#                 h_c = joint_enc.norm_c(h_c)
#                 h_b = joint_enc.norm_b(h_b)
#             h_fusion = torch.cat([h_f, h_c, h_b], dim=1)
#             z = joint_enc.encoder(h_fusion)
#             h_recon = joint_enc.decoder(z)
#             all_z.append(z.cpu())
#             all_fusion.append(h_fusion.cpu())
#             all_recon.append(h_recon.cpu())
#
#     Z = torch.cat(all_z).numpy()
#     H_fusion = torch.cat(all_fusion).numpy()
#     H_recon = torch.cat(all_recon).numpy()
#
#     # GMM clusters
#     gmm_cfg = checkpoint['gmm_config']
#     gmm = StreamingDPGMM(
#         gmm_cfg['max_components'], gmm_cfg['covariance_type'],
#         gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
#     gmm.model = checkpoint['gmm_model']
#     clusters = gmm.model.predict(Z)
#     gmm_probs = gmm.model.predict_proba(Z)
#     gmm_scores = gmm.model.score_samples(Z)
#
#     # Cluster summaries
#     summarizer = ClusterSummarizer(
#         feature_names=data['feature_names'], scaler=data['scaler'])
#     summaries = summarizer.summarize_all_clusters(
#         clusters, Z, X_train,
#         src_ips=src_train, dst_ips=dst_train,
#         df_raw=df_train_raw)
#
#     # Reconstruction errors
#     recon_errors = np.mean((H_fusion - H_recon) ** 2, axis=1)
#
#     return {
#         'Z': Z, 'H_fusion': H_fusion, 'H_recon': H_recon,
#         'X_train': X_train, 'y_train': y_train,
#         'src_train': src_train, 'dst_train': dst_train,
#         'df_raw': df_train_raw,
#         'clusters': clusters,
#         'gmm_probs': gmm_probs, 'gmm_scores': gmm_scores,
#         'recon_errors': recon_errors,
#         'summaries': summaries,
#         'feature_names': data['feature_names'],
#         'checkpoint': checkpoint,
#     }
#
#
# def run_validation():
#     os.makedirs(EVAL_DIR, exist_ok=True)
#
#     print(f"\n{'='*70}")
#     print(f"COMPREHENSIVE VALIDATION — {DATASET}")
#     print(f"{'='*70}")
#
#     data = load_and_encode()
#     y_train = data['y_train']
#     clusters = data['clusters']
#     Z = data['Z']
#     src_train = data['src_train']
#
#     gt_mapper = GroundTruthMapper(DATASET)
#     true_tactics = gt_mapper.map_labels(y_train)
#     tactic_list = gt_mapper.get_tactic_list()
#
#     all_metrics = {}
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 1: CLUSTERING QUALITY
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'='*70}")
#     print("SECTION 1: CLUSTERING QUALITY")
#     print(f"{'='*70}")
#
#     n_clusters = len(np.unique(clusters))
#     sample_size = min(5000, len(Z))
#
#     # Purity
#     purities = []
#     for k in np.unique(clusters):
#         mask = clusters == k
#         if mask.sum() < 2: continue
#         maj = Counter(y_train[mask]).most_common(1)[0]
#         purities.append(maj[1] / mask.sum())
#     mean_purity = np.mean(purities)
#     sizes = [np.sum(clusters == k) for k in np.unique(clusters)]
#     weighted_purity = np.average(purities, weights=sizes[:len(purities)])
#
#     # NMI / ARI at 3 levels
#     nmi_attack = normalized_mutual_info_score(y_train, clusters)
#     ari_attack = adjusted_rand_score(y_train, clusters)
#     nmi_tactic = normalized_mutual_info_score(true_tactics, clusters)
#     ari_tactic = adjusted_rand_score(true_tactics, clusters)
#
#     # Silhouette at 3 levels
#     sil_attack = silhouette_score(Z, y_train, sample_size=sample_size)
#     sil_tactic = silhouette_score(Z, true_tactics, sample_size=sample_size)
#     sil_cluster = silhouette_score(Z, clusters, sample_size=sample_size)
#
#     print(f"  Active clusters:        {n_clusters}")
#     print(f"  Mean cluster purity:    {mean_purity:.4f}")
#     print(f"  Weighted cluster purity:{weighted_purity:.4f}")
#     print(f"\n  {'Level':>15s} {'NMI':>8s} {'ARI':>8s} {'Silhouette':>12s}")
#     print(f"  {'-'*45}")
#     print(f"  {'Attack types':>15s} {nmi_attack:>8.4f} {ari_attack:>8.4f} {sil_attack:>12.4f}")
#     print(f"  {'Tactics':>15s} {nmi_tactic:>8.4f} {ari_tactic:>8.4f} {sil_tactic:>12.4f}")
#     print(f"  {'GMM Clusters':>15s} {'—':>8s} {'—':>8s} {sil_cluster:>12.4f}")
#
#     all_metrics.update({
#         'n_clusters': n_clusters,
#         'mean_purity': mean_purity, 'weighted_purity': weighted_purity,
#         'nmi_attack': nmi_attack, 'ari_attack': ari_attack,
#         'nmi_tactic': nmi_tactic, 'ari_tactic': ari_tactic,
#         'sil_attack': sil_attack, 'sil_tactic': sil_tactic,
#         'sil_cluster': sil_cluster,
#     })
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 2: SOFT CLUSTER SCORING
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'='*70}")
#     print("SECTION 2: SOFT CLUSTER BEHAVIOUR SCORING")
#     print(f"{'='*70}")
#
#     scorer = SoftClusterScorer()
#     cluster_scores = scorer.score_all_clusters(data['summaries'])
#
#     # Per-flow tactic from soft scoring
#     flow_tactics_soft = []
#     flow_bhv_soft = []
#     for i in range(len(clusters)):
#         k = int(clusters[i])
#         cs = cluster_scores.get(k, {})
#         flow_tactics_soft.append(cs.get('primary_tactic', 'None'))
#         flow_bhv_soft.append(cs.get('primary_behaviour', 'normal_activity'))
#     flow_tactics_soft = np.array(flow_tactics_soft)
#     flow_bhv_soft = np.array(flow_bhv_soft)
#
#     # Behaviour-level NMI/ARI
#     nmi_bhv = normalized_mutual_info_score(y_train, flow_bhv_soft)
#     ari_bhv = adjusted_rand_score(y_train, flow_bhv_soft)
#
#     # Silhouette by behaviour
#     unique_bhvs = np.unique(flow_bhv_soft)
#     if len(unique_bhvs) > 1:
#         sil_bhv = silhouette_score(Z, flow_bhv_soft, sample_size=sample_size)
#     else:
#         sil_bhv = 0.0
#
#     print(f"  {'Behaviours':>15s} {nmi_bhv:>8.4f} {ari_bhv:>8.4f} {sil_bhv:>12.4f}")
#     all_metrics.update({
#         'nmi_behaviour': nmi_bhv, 'ari_behaviour': ari_bhv,
#         'sil_behaviour': sil_bhv,
#     })
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 3: PER-FLOW TACTIC PREDICTION
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'='*70}")
#     print("SECTION 3: PER-FLOW TACTIC PREDICTION (soft cluster scoring)")
#     print(f"{'='*70}")
#
#     flow_acc = (flow_tactics_soft == true_tactics).mean()
#     print(f"\n  Per-flow accuracy: {flow_acc:.4f}")
#
#     report = classification_report(
#         true_tactics, flow_tactics_soft,
#         labels=tactic_list, output_dict=True, zero_division=0)
#     print(f"\n  {classification_report(true_tactics, flow_tactics_soft, labels=tactic_list, zero_division=0)}")
#
#     all_metrics['perflow_accuracy'] = flow_acc
#     all_metrics['perflow_macro_f1'] = report.get('macro avg', {}).get('f1-score', 0)
#     all_metrics['perflow_weighted_f1'] = report.get('weighted avg', {}).get('f1-score', 0)
#     for tactic in tactic_list:
#         if tactic in report:
#             all_metrics[f'perflow_{tactic}_f1'] = report[tactic]['f1-score']
#             all_metrics[f'perflow_{tactic}_precision'] = report[tactic]['precision']
#             all_metrics[f'perflow_{tactic}_recall'] = report[tactic]['recall']
#
#     # Confusion matrix plot
#     cm = confusion_matrix(true_tactics, flow_tactics_soft, labels=tactic_list)
#     fig, ax = plt.subplots(figsize=(8, 6))
#     cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
#     im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
#     ax.set_xticks(range(len(tactic_list)))
#     ax.set_yticks(range(len(tactic_list)))
#     ax.set_xticklabels(tactic_list, rotation=45, ha='right', fontsize=8)
#     ax.set_yticklabels(tactic_list, fontsize=8)
#     for i in range(len(tactic_list)):
#         for j in range(len(tactic_list)):
#             ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})',
#                     ha='center', va='center', fontsize=7)
#     ax.set_xlabel('Predicted'); ax.set_ylabel('True')
#     ax.set_title('Per-Flow Tactic Confusion Matrix')
#     fig.colorbar(im)
#     fig.tight_layout()
#     fig.savefig(os.path.join(EVAL_DIR, 'perflow_confusion.png'), dpi=150)
#     plt.close(fig)
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 4: WINDOW-LEVEL SOFT TACTIC PREDICTION
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'='*70}")
#     print("SECTION 4: WINDOW-LEVEL SOFT TACTIC PREDICTION")
#     print(f"{'='*70}")
#
#     for ws in [10, 20, 50]:
#         print(f"\n  --- Window size = {ws} ---")
#         aggregator = SoftWindowAggregator(
#             window_size=ws, tactic_threshold=0.10)
#         windows = aggregator.create_windows(
#             src_train, clusters, cluster_scores, y_true=y_train)
#
#         # Evaluate with SoftTacticEvaluator
#         soft_eval = SoftTacticEvaluator()
#         metrics, df = soft_eval.evaluate_windows(
#             windows, gt_mapper,
#             save_dir=os.path.join(EVAL_DIR, f'ws_{ws}'))
#
#         if ws == 20:  # Save primary window size metrics
#             for k, v in metrics.items():
#                 all_metrics[f'window_{k}'] = v
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 5: ANOMALY DETECTION (binary)
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'='*70}")
#     print("SECTION 5: ANOMALY DETECTION (AUC-ROC)")
#     print(f"{'='*70}")
#
#     # Binary labels: 1=attack, 0=benign
#     binary_true = (true_tactics != 'None').astype(int)
#
#     if len(np.unique(binary_true)) > 1:
#         # Score 1: reconstruction error
#         recon_auc = roc_auc_score(binary_true, data['recon_errors'])
#         recon_ap = average_precision_score(binary_true, data['recon_errors'])
#
#         # Score 2: negative GMM log-likelihood
#         neg_gmm = -data['gmm_scores']
#         gmm_auc = roc_auc_score(binary_true, neg_gmm)
#         gmm_ap = average_precision_score(binary_true, neg_gmm)
#
#         # Score 3: combined
#         combined = (data['recon_errors'] / data['recon_errors'].max() +
#                     neg_gmm / neg_gmm.max()) / 2
#         combined_auc = roc_auc_score(binary_true, combined)
#         combined_ap = average_precision_score(binary_true, combined)
#
#         print(f"  {'Score':>20s} {'AUC-ROC':>10s} {'AUC-PR':>10s}")
#         print(f"  {'-'*42}")
#         print(f"  {'Recon Error':>20s} {recon_auc:>10.4f} {recon_ap:>10.4f}")
#         print(f"  {'GMM (-loglik)':>20s} {gmm_auc:>10.4f} {gmm_ap:>10.4f}")
#         print(f"  {'Combined':>20s} {combined_auc:>10.4f} {combined_ap:>10.4f}")
#
#         all_metrics.update({
#             'auc_recon': recon_auc, 'ap_recon': recon_ap,
#             'auc_gmm': gmm_auc, 'ap_gmm': gmm_ap,
#             'auc_combined': combined_auc, 'ap_combined': combined_ap,
#         })
#
#         # ROC curve plot
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#         for name, scores, color in [
#             ('Recon Error', data['recon_errors'], 'blue'),
#             ('GMM Score', neg_gmm, 'green'),
#             ('Combined', combined, 'red'),
#         ]:
#             fpr, tpr, _ = roc_curve(binary_true, scores)
#             auc = roc_auc_score(binary_true, scores)
#             axes[0].plot(fpr, tpr, color=color,
#                          label=f'{name} (AUC={auc:.3f})')
#             prec, rec, _ = precision_recall_curve(binary_true, scores)
#             ap = average_precision_score(binary_true, scores)
#             axes[1].plot(rec, prec, color=color,
#                          label=f'{name} (AP={ap:.3f})')
#
#         axes[0].plot([0,1], [0,1], 'k--', alpha=0.3)
#         axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
#         axes[0].set_title('ROC Curve (Attack vs Benign)')
#         axes[0].legend(fontsize=8)
#         axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
#         axes[1].set_title('Precision-Recall Curve')
#         axes[1].legend(fontsize=8)
#         fig.tight_layout()
#         fig.savefig(os.path.join(EVAL_DIR, 'anomaly_detection.png'), dpi=150)
#         plt.close(fig)
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 6: CLUSTER LABELING STRATEGY COMPARISON
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'='*70}")
#     print("SECTION 6: CLUSTER LABELING STRATEGY COMPARISON")
#     print(f"{'='*70}")
#
#     from src.evaluation.labeling_eval import evaluate_all_strategies
#     labeling_results = evaluate_all_strategies(
#         cluster_summaries=data['summaries'],
#         cluster_assignments=clusters,
#         y_train=y_train,
#         gt_mapper=gt_mapper,
#         checkpoint=data.get('checkpoint'))
#
#     for r in labeling_results:
#         all_metrics[f"label_{r['name'].replace(' ', '_')}_agreement"] = r['hard_agreement']
#         all_metrics[f"label_{r['name'].replace(' ', '_')}_jaccard"] = r['soft_jaccard']
#         all_metrics[f"label_{r['name'].replace(' ', '_')}_kl"] = r['kl_divergence']
#
#     # ══════════════════════════════════════════════════════
#     # SUMMARY TABLE
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'='*70}")
#     print("SUMMARY — ALL METRICS FOR PAPER")
#     print(f"{'='*70}")
#
#     print(f"\n  CLUSTERING QUALITY:")
#     print(f"    Clusters:              {all_metrics.get('n_clusters')}")
#     print(f"    Mean Purity:           {all_metrics.get('mean_purity', 0):.4f}")
#     print(f"    Weighted Purity:       {all_metrics.get('weighted_purity', 0):.4f}")
#     print(f"    NMI (attack):          {all_metrics.get('nmi_attack', 0):.4f}")
#     print(f"    NMI (tactic):          {all_metrics.get('nmi_tactic', 0):.4f}")
#     print(f"    NMI (behaviour):       {all_metrics.get('nmi_behaviour', 0):.4f}")
#     print(f"    Silhouette (attack):   {all_metrics.get('sil_attack', 0):.4f}")
#     print(f"    Silhouette (tactic):   {all_metrics.get('sil_tactic', 0):.4f}")
#     print(f"    Silhouette (behaviour):{all_metrics.get('sil_behaviour', 0):.4f}")
#     print(f"    Silhouette (cluster):  {all_metrics.get('sil_cluster', 0):.4f}")
#
#     print(f"\n  PER-FLOW TACTIC PREDICTION:")
#     print(f"    Accuracy:              {all_metrics.get('perflow_accuracy', 0):.4f}")
#     print(f"    Macro F1:              {all_metrics.get('perflow_macro_f1', 0):.4f}")
#     print(f"    Weighted F1:           {all_metrics.get('perflow_weighted_f1', 0):.4f}")
#
#     print(f"\n  WINDOW-LEVEL (W=20):")
#     print(f"    Top-1 Accuracy:        {all_metrics.get('window_top1_accuracy', 0):.4f}")
#     print(f"    Coverage (Top-k):      {all_metrics.get('window_coverage', 0):.4f}")
#     print(f"    Mean Jaccard:          {all_metrics.get('window_mean_jaccard', 0):.4f}")
#     print(f"    Mean Soft Jaccard:     {all_metrics.get('window_mean_soft_jaccard', 0):.4f}")
#     print(f"    Mean Cosine:           {all_metrics.get('window_mean_cosine', 0):.4f}")
#     print(f"    Mean KL Divergence:    {all_metrics.get('window_mean_kl_divergence', 0):.4f}")
#
#     print(f"\n  ANOMALY DETECTION:")
#     print(f"    AUC-ROC (recon):       {all_metrics.get('auc_recon', 0):.4f}")
#     print(f"    AUC-ROC (GMM):         {all_metrics.get('auc_gmm', 0):.4f}")
#     print(f"    AUC-ROC (combined):    {all_metrics.get('auc_combined', 0):.4f}")
#
#     print(f"\n  CLUSTER LABELING COMPARISON:")
#     for r in labeling_results:
#         print(f"    {r['name']:<18s}: agree={r['hard_agreement']:.4f}  "
#               f"jaccard={r['soft_jaccard']:.4f}  kl={r['kl_divergence']:.4f}")
#
#     # Save all metrics
#     metrics_df = pd.DataFrame([all_metrics])
#     metrics_df.to_csv(os.path.join(EVAL_DIR, 'all_metrics.csv'), index=False)
#     print(f"\n  All metrics saved: {EVAL_DIR}/all_metrics.csv")
#     print(f"  Plots saved: {EVAL_DIR}/")
#
#
# if __name__ == '__main__':
#     run_validation()
"""
Comprehensive Validation Pipeline — TDSC Paper Metrics
=========================================================

Computes ALL metrics needed for the paper:
  1. Clustering quality (purity, NMI, ARI, silhouette at 3 levels)
  2. Per-flow tactic prediction (precision/recall/F1/confusion)
  3. Window-level soft tactic prediction (top-1, top-2)
  4. Soft multi-tactic evaluation (Jaccard, cosine, KL)
  5. Anomaly detection (AUC-ROC using reconstruction error)
  6. Multi-label per-tactic P/R/F1

Usage:
    python validation.py              # bot_iot
    python validation.py unsw_nb15    # unsw
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score, normalized_mutual_info_score,
    adjusted_rand_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'

if DATASET == 'unsw_nb15':
    from src.config.config_unsw import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,KB_PATH
    )
elif DATASET == 'cicids2018':
    from src.config.config_cicids import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,KB_PATH
    )
else:
    from src.config.config import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE, KB_PATH,
)

from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
from src.data.graph_builder import NetworkGraphBuilder
from src.encoder.feature import FeatureEncoder
from src.encoder.context import ContextEncoder
from src.encoder.behaviour import TransformerBehaviourEncoder
from src.encoder.jointencoder import JointEncoder
from src.streaming.streaming_gmm import StreamingDPGMM
from src.clustering.summarizer import ClusterSummarizer
from src.mapping.soft_cluster_scorer import SoftClusterScorer, SoftWindowAggregator
from src.evaluation.ground_truth import GroundTruthMapper
from src.evaluation.soft_eval import SoftTacticEvaluator

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "paper_results")


def load_and_encode():
    """Load data, encode, cluster — return everything needed."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)

    if DATASET == 'unsw_nb15' and 'Attack' in df.columns:
        from src.data.balancer import FastChronologicalBalancer
        from src.config.config_unsw import (
            MAX_BENIGN_SAMPLES, MAX_ATTACK_PER_CLASS, MIN_ATTACK_PER_CLASS)
        balancer = FastChronologicalBalancer(
            max_total_samples=120000, max_benign_samples=MAX_BENIGN_SAMPLES,
            max_attack_per_class=MAX_ATTACK_PER_CLASS,
            min_attack_per_class=MIN_ATTACK_PER_CLASS)
        df, _ = balancer.balance(df)
    elif DATASET == 'cicids2018' and 'Attack' in df.columns:
        from src.data.balancer import FastChronologicalBalancer
        from src.config.config_cicids import (
            MAX_BENIGN_SAMPLES, MAX_ATTACK_PER_CLASS, MIN_ATTACK_PER_CLASS)
        balancer = FastChronologicalBalancer(
            max_total_samples=120000, max_benign_samples=MAX_BENIGN_SAMPLES,
            max_attack_per_class=MAX_ATTACK_PER_CLASS,
            min_attack_per_class=MIN_ATTACK_PER_CLASS)
        df, _ = balancer.balance(df)

    preprocessor = Preprocessor()
    data = preprocessor.preprocess_dataset(df)
    X_train = data['X_train']
    y_train = data['y_train']
    src_train, dst_train = data['src_train'], data['dst_train']

    # Raw DataFrame for training split
    train_indices = data.get('train_indices', np.arange(len(X_train)))
    df_train_raw = df.iloc[train_indices].reset_index(drop=True)

    # Build encoder
    feature_enc = FeatureEncoder(cfg['feature_dim'], cfg['feature_output_dim'])
    context_enc = ContextEncoder(cfg['node_feature_dim'], 32,
                                  cfg['context_output_dim'], 0.0)
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

    # Graph + temporal
    builder = NetworkGraphBuilder(seq_length=cfg['seq_length'])
    graph, temporal_seqs = builder.build_graph(X_train, src_train, dst_train)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    with torch.no_grad():
        node_emb = context_enc.to(device)(graph, node_features)
        context_emb = builder.get_ip_embeddings(
            node_emb, src_train, dst_train).to(device)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    t_mask = torch.ones(len(X_train), temporal_seqs.shape[1]).to(device)

    # Extract z + reconstruction
    print("Extracting embeddings...")
    all_z, all_fusion, all_recon = [], [], []
    with torch.no_grad():
        for s in range(0, len(X_tensor), VAE_BATCH_SIZE):
            e = min(s + VAE_BATCH_SIZE, len(X_tensor))
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
            all_z.append(z.cpu())
            all_fusion.append(h_fusion.cpu())
            all_recon.append(h_recon.cpu())

    Z = torch.cat(all_z).numpy()
    H_fusion = torch.cat(all_fusion).numpy()
    H_recon = torch.cat(all_recon).numpy()

    # GMM clusters
    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']
    clusters = gmm.model.predict(Z)
    gmm_probs = gmm.model.predict_proba(Z)
    gmm_scores = gmm.model.score_samples(Z)

    # Cluster summaries
    summarizer = ClusterSummarizer(
        feature_names=data["feature_names"], scaler=data["scaler"], dataset=DATASET)
    summaries = summarizer.summarize_all_clusters(
        clusters, Z, X_train,
        src_ips=src_train, dst_ips=dst_train,
        df_raw=df_train_raw)
    # Reconstruction errors
    recon_errors = np.mean((H_fusion - H_recon) ** 2, axis=1)

    return {
        'Z': Z, 'H_fusion': H_fusion, 'H_recon': H_recon,
        'X_train': X_train, 'y_train': y_train,
        'src_train': src_train, 'dst_train': dst_train,
        'df_raw': df_train_raw,
        'clusters': clusters,
        'gmm_probs': gmm_probs, 'gmm_scores': gmm_scores,
        'recon_errors': recon_errors,
        'summaries': summaries,
        'feature_names': data['feature_names'],
        'checkpoint': checkpoint,
    }


def run_validation():
    os.makedirs(EVAL_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE VALIDATION — {DATASET}")
    print(f"{'='*70}")

    data = load_and_encode()
    y_train = data['y_train']
    clusters = data['clusters']
    Z = data['Z']
    src_train = data['src_train']

    gt_mapper = GroundTruthMapper(DATASET)
    true_tactics = gt_mapper.map_labels(y_train)
    tactic_list = gt_mapper.get_tactic_list()

    all_metrics = {}

    # ══════════════════════════════════════════════════════
    # SECTION 1: CLUSTERING QUALITY
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SECTION 1: CLUSTERING QUALITY")
    print(f"{'='*70}")

    n_clusters = len(np.unique(clusters))
    sample_size = min(5000, len(Z))

    # Purity
    purities = []
    for k in np.unique(clusters):
        mask = clusters == k
        if mask.sum() < 2: continue
        maj = Counter(y_train[mask]).most_common(1)[0]
        purities.append(maj[1] / mask.sum())
    mean_purity = np.mean(purities)
    sizes = [np.sum(clusters == k) for k in np.unique(clusters)]
    weighted_purity = np.average(purities, weights=sizes[:len(purities)])

    # NMI / ARI at 3 levels
    nmi_attack = normalized_mutual_info_score(y_train, clusters)
    ari_attack = adjusted_rand_score(y_train, clusters)
    nmi_tactic = normalized_mutual_info_score(true_tactics, clusters)
    ari_tactic = adjusted_rand_score(true_tactics, clusters)

    # Silhouette at 3 levels
    sil_attack = silhouette_score(Z, y_train, sample_size=sample_size)
    sil_tactic = silhouette_score(Z, true_tactics, sample_size=sample_size)
    sil_cluster = silhouette_score(Z, clusters, sample_size=sample_size)

    print(f"  Active clusters:        {n_clusters}")
    print(f"  Mean cluster purity:    {mean_purity:.4f}")
    print(f"  Weighted cluster purity:{weighted_purity:.4f}")
    print(f"\n  {'Level':>15s} {'NMI':>8s} {'ARI':>8s} {'Silhouette':>12s}")
    print(f"  {'-'*45}")
    print(f"  {'Attack types':>15s} {nmi_attack:>8.4f} {ari_attack:>8.4f} {sil_attack:>12.4f}")
    print(f"  {'Tactics':>15s} {nmi_tactic:>8.4f} {ari_tactic:>8.4f} {sil_tactic:>12.4f}")
    print(f"  {'GMM Clusters':>15s} {'—':>8s} {'—':>8s} {sil_cluster:>12.4f}")

    all_metrics.update({
        'n_clusters': n_clusters,
        'mean_purity': mean_purity, 'weighted_purity': weighted_purity,
        'nmi_attack': nmi_attack, 'ari_attack': ari_attack,
        'nmi_tactic': nmi_tactic, 'ari_tactic': ari_tactic,
        'sil_attack': sil_attack, 'sil_tactic': sil_tactic,
        'sil_cluster': sil_cluster,
    })

    # ══════════════════════════════════════════════════════
    # SECTION 2: SOFT CLUSTER SCORING
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SECTION 2: SOFT CLUSTER BEHAVIOUR SCORING")
    print(f"{'='*70}")

    # ── Rule-based scoring (SoftClusterScorer) ──
    print("\n  [Rule-based scoring]")
    scorer = SoftClusterScorer(DATASET)
    cluster_scores_rules = scorer.score_all_clusters(data['summaries'])

    # Build KB from rule-based scores
    from src.knowledge_base.kb import KnowledgeBase
    kb = KnowledgeBase()
    for k, cs in cluster_scores_rules.items():
        k = int(k)
        mask = clusters == k
        kb.add_from_soft_scorer(
            cluster_idx=k, scorer_result=cs,
            cluster_size=int(mask.sum()),
            cluster_proportion=float(mask.sum() / len(clusters)))
    print(f"  KB built: {len(kb.entries)} clusters")

    # ── LLM-based scoring (from checkpoint or live) ──
    print("\n  [LLM-based scoring]")
    checkpoint = data.get('checkpoint', {})
    cluster_scores_llm = {}

    # Try loading from checkpoint first
    cbm_hard = checkpoint.get('cluster_behaviour_map_hard', {})
    if cbm_hard:
        for k_str, entry in cbm_hard.items():
            k = int(k_str)
            if 'tactic_scores_llm' in entry and 'behaviour_scores_llm' in entry:
                cluster_scores_llm[k] = {
                    'behaviour_scores': entry['behaviour_scores_llm'],
                    'tactic_scores': entry['tactic_scores_llm'],
                    'primary_behaviour': entry.get('primary_behaviour', 'unknown'),
                    'primary_tactic': entry.get('primary_tactic', 'Unknown'),
                }
        print(f"  LLM scores loaded from checkpoint: {len(cluster_scores_llm)} clusters")
    else:
        # Try live LLM scoring
        try:
            from src.llm.reasoning import LLMLabeler
            labeler = LLMLabeler()
            if labeler.client is not None:
                from src.mapping.soft_cluster_scorer import get_behaviour_tactic_weights
                from collections import defaultdict as dd
                for k, summary in data['summaries'].items():
                    result = labeler.label_cluster_soft(summary, 0.0)
                    if result and result.get('behaviour_scores'):
                        tactic_scores = dd(float)
                        for bhv, score in result['behaviour_scores'].items():
                            btw = get_behaviour_tactic_weights(DATASET)
                            if bhv in btw:
                                for tac, w in btw[bhv].items():
                                    tactic_scores[tac] += score * w
                        total_ts = sum(tactic_scores.values())
                        if total_ts > 0:
                            tactic_scores = {t: s / total_ts
                                             for t, s in tactic_scores.items()}
                        cluster_scores_llm[int(k)] = {
                            'behaviour_scores': result['behaviour_scores'],
                            'tactic_scores': dict(tactic_scores),
                            'primary_behaviour': result['primary_behaviour'],
                            'primary_tactic': max(tactic_scores, key=tactic_scores.get),
                        }
                print(f"  LLM scores computed live: {len(cluster_scores_llm)} clusters")
            else:
                print(f"  LLM unavailable — using checkpoint mock labels")
        except Exception as e:
            print(f"  LLM scoring failed: {e}")

    has_llm_scores = len(cluster_scores_llm) > 0
    # has_llm_scores = False

    # Per-flow tactic from rule-based soft scoring
    flow_tactics_soft = []
    flow_bhv_soft = []
    for i in range(len(clusters)):
        k = int(clusters[i])
        cs = cluster_scores_rules.get(k, {})
        flow_tactics_soft.append(cs.get('primary_tactic', 'None'))
        flow_bhv_soft.append(cs.get('primary_behaviour', 'normal_activity'))
    flow_tactics_soft = np.array(flow_tactics_soft)
    # flow_bhv_soft = np.array(flow_bhv_soft)

    # # Behaviour-level NMI/ARI
    # nmi_bhv = normalized_mutual_info_score(y_train, flow_bhv_soft)
    # ari_bhv = adjusted_rand_score(y_train, flow_bhv_soft)
    #
    # # Silhouette by behaviour
    # unique_bhvs = np.unique(flow_bhv_soft)
    # if len(unique_bhvs) > 1:
    #     sil_bhv = silhouette_score(Z, flow_bhv_soft, sample_size=sample_size)
    # else:
    #     sil_bhv = 0.0
    #
    # print(f"  {'Behaviours':>15s} {nmi_bhv:>8.4f} {ari_bhv:>8.4f} {sil_bhv:>12.4f}")
    # all_metrics.update({
    #     'nmi_behaviour': nmi_bhv, 'ari_behaviour': ari_bhv,
    #     'sil_behaviour': sil_bhv,
    # })
    # ══════════════════════════════════════════════════════
    # SECTION 3: PER-FLOW TACTIC PREDICTION
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SECTION 3: PER-FLOW TACTIC PREDICTION")
    print(f"{'=' * 70}")

    # ── Rule-based per-flow ──
    print(f"\n  === RULE-BASED ===")
    flow_acc = (flow_tactics_soft == true_tactics).mean()
    print(f"  Per-flow accuracy: {flow_acc:.4f}")

    report = classification_report(
        true_tactics, flow_tactics_soft,
        labels=tactic_list, output_dict=True, zero_division=0, digits=4)
    print(f"\n{classification_report(true_tactics, flow_tactics_soft, labels=tactic_list, zero_division=0, digits=4)}")

    all_metrics['perflow_rules_accuracy'] = flow_acc
    all_metrics['perflow_rules_macro_f1'] = report.get('macro avg', {}).get('f1-score', 0)
    all_metrics['perflow_rules_weighted_f1'] = report.get('weighted avg', {}).get('f1-score', 0)
    for tactic in tactic_list:
        if tactic in report:
            all_metrics[f'perflow_rules_{tactic}_f1'] = report[tactic]['f1-score']
            all_metrics[f'perflow_rules_{tactic}_precision'] = report[tactic]['precision']
            all_metrics[f'perflow_rules_{tactic}_recall'] = report[tactic]['recall']

    # ── LLM-based per-flow ──
    if has_llm_scores:
        print(f"\n  === LLM-BASED ===")
        flow_tactics_llm = []
        for i in range(len(clusters)):
            k = int(clusters[i])
            cs = cluster_scores_llm.get(k, {})
            flow_tactics_llm.append(cs.get('primary_tactic', 'Unknown'))
        flow_tactics_llm = np.array(flow_tactics_llm)

        flow_acc_llm = (flow_tactics_llm == true_tactics).mean()
        print(f"  Per-flow accuracy: {flow_acc_llm:.4f}")

        report_llm = classification_report(
            true_tactics, flow_tactics_llm,
            labels=tactic_list, output_dict=True, zero_division=0, digits=4)
        print(f"\n{classification_report(true_tactics, flow_tactics_llm, labels=tactic_list, zero_division=0, digits=4)}")

        all_metrics['perflow_llm_accuracy'] = flow_acc_llm
        all_metrics['perflow_llm_macro_f1'] = report_llm.get('macro avg', {}).get('f1-score', 0)
        all_metrics['perflow_llm_weighted_f1'] = report_llm.get('weighted avg', {}).get('f1-score', 0)
        for tactic in tactic_list:
            if tactic in report_llm:
                all_metrics[f'perflow_llm_{tactic}_f1'] = report_llm[tactic]['f1-score']
                all_metrics[f'perflow_llm_{tactic}_precision'] = report_llm[tactic]['precision']
                all_metrics[f'perflow_llm_{tactic}_recall'] = report_llm[tactic]['recall']

        # Side-by-side summary
        print(f"\n  {'Tactic':>20s} {'Rules F1':>10s} {'LLM F1':>10s}")
        print(f"  {'-' * 42}")
        for tactic in tactic_list:
            r_f1 = report.get(tactic, {}).get('f1-score', 0)
            l_f1 = report_llm.get(tactic, {}).get('f1-score', 0)
            better = '←' if r_f1 > l_f1 else '→' if l_f1 > r_f1 else '='
            print(f"  {tactic:>20s} {r_f1:>10.4f} {l_f1:>10.4f} {better}")
        print(f"  {'Accuracy':>20s} {flow_acc:>10.4f} {flow_acc_llm:>10.4f}")

    # Confusion matrix plot (rule-based)
    cm = confusion_matrix(true_tactics, flow_tactics_soft, labels=tactic_list)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(tactic_list)))
    ax.set_yticks(range(len(tactic_list)))
    ax.set_xticklabels(tactic_list, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tactic_list, fontsize=8)
    for i in range(len(tactic_list)):
        for j in range(len(tactic_list)):
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.0%})',
                    ha='center', va='center', fontsize=7)
    ax.set_xlabel('Predicted');
    ax.set_ylabel('True')
    ax.set_title('Per-Flow Tactic Confusion Matrix (Rule-based)')
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, 'perflow_confusion_rules.png'), dpi=150)
    plt.close(fig)
    # # ══════════════════════════════════════════════════════
    # # SECTION 3: PER-FLOW TACTIC PREDICTION
    # # ══════════════════════════════════════════════════════
    # print(f"\n{'='*70}")
    # print("SECTION 3: PER-FLOW TACTIC PREDICTION (soft cluster scoring)")
    # print(f"{'='*70}")
    #
    # flow_acc = (flow_tactics_soft == true_tactics).mean()
    # print(f"\n  Per-flow accuracy: {flow_acc:.4f}")
    #
    # report = classification_report(
    #     true_tactics, flow_tactics_soft,
    #     labels=tactic_list, output_dict=True, zero_division=0)
    # print(f"\n  {classification_report(true_tactics, flow_tactics_soft, labels=tactic_list, zero_division=0)}")
    #
    # all_metrics['perflow_accuracy'] = flow_acc
    # all_metrics['perflow_macro_f1'] = report.get('macro avg', {}).get('f1-score', 0)
    # all_metrics['perflow_weighted_f1'] = report.get('weighted avg', {}).get('f1-score', 0)
    # for tactic in tactic_list:
    #     if tactic in report:
    #         all_metrics[f'perflow_{tactic}_f1'] = report[tactic]['f1-score']
    #         all_metrics[f'perflow_{tactic}_precision'] = report[tactic]['precision']
    #         all_metrics[f'perflow_{tactic}_recall'] = report[tactic]['recall']
    #
    # # Confusion matrix plot
    # cm = confusion_matrix(true_tactics, flow_tactics_soft, labels=tactic_list)
    # fig, ax = plt.subplots(figsize=(8, 6))
    # cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    # im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    # ax.set_xticks(range(len(tactic_list)))
    # ax.set_yticks(range(len(tactic_list)))
    # ax.set_xticklabels(tactic_list, rotation=45, ha='right', fontsize=8)
    # ax.set_yticklabels(tactic_list, fontsize=8)
    # for i in range(len(tactic_list)):
    #     for j in range(len(tactic_list)):
    #         ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})',
    #                 ha='center', va='center', fontsize=7)
    # ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    # ax.set_title('Per-Flow Tactic Confusion Matrix')
    # fig.colorbar(im)
    # fig.tight_layout()
    # fig.savefig(os.path.join(EVAL_DIR, 'perflow_confusion.png'), dpi=150)
    # plt.close(fig)

    # ══════════════════════════════════════════════════════
    # SECTION 4: WINDOW-LEVEL SOFT TACTIC PREDICTION
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SECTION 4: WINDOW-LEVEL SOFT TACTIC PREDICTION")
    print(f"{'=' * 70}")

    # Build cluster_scores from KB (rule-based)
    cluster_scores_kb = {}
    for k in np.unique(clusters):
        entry = kb.lookup(int(k))
        if entry:
            cluster_scores_kb[int(k)] = {
                'behaviour_scores': entry['behaviour_scores'],
                'tactic_scores': entry['tactic_scores'],
                'primary_behaviour': entry['primary_behaviour'],
                'primary_tactic': entry['primary_tactic'],
            }
        else:
            cluster_scores_kb[int(k)] = {
                'behaviour_scores': {'unknown': 1.0},
                'tactic_scores': {'Unknown': 1.0},
                'primary_behaviour': 'unknown',
                'primary_tactic': 'Unknown',
            }
    new_kb = KnowledgeBase()
    for k, cs in cluster_scores_rules.items():
        new_kb.entries[str(k)] = {
            'behaviour_scores': cs['behaviour_scores'],
            'tactic_scores': cs['tactic_scores'],
            'primary_behaviour': cs['primary_behaviour'],
            'primary_tactic': cs['primary_tactic'],
            'cluster_size': int((clusters == k).sum()),
            'cluster_proportion': float((clusters == k).sum()) / len(clusters),
        }
    new_kb.save(KB_PATH)
    print(f"KB rebuilt: {len(new_kb.entries)} entries saved to {KB_PATH}")
    # ── Rule-based windows ──
    print(f"\n  === RULE-BASED SCORING ===")
    for ws in [20, 50]:
        print(f"\n  --- Rule-based Window size = {ws} ---")
        aggregator = SoftWindowAggregator(
            window_size=ws, tactic_threshold=0.10)
        windows = aggregator.create_windows(
            src_train, clusters, cluster_scores_kb, y_true=y_train)

        soft_eval = SoftTacticEvaluator()
        metrics, df = soft_eval.evaluate_windows(
            windows, gt_mapper,
            save_dir=os.path.join(EVAL_DIR, f'rules_ws_{ws}'))

        if ws == 50:
            for k, v in metrics.items():
                all_metrics[f'window_rules_{k}'] = v

    # ── LLM-based windows ──
    if has_llm_scores:
        print(f"\n  === LLM-BASED SCORING ===")
        for ws in [20, 50]:
            print(f"\n  --- LLM Window size = {ws} ---")
            aggregator = SoftWindowAggregator(
                window_size=ws, tactic_threshold=0.10)
            windows_llm = aggregator.create_windows(
                src_train, clusters, cluster_scores_llm, y_true=y_train)

            soft_eval = SoftTacticEvaluator()
            metrics_llm, df_llm = soft_eval.evaluate_windows(
                windows_llm, gt_mapper,
                save_dir=os.path.join(EVAL_DIR, f'llm_ws_{ws}'))

            if ws == 50:
                for k, v in metrics_llm.items():
                    all_metrics[f'window_llm_{k}'] = v

    # ══════════════════════════════════════════════════════
    # SECTION 5: BAYESIAN FUSION (Rule-based × LLM at behaviour level)
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SECTION 5: BAYESIAN FUSION (behaviour-level)")
    print(f"{'=' * 70}")

    if has_llm_scores:
        # Fuse rule-based and LLM tactic scores per cluster
        # P_fused(tactic|cluster) = w1 × P_rules(tactic|cluster)
        #                         + w2 × P_llm(tactic|cluster)
        w1, w2 = 0.5, 0.5  # Rule-based weighted higher (better agreement)

        cluster_scores_fused = {}
        for k in np.unique(clusters):
            k = int(k)
            rules = cluster_scores_kb.get(k, {}).get('tactic_scores', {})
            llm = cluster_scores_llm.get(k, {}).get('tactic_scores', {})

            if not llm:
                cluster_scores_fused[k] = cluster_scores_kb.get(k, {})
                continue

            # Fuse tactic distributions
            all_tactics = set(list(rules.keys()) + list(llm.keys()))
            fused_tactics = {}
            for t in all_tactics:
                fused_tactics[t] = w1 * rules.get(t, 0) + w2 * llm.get(t, 0)

            # Normalize
            total = sum(fused_tactics.values())
            if total > 0:
                fused_tactics = {t: s / total for t, s in fused_tactics.items()}

            # Fuse behaviour distributions
            rules_bhv = cluster_scores_kb.get(k, {}).get('behaviour_scores', {})
            llm_bhv = cluster_scores_llm.get(k, {}).get('behaviour_scores', {})
            all_bhvs = set(list(rules_bhv.keys()) + list(llm_bhv.keys()))
            fused_bhv = {}
            for b in all_bhvs:
                fused_bhv[b] = w1 * rules_bhv.get(b, 0) + w2 * llm_bhv.get(b, 0)
            b_total = sum(fused_bhv.values())
            if b_total > 0:
                fused_bhv = {b: s / b_total for b, s in fused_bhv.items()}

            cluster_scores_fused[k] = {
                'tactic_scores': fused_tactics,
                'behaviour_scores': fused_bhv,
                'primary_tactic': max(fused_tactics, key=fused_tactics.get),
                'primary_behaviour': max(fused_bhv, key=fused_bhv.get),
            }

        print(f"  Fusion weights: rule={w1}, llm={w2}")
        print(f"  Fused clusters: {len(cluster_scores_fused)}")

        # Per-flow fused prediction
        flow_primaries_fused = []
        for i in range(len(clusters)):
            k = int(clusters[i])
            cs = cluster_scores_fused.get(k, {})
            flow_primaries_fused.append(cs.get('primary_tactic', 'Unknown'))
        flow_primaries_fused = np.array(flow_primaries_fused)

        fused_acc = (flow_primaries_fused == true_tactics).mean()
        print(f"\n  Per-flow accuracy (rules):  {flow_acc:.4f}")
        print(f"  Per-flow accuracy (fused):  {fused_acc:.4f}")
        print(f"  Improvement: {(fused_acc - flow_acc) * 100:+.2f}%")

        fused_report = classification_report(
            true_tactics, flow_primaries_fused,
            labels=tactic_list, output_dict=True, zero_division=0,digits=4)
        print(f"\n{classification_report(true_tactics, flow_primaries_fused, labels=tactic_list, zero_division=0, digits=4)}")

        all_metrics['perflow_accuracy_fused'] = fused_acc
        all_metrics['perflow_macro_f1_fused'] = fused_report.get(
            'macro avg', {}).get('f1-score', 0)

        # Window-level fused prediction
        for ws in [20, 50]:
            print(f"\n  --- Fused Window size = {ws} ---")
            aggregator = SoftWindowAggregator(
                window_size=ws, tactic_threshold=0.10)
            fused_windows = aggregator.create_windows(
                src_train, clusters, cluster_scores_fused, y_true=y_train)

            soft_eval = SoftTacticEvaluator()
            f_metrics, f_df = soft_eval.evaluate_windows(
                fused_windows, gt_mapper,
                save_dir=os.path.join(EVAL_DIR, f'fused_ws_{ws}'))

            if ws == 50:
                for k, v in f_metrics.items():
                    all_metrics[f'fusion_{k}'] = v
    else:
        print("  Skipped — LLM scores not available for fusion")

    # ══════════════════════════════════════════════════════
    # SECTION 6: ANOMALY DETECTION (binary)
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SECTION 6: ANOMALY DETECTION (AUC-ROC)")
    print(f"{'=' * 70}")

    # Binary labels: 1=attack, 0=benign
    binary_true = (true_tactics != 'None').astype(int)

    if len(np.unique(binary_true)) > 1:
        # Score 1: reconstruction error
        recon_auc = roc_auc_score(binary_true, data['recon_errors'])
        recon_ap = average_precision_score(binary_true, data['recon_errors'])

        # Score 2: negative GMM log-likelihood
        neg_gmm = -data['gmm_scores']
        gmm_auc = roc_auc_score(binary_true, neg_gmm)
        gmm_ap = average_precision_score(binary_true, neg_gmm)

        # Score 3: combined
        combined = (data['recon_errors'] / data['recon_errors'].max() +
                    neg_gmm / neg_gmm.max()) / 2
        combined_auc = roc_auc_score(binary_true, combined)
        combined_ap = average_precision_score(binary_true, combined)

        print(f"  {'Score':>20s} {'AUC-ROC':>10s} {'AUC-PR':>10s}")
        print(f"  {'-' * 42}")
        print(f"  {'Recon Error':>20s} {recon_auc:>10.4f} {recon_ap:>10.4f}")
        print(f"  {'GMM (-loglik)':>20s} {gmm_auc:>10.4f} {gmm_ap:>10.4f}")
        print(f"  {'Combined':>20s} {combined_auc:>10.4f} {combined_ap:>10.4f}")

        all_metrics.update({
            'auc_recon': recon_auc, 'ap_recon': recon_ap,
            'auc_gmm': gmm_auc, 'ap_gmm': gmm_ap,
            'auc_combined': combined_auc, 'ap_combined': combined_ap,
        })

        # ROC curve plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, scores, color in [
            ('Recon Error', data['recon_errors'], 'blue'),
            ('GMM Score', neg_gmm, 'green'),
            ('Combined', combined, 'red'),
        ]:
            fpr, tpr, _ = roc_curve(binary_true, scores)
            auc = roc_auc_score(binary_true, scores)
            axes[0].plot(fpr, tpr, color=color,
                         label=f'{name} (AUC={auc:.3f})')
            prec, rec, _ = precision_recall_curve(binary_true, scores)
            ap = average_precision_score(binary_true, scores)
            axes[1].plot(rec, prec, color=color,
                         label=f'{name} (AP={ap:.3f})')

        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes[0].set_xlabel('FPR');
        axes[0].set_ylabel('TPR')
        axes[0].set_title('ROC Curve (Attack vs Benign)')
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel('Recall');
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(EVAL_DIR, 'anomaly_detection.png'), dpi=150)
        plt.close(fig)

    # ══════════════════════════════════════════════════════
    # SECTION 7: CLUSTER LABELING STRATEGY COMPARISON
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SECTION 7: CLUSTER LABELING STRATEGY COMPARISON")
    print(f"{'=' * 70}")

    from src.evaluation.labeling_eval import evaluate_all_strategies
    labeling_results = evaluate_all_strategies(
        cluster_summaries=data['summaries'],
        cluster_assignments=clusters,
        y_train=y_train,
        gt_mapper=gt_mapper,
        checkpoint=data.get('checkpoint'))

    for r in labeling_results:
        all_metrics[f"label_{r['name'].replace(' ', '_')}_agreement"] = r['hard_agreement']
        all_metrics[f"label_{r['name'].replace(' ', '_')}_jaccard"] = r['soft_jaccard']
        all_metrics[f"label_{r['name'].replace(' ', '_')}_kl"] = r['kl_divergence']

    # ══════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SUMMARY — ALL METRICS FOR PAPER")
    print(f"{'=' * 70}")

    print(f"\n  CLUSTERING QUALITY:")
    print(f"    Clusters:              {all_metrics.get('n_clusters')}")
    print(f"    Mean Purity:           {all_metrics.get('mean_purity', 0):.4f}")
    print(f"    Weighted Purity:       {all_metrics.get('weighted_purity', 0):.4f}")
    print(f"    NMI (attack):          {all_metrics.get('nmi_attack', 0):.4f}")
    print(f"    NMI (tactic):          {all_metrics.get('nmi_tactic', 0):.4f}")
    print(f"    NMI (behaviour):       {all_metrics.get('nmi_behaviour', 0):.4f}")
    print(f"    Silhouette (attack):   {all_metrics.get('sil_attack', 0):.4f}")
    print(f"    Silhouette (tactic):   {all_metrics.get('sil_tactic', 0):.4f}")
    print(f"    Silhouette (behaviour):{all_metrics.get('sil_behaviour', 0):.4f}")
    print(f"    Silhouette (cluster):  {all_metrics.get('sil_cluster', 0):.4f}")

    print(f"\n  PER-FLOW TACTIC PREDICTION:")
    print(f"    Accuracy:              {all_metrics.get('perflow_accuracy', 0):.4f}")
    print(f"    Macro F1:              {all_metrics.get('perflow_macro_f1', 0):.4f}")
    print(f"    Weighted F1:           {all_metrics.get('perflow_weighted_f1', 0):.4f}")

    print(f"\n  WINDOW-LEVEL (W=20):")
    print(f"    Top-1 Accuracy:        {all_metrics.get('window_top1_accuracy', 0):.4f}")
    print(f"    Coverage (Top-k):      {all_metrics.get('window_coverage', 0):.4f}")
    print(f"    Mean Jaccard:          {all_metrics.get('window_mean_jaccard', 0):.4f}")
    print(f"    Mean Soft Jaccard:     {all_metrics.get('window_mean_soft_jaccard', 0):.4f}")
    print(f"    Mean Cosine:           {all_metrics.get('window_mean_cosine', 0):.4f}")
    print(f"    Mean KL Divergence:    {all_metrics.get('window_mean_kl_divergence', 0):.4f}")

    print(f"\n  ANOMALY DETECTION:")
    print(f"    AUC-ROC (recon):       {all_metrics.get('auc_recon', 0):.4f}")
    print(f"    AUC-ROC (GMM):         {all_metrics.get('auc_gmm', 0):.4f}")
    print(f"    AUC-ROC (combined):    {all_metrics.get('auc_combined', 0):.4f}")

    print(f"\n  CLUSTER LABELING COMPARISON:")
    for r in labeling_results:
        print(f"    {r['name']:<18s}: agree={r['hard_agreement']:.4f}  "
              f"jaccard={r['soft_jaccard']:.4f}  kl={r['kl_divergence']:.4f}")

    # Save all metrics
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(os.path.join(EVAL_DIR, 'all_metrics.csv'), index=False)
    print(f"\n  All metrics saved: {EVAL_DIR}/all_metrics.csv")
    print(f"  Plots saved: {EVAL_DIR}/")
    # # ══════════════════════════════════════════════════════
    # # SECTION 4: WINDOW-LEVEL SOFT TACTIC PREDICTION (KB-based)
    # # ══════════════════════════════════════════════════════
    # print(f"\n{'='*70}")
    # print("SECTION 4: WINDOW-LEVEL SOFT TACTIC PREDICTION (KB-based)")
    # print(f"{'='*70}")
    #
    # # Build cluster_scores from KB entries
    # cluster_scores_kb = {}
    # for k in np.unique(clusters):
    #     entry = kb.lookup(int(k))
    #     if entry:
    #         cluster_scores_kb[int(k)] = {
    #             'behaviour_scores': entry['behaviour_scores'],
    #             'tactic_scores': entry['tactic_scores'],
    #             'primary_behaviour': entry['primary_behaviour'],
    #             'primary_tactic': entry['primary_tactic'],
    #         }
    #     else:
    #         cluster_scores_kb[int(k)] = {
    #             'behaviour_scores': {'unknown': 1.0},
    #             'tactic_scores': {'Unknown': 1.0},
    #             'primary_behaviour': 'unknown',
    #             'primary_tactic': 'Unknown',
    #         }
    #
    # for ws in [10, 20, 50]:
    #     print(f"\n  --- Window size = {ws} ---")
    #     aggregator = SoftWindowAggregator(
    #         window_size=ws, tactic_threshold=0.10)
    #     windows = aggregator.create_windows(
    #         src_train, clusters, cluster_scores_kb, y_true=y_train)
    #
    #     soft_eval = SoftTacticEvaluator()
    #     metrics, df = soft_eval.evaluate_windows(
    #         windows, gt_mapper,
    #         save_dir=os.path.join(EVAL_DIR, f'ws_{ws}'))
    #
    #     if ws == 50:
    #         for k, v in metrics.items():
    #             all_metrics[f'window_{k}'] = v
    #
    # # ══════════════════════════════════════════════════════
    # # SECTION 5: ANOMALY DETECTION (binary)
    # # ══════════════════════════════════════════════════════
    # print(f"\n{'='*70}")
    # print("SECTION 5: ANOMALY DETECTION (AUC-ROC)")
    # print(f"{'='*70}")
    #
    # # Binary labels: 1=attack, 0=benign
    # binary_true = (true_tactics != 'None').astype(int)
    #
    # if len(np.unique(binary_true)) > 1:
    #     # Score 1: reconstruction error
    #     recon_auc = roc_auc_score(binary_true, data['recon_errors'])
    #     recon_ap = average_precision_score(binary_true, data['recon_errors'])
    #
    #     # Score 2: negative GMM log-likelihood
    #     neg_gmm = -data['gmm_scores']
    #     gmm_auc = roc_auc_score(binary_true, neg_gmm)
    #     gmm_ap = average_precision_score(binary_true, neg_gmm)
    #
    #     # Score 3: combined
    #     combined = (data['recon_errors'] / data['recon_errors'].max() +
    #                 neg_gmm / neg_gmm.max()) / 2
    #     combined_auc = roc_auc_score(binary_true, combined)
    #     combined_ap = average_precision_score(binary_true, combined)
    #
    #     print(f"  {'Score':>20s} {'AUC-ROC':>10s} {'AUC-PR':>10s}")
    #     print(f"  {'-'*42}")
    #     print(f"  {'Recon Error':>20s} {recon_auc:>10.4f} {recon_ap:>10.4f}")
    #     print(f"  {'GMM (-loglik)':>20s} {gmm_auc:>10.4f} {gmm_ap:>10.4f}")
    #     print(f"  {'Combined':>20s} {combined_auc:>10.4f} {combined_ap:>10.4f}")
    #
    #     all_metrics.update({
    #         'auc_recon': recon_auc, 'ap_recon': recon_ap,
    #         'auc_gmm': gmm_auc, 'ap_gmm': gmm_ap,
    #         'auc_combined': combined_auc, 'ap_combined': combined_ap,
    #     })
    #
    #     # ROC curve plot
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #     for name, scores, color in [
    #         ('Recon Error', data['recon_errors'], 'blue'),
    #         ('GMM Score', neg_gmm, 'green'),
    #         ('Combined', combined, 'red'),
    #     ]:
    #         fpr, tpr, _ = roc_curve(binary_true, scores)
    #         auc = roc_auc_score(binary_true, scores)
    #         axes[0].plot(fpr, tpr, color=color,
    #                      label=f'{name} (AUC={auc:.3f})')
    #         prec, rec, _ = precision_recall_curve(binary_true, scores)
    #         ap = average_precision_score(binary_true, scores)
    #         axes[1].plot(rec, prec, color=color,
    #                      label=f'{name} (AP={ap:.3f})')
    #
    #     axes[0].plot([0,1], [0,1], 'k--', alpha=0.3)
    #     axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    #     axes[0].set_title('ROC Curve (Attack vs Benign)')
    #     axes[0].legend(fontsize=8)
    #     axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    #     axes[1].set_title('Precision-Recall Curve')
    #     axes[1].legend(fontsize=8)
    #     fig.tight_layout()
    #     fig.savefig(os.path.join(EVAL_DIR, 'anomaly_detection.png'), dpi=150)
    #     plt.close(fig)
    #
    # # ══════════════════════════════════════════════════════
    # # SECTION 6: CLUSTER LABELING STRATEGY COMPARISON
    # # ══════════════════════════════════════════════════════
    # print(f"\n{'='*70}")
    # print("SECTION 6: CLUSTER LABELING STRATEGY COMPARISON")
    # print(f"{'='*70}")
    #
    # from src.evaluation.labeling_eval import evaluate_all_strategies
    # labeling_results = evaluate_all_strategies(
    #     cluster_summaries=data['summaries'],
    #     cluster_assignments=clusters,
    #     y_train=y_train,
    #     gt_mapper=gt_mapper,
    #     checkpoint=data.get('checkpoint'))
    #
    # for r in labeling_results:
    #     all_metrics[f"label_{r['name'].replace(' ', '_')}_agreement"] = r['hard_agreement']
    #     all_metrics[f"label_{r['name'].replace(' ', '_')}_jaccard"] = r['soft_jaccard']
    #     all_metrics[f"label_{r['name'].replace(' ', '_')}_kl"] = r['kl_divergence']
    #
    # # ══════════════════════════════════════════════════════
    # # SUMMARY TABLE
    # # ══════════════════════════════════════════════════════
    # print(f"\n{'='*70}")
    # print("SUMMARY — ALL METRICS FOR PAPER")
    # print(f"{'='*70}")
    #
    # print(f"\n  CLUSTERING QUALITY:")
    # print(f"    Clusters:              {all_metrics.get('n_clusters')}")
    # print(f"    Mean Purity:           {all_metrics.get('mean_purity', 0):.4f}")
    # print(f"    Weighted Purity:       {all_metrics.get('weighted_purity', 0):.4f}")
    # print(f"    NMI (attack):          {all_metrics.get('nmi_attack', 0):.4f}")
    # print(f"    NMI (tactic):          {all_metrics.get('nmi_tactic', 0):.4f}")
    # print(f"    NMI (behaviour):       {all_metrics.get('nmi_behaviour', 0):.4f}")
    # print(f"    Silhouette (attack):   {all_metrics.get('sil_attack', 0):.4f}")
    # print(f"    Silhouette (tactic):   {all_metrics.get('sil_tactic', 0):.4f}")
    # print(f"    Silhouette (behaviour):{all_metrics.get('sil_behaviour', 0):.4f}")
    # print(f"    Silhouette (cluster):  {all_metrics.get('sil_cluster', 0):.4f}")
    #
    # print(f"\n  PER-FLOW TACTIC PREDICTION:")
    # print(f"    Accuracy:              {all_metrics.get('perflow_accuracy', 0):.4f}")
    # print(f"    Macro F1:              {all_metrics.get('perflow_macro_f1', 0):.4f}")
    # print(f"    Weighted F1:           {all_metrics.get('perflow_weighted_f1', 0):.4f}")
    #
    # print(f"\n  WINDOW-LEVEL (W=20):")
    # print(f"    Top-1 Accuracy:        {all_metrics.get('window_top1_accuracy', 0):.4f}")
    # print(f"    Coverage (Top-k):      {all_metrics.get('window_coverage', 0):.4f}")
    # print(f"    Mean Jaccard:          {all_metrics.get('window_mean_jaccard', 0):.4f}")
    # print(f"    Mean Soft Jaccard:     {all_metrics.get('window_mean_soft_jaccard', 0):.4f}")
    # print(f"    Mean Cosine:           {all_metrics.get('window_mean_cosine', 0):.4f}")
    # print(f"    Mean KL Divergence:    {all_metrics.get('window_mean_kl_divergence', 0):.4f}")
    #
    # print(f"\n  ANOMALY DETECTION:")
    # print(f"    AUC-ROC (recon):       {all_metrics.get('auc_recon', 0):.4f}")
    # print(f"    AUC-ROC (GMM):         {all_metrics.get('auc_gmm', 0):.4f}")
    # print(f"    AUC-ROC (combined):    {all_metrics.get('auc_combined', 0):.4f}")
    #
    # print(f"\n  CLUSTER LABELING COMPARISON:")
    # for r in labeling_results:
    #     print(f"    {r['name']:<18s}: agree={r['hard_agreement']:.4f}  "
    #           f"jaccard={r['soft_jaccard']:.4f}  kl={r['kl_divergence']:.4f}")
    #
    # # Save all metrics
    # metrics_df = pd.DataFrame([all_metrics])
    # metrics_df.to_csv(os.path.join(EVAL_DIR, 'all_metrics.csv'), index=False)
    # print(f"\n  All metrics saved: {EVAL_DIR}/all_metrics.csv")
    # print(f"  Plots saved: {EVAL_DIR}/")


if __name__ == '__main__':
    run_validation()