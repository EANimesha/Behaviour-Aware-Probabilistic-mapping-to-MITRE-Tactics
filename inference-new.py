# """
# Inference Pipeline — KB-Based Soft Tactic Prediction
# ======================================================
#
# Uses the trained KB (soft behaviour/tactic distributions per cluster)
# for tactic prediction. No rule-based recomputation needed.
#
# Flow: new data → encoder → z → GMM cluster → KB lookup → window aggregation
#
# For zero-day: flows that don't fit any cluster (novelty) are flagged.
# """
#
# import os
# import numpy as np
# import pandas as pd
# import torch
# import logging
# import sys
# from collections import Counter
#
# from matplotlib import pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, silhouette_score, normalized_mutual_info_score
#
# from src.data.loader import load_dataset
# from src.data.preprocess import Preprocessor
# from src.data.graph_builder import NetworkGraphBuilder
# from src.encoder.feature import FeatureEncoder
# from src.encoder.context import ContextEncoder
# from src.encoder.behaviour import TransformerBehaviourEncoder
# from src.encoder.jointencoder import JointEncoder
# from src.evaluation.soft_eval import SoftTacticEvaluator
# from src.mapping.soft_cluster_scorer import SoftWindowAggregator
# from src.streaming.streaming_gmm import StreamingDPGMM
# from src.clustering.summarizer import ClusterSummarizer
# from src.llm.reasoning import LLMLabeler
# from src.knowledge_base.kb import KnowledgeBase
# from src.fusion.fusion import BayesianFusion
# from src.evaluation.ground_truth import GroundTruthMapper
# from src.evaluation.metrics import TacticEvaluator
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'
#
# if DATASET == 'unsw_nb15':
#     from src.config.config_unsw import (
#         DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
#         FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
#         Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
#     )
# elif DATASET == 'cicids2018':
#     from src.config.config_cicids import (
#         DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
#         FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
#         Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
#     )
# else:
#     from src.config.config import (
#         DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
#         FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
#         Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
#     )
#
# EVAL_DIR = os.path.join(CHECKPOINT_DIR, "inference_results")
# os.makedirs(EVAL_DIR, exist_ok=True)
#
# def load_checkpoint(model_path, kb_path, device='cpu'):
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#     cfg = checkpoint['config']
#
#     feature_enc = FeatureEncoder(cfg['feature_dim'], cfg['feature_output_dim'])
#     context_enc = ContextEncoder(cfg['node_feature_dim'], 32,
#                                   cfg['context_output_dim'], 0.0)
#     behaviour_enc = TransformerBehaviourEncoder(
#         cfg['feature_dim'], 64, cfg['behaviour_output_dim'], 2, 4, 0.0)
#
#     joint_enc = JointEncoder(
#         feature_enc, context_enc, behaviour_enc,
#         cfg['feature_output_dim'], cfg['context_output_dim'],
#         cfg['behaviour_output_dim'], cfg['z_dim'],
#         n_behaviours=cfg.get('n_behaviours', 5),
#         lambda_bhv=cfg.get('lambda_bhv', 1.0))
#
#     joint_enc.load_state_dict(checkpoint['joint_encoder_state'])
#     joint_enc.to(device)
#     joint_enc.eval()
#
#     gmm_cfg = checkpoint['gmm_config']
#     gmm = StreamingDPGMM(
#         gmm_cfg['max_components'], gmm_cfg['covariance_type'],
#         gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
#     gmm.model = checkpoint['gmm_model']
#
#     kb = KnowledgeBase()
#     kb.load(kb_path)
#
#     active = np.sum(gmm.model.weights_ > gmm_cfg['weight_threshold'])
#     print(f"  Loaded: encoder(z={cfg['z_dim']}), "
#           f"GMM({active} clusters), KB({len(kb.entries)} entries)")
#
#     return joint_enc, gmm, kb, checkpoint
#
#
# def extract_embeddings(joint_enc, graph, X_tensor, node_features,
#                        context_emb, temporal_seqs, temporal_mask,
#                        batch_size, device):
#     joint_enc.eval()
#     all_z = []
#     with torch.no_grad():
#         for s in range(0, len(X_tensor), batch_size):
#             e = min(s + batch_size, len(X_tensor))
#             z = joint_enc.get_embeddings(
#                 graph=graph, flow_features=X_tensor[s:e],
#                 node_features=node_features,
#                 context_embeddings=context_emb[s:e],
#                 temporal_sequences=temporal_seqs[s:e],
#                 temporal_mask=temporal_mask[s:e])
#             all_z.append(z.cpu())
#     return torch.cat(all_z, dim=0).numpy()
#
#
# def run_inference(test_csv_path=None, model_path=MODEL_PATH, kb_path=KB_PATH):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     joint_enc, gmm, kb, checkpoint = load_checkpoint(model_path, kb_path, device)
#     cfg = checkpoint['config']
#     saved_scaler = checkpoint['preprocessor_scaler']
#     saved_features = checkpoint['feature_names']
#
#     # ── Stage 1: Load + preprocess ──
#     print("\n" + "="*60)
#     print("STAGE 1: Load Test Data")
#     print("="*60)
#
#     preprocessor = Preprocessor()
#     if test_csv_path:
#         df = load_dataset(test_csv_path)
#         preprocessor.scaler = saved_scaler
#         preprocessor.fitted = True
#         preprocessor.final_feature_names = saved_features
#         X_test, src_test, dst_test = preprocessor.transform_new(df)
#         y_test = df['Attack'].values if 'Attack' in df.columns else None
#     else:
#         df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
#         data = preprocessor.preprocess_dataset(df)
#         X_test = data['X_test']
#         src_test, dst_test = data['src_test'], data['dst_test']
#         y_test = data['y_test']
#
#     print(f"  Test: {len(X_test)} flows")
#
#     # ── Stage 2: Graph + temporal ──
#     builder = NetworkGraphBuilder(seq_length=cfg['seq_length'])
#     graph, temporal_seqs = builder.build_graph(X_test, src_test, dst_test)
#     graph = graph.to(device)
#     temporal_seqs = temporal_seqs.to(device)
#     node_features = builder.node_features.to(device)
#
#     with torch.no_grad():
#         node_emb = joint_enc.context_encoder(graph, node_features)
#         context_emb = builder.get_ip_embeddings(
#             node_emb, src_test, dst_test).to(device)
#
#     X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
#     t_mask = torch.ones(len(X_test), temporal_seqs.shape[1]).to(device)
#
#     # ── Stage 3: Extract z ──
#     print("\n  Extracting z embeddings...")
#     Z = extract_embeddings(
#         joint_enc, graph, X_tensor, node_features,
#         context_emb, temporal_seqs, t_mask, VAE_BATCH_SIZE, device)
#     print(f"  Z_test: {Z.shape}")
#
#     # ── Stage 4: DP-GMM predict ──
#     print("\n" + "="*60)
#     print("STAGE 4: DP-GMM Predict")
#     print("="*60)
#
#     assignments, soft_probs, log_liks, novelty_flags = gmm.predict(Z)
#     clusters = gmm.model.predict(Z)
#     gmm_scores = gmm.model.score_samples(Z)
#
#     # Novelty detection
#     novelty_threshold = np.percentile(gmm_scores, 5)
#     novelty_mask = gmm_scores < novelty_threshold
#     n_novel = novelty_mask.sum()
#     print(f"  Clusters: {len(np.unique(clusters))} unique")
#     print(f"  Novelty flows: {n_novel} ({n_novel / len(Z) * 100:.1f}%)")
#
#     gt_mapper = GroundTruthMapper(DATASET)
#     true_tactics = gt_mapper.map_labels(y_test) if y_test is not None else None
#     tactic_list = gt_mapper.get_tactic_list()
#     all_metrics = {}
#
#     if y_test is None:
#         print("  No ground truth — skipping evaluation")
#         return
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 1: CLUSTERING QUALITY (test)
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'=' * 70}")
#     print("SECTION 1: CLUSTERING QUALITY (test data)")
#     print(f"{'=' * 70}")
#
#     sample_size = min(5000, len(Z))
#     sil_attack = silhouette_score(Z, y_test, sample_size=sample_size)
#     sil_tactic = silhouette_score(Z, true_tactics, sample_size=sample_size)
#     sil_cluster = silhouette_score(Z, clusters, sample_size=sample_size)
#     nmi_attack = normalized_mutual_info_score(y_test, clusters)
#     nmi_tactic = normalized_mutual_info_score(true_tactics, clusters)
#
#     purities, sizes = [], []
#     for k in np.unique(clusters):
#         mask = clusters == k
#         if mask.sum() < 2: continue
#         maj = Counter(y_test[mask]).most_common(1)[0]
#         purities.append(maj[1] / mask.sum())
#         sizes.append(mask.sum())
#     mean_purity = np.mean(purities)
#     weighted_purity = np.average(purities, weights=sizes)
#
#     print(f"  Mean purity:      {mean_purity:.4f}")
#     print(f"  Weighted purity:  {weighted_purity:.4f}")
#     print(f"  NMI (attack):     {nmi_attack:.4f}")
#     print(f"  NMI (tactic):     {nmi_tactic:.4f}")
#     print(f"  Sil (attack):     {sil_attack:.4f}")
#     print(f"  Sil (tactic):     {sil_tactic:.4f}")
#     print(f"  Sil (cluster):    {sil_cluster:.4f}")
#
#     all_metrics.update({
#         'mean_purity': mean_purity, 'weighted_purity': weighted_purity,
#         'nmi_attack': nmi_attack, 'nmi_tactic': nmi_tactic,
#         'sil_attack': sil_attack, 'sil_tactic': sil_tactic,
#         'sil_cluster': sil_cluster,
#     })
#
#     # ══════════════════════════════════════════════════════
#     # Build scoring sources
#     # ══════════════════════════════════════════════════════
#     # Rule-based: from KB
#     cluster_scores_rules = {}
#     for k in np.unique(clusters):
#         entry = kb.lookup(int(k))
#         if entry:
#             cluster_scores_rules[int(k)] = {
#                 'behaviour_scores': entry['behaviour_scores'],
#                 'tactic_scores': entry['tactic_scores'],
#                 'primary_behaviour': entry['primary_behaviour'],
#                 'primary_tactic': entry['primary_tactic'],
#             }
#         else:
#             cluster_scores_rules[int(k)] = {
#                 'behaviour_scores': {'unknown': 1.0},
#                 'tactic_scores': {'Unknown': 1.0},
#                 'primary_behaviour': 'unknown',
#                 'primary_tactic': 'Unknown',
#             }
#
#     # LLM-based: from checkpoint
#     cluster_scores_llm = {}
#     cbm_hard = checkpoint.get('cluster_behaviour_map_hard', {})
#     if cbm_hard:
#         for k_str, entry in cbm_hard.items():
#             k = int(k_str)
#             if 'tactic_scores_llm' in entry and 'behaviour_scores_llm' in entry:
#                 cluster_scores_llm[k] = {
#                     'behaviour_scores': entry['behaviour_scores_llm'],
#                     'tactic_scores': entry['tactic_scores_llm'],
#                     'primary_behaviour': entry.get('primary_behaviour', 'unknown'),
#                     'primary_tactic': entry.get('primary_tactic', 'Unknown'),
#                 }
#     has_llm_scores = len(cluster_scores_llm) > 0
#     print(f"\n  Rule-based clusters: {len(cluster_scores_rules)}")
#     print(f"  LLM-based clusters: {len(cluster_scores_llm)}")
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 2: PER-FLOW TACTIC PREDICTION
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'=' * 70}")
#     print("SECTION 2: PER-FLOW TACTIC PREDICTION")
#     print(f"{'=' * 70}")
#
#     # Rule-based per-flow
#     print(f"\n  === RULE-BASED ===")
#     flow_tactics_rules = np.array([
#         cluster_scores_rules.get(int(clusters[i]), {}).get('primary_tactic', 'Unknown')
#         for i in range(len(clusters))])
#     flow_acc_rules = (flow_tactics_rules == true_tactics).mean()
#     print(f"  Accuracy: {flow_acc_rules:.4f}")
#     report_rules = classification_report(
#         true_tactics, flow_tactics_rules,
#         labels=tactic_list, output_dict=True, zero_division=0)
#     print(classification_report(
#         true_tactics, flow_tactics_rules,
#         labels=tactic_list, zero_division=0))
#
#     all_metrics['perflow_rules_accuracy'] = flow_acc_rules
#     all_metrics['perflow_rules_macro_f1'] = report_rules.get('macro avg', {}).get('f1-score', 0)
#     all_metrics['perflow_rules_weighted_f1'] = report_rules.get('weighted avg', {}).get('f1-score', 0)
#     for t in tactic_list:
#         if t in report_rules:
#             all_metrics[f'perflow_rules_{t}_f1'] = report_rules[t]['f1-score']
#             all_metrics[f'perflow_rules_{t}_precision'] = report_rules[t]['precision']
#             all_metrics[f'perflow_rules_{t}_recall'] = report_rules[t]['recall']
#
#     # LLM-based per-flow
#     if has_llm_scores:
#         print(f"\n  === LLM-BASED ===")
#         flow_tactics_llm = np.array([
#             cluster_scores_llm.get(int(clusters[i]), {}).get('primary_tactic', 'Unknown')
#             for i in range(len(clusters))])
#         flow_acc_llm = (flow_tactics_llm == true_tactics).mean()
#         print(f"  Accuracy: {flow_acc_llm:.4f}")
#         report_llm = classification_report(
#             true_tactics, flow_tactics_llm,
#             labels=tactic_list, output_dict=True, zero_division=0)
#         print(classification_report(
#             true_tactics, flow_tactics_llm,
#             labels=tactic_list, zero_division=0))
#
#         all_metrics['perflow_llm_accuracy'] = flow_acc_llm
#         all_metrics['perflow_llm_macro_f1'] = report_llm.get('macro avg', {}).get('f1-score', 0)
#         for t in tactic_list:
#             if t in report_llm:
#                 all_metrics[f'perflow_llm_{t}_f1'] = report_llm[t]['f1-score']
#
#         # Side-by-side
#         print(f"\n  {'Tactic':>20s} {'Rules F1':>10s} {'LLM F1':>10s}")
#         print(f"  {'-' * 42}")
#         for t in tactic_list:
#             r = report_rules.get(t, {}).get('f1-score', 0)
#             l = report_llm.get(t, {}).get('f1-score', 0)
#             better = '←' if r > l else '→' if l > r else '='
#             print(f"  {t:>20s} {r:>10.4f} {l:>10.4f} {better}")
#         print(f"  {'Accuracy':>20s} {flow_acc_rules:>10.4f} {flow_acc_llm:>10.4f}")
#
#     # Confusion matrix
#     cm = confusion_matrix(true_tactics, flow_tactics_rules, labels=tactic_list)
#     fig, ax = plt.subplots(figsize=(8, 6))
#     cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
#     im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
#     ax.set_xticks(range(len(tactic_list)))
#     ax.set_yticks(range(len(tactic_list)))
#     ax.set_xticklabels(tactic_list, rotation=45, ha='right', fontsize=8)
#     ax.set_yticklabels(tactic_list, fontsize=8)
#     for i in range(len(tactic_list)):
#         for j in range(len(tactic_list)):
#             ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.0%})',
#                     ha='center', va='center', fontsize=7)
#     ax.set_xlabel('Predicted');
#     ax.set_ylabel('True')
#     ax.set_title('Per-Flow Tactic Confusion (Inference, Rule-based)')
#     fig.colorbar(im);
#     fig.tight_layout()
#     fig.savefig(os.path.join(EVAL_DIR, 'inference_confusion.png'), dpi=150)
#     plt.close(fig)
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 3: WINDOW-LEVEL SOFT TACTIC PREDICTION
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'=' * 70}")
#     print("SECTION 3: WINDOW-LEVEL SOFT TACTIC PREDICTION")
#     print(f"{'=' * 70}")
#
#     # Rule-based windows
#     print(f"\n  === RULE-BASED SCORING ===")
#     for ws in [20, 50]:
#         print(f"\n  --- Rule-based Window size = {ws} ---")
#         aggregator = SoftWindowAggregator(window_size=ws, tactic_threshold=0.10)
#         windows = aggregator.create_windows(
#             src_test, clusters, cluster_scores_rules, y_true=y_test)
#         soft_eval = SoftTacticEvaluator()
#         metrics, df = soft_eval.evaluate_windows(
#             windows, gt_mapper,
#             save_dir=os.path.join(EVAL_DIR, f'rules_ws_{ws}'))
#         if ws == 50:
#             for k, v in metrics.items():
#                 all_metrics[f'window_rules_{k}'] = v
#
#     # LLM-based windows
#     if has_llm_scores:
#         print(f"\n  === LLM-BASED SCORING ===")
#         for ws in [20, 50]:
#             print(f"\n  --- LLM Window size = {ws} ---")
#             aggregator = SoftWindowAggregator(window_size=ws, tactic_threshold=0.10)
#             windows_llm = aggregator.create_windows(
#                 src_test, clusters, cluster_scores_llm, y_true=y_test)
#             soft_eval = SoftTacticEvaluator()
#             metrics_llm, df_llm = soft_eval.evaluate_windows(
#                 windows_llm, gt_mapper,
#                 save_dir=os.path.join(EVAL_DIR, f'llm_ws_{ws}'))
#             if ws == 50:
#                 for k, v in metrics_llm.items():
#                     all_metrics[f'window_llm_{k}'] = v
#
#     # ══════════════════════════════════════════════════════
#     # SECTION 4: BAYESIAN FUSION (behaviour-level)
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'=' * 70}")
#     print("SECTION 4: BAYESIAN FUSION (behaviour-level)")
#     print(f"{'=' * 70}")
#
#     if has_llm_scores:
#         w1, w2 = 0.6, 0.4
#
#         cluster_scores_fused = {}
#         for k in np.unique(clusters):
#             k = int(k)
#             rules = cluster_scores_rules.get(k, {}).get('tactic_scores', {})
#             llm = cluster_scores_llm.get(k, {}).get('tactic_scores', {})
#             if not llm:
#                 cluster_scores_fused[k] = cluster_scores_rules.get(k, {})
#                 continue
#
#             all_tactics = set(list(rules.keys()) + list(llm.keys()))
#             fused_tactics = {}
#             for t in all_tactics:
#                 fused_tactics[t] = w1 * rules.get(t, 0) + w2 * llm.get(t, 0)
#             total = sum(fused_tactics.values())
#             if total > 0:
#                 fused_tactics = {t: s / total for t, s in fused_tactics.items()}
#
#             rules_bhv = cluster_scores_rules.get(k, {}).get('behaviour_scores', {})
#             llm_bhv = cluster_scores_llm.get(k, {}).get('behaviour_scores', {})
#             all_bhvs = set(list(rules_bhv.keys()) + list(llm_bhv.keys()))
#             fused_bhv = {}
#             for b in all_bhvs:
#                 fused_bhv[b] = w1 * rules_bhv.get(b, 0) + w2 * llm_bhv.get(b, 0)
#             b_total = sum(fused_bhv.values())
#             if b_total > 0:
#                 fused_bhv = {b: s / b_total for b, s in fused_bhv.items()}
#
#             cluster_scores_fused[k] = {
#                 'tactic_scores': fused_tactics,
#                 'behaviour_scores': fused_bhv,
#                 'primary_tactic': max(fused_tactics, key=fused_tactics.get),
#                 'primary_behaviour': max(fused_bhv, key=fused_bhv.get),
#             }
#
#         print(f"  Fusion weights: rule={w1}, llm={w2}")
#
#         # Per-flow fused
#         flow_fused = np.array([
#             cluster_scores_fused.get(int(clusters[i]), {}).get('primary_tactic', 'Unknown')
#             for i in range(len(clusters))])
#         fused_acc = (flow_fused == true_tactics).mean()
#         print(f"\n  Per-flow accuracy (rules):  {flow_acc_rules:.4f}")
#         print(f"  Per-flow accuracy (fused):  {fused_acc:.4f}")
#         print(f"  Improvement: {(fused_acc - flow_acc_rules) * 100:+.2f}%")
#         print(classification_report(
#             true_tactics, flow_fused,
#             labels=tactic_list, zero_division=0))
#
#         all_metrics['perflow_fused_accuracy'] = fused_acc
#
#         # Window-level fused
#         for ws in [20, 50]:
#             print(f"\n  --- Fused Window size = {ws} ---")
#             aggregator = SoftWindowAggregator(window_size=ws, tactic_threshold=0.10)
#             fused_windows = aggregator.create_windows(
#                 src_test, clusters, cluster_scores_fused, y_true=y_test)
#             soft_eval = SoftTacticEvaluator()
#             f_metrics, f_df = soft_eval.evaluate_windows(
#                 fused_windows, gt_mapper,
#                 save_dir=os.path.join(EVAL_DIR, f'fused_ws_{ws}'))
#             if ws == 50:
#                 for k, v in f_metrics.items():
#                     all_metrics[f'fusion_{k}'] = v
#     else:
#         print("  Skipped — LLM scores not available")
#
#     # # ══════════════════════════════════════════════════════
#     # # SECTION 5: ANOMALY DETECTION
#     # # ══════════════════════════════════════════════════════
#     # print(f"\n{'=' * 70}")
#     # print("SECTION 5: ANOMALY DETECTION (AUC-ROC)")
#     # print(f"{'=' * 70}")
#     #
#     # binary_true = (true_tactics != 'None').astype(int)
#     # if len(np.unique(binary_true)) > 1:
#     #     neg_gmm = -gmm_scores
#     #     combined = (recon_errors / recon_errors.max() +
#     #                 neg_gmm / neg_gmm.max()) / 2
#     #
#     #     for name, scores in [('Recon Error', recon_errors),
#     #                          ('GMM (-loglik)', neg_gmm),
#     #                          ('Combined', combined)]:
#     #         auc = roc_auc_score(binary_true, scores)
#     #         ap = average_precision_score(binary_true, scores)
#     #         print(f"  {name:>15s}: AUC-ROC={auc:.4f}, AUC-PR={ap:.4f}")
#     #         key = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
#     #         all_metrics[f'auc_{key}'] = auc
#     #         all_metrics[f'ap_{key}'] = ap
#
#     # Novelty summary
#     if n_novel > 0:
#         print(f"\n  NOVELTY: {n_novel} flows flagged as potential zero-day")
#         if y_test is not None:
#             novel_attacks = Counter(y_test[novelty_mask])
#             print(f"  Novel flow attacks: {dict(novel_attacks)}")
#
#     # ══════════════════════════════════════════════════════
#     # SUMMARY
#     # ══════════════════════════════════════════════════════
#     print(f"\n{'=' * 70}")
#     print("SUMMARY — INFERENCE METRICS")
#     print(f"{'=' * 70}")
#
#     print(f"\n  CLUSTERING (test):")
#     print(f"    Mean Purity:      {all_metrics.get('mean_purity', 0):.4f}")
#     print(f"    Weighted Purity:  {all_metrics.get('weighted_purity', 0):.4f}")
#     print(f"    NMI (tactic):     {all_metrics.get('nmi_tactic', 0):.4f}")
#     print(f"    Sil (tactic):     {all_metrics.get('sil_tactic', 0):.4f}")
#
#     print(f"\n  PER-FLOW:")
#     print(f"    Rules accuracy:   {all_metrics.get('perflow_rules_accuracy', 0):.4f}")
#     if has_llm_scores:
#         print(f"    LLM accuracy:     {all_metrics.get('perflow_llm_accuracy', 0):.4f}")
#         print(f"    Fused accuracy:   {all_metrics.get('perflow_fused_accuracy', 0):.4f}")
#
#     print(f"\n  WINDOW (W=50):")
#     print(f"    Rules Top-1:      {all_metrics.get('window_rules_top1_accuracy', 0):.4f}")
#     print(f"    Rules Cosine:     {all_metrics.get('window_rules_mean_cosine', 0):.4f}")
#     if has_llm_scores:
#         print(f"    LLM Top-1:        {all_metrics.get('window_llm_top1_accuracy', 0):.4f}")
#         print(f"    Fused Top-1:      {all_metrics.get('fusion_top1_accuracy', 0):.4f}")
#         print(f"    Fused Cosine:     {all_metrics.get('fusion_mean_cosine', 0):.4f}")
#
#     # Save all metrics
#     metrics_df = pd.DataFrame([all_metrics])
#     metrics_df.to_csv(os.path.join(EVAL_DIR, 'inference_metrics.csv'), index=False)
#     print(f"\n  All metrics saved: {EVAL_DIR}/")
#
#
# if __name__ == '__main__':
#     run_inference()
"""
Inference Pipeline — KB-Based Soft Tactic Prediction
======================================================

Uses the trained KB (soft behaviour/tactic distributions per cluster)
for tactic prediction. No rule-based recomputation needed.

Flow: new data → encoder → z → GMM cluster → KB lookup → window aggregation

Sections:
  1. Clustering Quality (test embeddings)
  2. Per-Flow Tactic Prediction (rule-based + LLM, side-by-side)
  3. Window-Level Soft Tactic (rule-based + LLM, W=20,50)
  4. Bayesian Fusion (per-flow + window)
  5. Anomaly Detection (AUC-ROC)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score, normalized_mutual_info_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'bot_iot'

if DATASET == 'unsw_nb15':
    from src.config.config_unsw import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
    )
elif DATASET == 'cicids2018':
    from src.config.config_cicids import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
    )
else:
    from src.config.config import (
        DATA_PATH, SAMPLE_SIZE, CHECKPOINT_DIR, MODEL_PATH, KB_PATH,
        FEATURE_OUTPUT_DIM, CONTEXT_OUTPUT_DIM, BEHAVIOUR_OUTPUT_DIM,
        Z_DIM, SEQ_LENGTH, NODE_FEATURE_DIM, VAE_BATCH_SIZE,
    )

from src.data.loader import load_dataset
from src.data.preprocess import Preprocessor
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

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "inference_results")
os.makedirs(EVAL_DIR, exist_ok=True)


def load_checkpoint(model_path, kb_path, device='cpu'):
    """Load encoder, GMM, and KB from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    feature_enc = FeatureEncoder(cfg['feature_dim'], cfg['feature_output_dim'])
    context_enc = ContextEncoder(
        cfg.get('node_feature_dim', NODE_FEATURE_DIM), 32,
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
    joint_enc.to(device)
    joint_enc.eval()

    gmm_cfg = checkpoint['gmm_config']
    gmm = StreamingDPGMM(
        gmm_cfg['max_components'], gmm_cfg['covariance_type'],
        gmm_cfg['weight_threshold'], gmm_cfg['novelty_threshold'])
    gmm.model = checkpoint['gmm_model']

    kb = KnowledgeBase()
    kb.load(kb_path)

    active = np.sum(gmm.model.weights_ > gmm_cfg['weight_threshold'])
    print(f"  Loaded: encoder(z={cfg['z_dim']}), "
          f"GMM({active} clusters), KB({len(kb.entries)} entries)")

    return joint_enc, gmm, kb, checkpoint


def encode_test_data(joint_enc, cfg, df_test, device, checkpoint):
    """Encode test data using the SAME preprocessor as training."""
    preprocessor = Preprocessor(clip_extremes=(DATASET != 'bot_iot'))

    # Use saved scaler and feature names from training
    if 'preprocessor_scaler' in checkpoint:
        preprocessor.scaler = checkpoint['preprocessor_scaler']
        preprocessor.fitted = True
        preprocessor.final_feature_names = checkpoint.get('feature_names', [])
        X_test, src_test, dst_test = preprocessor.transform_new(df_test)
    else:
        test_data = preprocessor.preprocess_dataset(df_test)
        X_test = test_data['X_train']
        src_test = test_data['src_train']
        dst_test = test_data['dst_train']

    builder = NetworkGraphBuilder(seq_length=cfg.get('seq_length', SEQ_LENGTH))
    graph, temporal_seqs = builder.build_graph(X_test, src_test, dst_test)
    graph = graph.to(device)
    temporal_seqs = temporal_seqs.to(device)
    node_features = builder.node_features.to(device)

    with torch.no_grad():
        node_emb = joint_enc.context_encoder(graph, node_features)
        context_emb = builder.get_ip_embeddings(
            node_emb, src_test, dst_test).to(device)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    t_mask = torch.ones(len(X_test), temporal_seqs.shape[1]).to(device)

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
    recon_errors = np.mean((H_fusion - H_recon) ** 2, axis=1)

    return Z, recon_errors, src_test, dst_test


def run_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"INFERENCE PIPELINE — {DATASET}")
    print(f"{'='*70}")

    # Load checkpoint + KB
    joint_enc, gmm, kb, checkpoint = load_checkpoint(
        MODEL_PATH, KB_PATH, device)
    cfg = checkpoint['config']

    # Load data
    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)

    # Balance if needed
    if DATASET == 'unsw_nb15' and 'Attack' in df.columns:
        from src.data.balancer import FastChronologicalBalancer
        from src.config.config_unsw import (
            MAX_BENIGN_SAMPLES, MAX_ATTACK_PER_CLASS, MIN_ATTACK_PER_CLASS)
        balancer = FastChronologicalBalancer(
            max_total_samples=120000,
            max_benign_samples=MAX_BENIGN_SAMPLES,
            max_attack_per_class=MAX_ATTACK_PER_CLASS,
            min_attack_per_class=MIN_ATTACK_PER_CLASS)
        df, _ = balancer.balance(df)
    elif DATASET == 'cicids2018' and 'Attack' in df.columns:
        from src.data.balancer import FastChronologicalBalancer
        from src.config.config_cicids import (
            MAX_BENIGN_SAMPLES, MAX_ATTACK_PER_CLASS, MIN_ATTACK_PER_CLASS)
        balancer = FastChronologicalBalancer(
            max_total_samples=120000,
            max_benign_samples=MAX_BENIGN_SAMPLES,
            max_attack_per_class=MAX_ATTACK_PER_CLASS,
            min_attack_per_class=MIN_ATTACK_PER_CLASS)
        df, _ = balancer.balance(df)

    # Train/test split (same as training)
    indices = np.arange(len(df))
    if 'Attack' in df.columns:
        _, idx_test = train_test_split(
            indices, test_size=0.2,
            stratify=df['Attack'].values, random_state=42)
    else:
        _, idx_test = train_test_split(
            indices, test_size=0.2, random_state=42)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    y_test = df_test['Attack'].values if 'Attack' in df_test.columns else None
    print(f"  Test flows: {len(df_test):,}")

    # Encode test data
    print("  Encoding test flows...")
    t1 = time.perf_counter()
    Z, recon_errors, src_test, dst_test = encode_test_data(
        joint_enc, cfg, df_test, device, checkpoint)
    print(f"  Embeddings: {Z.shape}")
    print(f"  Embedding time to generate: {time.perf_counter() - t1:.4f}s")

    # GMM clusters
    t2 = time.perf_counter()
    clusters = gmm.model.predict(Z)
    gmm_scores = gmm.model.score_samples(Z)
    print(f"  Clustering time: {time.perf_counter() - t2:.4f}s")


    # Novelty detection
    novelty_threshold = np.percentile(gmm_scores, 5)
    novelty_mask = gmm_scores < novelty_threshold
    n_novel = novelty_mask.sum()
    print(f"  Clusters: {len(np.unique(clusters))} unique")
    print(f"  Novelty flows: {n_novel} ({n_novel/len(Z)*100:.1f}%)")

    gt_mapper = GroundTruthMapper(DATASET)
    true_tactics = gt_mapper.map_labels(y_test) if y_test is not None else None
    tactic_list = gt_mapper.get_tactic_list()
    all_metrics = {}

    if y_test is None:
        print("  No ground truth — skipping evaluation")
        return

    # ══════════════════════════════════════════════════════
    # SECTION 1: CLUSTERING QUALITY (test)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SECTION 1: CLUSTERING QUALITY (test data)")
    print(f"{'='*70}")

    sample_size = min(5000, len(Z))
    sil_attack = silhouette_score(Z, y_test, sample_size=sample_size)
    sil_tactic = silhouette_score(Z, true_tactics, sample_size=sample_size)
    sil_cluster = silhouette_score(Z, clusters, sample_size=sample_size)
    nmi_attack = normalized_mutual_info_score(y_test, clusters)
    nmi_tactic = normalized_mutual_info_score(true_tactics, clusters)

    purities, sizes = [], []
    for k in np.unique(clusters):
        mask = clusters == k
        if mask.sum() < 2: continue
        maj = Counter(y_test[mask]).most_common(1)[0]
        purities.append(maj[1] / mask.sum())
        sizes.append(mask.sum())
    mean_purity = np.mean(purities)
    weighted_purity = np.average(purities, weights=sizes)

    print(f"  Mean purity:      {mean_purity:.4f}")
    print(f"  Weighted purity:  {weighted_purity:.4f}")
    print(f"  NMI (attack):     {nmi_attack:.4f}")
    print(f"  NMI (tactic):     {nmi_tactic:.4f}")
    print(f"  Sil (attack):     {sil_attack:.4f}")
    print(f"  Sil (tactic):     {sil_tactic:.4f}")
    print(f"  Sil (cluster):    {sil_cluster:.4f}")

    all_metrics.update({
        'mean_purity': mean_purity, 'weighted_purity': weighted_purity,
        'nmi_attack': nmi_attack, 'nmi_tactic': nmi_tactic,
        'sil_attack': sil_attack, 'sil_tactic': sil_tactic,
        'sil_cluster': sil_cluster,
    })

    # ══════════════════════════════════════════════════════
    # Build scoring sources (from KB + checkpoint)
    # ══════════════════════════════════════════════════════

    # Rule-based: from KB
    cluster_scores_rules = {}
    for k in np.unique(clusters):
        entry = kb.lookup(int(k))
        if entry:
            cluster_scores_rules[int(k)] = {
                'behaviour_scores': entry['behaviour_scores'],
                'tactic_scores': entry['tactic_scores'],
                'primary_behaviour': entry['primary_behaviour'],
                'primary_tactic': entry['primary_tactic'],
            }
        else:
            cluster_scores_rules[int(k)] = {
                'behaviour_scores': {'unknown': 1.0},
                'tactic_scores': {'Unknown': 1.0},
                'primary_behaviour': 'unknown',
                'primary_tactic': 'Unknown',
            }

    # LLM-based: from checkpoint
    cluster_scores_llm = {}
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
    has_llm_scores = len(cluster_scores_llm) > 0
    print(f"\n  Rule-based clusters: {len(cluster_scores_rules)}")
    print(f"  LLM-based clusters: {len(cluster_scores_llm)}")

    # ══════════════════════════════════════════════════════
    # SECTION 2: PER-FLOW TACTIC PREDICTION
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SECTION 2: PER-FLOW TACTIC PREDICTION")
    print(f"{'='*70}")

    # Rule-based per-flow
    print(f"\n  === RULE-BASED ===")
    flow_tactics_rules = np.array([
        cluster_scores_rules.get(int(clusters[i]), {}).get(
            'primary_tactic', 'Unknown')
        for i in range(len(clusters))])
    flow_acc_rules = (flow_tactics_rules == true_tactics).mean()
    print(f"  Accuracy: {flow_acc_rules:.4f}")
    report_rules = classification_report(
        true_tactics, flow_tactics_rules,
        labels=tactic_list, output_dict=True, zero_division=0, digits=4)
    print(classification_report(
        true_tactics, flow_tactics_rules,
        labels=tactic_list, zero_division=0, digits=4))

    all_metrics['perflow_rules_accuracy'] = flow_acc_rules
    all_metrics['perflow_rules_macro_f1'] = report_rules.get(
        'macro avg', {}).get('f1-score', 0)
    all_metrics['perflow_rules_weighted_f1'] = report_rules.get(
        'weighted avg', {}).get('f1-score', 0)
    for t in tactic_list:
        if t in report_rules:
            all_metrics[f'perflow_rules_{t}_f1'] = report_rules[t]['f1-score']
            all_metrics[f'perflow_rules_{t}_precision'] = report_rules[t]['precision']
            all_metrics[f'perflow_rules_{t}_recall'] = report_rules[t]['recall']

    # LLM-based per-flow
    if has_llm_scores:
        print(f"\n  === LLM-BASED ===")
        flow_tactics_llm = np.array([
            cluster_scores_llm.get(int(clusters[i]), {}).get(
                'primary_tactic', 'Unknown')
            for i in range(len(clusters))])
        flow_acc_llm = (flow_tactics_llm == true_tactics).mean()
        print(f"  Accuracy: {flow_acc_llm:.4f}")
        report_llm = classification_report(
            true_tactics, flow_tactics_llm,
            labels=tactic_list, output_dict=True, zero_division=0, digits=4)
        print(classification_report(
            true_tactics, flow_tactics_llm,
            labels=tactic_list, zero_division=0, digits=4))

        all_metrics['perflow_llm_accuracy'] = flow_acc_llm
        all_metrics['perflow_llm_macro_f1'] = report_llm.get(
            'macro avg', {}).get('f1-score', 0)
        for t in tactic_list:
            if t in report_llm:
                all_metrics[f'perflow_llm_{t}_f1'] = report_llm[t]['f1-score']

        # Side-by-side
        print(f"\n  {'Tactic':>20s} {'Rules F1':>10s} {'LLM F1':>10s}")
        print(f"  {'-'*42}")
        for t in tactic_list:
            r = report_rules.get(t, {}).get('f1-score', 0)
            l = report_llm.get(t, {}).get('f1-score', 0)
            better = '←' if r > l else '→' if l > r else '='
            print(f"  {t:>20s} {r:>10.4f} {l:>10.4f} {better}")
        print(f"  {'Accuracy':>20s} {flow_acc_rules:>10.4f} {flow_acc_llm:>10.4f}")

    # Confusion matrix
    cm = confusion_matrix(true_tactics, flow_tactics_rules, labels=tactic_list)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(tactic_list)))
    ax.set_yticks(range(len(tactic_list)))
    ax.set_xticklabels(tactic_list, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tactic_list, fontsize=8)
    for i in range(len(tactic_list)):
        for j in range(len(tactic_list)):
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})',
                    ha='center', va='center', fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Per-Flow Tactic Confusion (Inference, Rule-based)')
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, 'inference_confusion.png'), dpi=150)
    plt.close(fig)

    # ══════════════════════════════════════════════════════
    # SECTION 3: WINDOW-LEVEL SOFT TACTIC PREDICTION
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SECTION 3: WINDOW-LEVEL SOFT TACTIC PREDICTION")
    print(f"{'='*70}")

    # Rule-based windows
    print(f"\n  === RULE-BASED SCORING ===")
    for ws in [20, 50]:
        print(f"\n  --- Rule-based Window size = {ws} ---")
        aggregator = SoftWindowAggregator(
            window_size=ws, tactic_threshold=0.10)
        windows = aggregator.create_windows(
            src_test, clusters, cluster_scores_rules, y_true=y_test)
        soft_eval = SoftTacticEvaluator()
        metrics, df_res = soft_eval.evaluate_windows(
            windows, gt_mapper,
            save_dir=os.path.join(EVAL_DIR, f'rules_ws_{ws}'))
        if ws == 50:
            for k, v in metrics.items():
                all_metrics[f'window_rules_{k}'] = v

    # LLM-based windows
    if has_llm_scores:
        print(f"\n  === LLM-BASED SCORING ===")
        for ws in [20, 50]:
            print(f"\n  --- LLM Window size = {ws} ---")
            aggregator = SoftWindowAggregator(
                window_size=ws, tactic_threshold=0.10)
            windows_llm = aggregator.create_windows(
                src_test, clusters, cluster_scores_llm, y_true=y_test)
            soft_eval = SoftTacticEvaluator()
            metrics_llm, df_llm = soft_eval.evaluate_windows(
                windows_llm, gt_mapper,
                save_dir=os.path.join(EVAL_DIR, f'llm_ws_{ws}'))
            if ws == 50:
                for k, v in metrics_llm.items():
                    all_metrics[f'window_llm_{k}'] = v

    # ══════════════════════════════════════════════════════
    # SECTION 4: BAYESIAN FUSION (behaviour-level)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SECTION 4: BAYESIAN FUSION (behaviour-level)")
    print(f"{'='*70}")

    if has_llm_scores:
        w1, w2 = 0.6, 0.4

        cluster_scores_fused = {}
        for k in np.unique(clusters):
            k = int(k)
            rules = cluster_scores_rules.get(k, {}).get('tactic_scores', {})
            llm = cluster_scores_llm.get(k, {}).get('tactic_scores', {})
            if not llm:
                cluster_scores_fused[k] = cluster_scores_rules.get(k, {})
                continue

            all_tactics = set(list(rules.keys()) + list(llm.keys()))
            fused_tactics = {}
            for t in all_tactics:
                fused_tactics[t] = w1 * rules.get(t, 0) + w2 * llm.get(t, 0)
            total = sum(fused_tactics.values())
            if total > 0:
                fused_tactics = {t: s / total for t, s in fused_tactics.items()}

            rules_bhv = cluster_scores_rules.get(k, {}).get(
                'behaviour_scores', {})
            llm_bhv = cluster_scores_llm.get(k, {}).get(
                'behaviour_scores', {})
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

        # Per-flow fused
        flow_fused = np.array([
            cluster_scores_fused.get(int(clusters[i]), {}).get(
                'primary_tactic', 'Unknown')
            for i in range(len(clusters))])
        fused_acc = (flow_fused == true_tactics).mean()
        print(f"\n  Per-flow accuracy (rules):  {flow_acc_rules:.4f}")
        print(f"  Per-flow accuracy (fused):  {fused_acc:.4f}")
        print(f"  Improvement: {(fused_acc - flow_acc_rules)*100:+.2f}%")
        print(classification_report(
            true_tactics, flow_fused,
            labels=tactic_list, zero_division=0, digits=4))

        all_metrics['perflow_fused_accuracy'] = fused_acc

        # Window-level fused
        for ws in [20, 50]:
            print(f"\n  --- Fused Window size = {ws} ---")
            aggregator = SoftWindowAggregator(
                window_size=ws, tactic_threshold=0.10)
            fused_windows = aggregator.create_windows(
                src_test, clusters, cluster_scores_fused, y_true=y_test)
            soft_eval = SoftTacticEvaluator()
            f_metrics, f_df = soft_eval.evaluate_windows(
                fused_windows, gt_mapper,
                save_dir=os.path.join(EVAL_DIR, f'fused_ws_{ws}'))
            if ws == 50:
                for k, v in f_metrics.items():
                    all_metrics[f'fusion_{k}'] = v
    else:
        print("  Skipped — LLM scores not available")

    # ══════════════════════════════════════════════════════
    # SECTION 5: ANOMALY DETECTION
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SECTION 5: ANOMALY DETECTION (AUC-ROC)")
    print(f"{'='*70}")

    binary_true = (true_tactics != 'None').astype(int)
    if len(np.unique(binary_true)) > 1:
        neg_gmm = -gmm_scores
        combined = (recon_errors / (recon_errors.max() + 1e-10) +
                    neg_gmm / (neg_gmm.max() + 1e-10)) / 2

        for name, scores in [('Recon Error', recon_errors),
                              ('GMM (-loglik)', neg_gmm),
                              ('Combined', combined)]:
            auc = roc_auc_score(binary_true, scores)
            ap = average_precision_score(binary_true, scores)
            print(f"  {name:>15s}: AUC-ROC={auc:.4f}, AUC-PR={ap:.4f}")
            key = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            all_metrics[f'auc_{key}'] = auc
            all_metrics[f'ap_{key}'] = ap

    # Novelty summary
    if n_novel > 0:
        print(f"\n  NOVELTY: {n_novel} flows flagged as potential zero-day")
        if y_test is not None:
            novel_attacks = Counter(y_test[novelty_mask])
            print(f"  Novel flow attacks: {dict(novel_attacks)}")

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY — INFERENCE METRICS")
    print(f"{'='*70}")

    print(f"\n  CLUSTERING (test):")
    print(f"    Mean Purity:      {all_metrics.get('mean_purity', 0):.4f}")
    print(f"    Weighted Purity:  {all_metrics.get('weighted_purity', 0):.4f}")
    print(f"    NMI (tactic):     {all_metrics.get('nmi_tactic', 0):.4f}")
    print(f"    Sil (tactic):     {all_metrics.get('sil_tactic', 0):.4f}")

    print(f"\n  PER-FLOW:")
    print(f"    Rules accuracy:   {all_metrics.get('perflow_rules_accuracy', 0):.4f}")
    if has_llm_scores:
        print(f"    LLM accuracy:     {all_metrics.get('perflow_llm_accuracy', 0):.4f}")
        print(f"    Fused accuracy:   {all_metrics.get('perflow_fused_accuracy', 0):.4f}")

    print(f"\n  WINDOW (W=50):")
    print(f"    Rules Top-1:      {all_metrics.get('window_rules_top1_accuracy', 0):.4f}")
    print(f"    Rules Cosine:     {all_metrics.get('window_rules_mean_cosine', 0):.4f}")
    if has_llm_scores:
        print(f"    LLM Top-1:        {all_metrics.get('window_llm_top1_accuracy', 0):.4f}")
        print(f"    Fused Top-1:      {all_metrics.get('fusion_top1_accuracy', 0):.4f}")
        print(f"    Fused Cosine:     {all_metrics.get('fusion_mean_cosine', 0):.4f}")

    # Save all metrics
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(os.path.join(EVAL_DIR, 'inference_metrics.csv'), index=False)
    print(f"\n  All metrics saved: {EVAL_DIR}/")


if __name__ == '__main__':
    t0 = time.perf_counter()
    run_inference()
    time_taken= time.perf_counter() - t0
    print("Time taken: ", time_taken)