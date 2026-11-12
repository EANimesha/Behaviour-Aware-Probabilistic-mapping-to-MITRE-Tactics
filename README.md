# BGTD-NID: Behaviour-Guided Tactic Discovery for Network Intrusion Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Behaviour-Guided Tactic Discovery with Knowledge-Based Semantic Reasoning for Network Intrusion Detection**
>
> *Submitted to Knowledge-Based Systems (Elsevier)*

BGTD-NID is an unsupervised framework that discovers network behaviours from raw NetFlow traffic and maps them to [MITRE ATT&CK](https://attack.mitre.org/) tactics — without requiring attack-type or tactic labels during training. It provides traceable semantic explanations for every tactic prediction through a portable Knowledge Base.

---


## Key Features

- **Unsupervised tactic attribution** — no attack-type or tactic labels needed during training
- **Three-track encoder** — jointly captures flow statistics (MLP), IP communication topology (GraphSAGE), and temporal dynamics (Transformer)
- **Dual semantic labelling** — independent rule-based and LLM-based scorers fused via weighted probabilistic combination
- **Portable Knowledge Base** — stores fused behaviour/tactic distributions with traceable explanation chains
- **Leakage-safe evaluation (v3)** — per-class chronological train / gap / test split; fusion weight and thresholds selected on validation only
- **Streaming adaptation** — online EM updates, log-likelihood drift monitoring and a Dynamic KB that re-labels clusters per batch
- **Cross-dataset evaluation** — NF-BoT-IoT-v3 (primary) with label-free transfer to NF-UNSW-NB15-v3; the original v2 experiments (incl. NF-CICIDS-2018-v2) are kept in `src/v2_runs/`

## Results Summary

**NF-BoT-IoT-v3 (chronological split, test set, final model seed 44, fused w_r = 0.7 selected on validation):**

| Metric | Rules | LLM | Fused |
|--------|-------|-----|-------|
| Per-flow tactic accuracy (%) | 88.80 | 91.12 | 89.31 |
| Per-flow macro-F1 (%) | 66.09 | 68.17 | 66.45 |
| Window Top-1 accuracy, W=50 (%) | 91.72 | 90.45 | 91.72 |

Scope: Impact and Reconnaissance are attributed reliably; low-volume Exfiltration (Theft), whose flows resemble benign traffic at flow level, is not.

**Cross-dataset transfer (NF-UNSW-NB15-v3, label-free, w_r = 0.5 fixed a priori):** fused macro-F1 0.115 (all tactics), 0.306 (shared tactics), W50 Top-1 0.395.

**Streaming (NF-BoT-IoT-v3):** online adaptation improves macro-F1 over a static model in all three phases (0.685 vs 0.665, 0.629 vs 0.501, 0.430 vs 0.112). Drift detection: precision 1.00, recall 0.33. Likelihood-based novelty detection does not separate unseen attacks on this dataset (leave-one-attack-out AUC 0.28–0.59).

Results for the v2 datasets (random split, original submission) are reproducible from `src/v2_runs/`.

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 32GB RAM

### Setup

```bash
git clone https://github.com/<your-username>/bgtd-nid.git
cd bgtd-nid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.12.0
dgl>=1.0.0
scikit-learn>=1.1.0
numpy>=1.21.0
pandas>=1.4.0
matplotlib>=3.5.0
openai>=1.0.0        # For LLM-based scoring (Stage 5)
anthropic>=0.20.0    # For LLM-as-judge evaluation (optional)
```

### Data Setup

Download the standardised NetFlow datasets from [Sarhan et al. (2022)](https://staff.itee.uq.edu.au/marius/NIDS_datasets/):

```bash
mkdir -p data/
# Place dataset CSV files:
#   data/NF-BoT-IoT-v3.csv        (v3 runs, primary)
#   data/NF-UNSW-NB15-v3.csv      (v3 transfer target)
#   data/NF-BoT-IoT-v2.csv, NF-UNSW-NB15-v2.csv, NF-CSE-CIC-IDS2018-v2.csv  (v2 runs)
```

### API Keys (optional)

```bash
# For LLM-based behaviour scoring (Stage 5)
export OPENAI_API_KEY="sk-..."

# For multi-LLM explanation quality evaluation
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

Keys are read from environment variables only; never commit them.

## Project Structure

```
bgtd-nid/
├── main.py                     # Training pipeline (BoT-IoT)
├── main_unsw.py                # Training pipeline (UNSW-NB15)
├── main_cicids.py              # Training pipeline (CICIDS-2018)
├── validation.py               # Comprehensive validation & metrics
├── inference.py                # KB-based inference (no LLM calls)
├── baselines.py                # 6 baseline comparisons + plots
├── ablation_studies.py         # Clustering/window/fusion/covariance ablations
├── ablation_encoder.py         # Encoder track ablation (joint training)
├── streaming_pipeline.py           # Three-phase adaptation + drift + novelty
├── explainability_eval.py      # LLM-as-judge quality scoring
├── plot_convergence.py         # DP-GMM ELBO convergence plot
├── inject_kb_features.py       # Add features to existing KB (no retrain)
├── analyse_clusters.py         # Fisher discriminability analysis
├── evaluate_embeddings.py      # 9 embedding diagnostic plots
│
├── src/
│   ├── v2_runs/                # Original submission (v2 datasets, random split)
│   │   ├── main.py, main_unsw.py, main_cicids.py
│   │   └── ...                 # v2 validation / inference / ablation / streaming scripts
│   │
│   ├── v3_runs/                # Revision (v3 datasets, leakage-safe chronological split)
│   │   ├── training_v3.py              # Training (SEED, CHECKPOINT_DIR, TRANSFER_KB, HOLDOUT_ATTACK via env)
│   │   ├── inference_v3.py             # Test inference + timing (no LLM calls)
│   │   ├── sweep_fusion_window_v3.py   # Fusion weight (validation) + window-size sensitivity
│   │   ├── run_ablation_v3.py          # Encoder / clustering ablations over seeds
│   │   ├── baseline_experiments_v3.py  # IF, AE, K-Means oracle, RF, MLP, E-GraphSAGE
│   │   ├── streaming_v3.py             # Three-phase adaptation, drift, leave-one-attack-out novelty
│   │   ├── novelty_loo_summary.py      # Collects leave-one-attack-out results
│   │   ├── explainability_v3_extra.py  # Complementarity, hallucination, sensitivity, judges
│   │   ├── cluster_purity_table_v3.py  # Per-cluster size / purity / fused labels
│   │   ├── explain_chain_v3.py         # Explanation chains, heuristic-semantic matrix, flow trace
│   │   └── check_split_dates.py        # Train/test date ranges per class
│   │
│   ├── config/
│   │   ├── config.py           # BoT-IoT configuration
│   │   ├── config_unsw.py      # UNSW-NB15 configuration
│   │   ├── config_cicids.py    # CICIDS-2018 configuration
│   │   └── config_v3.py        # v3 datasets (paths, dims, CHECKPOINT_DIR override)
│   │
│   ├── data/
│   │   ├── loader.py           # Dataset loading utility
│   │   ├── preprocess.py       # Feature engineering (conditional clip_extremes)
│   │   ├── graph_builder.py    # IP communication graph + temporal sequences
│   │   └── balancer.py         # Chronological class balancing
│   │
│   ├── encoder/
│   │   ├── feature.py          # MLP feature encoder (Track 1)
│   │   ├── context.py          # GraphSAGE context encoder (Track 2)
│   │   ├── behaviour.py        # Transformer temporal encoder (Track 3)
│   │   ├── jointencoder.py     # Behaviour-aware AE with LayerNorm fusion
│   │   └── behaviour_scorer.py # Heuristic behaviour target computation
│   │
│   ├── clustering/
│   │   └── summarizer.py       # Raw feature passthrough + cluster summaries
│   │
│   ├── mapping/
│   │   ├── soft_cluster_scorer.py  # Dataset-aware scoring rules + tactic weights
│   │   └── behaviour_mapping.py    # Behaviour-to-tactic mapping
│   │
│   ├── llm/
│   │   └── reasoning.py        # Dataset-aware LLM prompts (GPT-4)
│   │
│   ├── knowledge_base/
│   │   ├── kb.py               # Knowledge Base (load/save/lookup)
│   │   └── dynamic_kb.py       # Dynamic KB for streaming adaptation
│   │
│   ├── fusion/
│   │   └── fusion.py           # Behaviour-level weighted fusion
│   │
│   ├── streaming/
│   │   └── streaming_gmm.py    # Online EM + drift detection
│   │
│   └── evaluation/
│       ├── ground_truth.py     # Attack-type → tactic mapping (3 datasets)
│       ├── soft_eval.py        # Window-level soft distributional metrics
│       └── metrics.py          # Per-flow tactic evaluator
│
├── checkpoints/                # Saved models and KB (created at runtime)
└── data/                       # Dataset CSV files (not tracked)
```

## Usage

### v3 (revision)

Run from the repository root; scripts read `V3_DATASET` and `CHECKPOINT_DIR` from the environment.

```bash
# 1. Train (final model: seed 44)
SEED=44 V3_DATASET=bot_iot CHECKPOINT_DIR=checkpoints_botv3 python src/v3_runs/training_v3.py

# 2. Inference on the held-out test split (no LLM calls)
FUSION_W_RULES=0.7 V3_DATASET=bot_iot CHECKPOINT_DIR=checkpoints_botv3 python src/v3_runs/inference_v3.py

# 3. Fusion-weight (validation) and window-size sensitivity
V3_DATASET=bot_iot CHECKPOINT_DIR=checkpoints_botv3 python src/v3_runs/sweep_fusion_window_v3.py

# 4. Ablations (3 seeds; LLM on, as in the final model)
V3_DATASET=bot_iot python src/v3_runs/run_ablation_v3.py                 # encoder tracks
V3_DATASET=bot_iot python src/v3_runs/run_ablation_v3.py --study cluster # K-Means / GMM / DP-GMM

# 5. Baselines
FUSION_W_RULES=0.7 V3_DATASET=bot_iot CHECKPOINT_DIR=checkpoints_botv3 python src/v3_runs/baseline_experiments_v3.py

# 6. Streaming (three-phase adaptation + drift); STREAM_SHUFFLE=0 for time order
V3_DATASET=bot_iot CHECKPOINT_DIR=checkpoints_botv3 python src/v3_runs/streaming_v3.py bot_iot
#    Leave-one-attack-out novelty (train one HOLDOUT_ATTACK model per class first)
NOVELTY_ONLY=1 HOLDOUT_CKPT_DIR=checkpoints_botv3_holdout_Theft CHECKPOINT_DIR=checkpoints_botv3 V3_DATASET=bot_iot python src/v3_runs/streaming_v3.py bot_iot
python src/v3_runs/novelty_loo_summary.py checkpoints_botv3/streaming_results

# 7. Explainability
FUSION_W_RULES=0.7 V3_DATASET=bot_iot python src/v3_runs/explainability_v3_extra.py 1 2 3 4 5
FUSION_W_RULES=0.7 V3_DATASET=bot_iot python src/v3_runs/cluster_purity_table_v3.py
EXAMPLE_CLUSTERS=14,1,3,19 TRACE_CLUSTER=14 FUSION_W_RULES=0.7 V3_DATASET=bot_iot python src/v3_runs/explain_chain_v3.py

# 8. Transfer to NF-UNSW-NB15-v3 (label-free refit, BoT-IoT knowledge frozen)
TRANSFER_KB=bot_iot V3_DATASET=unsw_nb15 CHECKPOINT_DIR=checkpoints_unswv3_transfer python src/v3_runs/training_v3.py
FUSION_W_RULES=0.5 TRANSFER_KB=bot_iot V3_DATASET=unsw_nb15 CHECKPOINT_DIR=checkpoints_unswv3_transfer python src/v3_runs/inference_v3.py
```

### v2 (original submission)

```bash
python src/v2_runs/main.py          # NF-BoT-IoT-v2
python src/v2_runs/main_unsw.py     # NF-UNSW-NB15-v2
python src/v2_runs/main_cicids.py   # NF-CICIDS-2018-v2
python validation.py bot_iot
python inference.py bot_iot
python ablation_studies.py bot_iot cluster   # also: window | fusion | covariance
python ablation_encoder.py bot_iot
python streaming_pipeline.py bot_iot
python explainability_eval.py bot_iot
python baselines.py bot_iot
```

## Three Operational Modes

```
Training:   Stages 1-7  →  Learns representations, discovers clusters,
                            builds KB with fused tactic distributions.

Inference:  Stage 4 + KB lookup + window aggregation
                        →  Per-flow and per-window tactic prediction.
                            No LLM calls, no rule recomputation.

Streaming:  Inference + online EM + drift detection + Dynamic KB
                        →  Batch-level adaptation under evolving traffic.
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Bayesian fusion at behaviour level** (not cluster level) | GMM posteriors are near-deterministic (>0.95); semantic uncertainty persists between rule and LLM interpretations |
| **Raw feature passthrough** in cluster summaries | Inverse-transforming log-scaled/standardised features produces artifacts (e.g., OUT_BYTES=0 for 728K-byte clusters) |
| **LayerNorm per track** before fusion | GraphSAGE dominates 60-80% of fusion norm without it, drowning out discriminative MLP features |
| **DP-GMM** over fixed-K alternatives | Automatic cluster count determination; matches GMM quality at equal K while eliminating manual K specification |
| **Window aggregation** (W=50) | Fixed a priori; aggregates per-flow predictions per source IP |
| **Fusion weight** w_r = 0.7 | Selected on the validation gap (v3); never tuned on test |

## Datasets

All datasets use the [standardised NetFlow representation](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) by Sarhan et al. (2022).

| Dataset | Flows | Attack Types | MITRE Tactics | Role |
|---------|-------|-------------|---------------|------|
| NF-BoT-IoT-v3 | sampled | 4 | 4 | Primary benchmark (revision) |
| NF-UNSW-NB15-v3 | sampled | 9 | 7 | Label-free transfer target (revision) |
| NF-BoT-IoT-v2 | ~3.6M (42K sampled) | 4 | 4 | Original submission |
| NF-UNSW-NB15-v2 | ~2.5M (65K sampled) | 9 | 7 | Original submission |
| NF-CICIDS-2018-v2 | ~16M (120K sampled) | 14 | 6 | Original submission |

## Configuration

Dataset-specific settings are in `src/config/` (`config_v3.py` for the v3 runs):

```python
# Key hyperparameters (src/config/config.py)
SAMPLE_SIZE = 42000
FEATURE_OUTPUT_DIM = 32        # d_f
CONTEXT_OUTPUT_DIM = 32        # per node (d_c = 64 after src||dst concat)
BEHAVIOUR_OUTPUT_DIM = 32      # d_b
Z_DIM = 64                    # d_z
SEQ_LENGTH = 10                # K_seq
GMM_MAX_COMPONENTS = 20        # K_max
WINDOW_SIZE = 50               # W
TACTIC_THRESHOLD = 0.10        # τ
LAMBDA_BHV = 1.0               # λ (behaviour loss weight)
```

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{bgtdnid2026,
  title={BGTD-NID: Behaviour-Guided Tactic Discovery with 
         Knowledge-Based Semantic Reasoning for Network 
         Intrusion Detection},
  journal={Knowledge-Based Systems},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Sarhan et al.](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) for the standardised NetFlow dataset representations
- [MITRE ATT&CK](https://attack.mitre.org/) for the adversary tactic framework
- [DGL](https://www.dgl.ai/) for the graph neural network library
