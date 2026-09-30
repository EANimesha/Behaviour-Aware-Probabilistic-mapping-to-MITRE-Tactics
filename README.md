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
- **Streaming adaptation** — online EM updates, drift detection via log-likelihood monitoring, Dynamic KB, and novelty detection for zero-day attacks
- **Cross-dataset evaluation** — tested on NF-BoT-IoT-v2, NF-UNSW-NB15-v2, and NF-CICIDS-2018-v2

## Results Summary

| Metric | NF-BoT-IoT-v2 | NF-UNSW-NB15-v2 | NF-CICIDS-2018-v2 |
|--------|---------------|-----------------|-------------------|
| Cluster purity (%) | 78.22 | 67.88 | 84.38 |
| Per-flow tactic accuracy (%) | 64.11 | 52.30 | 69.72 |
| Window Top-1 accuracy (%) | 73.86 | 55.62 | 84.74 |
| Window Multi-Tactic Coverage (%) | 97.73 | 86.98 | 92.74 |
| Window cosine similarity (%) | 87.61 | 81.77 | 79.05 |
| Hopkins statistic | 0.998 | 0.991 | 0.992 |

**Streaming capabilities (BoT-IoT):**
- Drift detection F1: 0.80 (1-batch latency)
- Novelty detection AUC-ROC: 0.823 (held-out Theft attack)
- Online EM log-likelihood improvement: +0.72

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
#   data/NF-BoT-IoT-v2.csv
#   data/NF-UNSW-NB15-v2.csv
#   data/NF-CSE-CIC-IDS2018-v2.csv
```

### API Keys (optional)

```bash
# For LLM-based behaviour scoring (Stage 5)
export OPENAI_API_KEY="sk-..."

# For multi-LLM explanation quality evaluation
export ANTHROPIC_API_KEY="sk-ant-..."
```

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
│   ├── config/
│   │   ├── config.py           # BoT-IoT configuration
│   │   ├── config_unsw.py      # UNSW-NB15 configuration
│   │   └── config_cicids.py    # CICIDS-2018 configuration
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

### 1. Training

```bash
# Train on NF-BoT-IoT-v2 (primary dataset)
python main.py

# Train on other datasets
python main_unsw.py
python main_cicids.py
```

Training outputs:
- `checkpoints/model_checkpoint.pt` — encoder + GMM + cluster assignments
- `checkpoints/knowledge_base.json` — fused behaviour/tactic KB

### 2. Validation

```bash
python validation.py bot_iot
```

Computes: cluster purity, NMI/ARI, silhouette scores, per-flow tactic accuracy, window-level metrics, and generates the cluster composition plot.

### 3. Inference

```bash
python inference.py bot_iot
```

Runs KB-based tactic prediction on held-out test data. No LLM calls — uses pre-computed KB entries.

### 4. Ablation Studies

```bash
# Clustering method comparison (K-Means vs GMM vs DP-GMM)
python ablation_studies.py bot_iot cluster

# Window size sensitivity (W = 5, 10, 20, 30, 50, 75, 100, 200)
python ablation_studies.py bot_iot window

# Fusion weight sensitivity (w_r = 0.0 to 1.0)
python ablation_studies.py bot_iot fusion

# Covariance type comparison
python ablation_studies.py bot_iot covariance

# Encoder track ablation (requires retraining per config)
python ablation_encoder.py bot_iot
python ablation_encoder.py cicids2018
```

### 5. Streaming Evaluation

```bash
python streaming_pipeline.py bot_iot
```

Runs three evaluations:
1. **Three-phase adaptation** — stable → drift injection → recovery (Static GMM vs Online GMM + Dynamic KB)
2. **Drift detection** — log-likelihood monitoring with simulated distribution shift
3. **Novelty detection** — hold-out attack type (Theft/Exfiltration) flagging via log-likelihood thresholding

### 6. Explainability Evaluation

```bash
# Requires OPENAI_API_KEY and optionally ANTHROPIC_API_KEY
python explainability_eval.py bot_iot
```

Evaluates explanation quality across three dimensions: consistency (rule vs LLM agreement), feature alignment (cited features match Fisher-derived signatures), and semantic quality (multi-LLM judge scoring).

### 7. Baseline Comparison

```bash
python baselines.py bot_iot
```

Compares against: Isolation Forest, Plain Autoencoder, K-Means (oracle), Random Forest, MLP Classifier, E-GraphSAGE.

## Three Operational Modes

```
Training:   Stages 1-7  →  Learns representations, discovers clusters,
                            builds KB with fused tactic distributions.

Inference:  Stage 4 + KB lookup + window aggregation
                        →  Per-flow and per-window tactic prediction.
                            No LLM calls, no rule recomputation.

Streaming:  Inference + online EM + drift detection + Dynamic KB
                        →  Continuous adaptation under evolving traffic.
                            Novelty flagging for zero-day attacks.
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Bayesian fusion at behaviour level** (not cluster level) | GMM posteriors are near-deterministic (>0.95); semantic uncertainty persists between rule and LLM interpretations |
| **Raw feature passthrough** in cluster summaries | Inverse-transforming log-scaled/standardised features produces artifacts (e.g., OUT_BYTES=0 for 728K-byte clusters) |
| **LayerNorm per track** before fusion | GraphSAGE dominates 60-80% of fusion norm without it, drowning out discriminative MLP features |
| **DP-GMM** over fixed-K alternatives | Automatic cluster count determination; matches GMM quality at equal K while eliminating manual K specification |
| **Window aggregation** (W=50) | Per-flow predictions lack temporal context; window-level Multi-Tactic Coverage reaches 97.73% vs 64% per-flow |

## Datasets

All datasets use the [standardised NetFlow representation](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) by Sarhan et al. (2022).

| Dataset | Flows | Attack Types | MITRE Tactics | Role |
|---------|-------|-------------|---------------|------|
| NF-BoT-IoT-v2 | ~3.6M (42K sampled) | 4 | 4 | Primary benchmark |
| NF-UNSW-NB15-v2 | ~2.5M (65K sampled) | 9 | 7 | Limitation case |
| NF-CICIDS-2018-v2 | ~16M (120K sampled) | 14 | 6 | Cross-dataset validation |

## Configuration

Dataset-specific settings are in `src/config/`:

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
