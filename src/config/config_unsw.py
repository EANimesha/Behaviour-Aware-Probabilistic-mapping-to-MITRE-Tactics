"""
Configuration for NF-UNSW-NB15-v2 Dataset
==========================================

Same 43 NetFlow features as NF-BoT-IoT-v2 (Sarhan et al., 2021).
Differences from BoT-IoT:
  - 9 attack types (vs 4): Fuzzers, Analysis, Backdoors, DoS, Exploits,
    Generic, Reconnaissance, Shellcode, Worms
  - Class balance: ~87% benign, ~13% attack (vs BoT-IoT's 99.99% attack)
  - Total flows: ~2.5M (vs BoT-IoT's ~3.6M)

MITRE ATT&CK tactic mapping for UNSW-NB15 attack types:
  Reconnaissance  -> TA0043 Reconnaissance
  Fuzzers         -> TA0001 Initial Access (fuzzing for vulns)
  Exploits        -> TA0002 Execution
  Generic         -> TA0040 Impact (generic attacks)
  DoS             -> TA0040 Impact
  Analysis        -> TA0007 Discovery
  Backdoors       -> TA0003 Persistence
  Shellcode       -> TA0002 Execution
  Worms           -> TA0008 Lateral Movement
"""

import os

# ── Data ──
DATASET_NAME = "NF-UNSW-NB15-v2"
DATA_PATH = os.getenv(
    "DATA_PATH",
    "/home/nimesha/Downloads/PythonProject1_updated/PythonProject1/data/NF-UNSW-NB15-v2.csv"
)
SAMPLE_SIZE = None

# Attack types in this dataset
ATTACK_TYPES = [
    'Benign', 'Reconnaissance', 'Fuzzers', 'Exploits', 'Generic',
    'DoS', 'Analysis', 'Backdoors', 'Shellcode', 'Worms'
]

# Suggested MITRE ATT&CK mapping (used by mock labeler)
ATTACK_TO_MITRE = {
    'Benign': ('Benign', 'N/A'),
    'Reconnaissance': ('Reconnaissance', 'TA0043'),
    'Fuzzers': ('Initial Access', 'TA0001'),
    'Exploits': ('Execution', 'TA0002'),
    'Generic': ('Impact', 'TA0040'),
    'DoS': ('Impact', 'TA0040'),
    'Analysis': ('Discovery', 'TA0007'),
    'Backdoors': ('Persistence', 'TA0003'),
    'Shellcode': ('Execution', 'TA0002'),
    'Worms': ('Lateral Movement', 'TA0008'),
}

# ── Data Balancing ──
# UNSW-NB15 is 96% benign — needs balancing for GMM to find attack clusters
BENIGN_RATIO = 1.0  # Target benign:attack ratio
MIN_SAMPLES_PER_CLASS = 200  # Boost rare classes (Worms=164) above this
MAX_TOTAL_SAMPLES = 120000  # Hard cap for memory/GPU

# FastChronologicalBalancer params
MAX_BENIGN_SAMPLES = 30_000
MAX_ATTACK_PER_CLASS = 12_000
MIN_ATTACK_PER_CLASS = 1_000

# ── Encoder ──
FEATURE_OUTPUT_DIM = 32
CONTEXT_OUTPUT_DIM = 32
BEHAVIOUR_OUTPUT_DIM = 32
Z_DIM = 64
SEQ_LENGTH = 10
NODE_FEATURE_DIM = 10

# ── VAE Training ──
# UNSW-NB15 is larger (~2.5M flows) so we may want more epochs
# but the benign-heavy imbalance means clusters form differently
VAE_EPOCHS = 30
VAE_BATCH_SIZE = 512
VAE_LR = 1e-3
VAE_BETA = 1.0
VAE_BETA_WARMUP = 10

# ── DP-GMM ──
# More attack types -> may need more components
GMM_MAX_COMPONENTS = 25
GMM_COVARIANCE_TYPE = 'diag'
GMM_WEIGHT_THRESHOLD = 0.01
GMM_NOVELTY_THRESHOLD = None  # Auto-calibrated from training data (1st percentile)

# ── Streaming ──
WINDOW_SIZE = 1000
STREAMING_LR = 0.01

# ── Fusion ──
DEFAULT_W1_GMM = 0.6
DEFAULT_W2_LLM = 0.4

# ── LLM ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4"
LLM_CONFIDENCE_THRESHOLD = 0.9

# ── Checkpoints ──
CHECKPOINT_DIR = "checkpoints_unsw"
KB_PATH = os.path.join(CHECKPOINT_DIR, "knowledge_base.json")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_checkpoint.pt")
