"""
Configuration for the Cyber Threat Detection System.
"""
import os

# ── Data ──
DATA_PATH = os.getenv(
    "DATA_PATH",
    "/home/nimesha/Documents/TW3_POC/datasets/NF-BoT-IoT-v2-sample.csv"
)
SAMPLE_SIZE = None  # None = full dataset

# ── Encoder ──
FEATURE_OUTPUT_DIM = 32
CONTEXT_OUTPUT_DIM = 32  # GraphSAGE per-node output (flow gets 2x: src||dst)
BEHAVIOUR_OUTPUT_DIM = 32
Z_DIM = 64  # VAE latent dimension
SEQ_LENGTH = 10  # Temporal sequence length
NODE_FEATURE_DIM = 10  # Aggregated IP node features

# ── VAE Training ──
VAE_EPOCHS = 30
VAE_BATCH_SIZE = 512
VAE_LR = 1e-3

# ── Behaviour-Aware AE ──
LAMBDA_BHV = 1.0  # Weight for behaviour prediction loss
N_BEHAVIOURS = 5  # scan, flood, exfil, beacon, lateral

# ── DP-GMM ──
GMM_MAX_COMPONENTS = 20
GMM_COVARIANCE_TYPE = 'diag'
GMM_WEIGHT_THRESHOLD = 0.01
GMM_NOVELTY_THRESHOLD = None  # Auto-calibrated from training data (1st percentile)
# INFO:src.streaming.streaming_gmm:Novelty threshold calibrated from training data: 122.71 (1.0th percentile)
# INFO:src.streaming.streaming_gmm:Training log-likelihood range: [-193.96, 276.90], median=270.22

# ── Streaming ──
WINDOW_SIZE = 1000
STREAMING_LR = 0.01  # Online EM learning rate

# ── Fusion ──
DEFAULT_W1_GMM = 0.5  # Default GMM weight in Bayesian fusion
DEFAULT_W2_LLM = 0.5  # Default LLM weight

# ── LLM ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4"
LLM_CONFIDENCE_THRESHOLD = 0.5  # Below this -> call LLM during inference

# ── Checkpoints ──
CHECKPOINT_DIR = "checkpoints"
KB_PATH = os.path.join(CHECKPOINT_DIR, "knowledge_base.json")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_checkpoint.pt")
