"""
Configuration — NF-CSE-CIC-IDS2018-v2 (Sampled)
==================================================
"""
import os

# Dataset
DATA_PATH = 'data/NF-CIC-IDS2018-v2-sampled.csv'
SAMPLE_SIZE = None  # Already sampled

# Encoder dimensions
FEATURE_OUTPUT_DIM = 32
CONTEXT_OUTPUT_DIM = 32
BEHAVIOUR_OUTPUT_DIM = 32
Z_DIM = 64
SEQ_LENGTH = 10
NODE_FEATURE_DIM = 10  # Graph builder produces 10D node features (degree metrics)

# Training
VAE_EPOCHS = 30
VAE_BATCH_SIZE = 256
VAE_LR = 1e-3

# DP-GMM
GMM_MAX_COMPONENTS = 30  # More attack types than BoT-IoT
GMM_COVARIANCE_TYPE = 'diag'

# Fusion weights
DEFAULT_W1_GMM = 0.6
DEFAULT_W2_LLM = 0.4

# LLM
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
LLM_MODEL = 'gpt-4'

# Paths
CHECKPOINT_DIR = 'checkpoints_cicids'
KB_PATH = os.path.join(CHECKPOINT_DIR, 'knowledge_base.json')
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'model_checkpoint.pt')

# FastChronologicalBalancer params (dataset already sampled, light balancing)
MAX_TOTAL_SAMPLES = 120_000
MAX_BENIGN_SAMPLES = 30_000
MAX_ATTACK_PER_CLASS = 15_000
MIN_ATTACK_PER_CLASS = 500
BENIGN_RATIO = 2.0
MIN_SAMPLES_PER_CLASS = 500