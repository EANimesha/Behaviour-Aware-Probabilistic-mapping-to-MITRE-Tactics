import torch
from torch import nn


class FeatureEncoder(nn.Module):
    """Track 1: MLP-based per-flow feature encoder.

    Maps raw flow features (d-dimensional) to a 32-d embedding.
    """

    def __init__(self, feat_in, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_in, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)