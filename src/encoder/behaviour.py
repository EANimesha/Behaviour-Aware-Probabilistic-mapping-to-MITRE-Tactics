import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """Standard positional encoding for Transformer."""

    def __init__(self, d_model, max_seq_length=20):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBehaviourEncoder(nn.Module):
    """Track 3: Transformer-based temporal behaviour encoder with attention."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=32,
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()

        # Project input to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim, max_seq_length=20)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)

        print(
            f"TransformerBehaviourEncoder: {input_dim} -> Transformer({hidden_dim}, {num_layers} layers) -> {output_dim}")

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_length, input_dim) temporal sequences
            mask: (batch_size, seq_length) binary mask for padding
        Returns:
            (batch_size, output_dim) embeddings
        """
        # Project to hidden dim
        x = self.input_proj(x)  # (batch_size, seq_length, hidden_dim)

        # Add positional encoding
        x = self.pos_enc(x)

        # Create attention mask for padding
        if mask is not None:
            # mask is (batch_size, seq_length), 1 = valid, 0 = padding
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2).bool()  # (batch, 1, 1, seq_len)
        else:
            attn_mask = None

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=mask == 0 if mask is not None else None)

        # Mean pooling (masked)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_length, 1)
            x_masked = x * mask_expanded
            embeddings = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-10)
        else:
            embeddings = x.mean(dim=1)

        # Output projection
        embeddings = self.fc(embeddings)
        return embeddings

