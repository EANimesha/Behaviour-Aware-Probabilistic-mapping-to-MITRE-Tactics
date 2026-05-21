"""
Joint Encoder with Variational AutoEncoder (VAE)
=================================================

Fuses three encoder tracks (Feature, Context, Behaviour) via concatenation,
then learns a smooth latent space using a VAE.

Training signal: reconstruction of the fused 3-track input.
Reparameterization: z = mu + sigma * epsilon

The VAE provides:
  - Regularised latent space ideal for DP-GMM clustering
  - Meaningful distances in latent space (similar flows cluster together)
  - Uncertainty quantification via the learned variance
"""

import torch
from torch import nn
import torch.nn.functional as F


# class VAEEncoder(nn.Module):
#     """Encoder half of the VAE: fused input -> mu, log_var."""
#
#     def __init__(self, input_dim, hidden_dim=128, z_dim=64):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
#
#         # Two heads: mean and log-variance
#         self.fc_mu = nn.Linear(hidden_dim // 2, z_dim)
#         self.fc_log_var = nn.Linear(hidden_dim // 2, z_dim)
#
#     def forward(self, x):
#         h = F.relu(self.bn1(self.fc1(x)))
#         h = F.relu(self.bn2(self.fc2(h)))
#         mu = self.fc_mu(h)
#         log_var = self.fc_log_var(h)
#         return mu, log_var
#
#
# class VAEDecoder(nn.Module):
#     """Decoder half of the VAE: z -> reconstructed fused input."""
#
#     def __init__(self, z_dim, hidden_dim=128, output_dim=96):
#         super().__init__()
#         self.fc1 = nn.Linear(z_dim, hidden_dim // 2)
#         self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
#         self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.fc_out = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, z):
#         h = F.relu(self.bn1(self.fc1(z)))
#         h = F.relu(self.bn2(self.fc2(h)))
#         return self.fc_out(h)
#
#
# class JointEncoder(nn.Module):
#     """Fuses three tracks via concatenation + VAE for latent space learning.
#
#     Architecture (from Training-new.jpg, Stage 3):
#         h_f (32d) + h_c (64d) + h_b (32d) -> h_fusion (128d)
#         h_fusion -> VAE Encoder -> mu, log_var
#         z = mu + sigma * epsilon  (reparameterization)
#         z -> VAE Decoder -> h_fusion_reconstructed
#
#     Loss: L_recon (MSE) + beta * L_KL (KL divergence)
#     """
#
#     def __init__(self, feature_encoder, context_encoder, behaviour_encoder,
#                  feature_output_dim=32, context_output_dim=32,
#                  behaviour_output_dim=32, z_dim=64, beta=1.0):
#         super().__init__()
#
#         self.feature_encoder = feature_encoder
#         self.context_encoder = context_encoder
#         self.behaviour_encoder = behaviour_encoder
#         self.beta = beta  # KL weight (beta-VAE)
#
#         # Fusion input: concat of all three tracks
#         # Context gives 2 * context_output_dim because we concat src + dst
#         self.fusion_input_dim = (feature_output_dim +
#                                  2 * context_output_dim +
#                                  behaviour_output_dim)
#
#         # VAE components
#         self.vae_encoder = VAEEncoder(
#             input_dim=self.fusion_input_dim,
#             hidden_dim=self.fusion_input_dim * 2,
#             z_dim=z_dim
#         )
#         self.vae_decoder = VAEDecoder(
#             z_dim=z_dim,
#             hidden_dim=self.fusion_input_dim * 2,
#             output_dim=self.fusion_input_dim
#         )
#
#         print(f"JointEncoder (VAE): {self.fusion_input_dim} "
#               f"-> VAE(z={z_dim}) -> reconstruct {self.fusion_input_dim}")
#
#     @staticmethod
#     def reparameterize(mu, log_var):
#         """Reparameterization trick: z = mu + sigma * epsilon."""
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + std * eps
#
#     def encode(self, graph, flow_features, node_features, context_embeddings,
#                temporal_sequences, temporal_mask=None):
#         """Run the three-track encoder + VAE encoder.
#
#         Returns:
#             z: (batch, z_dim) latent embeddings
#             mu: (batch, z_dim) mean
#             log_var: (batch, z_dim) log variance
#             h_fusion: (batch, fusion_input_dim) fused input (reconstruction target)
#         """
#         device = next(self.parameters()).device
#
#         # Move inputs to device
#         flow_features = flow_features.to(device)
#         context_embeddings = context_embeddings.to(device)
#         temporal_sequences = temporal_sequences.to(device)
#         if temporal_mask is not None:
#             temporal_mask = temporal_mask.to(device)
#
#         # Track 1: Feature encoding
#         h_f = self.feature_encoder(flow_features)           # (batch, 32)
#
#         # Track 2: Context (already computed as src||dst embeddings)
#         h_c = context_embeddings                             # (batch, 64)
#
#         # Track 3: Behaviour encoding
#         h_b = self.behaviour_encoder(temporal_sequences, temporal_mask)  # (batch, 32)
#
#         # Fuse embeddings: h_fusion = [h_f || h_c || h_b]
#         h_fusion = torch.cat([h_f, h_c, h_b], dim=1)       # (batch, 128)
#
#         # VAE encode -> mu, log_var
#         mu, log_var = self.vae_encoder(h_fusion)
#
#         # Reparameterize
#         z = self.reparameterize(mu, log_var)                 # (batch, z_dim)
#
#         return z, mu, log_var, h_fusion
#
#     def decode(self, z):
#         """Decode latent z back to fused representation."""
#         return self.vae_decoder(z)
#
#     def forward(self, graph, flow_features, node_features, context_embeddings,
#                 temporal_sequences, temporal_mask=None):
#         """Full forward pass: encode + decode.
#
#         Returns:
#             z: (batch, z_dim) latent embeddings
#             mu: (batch, z_dim) mean
#             log_var: (batch, z_dim) log variance
#             h_fusion: (batch, fusion_dim) original fused input
#             h_recon: (batch, fusion_dim) reconstructed fused input
#         """
#         z, mu, log_var, h_fusion = self.encode(
#             graph, flow_features, node_features,
#             context_embeddings, temporal_sequences, temporal_mask
#         )
#         h_recon = self.decode(z)
#         return z, mu, log_var, h_fusion, h_recon
#
#     def compute_loss(self, h_fusion, h_recon, mu, log_var):
#         """Compute VAE loss = reconstruction + beta * KL divergence.
#
#         Args:
#             h_fusion: (batch, fusion_dim) original fused input
#             h_recon: (batch, fusion_dim) reconstructed fused input
#             mu: (batch, z_dim) mean
#             log_var: (batch, z_dim) log variance
#
#         Returns:
#             total_loss, recon_loss, kl_loss (all scalars)
#         """
#         # Reconstruction loss (MSE)
#         recon_loss = F.mse_loss(h_recon, h_fusion, reduction='mean')
#
#         # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
#         kl_loss = -0.5 * torch.mean(
#             torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
#         )
#
#         total_loss = recon_loss + self.beta * kl_loss
#         return total_loss, recon_loss, kl_loss
#
#     def get_embeddings(self, graph, flow_features, node_features,
#                        context_embeddings, temporal_sequences,
#                        temporal_mask=None):
#         """Inference mode: return just mu (no sampling noise)."""
#         self.eval()
#         with torch.no_grad():
#             _, mu, _, _ = self.encode(
#                 graph, flow_features, node_features,
#                 context_embeddings, temporal_sequences, temporal_mask
#             )
#         # Use mu directly for deterministic inference
#         return mu







import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder: fused input -> z (single deterministic output)."""

    def __init__(self, input_dim, hidden_dim=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, z_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Decoder: z -> reconstructed fused input."""

    def __init__(self, z_dim, hidden_dim=128, output_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)
#
# class JointEncoder(nn.Module):
#     """Fuses three tracks via concatenation + AE for latent space learning.
#
#     Architecture:
#         h_f (32d) + h_c (64d) + h_b (32d) -> h_fusion (128d)
#         h_fusion -> Encoder -> z (64d)
#         z -> Decoder -> h_fusion_reconstructed (128d)
#
#     Loss: MSE(h_fusion, h_recon) — reconstruction only, no KL.
#     """
#
#     def __init__(self, feature_encoder, context_encoder, behaviour_encoder,
#                  feature_output_dim=32, context_output_dim=32,
#                  behaviour_output_dim=32, z_dim=64, **kwargs):
#         super().__init__()
#
#         self.feature_encoder = feature_encoder
#         self.context_encoder = context_encoder
#         self.behaviour_encoder = behaviour_encoder
#
#         # Fusion input: concat of all three tracks
#         # Context gives 2 * context_output_dim because we concat src + dst
#         self.fusion_input_dim = (feature_output_dim +
#                                  2 * context_output_dim +
#                                  behaviour_output_dim)
#
#         # AE components
#         self.encoder = Encoder(
#             input_dim=self.fusion_input_dim,
#             hidden_dim=self.fusion_input_dim * 2,
#             z_dim=z_dim
#         )
#         self.decoder = Decoder(
#             z_dim=z_dim,
#             hidden_dim=self.fusion_input_dim * 2,
#             output_dim=self.fusion_input_dim
#         )
#
#         print(f"JointEncoder (AE): {self.fusion_input_dim} "
#               f"-> Encoder(z={z_dim}) -> reconstruct {self.fusion_input_dim}")
#
#     def encode(self, graph, flow_features, node_features, context_embeddings,
#                temporal_sequences, temporal_mask=None):
#         """Run the three-track encoder + AE encoder.
#
#         Returns:
#             z: (batch, z_dim) latent embeddings
#             h_fusion: (batch, fusion_input_dim) fused input (reconstruction target)
#         """
#         device = next(self.parameters()).device
#
#         flow_features = flow_features.to(device)
#         context_embeddings = context_embeddings.to(device)
#         temporal_sequences = temporal_sequences.to(device)
#         if temporal_mask is not None:
#             temporal_mask = temporal_mask.to(device)
#
#         # Track 1: Feature encoding
#         h_f = self.feature_encoder(flow_features)  # (batch, 32)
#
#         # Track 2: Context (already computed as src||dst embeddings)
#         h_c = context_embeddings  # (batch, 64)
#
#         # Track 3: Behaviour encoding
#         h_b = self.behaviour_encoder(temporal_sequences, temporal_mask)  # (batch, 32)
#
#         # Fuse: h_fusion = [h_f || h_c || h_b]
#         h_fusion = torch.cat([h_f, h_c, h_b], dim=1)  # (batch, 128)
#
#         # Encode to latent
#         z = self.encoder(h_fusion)  # (batch, z_dim)
#
#         return z, h_fusion
#
#     def decode(self, z):
#         """Decode latent z back to fused representation."""
#         return self.decoder(z)
#
#     def forward(self, graph, flow_features, node_features, context_embeddings,
#                 temporal_sequences, temporal_mask=None):
#         """Full forward pass: encode + decode.
#
#         Returns:
#             z: (batch, z_dim) latent embeddings
#             h_fusion: (batch, fusion_dim) original fused input
#             h_recon: (batch, fusion_dim) reconstructed fused input
#         """
#         z, h_fusion = self.encode(
#             graph, flow_features, node_features,
#             context_embeddings, temporal_sequences, temporal_mask
#         )
#         h_recon = self.decode(z)
#         return z, h_fusion, h_recon
#
#     def compute_loss(self, h_fusion, h_recon):
#         """Compute AE loss = reconstruction MSE only.
#
#         Args:
#             h_fusion: (batch, fusion_dim) original fused input
#             h_recon: (batch, fusion_dim) reconstructed fused input
#
#         Returns:
#             loss: scalar
#         """
#         return F.mse_loss(h_recon, h_fusion, reduction='mean')
#
#     def get_embeddings(self, graph, flow_features, node_features,
#                        context_embeddings, temporal_sequences,
#                        temporal_mask=None):
#         """Inference mode: return z embeddings."""
#         self.eval()
#         with torch.no_grad():
#             z, _ = self.encode(
#                 graph, flow_features, node_features,
#                 context_embeddings, temporal_sequences, temporal_mask
#             )
#         return z









class BehaviourHead(nn.Module):
    """Auxiliary head: z -> predicted behaviour scores.

    Predicts 5 behaviour dimensions (scan, flood, exfil, beacon, lateral)
    from the latent embedding. This forces z to encode tactic-relevant
    structure rather than just reconstruction-friendly features.
    """

    def __init__(self, z_dim, n_behaviours=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_behaviours),
            nn.Sigmoid(),  # Output in [0, 1] to match behaviour scores
        )

    def forward(self, z):
        return self.net(z)


class JointEncoder(nn.Module):
    """Fuses three tracks + behaviour-aware AE.

    Architecture:
        h_f (32d) + h_c (64d) + h_b (32d) -> h_fusion (128d)
        h_fusion -> Encoder -> z (64d)
        z -> Decoder -> h_recon (128d)         [reconstruction path]
        z -> BehaviourHead -> pred_scores (5d)  [behaviour path]

    Loss:
        L = L_recon + lambda_bhv * L_behaviour
        L_recon     = MSE(h_fusion, h_recon)
        L_behaviour = MSE(pred_scores, true_scores)
    """

    def __init__(self, feature_encoder, context_encoder, behaviour_encoder,
                 feature_output_dim=32, context_output_dim=32,
                 behaviour_output_dim=32, z_dim=64,
                 n_behaviours=5, lambda_bhv=1.0, **kwargs):
        super().__init__()

        self.feature_encoder = feature_encoder
        self.context_encoder = context_encoder
        self.behaviour_encoder = behaviour_encoder
        self.lambda_bhv = lambda_bhv

        # Fusion input dim
        self.fusion_input_dim = (feature_output_dim +
                                 2 * context_output_dim +
                                 behaviour_output_dim)

        # AE components
        self.encoder = Encoder(
            input_dim=self.fusion_input_dim,
            hidden_dim=self.fusion_input_dim * 2,
            z_dim=z_dim
        )
        self.decoder = Decoder(
            z_dim=z_dim,
            hidden_dim=self.fusion_input_dim * 2,
            output_dim=self.fusion_input_dim
        )

        # Behaviour prediction head
        self.behaviour_head = BehaviourHead(z_dim, n_behaviours)

        print(f"JointEncoder (Behaviour-Aware AE): "
              f"{self.fusion_input_dim} -> z({z_dim}) -> "
              f"recon({self.fusion_input_dim}) + behaviour({n_behaviours})")
        print(f"  lambda_bhv={lambda_bhv}")

    def encode(self, graph, flow_features, node_features, context_embeddings,
               temporal_sequences, temporal_mask=None):
        """Run three-track encoder + AE encoder.

        Returns:
            z, h_fusion
        """
        device = next(self.parameters()).device

        flow_features = flow_features.to(device)
        context_embeddings = context_embeddings.to(device)
        temporal_sequences = temporal_sequences.to(device)
        if temporal_mask is not None:
            temporal_mask = temporal_mask.to(device)

        h_f = self.feature_encoder(flow_features)
        h_c = context_embeddings
        h_b = self.behaviour_encoder(temporal_sequences, temporal_mask)

        h_fusion = torch.cat([h_f, h_c, h_b], dim=1)
        z = self.encoder(h_fusion)

        return z, h_fusion

    def decode(self, z):
        return self.decoder(z)

    def predict_behaviour(self, z):
        return self.behaviour_head(z)

    def forward(self, graph, flow_features, node_features, context_embeddings,
                temporal_sequences, temporal_mask=None):
        """Full forward pass.

        Returns:
            z, h_fusion, h_recon, pred_behaviour
        """
        z, h_fusion = self.encode(
            graph, flow_features, node_features,
            context_embeddings, temporal_sequences, temporal_mask
        )
        h_recon = self.decode(z)
        pred_behaviour = self.predict_behaviour(z)
        return z, h_fusion, h_recon, pred_behaviour

    def compute_loss(self, h_fusion, h_recon, pred_behaviour=None,
                     true_behaviour=None):
        """Compute combined loss.

        L = L_recon + lambda_bhv * L_behaviour

        Args:
            h_fusion: original fused input
            h_recon: reconstructed fused input
            pred_behaviour: predicted behaviour scores from head
            true_behaviour: ground truth behaviour scores from BehaviourScorer

        Returns:
            total_loss, recon_loss, behaviour_loss
        """
        recon_loss = F.mse_loss(h_recon, h_fusion, reduction='mean')

        if pred_behaviour is not None and true_behaviour is not None:
            bhv_loss = F.mse_loss(pred_behaviour, true_behaviour, reduction='mean')
            total_loss = recon_loss + self.lambda_bhv * bhv_loss
        else:
            bhv_loss = torch.tensor(0.0)
            total_loss = recon_loss

        return total_loss, recon_loss, bhv_loss

    def get_embeddings(self, graph, flow_features, node_features,
                       context_embeddings, temporal_sequences,
                       temporal_mask=None):
        """Inference mode: return z embeddings."""
        self.eval()
        with torch.no_grad():
            z, _ = self.encode(
                graph, flow_features, node_features,
                context_embeddings, temporal_sequences, temporal_mask
            )
        return z
