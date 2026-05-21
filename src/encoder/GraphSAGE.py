"""
Custom GraphSAGE Implementation for DGL
========================================

Implements GraphSAGE (Graph Sampling And AggregatinG Embeddings) from scratch
using DGL primitives, based on official DGL examples.

This works with any DGL version and avoids import issues.

Reference: https://github.com/dmlc/dgl/blob/master/examples/core/graphsage/node_classification.py
"""

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAGEConv(nn.Module):
    """
    Single-layer GraphSAGE convolution.

    Aggregates neighbor embeddings and combines with node's own embedding.
    """

    def __init__(self, in_feats, out_feats, aggregator_type='mean'):
        """
        Args:
            in_feats: Input feature dimension
            out_feats: Output feature dimension
            aggregator_type: 'mean', 'sum', 'pool', or 'lstm'
        """
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggregator_type = aggregator_type

        # Weight matrix for combining node's own features
        self.weight = nn.Linear(in_feats, out_feats)

        # Weight matrix for aggregated neighbor features
        self.agg_weight = nn.Linear(in_feats, out_feats)

        # Different aggregators
        if aggregator_type == 'mean':
            self.aggregator = self._mean_aggregator
        elif aggregator_type == 'sum':
            self.aggregator = self._sum_aggregator
        elif aggregator_type == 'pool':
            self.aggregator = self._pool_aggregator
        elif aggregator_type == 'lstm':
            self.aggregator = self._lstm_aggregator
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")

        logger.info(f"SAGEConv: {in_feats} -> {out_feats} (aggregator={aggregator_type})")

    def _mean_aggregator(self, g, feat):
        """Aggregate by computing mean of neighbors. Handles isolated nodes."""
        g.ndata['h'] = feat
        # Initialize aggregation with zeros (handles nodes with no incoming edges)
        g.ndata['agg_h'] = torch.zeros_like(feat)
        # Message passing: send neighbor features
        g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'agg_h'))
        # Replace NaN with zeros (handles isolated nodes)
        g.ndata['agg_h'] = torch.where(
            torch.isnan(g.ndata['agg_h']),
            torch.zeros_like(g.ndata['agg_h']),
            g.ndata['agg_h']
        )
        return g.ndata['agg_h']

    def _sum_aggregator(self, g, feat):
        """Aggregate by computing sum of neighbors. Handles isolated nodes."""
        g.ndata['h'] = feat
        # Initialize aggregation with zeros
        g.ndata['agg_h'] = torch.zeros_like(feat)
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'agg_h'))
        return g.ndata['agg_h']

    def _pool_aggregator(self, g, feat):
        """Aggregate by max-pooling neighbors. Handles isolated nodes."""
        g.ndata['h'] = feat
        # Initialize with very small values (for max pooling)
        g.ndata['agg_h'] = torch.full_like(feat, -1e9)
        g.update_all(fn.copy_u('h', 'm'), fn.max('m', 'agg_h'))
        # Replace -inf with zeros (handles isolated nodes)
        g.ndata['agg_h'] = torch.where(
            torch.isinf(g.ndata['agg_h']),
            torch.zeros_like(g.ndata['agg_h']),
            g.ndata['agg_h']
        )
        return g.ndata['agg_h']

    def _lstm_aggregator(self, g, feat):
        """Aggregate using LSTM (simplified for this context)."""
        # For simplicity, use mean aggregation
        # Full LSTM aggregation would require keeping neighbor order
        return self._mean_aggregator(g, feat)

    def forward(self, g, feat):
        """
        Args:
            g: DGL graph
            feat: (num_nodes, in_feats) node features

        Returns:
            (num_nodes, out_feats) updated node features
        """
        # Ensure feat is on the same device as model parameters
        device = next(self.parameters()).device
        feat = feat.to(device)
        g = g.to(device)
        # Aggregate neighbor features
        agg_feat = self.aggregator(g, feat)

        # Transform node's own features
        h = self.weight(feat)

        # Transform aggregated neighbor features
        agg_h = self.agg_weight(agg_feat)

        # Combine: h_new = h_node + h_neighbors
        h_combined = h + agg_h

        return h_combined


class GraphSAGE(nn.Module):
    """
    Multi-layer GraphSAGE model.

    Stacks multiple SAGEConv layers to create deeper graph embeddings.
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 activation=None, dropout=0.0, aggregator_type='mean'):
        """
        Args:
            in_feats: Input feature dimension
            n_hidden: Hidden feature dimension
            n_classes: Output feature dimension (final layer)
            n_layers: Number of convolution layers
            activation: Activation function (e.g., nn.ReLU())
            dropout: Dropout rate
            aggregator_type: 'mean', 'sum', 'pool', 'lstm'
        """
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Build layers
        self.layers = nn.ModuleList()

        # First layer: in_feats -> n_hidden
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))

        # Hidden layers: n_hidden -> n_hidden
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))

        # Last layer: n_hidden -> n_classes
        if n_layers > 1:
            self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))
        else:
            self.layers.append(SAGEConv(in_feats, n_classes, aggregator_type))

        logger.info(f"GraphSAGE: {n_layers} layers, {in_feats} -> {n_hidden} -> {n_classes}")

    def forward(self, g, feat):
        """
        Args:
            g: DGL graph
            feat: (num_nodes, in_feats) node features

        Returns:
            (num_nodes, n_classes) final node embeddings
        """
        for i, layer in enumerate(self.layers):
            feat = layer(g, feat)

            # Apply activation (except on last layer)
            if i < len(self.layers) - 1:
                if self.activation:
                    feat = self.activation(feat)
                if self.dropout:
                    feat = self.dropout(feat)

        return feat

# class SAGEConvWithAttention(nn.Module):
#     """
#     GraphSAGE with attention-weighted aggregation.
#
#     Weights neighbor contributions based on learned attention scores.
#     """
#
#     def __init__(self, in_feats, out_feats, use_attention=True):
#         """
#         Args:
#             in_feats: Input feature dimension
#             out_feats: Output feature dimension
#             use_attention: Whether to use attention weighting
#         """
#         super().__init__()
#         self.in_feats = in_feats
#         self.out_feats = out_feats
#         self.use_attention = use_attention
#
#         self.weight = nn.Linear(in_feats, out_feats)
#         self.agg_weight = nn.Linear(in_feats, out_feats)
#
#         if use_attention:
#             # Attention mechanism
#             self.attn = nn.Linear(in_feats, 1)
#
#         logger.info(f"SAGEConvWithAttention: {in_feats} -> {out_feats} (attention={use_attention})")
#
#     def forward(self, g, feat):
#         """
#         Args:
#             g: DGL graph
#             feat: (num_nodes, in_feats) node features
#
#         Returns:
#             (num_nodes, out_feats) updated node features
#         """
#         g.ndata['h'] = feat
#
#         if self.use_attention:
#             # Compute attention scores
#             g.ndata['attn'] = self.attn(feat)
#
#             # Message passing with attention
#             def message_func(edges):
#                 return {
#                     'm': edges.src['h'],
#                     'a': torch.sigmoid(edges.src['attn'])  # Attention weights
#                 }
#
#             def reduce_func(nodes):
#                 # Weight messages by attention
#                 m = nodes.mailbox['m']  # (num_nodes, num_neighbors, in_feats)
#                 a = nodes.mailbox['a']  # (num_nodes, num_neighbors, 1)
#
#                 # Weighted mean
#                 weighted_m = (m * a).sum(dim=1) / (a.sum(dim=1) + 1e-8)
#                 return {'agg_h': weighted_m}
#
#             g.update_all(message_func, reduce_func)
#             agg_feat = g.ndata['agg_h']
#         else:
#             # Simple mean aggregation
#             g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'agg_h'))
#             agg_feat = g.ndata['agg_h']
#
#         # Transform and combine
#         h = self.weight(feat)
#         agg_h = self.agg_weight(agg_feat)
#         h_combined = h + agg_h
#
#         return h_combined