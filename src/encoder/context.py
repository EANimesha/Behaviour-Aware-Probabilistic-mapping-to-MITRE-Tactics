from torch import nn
import dgl

from src.encoder.GraphSAGE import GraphSAGE


class ContextEncoder(nn.Module):
    """Track 2: DGL GraphSAGE-based network context encoder."""

    def __init__(self, node_feature_dim, hidden_dim=32, output_dim=32, dropout=0.1):
        super().__init__()
        # DGL GraphSAGE with 2 layers
        self.sage = GraphSAGE(
            in_feats=node_feature_dim,
            n_hidden=hidden_dim,
            n_classes=output_dim,
            n_layers=2,
            activation=nn.ReLU(),
            dropout=dropout,
            aggregator_type='mean'  # 'mean', 'gcn', 'pool', 'lstm'
        )

        print(f"ContextEncoder (DGL GraphSAGE): {node_feature_dim} -> {hidden_dim} -> {output_dim}")

    def forward(self, graph, node_features):
        """
        Args:
            graph: dgl.DGLGraph
            node_features: (num_nodes, node_feature_dim) node features

        Returns:
            (num_nodes, output_dim) node embeddings
        """
        # GraphSAGE forward pass
        node_embeddings = self.sage(graph, node_features)
        return node_embeddings
