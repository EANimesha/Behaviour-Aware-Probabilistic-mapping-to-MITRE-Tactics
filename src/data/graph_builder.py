"""
Graph Builder Module for Network Context Encoding (DGL-based)
Builds a directed graph where nodes are unique IPs and edges are network flows.
Computes 10-dimensional node features (aggregated per-IP statistics).
Generates temporal sequences (last N flows per source IP).
Uses Deep Graph Library (DGL) for efficient graph operations.
"""

import numpy as np
import pandas as pd
import torch
import dgl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkGraphBuilder:
    """Build network flow graphs for context-aware encoding."""

    def __init__(self, seq_length=10):
        """
        Args:
            seq_length: Number of past flows to include in temporal sequences.
        """
        self.seq_length = seq_length
        self.ip_to_idx = {}
        self.node_features = None
        self.edge_index = None
        self.edge_weights = None
        self.temporal_sequences = None

    def build_graph(self, X_scaled, src_ips, dst_ips):
        """
        Build DGL graph structure and compute node features.

        Args:
            X_scaled: (N, feature_dim) scaled flow features
            src_ips: (N,) source IP addresses
            dst_ips: (N,) destination IP addresses

        Returns:
            graph (dgl.DGLGraph): DGL graph with node features
            temporal_sequences: (N, seq_length, feature_dim) temporal context
        """
        # Create IP to index mapping
        unique_ips = np.unique(np.concatenate([src_ips, dst_ips]))
        self.ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
        n_nodes = len(unique_ips)

        logger.info(f"Graph has {n_nodes} unique IP nodes")
        logger.info(f"Processing {len(src_ips)} flows")

        # Initialize node feature matrix (10 dims per IP)
        node_features = np.zeros((n_nodes, 10), dtype=np.float32)

        # Build edge list and aggregate per-IP statistics
        src_indices = []
        dst_indices = []
        flow_stats = {ip: {'as_src': [], 'as_dst': []} for ip in unique_ips}

        # Create a flow dataframe for easier processing
        flows_df = pd.DataFrame({
            'src_idx': [self.ip_to_idx[ip] for ip in src_ips],
            'dst_idx': [self.ip_to_idx[ip] for ip in dst_ips],
            'features': list(X_scaled)
        })

        # Add edges and group statistics
        for row in flows_df.itertuples(index=False):
            src_idx, dst_idx = row.src_idx, row.dst_idx
            src_indices.append(src_idx)
            dst_indices.append(dst_idx)

            src_ip = unique_ips[src_idx]
            dst_ip = unique_ips[dst_idx]

            flow_stats[src_ip]['as_src'].append(row.features)
            flow_stats[dst_ip]['as_dst'].append(row.features)

        logger.info(f"Graph has {len(src_indices)} directed edges")

        # Compute node features from aggregated flow statistics
        for ip_idx, ip in enumerate(unique_ips):
            node_features[ip_idx] = self._compute_node_features(
                flow_stats[ip], X_scaled
            )

        # Create DGL graph
        graph = dgl.graph((src_indices, dst_indices), num_nodes=n_nodes)
        graph.ndata['feat'] = torch.FloatTensor(node_features)

        self.node_features = torch.FloatTensor(node_features)
        self.src_indices = np.array(src_indices)
        self.dst_indices = np.array(dst_indices)

        # Build temporal sequences for behaviour encoding
        temporal_sequences = self._build_temporal_sequences(
            flows_df, X_scaled
        )
        self.temporal_sequences = temporal_sequences

        logger.info(f"Node features shape: {self.node_features.shape}")
        logger.info(f"Temporal sequences shape: {temporal_sequences.shape}")

        return graph, temporal_sequences

    def _compute_node_features(self, flow_stats, X_scaled):
        """
        Compute 10-dimensional node feature vector for an IP.

        Features (5 as source, 5 as destination):
          As source: bytes_sent, pkts_sent, flow_count, fan_out, avg_pkt_size
          As dest:   bytes_recv, pkts_recv, flow_count, fan_in, avg_throughput

        New feature indices (from updated preprocessor):
          0: IN_PKTS, 1: OUT_BYTES, 2: OUT_PKTS
        """
        features = np.zeros(10, dtype=np.float32)

        # As source (indices 0-4)
        as_src = flow_stats['as_src']
        if len(as_src) > 0:
            as_src = np.array(as_src)
            features[0] = np.log1p(np.sum(as_src[:, 1]))   # log bytes sent (OUT_BYTES)
            features[1] = np.log1p(np.sum(as_src[:, 2]))   # log packets sent (OUT_PKTS)
            features[2] = np.log1p(len(as_src))             # log flow count
            features[3] = np.log1p(len(np.unique(as_src[:, 0] if as_src.shape[1] > 0 else [])))  # fan-out proxy
            features[4] = np.mean(as_src[:, 1]) if as_src.shape[1] > 1 else 0.0  # avg bytes

        # As destination (indices 5-9)
        as_dst = flow_stats['as_dst']
        if len(as_dst) > 0:
            as_dst = np.array(as_dst)
            features[5] = np.log1p(np.sum(as_dst[:, 1]))   # log bytes recv
            features[6] = np.log1p(np.sum(as_dst[:, 0]))   # log packets recv (IN_PKTS)
            features[7] = np.log1p(len(as_dst))             # log flow count
            features[8] = np.log1p(len(np.unique(as_dst[:, 0] if as_dst.shape[1] > 0 else [])))  # fan-in proxy
            features[9] = np.mean(as_dst[:, 1]) if as_dst.shape[1] > 1 else 0.0  # avg bytes

        return features

    def _build_temporal_sequences(self, flows_df, X_scaled):
        """
        Build temporal sequences: last seq_length flows per source IP.
        Returns zero-padded (N, seq_length, feature_dim) tensor.
        """
        n_flows = len(flows_df)
        feature_dim = X_scaled.shape[1]
        temporal_seqs = np.zeros((n_flows, self.seq_length, feature_dim), dtype=np.float32)

        # Group flows by source IP
        flows_df['src_ip'] = [list(self.ip_to_idx.keys())[idx] for idx in flows_df['src_idx']]

        # For each source IP, track its flows chronologically
        for src_ip, group in flows_df.groupby('src_ip'):
            indices = group.index.values

            # For each flow from this source, include the last seq_length flows
            for i, flow_idx in enumerate(indices):
                # Get the last seq_length flows (including current)
                start = max(0, i - self.seq_length + 1)
                seq_indices = indices[start:i+1]

                # Zero-padded if less than seq_length
                seq_data = X_scaled[seq_indices]
                temporal_seqs[flow_idx, -len(seq_data):, :] = seq_data

        return torch.FloatTensor(temporal_seqs)

    def get_ip_embeddings(self, node_embeddings, src_ips, dst_ips, ip_to_idx=None):
        """
        Given node embeddings from GraphSAGE, extract embeddings for each flow's src/dst pair.

        Args:
            node_embeddings: (num_nodes, embedding_dim) from GraphSAGE
            src_ips: (N,) source IP addresses
            dst_ips: (N,) destination IP addresses

        Returns:
            context_embeddings: (N, 2*embedding_dim) concatenated src/dst embeddings
        """
        n_flows = len(src_ips)
        embedding_dim = node_embeddings.shape[1]
        context_embeddings = np.zeros((n_flows, 2 * embedding_dim), dtype=np.float32)

        # Use provided ip_to_idx or fall back to stored one
        if ip_to_idx is None:
            ip_to_idx = self.ip_to_idx

        # Convert node_embeddings to numpy if it's a tensor
        if hasattr(node_embeddings, 'cpu'):
            node_embeddings_np = node_embeddings.cpu().detach().numpy()
        else:
            node_embeddings_np = np.array(node_embeddings)

        # Check for NaN in input embeddings
        if np.isnan(node_embeddings_np).any():
            logger.warning("⚠️  Input node_embeddings contain NaN values!")
            # Replace NaN with zeros
            node_embeddings_np = np.nan_to_num(node_embeddings_np, nan=0.0)

        unseen_count = 0

        for i in range(n_flows):
            # Get embeddings, using zero vector for unseen IPs (inductive learning)
            if src_ips[i] in ip_to_idx:
                src_idx = ip_to_idx[src_ips[i]]
                src_emb = node_embeddings_np[src_idx]
            else:
                # Unseen IP: use zero embedding (inductive capability)
                src_emb = np.zeros(embedding_dim, dtype=np.float32)
                unseen_count += 1

            if dst_ips[i] in ip_to_idx:
                dst_idx = ip_to_idx[dst_ips[i]]
                dst_emb = node_embeddings_np[dst_idx]
            else:
                # Unseen IP: use zero embedding (inductive capability)
                dst_emb = np.zeros(embedding_dim, dtype=np.float32)
                unseen_count += 1

            context_embeddings[i] = np.concatenate([src_emb, dst_emb])

        # Log unseen IPs
        if unseen_count > 0:
            logger.info(f"ℹ️  Unseen IPs: {unseen_count} flows (using zero embeddings for new IPs)")

        # Check output for NaN
        if np.isnan(context_embeddings).any():
            logger.warning("⚠️  Output context_embeddings contain NaN - replacing with zeros")
            context_embeddings = np.nan_to_num(context_embeddings, nan=0.0)

        return torch.FloatTensor(context_embeddings)