# """
# Graph Builder Module for Network Context Encoding (DGL-based)
# Builds a directed graph where nodes are unique IPs and edges are network flows.
# Computes 10-dimensional node features (aggregated per-IP statistics).
# Generates temporal sequences (last N flows per source IP).
# Uses Deep Graph Library (DGL) for efficient graph operations.
# """
#
# import numpy as np
# import pandas as pd
# import torch
# import dgl
# import logging
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class NetworkGraphBuilder:
#     """Build network flow graphs for context-aware encoding."""
#
#     def __init__(self, seq_length=10):
#         """
#         Args:
#             seq_length: Number of past flows to include in temporal sequences.
#         """
#         self.seq_length = seq_length
#         self.ip_to_idx = {}
#         self.node_features = None
#         self.edge_index = None
#         self.edge_weights = None
#         self.temporal_sequences = None
#
#     def build_graph(self, X_scaled, src_ips, dst_ips):
#         """
#         Build DGL graph structure and compute node features.
#
#         Args:
#             X_scaled: (N, feature_dim) scaled flow features
#             src_ips: (N,) source IP addresses
#             dst_ips: (N,) destination IP addresses
#
#         Returns:
#             graph (dgl.DGLGraph): DGL graph with node features
#             temporal_sequences: (N, seq_length, feature_dim) temporal context
#         """
#         # Create IP to index mapping
#         unique_ips = np.unique(np.concatenate([src_ips, dst_ips]))
#         self.ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
#         n_nodes = len(unique_ips)
#
#         logger.info(f"Graph has {n_nodes} unique IP nodes")
#         logger.info(f"Processing {len(src_ips)} flows")
#
#         # Initialize node feature matrix (10 dims per IP)
#         node_features = np.zeros((n_nodes, 10), dtype=np.float32)
#
#         # Build edge list and aggregate per-IP statistics
#         src_indices = []
#         dst_indices = []
#         flow_stats = {ip: {'as_src': [], 'as_dst': []} for ip in unique_ips}
#
#         # Create a flow dataframe for easier processing
#         flows_df = pd.DataFrame({
#             'src_idx': [self.ip_to_idx[ip] for ip in src_ips],
#             'dst_idx': [self.ip_to_idx[ip] for ip in dst_ips],
#             'features': list(X_scaled)
#         })
#
#         # Add edges and group statistics
#         for row in flows_df.itertuples(index=False):
#             src_idx, dst_idx = row.src_idx, row.dst_idx
#             src_indices.append(src_idx)
#             dst_indices.append(dst_idx)
#
#             src_ip = unique_ips[src_idx]
#             dst_ip = unique_ips[dst_idx]
#
#             flow_stats[src_ip]['as_src'].append(row.features)
#             flow_stats[dst_ip]['as_dst'].append(row.features)
#
#         logger.info(f"Graph has {len(src_indices)} directed edges")
#
#         # Compute node features from aggregated flow statistics
#         for ip_idx, ip in enumerate(unique_ips):
#             node_features[ip_idx] = self._compute_node_features(
#                 flow_stats[ip], X_scaled
#             )
#
#         # Create DGL graph
#         graph = dgl.graph((src_indices, dst_indices), num_nodes=n_nodes)
#         graph.ndata['feat'] = torch.FloatTensor(node_features)
#
#         self.node_features = torch.FloatTensor(node_features)
#         self.src_indices = np.array(src_indices)
#         self.dst_indices = np.array(dst_indices)
#
#         # Build temporal sequences for behaviour encoding
#         temporal_sequences = self._build_temporal_sequences(
#             flows_df, X_scaled
#         )
#         self.temporal_sequences = temporal_sequences
#
#         logger.info(f"Node features shape: {self.node_features.shape}")
#         logger.info(f"Temporal sequences shape: {temporal_sequences.shape}")
#
#         return graph, temporal_sequences
#
#     def _compute_node_features(self, flow_stats, X_scaled):
#         """
#         Compute 10-dimensional node feature vector for an IP.
#
#         Features (5 as source, 5 as destination):
#           As source: bytes_sent, pkts_sent, flow_count, fan_out, avg_pkt_size
#           As dest:   bytes_recv, pkts_recv, flow_count, fan_in, avg_throughput
#
#         New feature indices (from updated preprocessor):
#           0: IN_PKTS, 1: OUT_BYTES, 2: OUT_PKTS
#         """
#         features = np.zeros(10, dtype=np.float32)
#
#         # As source (indices 0-4)
#         as_src = flow_stats['as_src']
#         if len(as_src) > 0:
#             as_src = np.array(as_src)
#             features[0] = np.log1p(np.sum(as_src[:, 1]))   # log bytes sent (OUT_BYTES)
#             features[1] = np.log1p(np.sum(as_src[:, 2]))   # log packets sent (OUT_PKTS)
#             features[2] = np.log1p(len(as_src))             # log flow count
#             features[3] = np.log1p(len(np.unique(as_src[:, 0] if as_src.shape[1] > 0 else [])))  # fan-out proxy
#             features[4] = np.mean(as_src[:, 1]) if as_src.shape[1] > 1 else 0.0  # avg bytes
#
#         # As destination (indices 5-9)
#         as_dst = flow_stats['as_dst']
#         if len(as_dst) > 0:
#             as_dst = np.array(as_dst)
#             features[5] = np.log1p(np.sum(as_dst[:, 1]))   # log bytes recv
#             features[6] = np.log1p(np.sum(as_dst[:, 0]))   # log packets recv (IN_PKTS)
#             features[7] = np.log1p(len(as_dst))             # log flow count
#             features[8] = np.log1p(len(np.unique(as_dst[:, 0] if as_dst.shape[1] > 0 else [])))  # fan-in proxy
#             features[9] = np.mean(as_dst[:, 1]) if as_dst.shape[1] > 1 else 0.0  # avg bytes
#
#         return features
#
#     def _build_temporal_sequences(self, flows_df, X_scaled):
#         """
#         Build temporal sequences: last seq_length flows per source IP.
#         Returns zero-padded (N, seq_length, feature_dim) tensor.
#         """
#         n_flows = len(flows_df)
#         feature_dim = X_scaled.shape[1]
#         temporal_seqs = np.zeros((n_flows, self.seq_length, feature_dim), dtype=np.float32)
#
#         # Group flows by source IP
#         flows_df['src_ip'] = [list(self.ip_to_idx.keys())[idx] for idx in flows_df['src_idx']]
#
#         # For each source IP, track its flows chronologically
#         for src_ip, group in flows_df.groupby('src_ip'):
#             indices = group.index.values
#
#             # For each flow from this source, include the last seq_length flows
#             for i, flow_idx in enumerate(indices):
#                 # Get the last seq_length flows (including current)
#                 start = max(0, i - self.seq_length + 1)
#                 seq_indices = indices[start:i+1]
#
#                 # Zero-padded if less than seq_length
#                 seq_data = X_scaled[seq_indices]
#                 temporal_seqs[flow_idx, -len(seq_data):, :] = seq_data
#
#         return torch.FloatTensor(temporal_seqs)
#
#     def get_ip_embeddings(self, node_embeddings, src_ips, dst_ips, ip_to_idx=None):
#         """
#         Given node embeddings from GraphSAGE, extract embeddings for each flow's src/dst pair.
#
#         Args:
#             node_embeddings: (num_nodes, embedding_dim) from GraphSAGE
#             src_ips: (N,) source IP addresses
#             dst_ips: (N,) destination IP addresses
#
#         Returns:
#             context_embeddings: (N, 2*embedding_dim) concatenated src/dst embeddings
#         """
#         n_flows = len(src_ips)
#         embedding_dim = node_embeddings.shape[1]
#         context_embeddings = np.zeros((n_flows, 2 * embedding_dim), dtype=np.float32)
#
#         # Use provided ip_to_idx or fall back to stored one
#         if ip_to_idx is None:
#             ip_to_idx = self.ip_to_idx
#
#         # Convert node_embeddings to numpy if it's a tensor
#         if hasattr(node_embeddings, 'cpu'):
#             node_embeddings_np = node_embeddings.cpu().detach().numpy()
#         else:
#             node_embeddings_np = np.array(node_embeddings)
#
#         # Check for NaN in input embeddings
#         if np.isnan(node_embeddings_np).any():
#             logger.warning("⚠️  Input node_embeddings contain NaN values!")
#             # Replace NaN with zeros
#             node_embeddings_np = np.nan_to_num(node_embeddings_np, nan=0.0)
#
#         unseen_count = 0
#
#         for i in range(n_flows):
#             # Get embeddings, using zero vector for unseen IPs (inductive learning)
#             if src_ips[i] in ip_to_idx:
#                 src_idx = ip_to_idx[src_ips[i]]
#                 src_emb = node_embeddings_np[src_idx]
#             else:
#                 # Unseen IP: use zero embedding (inductive capability)
#                 src_emb = np.zeros(embedding_dim, dtype=np.float32)
#                 unseen_count += 1
#
#             if dst_ips[i] in ip_to_idx:
#                 dst_idx = ip_to_idx[dst_ips[i]]
#                 dst_emb = node_embeddings_np[dst_idx]
#             else:
#                 # Unseen IP: use zero embedding (inductive capability)
#                 dst_emb = np.zeros(embedding_dim, dtype=np.float32)
#                 unseen_count += 1
#
#             context_embeddings[i] = np.concatenate([src_emb, dst_emb])
#
#         # Log unseen IPs
#         if unseen_count > 0:
#             logger.info(f"ℹ️  Unseen IPs: {unseen_count} flows (using zero embeddings for new IPs)")
#
#         # Check output for NaN
#         if np.isnan(context_embeddings).any():
#             logger.warning("⚠️  Output context_embeddings contain NaN - replacing with zeros")
#             context_embeddings = np.nan_to_num(context_embeddings, nan=0.0)
#
#         return torch.FloatTensor(context_embeddings)
"""
Graph Builder Module for Network Context Encoding (DGL-based)
Builds a directed graph where nodes are unique IPs and edges are network flows.
Computes 10-dimensional node features (aggregated per-IP statistics).
Generates causal temporal sequences (previous N flows per source IP).
Uses Deep Graph Library (DGL) for efficient graph operations.
"""

import numpy as np
import pandas as pd
import torch
import dgl
import logging

from src.data.preprocess import GRAPH_COLS

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
        self.node_mean = None  # node-feature scaler, fitted on the training graph
        self.node_std = None
        self.ip_to_idx = {}
        self.node_features = None
        self.edge_index = None
        self.edge_weights = None
        self.temporal_sequences = None

    def build_graph(self, X_scaled, src_ips, dst_ips, fit_node_scaler=None):
        """
        Build DGL graph structure and compute node features from the given
        flows only. Pass ONLY flows the model is allowed to see (e.g. the
        train period, or everything before a test window) - never future flows.

        Node features are z-scored per column so the context track has the
        same scale as the other tracks (raw log-sums/counts are ~0-20 and made
        the GraphSAGE output dominate the fused embedding). The mean/std are
        fitted on the first graph built (the training graph) and reused for
        later graphs (test windows), unless fit_node_scaler is given.

        Args:
            X_scaled: (N, len(GRAPH_COLS)) raw graph columns (data['G_all'] rows)
            src_ips: (N,) source IP addresses
            dst_ips: (N,) destination IP addresses

        Returns:
            graph (dgl.DGLGraph): DGL graph with node features
        """
        unique_ips = pd.unique(np.concatenate([src_ips, dst_ips]))
        self.ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
        n_nodes = len(unique_ips)

        ip_index = pd.Index(unique_ips)
        src_indices = ip_index.get_indexer(src_ips)
        dst_indices = ip_index.get_indexer(dst_ips)

        logger.info(f"Graph has {n_nodes} unique IP nodes, {len(src_indices)} directed edges")

        node_features = self._compute_node_features(X_scaled, src_indices, dst_indices, n_nodes)

        if fit_node_scaler is None:
            fit_node_scaler = getattr(self, 'node_mean', None) is None
        if fit_node_scaler:
            self.node_mean = node_features.mean(axis=0)
            self.node_std = node_features.std(axis=0) + 1e-6
        node_features = ((node_features - self.node_mean) / self.node_std).astype(np.float32)

        graph = dgl.graph((torch.as_tensor(src_indices), torch.as_tensor(dst_indices)), num_nodes=n_nodes)
        graph.ndata['feat'] = torch.FloatTensor(node_features)

        self.node_features = torch.FloatTensor(node_features)
        self.src_indices = src_indices
        self.dst_indices = dst_indices

        logger.info(f"Node features shape: {self.node_features.shape}")

        return graph

    @staticmethod
    def _signed_log(x):
        """log1p that keeps the sign (scaled features can be negative)."""
        return np.sign(x) * np.log1p(np.abs(x))

    def _compute_node_features(self, X_scaled, src_indices, dst_indices, n_nodes):
        """
        Compute 10-dimensional node feature vector for every IP (vectorised).
        X_scaled here is the raw graph matrix (GRAPH_COLS: log IN_BYTES,
        log IN_PKTS, L4_DST_PORT, PROTOCOL), so dropping model features does
        not affect the graph.

        Features (5 as source, 5 as destination):
          As source: bytes_sent, pkts_sent, flow_count, fan_out, dst_port_diversity
          As dest:   bytes_recv, pkts_recv, flow_count, fan_in, protocol_diversity
        """
        col = {name: i for i, name in enumerate(GRAPH_COLS)}
        features = np.zeros((n_nodes, 10), dtype=np.float32)
        slog = self._signed_log

        def n_unique(values, groups):
            counts = pd.Series(values).groupby(groups).nunique()
            out = np.zeros(n_nodes, dtype=np.float32)
            out[counts.index.to_numpy()] = np.log1p(counts.to_numpy())
            return out

        # As source (indices 0-4)
        features[:, 0] = slog(np.bincount(src_indices, X_scaled[:, col['IN_BYTES']], n_nodes))
        features[:, 1] = slog(np.bincount(src_indices, X_scaled[:, col['IN_PKTS']], n_nodes))
        features[:, 2] = np.log1p(np.bincount(src_indices, minlength=n_nodes))
        features[:, 3] = n_unique(dst_indices, src_indices)  # fan-out: unique dst IPs
        features[:, 4] = n_unique(X_scaled[:, col['L4_DST_PORT']], src_indices)

        # As destination (indices 5-9)
        features[:, 5] = slog(np.bincount(dst_indices, X_scaled[:, col['IN_BYTES']], n_nodes))
        features[:, 6] = slog(np.bincount(dst_indices, X_scaled[:, col['IN_PKTS']], n_nodes))
        features[:, 7] = np.log1p(np.bincount(dst_indices, minlength=n_nodes))
        features[:, 8] = n_unique(src_indices, dst_indices)  # fan-in: unique src IPs
        features[:, 9] = n_unique(X_scaled[:, col['PROTOCOL']], dst_indices)

        return features

    def build_causal_sequences(self, X_all, src_all, target_positions, history_positions=None):
        """
        Temporal sequence for each target flow: the previous seq_length - 1
        flows from the same source IP that happened BEFORE it, plus the flow
        itself. Built on the full, unbalanced timeline, so the neighbours are
        the real consecutive flows (balancing never thins them).

        Args:
            X_all: (T, feature_dim) scaled features of the whole timeline
            src_all: (T,) source IP of each row
            target_positions: (N,) time positions of the flows to encode
            history_positions: time positions allowed as history. For train
                flows pass the train-period rows (no test data in training
                inputs); for test flows pass None = all earlier traffic.

        Returns:
            sequences: (N, seq_length, feature_dim), zero-padded at the front
            mask: (N, seq_length), 1 = real flow, 0 = padding
        """
        target_positions = np.asarray(target_positions)
        n, L, d = len(target_positions), self.seq_length, X_all.shape[1]
        sequences = np.zeros((n, L, d), dtype=np.float32)
        mask = np.zeros((n, L), dtype=np.float32)

        if history_positions is None:
            history_positions = np.arange(len(X_all))
        history_positions = np.sort(np.asarray(history_positions))

        # Sorted history positions per source IP
        hist_by_ip = pd.Series(history_positions).groupby(src_all[history_positions]).apply(np.asarray).to_dict()
        empty = np.array([], dtype=int)

        for i, t in enumerate(target_positions):
            hist = hist_by_ip.get(src_all[t], empty)
            k = np.searchsorted(hist, t, side='left')  # flows strictly before t
            seq_idx = np.append(hist[max(0, k - (L - 1)):k], t)
            sequences[i, -len(seq_idx):] = X_all[seq_idx]
            mask[i, -len(seq_idx):] = 1.0

        return torch.FloatTensor(sequences), torch.FloatTensor(mask)

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