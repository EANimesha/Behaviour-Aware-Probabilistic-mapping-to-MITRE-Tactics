"""
Cluster Summarization Module
=============================

For each cluster k discovered by DP-GMM, compute:
  - Feature Importance (in ORIGINAL scale, not z-scores)
  - Behavioural Pattern (decoded to real units)
  - Graph Properties (connectivity patterns)
  - Categorical Feature Decoding (protocol/port proportions)
  - Representative Samples (top flows nearest to centroid)

Output: Cluster k -> Text Summary (Summary_k)
This summary is fed to the LLM for semantic tactic labeling.

KEY: All values shown to the LLM are in original/interpretable scale,
not StandardScaler z-scores. This is critical for GPT-4 to produce
meaningful tactic assignments.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Features that were log1p-transformed before scaling (from preprocess.py)
LOG_FEATURES = {
    'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
}

# Known one-hot feature prefixes for categorical decoding
CATEGORICAL_PREFIXES = {
    'SRC_PORT': ['SRC_PORT_WELL_KNOWN', 'SRC_PORT_REGISTERED', 'SRC_PORT_EPHEMERAL'],
    'DST_PORT': ['DST_PORT_WELL_KNOWN', 'DST_PORT_REGISTERED', 'DST_PORT_EPHEMERAL'],
    'PROTO': ['PROTO_TCP', 'PROTO_UDP', 'PROTO_ICMP', 'PROTO_OTHER'],
    'L7': ['L7_0', 'L7_7', 'L7_118', 'L7_OTHER'],
}

# Human-readable labels for categorical values
CATEGORICAL_LABELS = {
    'SRC_PORT_WELL_KNOWN': 'well-known ports (0-1023)',
    'SRC_PORT_REGISTERED': 'registered ports (1024-49151)',
    'SRC_PORT_EPHEMERAL': 'ephemeral ports (49152-65535)',
    'DST_PORT_WELL_KNOWN': 'well-known ports (0-1023)',
    'DST_PORT_REGISTERED': 'registered ports (1024-49151)',
    'DST_PORT_EPHEMERAL': 'ephemeral ports (49152-65535)',
    'PROTO_TCP': 'TCP',
    'PROTO_UDP': 'UDP',
    'PROTO_ICMP': 'ICMP',
    'PROTO_OTHER': 'Other protocol',
    'L7_0': 'Unknown/Unclassified',
    'L7_7': 'HTTP',
    'L7_118': 'DNS',
    'L7_OTHER': 'Other L7 protocol',
    'ICMP_ACTIVE': 'ICMP active',
}

# Units for human-readable display
FEATURE_UNITS = {
    'OUT_BYTES': 'bytes', 'IN_PKTS': 'packets', 'OUT_PKTS': 'packets',
    'FLOW_DURATION_MILLISECONDS': 'ms', 'DURATION_IN': 'ms', 'DURATION_OUT': 'ms',
    'MIN_TTL': '', 'LONGEST_FLOW_PKT': 'bytes', 'SHORTEST_FLOW_PKT': 'bytes',
    'MIN_IP_PKT_LEN': 'bytes',
    'SRC_TO_DST_SECOND_BYTES': 'bytes/sec', 'DST_TO_SRC_SECOND_BYTES': 'bytes/sec',
    'SRC_TO_DST_AVG_THROUGHPUT': 'bps', 'DST_TO_SRC_AVG_THROUGHPUT': 'bps',
    'NUM_PKTS_UP_TO_128_BYTES': 'packets', 'NUM_PKTS_128_TO_256_BYTES': 'packets',
    'TCP_WIN_MAX_IN': '', 'TCP_WIN_MAX_OUT': '',
    'TCP_FLAGS': '', 'CLIENT_TCP_FLAGS': '', 'SERVER_TCP_FLAGS': '',
}


class ClusterSummarizer:
    """Generate rich text summaries for each DP-GMM cluster.

    Inverse-transforms scaled features back to original scale so
    the LLM receives interpretable values (bytes, packets, ms)
    rather than z-scores.
    """

    def __init__(self, feature_names=None, scaler=None,
                 top_k_features=10, n_representative=5):
        """
        Args:
            feature_names: list of feature column names from preprocessor
            scaler: fitted StandardScaler (for inverse transform)
            top_k_features: number of top features to highlight
            n_representative: number of representative samples to include
        """
        self.feature_names = feature_names or []
        self.scaler = scaler
        self.top_k_features = top_k_features
        self.n_representative = n_representative

        # Build index maps
        self._name_to_idx = {
            name: i for i, name in enumerate(self.feature_names)
        }
        # Identify which feature indices are one-hot categoricals
        self._categorical_indices = {}
        for prefix, cols in CATEGORICAL_PREFIXES.items():
            indices = []
            for col in cols:
                if col in self._name_to_idx:
                    indices.append((col, self._name_to_idx[col]))
            if indices:
                self._categorical_indices[prefix] = indices

    def _inverse_transform_means(self, scaled_means):
        """Convert scaled cluster means back to original scale.

        Steps: inverse StandardScaler -> inverse log1p (expm1) for
        log-transformed features.

        Args:
            scaled_means: (feature_dim,) mean in scaled space

        Returns:
            original_means: (feature_dim,) mean in original scale
        """
        if self.scaler is None:
            return scaled_means

        # Inverse scaler: x_orig = x_scaled * std + mean
        original = (scaled_means * self.scaler.scale_) + self.scaler.mean_

        # Inverse log1p for log-transformed features
        for fname in LOG_FEATURES:
            if fname in self._name_to_idx:
                idx = self._name_to_idx[fname]
                if idx < len(original):
                    original[idx] = np.expm1(max(original[idx], 0))

        return original

    def summarize_cluster(self, cluster_idx, cluster_flows, cluster_raw_features,
                          all_flows, src_ips=None, dst_ips=None,
                          flow_indices=None):
        """Generate a comprehensive text summary for one cluster."""
        n_cluster = len(cluster_flows)
        n_total = len(all_flows)

        # Inverse-transform cluster and global means to original scale
        cluster_mean_scaled = np.mean(cluster_raw_features, axis=0)
        global_mean_scaled = np.mean(all_flows, axis=0)

        cluster_mean_orig = self._inverse_transform_means(cluster_mean_scaled)
        global_mean_orig = self._inverse_transform_means(global_mean_scaled)

        # Feature importance (using original-scale values)
        feat_importance = self._compute_feature_importance(
            cluster_mean_orig, global_mean_orig, cluster_raw_features, all_flows
        )

        # Categorical decoding
        categoricals = self._decode_categoricals(cluster_raw_features)

        # Behavioural pattern (using original-scale values)
        behaviour = self._compute_behaviour_pattern(
            cluster_mean_orig, cluster_raw_features
        )

        # Graph properties
        graph_props = self._compute_graph_properties(src_ips, dst_ips)

        # Representative samples
        representatives = self._get_representative_samples(
            cluster_flows, cluster_raw_features
        )

        # Build text summary with original-scale values
        summary_text = self._build_text_summary(
            cluster_idx, n_cluster, n_total,
            feat_importance, categoricals, behaviour, graph_props,
            representatives
        )

        stats = {
            'cluster_idx': cluster_idx,
            'size': n_cluster,
            'proportion': n_cluster / n_total,
            'feature_importance': feat_importance,
            'categoricals': categoricals,
            'behaviour': behaviour,
            'graph_properties': graph_props,
            'representative_samples': representatives,
        }

        logger.info(f"Cluster {cluster_idx}: {n_cluster} flows "
                     f"({n_cluster/n_total*100:.1f}%)")

        return {'text': summary_text, 'stats': stats}

    def _compute_feature_importance(self, cluster_orig, global_orig,
                                     cluster_scaled, all_scaled):
        """Compute which features deviate most, showing original-scale values."""
        # Use scaled values for effect size computation (proper standardisation)
        global_mean_s = np.mean(all_scaled, axis=0)
        global_std_s = np.std(all_scaled, axis=0) + 1e-8
        cluster_mean_s = np.mean(cluster_scaled, axis=0)
        effect_sizes = (cluster_mean_s - global_mean_s) / global_std_s

        n_features = min(len(effect_sizes), len(self.feature_names))
        top_indices = np.argsort(np.abs(effect_sizes))[::-1][:self.top_k_features]

        importance = []
        for idx in top_indices:
            if idx >= n_features:
                continue

            fname = self.feature_names[idx]
            unit = FEATURE_UNITS.get(fname, '')

            # Skip one-hot categorical features (handled separately)
            if any(fname.startswith(p) for p in
                   ['SRC_PORT_', 'DST_PORT_', 'PROTO_', 'L7_', 'ICMP_']):
                continue

            importance.append({
                'name': fname,
                'effect_size': float(effect_sizes[idx]),
                'cluster_value': float(cluster_orig[idx]),
                'global_value': float(global_orig[idx]),
                'unit': unit,
                'direction': 'HIGH' if effect_sizes[idx] > 0 else 'LOW',
            })

        return importance[:self.top_k_features]

    def _decode_categoricals(self, cluster_features):
        """Decode one-hot encoded features into readable proportions.

        e.g., "Protocol: 78% TCP, 15% UDP, 7% ICMP"
        """
        categoricals = {}
        cluster_mean = np.mean(cluster_features, axis=0)

        for prefix, col_indices in self._categorical_indices.items():
            # Get the mean values of the one-hot columns
            # After inverse scaling, these approximate proportions
            values = {}
            for col_name, idx in col_indices:
                # The scaled mean of a one-hot column approximates
                # the proportion (higher = more prevalent)
                raw_val = cluster_mean[idx]
                label = CATEGORICAL_LABELS.get(col_name, col_name)
                values[label] = float(raw_val)

            # Sort by value (highest proportion first)
            sorted_vals = sorted(values.items(), key=lambda x: x[1], reverse=True)
            categoricals[prefix] = sorted_vals

        # ICMP active
        if 'ICMP_ACTIVE' in self._name_to_idx:
            idx = self._name_to_idx['ICMP_ACTIVE']
            icmp_val = cluster_mean[idx]
            categoricals['ICMP'] = [('ICMP present', float(icmp_val))]

        return categoricals

    def _compute_behaviour_pattern(self, cluster_orig, cluster_features):
        """Characterise behaviour using ORIGINAL-SCALE values."""
        if len(cluster_features) == 0:
            return {}

        pattern = {}
        n = self._name_to_idx

        # Byte volume
        if 'OUT_BYTES' in n:
            val = cluster_orig[n['OUT_BYTES']]
            pattern['avg_out_bytes'] = float(val)
            pattern['volume_class'] = (
                'TINY' if val < 100 else
                'SMALL' if val < 1000 else
                'MEDIUM' if val < 10000 else
                'LARGE' if val < 100000 else
                'VERY_LARGE'
            )

        # Packet counts
        if 'IN_PKTS' in n:
            pattern['avg_in_pkts'] = float(cluster_orig[n['IN_PKTS']])
        if 'OUT_PKTS' in n:
            pattern['avg_out_pkts'] = float(cluster_orig[n['OUT_PKTS']])

        # Duration in milliseconds
        if 'FLOW_DURATION_MILLISECONDS' in n:
            dur_ms = cluster_orig[n['FLOW_DURATION_MILLISECONDS']]
            pattern['avg_duration_ms'] = float(dur_ms)
            pattern['duration_class'] = (
                'VERY_SHORT (<10ms)' if dur_ms < 10 else
                'SHORT (10-100ms)' if dur_ms < 100 else
                'MEDIUM (100ms-1s)' if dur_ms < 1000 else
                'LONG (1-10s)' if dur_ms < 10000 else
                'VERY_LONG (>10s)'
            )

        # Throughput (bytes/sec)
        if 'SRC_TO_DST_SECOND_BYTES' in n and 'DST_TO_SRC_SECOND_BYTES' in n:
            src_bps = cluster_orig[n['SRC_TO_DST_SECOND_BYTES']]
            dst_bps = cluster_orig[n['DST_TO_SRC_SECOND_BYTES']]
            pattern['src_to_dst_bytes_sec'] = float(src_bps)
            pattern['dst_to_src_bytes_sec'] = float(dst_bps)
            ratio = src_bps / (dst_bps + 1e-8)
            pattern['throughput_ratio'] = float(ratio)
            pattern['traffic_direction'] = (
                'OUTBOUND_HEAVY' if ratio > 3.0 else
                'INBOUND_HEAVY' if ratio < 0.33 else
                'BALANCED'
            )

        # Packet sizes
        if 'LONGEST_FLOW_PKT' in n:
            pattern['avg_longest_pkt_bytes'] = float(
                cluster_orig[n['LONGEST_FLOW_PKT']]
            )
        if 'SHORTEST_FLOW_PKT' in n:
            pattern['avg_shortest_pkt_bytes'] = float(
                cluster_orig[n['SHORTEST_FLOW_PKT']]
            )

        # TCP flags
        if 'TCP_FLAGS' in n:
            pattern['avg_tcp_flags'] = float(cluster_orig[n['TCP_FLAGS']])
        if 'CLIENT_TCP_FLAGS' in n:
            pattern['avg_client_tcp_flags'] = float(
                cluster_orig[n['CLIENT_TCP_FLAGS']]
            )

        # TTL
        if 'MIN_TTL' in n:
            pattern['avg_min_ttl'] = float(cluster_orig[n['MIN_TTL']])

        return pattern

    def _compute_graph_properties(self, src_ips, dst_ips):
        """Compute network topology properties of this cluster."""
        if src_ips is None or dst_ips is None:
            return {'available': False}

        unique_src = np.unique(src_ips)
        unique_dst = np.unique(dst_ips)
        unique_pairs = len(set(zip(src_ips, dst_ips)))

        n_flows = len(src_ips)
        fan_out = n_flows / max(len(unique_src), 1)
        fan_in = n_flows / max(len(unique_dst), 1)

        props = {
            'available': True,
            'n_unique_sources': int(len(unique_src)),
            'n_unique_destinations': int(len(unique_dst)),
            'n_unique_pairs': int(unique_pairs),
            'avg_fan_out': float(fan_out),
            'avg_fan_in': float(fan_in),
        }

        if len(unique_src) == 1 and len(unique_dst) > 10:
            props['pattern'] = 'ONE_TO_MANY (scanning/spray)'
        elif len(unique_src) > 10 and len(unique_dst) == 1:
            props['pattern'] = 'MANY_TO_ONE (DDoS/service)'
        elif len(unique_src) == 1 and len(unique_dst) == 1:
            props['pattern'] = 'ONE_TO_ONE (targeted/tunnel)'
        elif unique_pairs < min(len(unique_src), len(unique_dst)) * 0.3:
            props['pattern'] = 'SPARSE (selected targets)'
        else:
            props['pattern'] = 'DISTRIBUTED'

        return props

    def _get_representative_samples(self, cluster_embeddings, cluster_features):
        """Find flows nearest to centroid in latent space."""
        centroid = np.mean(cluster_embeddings, axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        top_indices = np.argsort(distances)[:self.n_representative]

        representatives = []
        for idx in top_indices:
            sample = {'distance_to_centroid': float(distances[idx])}
            feat = cluster_features[idx]
            for i, val in enumerate(feat[:min(len(feat), len(self.feature_names))]):
                sample[self.feature_names[i]] = float(val)
            representatives.append(sample)

        return representatives

    @staticmethod
    def _decode_tcp_flags(flag_value):
        """Decode TCP flag bitmask to human-readable flag names."""
        flag_val = int(round(flag_value))
        if flag_val == 0:
            return "none"
        names = []
        flag_map = [
            (0x01, 'FIN'), (0x02, 'SYN'), (0x04, 'RST'), (0x08, 'PSH'),
            (0x10, 'ACK'), (0x20, 'URG'), (0x40, 'ECE'), (0x80, 'CWR'),
        ]
        for mask, name in flag_map:
            if flag_val & mask:
                names.append(name)
        return '+'.join(names) if names else f"0x{flag_val:02x}"

    @staticmethod
    def _z_strength(z):
        """Describe z-score strength in plain language."""
        az = abs(z)
        if az > 2.0:
            return "STRONGLY"
        elif az > 1.0:
            return "moderately"
        elif az > 0.5:
            return "slightly"
        else:
            return "near average"

    def _build_text_summary(self, cluster_idx, n_cluster, n_total,
                            feat_importance, categoricals, behaviour,
                            graph_props, representatives):
        """Build human-readable text summary for the LLM.

        Includes:
          - Original-scale values with multipliers vs global average
          - Categorical z-scores with plain-English interpretation
          - TCP flag decoding
          - Throughput ratio explanation
          - TTL interpretation
        """
        lines = []
        pct = n_cluster / n_total * 100
        size_note = ("a tiny specialized cluster" if pct < 1 else
                     "a small cluster" if pct < 5 else
                     "a moderate cluster" if pct < 20 else
                     "a large cluster (likely benign-dominant)")
        lines.append(f"=== Cluster {cluster_idx} Summary ===")
        lines.append(f"Size: {n_cluster:,} flows "
                      f"({pct:.1f}% of dataset — {size_note})")

        # ── Feature importance with multipliers ──
        lines.append("\nTop Distinguishing Features "
                      "(original scale, compared to dataset average):")
        for fi in feat_importance[:7]:
            unit = fi.get('unit', '')
            cv = fi['cluster_value']
            gv = fi['global_value']
            direction = "HIGHER" if fi['direction'] == 'HIGH' else "LOWER"

            # Format values
            if abs(cv) > 10000:
                cv_str = f"{cv:,.0f}"
                gv_str = f"{gv:,.0f}"
            elif abs(cv) > 10:
                cv_str = f"{cv:.1f}"
                gv_str = f"{gv:.1f}"
            else:
                cv_str = f"{cv:.3f}"
                gv_str = f"{gv:.3f}"

            # Compute multiplier
            if abs(gv) > 0.001:
                mult = cv / gv
                if mult > 1:
                    mult_str = f" — {mult:.0f}x above normal"
                elif mult < 1 and mult > 0:
                    mult_str = f" — {1/mult:.0f}x below normal"
                else:
                    mult_str = ""
            else:
                mult_str = ""

            lines.append(f"  - {fi['name']}: {cv_str} {unit} "
                          f"({direction} than global avg {gv_str} {unit}"
                          f"{mult_str})")

        # ── Categorical features with z-score interpretation ──
        if categoricals:
            lines.append("\nProtocol & Port Distribution "
                          "(z-score: positive = more prevalent than dataset "
                          "average, negative = less prevalent):")

            if 'PROTO' in categoricals:
                parts = []
                for label, val in categoricals['PROTO']:
                    strength = self._z_strength(val)
                    direction = "elevated" if val > 0 else "reduced"
                    parts.append(f"{label}: z={val:+.2f} ({strength} {direction})")
                lines.append(f"  - Protocol: {parts[0]}")
                for p in parts[1:]:
                    lines.append(f"    {p}")
                # Highlight dominant non-standard protocol
                top_proto = categoricals['PROTO'][0]
                if 'Other' in top_proto[0] and top_proto[1] > 1.0:
                    lines.append(f"    NOTE: Non-TCP/UDP/ICMP protocol is "
                                  f"dominant — may be GRE, SCTP, IGMP, "
                                  f"or tunneling/encapsulation protocol")

            if 'DST_PORT' in categoricals:
                parts = []
                for label, val in categoricals['DST_PORT']:
                    strength = self._z_strength(val)
                    direction = "elevated" if val > 0 else "reduced"
                    parts.append(f"{label}: z={val:+.2f} ({strength} {direction})")
                lines.append(f"  - Destination ports: {', '.join(parts)}")

            if 'SRC_PORT' in categoricals:
                parts = []
                for label, val in categoricals['SRC_PORT']:
                    strength = self._z_strength(val)
                    direction = "elevated" if val > 0 else "reduced"
                    parts.append(f"{label}: z={val:+.2f} ({strength} {direction})")
                lines.append(f"  - Source ports: {', '.join(parts)}")

            if 'L7' in categoricals:
                parts = []
                for label, val in categoricals['L7']:
                    strength = self._z_strength(val)
                    direction = "elevated" if val > 0 else "reduced"
                    parts.append(f"{label}: z={val:+.2f} ({strength} {direction})")
                lines.append(f"  - App layer protocol: {', '.join(parts)}")

            if 'ICMP' in categoricals:
                val = categoricals['ICMP'][0][1]
                lines.append(f"  - ICMP: {'prevalent' if val > 0.5 else 'rare'} "
                              f"in this cluster (z={val:+.2f})")

        # ── Behavioural pattern with interpretations ──
        lines.append("\nBehavioural Pattern:")
        if behaviour.get('avg_duration_ms') is not None:
            lines.append(f"  - Flow duration: "
                          f"{behaviour['avg_duration_ms']:,.0f} ms "
                          f"({behaviour.get('duration_class', '')})")

        if behaviour.get('avg_out_bytes') is not None:
            lines.append(f"  - Bytes sent per flow: "
                          f"{behaviour['avg_out_bytes']:,.0f} bytes "
                          f"({behaviour.get('volume_class', '')})")

        if behaviour.get('avg_in_pkts') is not None:
            lines.append(f"  - Packets in: "
                          f"{behaviour['avg_in_pkts']:,.0f}")
        if behaviour.get('avg_out_pkts') is not None:
            lines.append(f"  - Packets out: "
                          f"{behaviour['avg_out_pkts']:,.0f}")

        # Throughput direction with plain-English explanation
        if behaviour.get('traffic_direction'):
            ratio = behaviour.get('throughput_ratio', 1.0)
            dir_label = behaviour['traffic_direction']
            if dir_label == 'INBOUND_HEAVY':
                inv_ratio = 1.0 / max(ratio, 0.001)
                lines.append(
                    f"  - Traffic direction: INBOUND_HEAVY — destinations "
                    f"send back {inv_ratio:.0f}x more data than sources "
                    f"send out (ratio={ratio:.2f}). Small requests triggering "
                    f"large responses (possible amplification or enumeration)")
            elif dir_label == 'OUTBOUND_HEAVY':
                lines.append(
                    f"  - Traffic direction: OUTBOUND_HEAVY — sources send "
                    f"{ratio:.0f}x more data than they receive back "
                    f"(ratio={ratio:.2f}). Possible data exfiltration or upload")
            else:
                lines.append(
                    f"  - Traffic direction: BALANCED "
                    f"(ratio={ratio:.2f})")

        if behaviour.get('avg_longest_pkt_bytes') is not None:
            lines.append(f"  - Largest packet: "
                          f"{behaviour['avg_longest_pkt_bytes']:,.0f} bytes")

        # TCP flags decoded
        if behaviour.get('avg_tcp_flags') is not None:
            flags_val = behaviour['avg_tcp_flags']
            decoded = self._decode_tcp_flags(flags_val)
            lines.append(f"  - TCP flags: {flags_val:.1f} "
                          f"(decoded: {decoded})")

        # TTL with interpretation
        if behaviour.get('avg_min_ttl') is not None:
            ttl = behaviour['avg_min_ttl']
            if ttl < 32:
                ttl_note = "low — may indicate distant source, proxied " \
                           "traffic, or TTL manipulation"
            elif ttl < 64:
                ttl_note = "moderate"
            elif ttl < 128:
                ttl_note = "typical (likely Linux/Unix origin, TTL~64)"
            else:
                ttl_note = "high (likely Windows origin, TTL~128)"
            lines.append(f"  - Min TTL: {ttl:.0f} ({ttl_note})")

        # ── Graph properties ──
        if graph_props.get('available'):
            n_src = graph_props['n_unique_sources']
            n_dst = graph_props['n_unique_destinations']
            lines.append("\nNetwork Topology:")
            lines.append(f"  - {n_src:,} unique sources -> "
                          f"{n_dst:,} unique destinations "
                          f"(fan-out ratio {n_dst/max(n_src,1):.0f}:1)")
            lines.append(f"  - {graph_props['n_unique_pairs']:,} unique "
                          f"src-dst pairs "
                          f"(each source hits ~"
                          f"{graph_props['n_unique_pairs']/max(n_src,1):.0f} "
                          f"targets)")
            lines.append(f"  - Connectivity: {graph_props['pattern']}")
            lines.append(f"  - Avg fan-out: {graph_props['avg_fan_out']:.1f}, "
                          f"Avg fan-in: {graph_props['avg_fan_in']:.1f}")

        return "\n".join(lines)

    def summarize_all_clusters(self, cluster_assignments, Z_train, X_train,
                                src_ips=None, dst_ips=None):
        """Generate summaries for all discovered clusters."""
        summaries = {}
        unique_clusters = np.unique(cluster_assignments)

        logger.info(f"Summarizing {len(unique_clusters)} clusters...")

        for k in unique_clusters:
            mask = cluster_assignments == k
            if mask.sum() < 2:
                continue

            cluster_z = Z_train[mask]
            cluster_x = X_train[mask]
            cluster_src = src_ips[mask] if src_ips is not None else None
            cluster_dst = dst_ips[mask] if dst_ips is not None else None

            summaries[k] = self.summarize_cluster(
                cluster_idx=k,
                cluster_flows=cluster_z,
                cluster_raw_features=cluster_x,
                all_flows=X_train,
                src_ips=cluster_src,
                dst_ips=cluster_dst,
            )

        logger.info(f"Generated {len(summaries)} cluster summaries")
        return summaries