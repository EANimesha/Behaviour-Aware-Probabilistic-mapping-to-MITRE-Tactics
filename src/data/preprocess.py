# """
# Preprocessing Module for NF-BoT-IoT-v2
# ========================================
#
# Based on EDA notebook findings:
#   1. Clean: remove inf/NaN, handle missing values
#   2. Drop: correlated, VIF, near-zero variance, and useless features
#   3. Engineer: categorical features (ports, protocols, ICMP) into
#      meaningful bucketed + one-hot representations
#   4. Transform: log1p on skewed numeric features
#   5. Scale: StandardScaler on numeric features only
#   6. Split: stratified train/test by Attack type
# """
#
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import logging
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # ── Column classifications ──
#
# IDENTIFIER_COLS = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
# LABEL_COLS = ['Label', 'Attack']
#
# # Categorical features that need engineering, NOT numeric scaling
# CATEGORICAL_COLS = [
#     'PROTOCOL',              # 6=TCP, 17=UDP, 1=ICMP
#     'L7_PROTO',              # application protocol codes
#     'TCP_FLAGS',             # bitmask — treat as numeric
#     'CLIENT_TCP_FLAGS',      # bitmask — treat as numeric
#     'SERVER_TCP_FLAGS',      # bitmask — treat as numeric
#     'ICMP_TYPE',             # ICMP message type codes
#     'ICMP_IPV4_TYPE',        # ICMP subtype codes
#     'DNS_QUERY_TYPE',        # 1=A, 2=NS, 5=CNAME, 28=AAAA
#     'FTP_COMMAND_RET_CODE',  # 200, 331, 530, 550
#     'L4_SRC_PORT',           # port numbers
#     'L4_DST_PORT',           # port numbers
# ]
#
# # ── Features to DROP (from EDA notebook) ──
#
# # Correlation > 0.94 with another feature
# DROP_CORRELATION = [
#     'MAX_TTL',                     # correlated with MIN_TTL
#     'MAX_IP_PKT_LEN',             # correlated with LONGEST_FLOW_PKT
#     'RETRANSMITTED_IN_PKTS',      # correlated with RETRANSMITTED_IN_BYTES
#     'NUM_PKTS_1024_TO_1514_BYTES', # correlated with OUT_BYTES
# ]
#
# # High VIF (multicollinearity)
# DROP_VIF = [
#     'IN_BYTES',                    # VIF too high
#     'NUM_PKTS_512_TO_1024_BYTES',  # VIF too high
#     'NUM_PKTS_256_TO_512_BYTES',   # VIF too high
# ]
#
# # Near-zero variance (>90% zeros — no discriminative power)
# DROP_ZERO_VARIANCE = [
#     'RETRANSMITTED_IN_BYTES',   # 91.7% zeros
#     'RETRANSMITTED_OUT_BYTES',  # 97.6% zeros
#     'RETRANSMITTED_OUT_PKTS',   # 97.6% zeros
#     'DNS_TTL_ANSWER',           # 89.9% zeros (also near-zero)
#     'DNS_QUERY_TYPE',           # mostly zeros, categorical
#     'ICMP_IPV4_TYPE',           # 89.9% zeros
# ]
#
# # No predictive value
# DROP_USELESS = [
#     'DNS_QUERY_ID',             # random transaction ID
#     'FTP_COMMAND_RET_CODE',     # mostly zeros, rarely used
# ]
#
# # All features to drop (union, deduplicated)
# ALL_DROPS = list(set(
#     DROP_CORRELATION + DROP_VIF + DROP_ZERO_VARIANCE + DROP_USELESS
# ))
#
# # ── Final numeric features (after drops) to log-transform ──
# LOG_FEATURES = [
#     'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
#     'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
#     'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
#     'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
#     'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
# ]
#
# # Final numeric features to scale (after EDA cleanup)
# NUMERIC_FEATURES = [
#     'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
#     'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
#     'MIN_TTL',
#     'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
#     'MIN_IP_PKT_LEN',
#     'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
#     'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
#     'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
#     'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
# ]
#
# # TCP flag bitmask features — keep as numeric (ordinal intensity)
# TCP_FLAG_FEATURES = ['TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS']
#
#
# class Preprocessor:
#     """Full preprocessing pipeline based on EDA notebook findings.
#
#     Handles:
#       - Cleaning (inf/NaN)
#       - Feature drops (correlation, VIF, zero-variance, useless)
#       - Categorical feature engineering (ports, protocols, ICMP)
#       - Log-transform on skewed numerics
#       - StandardScaler on numerics + TCP flags
#       - Stratified train/test split
#     """
#
#     def __init__(self, clip_extremes=False):
#         """
#         Args:
#             clip_extremes: if True, clip to 0.1th/99.9th percentile.
#                            Needed for CIC-IDS2018/UNSW (overflow prevention).
#                            Not for BoT-IoT (extreme values are the signal).
#         """
#         self.scaler = StandardScaler()
#         self.fitted = False
#         self.final_feature_names = []
#         self._clip_extremes = clip_extremes
#
#     def preprocess_dataset(self, df, test_size=0.2, stratify_col='Attack',
#                            random_state=42):
#         """Full pipeline: clean -> engineer -> transform -> split -> scale.
#
#         Args:
#             df: raw DataFrame from loader
#             test_size: fraction for test split
#             stratify_col: column to stratify on
#             random_state: reproducibility seed
#
#         Returns:
#             dict with X_train, X_test, y_train, y_test, src/dst IPs,
#             feature_names, scaler, etc.
#         """
#         df = self._clean(df)
#
#         # Preserve IPs and labels before feature engineering
#         src_ips = df['IPV4_SRC_ADDR'].values
#         dst_ips = df['IPV4_DST_ADDR'].values
#         y_attack = df['Attack'].values if 'Attack' in df.columns else None
#         y_binary = df['Label'].values if 'Label' in df.columns else None
#
#         # ── Feature Engineering ──
#         X = self._engineer_features(df)
#
#         self.final_feature_names = list(X.columns)
#         logger.info(f"Final feature count: {len(self.final_feature_names)}")
#         logger.info(f"Features: {self.final_feature_names}")
#
#         # ── Stratified Split ──
#         indices = np.arange(len(X))
#         stratify = y_attack if stratify_col == 'Attack' else y_binary
#
#         idx_train, idx_test = train_test_split(
#             indices, test_size=test_size,
#             stratify=stratify, random_state=random_state
#         )
#
#         X_train_df = X.iloc[idx_train].copy()
#         X_test_df = X.iloc[idx_test].copy()
#
#         # ── Log Transform (before scaling) ──
#         X_train_df = self._apply_log_transform(X_train_df)
#         X_test_df = self._apply_log_transform(X_test_df)
#
#         # ── Scale: fit on train, transform both ──
#         X_train_scaled = self._fit_transform_scale(X_train_df)
#         X_test_scaled = self._transform_scale(X_test_df)
#
#         logger.info(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
#
#         return {
#             'X_train': X_train_scaled,
#             'X_test': X_test_scaled,
#             'y_train': y_attack[idx_train] if y_attack is not None else None,
#             'y_test': y_attack[idx_test] if y_attack is not None else None,
#             'y_binary_train': y_binary[idx_train] if y_binary is not None else None,
#             'y_binary_test': y_binary[idx_test] if y_binary is not None else None,
#             'src_train': src_ips[idx_train],
#             'src_test': src_ips[idx_test],
#             'dst_train': dst_ips[idx_train],
#             'dst_test': dst_ips[idx_test],
#             'feature_names': self.final_feature_names,
#             'scaler': self.scaler,
#             'train_indices': idx_train,
#             'test_indices': idx_test,
#         }
#
#     def transform_new(self, df):
#         """Transform new data (inference/streaming) using fitted scaler.
#
#         Args:
#             df: raw DataFrame (must contain same columns as training data)
#
#         Returns:
#             X_scaled: (N, feature_dim) numpy array
#             src_ips, dst_ips: IP arrays
#         """
#         if not self.fitted:
#             raise ValueError("Preprocessor not fitted. Call preprocess_dataset first.")
#
#         df = self._clean(df)
#         src_ips = df['IPV4_SRC_ADDR'].values
#         dst_ips = df['IPV4_DST_ADDR'].values
#
#         X = self._engineer_features(df)
#
#         # Ensure same columns in same order
#         for col in self.final_feature_names:
#             if col not in X.columns:
#                 X[col] = 0
#         X = X[self.final_feature_names]
#
#         X = self._apply_log_transform(X)
#         X_scaled = self._transform_scale(X)
#
#         return X_scaled, src_ips, dst_ips
#
#     # ────────────────────────────────────────────────────────
#     # Internal methods
#     # ────────────────────────────────────────────────────────
#
#     def _clean(self, df):
#         """Handle missing values, infinities, duplicates."""
#         n_before = len(df)
#
#         # Replace inf with NaN
#         df = df.replace([np.inf, -np.inf], np.nan)
#
#         # Drop rows missing critical columns
#         critical = [c for c in IDENTIFIER_COLS + LABEL_COLS if c in df.columns]
#         df = df.dropna(subset=critical)
#
#         # Fill remaining NaN in numeric columns with 0
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         df[numeric_cols] = df[numeric_cols].fillna(0)
#
#         logger.info(f"Cleaned: {n_before:,} -> {len(df):,} rows")
#         return df
#
#     def _engineer_features(self, df):
#         """Build the final feature matrix with engineered categoricals.
#
#         Based on EDA notebook cells 28-32:
#           - Ports: bucketed into well_known/registered/ephemeral
#           - PROTOCOL: one-hot (TCP, UDP, ICMP)
#           - L7_PROTO: bucketed (0, 7, 118, Other)
#           - ICMP_TYPE: binary (0 vs non-0)
#           - TCP flags: kept as numeric
#         """
#         parts = []
#
#         # ── 1. Numeric features ──
#         available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
#         parts.append(df[available_numeric].copy())
#
#         # ── 2. TCP flags (keep as numeric — ordinal bitmask values) ──
#         available_tcp = [c for c in TCP_FLAG_FEATURES if c in df.columns]
#         parts.append(df[available_tcp].copy())
#
#         # ── 3. Port bucketing (src & dst) ──
#         for port_col, prefix in [('L4_SRC_PORT', 'SRC_PORT'),
#                                   ('L4_DST_PORT', 'DST_PORT')]:
#             if port_col in df.columns:
#                 port_df = self._bucket_ports(df[port_col], prefix)
#                 parts.append(port_df)
#
#         # ── 4. PROTOCOL one-hot ──
#         if 'PROTOCOL' in df.columns:
#             proto_df = self._encode_protocol(df['PROTOCOL'])
#             parts.append(proto_df)
#
#         # ── 5. L7_PROTO bucketing ──
#         if 'L7_PROTO' in df.columns:
#             l7_df = self._encode_l7_proto(df['L7_PROTO'])
#             parts.append(l7_df)
#
#         # ── 6. ICMP_TYPE binary ──
#         if 'ICMP_TYPE' in df.columns:
#             icmp_binary = (df['ICMP_TYPE'] != 0).astype(np.float32)
#             parts.append(pd.DataFrame(
#                 {'ICMP_ACTIVE': icmp_binary.values}, index=df.index
#             ))
#
#         # Combine all parts
#         X = pd.concat(parts, axis=1)
#
#         # Safety check: ensure no dropped features leaked through
#         leaked = [c for c in X.columns if c in ALL_DROPS]
#         if leaked:
#             logger.warning(f"Dropping leaked features: {leaked}")
#             X = X.drop(columns=leaked)
#
#         # Ensure float32
#         X = X.astype(np.float32)
#
#         return X
#
#     def _bucket_ports(self, port_series, prefix):
#         """Bucket ports into: well_known (0-1023), registered (1024-49151),
#         ephemeral (49152-65535). Returns one-hot DataFrame."""
#         buckets = pd.cut(
#             port_series.clip(0, 65535),
#             bins=[-1, 1023, 49151, 65535],
#             labels=[f'{prefix}_WELL_KNOWN', f'{prefix}_REGISTERED',
#                     f'{prefix}_EPHEMERAL']
#         )
#         return pd.get_dummies(buckets, dtype=np.float32)
#
#     def _encode_protocol(self, proto_series):
#         """One-hot encode PROTOCOL: TCP(6), UDP(17), ICMP(1), Other."""
#         proto_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
#         mapped = proto_series.map(proto_map).fillna('OTHER')
#         dummies = pd.get_dummies(mapped, prefix='PROTO', dtype=np.float32)
#         return dummies
#
#     def _encode_l7_proto(self, l7_series):
#         """Bucket L7_PROTO: 0 (unknown), 7 (HTTP), 118 (specific), Other."""
#         l7_mapped = l7_series.copy()
#         keep_values = {0.0, 7.0, 118.0}
#         l7_mapped = l7_mapped.apply(
#             lambda x: str(int(x)) if x in keep_values else 'OTHER'
#         )
#         dummies = pd.get_dummies(l7_mapped, prefix='L7', dtype=np.float32)
#         return dummies
#
#     def _apply_log_transform(self, X_df):
#         """Apply log1p to skewed features."""
#         X = X_df.copy()
#         for col in LOG_FEATURES:
#             if col in X.columns:
#                 X[col] = np.log1p(np.clip(X[col].values, 0, None))
#         return X
#
#     def _fit_transform_scale(self, X_df):
#         """Fit scaler on training data and transform."""
#         self.scaler.fit(X_df.values)
#         self.fitted = True
#         return self.scaler.transform(X_df.values).astype(np.float32)
#
#     def _transform_scale(self, X_df):
#         """Transform using already-fitted scaler."""
#         return self.scaler.transform(X_df.values).astype(np.float32)
#     # def _fit_transform_scale(self, X_df):
#     #     """Fit scaler on training data and transform.
#     #     Handles inf/nan for all datasets. Clipping only when enabled.
#     #     """
#     #     X = X_df.values.astype(np.float64)  # float64 to avoid overflow
#     #
#     #     # Replace inf with nan (needed for ALL datasets)
#     #     X = np.where(np.isinf(X), np.nan, X)
#     #
#     #     # Replace nan with column median (needed for ALL datasets)
#     #     for col in range(X.shape[1]):
#     #         mask = np.isnan(X[:, col])
#     #         if mask.any():
#     #             median = np.nanmedian(X[:, col])
#     #             X[mask, col] = median if np.isfinite(median) else 0.0
#     #
#     #     # Clip extreme values ONLY if enabled
#     #     # BoT-IoT: no clipping needed (extreme values ARE the signal)
#     #     # CIC-IDS2018/UNSW: clipping prevents overflow in StandardScaler
#     #     if self._clip_extremes:
#     #         for col in range(X.shape[1]):
#     #             lo = np.percentile(X[:, col], 0.1)
#     #             hi = np.percentile(X[:, col], 99.9)
#     #             X[:, col] = np.clip(X[:, col], lo, hi)
#     #
#     #     self.scaler.fit(X)
#     #     self.fitted = True
#     #     return self.scaler.transform(X).astype(np.float32)
#     #
#     # def _transform_scale(self, X_df):
#     #     """Transform using already-fitted scaler."""
#     #     X = X_df.values.astype(np.float64)
#     #     X = np.where(np.isinf(X), np.nan, X)
#     #     for col in range(X.shape[1]):
#     #         mask = np.isnan(X[:, col])
#     #         if mask.any():
#     #             median = np.nanmedian(X[:, col])
#     #             X[mask, col] = median if np.isfinite(median) else 0.0
#     #     return self.scaler.transform(X).astype(np.float32)

"""
Preprocessing Module for NF-v3 datasets (leakage-safe)
=======================================================

Feature choices come from the train-split feature analysis notebook
(feature_analysis_nf_v3.ipynb -> features_<dataset>.json):
  1. Clean: remove inf/NaN, handle missing values
  2. Sort by FLOW_START_MILLISECONDS (time order; never used as a feature)
  3. Split: per-class chronological split  | train | gap (dropped) | test |
  4. Remove test flows identical to a train flow
  5. Balance: select train/test rows (train may oversample, test never)
  6. Engineer: numeric features from the JSON, TCP flags, port buckets,
     PROTOCOL / L7_PROTO one-hot, ICMP active flag
  7. Transform: log1p on skewed numeric features (from the JSON)
  8. Scale: StandardScaler fit on train-period rows only

Every row of the timeline is transformed (X_all), so temporal sequences and
graphs can use the real, unthinned history; idx_train / idx_test say which
rows are encoded and evaluated.
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Column classifications ──

IDENTIFIER_COLS = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
LABEL_COLS = ['Label', 'Attack']
# Time stamps: used ONLY to sort flows, never as model features
# (each attack was captured in its own time window, so they would leak the class)
TIME_COLS = ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS']

# All NF-v3 flow feature columns (raw). Used for column checks and duplicate removal.
RAW_FEATURE_COLS = [
    'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
    'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES',
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
    'ICMP_TYPE', 'ICMP_IPV4_TYPE',
    'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER',
    'FTP_COMMAND_RET_CODE',
    'SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX', 'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
    'DST_TO_SRC_IAT_MIN', 'DST_TO_SRC_IAT_MAX', 'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV',
]

# TCP flag bitmask features — keep as numeric (ordinal intensity)
TCP_FLAG_FEATURES = ['TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS']

# Raw columns used for IP-graph node features (independent of feature drops)
GRAPH_COLS = ['IN_BYTES', 'IN_PKTS', 'L4_DST_PORT', 'PROTOCOL']

# Fixed category sets, so train and test always get the same one-hot columns
PORT_BUCKETS = ['WELL_KNOWN', 'REGISTERED', 'EPHEMERAL']
PROTOCOLS = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}


def load_feature_config(path):
    """Read features_<dataset>.json written by the feature analysis notebook."""
    with open(path) as f:
        cfg = json.load(f)
    logger.info(f"Feature config {path}: {len(cfg['numeric_features'])} numeric, "
                f"{len(cfg['log_features'])} log, L7 buckets {cfg['l7_proto_buckets']}")
    return cfg


def check_columns(df: pd.DataFrame, numeric_features):
    """Stop early with a clear message if the CSV is missing expected columns."""
    required = (IDENTIFIER_COLS + ['Attack'] + list(numeric_features) + TCP_FLAG_FEATURES
                + ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'ICMP_TYPE'] + GRAPH_COLS)
    missing = sorted(set(c for c in required if c not in df.columns))
    if missing:
        raise ValueError(f"Dataset is missing {len(missing)} expected column(s): {missing}")
    if not any(c in df.columns for c in TIME_COLS):
        print("  ⚠️  No FLOW_START_MILLISECONDS column: CSV row order is used as time order")


class Preprocessor:
    """Leakage-safe preprocessing pipeline.

    Handles:
      - Cleaning (inf/NaN) and time sorting
      - Per-class chronological split with a gap, duplicate removal
      - Row selection by the balancers
      - Feature engineering (numeric, TCP flags, ports, protocols, ICMP)
      - Log-transform on skewed numerics
      - StandardScaler fit on train-period rows only
    """

    def __init__(self, feature_config):
        """
        Args:
            feature_config: dict from load_feature_config() (or a path to the JSON).
        """
        if isinstance(feature_config, str):
            feature_config = load_feature_config(feature_config)
        self.numeric_features = list(feature_config['numeric_features'])
        self.log_features = list(feature_config['log_features'])
        self.l7_keep = sorted(float(v) for v in feature_config['l7_proto_buckets'])
        self.scaler = StandardScaler()
        self.fitted = False
        self.final_feature_names = []

    # ────────────────────────────────────────────────────────
    # Split helpers
    # ────────────────────────────────────────────────────────

    def chronological_split(self, df: pd.DataFrame, test_size=0.15, gap_size=0.05,
                            stratify_col='Attack'):
        """
        Per-class chronological split with a gap.

        Within every class (rows already in time order):
            | earliest flows -> TRAIN | gap_size -> DROPPED | latest test_size -> TEST |

        - Every class appears in train and test, even classes that only occur
          at the end of the capture.
        - For each class, all train flows happen before all test flows.
        - The gap removes the flows around the cut, so one burst/session does
          not continue from train into test.

        Returns:
            idx_train, idx_gap, idx_test: positional indices, each sorted by time
        """
        labels = df[stratify_col].values
        idx_train, idx_gap, idx_test = [], [], []

        for label in pd.unique(labels):
            positions = np.where(labels == label)[0]  # already in time order
            n_test = int(round(len(positions) * test_size))
            n_gap = int(round(len(positions) * gap_size))

            if n_test == 0 or len(positions) - n_test - n_gap < 1:
                print(f"  ⚠️  Class '{label}' has {len(positions)} rows - kept in train only")
                idx_train.extend(positions)
                continue

            n_train = len(positions) - n_test - n_gap
            idx_train.extend(positions[:n_train])
            idx_gap.extend(positions[n_train:n_train + n_gap])
            idx_test.extend(positions[n_train + n_gap:])

        return (np.sort(np.array(idx_train, dtype=int)),
                np.sort(np.array(idx_gap, dtype=int)),
                np.sort(np.array(idx_test, dtype=int)))

    def remove_cross_split_duplicates(self, df: pd.DataFrame, idx_train, idx_test, decimals=6):
        """
        Drop test flows whose raw feature vector exactly matches a training
        flow (after rounding), so the same flow cannot be scored in both sets.

        Returns:
            idx_test without duplicates, number dropped
        """
        cols = [c for c in RAW_FEATURE_COLS if c in df.columns]
        rows = np.round(df[cols].to_numpy(dtype=np.float64), decimals)
        keys = pd.util.hash_pandas_object(pd.DataFrame(rows), index=False).to_numpy()
        train_keys = set(keys[idx_train])
        keep = np.array([keys[i] not in train_keys for i in idx_test], dtype=bool)
        return idx_test[keep], int((~keep).sum())

    # ────────────────────────────────────────────────────────
    # Main pipeline
    # ────────────────────────────────────────────────────────

    def preprocess_dataset(self, df: pd.DataFrame, test_size=0.15, gap_size=0.05,
                           stratify_col='Attack', time_col=None, train_balancer=None,
                           test_balancer=None, drop_duplicates=True):
        """
        Args:
            time_col: Optional timestamp column. If None, auto-detects
                FLOW_START_MILLISECONDS; without it, CSV row order is used.
            train_balancer / test_balancer: FastChronologicalBalancer, or None.
                Build test_balancer with allow_oversample=False.

        Returns dict with:
            X_all, G_all, src_all, dst_all, y_all: every row of the timeline
                (X_all = model features, G_all = raw graph columns)
            idx_train_split: all train-period rows (for graph + sequence history)
            idx_train, idx_test: selected rows to encode / evaluate (time
                positions; idx_train may repeat rows of oversampled classes)
            X_train, X_test, src_*, dst_*, y_*: the selected rows, for convenience
            df_raw: cleaned, time-sorted raw DataFrame (df_raw.iloc[idx_train]
                gives the raw rows of the training flows)
        """
        check_columns(df, self.numeric_features)
        df = self._clean(df)

        # Put rows in time order and make the index the time position
        if time_col is None and 'FLOW_START_MILLISECONDS' in df.columns:
            time_col = 'FLOW_START_MILLISECONDS'
        if time_col is not None:
            df = df.sort_values(time_col, kind='stable')
        df = df.reset_index(drop=True)

        # ── Split BEFORE balancing so oversampled duplicates cannot leak into test ──
        idx_train_split, idx_gap, idx_test_split = self.chronological_split(
            df, test_size, gap_size, stratify_col)
        print(f"  Split: train {len(idx_train_split):,}, gap (dropped) {len(idx_gap):,}, "
              f"test {len(idx_test_split):,}")

        n_dup = 0
        if drop_duplicates:
            idx_test_split, n_dup = self.remove_cross_split_duplicates(df, idx_train_split, idx_test_split)
            print(f"  Removed {n_dup:,} test flows identical to a train flow")

        # ── Balancing = selection of rows; the timeline itself stays complete ──
        balance_stats = {}
        idx_train, idx_test = idx_train_split, idx_test_split
        if train_balancer is not None:
            print("  Balancing train split...")
            idx_train, balance_stats['train'] = train_balancer.select(df.iloc[idx_train_split])
        if test_balancer is not None:
            print("  Balancing test split...")
            idx_test, balance_stats['test'] = test_balancer.select(df.iloc[idx_test_split])

        self._print_class_counts(df.iloc[idx_train], df.iloc[idx_test], stratify_col)

        # ── Feature engineering + log transform (row-wise, no statistics) ──
        X = self._apply_log_transform(self._engineer_features(df))
        self.final_feature_names = list(X.columns)
        logger.info(f"Final feature count: {len(self.final_feature_names)}")
        logger.info(f"Features: {self.final_feature_names}")

        # ── Scale: fit on train-period rows only, transform every row ──
        X_all = X.to_numpy(dtype=np.float64)
        self.scaler.fit(X_all[idx_train_split])
        self.fitted = True
        X_all = self.scaler.transform(X_all).astype(np.float32)

        G_all = self._graph_features(df)
        src_all = df['IPV4_SRC_ADDR'].values
        dst_all = df['IPV4_DST_ADDR'].values
        y_all = df['Attack'].values if 'Attack' in df.columns else None
        y_bin = df['Label'].values if 'Label' in df.columns else None

        print(f"Train: {len(idx_train):,} flows, Test: {len(idx_test):,} flows, "
              f"Timeline: {len(df):,} flows, Features: {X_all.shape[1]}")

        return {
            'X_all': X_all,
            'G_all': G_all,
            'src_all': src_all,
            'dst_all': dst_all,
            'y_all': y_all,
            'idx_train_split': idx_train_split,
            'idx_test_split': idx_test_split,
            'idx_train': idx_train,
            'idx_test': idx_test,
            'X_train': X_all[idx_train],
            'X_test': X_all[idx_test],
            'y_train': y_all[idx_train] if y_all is not None else None,
            'y_test': y_all[idx_test] if y_all is not None else None,
            'y_binary_train': y_bin[idx_train] if y_bin is not None else None,
            'y_binary_test': y_bin[idx_test] if y_bin is not None else None,
            'src_train': src_all[idx_train],
            'src_test': src_all[idx_test],
            'dst_train': dst_all[idx_train],
            'dst_test': dst_all[idx_test],
            'feature_names': self.final_feature_names,
            'scaler': self.scaler,
            'balance_stats': balance_stats,
            'n_duplicates_removed': n_dup,
            'df_raw': df,  # cleaned + time-sorted raw rows; row i = time position i
        }

    def transform_new(self, df):
        """Transform new data (inference/streaming) using the fitted scaler.

        Returns:
            X_scaled: (N, feature_dim) numpy array
            src_ips, dst_ips: IP arrays
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call preprocess_dataset first.")

        df = self._clean(df)
        X = self._apply_log_transform(self._engineer_features(df))[self.final_feature_names]
        X_scaled = self.scaler.transform(X.to_numpy(dtype=np.float64)).astype(np.float32)
        return X_scaled, df['IPV4_SRC_ADDR'].values, df['IPV4_DST_ADDR'].values

    # ────────────────────────────────────────────────────────
    # Internal methods
    # ────────────────────────────────────────────────────────

    def _clean(self, df):
        """Handle missing values and infinities."""
        n_before = len(df)

        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop rows missing critical columns
        critical = [c for c in IDENTIFIER_COLS + LABEL_COLS if c in df.columns]
        df = df.dropna(subset=critical)

        # Fill remaining NaN in numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        logger.info(f"Cleaned: {n_before:,} -> {len(df):,} rows")
        return df

    def _engineer_features(self, df):
        """Build the final feature matrix with engineered categoricals.

          - Numeric: the features kept by the feature analysis (JSON)
          - TCP flags: kept as numeric
          - Ports: bucketed into well_known/registered/ephemeral (one-hot)
          - PROTOCOL: one-hot (TCP, UDP, ICMP, OTHER)
          - L7_PROTO: one-hot of the JSON buckets + OTHER
          - ICMP_TYPE: binary (0 vs non-0)
        """
        parts = [df[self.numeric_features].copy(), df[TCP_FLAG_FEATURES].copy()]

        for port_col, prefix in [('L4_SRC_PORT', 'SRC_PORT'), ('L4_DST_PORT', 'DST_PORT')]:
            parts.append(self._bucket_ports(df[port_col], prefix))

        parts.append(self._encode_protocol(df['PROTOCOL']))
        parts.append(self._encode_l7_proto(df['L7_PROTO']))
        parts.append(pd.DataFrame({'ICMP_ACTIVE': (df['ICMP_TYPE'] != 0).astype(np.float32)},
                                  index=df.index))

        return pd.concat(parts, axis=1).astype(np.float32)

    def _one_hot(self, values, categories, prefix):
        """One-hot with a fixed category list (same columns for any subset of rows)."""
        cat = pd.Categorical(values, categories=categories)
        return pd.get_dummies(cat, prefix=prefix, dtype=np.float32).set_axis(values.index)

    def _bucket_ports(self, port_series, prefix):
        """Bucket ports into: well_known (0-1023), registered (1024-49151),
        ephemeral (49152-65535). Returns one-hot DataFrame."""
        buckets = pd.cut(port_series.clip(0, 65535), bins=[-1, 1023, 49151, 65535],
                         labels=PORT_BUCKETS).astype(str)
        return self._one_hot(pd.Series(buckets, index=port_series.index), PORT_BUCKETS, prefix)

    def _encode_protocol(self, proto_series):
        """One-hot encode PROTOCOL: TCP(6), UDP(17), ICMP(1), Other."""
        mapped = proto_series.map(PROTOCOLS).fillna('OTHER')
        return self._one_hot(mapped, list(PROTOCOLS.values()) + ['OTHER'], 'PROTO')

    def _encode_l7_proto(self, l7_series):
        """Bucket L7_PROTO into the values chosen by the feature analysis + Other."""
        names = [str(int(v)) if float(v).is_integer() else str(v) for v in self.l7_keep]
        lookup = dict(zip(self.l7_keep, names))
        mapped = l7_series.astype(float).map(lookup).fillna('OTHER')
        return self._one_hot(mapped, names + ['OTHER'], 'L7')

    def _apply_log_transform(self, X_df):
        """Apply log1p to skewed features."""
        X = X_df.copy()
        for col in self.log_features:
            if col in X.columns:
                X[col] = np.log1p(np.clip(X[col].values, 0, None))
        return X

    def _graph_features(self, df):
        """Raw columns for IP-graph node features (bytes/packets log-scaled)."""
        G = df[GRAPH_COLS].to_numpy(dtype=np.float64)
        G[:, 0] = np.log1p(np.clip(G[:, 0], 0, None))  # IN_BYTES
        G[:, 1] = np.log1p(np.clip(G[:, 1], 0, None))  # IN_PKTS
        return G.astype(np.float32)

    def _print_class_counts(self, df_train, df_test, col):
        """Show per-class counts so a missing class is visible immediately."""
        counts = pd.DataFrame({
            'train': df_train[col].value_counts(),
            'test': df_test[col].value_counts()
        }).fillna(0).astype(int)
        print("  Class counts after split:")
        print(counts.to_string())
