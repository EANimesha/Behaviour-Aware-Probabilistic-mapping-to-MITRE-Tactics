"""
Preprocessing Module for NF-BoT-IoT-v2
========================================

Based on EDA notebook findings:
  1. Clean: remove inf/NaN, handle missing values
  2. Drop: correlated, VIF, near-zero variance, and useless features
  3. Engineer: categorical features (ports, protocols, ICMP) into
     meaningful bucketed + one-hot representations
  4. Transform: log1p on skewed numeric features
  5. Scale: StandardScaler on numeric features only
  6. Split: stratified train/test by Attack type
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Column classifications ──

IDENTIFIER_COLS = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
LABEL_COLS = ['Label', 'Attack']

# Categorical features that need engineering, NOT numeric scaling
CATEGORICAL_COLS = [
    'PROTOCOL',              # 6=TCP, 17=UDP, 1=ICMP
    'L7_PROTO',              # application protocol codes
    'TCP_FLAGS',             # bitmask — treat as numeric
    'CLIENT_TCP_FLAGS',      # bitmask — treat as numeric
    'SERVER_TCP_FLAGS',      # bitmask — treat as numeric
    'ICMP_TYPE',             # ICMP message type codes
    'ICMP_IPV4_TYPE',        # ICMP subtype codes
    'DNS_QUERY_TYPE',        # 1=A, 2=NS, 5=CNAME, 28=AAAA
    'FTP_COMMAND_RET_CODE',  # 200, 331, 530, 550
    'L4_SRC_PORT',           # port numbers
    'L4_DST_PORT',           # port numbers
]

# ── Features to DROP (from EDA notebook) ──

# Correlation > 0.94 with another feature
DROP_CORRELATION = [
    'MAX_TTL',                     # correlated with MIN_TTL
    'MAX_IP_PKT_LEN',             # correlated with LONGEST_FLOW_PKT
    'RETRANSMITTED_IN_PKTS',      # correlated with RETRANSMITTED_IN_BYTES
    'NUM_PKTS_1024_TO_1514_BYTES', # correlated with OUT_BYTES
]

# High VIF (multicollinearity)
DROP_VIF = [
    'IN_BYTES',                    # VIF too high
    'NUM_PKTS_512_TO_1024_BYTES',  # VIF too high
    'NUM_PKTS_256_TO_512_BYTES',   # VIF too high
]

# Near-zero variance (>90% zeros — no discriminative power)
DROP_ZERO_VARIANCE = [
    'RETRANSMITTED_IN_BYTES',   # 91.7% zeros
    'RETRANSMITTED_OUT_BYTES',  # 97.6% zeros
    'RETRANSMITTED_OUT_PKTS',   # 97.6% zeros
    'DNS_TTL_ANSWER',           # 89.9% zeros (also near-zero)
    'DNS_QUERY_TYPE',           # mostly zeros, categorical
    'ICMP_IPV4_TYPE',           # 89.9% zeros
]

# No predictive value
DROP_USELESS = [
    'DNS_QUERY_ID',             # random transaction ID
    'FTP_COMMAND_RET_CODE',     # mostly zeros, rarely used
]

# All features to drop (union, deduplicated)
ALL_DROPS = list(set(
    DROP_CORRELATION + DROP_VIF + DROP_ZERO_VARIANCE + DROP_USELESS
))

# ── Final numeric features (after drops) to log-transform ──
LOG_FEATURES = [
    'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
]

# Final numeric features to scale (after EDA cleanup)
NUMERIC_FEATURES = [
    'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'MIN_TTL',
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
]

# TCP flag bitmask features — keep as numeric (ordinal intensity)
TCP_FLAG_FEATURES = ['TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS']


class Preprocessor:
    """Full preprocessing pipeline based on EDA notebook findings.

    Handles:
      - Cleaning (inf/NaN)
      - Feature drops (correlation, VIF, zero-variance, useless)
      - Categorical feature engineering (ports, protocols, ICMP)
      - Log-transform on skewed numerics
      - StandardScaler on numerics + TCP flags
      - Stratified train/test split
    """

    def __init__(self, clip_extremes=False):
        """
        Args:
            clip_extremes: if True, clip to 0.1th/99.9th percentile.
                           Needed for CIC-IDS2018/UNSW (overflow prevention).
                           Not for BoT-IoT (extreme values are the signal).
        """
        self.scaler = StandardScaler()
        self.fitted = False
        self.final_feature_names = []
        self._clip_extremes = clip_extremes

    def preprocess_dataset(self, df, test_size=0.2, stratify_col='Attack',
                           random_state=42):
        """Full pipeline: clean -> engineer -> transform -> split -> scale.

        Args:
            df: raw DataFrame from loader
            test_size: fraction for test split
            stratify_col: column to stratify on
            random_state: reproducibility seed

        Returns:
            dict with X_train, X_test, y_train, y_test, src/dst IPs,
            feature_names, scaler, etc.
        """
        df = self._clean(df)

        # Preserve IPs and labels before feature engineering
        src_ips = df['IPV4_SRC_ADDR'].values
        dst_ips = df['IPV4_DST_ADDR'].values
        y_attack = df['Attack'].values if 'Attack' in df.columns else None
        y_binary = df['Label'].values if 'Label' in df.columns else None

        # ── Feature Engineering ──
        X = self._engineer_features(df)

        self.final_feature_names = list(X.columns)
        logger.info(f"Final feature count: {len(self.final_feature_names)}")
        logger.info(f"Features: {self.final_feature_names}")

        # ── Stratified Split ──
        indices = np.arange(len(X))
        stratify = y_attack if stratify_col == 'Attack' else y_binary

        idx_train, idx_test = train_test_split(
            indices, test_size=test_size,
            stratify=stratify, random_state=random_state
        )

        X_train_df = X.iloc[idx_train].copy()
        X_test_df = X.iloc[idx_test].copy()

        # ── Log Transform (before scaling) ──
        X_train_df = self._apply_log_transform(X_train_df)
        X_test_df = self._apply_log_transform(X_test_df)

        # ── Scale: fit on train, transform both ──
        X_train_scaled = self._fit_transform_scale(X_train_df)
        X_test_scaled = self._transform_scale(X_test_df)

        logger.info(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_attack[idx_train] if y_attack is not None else None,
            'y_test': y_attack[idx_test] if y_attack is not None else None,
            'y_binary_train': y_binary[idx_train] if y_binary is not None else None,
            'y_binary_test': y_binary[idx_test] if y_binary is not None else None,
            'src_train': src_ips[idx_train],
            'src_test': src_ips[idx_test],
            'dst_train': dst_ips[idx_train],
            'dst_test': dst_ips[idx_test],
            'feature_names': self.final_feature_names,
            'scaler': self.scaler,
            'train_indices': idx_train,
            'test_indices': idx_test,
        }

    def transform_new(self, df):
        """Transform new data (inference/streaming) using fitted scaler.

        Args:
            df: raw DataFrame (must contain same columns as training data)

        Returns:
            X_scaled: (N, feature_dim) numpy array
            src_ips, dst_ips: IP arrays
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call preprocess_dataset first.")

        df = self._clean(df)
        src_ips = df['IPV4_SRC_ADDR'].values
        dst_ips = df['IPV4_DST_ADDR'].values

        X = self._engineer_features(df)

        # Ensure same columns in same order
        for col in self.final_feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.final_feature_names]

        X = self._apply_log_transform(X)
        X_scaled = self._transform_scale(X)

        return X_scaled, src_ips, dst_ips

    # ────────────────────────────────────────────────────────
    # Internal methods
    # ────────────────────────────────────────────────────────

    def _clean(self, df):
        """Handle missing values, infinities, duplicates."""
        n_before = len(df)

        # Replace inf with NaN
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

        Based on EDA notebook cells 28-32:
          - Ports: bucketed into well_known/registered/ephemeral
          - PROTOCOL: one-hot (TCP, UDP, ICMP)
          - L7_PROTO: bucketed (0, 7, 118, Other)
          - ICMP_TYPE: binary (0 vs non-0)
          - TCP flags: kept as numeric
        """
        parts = []

        # ── 1. Numeric features ──
        available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
        parts.append(df[available_numeric].copy())

        # ── 2. TCP flags (keep as numeric — ordinal bitmask values) ──
        available_tcp = [c for c in TCP_FLAG_FEATURES if c in df.columns]
        parts.append(df[available_tcp].copy())

        # ── 3. Port bucketing (src & dst) ──
        for port_col, prefix in [('L4_SRC_PORT', 'SRC_PORT'),
                                  ('L4_DST_PORT', 'DST_PORT')]:
            if port_col in df.columns:
                port_df = self._bucket_ports(df[port_col], prefix)
                parts.append(port_df)

        # ── 4. PROTOCOL one-hot ──
        if 'PROTOCOL' in df.columns:
            proto_df = self._encode_protocol(df['PROTOCOL'])
            parts.append(proto_df)

        # ── 5. L7_PROTO bucketing ──
        if 'L7_PROTO' in df.columns:
            l7_df = self._encode_l7_proto(df['L7_PROTO'])
            parts.append(l7_df)

        # ── 6. ICMP_TYPE binary ──
        if 'ICMP_TYPE' in df.columns:
            icmp_binary = (df['ICMP_TYPE'] != 0).astype(np.float32)
            parts.append(pd.DataFrame(
                {'ICMP_ACTIVE': icmp_binary.values}, index=df.index
            ))

        # Combine all parts
        X = pd.concat(parts, axis=1)

        # Safety check: ensure no dropped features leaked through
        leaked = [c for c in X.columns if c in ALL_DROPS]
        if leaked:
            logger.warning(f"Dropping leaked features: {leaked}")
            X = X.drop(columns=leaked)

        # Ensure float32
        X = X.astype(np.float32)

        return X

    def _bucket_ports(self, port_series, prefix):
        """Bucket ports into: well_known (0-1023), registered (1024-49151),
        ephemeral (49152-65535). Returns one-hot DataFrame."""
        buckets = pd.cut(
            port_series.clip(0, 65535),
            bins=[-1, 1023, 49151, 65535],
            labels=[f'{prefix}_WELL_KNOWN', f'{prefix}_REGISTERED',
                    f'{prefix}_EPHEMERAL']
        )
        return pd.get_dummies(buckets, dtype=np.float32)

    def _encode_protocol(self, proto_series):
        """One-hot encode PROTOCOL: TCP(6), UDP(17), ICMP(1), Other."""
        proto_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
        mapped = proto_series.map(proto_map).fillna('OTHER')
        dummies = pd.get_dummies(mapped, prefix='PROTO', dtype=np.float32)
        return dummies

    def _encode_l7_proto(self, l7_series):
        """Bucket L7_PROTO: 0 (unknown), 7 (HTTP), 118 (specific), Other."""
        l7_mapped = l7_series.copy()
        keep_values = {0.0, 7.0, 118.0}
        l7_mapped = l7_mapped.apply(
            lambda x: str(int(x)) if x in keep_values else 'OTHER'
        )
        dummies = pd.get_dummies(l7_mapped, prefix='L7', dtype=np.float32)
        return dummies

    def _apply_log_transform(self, X_df):
        """Apply log1p to skewed features."""
        X = X_df.copy()
        for col in LOG_FEATURES:
            if col in X.columns:
                X[col] = np.log1p(np.clip(X[col].values, 0, None))
        return X

    # def _fit_transform_scale(self, X_df):
    #     """Fit scaler on training data and transform.
    #     Handles inf/nan and extreme values from large datasets.
    #     """
    #     X = X_df.values.astype(np.float64)  # float64 to avoid overflow
    #
    #     # Replace inf with nan
    #     X = np.where(np.isinf(X), np.nan, X)
    #
    #     # Replace nan with column median
    #     for col in range(X.shape[1]):
    #         mask = np.isnan(X[:, col])
    #         if mask.any():
    #             median = np.nanmedian(X[:, col])
    #             X[mask, col] = median if np.isfinite(median) else 0.0
    #
    #     # Clip extreme values (per column, 0.1th and 99.9th percentile)
    #     for col in range(X.shape[1]):
    #         lo = np.percentile(X[:, col], 0.1)
    #         hi = np.percentile(X[:, col], 99.9)
    #         X[:, col] = np.clip(X[:, col], lo, hi)
    #
    #     self.scaler.fit(X)
    #     self.fitted = True
    #     return self.scaler.transform(X).astype(np.float32)
    #
    # def _transform_scale(self, X_df):
    #     """Transform using already-fitted scaler."""
    #     X = X_df.values.astype(np.float64)
    #     X = np.where(np.isinf(X), np.nan, X)
    #     for col in range(X.shape[1]):
    #         mask = np.isnan(X[:, col])
    #         if mask.any():
    #             median = np.nanmedian(X[:, col])
    #             X[mask, col] = median if np.isfinite(median) else 0.0
    #     return self.scaler.transform(X).astype(np.float32)
    def _fit_transform_scale(self, X_df):
        """Fit scaler on training data and transform."""
        self.scaler.fit(X_df.values)
        self.fitted = True
        return self.scaler.transform(X_df.values).astype(np.float32)

    def _transform_scale(self, X_df):
        """Transform using already-fitted scaler."""
        return self.scaler.transform(X_df.values).astype(np.float32)
    # def _fit_transform_scale(self, X_df):
    #     """Fit scaler on training data and transform.
    #     Handles inf/nan for all datasets. Clipping only when enabled.
    #     """
    #     X = X_df.values.astype(np.float64)  # float64 to avoid overflow
    #
    #     # Replace inf with nan (needed for ALL datasets)
    #     X = np.where(np.isinf(X), np.nan, X)
    #
    #     # Replace nan with column median (needed for ALL datasets)
    #     for col in range(X.shape[1]):
    #         mask = np.isnan(X[:, col])
    #         if mask.any():
    #             median = np.nanmedian(X[:, col])
    #             X[mask, col] = median if np.isfinite(median) else 0.0
    #
    #     # Clip extreme values ONLY if enabled
    #     # BoT-IoT: no clipping needed (extreme values ARE the signal)
    #     # CIC-IDS2018/UNSW: clipping prevents overflow in StandardScaler
    #     if self._clip_extremes:
    #         for col in range(X.shape[1]):
    #             lo = np.percentile(X[:, col], 0.1)
    #             hi = np.percentile(X[:, col], 99.9)
    #             X[:, col] = np.clip(X[:, col], lo, hi)
    #
    #     self.scaler.fit(X)
    #     self.fitted = True
    #     return self.scaler.transform(X).astype(np.float32)
    #
    # def _transform_scale(self, X_df):
    #     """Transform using already-fitted scaler."""
    #     X = X_df.values.astype(np.float64)
    #     X = np.where(np.isinf(X), np.nan, X)
    #     for col in range(X.shape[1]):
    #         mask = np.isnan(X[:, col])
    #         if mask.any():
    #             median = np.nanmedian(X[:, col])
    #             X[mask, col] = median if np.isfinite(median) else 0.0
    #     return self.scaler.transform(X).astype(np.float32)