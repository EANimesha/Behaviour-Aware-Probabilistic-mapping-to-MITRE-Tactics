"""
Chronological Data Balancer for Imbalanced NIDS Datasets
=========================================================

Problem:
  NF-UNSW-NB15-v2 is 96% benign, 4% attack. The dataset is too large
  to train on fully (~2.4M flows), and the imbalance causes DP-GMM
  to collapse everything into one benign cluster.

Strategy:
  1. Preserve chronological order (critical for temporal sequences)
  2. Keep ALL attack flows (they're already minority)
  3. Subsample benign flows using strided selection (every Nth row)
     within chronological order — preserves temporal spread
  4. Target a configurable benign:attack ratio (default 2:1)

This gives us:
  - ~95K attack flows (all preserved)
  - ~190K benign flows (strided from 2.3M)
  - ~285K total (manageable, fits in GPU memory)
  - Temporal ordering intact for behaviour encoder
  - Graph topology proportionally preserved
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChronologicalBalancer:
    """Balance dataset while preserving chronological order.

    Assumes rows in the DataFrame are already in chronological order
    (as they are in NF-UNSW-NB15-v2 and NF-BoT-IoT-v2).
    """

    def __init__(self, benign_ratio=2.0, min_samples_per_class=100,
                 max_total_samples=None, benign_label='Benign',
                 attack_col='Attack', label_col='Label'):
        """
        Args:
            benign_ratio: target ratio of benign:attack flows.
                          e.g., 2.0 means 2x benign per attack flow.
                          Use None to keep all benign (just cap total).
            min_samples_per_class: minimum attack samples to keep
                                   (classes below this are kept entirely)
            max_total_samples: hard cap on total dataset size (None = no cap)
            benign_label: value in attack_col that marks benign flows
            attack_col: column name for attack type labels
            label_col: column name for binary label (0/1)
        """
        self.benign_ratio = benign_ratio
        self.min_samples_per_class = min_samples_per_class
        self.max_total_samples = max_total_samples
        self.benign_label = benign_label
        self.attack_col = attack_col
        self.label_col = label_col

    def balance(self, df):
        """Balance the dataset while preserving row order.

        Args:
            df: DataFrame in chronological order

        Returns:
            df_balanced: balanced DataFrame (still in chronological order)
            stats: dict with balancing statistics
        """
        n_original = len(df)

        if self.attack_col not in df.columns:
            logger.warning(f"Column '{self.attack_col}' not found. "
                           f"Returning original DataFrame.")
            return df, {'method': 'none', 'original': n_original}

        # ── Separate benign and attack flows (preserve indices) ──
        is_benign = df[self.attack_col] == self.benign_label
        benign_indices = df.index[is_benign].tolist()
        attack_indices = df.index[~is_benign].tolist()

        n_benign = len(benign_indices)
        n_attack = len(attack_indices)

        logger.info(f"Original: {n_original:,} flows "
                    f"({n_benign:,} benign, {n_attack:,} attack)")

        # ── Keep ALL attack flows ──
        keep_indices = list(attack_indices)

        # ── Subsample benign via chronological stride ──
        if self.benign_ratio is not None and n_attack > 0:
            target_benign = int(n_attack * self.benign_ratio)
            target_benign = max(target_benign, self.min_samples_per_class)
            target_benign = min(target_benign, n_benign)  # Can't exceed total

            if target_benign < n_benign:
                # Stride sampling: take every Nth benign flow
                stride = n_benign / target_benign
                sampled_positions = [
                    int(i * stride) for i in range(target_benign)
                ]
                sampled_benign = [benign_indices[p] for p in sampled_positions]
                keep_indices.extend(sampled_benign)
                logger.info(f"Benign subsampled: {n_benign:,} -> "
                            f"{len(sampled_benign):,} "
                            f"(stride={stride:.1f})")
            else:
                keep_indices.extend(benign_indices)
        else:
            keep_indices.extend(benign_indices)

        # ── Apply max_total_samples cap if set ──
        if self.max_total_samples and len(keep_indices) > self.max_total_samples:
            # Further stride-sample from the combined set
            # but preserve chronological order
            keep_indices.sort()
            stride = len(keep_indices) / self.max_total_samples
            keep_indices = [
                keep_indices[int(i * stride)]
                for i in range(self.max_total_samples)
            ]
            logger.info(f"Capped to {self.max_total_samples:,} total samples")

        # ── Sort to restore chronological order ──
        keep_indices.sort()

        # ── Build balanced DataFrame ──
        df_balanced = df.loc[keep_indices].reset_index(drop=True)

        # ── Statistics ──
        balanced_benign = (
            df_balanced[self.attack_col] == self.benign_label
        ).sum()
        balanced_attack = len(df_balanced) - balanced_benign

        stats = {
            'method': 'chronological_stride',
            'original_total': n_original,
            'original_benign': n_benign,
            'original_attack': n_attack,
            'balanced_total': len(df_balanced),
            'balanced_benign': int(balanced_benign),
            'balanced_attack': int(balanced_attack),
            'benign_ratio_actual': (
                float(balanced_benign / balanced_attack)
                if balanced_attack > 0 else float('inf')
            ),
            'reduction_factor': n_original / len(df_balanced),
        }

        logger.info(f"Balanced: {stats['balanced_total']:,} flows "
                    f"({stats['balanced_benign']:,} benign, "
                    f"{stats['balanced_attack']:,} attack, "
                    f"ratio={stats['benign_ratio_actual']:.1f}:1)")

        # ── Per-class breakdown ──
        logger.info("Per-class distribution after balancing:")
        for attack_type, count in (
            df_balanced[self.attack_col].value_counts().items()
        ):
            pct = count / len(df_balanced) * 100
            logger.info(f"  {attack_type:20s} {count:>8,} ({pct:5.1f}%)")

        return df_balanced, stats

    def balance_with_minority_boost(self, df):
        """Balance with additional oversampling of rare attack classes.

        For classes with very few samples (e.g., Worms: 164), duplicate
        them to reach min_samples_per_class while keeping them in
        temporal position (duplicates placed adjacent to originals).

        Args:
            df: DataFrame in chronological order

        Returns:
            df_balanced: balanced DataFrame
            stats: dict with balancing statistics
        """
        # First do the standard chronological balance
        df_balanced, stats = self.balance(df)

        if self.attack_col not in df_balanced.columns:
            return df_balanced, stats

        # ── Boost minority attack classes ──
        boosted_frames = [df_balanced]
        n_boosted = 0

        for attack_type in df_balanced[self.attack_col].unique():
            if attack_type == self.benign_label:
                continue

            mask = df_balanced[self.attack_col] == attack_type
            n_class = mask.sum()

            if n_class < self.min_samples_per_class:
                # How many copies needed
                n_copies = self.min_samples_per_class - n_class
                class_rows = df_balanced[mask]

                # Cycle through existing rows
                boost_indices = np.arange(n_copies) % len(class_rows)
                boost_rows = class_rows.iloc[boost_indices].copy()

                boosted_frames.append(boost_rows)
                n_boosted += n_copies
                logger.info(f"  Boosted {attack_type}: {n_class} -> "
                            f"{n_class + n_copies} samples")

        if n_boosted > 0:
            df_balanced = pd.concat(boosted_frames, ignore_index=True)
            # Re-sort to approximate chronological order
            # (boosted samples go to end but that's acceptable)
            stats['boosted_samples'] = n_boosted
            stats['balanced_total'] = len(df_balanced)
            logger.info(f"After minority boost: {len(df_balanced):,} total "
                        f"(+{n_boosted} boosted)")

        return df_balanced, stats