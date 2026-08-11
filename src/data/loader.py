import pandas as pd


def load_dataset(csv_path: str, sample_size: int = None,
                 random_state: int = 42) -> pd.DataFrame:
    """Load the NF-BoT-IoT-v2 CSV. Optionally sample for faster experimentation."""
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path) #, index_col=0
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    if sample_size is not None and sample_size < len(df):
        # Stratified sample by Attack to keep class proportions
        df = df.groupby('Attack', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // df['Attack'].nunique()),
                               random_state=random_state)
        ).reset_index(drop=True)
        print(f"  Sampled to {len(df):,} rows (stratified by Attack)")

    return df