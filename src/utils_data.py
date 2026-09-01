"""Shared helpers for loading feature matrices and building chronological splits.

Used by 02_build_splits.py (to create the splits) and 03_train_baselines.py
(to load them back for training). Centralising this logic keeps the label
normalisation and split boundaries identical across every script that
touches the three datasets.
"""

import pandas as pd

LABEL_MAP = {
    "normal": 0,
    "success": 0,
    "anomaly": 1,
    "fail": 1,
}


def normalize_label(series):
    """Map a Label column's mixed vocabulary (Normal/Success/Anomaly/Fail) to binary 0/1.

    Raises ValueError if any value isn't recognised, so silent mislabeling
    can't slip through.
    """
    lowered = series.astype(str).str.strip().str.lower()
    unknown = set(lowered.unique()) - set(LABEL_MAP.keys())
    if unknown:
        raise ValueError(f"Unrecognised label value(s): {unknown}")
    return lowered.map(LABEL_MAP).astype(int)


def get_feature_columns(df, prefix="E"):
    """Return the sorted list of event-count feature columns (E1, E2, ... En) in a dataframe."""
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


def chronological_split(df, train_frac=0.70, val_frac=0.15):
    """Split a chronologically-ordered dataframe into train/val/test by row position.

    Never shuffles: log data is time-ordered and evaluating on a random
    split would leak future event patterns into training, giving an
    optimistic bias that doesn't hold up on genuinely unseen future
    incidents.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def print_class_balance(name, df, label_col="Label"):
    """Print row count and positive-class rate for a split, for imbalance sanity checks."""
    n = len(df)
    pos = int(df[label_col].sum())
    rate = (pos / n * 100) if n else 0.0
    print(f"    {name}: {n:,} rows, {pos:,} anomalous ({rate:.2f}%)")


def load_split(dataset_dir, split_name):
    """Load a previously-built split CSV (train/val/test) for a dataset.

    `dataset_dir` is the dataset's processed-data folder, e.g.
    data/processed/HDFS/. Returns (X, y, feature_columns).
    """
    path = f"{dataset_dir}/splits/{split_name}.csv"
    df = pd.read_csv(path)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Label"]
    return X, y, feature_cols
