"""Shared BGL session-windowing logic, used by both the window-size ablation
(08_ablation_window_size.py) and the SHAP stability analysis
(09_shap_stability_window.py) so the two never drift apart on how a window
size maps to a feature matrix.
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
STRUCTURED_PATH = os.path.join(PROCESSED_DIR, "BGL", "BGL_structured.csv")
TEMPLATES_PATH = os.path.join(PROCESSED_DIR, "BGL", "BGL_templates.csv")

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def load_structured_log_and_event_ids():
    """Load BGL's per-line Drain3 output and its sorted event-id vocabulary.

    Returns (df, sorted_event_ids). The event vocabulary is fixed regardless
    of window size — only the session aggregation depends on window size.
    """
    df = pd.read_csv(STRUCTURED_PATH)
    templates_df = pd.read_csv(TEMPLATES_PATH)
    sorted_event_ids = sorted(templates_df["EventId"].tolist(), key=lambda e: int(e[1:]))
    return df, sorted_event_ids


def build_windows_sparse(df, window_size, sorted_event_ids):
    """Aggregate per-line (Timestamp, Label, EventId) rows into fixed-size window sessions.

    Returns a sparse (n_windows x n_events) count matrix (float32, since
    LightGBM's sparse CSR ingestion requires float data) and a per-window
    binary label array. Built via a sparse COO matrix rather than a dense
    pivot/crosstab so window_size=10's ~475,000 windows x 987 event columns
    doesn't require multiple GB of dense memory.
    """
    window_id = np.arange(len(df)) // window_size
    n_windows = int(window_id[-1]) + 1

    event_id_to_col = {eid: i for i, eid in enumerate(sorted_event_ids)}
    event_col_idx = df["EventId"].map(event_id_to_col).values
    data = np.ones(len(df), dtype=np.float32)

    counts = sp.coo_matrix(
        (data, (window_id, event_col_idx)), shape=(n_windows, len(sorted_event_ids))
    ).tocsr()
    counts.sum_duplicates()

    labels = df.groupby(window_id)["Label"].max().values
    return counts, labels


def chronological_split_sparse(X, y):
    """Split a sparse (X, y) pair into train/val/test by row position (70/15/15), no shuffling."""
    n = X.shape[0]
    train_end = int(n * TRAIN_FRAC)
    test_start = int(n * (TRAIN_FRAC + VAL_FRAC))
    return (
        X[:train_end], y[:train_end],
        X[train_end:test_start], y[train_end:test_start],
        X[test_start:], y[test_start:],
    )
