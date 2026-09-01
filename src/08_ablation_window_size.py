"""Ablation study: effect of BGL's session-window size on RQ1 model performance.

BGL has no natural session key (unlike HDFS's per-block sessions), so
01_preprocess_bgl.py aggregates fixed-size windows of consecutive log lines
into sessions — window_size=100 was the arbitrary choice used everywhere
else in this project. This script tests whether that choice matters by
rebuilding the event-occurrence matrix at window_size=10, 20, and 40 and
comparing test-set performance against the existing window_size=100 build.

Windows are rebuilt directly from BGL_structured.csv (the per-line Drain3
template assignments already produced by 01_preprocess_bgl.py) rather than
re-running Drain3, since the template vocabulary doesn't depend on
windowing — only the session aggregation does. A sparse matrix is used
throughout (window counts are mostly zero across 987 possible event
templates) so window_size=10's ~475,000 windows stay memory-tractable.

Identical chronological 70/15/15 split logic to utils_data.chronological_split
is applied by row position for every window size, and window_size=100 reuses
the existing data/processed/BGL/splits/{train,test}.csv rather than being
rebuilt, per the ablation's own baseline.
"""

import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from utils_bgl_windows import build_windows_sparse, load_structured_log_and_event_ids
from utils_data import get_feature_columns

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

REBUILD_WINDOW_SIZES = [10, 20, 40]
EXISTING_WINDOW_SIZE = 100
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
RANDOM_STATE = 42


def build_models(scale_pos_weight):
    """Return {model_name: estimator} matching this ablation's exact spec."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, verbosity=-1,
        ),
    }


def evaluate(model, X, y):
    """Compute F1/Precision/Recall/AUC-ROC for a fitted model on (X, y)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else float("nan"),
    }


def run_rebuilt_window_size(window_size, df, sorted_event_ids):
    """Rebuild windows at the given size, split chronologically, train and evaluate both models."""
    print(f"\n=== window_size={window_size} (rebuilt from BGL_structured.csv) ===")
    X, y = build_windows_sparse(df, window_size, sorted_event_ids)
    n = X.shape[0]
    print(f"  {n:,} windows, {int(y.sum()):,} anomalous ({y.mean() * 100:.2f}%)")

    train_end = int(n * TRAIN_FRAC)
    test_start = int(n * (TRAIN_FRAC + VAL_FRAC))
    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[test_start:], y[test_start:]
    print(f"  train={X_train.shape[0]:,} ({int(y_train.sum())} anomalous), "
          f"test={X_test.shape[0]:,} ({int(y_test.sum())} anomalous)")

    return train_and_evaluate(window_size, X_train, y_train, X_test, y_test)


def run_existing_window_size():
    """Load the existing window_size=100 chronological splits rather than rebuilding them."""
    print(f"\n=== window_size={EXISTING_WINDOW_SIZE} (existing splits, not rebuilt) ===")
    splits_dir = os.path.join(PROCESSED_DIR, "BGL", "splits")
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    feature_cols = get_feature_columns(train_df)

    X_train, y_train = train_df[feature_cols].values, train_df["Label"].values
    X_test, y_test = test_df[feature_cols].values, test_df["Label"].values
    print(f"  train={X_train.shape[0]:,} ({int(y_train.sum())} anomalous), "
          f"test={X_test.shape[0]:,} ({int(y_test.sum())} anomalous)")

    return train_and_evaluate(EXISTING_WINDOW_SIZE, X_train, y_train, X_test, y_test)


def train_and_evaluate(window_size, X_train, y_train, X_test, y_test):
    """Train RandomForest and LightGBM on (X_train, y_train) and evaluate on (X_test, y_test)."""
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    rows = []
    for model_name, model in build_models(scale_pos_weight).items():
        print(f"  Training {model_name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        print(f"    [{model_name}] F1={metrics['f1']:.4f} Precision={metrics['precision']:.4f} "
              f"Recall={metrics['recall']:.4f} AUC-ROC={metrics['auc_roc']:.4f}")
        rows.append({"window_size": window_size, "model": model_name, **metrics})
    return rows


def make_plot(results_df):
    """Save a line plot of F1 vs window_size, one line per model."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, group in results_df.groupby("model"):
        group = group.sort_values("window_size")
        ax.plot(group["window_size"], group["f1"], marker="o", label=model_name)
    ax.set_xlabel("Window size (log lines per session)")
    ax.set_ylabel("Test F1")
    ax.set_title("BGL: Test F1 vs Session Window Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_path = os.path.join(FIGURES_DIR, "ablation_window_size.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")


def main():
    """Run the BGL window-size ablation across {10, 20, 40, 100} and save results + plot."""
    print("STEP: BGL window-size ablation study")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        print("Loading structured log and event vocabulary...")
        df, sorted_event_ids = load_structured_log_and_event_ids()
        print(f"  {len(df):,} lines loaded, {len(sorted_event_ids)} event templates "
              f"(fixed vocabulary, independent of window size)")

        all_rows = []
        for ws in REBUILD_WINDOW_SIZES:
            all_rows.extend(run_rebuilt_window_size(ws, df, sorted_event_ids))
        all_rows.extend(run_existing_window_size())

        results_df = pd.DataFrame(all_rows)[
            ["window_size", "model", "precision", "recall", "f1", "auc_roc"]
        ].sort_values(["window_size", "model"])
        out_path = os.path.join(RESULTS_DIR, "ablation_window_size.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
        print(results_df.to_string(index=False))

        make_plot(results_df)

        print("\nSTEP COMPLETE: BGL window-size ablation finished successfully")

    except Exception:
        print("\nERROR during BGL window-size ablation:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
