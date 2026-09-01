"""Temporal drift analysis on BGL's chronological test set (RQ1 follow-up).

Splits the existing window_size=100 chronological test split
(data/processed/BGL/splits/test.csv) into 3 equal sequential chunks —
earliest, middle, latest — and evaluates whether F1/Precision/Recall/AUC-ROC
degrade as time progresses within the test period. This is a direct probe
for concept drift, complementing the val->test drop already observed in
03_train_baselines.py: that comparison showed *train-period* patterns don't
transfer to the test period; this one asks whether performance is even
stable *within* the test period itself.

RandomForest is loaded from the existing saved model
(results/models/BGL_RandomForest.pkl, from 04_explain_rq2.py). LightGBM has
no saved model artifact from 03_train_baselines.py (only its metrics were
recorded there), so it is retrained here with identical hyperparameters
(n_estimators=300, scale_pos_weight from the train split, random_state=42)
and persisted for reuse.
"""

import os
import pickle
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from utils_data import get_feature_columns

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
RANDOM_STATE = 42


def load_or_train_models(X_train, y_train):
    """Load the existing RandomForest and retrain+save LightGBM with 03_train_baselines.py's settings."""
    rf_path = os.path.join(MODELS_DIR, "BGL_RandomForest.pkl")
    print(f"  Loading existing model: {rf_path}")
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)

    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  Training LightGBM (n_estimators=300, scale_pos_weight={scale_pos_weight:.2f})...")
    lgbm_model = LGBMClassifier(
        n_estimators=300, scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, verbosity=-1
    )
    lgbm_model.fit(X_train, y_train)

    lgbm_path = os.path.join(MODELS_DIR, "BGL_LightGBM.pkl")
    with open(lgbm_path, "wb") as f:
        pickle.dump(lgbm_model, f)
    print(f"  Saved model: {lgbm_path}")

    return {"RandomForest": rf_model, "LightGBM": lgbm_model}


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


def make_plot(results_df):
    """Save F1 vs chunk for both models, with per-chunk anomaly count on a secondary y-axis."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    for model_name, group in results_df.groupby("model"):
        group = group.sort_values("chunk")
        ax1.plot(group["chunk"], group["f1"], marker="o", label=model_name)
    ax1.set_xlabel("Test-set chunk (1=earliest, 3=latest)")
    ax1.set_ylabel("F1")
    ax1.set_xticks([1, 2, 3])
    ax1.legend(loc="upper left")

    anomaly_counts = results_df.drop_duplicates("chunk").sort_values("chunk")
    ax2 = ax1.twinx()
    ax2.bar(anomaly_counts["chunk"], anomaly_counts["anomaly_count_in_chunk"],
            alpha=0.15, color="gray", width=0.5)
    ax2.set_ylabel("Anomaly count in chunk")

    plt.title("BGL Temporal Drift: F1 vs Test-Set Chunk")
    fig_path = os.path.join(FIGURES_DIR, "temporal_drift_bgl.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")


def main():
    """Split BGL's chronological test set into 3 sequential chunks and evaluate both models on each."""
    print("STEP: BGL temporal drift analysis")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        splits_dir = os.path.join(PROCESSED_DIR, "BGL", "splits")
        train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
        feature_cols = get_feature_columns(train_df)

        X_train, y_train = train_df[feature_cols], train_df["Label"]
        print(f"  Test set: {len(test_df):,} windows, {int(test_df['Label'].sum())} anomalous")

        models = load_or_train_models(X_train, y_train)

        n = len(test_df)
        chunk_size = n // 3
        chunks = {
            1: test_df.iloc[:chunk_size],
            2: test_df.iloc[chunk_size:2 * chunk_size],
            3: test_df.iloc[2 * chunk_size:],  # remainder folded into chunk 3
        }

        rows = []
        for chunk_id, chunk_df in chunks.items():
            X_chunk, y_chunk = chunk_df[feature_cols], chunk_df["Label"]
            anomaly_count = int(y_chunk.sum())
            print(f"\n  Chunk {chunk_id}: {len(chunk_df):,} windows, {anomaly_count} anomalous")
            for model_name, model in models.items():
                metrics = evaluate(model, X_chunk, y_chunk)
                print(f"    [{model_name}] F1={metrics['f1']:.4f} Precision={metrics['precision']:.4f} "
                      f"Recall={metrics['recall']:.4f} AUC-ROC={metrics['auc_roc']:.4f}")
                rows.append({
                    "model": model_name,
                    "chunk": chunk_id,
                    "chunk_size": len(chunk_df),
                    "anomaly_count_in_chunk": anomaly_count,
                    **metrics,
                })

        results_df = pd.DataFrame(rows)[
            ["model", "chunk", "chunk_size", "anomaly_count_in_chunk",
             "precision", "recall", "f1", "auc_roc"]
        ]
        out_path = os.path.join(RESULTS_DIR, "temporal_drift_bgl.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
        print(results_df.to_string(index=False))

        make_plot(results_df)

        print("\nSTEP COMPLETE: BGL temporal drift analysis finished successfully")

    except Exception:
        print("\nERROR during BGL temporal drift analysis:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
