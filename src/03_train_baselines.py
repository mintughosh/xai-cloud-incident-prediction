"""Train and evaluate Logistic Regression, Random Forest, and LightGBM for RQ1.

Evaluation strategy per dataset:
  - HDFS, BGL: train on the chronological train split, evaluate on val and
    test splits separately (see 02_build_splits.py for the 70/15/15 split).
  - OpenStack: chronological split is structurally impossible here (see
    02_build_splits.py docstring — normal and abnormal logs come from
    disjoint time periods), so this is a documented methodological
    exception: stratified k-fold CV over the full matrix instead. With only
    4 anomalous instances total, k is capped at 4 (sklearn's
    StratifiedKFold requires n_splits <= the smallest class's member
    count), not the more typical k=5, so every fold's test set contains at
    least one anomaly.

All models use imbalance-aware settings: class_weight='balanced' for
Logistic Regression / Random Forest, scale_pos_weight for LightGBM. Never
raw accuracy — F1, Precision, Recall, and AUC-ROC only.
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils_data import get_feature_columns

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "model_results.csv")

CHRONOLOGICAL_DATASETS = ["HDFS", "BGL"]
KFOLD_DATASETS = ["OpenStack"]
RANDOM_STATE = 42


def build_models(scale_pos_weight):
    """Return {model_name: estimator} with imbalance-aware settings applied.

    Logistic Regression is wrapped in a StandardScaler pipeline since raw
    event counts have very different scales across templates; tree-based
    models don't need scaling.
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, verbosity=-1,
        ),
    }


def evaluate(model, X, y):
    """Fit-independent evaluation: compute F1/Precision/Recall/AUC-ROC for predictions on (X, y)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "N": len(y),
        "N_Positive": int(y.sum()),
        "F1": f1_score(y, y_pred, zero_division=0),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "AUC_ROC": roc_auc_score(y, y_proba) if y.nunique() > 1 else float("nan"),
    }


def run_chronological_dataset(name):
    """Train on the train split, evaluate on val and test splits, for one chronological dataset."""
    print(f"\n=== {name} (chronological train/val/test) ===")
    splits_dir = os.path.join(PROCESSED_DIR, name, "splits")

    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))

    feature_cols = get_feature_columns(train_df)
    X_train, y_train = train_df[feature_cols], train_df["Label"]
    X_val, y_val = val_df[feature_cols], val_df["Label"]
    X_test, y_test = test_df[feature_cols], test_df["Label"]

    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  Features: {len(feature_cols)}, train scale_pos_weight={scale_pos_weight:.2f}")

    rows = []
    for model_name, model in build_models(scale_pos_weight).items():
        print(f"  Training {model_name}...")
        model.fit(X_train, y_train)

        for eval_name, X_eval, y_eval in [("val", X_val, y_val), ("test", X_test, y_test)]:
            metrics = evaluate(model, X_eval, y_eval)
            print(f"    [{eval_name}] F1={metrics['F1']:.4f} Precision={metrics['Precision']:.4f} "
                  f"Recall={metrics['Recall']:.4f} AUC-ROC={metrics['AUC_ROC']:.4f}")
            rows.append({"Dataset": name, "Model": model_name, "EvalSet": eval_name, **metrics})

    return rows


def run_kfold_dataset(name, k=4):
    """Run stratified k-fold CV over the full matrix, reporting mean +/- std per model.

    k defaults to 4 (not 5) because OpenStack has only 4 anomalous
    instances total and sklearn's StratifiedKFold requires n_splits to not
    exceed the smallest class's member count.
    """
    print(f"\n=== {name} (stratified {k}-fold CV, documented exception) ===")
    full_df = pd.read_csv(os.path.join(PROCESSED_DIR, name, "splits", "full.csv"))
    feature_cols = get_feature_columns(full_df)
    X, y = full_df[feature_cols], full_df["Label"]
    print(f"  Features: {len(feature_cols)}, total anomalies: {int(y.sum())} / {len(y)}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {model_name: [] for model_name in build_models(1.0)}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        for model_name, model in build_models(scale_pos_weight).items():
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            fold_metrics[model_name].append(metrics)
            print(f"  Fold {fold_idx}/{k} [{model_name}] F1={metrics['F1']:.4f} "
                  f"Precision={metrics['Precision']:.4f} Recall={metrics['Recall']:.4f} "
                  f"AUC-ROC={metrics['AUC_ROC']:.4f}")

    rows = []
    for model_name, folds in fold_metrics.items():
        folds_df = pd.DataFrame(folds)
        mean_row = folds_df.mean(numeric_only=True).to_dict()
        std_row = folds_df.std(numeric_only=True).to_dict()
        print(f"  {model_name} CV mean: F1={mean_row['F1']:.4f} Precision={mean_row['Precision']:.4f} "
              f"Recall={mean_row['Recall']:.4f} AUC-ROC={mean_row['AUC_ROC']:.4f}")
        rows.append({"Dataset": name, "Model": model_name, "EvalSet": "cv_mean", **mean_row})
        rows.append({"Dataset": name, "Model": model_name, "EvalSet": "cv_std", **std_row})

    return rows


def main():
    """Train LR/RF/LightGBM on HDFS and BGL (chronological) and OpenStack (k-fold), saving results."""
    print("STEP: Training baseline models for RQ1")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        all_rows = []
        for name in CHRONOLOGICAL_DATASETS:
            all_rows.extend(run_chronological_dataset(name))
        for name in KFOLD_DATASETS:
            all_rows.extend(run_kfold_dataset(name))

        results_df = pd.DataFrame(all_rows)
        results_df.to_csv(RESULTS_PATH, index=False)
        print(f"\nSaved all results: {RESULTS_PATH} ({len(results_df)} rows)")

        print("\nSTEP COMPLETE: Baseline model training finished successfully")

    except Exception:
        print("\nERROR during baseline model training:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
