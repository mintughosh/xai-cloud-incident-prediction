"""SHAP stability across BGL session-window sizes.

08_ablation_window_size.py trained Random Forest models for window_size in
{10, 20, 40, 100} but only evaluated them in-memory — none of the {10, 20,
40} models were persisted to disk, only the existing window_size=100
baseline (results/models/BGL_RandomForest.pkl). This script retrains those
three with identical hyperparameters (n_estimators=100,
class_weight='balanced', random_state=42 — matching build_models() in
08_ablation_window_size.py) and saves them, so the SHAP analysis here is
tied to a concrete, reloadable model artifact per window size, not a
transient in-memory object.

Because the event-template vocabulary (E1..E987) is identical across every
window size (only the session aggregation changes, not the Drain3
parsing), the top-10 feature sets from each window size are directly
comparable via Jaccard similarity without any relabeling.
"""

import os
import pickle
import sys
import traceback

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier

from utils_bgl_windows import (
    build_windows_sparse,
    chronological_split_sparse,
    load_structured_log_and_event_ids,
)
from utils_data import get_feature_columns
from utils_explain import extract_positive_class_shap, stratified_sample

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

REBUILD_WINDOW_SIZES = [10, 20, 40]
EXISTING_WINDOW_SIZE = 100
SAMPLE_SIZE = 200
BACKGROUND_SIZE = 100
TOP_N = 10
RANDOM_STATE = 42


def get_model_and_test_sample(window_size, df, sorted_event_ids):
    """Train (window sizes 10/20/40) or load (window size 100) the RF model and a test sample.

    Returns (model, X_sample, feature_cols, background) — X_sample and
    background are dense DataFrames (SHAP TreeExplainer and this project's
    stratified_sample helper both expect array-like/DataFrame input, and a
    100-200 row sample is small enough that densifying it, unlike the full
    sparse training matrices, is fine). background is a 100-row sample of
    the train split, required because class_weight='balanced' RandomForest
    needs feature_perturbation='interventional' to produce correct SHAP
    values (see compute_top10()'s docstring).
    """
    if window_size == EXISTING_WINDOW_SIZE:
        print(f"\n=== window_size={window_size} (existing model, not retrained) ===")
        model_path = os.path.join(MODELS_DIR, "BGL_RandomForest.pkl")
        print(f"  Loading model: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "BGL", "splits", "train.csv"))
        test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "BGL", "splits", "test.csv"))
        feature_cols = get_feature_columns(test_df)
        X_test, y_test = test_df[feature_cols], test_df["Label"]
        background = train_df[feature_cols].sample(BACKGROUND_SIZE, random_state=RANDOM_STATE)
    else:
        print(f"\n=== window_size={window_size} (retrained to match 08_ablation_window_size.py) ===")
        X, y = build_windows_sparse(df, window_size, sorted_event_ids)
        X_train, y_train, _, _, X_test_sparse, y_test = chronological_split_sparse(X, y)
        print(f"  train={X_train.shape[0]:,} ({int(y_train.sum())} anomalous), "
              f"test={X_test_sparse.shape[0]:,} ({int(y_test.sum())} anomalous)")

        model = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        print("  Training RandomForest...")
        model.fit(X_train, y_train)

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"BGL_RandomForest_w{window_size}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved model: {model_path}")

        feature_cols = sorted_event_ids
        X_test = pd.DataFrame(X_test_sparse.toarray(), columns=feature_cols)
        y_test = pd.Series(y_test)
        background_idx = np.random.RandomState(RANDOM_STATE).choice(
            X_train.shape[0], size=min(BACKGROUND_SIZE, X_train.shape[0]), replace=False
        )
        background = pd.DataFrame(X_train[background_idx].toarray(), columns=feature_cols)

    X_sample, y_sample = stratified_sample(X_test, y_test, SAMPLE_SIZE, RANDOM_STATE)
    print(f"  Sampled {len(X_sample)} test instances ({int(y_sample.sum())} anomalous) for SHAP")
    return model, X_sample, feature_cols, background


def compute_top10(model, X_sample, feature_cols, background):
    """Run SHAP TreeExplainer and return the top-10 feature names by mean |SHAP value|.

    class_weight='balanced' RandomForest corrupts TreeExplainer's default
    'tree_path_dependent' algorithm outright (confirmed: SHAP magnitudes up
    to ~1e11-1e27 for a model whose predict_proba is bounded in [0, 1]), not
    just a failed additivity check. feature_perturbation='interventional'
    with an explicit background sample is the fix — see 04_explain_rq2.py's
    run_shap() docstring for the full diagnosis.
    """
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    explanation = explainer(X_sample)
    shap_values = extract_positive_class_shap(explanation)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)
    return list(ranking.head(TOP_N).index)


def jaccard(set_a, set_b):
    """Jaccard similarity: |intersection| / |union| of two feature sets."""
    a, b = set(set_a), set(set_b)
    return len(a & b) / len(a | b)


def main():
    """Compute top-10 SHAP feature sets for BGL at 4 window sizes and their pairwise Jaccard similarity."""
    print("STEP: SHAP stability across BGL window sizes")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        print("Loading structured log and event vocabulary...")
        df, sorted_event_ids = load_structured_log_and_event_ids()
        print(f"  {len(df):,} lines loaded, {len(sorted_event_ids)} event templates")

        window_sizes = REBUILD_WINDOW_SIZES + [EXISTING_WINDOW_SIZE]
        top10_by_window = {}

        for ws in window_sizes:
            model, X_sample, feature_cols, background = get_model_and_test_sample(ws, df, sorted_event_ids)
            top10 = compute_top10(model, X_sample, feature_cols, background)
            top10_by_window[ws] = top10
            print(f"  Top-{TOP_N} SHAP features: {top10}")

        print("\n=== Top-10 SHAP features side by side ===")
        side_by_side = pd.DataFrame({f"w{ws}": top10_by_window[ws] for ws in window_sizes})
        print(side_by_side.to_string(index=False))

        print("\nComputing pairwise Jaccard similarity...")
        pairs = [(10, 20), (10, 40), (10, 100), (20, 40), (20, 100), (40, 100)]
        rows = []
        for a, b in pairs:
            top10_a, top10_b = top10_by_window[a], top10_by_window[b]
            common = sorted(set(top10_a) & set(top10_b), key=lambda e: int(e[1:]))
            sim = jaccard(top10_a, top10_b)
            print(f"  ({a} vs {b}): jaccard={sim:.4f}, common={common}")
            rows.append({
                "window_a": a,
                "window_b": b,
                "jaccard_similarity": sim,
                "features_in_common": ";".join(common),
                "top10_a": ";".join(top10_a),
                "top10_b": ";".join(top10_b),
            })

        results_df = pd.DataFrame(rows)
        out_path = os.path.join(RESULTS_DIR, "shap_stability_window.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

        print("\nSTEP COMPLETE: SHAP stability analysis finished successfully")

    except Exception:
        print("\nERROR during SHAP stability analysis:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
