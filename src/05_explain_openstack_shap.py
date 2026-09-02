"""RQ3-only SHAP analysis for OpenStack.

OpenStack is excluded from RQ2's quantitative SHAP/LIME evaluation (no
Spearman agreement, no faithfulness test) because it has only 4 anomalous
instances across the entire dataset (see 02_build_splits.py / 03's
near-zero classification scores) — far too few for the deletion/comparison
statistics in 04_explain_rq2.py to be meaningful. It is retained here
purely for RQ3's qualitative question: do the top SHAP-ranked features map
to operationally recognisable failure patterns an on-call engineer would
recognise?

The model is trained on the full OpenStack matrix (no held-out split, for
the same reason: there is no way to hold out anomalies without leaving
either train or test with zero positive examples). LightGBM is used for
consistency with the tree-based explainer approach used elsewhere.
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
import shap
from lightgbm import LGBMClassifier

from utils_data import get_feature_columns
from utils_explain import extract_positive_class_shap

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
SHAP_VALUES_DIR = os.path.join(RESULTS_DIR, "shap_values")

RANDOM_STATE = 42


def main():
    """Train LightGBM on the full OpenStack matrix and run SHAP for qualitative RQ3 analysis."""
    print("STEP: OpenStack SHAP analysis (RQ3 qualitative only, excluded from RQ2 quantitative evaluation)")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SHAP_VALUES_DIR, exist_ok=True)

    try:
        matrix_path = os.path.join(PROCESSED_DIR, "OpenStack", "splits", "full.csv")
        print(f"  Loading: {matrix_path}")
        df = pd.read_csv(matrix_path)
        feature_cols = get_feature_columns(df)
        X, y = df[feature_cols], df["Label"]
        print(f"  {len(df):,} instance sessions, {len(feature_cols)} features, "
              f"{int(y.sum())} anomalous ({y.mean() * 100:.2f}%)")
        print("  NOTE: no held-out test split — with only 4 anomalies total, holding out "
              "any of them would leave train or test with zero positives, so RQ1-style "
              "generalisation metrics are not meaningful for this dataset (see "
              "02_build_splits.py). Model is trained on all data for RQ3 explanation only.")

        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        # With only 4 positives, the raw imbalance ratio (~516x) as
        # scale_pos_weight lets LightGBM build deep, narrow splits that
        # perfectly isolate the 4 anomalies and produce pathologically
        # large raw-margin outputs (confirmed here: an initial unregularised
        # run gave the single most common feature value in the whole
        # dataset, E3=2 (present in 1,777/2,069 rows), a SHAP magnitude of
        # >10,000 — a clear overfitting artifact, not a real pattern). A
        # sqrt-dampened weight plus shallow, wide-leaf trees keeps the model
        # from memorising individual rows while still biasing it toward the
        # minority class.
        scale_pos_weight = (n_neg / max(n_pos, 1)) ** 0.5
        model = LGBMClassifier(
            n_estimators=100, max_depth=4, num_leaves=15, min_child_samples=5,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, verbosity=-1,
        )
        print(f"  Training regularised LightGBM on full OpenStack matrix "
              f"(scale_pos_weight={scale_pos_weight:.2f}, max_depth=4, num_leaves=15)...")
        model.fit(X, y)

        model_path = os.path.join(MODELS_DIR, "OpenStack_LightGBM.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved model: {model_path}")

        print("  Running SHAP TreeExplainer on all instances...")
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X)
        shap_values = extract_positive_class_shap(explanation)

        npy_path = os.path.join(SHAP_VALUES_DIR, "OpenStack_shap_values.npy")
        np.save(npy_path, shap_values)
        print(f"  Saved SHAP values: {npy_path} (shape={shap_values.shape})")

        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
        plt.title("OpenStack: SHAP Summary Plot (Feature Impact on Anomaly Prediction, Qualitative Only)")
        summary_path = os.path.join(FIGURES_DIR, "OpenStack_shap_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP summary plot: {summary_path}")

        anomaly_positions = np.where(y.values == 1)[0]
        is_multiclass_output = np.array(explanation.values).ndim == 3
        pos_explanation = explanation[:, :, -1] if is_multiclass_output else explanation
        for i, pos in enumerate(anomaly_positions):
            plt.figure()
            shap.plots.waterfall(pos_explanation[int(pos)], show=False)
            plt.title(f"OpenStack: SHAP Waterfall — Anomalous Instance {i + 1}/{len(anomaly_positions)}")
            waterfall_path = os.path.join(FIGURES_DIR, f"OpenStack_shap_waterfall_{i + 1}.png")
            plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved SHAP waterfall plot for anomalous instance {i + 1}/{len(anomaly_positions)}: "
                  f"{waterfall_path}")

        ranking_df = pd.DataFrame({
            "Feature": feature_cols,
            "MeanAbsSHAP": np.abs(shap_values).mean(axis=0),
        }).sort_values("MeanAbsSHAP", ascending=False)
        ranking_path = os.path.join(RESULTS_DIR, "feature_ranking_OpenStack.csv")
        ranking_df.to_csv(ranking_path, index=False)
        print(f"  Saved feature ranking: {ranking_path}")
        print(ranking_df.head(10).to_string(index=False))

        print("\nSTEP COMPLETE: OpenStack SHAP analysis finished successfully")

    except Exception:
        print("\nERROR during OpenStack SHAP analysis:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
