"""RQ2: SHAP and LIME explainability analysis for the best model per dataset.

Best model per dataset (chosen from 03_train_baselines.py's test-set results):
  - HDFS: LightGBM (highest test F1)
  - BGL:  Random Forest (best test-set generalisation: highest test
          Precision/F1 among the three models, despite the chronological
          val->test performance drop all models show on BGL)

Each model is retrained here (identical hyperparameters to
03_train_baselines.py) on its dataset's train split and persisted to
results/models/, so the explanations below are traceable to a concrete,
reproducible model artifact rather than an in-memory object.

Pipeline per dataset:
  1. SHAP TreeExplainer on a class-stratified 200-instance sample of the
     test split.
  2. Save raw SHAP values as .npy.
  3. SHAP summary plot (global feature importance).
  4. SHAP waterfall plot for one anomalous instance.
  5. LIME explanations for the same 200 instances (so SHAP and LIME are
     compared on identical inputs).
  6. Save a per-instance LIME weight matrix as CSV.
  7. Faithfulness (deletion) test at k=1,2,3,5,10 for SHAP, LIME, and a
     random-ranking control.
  8. Spearman rank correlation between SHAP's and LIME's global feature
     importances.
  9. All numeric results saved as CSV in results/; all plots saved as PNG
     at dpi=150 in results/figures/.
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
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier

from utils_data import get_feature_columns
from utils_explain import (
    compute_spearman_agreement,
    extract_positive_class_shap,
    faithfulness_test,
    get_lime_feature_weights,
    random_baseline_faithfulness,
    stratified_sample,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
SHAP_VALUES_DIR = os.path.join(RESULTS_DIR, "shap_values")

SAMPLE_SIZE = 200
LIME_NUM_SAMPLES = 1000
FAITHFULNESS_K = [1, 2, 3, 5, 10]
RANDOM_STATE = 42

DATASET_CONFIG = {
    "HDFS": {
        "model_name": "LightGBM",
        "build_model": lambda spw: LGBMClassifier(
            n_estimators=300, scale_pos_weight=spw, random_state=RANDOM_STATE, verbosity=-1
        ),
    },
    "BGL": {
        "model_name": "RandomForest",
        "build_model": lambda spw: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    },
}


def load_splits(name):
    """Load train/val/test splits and return (X_train, y_train, X_test, y_test, feature_cols)."""
    splits_dir = os.path.join(PROCESSED_DIR, name, "splits")
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    feature_cols = get_feature_columns(train_df)
    return (
        train_df[feature_cols], train_df["Label"],
        test_df[feature_cols], test_df["Label"],
        feature_cols,
    )


def train_and_save_model(name, config, X_train, y_train):
    """Train the dataset's designated best model and persist it as a pickle file."""
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    model = config["build_model"](scale_pos_weight)
    print(f"  Training {config['model_name']} on {name} train split ({len(X_train):,} rows)...")
    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{name}_{config['model_name']}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model: {model_path}")
    return model


def run_shap(name, model, X_sample, feature_cols):
    """Run SHAP TreeExplainer, save raw values, and produce summary + waterfall plots."""
    print("  Running SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    # RandomForestClassifier(class_weight='balanced') reweights samples during
    # training, which breaks TreeExplainer's internal additivity assumption
    # (it expects leaf values to reflect raw, unweighted sample counts). This
    # is a documented SHAP/sklearn interaction, not a sign of incorrect SHAP
    # values, so the strict additivity check is disabled here.
    explanation = explainer(X_sample, check_additivity=False)

    is_multiclass_output = np.array(explanation.values).ndim == 3
    pos_explanation = explanation[:, :, -1] if is_multiclass_output else explanation
    shap_values = extract_positive_class_shap(explanation)

    os.makedirs(SHAP_VALUES_DIR, exist_ok=True)
    npy_path = os.path.join(SHAP_VALUES_DIR, f"{name}_shap_values.npy")
    np.save(npy_path, shap_values)
    print(f"  Saved SHAP values: {npy_path} (shape={shap_values.shape})")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
    summary_path = os.path.join(FIGURES_DIR, f"{name}_shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP summary plot: {summary_path}")

    return shap_values, pos_explanation


def save_waterfall_plot(name, pos_explanation, y_sample):
    """Save a SHAP waterfall plot for the first anomalous instance in the sample."""
    anomaly_positions = np.where(y_sample.values == 1)[0]
    if len(anomaly_positions) == 0:
        print(f"  WARNING: no anomalous instance in the {name} sample, skipping waterfall plot")
        return
    idx = int(anomaly_positions[0])
    plt.figure()
    shap.plots.waterfall(pos_explanation[idx], show=False)
    waterfall_path = os.path.join(FIGURES_DIR, f"{name}_shap_waterfall.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP waterfall plot (instance {idx}, true label=anomaly): {waterfall_path}")


def run_lime(name, model, X_train, X_sample, feature_cols):
    """Run LIME on every sampled instance and return a per-instance weight matrix."""
    print(f"  Running LIME on {len(X_sample)} instances...")
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_cols,
        class_names=["Normal", "Anomaly"],
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_STATE,
    )

    weight_rows = []
    for i in range(len(X_sample)):
        exp = explainer.explain_instance(
            X_sample.iloc[i].values,
            model.predict_proba,
            num_features=len(feature_cols),
            num_samples=LIME_NUM_SAMPLES,
        )
        weight_rows.append(get_lime_feature_weights(exp, feature_cols))
        if (i + 1) % 50 == 0:
            print(f"    LIME progress: {i + 1}/{len(X_sample)}")

    lime_df = pd.DataFrame(weight_rows, columns=feature_cols)
    out_path = os.path.join(RESULTS_DIR, f"lime_ranking_{name}.csv")
    lime_df.to_csv(out_path, index=False)
    print(f"  Saved LIME weight matrix: {out_path}")
    return lime_df


def run_faithfulness_and_agreement(name, model, X_sample, shap_values, lime_df, feature_cols):
    """Run the faithfulness (deletion) test and SHAP-LIME Spearman agreement, saving both as CSV."""
    print("  Running faithfulness (deletion) test...")
    lime_matrix = lime_df[feature_cols].values
    rows = []
    rows.extend(faithfulness_test(model, X_sample, shap_values, feature_cols, FAITHFULNESS_K, "SHAP"))
    rows.extend(faithfulness_test(model, X_sample, lime_matrix, feature_cols, FAITHFULNESS_K, "LIME"))
    rows.extend(random_baseline_faithfulness(model, X_sample, feature_cols, FAITHFULNESS_K))
    faith_df = pd.DataFrame(rows)
    faith_df.insert(0, "Dataset", name)
    faith_path = os.path.join(RESULTS_DIR, f"faithfulness_{name}.csv")
    faith_df.to_csv(faith_path, index=False)
    print(f"  Saved faithfulness results: {faith_path}")
    print(faith_df.to_string(index=False))

    print("  Computing SHAP-LIME Spearman rank agreement...")
    corr, pvalue, shap_imp, lime_imp = compute_spearman_agreement(shap_values, lime_df, feature_cols)
    ranking_df = pd.DataFrame({
        "Feature": feature_cols,
        "MeanAbsSHAP": shap_imp,
        "MeanAbsLIME": lime_imp,
    }).sort_values("MeanAbsSHAP", ascending=False)
    ranking_path = os.path.join(RESULTS_DIR, f"feature_ranking_{name}.csv")
    ranking_df.to_csv(ranking_path, index=False)
    print(f"  Saved feature ranking: {ranking_path}")
    print(f"  Spearman correlation (SHAP vs LIME): r={corr:.4f}, p={pvalue:.4g}")

    return {"Dataset": name, "SpearmanR": corr, "PValue": pvalue, "N_Features": len(feature_cols)}


def process_dataset(name, config):
    """Run the full RQ2 pipeline (SHAP + LIME + faithfulness + agreement) for one dataset."""
    print(f"\n=== {name}: explaining {config['model_name']} ===")
    X_train, y_train, X_test, y_test, feature_cols = load_splits(name)
    model = train_and_save_model(name, config, X_train, y_train)

    X_sample, y_sample = stratified_sample(X_test, y_test, SAMPLE_SIZE, RANDOM_STATE)
    print(f"  Sampled {len(X_sample)} test instances ({int(y_sample.sum())} anomalous)")

    shap_values, pos_explanation = run_shap(name, model, X_sample, feature_cols)
    save_waterfall_plot(name, pos_explanation, y_sample)

    lime_df = run_lime(name, model, X_train, X_sample, feature_cols)
    spearman_row = run_faithfulness_and_agreement(name, model, X_sample, shap_values, lime_df, feature_cols)
    return spearman_row


def main():
    """Run RQ2 SHAP/LIME analysis for HDFS (LightGBM) and BGL (Random Forest)."""
    print("STEP: RQ2 explainability analysis (SHAP + LIME)")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        spearman_rows = []
        for name, config in DATASET_CONFIG.items():
            spearman_rows.append(process_dataset(name, config))

        spearman_df = pd.DataFrame(spearman_rows)
        spearman_path = os.path.join(RESULTS_DIR, "shap_lime_spearman.csv")
        spearman_df.to_csv(spearman_path, index=False)
        print(f"\nSaved combined Spearman agreement summary: {spearman_path}")
        print(spearman_df.to_string(index=False))

        print("\nSTEP COMPLETE: RQ2 explainability analysis finished successfully")

    except Exception:
        print("\nERROR during RQ2 explainability analysis:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
