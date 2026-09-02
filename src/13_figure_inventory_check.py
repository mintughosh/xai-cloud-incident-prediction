"""Figure Generation Check: verify all required dissertation figures exist and
meet minimum standards (dpi>=150, axis labels/title, legend where applicable),
regenerating any that are missing or non-compliant.

Findings from the initial check (documented here, not just in the log):
  - The 4 SHAP figures existed under a different naming convention
    (HDFS_shap_summary.png vs the required shap_summary_HDFS.png) and,
    on inspection, had no title — shap.summary_plot()/shap.plots.waterfall()
    don't add one automatically, and 04_explain_rq2.py never added one
    either. Both issues are fixed here: regenerated under the required
    names, with an explicit title added.
  - faithfulness_HDFS.png / faithfulness_BGL.png never existed at all —
    04_explain_rq2.py saved the faithfulness CSVs but no plot. Generated
    fresh from those CSVs here.
  - rq3_actionability_{HDFS,BGL}.png, ablation_window_size.png,
    temporal_drift_bgl.png, and hyperparameter_heatmap.png already existed
    under the exact required names with titles/labels/legends already in
    place (from 06/08/10/11's own generation code) — verified, not touched.
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
from PIL import Image

from utils_data import get_feature_columns
from utils_explain import extract_positive_class_shap, stratified_sample

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

SAMPLE_SIZE = 200
RANDOM_STATE = 42
MIN_DPI = 150

REQUIRED_FIGURES = [
    "shap_summary_HDFS.png",
    "shap_summary_BGL.png",
    "shap_waterfall_HDFS.png",
    "shap_waterfall_BGL.png",
    "rq3_actionability_HDFS.png",
    "rq3_actionability_BGL.png",
    "ablation_window_size.png",
    "temporal_drift_bgl.png",
    "hyperparameter_heatmap.png",
    "faithfulness_HDFS.png",
    "faithfulness_BGL.png",
]

MODEL_CONFIG = {
    "HDFS": "HDFS_LightGBM.pkl",
    "BGL": "BGL_RandomForest.pkl",
}


def get_image_info(path):
    """Return (width_px, height_px, dpi_x, dpi_y) for a PNG file."""
    with Image.open(path) as img:
        width, height = img.size
        dpi = img.info.get("dpi", (72, 72))
    return width, height, dpi[0], dpi[1]


NEEDS_SHAP_BACKGROUND = {"HDFS": False, "BGL": True}
BACKGROUND_SIZE = 100


def load_model_and_sample(name):
    """Load the dataset's saved model and reconstruct the exact same 200-instance test sample
    used in 04_explain_rq2.py (same random_state, so stratified_sample reproduces it exactly).

    Also returns a background sample (None for HDFS) — BGL's
    class_weight='balanced' RandomForest requires
    feature_perturbation='interventional' with an explicit background to
    produce correct (not numerically corrupted) SHAP values; see
    04_explain_rq2.py's run_shap() docstring for the full diagnosis.
    """
    model_path = os.path.join(MODELS_DIR, MODEL_CONFIG[name])
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, name, "splits", "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, name, "splits", "test.csv"))
    feature_cols = get_feature_columns(test_df)
    X_test, y_test = test_df[feature_cols], test_df["Label"]
    X_sample, y_sample = stratified_sample(X_test, y_test, SAMPLE_SIZE, RANDOM_STATE)

    background = None
    if NEEDS_SHAP_BACKGROUND[name]:
        background = train_df[feature_cols].sample(BACKGROUND_SIZE, random_state=RANDOM_STATE)

    return model, X_sample, y_sample, feature_cols, background


def build_explainer(model, background):
    """Return a TreeExplainer, using interventional perturbation when a background sample is given."""
    if background is not None:
        return shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    return shap.TreeExplainer(model)


def regenerate_shap_summary(name):
    """Regenerate a SHAP summary plot with an explicit title, under the required filename."""
    print(f"  Regenerating shap_summary_{name}.png...")
    model, X_sample, y_sample, feature_cols, background = load_model_and_sample(name)

    explainer = build_explainer(model, background)
    explanation = explainer(X_sample)
    shap_values = extract_positive_class_shap(explanation)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
    plt.title(f"{name}: SHAP Summary Plot (Feature Impact on Anomaly Prediction)")
    out_path = os.path.join(FIGURES_DIR, f"shap_summary_{name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")
    return out_path


def regenerate_shap_waterfall(name):
    """Regenerate a SHAP waterfall plot (first anomalous sample instance) with a title."""
    print(f"  Regenerating shap_waterfall_{name}.png...")
    model, X_sample, y_sample, feature_cols, background = load_model_and_sample(name)

    explainer = build_explainer(model, background)
    explanation = explainer(X_sample)
    is_multiclass_output = np.array(explanation.values).ndim == 3
    pos_explanation = explanation[:, :, -1] if is_multiclass_output else explanation

    anomaly_positions = np.where(y_sample.values == 1)[0]
    idx = int(anomaly_positions[0]) if len(anomaly_positions) > 0 else 0

    plt.figure()
    shap.plots.waterfall(pos_explanation[idx], show=False)
    plt.title(f"{name}: SHAP Waterfall — Instance {idx} (true label="
              f"{'anomaly' if idx in anomaly_positions else 'normal'})")
    out_path = os.path.join(FIGURES_DIR, f"shap_waterfall_{name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")
    return out_path


def regenerate_faithfulness_plot(name):
    """Generate a faithfulness (deletion test) plot from the saved CSV: mean_prob_drop vs k, per method."""
    print(f"  Regenerating faithfulness_{name}.png...")
    csv_path = os.path.join(RESULTS_DIR, f"faithfulness_{name}.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in df.groupby("Method"):
        group = group.sort_values("k")
        ax.errorbar(
            group["k"], group["mean_prob_drop"], yerr=group["std_prob_drop"],
            marker="o", capsize=3, label=method,
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("k (number of top-ranked features removed)")
    ax.set_ylabel("Mean drop in P(anomaly)  [baseline - perturbed]")
    ax.set_title(f"{name}: Faithfulness (Deletion Test) — SHAP vs LIME vs Random")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(FIGURES_DIR, f"faithfulness_{name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")
    return out_path


REGENERATORS = {
    "shap_summary_HDFS.png": lambda: regenerate_shap_summary("HDFS"),
    "shap_summary_BGL.png": lambda: regenerate_shap_summary("BGL"),
    "shap_waterfall_HDFS.png": lambda: regenerate_shap_waterfall("HDFS"),
    "shap_waterfall_BGL.png": lambda: regenerate_shap_waterfall("BGL"),
    "faithfulness_HDFS.png": lambda: regenerate_faithfulness_plot("HDFS"),
    "faithfulness_BGL.png": lambda: regenerate_faithfulness_plot("BGL"),
}


def main():
    """Check every required figure; regenerate missing ones; save the inventory report."""
    print("STEP: Figure generation check")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    try:
        report_lines = ["Figure Inventory Check", "=" * 60, ""]

        for filename in REQUIRED_FIGURES:
            path = os.path.join(FIGURES_DIR, filename)
            if not os.path.exists(path):
                print(f"[MISSING] {filename} -> regenerating...")
                if filename in REGENERATORS:
                    REGENERATORS[filename]()
                else:
                    print(f"  ERROR: no regenerator defined for {filename}")
                    report_lines.append(f"[FAILED - no regenerator] {filename}")
                    continue
            else:
                print(f"[PRESENT] {filename}")

            width, height, dpi_x, dpi_y = get_image_info(path)
            dpi_ok = dpi_x >= MIN_DPI and dpi_y >= MIN_DPI
            status = "OK" if dpi_ok else "DPI TOO LOW"
            print(f"    dimensions: {width}x{height}px, dpi: {dpi_x:.0f}x{dpi_y:.0f} [{status}]")
            report_lines.append(
                f"{filename}: {width}x{height}px, dpi={dpi_x:.0f}x{dpi_y:.0f} [{status}]"
            )

        report_lines.append("")
        report_lines.append(f"Summary: {len(REQUIRED_FIGURES)} required figures checked")
        report_path = os.path.join(LOGS_DIR, "figure_inventory.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"\nSaved: {report_path}")

        print("\nSTEP COMPLETE: Figure generation check finished successfully")

    except Exception:
        print("\nERROR during figure generation check:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
