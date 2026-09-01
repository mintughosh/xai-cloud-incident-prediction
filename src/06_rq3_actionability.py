"""RQ3: Do top SHAP-ranked features map to operationally recognisable failure patterns?

For HDFS and BGL, the top-15 SHAP features (by mean |SHAP value|, from
results/feature_ranking_{dataset}.csv produced in 04_explain_rq2.py) are
matched to their Drain3 template text and assigned one of six operational
categories via keyword rules (utils_rq3.categorize_template). Actionability
coverage = the fraction of those 15 features whose template text is
specific enough to categorise as a recognisable failure type (i.e. not
"Unknown").

Template sources:
  - HDFS: LogPai's own pre-processed template list
    (data/raw/HDFS_v1/preprocessed/HDFS.log_templates.csv), since HDFS was
    parsed by LogPai, not by our own Drain3 run.
  - BGL: our own Drain3 output from 01_preprocess_bgl.py
    (data/processed/BGL/BGL_templates.csv). Note: this is NOT under logs/ —
    logs/ holds this project's run logs (stdout/stderr captures), not
    Drain3's template artifacts, which 01_preprocess_bgl.py writes to
    data/processed/BGL/ alongside the event-occurrence matrix.
"""

import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils_rq3 import CATEGORY_COLORS, categorize_template, truncate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TOP_N = 15

DATASET_CONFIG = {
    "HDFS": {
        "ranking_path": os.path.join(RESULTS_DIR, "feature_ranking_HDFS.csv"),
        "templates_path": os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "HDFS_v1",
            "preprocessed", "HDFS.log_templates.csv",
        ),
    },
    "BGL": {
        "ranking_path": os.path.join(RESULTS_DIR, "feature_ranking_BGL.csv"),
        "templates_path": os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "BGL", "BGL_templates.csv"
        ),
    },
}


def build_actionability_table(name, config):
    """Match a dataset's top-N SHAP features to template text and assign operational categories."""
    print(f"\n=== {name} ===")
    print(f"  Loading SHAP ranking: {config['ranking_path']}")
    ranking_df = pd.read_csv(config["ranking_path"])

    print(f"  Loading templates: {config['templates_path']}")
    if not os.path.exists(config["templates_path"]):
        raise FileNotFoundError(
            f"Template file not found for {name}: {config['templates_path']}"
        )
    templates_df = pd.read_csv(config["templates_path"])

    top_df = ranking_df.head(TOP_N).reset_index(drop=True)
    top_df = top_df.merge(
        templates_df[["EventId", "EventTemplate"]],
        left_on="Feature", right_on="EventId", how="left",
    )

    missing = top_df["EventTemplate"].isna().sum()
    if missing:
        print(f"  WARNING: {missing}/{len(top_df)} top features had no matching template")

    top_df["EventTemplate"] = top_df["EventTemplate"].fillna("(no matching template found)")
    top_df["Category"] = top_df["EventTemplate"].apply(categorize_template)
    top_df.insert(0, "rank", range(1, len(top_df) + 1))
    top_df = top_df.rename(columns={
        "EventId": "event_id", "EventTemplate": "template_text", "MeanAbsSHAP": "mean_shap",
        "Category": "category",
    })[["rank", "event_id", "template_text", "mean_shap", "category"]]

    print(f"  Top {TOP_N} SHAP features and categories:")
    for _, row in top_df.iterrows():
        print(f"    [{row['rank']:>2}] {row['event_id']}: {row['template_text']}  -> {row['category']}")

    out_path = os.path.join(RESULTS_DIR, f"rq3_actionability_{name}.csv")
    top_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    non_unknown = (top_df["category"] != "Unknown").sum()
    coverage = non_unknown / len(top_df)
    print(f"  actionability_coverage = {non_unknown}/{len(top_df)} = {coverage:.2%}")

    make_bar_chart(name, top_df)

    dominant_category = top_df["category"].value_counts().idxmax()
    return {
        "dataset": name,
        "total_features_checked": len(top_df),
        "categorised": int(non_unknown),
        "coverage_pct": round(coverage * 100, 2),
        "dominant_category": dominant_category,
    }


def make_bar_chart(name, top_df):
    """Save a horizontal bar chart of the top-N SHAP features, coloured by operational category."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_df = top_df.iloc[::-1]  # highest-ranked feature at the top of the chart
    labels = [truncate(t, 50) for t in plot_df["template_text"]]
    colors = [CATEGORY_COLORS[c] for c in plot_df["category"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, plot_df["mean_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{name}: Top {len(top_df)} SHAP Features by Operational Category")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()]
    ax.legend(handles, CATEGORY_COLORS.keys(), loc="lower right", fontsize=8)

    fig_path = os.path.join(FIGURES_DIR, f"rq3_actionability_{name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")


def main():
    """Run RQ3 actionability analysis for HDFS and BGL, and save a combined coverage summary."""
    print("STEP: RQ3 actionability analysis (HDFS, BGL)")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        coverage_rows = []
        for name, config in DATASET_CONFIG.items():
            coverage_rows.append(build_actionability_table(name, config))

        coverage_df = pd.DataFrame(coverage_rows)
        coverage_path = os.path.join(RESULTS_DIR, "rq3_coverage.csv")
        coverage_df.to_csv(coverage_path, index=False)
        print(f"\nSaved coverage summary: {coverage_path}")
        print(coverage_df.to_string(index=False))

        print("\nSTEP COMPLETE: RQ3 actionability analysis finished successfully")

    except Exception:
        print("\nERROR during RQ3 actionability analysis:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
