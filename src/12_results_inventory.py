"""Generate a complete inventory of results/ and a validation checklist.

Two outputs:
  - logs/results_inventory.txt: every file in results/ and results/figures/
    with its size, and for each CSV in results/ (not results/figures/),
    its row count, column names, and first 3 rows.
  - logs/results_checklist.txt: PRESENT/MISSING for a fixed list of
    expected deliverable files from RQ1-RQ3 and the follow-up ablation
    studies.
"""

import os
import sys
import traceback

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

EXPECTED_FILES = [
    "results/model_results.csv",
    "results/shap_lime_spearman.csv",
    "results/faithfulness_HDFS.csv",
    "results/faithfulness_BGL.csv",
    "results/rq3_actionability_HDFS.csv",
    "results/rq3_actionability_BGL.csv",
    "results/rq3_coverage.csv",
    "results/ablation_window_size.csv",
    "results/temporal_drift_bgl.csv",
    "results/hyperparameter_sensitivity.csv",
    "results/shap_stability_window.csv",
    "results/feature_ranking_HDFS.csv",
    "results/feature_ranking_BGL.csv",
    "results/lime_ranking_HDFS.csv",
    "results/lime_ranking_BGL.csv",
]


def human_size(num_bytes):
    """Convert a byte count into a human-readable string (e.g. '709.4 MB')."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def list_directory_files(directory):
    """Return sorted (filename, size_bytes) pairs for files directly inside a directory."""
    if not os.path.isdir(directory):
        return []
    entries = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            entries.append((name, os.path.getsize(path)))
    return entries


def build_inventory():
    """Build the full results/ + results/figures/ inventory, with CSV previews, as a list of lines."""
    lines = ["Results Inventory", "=" * 60, ""]

    lines.append("results/ (top level)")
    lines.append("-" * 60)
    top_level_files = list_directory_files(RESULTS_DIR)
    for name, size in top_level_files:
        lines.append(f"  {name}  ({human_size(size)})")

    lines.append("")
    lines.append("results/figures/")
    lines.append("-" * 60)
    figure_files = list_directory_files(FIGURES_DIR)
    for name, size in figure_files:
        lines.append(f"  {name}  ({human_size(size)})")

    lines.append("")
    lines.append("=" * 60)
    lines.append("CSV previews (results/ top level only)")
    lines.append("=" * 60)

    csv_files = [name for name, _ in top_level_files if name.endswith(".csv")]
    for name in csv_files:
        path = os.path.join(RESULTS_DIR, name)
        lines.append("")
        lines.append(f"--- {name} ---")
        try:
            df = pd.read_csv(path)
            lines.append(f"  rows: {len(df):,}")
            lines.append(f"  columns: {list(df.columns)}")
            lines.append("  first 3 rows:")
            preview = df.head(3).to_string(index=False)
            for preview_line in preview.splitlines():
                lines.append(f"    {preview_line}")
        except Exception as e:
            lines.append(f"  ERROR reading CSV: {e}")

    return lines


def build_checklist():
    """Check each expected deliverable file for existence and return PRESENT/MISSING lines."""
    lines = ["Results Validation Checklist", "=" * 60, ""]
    project_root = os.path.join(os.path.dirname(__file__), "..")

    n_present = 0
    for rel_path in EXPECTED_FILES:
        full_path = os.path.join(project_root, rel_path)
        exists = os.path.exists(full_path)
        status = "PRESENT" if exists else "MISSING"
        if exists:
            n_present += 1
        lines.append(f"[{status}] {rel_path}")

    lines.append("")
    lines.append(f"Summary: {n_present}/{len(EXPECTED_FILES)} expected files present")
    return lines


def main():
    """Generate the results inventory and validation checklist, saving both under logs/."""
    print("STEP: Generating results inventory and validation checklist")
    os.makedirs(LOGS_DIR, exist_ok=True)

    try:
        print("  Building inventory of results/ and results/figures/...")
        inventory_lines = build_inventory()
        inventory_path = os.path.join(LOGS_DIR, "results_inventory.txt")
        with open(inventory_path, "w") as f:
            f.write("\n".join(inventory_lines) + "\n")
        print(f"  Saved: {inventory_path}")

        print("\n  Running validation checklist...")
        checklist_lines = build_checklist()
        for line in checklist_lines:
            print(f"  {line}")
        checklist_path = os.path.join(LOGS_DIR, "results_checklist.txt")
        with open(checklist_path, "w") as f:
            f.write("\n".join(checklist_lines) + "\n")
        print(f"\n  Saved: {checklist_path}")

        print("\nSTEP COMPLETE: Results inventory and checklist generated successfully")

    except Exception:
        print("\nERROR during results inventory generation:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
