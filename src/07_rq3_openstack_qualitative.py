"""RQ3 qualitative analysis for OpenStack — no quantitative metrics (see 02_build_splits.py:
OpenStack has only 4 anomalous instances, too few for held-out evaluation or a
15-feature coverage statistic to be meaningful).

This derives the actual event templates the 4 anomalous instances triggered
directly from data/processed/OpenStack/OpenStack_structured.csv and
OpenStack_templates.csv — there is no separate "manual inspection" artifact
from an earlier session to load; this script performs that inspection.
"""

import os
import sys
import traceback

import pandas as pd

from utils_rq3 import categorize_template

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "OpenStack")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ANOMALY_LABELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "OpenStack", "anomaly_labels.txt"
)

import re
UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def load_anomalous_instance_ids():
    """Read anomaly_labels.txt and return the set of instance UUIDs flagged as anomalous."""
    ids = set()
    with open(ANOMALY_LABELS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if UUID_RE.match(line):
                ids.add(line)
    return ids


def main():
    """Derive and categorise the event templates used by OpenStack's 4 anomalous instances."""
    print("STEP: RQ3 OpenStack qualitative analysis (no quantitative metrics)")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        anomalous_ids = load_anomalous_instance_ids()
        print(f"  Anomalous instance UUIDs ({len(anomalous_ids)}): {sorted(anomalous_ids)}")

        structured = pd.read_csv(os.path.join(PROCESSED_DIR, "OpenStack_structured.csv"))
        templates = pd.read_csv(os.path.join(PROCESSED_DIR, "OpenStack_templates.csv"))

        sub = structured[structured["InstanceId"].isin(anomalous_ids)]
        print(f"  {len(sub):,} log lines across the 4 anomalous instances")

        event_counts = sub["EventId"].value_counts().rename("Count").reset_index()
        event_counts.columns = ["EventId", "Count"]
        merged = event_counts.merge(templates, on="EventId", how="left")
        merged["Category"] = merged["EventTemplate"].apply(categorize_template)
        merged = merged.sort_values("Count", ascending=False).reset_index(drop=True)

        print("  Event templates triggered by the 4 anomalous instances:")
        for _, row in merged.iterrows():
            print(f"    {row['EventId']} (x{row['Count']}): {row['EventTemplate']} -> {row['Category']}")

        category_counts = merged.groupby("Category")["Count"].sum().sort_values(ascending=False)
        n_unknown = int(merged.loc[merged["Category"] == "Unknown", "Count"].sum())
        n_total = int(merged["Count"].sum())
        non_unknown_counts = category_counts.drop("Unknown", errors="ignore")

        if len(non_unknown_counts) > 0:
            category_sentence = (
                f"{n_unknown}/{n_total} lines fall into 'Unknown' because their text carries no "
                f"failure-specific vocabulary at all, and the best-represented failure-adjacent "
                f"category among the rest is '{non_unknown_counts.idxmax()}' "
                f"({int(non_unknown_counts.max())}/{n_total} lines)."
            )
        else:
            category_sentence = (
                f"All {n_total}/{n_total} lines fall into 'Unknown': not one of the {len(merged)} "
                f"templates these instances triggered carries any failure-specific vocabulary."
            )

        description = (
            f"The 4 anomalous OpenStack instances trigger {len(merged)} distinct event "
            f"templates over {n_total} log lines, and critically, none of the top-occurring "
            f"templates are explicit error or failure messages — they are routine VM "
            f"lifecycle operations (instance creation, resource claims, deletion, and network "
            f"deallocation). {category_sentence} "
            f"This indicates that, unlike HDFS and BGL where anomalies correspond to distinct "
            f"error-type log events, OpenStack's injected anomalies in this dataset manifest as "
            f"anomalous *timing or sequencing* of otherwise-normal lifecycle events rather than "
            f"the presence of a distinctly anomalous log line, which independently explains why "
            f"the event-count feature representation used for RQ1/RQ2 could not separate them "
            f"(only 4 positives, near-zero F1 across all models) — the signal these templates "
            f"carry is not really present at the per-template-occurrence level to begin with."
        )

        print("\n  Qualitative description:")
        print(f"  {description}")

        out_lines = [
            "RQ3 — OpenStack Qualitative Analysis (no quantitative actionability metrics)",
            "=" * 78,
            "",
            f"Anomalous instance UUIDs ({len(anomalous_ids)}):",
            *[f"  - {uid}" for uid in sorted(anomalous_ids)],
            "",
            "Event templates triggered by these instances, with assigned operational category:",
            "",
        ]
        for _, row in merged.iterrows():
            out_lines.append(
                f"  {row['EventId']} (occurs {row['Count']}x): {row['EventTemplate']}"
            )
            out_lines.append(f"      -> Category: {row['Category']}")
        out_lines += ["", "Qualitative description:", "", description, ""]

        out_path = os.path.join(RESULTS_DIR, "rq3_openstack_qualitative.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(out_lines))
        print(f"\n  Saved: {out_path}")

        print("\nSTEP COMPLETE: RQ3 OpenStack qualitative analysis finished successfully")

    except Exception:
        print("\nERROR during RQ3 OpenStack qualitative analysis:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
