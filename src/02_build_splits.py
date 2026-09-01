"""Build chronological 70/15/15 train/val/test splits for HDFS, BGL, and OpenStack.

Chronological ordering assumption per dataset:
  - HDFS: LogPai's Event_occurrence_matrix.csv lists blocks in the order they
    first appeared while parsing HDFS.log sequentially, so row order is used
    directly as chronological order (no independent timestamp column exists
    in this pre-processed file).
  - BGL: sessions are fixed-size windows built in log order in
    01_preprocess_bgl.py; sorted here by WindowStartTime for explicitness.

OpenStack is a documented exception: openstack_abnormal.log was collected
on 2017-05-14, entirely before openstack_normal1/2.log (2017-05-16/17). All
4 anomalous instances therefore fall in the earliest ~8% of any
chronologically-ordered timeline, so a real chronological split always
puts every anomaly in train and leaves val/test with zero positives. A
70/15/15 split is structurally impossible here regardless of session
granularity, so OpenStack instead gets its full normalized matrix saved
(no split) for 03_train_baselines.py to evaluate with stratified k-fold CV.

The HDFS event matrix's 'Type' column is dropped: it encodes the failure
reason code and is only non-null on Fail rows, which is direct label
leakage if left in as a feature.
"""

import os
import sys
import traceback

import pandas as pd

from utils_data import chronological_split, normalize_label, print_class_balance

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

DATASETS = {
    "HDFS": {
        "matrix_path": os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "HDFS_v1",
            "preprocessed", "Event_occurrence_matrix.csv",
        ),
        "id_col": "BlockId",
        "sort_col": None,
        "drop_cols": ["Type"],
        "split_strategy": "chronological",
    },
    "BGL": {
        "matrix_path": os.path.join(PROCESSED_DIR, "BGL", "BGL_event_occurrence_matrix.csv"),
        "id_col": "WindowId",
        "sort_col": "WindowStartTime",
        "drop_cols": [],
        "split_strategy": "chronological",
    },
    "OpenStack": {
        "matrix_path": os.path.join(PROCESSED_DIR, "OpenStack", "OpenStack_event_occurrence_matrix.csv"),
        "id_col": "InstanceId",
        "sort_col": "FirstSeenTime",
        "drop_cols": [],
        "split_strategy": "kfold",
    },
}


def build_dataset_splits(name, config):
    """Load a dataset's feature matrix, normalize labels, sort chronologically, and split it."""
    print(f"\n=== {name} ===")
    print(f"  Loading: {config['matrix_path']}")
    df = pd.read_csv(config["matrix_path"])
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    if config["drop_cols"]:
        present = [c for c in config["drop_cols"] if c in df.columns]
        if present:
            print(f"  Dropping leakage/non-feature columns: {present}")
            df = df.drop(columns=present)

    if config["sort_col"]:
        print(f"  Sorting chronologically by: {config['sort_col']}")
        df = df.sort_values(config["sort_col"]).reset_index(drop=True)
    else:
        print("  Using existing row order as chronological order (no timestamp column available)")

    print("  Normalizing labels to binary (0=normal, 1=anomaly)...")
    df["Label"] = normalize_label(df["Label"])
    print(f"  Overall anomaly rate: {df['Label'].mean() * 100:.2f}% "
          f"({int(df['Label'].sum()):,} / {len(df):,})")

    out_dir = os.path.join(PROCESSED_DIR, name, "splits")
    os.makedirs(out_dir, exist_ok=True)

    if config["split_strategy"] == "chronological":
        train_df, val_df, test_df = chronological_split(df)
        print("  Split sizes and class balance:")
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            out_path = os.path.join(out_dir, f"{split_name}.csv")
            split_df.to_csv(out_path, index=False)
            print_class_balance(split_name, split_df)
            print(f"      Saved: {out_path}")
    else:
        # kfold strategy: no split here. Save the full normalized matrix;
        # 03_train_baselines.py runs stratified k-fold CV over it directly.
        out_path = os.path.join(out_dir, "full.csv")
        df.to_csv(out_path, index=False)
        print("  Split strategy: stratified k-fold (no train/val/test split saved)")
        print_class_balance("full", df)
        print(f"      Saved: {out_path}")


def main():
    """Build chronological splits for all three datasets."""
    print("STEP: Building chronological train/val/test splits (70/15/15)")
    try:
        for name, config in DATASETS.items():
            if not os.path.exists(config["matrix_path"]):
                print(f"\n=== {name} ===\n  SKIPPED: matrix not found at {config['matrix_path']}")
                continue
            build_dataset_splits(name, config)

        print("\nSTEP COMPLETE: All available dataset splits built successfully")

    except Exception:
        print("\nERROR during split building:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
