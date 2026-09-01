"""Inspect raw and pre-processed dataset files before running Drain3 parsing.

Prints file sizes, line counts, and sample lines for BGL, OpenStack, and
HDFS (pre-processed) so parsing decisions in later scripts are grounded in
what the data actually looks like, rather than assumptions.
"""

import os
import sys
import traceback

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def human_size(num_bytes):
    """Convert a byte count into a human-readable string (e.g. '709.4 MB')."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def count_lines(path, limit=None):
    """Count lines in a text file, optionally stopping early at `limit` lines."""
    count = 0
    with open(path, "r", errors="replace") as f:
        for _ in f:
            count += 1
            if limit is not None and count >= limit:
                return count, True
    return count, False


def show_sample_lines(path, n=5):
    """Print the first n lines of a text file."""
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(f"    [{i}] {line.rstrip()}")


def inspect_log_file(label, path, sample_n=5, count_cap=2_000_000):
    """Print size, sample lines, and an approximate/exact line count for a raw log file."""
    print(f"\n=== {label} ===")
    print(f"  Path: {path}")
    if not os.path.exists(path):
        print("  MISSING FILE")
        return
    size = os.path.getsize(path)
    print(f"  Size: {human_size(size)}")
    print(f"  Sample lines:")
    show_sample_lines(path, sample_n)

    print(f"  Counting lines (cap {count_cap:,} for large files)...")
    n, hit_cap = count_lines(path, limit=count_cap)
    if hit_cap:
        print(f"  Line count: >= {n:,} (stopped early, file is larger)")
    else:
        print(f"  Line count: {n:,}")


def inspect_csv_file(label, path, nrows_preview=5):
    """Print shape, columns, dtypes, and a head preview for a CSV file."""
    print(f"\n=== {label} ===")
    print(f"  Path: {path}")
    if not os.path.exists(path):
        print("  MISSING FILE")
        return
    size = os.path.getsize(path)
    print(f"  Size: {human_size(size)}")
    try:
        df = pd.read_csv(path, nrows=nrows_preview)
        full_cols = df.columns.tolist()
        print(f"  Columns ({len(full_cols)}): {full_cols[:10]}{' ...' if len(full_cols) > 10 else ''}")
        print(f"  Dtypes (first 10 cols):\n{df.dtypes.head(10)}")
        print(f"  Head:\n{df.head(nrows_preview)}")
    except Exception:
        print("  ERROR reading CSV:")
        traceback.print_exc()


def inspect_npz_file(label, path):
    """Print array names and shapes for a .npz file."""
    import numpy as np

    print(f"\n=== {label} ===")
    print(f"  Path: {path}")
    if not os.path.exists(path):
        print("  MISSING FILE")
        return
    size = os.path.getsize(path)
    print(f"  Size: {human_size(size)}")
    try:
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            arr = data[key]
            print(f"  Array '{key}': shape={arr.shape}, dtype={arr.dtype}")
    except Exception:
        print("  ERROR reading npz:")
        traceback.print_exc()


def main():
    """Run inspection over BGL, OpenStack, and pre-processed HDFS files."""
    print("STEP: Inspecting raw and pre-processed dataset files")
    print(f"RAW_DIR = {os.path.abspath(RAW_DIR)}")

    try:
        # --- BGL ---
        print("\n########## BGL ##########")
        inspect_log_file("BGL.log", os.path.join(RAW_DIR, "BGL", "BGL.log"))

        # --- OpenStack ---
        print("\n########## OpenStack ##########")
        inspect_log_file(
            "openstack_normal1.log",
            os.path.join(RAW_DIR, "OpenStack", "openstack_normal1.log"),
        )
        inspect_log_file(
            "openstack_normal2.log",
            os.path.join(RAW_DIR, "OpenStack", "openstack_normal2.log"),
        )
        inspect_log_file(
            "openstack_abnormal.log",
            os.path.join(RAW_DIR, "OpenStack", "openstack_abnormal.log"),
        )
        inspect_log_file(
            "anomaly_labels.txt",
            os.path.join(RAW_DIR, "OpenStack", "anomaly_labels.txt"),
        )

        # --- HDFS (already pre-processed by LogPai) ---
        print("\n########## HDFS (pre-processed) ##########")
        hdfs_pre = os.path.join(RAW_DIR, "HDFS_v1", "preprocessed")
        inspect_csv_file(
            "anomaly_label.csv", os.path.join(hdfs_pre, "anomaly_label.csv")
        )
        inspect_csv_file(
            "Event_occurrence_matrix.csv",
            os.path.join(hdfs_pre, "Event_occurrence_matrix.csv"),
        )
        inspect_csv_file(
            "HDFS.log_templates.csv",
            os.path.join(hdfs_pre, "HDFS.log_templates.csv"),
        )
        inspect_csv_file(
            "Event_traces.csv", os.path.join(hdfs_pre, "Event_traces.csv")
        )
        inspect_npz_file("HDFS.npz", os.path.join(hdfs_pre, "HDFS.npz"))

        print("\nSTEP COMPLETE: Data inspection finished successfully")

    except Exception:
        print("\nERROR during data inspection:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
