"""Parse BGL.log with Drain3 and build a session-level event-occurrence matrix.

BGL log line format (whitespace-separated, content is free text):
    <Label> <TimestampEpoch> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>

Label field: '-' means normal, any other value means an anomaly alert code.
BGL has no block/request ID, so "sessions" are built as fixed-size,
non-overlapping windows of WINDOW_SIZE consecutive log lines (a standard
approach for node-level logs without a natural session key, e.g. DeepLog /
LogAnomaly-style windowing). A window is labelled anomalous if any line in
it is anomalous.

Output mirrors the shape of the HDFS Event_occurrence_matrix.csv so the
step-2 loader can treat all three datasets uniformly:
    WindowId, Label, WindowStartTime, E1..Ek
"""

import os
import sys
import traceback

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

RAW_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "BGL", "BGL.log"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "BGL")
STRUCTURED_OUT = os.path.join(OUT_DIR, "BGL_structured.csv")
TEMPLATES_OUT = os.path.join(OUT_DIR, "BGL_templates.csv")
MATRIX_OUT = os.path.join(OUT_DIR, "BGL_event_occurrence_matrix.csv")

WINDOW_SIZE = 100  # log lines per session window
PROGRESS_EVERY = 500_000


def build_template_miner():
    """Configure and return a Drain3 TemplateMiner that masks variable tokens before clustering.

    Without masking, variable content (memory addresses, IPs, numeric IDs)
    causes near-identical messages to fragment into separate templates,
    inflating the event vocabulary far beyond the true number of log event
    types. Masks are applied in order: IP addresses, hex addresses, then
    generic decimal numbers.
    """
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth = 4
    config.profiling_enabled = False
    config.masking_instructions = [
        MaskingInstruction(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?", "IP"),
        MaskingInstruction(r"0x[0-9a-fA-F]+", "HEX"),
        MaskingInstruction(r"\d+", "NUM"),
    ]
    return TemplateMiner(config=config)


def parse_line(line):
    """Split a raw BGL log line into (label, timestamp_epoch, content).

    Returns None if the line does not have enough fields to parse.
    """
    parts = line.rstrip("\n").split(maxsplit=9)
    if len(parts) < 9:
        return None
    label = parts[0]
    timestamp_epoch = parts[1]
    content = parts[9] if len(parts) > 9 else ""
    return label, timestamp_epoch, content


def main():
    """Run Drain3 parsing over BGL.log, then aggregate into fixed-size session windows."""
    print("STEP: Preprocessing BGL.log with Drain3")
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        miner = build_template_miner()

        rows = []  # (timestamp_epoch, is_anomaly, event_id)
        skipped = 0

        print(f"Reading and parsing: {os.path.abspath(RAW_LOG_PATH)}")
        with open(RAW_LOG_PATH, "r", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                parsed = parse_line(line)
                if parsed is None:
                    skipped += 1
                    continue
                label, ts_epoch, content = parsed
                result = miner.add_log_message(content)
                event_id = result["cluster_id"]
                is_anomaly = 0 if label == "-" else 1
                rows.append((ts_epoch, is_anomaly, f"E{event_id}"))

                if i % PROGRESS_EVERY == 0:
                    print(f"  Processed {i:,} lines... "
                          f"({len(list(miner.drain.clusters))} templates so far)")

        clusters = list(miner.drain.clusters)
        print(f"Finished parsing. Total lines: {len(rows):,}, skipped (malformed): {skipped:,}")
        print(f"Total unique templates discovered: {len(clusters)}")

        # --- Save structured log + templates ---
        print("Saving structured log and template list...")
        structured_df = pd.DataFrame(rows, columns=["Timestamp", "Label", "EventId"])
        structured_df.to_csv(STRUCTURED_OUT, index=False)
        print(f"  Saved: {STRUCTURED_OUT} ({len(structured_df):,} rows)")

        templates = [
            {"EventId": f"E{c.cluster_id}", "EventTemplate": c.get_template()}
            for c in clusters
        ]
        templates_df = pd.DataFrame(templates)
        templates_df.to_csv(TEMPLATES_OUT, index=False)
        print(f"  Saved: {TEMPLATES_OUT} ({len(templates_df):,} templates)")

        # --- Build fixed-size session windows ---
        print(f"Building fixed-size windows of {WINDOW_SIZE} lines...")
        all_event_ids = sorted(templates_df["EventId"].tolist(), key=lambda e: int(e[1:]))
        num_windows = (len(structured_df) + WINDOW_SIZE - 1) // WINDOW_SIZE

        window_rows = []
        for w in range(num_windows):
            start = w * WINDOW_SIZE
            end = min(start + WINDOW_SIZE, len(structured_df))
            chunk = structured_df.iloc[start:end]

            counts = chunk["EventId"].value_counts().to_dict()
            row = {
                "WindowId": f"W{w}",
                "Label": "Anomaly" if chunk["Label"].sum() > 0 else "Normal",
                "WindowStartTime": chunk["Timestamp"].iloc[0],
            }
            for eid in all_event_ids:
                row[eid] = counts.get(eid, 0)
            window_rows.append(row)

            if (w + 1) % 5000 == 0:
                print(f"  Built {w + 1:,}/{num_windows:,} windows...")

        matrix_df = pd.DataFrame(window_rows)
        matrix_df.to_csv(MATRIX_OUT, index=False)
        print(f"  Saved: {MATRIX_OUT} ({len(matrix_df):,} windows, {len(all_event_ids)} event columns)")
        print(f"  Anomalous windows: {(matrix_df['Label'] == 'Anomaly').sum():,} "
              f"/ {len(matrix_df):,} "
              f"({(matrix_df['Label'] == 'Anomaly').mean() * 100:.2f}%)")

        print("\nSTEP COMPLETE: BGL preprocessing finished successfully")

    except Exception:
        print("\nERROR during BGL preprocessing:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
