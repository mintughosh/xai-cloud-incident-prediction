"""Parse OpenStack logs with Drain3 and build a per-instance event-occurrence matrix.

OpenStack logs have no per-line anomaly label. Instead, `anomaly_labels.txt`
lists VM instance UUIDs that were deliberately injected with anomalies in
openstack_abnormal.log. Sessions are therefore built per VM instance UUID
(analogous to HDFS's per-block sessions), extracted from lines containing
"[instance: <uuid>]". Lines with no instance UUID are parsed for template
mining (to keep Drain3's template vocabulary complete) but are not attributed
to any session.

An instance session is labelled anomalous only if its UUID appears in
anomaly_labels.txt; all other instance sessions (from normal1, normal2, and
any instance in abnormal.log not on the injected list) are labelled normal.

Output mirrors the shape of the HDFS Event_occurrence_matrix.csv:
    InstanceId, Label, FirstSeenTime, E1..Ek
"""

import os
import re
import sys
import traceback

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "OpenStack")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "OpenStack")
STRUCTURED_OUT = os.path.join(OUT_DIR, "OpenStack_structured.csv")
TEMPLATES_OUT = os.path.join(OUT_DIR, "OpenStack_templates.csv")
MATRIX_OUT = os.path.join(OUT_DIR, "OpenStack_event_occurrence_matrix.csv")
ANOMALY_LABELS_PATH = os.path.join(RAW_DIR, "anomaly_labels.txt")

LOG_FILES = [
    ("openstack_normal1.log", "normal1"),
    ("openstack_normal2.log", "normal2"),
    ("openstack_abnormal.log", "abnormal"),
]

INSTANCE_RE = re.compile(r"\[instance:\s*([0-9a-fA-F-]{36})\]")
UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")

# nova log line prefix: "<logfile>.<rotation> <YYYY-MM-DD HH:MM:SS.mmm> <pid> <LEVEL> <logger> [<req...>] <content>"
LINE_RE = re.compile(
    r"^\S+\s+(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+\d+\s+\S+\s+\S+\s+\[.*?\]\s+(?P<content>.*)$"
)


def build_template_miner():
    """Configure and return a Drain3 TemplateMiner that masks variable tokens before clustering.

    OpenStack nova messages are dense with UUIDs (instance/request/tenant
    ids), IPs, and file paths that vary per request but don't change the
    underlying event type. Without masking these, Drain3 fragments what
    should be one template into thousands of near-duplicates. Masks are
    applied in order: UUIDs, IP addresses, hex addresses, then generic
    decimal numbers.
    """
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth = 4
    config.profiling_enabled = False
    config.masking_instructions = [
        MaskingInstruction(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "UUID"),
        MaskingInstruction(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?", "IP"),
        MaskingInstruction(r"0x[0-9a-fA-F]+", "HEX"),
        MaskingInstruction(r"\d+", "NUM"),
    ]
    return TemplateMiner(config=config)


def load_anomalous_instance_ids():
    """Read anomaly_labels.txt and return the set of instance UUIDs flagged as anomalous."""
    ids = set()
    with open(ANOMALY_LABELS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if UUID_RE.match(line):
                ids.add(line)
    return ids


def parse_line(line):
    """Extract (timestamp, content) from a raw OpenStack nova log line.

    Falls back to treating the whole line as content (with no timestamp) if
    the expected nova log prefix format doesn't match.
    """
    m = LINE_RE.match(line.rstrip("\n"))
    if m:
        return m.group("timestamp"), m.group("content")
    return None, line.rstrip("\n")


def main():
    """Run Drain3 parsing over all three OpenStack log files, then build per-instance sessions."""
    print("STEP: Preprocessing OpenStack logs with Drain3")
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        anomalous_ids = load_anomalous_instance_ids()
        print(f"Loaded {len(anomalous_ids)} anomalous instance UUIDs from anomaly_labels.txt")

        miner = build_template_miner()
        rows = []  # (source, timestamp, instance_id, event_id)

        for filename, source in LOG_FILES:
            path = os.path.join(RAW_DIR, filename)
            print(f"Reading and parsing: {os.path.abspath(path)}")
            with open(path, "r", errors="replace") as f:
                for line in f:
                    timestamp, content = parse_line(line)
                    result = miner.add_log_message(content)
                    event_id = result["cluster_id"]

                    m = INSTANCE_RE.search(line)
                    instance_id = m.group(1) if m else None

                    rows.append((source, timestamp, instance_id, f"E{event_id}"))
            print(f"  Done: {filename} ({sum(1 for r in rows if r[0] == source):,} lines)")

        clusters = list(miner.drain.clusters)
        print(f"Total lines parsed: {len(rows):,}")
        print(f"Total unique templates discovered: {len(clusters)}")

        structured_df = pd.DataFrame(
            rows, columns=["Source", "Timestamp", "InstanceId", "EventId"]
        )
        structured_df.to_csv(STRUCTURED_OUT, index=False)
        print(f"  Saved: {STRUCTURED_OUT} ({len(structured_df):,} rows)")

        templates_df = pd.DataFrame(
            [{"EventId": f"E{c.cluster_id}", "EventTemplate": c.get_template()} for c in clusters]
        )
        templates_df.to_csv(TEMPLATES_OUT, index=False)
        print(f"  Saved: {TEMPLATES_OUT} ({len(templates_df):,} templates)")

        # --- Build per-instance sessions (drop lines with no instance id) ---
        print("Building per-instance sessions...")
        session_df = structured_df.dropna(subset=["InstanceId"])
        print(f"  {len(session_df):,} / {len(structured_df):,} lines carry an instance id")

        all_event_ids = sorted(templates_df["EventId"].tolist(), key=lambda e: int(e[1:]))
        window_rows = []
        for instance_id, group in session_df.groupby("InstanceId"):
            counts = group["EventId"].value_counts().to_dict()
            row = {
                "InstanceId": instance_id,
                "Label": "Anomaly" if instance_id in anomalous_ids else "Normal",
                "FirstSeenTime": group["Timestamp"].dropna().min() if group["Timestamp"].notna().any() else None,
            }
            for eid in all_event_ids:
                row[eid] = counts.get(eid, 0)
            window_rows.append(row)

        matrix_df = pd.DataFrame(window_rows).sort_values("FirstSeenTime").reset_index(drop=True)
        matrix_df.to_csv(MATRIX_OUT, index=False)
        print(f"  Saved: {MATRIX_OUT} ({len(matrix_df):,} instance sessions, {len(all_event_ids)} event columns)")
        n_anom = (matrix_df["Label"] == "Anomaly").sum()
        print(f"  Anomalous sessions: {n_anom:,} / {len(matrix_df):,} ({n_anom / len(matrix_df) * 100:.2f}%)")

        missing = anomalous_ids - set(matrix_df["InstanceId"])
        if missing:
            print(f"  WARNING: {len(missing)} anomalous instance UUID(s) from anomaly_labels.txt "
                  f"were never seen in the logs: {missing}")

        print("\nSTEP COMPLETE: OpenStack preprocessing finished successfully")

    except Exception:
        print("\nERROR during OpenStack preprocessing:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
