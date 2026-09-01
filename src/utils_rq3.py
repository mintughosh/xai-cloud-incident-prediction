"""Shared helpers for RQ3: mapping top SHAP features to operational failure categories.

Categorisation is rule-based keyword matching on template text, checked in
a fixed priority order so overlapping terms (e.g. "disk" appears in both
hardware-failure and resource-exhaustion vocabulary) resolve consistently.
"""

import re

CATEGORIES = [
    "Data integrity",
    "Resource exhaustion",
    "Network fault",
    "Hardware failure",
    "Application error",
    "Unknown",
]

# Consistent colour scheme across all RQ3 plots (HDFS, BGL).
CATEGORY_COLORS = {
    "Hardware failure": "#d62728",
    "Network fault": "#1f77b4",
    "Application error": "#ff7f0e",
    "Resource exhaustion": "#9467bd",
    "Data integrity": "#2ca02c",
    "Unknown": "#7f7f7f",
}

_RULES = [
    ("Data integrity", [
        "checksum", "corrupt", "replica", "replication", "invalid block",
        "mismatch", "verification failed", "verification succeeded",
        "bad crc", "does not belong to any file", "invalidset",
        "redundant addstoredblock",
    ]),
    ("Resource exhaustion", [
        "buffer full", "queue full", "threshold exceeded", "exceeded quota",
        "out of memory", "no space left", "disk full", "not enough space",
        "exceeds", "resource temporarily unavailable",
    ]),
    ("Network fault", [
        "connection refused", "connection reset", "connection timed out",
        "timeout", "timed out", "socket", "unreachable", "packet",
        "network unreachable", "broken pipe", "retry connecting",
        "failed to connect", "lustre mount",
        "link training", "link failed",
    ]),
    ("Hardware failure", [
        "disk error", "disk failure", "parity error", "machine check",
        "cache parity", "ecc", "dimm", "power supply", "fan speed",
        "kernel panic", "cpu error", "cpu failure", "memory error",
        "hardware error", "bad sector",
        "tlb error", "storage interrupt", "ce sym", "program interrupt",
        "chip status", "instruction cache", "data address:",
    ]),
    ("Application error", [
        "exception", "assert", "abort", "crash", "traceback", "fail to",
        "failed to", "error code", "exit code", "terminated", "stack trace",
        "unable to", "denied", "panic", "generating core", "core dump",
        "missing or invalid", "unexpected error",
        # Generic catch-alls: checked last within this category, and this
        # category is checked after all more-specific ones above, so a
        # bare "error"/"failed" only lands here when no domain-specific
        # vocabulary (hardware, network, integrity, resource) matched first.
        "error", "failed",
    ]),
]


def categorize_template(template_text):
    """Assign one operational category to a log template using ordered keyword rules.

    Rule order matters: Data integrity and Resource exhaustion are checked
    before Hardware failure / Network fault since terms like "disk" or
    "full" are ambiguous between categories, and the more specific failure
    semantics (corruption, exhaustion) should take precedence over the
    generic component name.
    """
    text = str(template_text).lower()
    for category, keywords in _RULES:
        for kw in keywords:
            if kw in text:
                return category
    return "Unknown"


def truncate(text, max_len=50):
    """Truncate template text to max_len characters for use as a plot label."""
    text = str(text)
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def strip_uuid_markers(text):
    """Collapse repeated Drain3 wildcard markers for slightly more readable labels."""
    return re.sub(r"(<\*>\s*)+", "<*> ", str(text)).strip()
