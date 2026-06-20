# src/reports/report_generator.py
"""
Module 4 — Automated Report Generation
Produces structured, immutable compliance records for every detected violation.
Outputs: JSON (per-event), append-only CSV audit log, SQLite DB records.
"""

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_CSV = ROOT / "outputs" / "compliance_audit.csv"
AUDIT_JSON = ROOT / "outputs" / "compliance_audit.jsonl"

# CSV columns (Module 4 required fields)
CSV_FIELDS = [
    "event_id", "timestamp", "clip_id", "zone", "behavior_class",
    "policy_rule_ref", "event_description", "severity", "escalation_action",
    "confidence", "frame_number", "frame_path",
]


def generate_report(
    clip_id: str,
    behavior_class: str,
    policy_rule_ref: str,
    event_description: str,
    severity: str,
    escalation_action: str,
    zone: str = "Zone-1",
    confidence: float = 0.0,
    frame_number: int = 0,
    frame_path: str = "",
) -> dict:
    """
    Generate a complete compliance report for a single violation event.
    Writes JSON file, appends to CSV and JSONL audit logs.
    Returns the report dict.
    """
    report = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clip_id": clip_id,
        "zone": zone,
        "behavior_class": behavior_class,
        "policy_rule_ref": policy_rule_ref,
        "event_description": event_description,
        "severity": severity,
        "escalation_action": escalation_action,
        "confidence": round(confidence, 4),
        "frame_number": frame_number,
        "frame_path": frame_path,
    }

    _write_json_report(report)
    _append_csv(report)
    _append_jsonl(report)

    return report


def _write_json_report(report: dict):
    """Write individual JSON file per event."""
    ts = report["timestamp"].replace(":", "-").replace("T", "_").replace("Z", "")
    filename = f"{ts}_{report['event_id'][:8]}.json"
    path = REPORTS_DIR / filename
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def _append_csv(report: dict):
    """Append to append-only audit CSV."""
    write_header = not AUDIT_CSV.exists()
    with open(AUDIT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(report)


def _append_jsonl(report: dict):
    """Append to append-only JSONL audit log."""
    with open(AUDIT_JSON, "a") as f:
        f.write(json.dumps(report) + "\n")


def load_audit_csv() -> list[dict]:
    """Load full audit CSV as list of dicts."""
    if not AUDIT_CSV.exists():
        return []
    with open(AUDIT_CSV) as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_audit_jsonl() -> list[dict]:
    """Load full JSONL audit log."""
    if not AUDIT_JSON.exists():
        return []
    records = []
    with open(AUDIT_JSON) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
