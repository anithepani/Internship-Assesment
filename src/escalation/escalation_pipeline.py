# src/escalation/escalation_pipeline.py
"""
Module 3 — Escalation Pipeline
Routes violations to correct downstream channel based on severity tier.

LOW / MEDIUM  → persistent DB log only
HIGH / CRITICAL → real-time alert trigger + DB log
"""

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.policy_rules import ESCALATION_ROUTING, SEVERITY_COLORS


# ── In-process pub/sub alert queue ──────────────────────────────────────────
# Dashboard reads from this queue to render real-time alerts.
# Thread-safe deque; max 100 unread alerts.

_alert_queue: deque[dict] = deque(maxlen=100)
_alert_lock = threading.Lock()
_alert_callbacks: list = []   # callable(alert_dict) hooks for WebSocket etc.


def subscribe_alerts(callback):
    """Register a callback function that receives alert dicts in real time."""
    _alert_callbacks.append(callback)


def _push_alert(alert: dict):
    """Internal: push alert to queue and notify subscribers."""
    with _alert_lock:
        _alert_queue.append(alert)
    for cb in _alert_callbacks:
        try:
            cb(alert)
        except Exception:
            pass


def get_pending_alerts() -> list[dict]:
    """Drain and return all pending real-time alerts. Called by dashboard."""
    with _alert_lock:
        alerts = list(_alert_queue)
        _alert_queue.clear()
    return alerts


# ── Core escalation function ─────────────────────────────────────────────────

def escalate(event: dict) -> dict:
    """
    Process a violation event through the escalation pipeline.

    Args:
        event: dict with keys:
            event_id, clip_id, zone, behavior_class, policy_rule_ref,
            event_description, severity, confidence, frame_number, frame_path

    Returns:
        Updated event dict with escalation_action field populated.
    """
    severity = event.get("severity", "LOW")
    routing = ESCALATION_ROUTING.get(severity, ESCALATION_ROUTING["LOW"])

    event["escalation_action"] = routing["action_label"]

    # ── DB log (always) ─────────────────────────────────────────────────────
    if routing["db_log"]:
        _write_db_log(event)

    # ── Real-time alert (HIGH / CRITICAL only) ───────────────────────────────
    if routing["realtime_alert"]:
        alert = _build_alert(event)
        _push_alert(alert)
        _write_alert_json(alert)
        print(f"[ALERT] {severity} -- {event['behavior_class']} in {event['zone']} ({event['clip_id']})")
    else:
        print(f"[LOG]   {severity} -- {event['behavior_class']} in {event['zone']} ({event['clip_id']})")

    return event


def _write_db_log(event: dict):
    """Persist event to SQLite."""
    try:
        from src.reports.database import log_event
        log_event(
            clip_id=event.get("clip_id", "unknown"),
            behavior_class=event.get("behavior_class", ""),
            policy_rule_ref=event.get("policy_rule_ref", ""),
            event_description=event.get("event_description", ""),
            severity=event.get("severity", "LOW"),
            escalation_action=event.get("escalation_action", "Logged to DB"),
            zone=event.get("zone", "Zone-1"),
            confidence=event.get("confidence", 0.0),
            frame_number=event.get("frame_number", 0),
            frame_path=event.get("frame_path", ""),
        )
    except Exception as e:
        print(f"[Escalation] DB log failed: {e}")


def _build_alert(event: dict) -> dict:
    """Build the alert payload for real-time notification."""
    return {
        "alert_id": event.get("event_id", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": event.get("severity"),
        "color": SEVERITY_COLORS.get(event.get("severity", "LOW"), "#ffffff"),
        "behavior_class": event.get("behavior_class", ""),
        "clip_id": event.get("clip_id", ""),
        "zone": event.get("zone", "Zone-1"),
        "policy_rule_ref": event.get("policy_rule_ref", ""),
        "event_description": event.get("event_description", ""),
        "confidence": event.get("confidence", 0.0),
        "frame_path": event.get("frame_path", ""),
    }


def _write_alert_json(alert: dict):
    """Append alert to outputs/alerts.jsonl for persistence."""
    alerts_file = ROOT / "outputs" / "alerts.jsonl"
    alerts_file.parent.mkdir(parents=True, exist_ok=True)
    with open(alerts_file, "a") as f:
        f.write(json.dumps(alert) + "\n")
