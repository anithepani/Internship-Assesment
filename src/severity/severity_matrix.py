# src/severity/severity_matrix.py
"""
Module 2 — Severity Categorization Matrix
Assigns risk tier to each detected violation based on policy signals.

Severity mapping derived from OHS policy document:
- CRITICAL: Safe Walkway Violation — highest-frequency, WARNING callout, Section 3.3.2
- HIGH:     Unauthorized Intervention — CRITICAL SAFETY NOTICE, Section 4.3.2
- HIGH:     Forklift Overload — CRITICAL SAFETY NOTICE, Section 6.3.2
- MEDIUM:   Opened Panel Cover — WARNING callout, state-based, Section 5.2.2
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.policy_rules import COMPLIANCE_RULES, ESCALATION_ROUTING


# ── Core severity assignment ────────────────────────────────────────────────

def assign_severity(rule_key: str, context: dict | None = None) -> dict:
    """
    Assign a severity tier to a detected violation.

    Args:
        rule_key:  Key in COMPLIANCE_RULES (e.g. 'walkway_violation')
        context:   Optional dict with detection context:
                   - personnel_in_frame: bool
                   - distance_to_hazard: float (0-1, 1=closest)
                   - confidence: float (detection confidence)

    Returns dict with:
        severity, tier_label, rationale, escalation_action, realtime_alert
    """
    rule = COMPLIANCE_RULES.get(rule_key)
    if not rule:
        return _default_severity()

    base_severity = rule["severity"]

    # Context-based upgrade/downgrade (optional refinement)
    if context:
        base_severity = _apply_context_modifiers(base_severity, rule_key, context)

    routing = ESCALATION_ROUTING[base_severity]

    return {
        "severity": base_severity,
        "tier_label": _tier_label(base_severity),
        "rationale": rule["severity_rationale"],
        "escalation_action": routing["action_label"],
        "realtime_alert": routing["realtime_alert"],
        "db_log": routing["db_log"],
    }


def assign_severity_by_class_id(class_id: int, context: dict | None = None) -> dict:
    """Convenience wrapper — takes integer class ID (0-3)."""
    from config.policy_rules import CLASS_ID_TO_RULE
    rule_key = CLASS_ID_TO_RULE.get(class_id)
    if not rule_key:
        return _default_severity()
    return assign_severity(rule_key, context)


# ── Context modifiers ───────────────────────────────────────────────────────

def _apply_context_modifiers(severity: str, rule_key: str, context: dict) -> str:
    """
    Adjust severity based on detection context.
    Policy principle: personnel proximity escalates risk.
    """
    TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def upgrade(s):
        idx = TIERS.index(s)
        return TIERS[min(idx + 1, len(TIERS) - 1)]

    def downgrade(s):
        idx = TIERS.index(s)
        return TIERS[max(idx - 1, 0)]

    # Panel cover: upgrade if personnel are confirmed in frame near panel
    if rule_key == "opened_panel_cover":
        if context.get("personnel_in_frame") and context.get("distance_to_hazard", 0) > 0.6:
            severity = upgrade(severity)  # MEDIUM → HIGH

    # Forklift: downgrade to MEDIUM if very low confidence (uncertain block count)
    if rule_key == "forklift_overload":
        if context.get("confidence", 1.0) < 0.4:
            severity = downgrade(severity)

    return severity


# ── Helpers ─────────────────────────────────────────────────────────────────

def _tier_label(severity: str) -> str:
    labels = {
        "LOW":      "LOW RISK",
        "MEDIUM":   "MEDIUM RISK",
        "HIGH":     "HIGH RISK",
        "CRITICAL": "CRITICAL RISK",
    }
    return labels.get(severity, severity)


def _default_severity() -> dict:
    return {
        "severity": "MEDIUM",
        "tier_label": "MEDIUM RISK",
        "rationale": "Unknown rule — defaulting to MEDIUM.",
        "escalation_action": "Logged to DB",
        "realtime_alert": False,
        "db_log": True,
    }


# ── Severity summary for multiple events ────────────────────────────────────

def aggregate_severity(severities: list[str]) -> str:
    """Return the highest severity from a list (for multi-violation clips)."""
    TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if not severities:
        return "LOW"
    return max(severities, key=lambda s: TIERS.index(s) if s in TIERS else 0)


if __name__ == "__main__":
    for rule_key in ["walkway_violation", "unauthorized_intervention", "opened_panel_cover", "forklift_overload"]:
        result = assign_severity(rule_key)
        print(f"\n{rule_key}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
