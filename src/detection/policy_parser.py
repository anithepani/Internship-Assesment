# src/detection/policy_parser.py
"""
Module 1 (support) — Policy Parser
Parses the OHS compliance policy PDF and extracts structured rules.
Uses LLM (Anthropic Claude) for extraction; falls back to config/policy_rules.py.
"""

import json
import os
import sys
from pathlib import Path

# ── Add project root to path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def extract_rules_from_pdf(pdf_path: str) -> dict:
    """
    Extract compliance rules from the OHS policy PDF using Claude.
    Returns a dict matching the structure of config/policy_rules.py.
    Falls back to static config if extraction fails.
    """
    try:
        import anthropic
        import fitz  # PyMuPDF

        # Read PDF text
        doc = fitz.open(pdf_path)
        full_text = "\n\n".join(page.get_text() for page in doc)
        doc.close()

        client = anthropic.Anthropic()

        prompt = f"""You are parsing a factory OHS compliance policy document.
Extract the 4 behavioral compliance rules defined in this document.

For each rule return a JSON object with exactly these fields:
- class_id: integer (0-3, in order: walkway, intervention, panel, forklift)
- behavior_class: the name of the UNSAFE behavior
- safe_behavior: the name of the SAFE/compliant behavior
- domain: operational domain name
- policy_section: section reference (e.g. "Section 3.3.2")
- observable_indicator: what a camera would observe to detect the unsafe behavior
- hazard_context: one sentence describing the physical hazard
- severity: one of LOW / MEDIUM / HIGH / CRITICAL
- severity_rationale: why this severity was assigned based on policy language
- alert_callout: WARNING or CRITICAL SAFETY NOTICE (from the policy boxes)

Severity mapping rule from the assessment:
- CRITICAL: highest-frequency behavior explicitly flagged in a WARNING callout
- HIGH: CRITICAL SAFETY NOTICE in the policy, active personnel risk
- MEDIUM: WARNING callout, state-based condition (no confirmed personnel exposure)
- LOW: (none in this policy, but include if applicable)

Return ONLY a JSON array of 4 rule objects. No prose, no markdown fences.

POLICY TEXT:
{full_text[:12000]}
"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        rules_list = json.loads(raw)

        # Convert list → dict keyed by rule name
        key_map = {
            0: "walkway_violation",
            1: "unauthorized_intervention",
            2: "opened_panel_cover",
            3: "forklift_overload",
        }
        rules = {}
        for r in rules_list:
            key = key_map.get(r.get("class_id", -1))
            if key:
                rules[key] = r

        print(f"[PolicyParser] Extracted {len(rules)} rules from PDF via LLM.")
        return rules

    except Exception as e:
        print(f"[PolicyParser] LLM extraction failed ({e}), using static config.")
        from config.policy_rules import COMPLIANCE_RULES
        return COMPLIANCE_RULES


def get_rules() -> dict:
    """
    Return compliance rules. Tries PDF extraction first, falls back to static config.
    """
    pdf_candidates = [
        ROOT / "compliance_policy.pdf",
        ROOT / "data" / "compliance_policy.pdf",
        Path("/mnt/user-data/uploads/Compliance_Policy_Manual.pdf"),
    ]
    for p in pdf_candidates:
        if p.exists():
            return extract_rules_from_pdf(str(p))

    print("[PolicyParser] No PDF found — using static config.")
    from config.policy_rules import COMPLIANCE_RULES
    return COMPLIANCE_RULES


if __name__ == "__main__":
    rules = get_rules()
    for k, v in rules.items():
        print(f"\n{'='*60}")
        print(f"Rule: {k}")
        for field, val in v.items():
            print(f"  {field}: {val}")
