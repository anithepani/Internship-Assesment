#!/usr/bin/env python3
# main.py
"""
Factory Compliance & Alert Escalation System — Main Entry Point
KMP-OHS-POL-001

Usage:
    python main.py --mode detect   # Run detection on clips, log results
    python main.py --mode dashboard # Launch Streamlit dashboard
    python main.py --mode full     # Run detection then launch dashboard
    python main.py --mode test     # Run on sample/demo data to verify pipeline

Dataset: https://www.kaggle.com/datasets/trnhhnggiang/video-dataset-for-safe-and-unsafe-behaviours
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "clips"
OUTPUTS_DIR = ROOT / "outputs"


def run_detection(clips_dir: str, model_weights: str | None = None, zone: str = "Zone-1"):
    """Run the detection pipeline on all clips in a directory."""
    from src.reports.database import init_db
    from src.detection.detection_engine import process_directory, load_yolo_model

    init_db()
    print(f"[Main] Loading detection model...")
    model = load_yolo_model(model_weights)

    print(f"[Main] Processing clips in: {clips_dir}")
    events = process_directory(clips_dir, model=model, zone=zone)
    print(f"\n[Main] [OK] Detection complete. {len(events)} violations recorded.")
    print(f"[Main] Reports: {OUTPUTS_DIR / 'reports'}")
    print(f"[Main] Audit CSV: {OUTPUTS_DIR / 'compliance_audit.csv'}")
    print(f"[Main] Database: {OUTPUTS_DIR / 'compliance.db'}")
    return events


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_path = ROOT / "src" / "dashboard" / "app.py"
    print(f"[Main] Launching dashboard at http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ])


def run_demo():
    """
    Generate synthetic demo events to populate the dashboard without real video.
    Useful for testing the pipeline end-to-end.
    """
    import uuid
    from datetime import datetime, timezone, timedelta
    from src.reports.database import init_db
    from src.reports.report_generator import generate_report
    from src.escalation.escalation_pipeline import escalate
    from config.policy_rules import COMPLIANCE_RULES, CLASS_ID_TO_RULE
    from src.severity.severity_matrix import assign_severity_by_class_id

    init_db()
    print("[Demo] Generating synthetic violation events...")

    demo_events = [
        # CRITICAL: Walkway violation
        dict(class_id=0, clip_id="demo_clip_001", confidence=0.87, frame_number=45, zone="Zone-1"),
        dict(class_id=0, clip_id="demo_clip_002", confidence=0.72, frame_number=120, zone="Zone-2"),
        dict(class_id=0, clip_id="demo_clip_003", confidence=0.91, frame_number=200, zone="Zone-1"),
        # HIGH: Unauthorized intervention
        dict(class_id=1, clip_id="demo_clip_004", confidence=0.68, frame_number=80, zone="Zone-2"),
        dict(class_id=1, clip_id="demo_clip_005", confidence=0.55, frame_number=300, zone="Zone-1"),
        # MEDIUM: Open panel
        dict(class_id=2, clip_id="demo_clip_006", confidence=0.79, frame_number=15, zone="Zone-3"),
        dict(class_id=2, clip_id="demo_clip_007", confidence=0.82, frame_number=60, zone="Zone-3"),
        # HIGH: Forklift overload
        dict(class_id=3, clip_id="demo_clip_008", confidence=0.61, frame_number=250, zone="Zone-2"),
        dict(class_id=3, clip_id="demo_clip_009", confidence=0.74, frame_number=180, zone="Zone-2"),
    ]

    for i, demo in enumerate(demo_events):
        class_id = demo["class_id"]
        rule_key = CLASS_ID_TO_RULE[class_id]
        rule = COMPLIANCE_RULES[rule_key]

        sev = assign_severity_by_class_id(class_id, context={"confidence": demo["confidence"]})

        description = (
            f"[DEMO] {rule['behavior_class']} detected in {demo['clip_id']}. "
            f"{rule['observable_indicator']}. "
            f"Confidence: {demo['confidence']:.0%}."
        )

        event = {
            "event_id": str(uuid.uuid4()),
            "class_id": class_id,
            "behavior_class": rule["behavior_class"],
            "policy_rule_ref": rule["policy_section"],
            "event_description": description,
            "severity": sev["severity"],
            "escalation_action": sev["escalation_action"],
            "clip_id": demo["clip_id"],
            "zone": demo["zone"],
            "confidence": demo["confidence"],
            "frame_number": demo["frame_number"],
            "frame_path": "",
        }

        report = generate_report(**{k: v for k, v in event.items() if k != "event_id" and k != "class_id"})
        escalate(event)
        print(f"  [{sev['severity']:8s}] {rule['behavior_class']} -- {demo['clip_id']}")

    print(f"\n[Demo] [OK] {len(demo_events)} demo events generated.")
    print("[Demo] Launch dashboard to view: python main.py --mode dashboard")


def setup_kaggle_dataset():
    """Download the Kaggle dataset if kaggle CLI is configured."""
    dataset = "trnhhnggiang/video-dataset-for-safe-and-unsafe-behaviours"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Downloading Kaggle dataset: {dataset}")
    print(f"[Setup] This is ~10 GB. Downloading to {DATA_DIR}...")
    result = subprocess.run([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", dataset, "-p", str(DATA_DIR), "--unzip"
    ])
    if result.returncode == 0:
        print("[Setup] [OK] Dataset downloaded.")
    else:
        print("[Setup] [FAIL] Download failed. Ensure kaggle.json is at ~/.kaggle/kaggle.json")
        print("[Setup] Get your API key at: https://www.kaggle.com/account")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factory Compliance & Alert Escalation System")
    parser.add_argument(
        "--mode",
        choices=["detect", "dashboard", "full", "demo", "setup-kaggle"],
        default="demo",
        help="Execution mode",
    )
    parser.add_argument("--clips", default=str(DATA_DIR), help="Path to video clips directory")
    parser.add_argument("--weights", default=None, help="Path to custom YOLO weights file")
    parser.add_argument("--zone", default="Zone-1", help="Facility zone label")

    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "detect":
        run_detection(args.clips, args.weights, args.zone)

    elif args.mode == "dashboard":
        launch_dashboard()

    elif args.mode == "full":
        run_detection(args.clips, args.weights, args.zone)
        launch_dashboard()

    elif args.mode == "demo":
        run_demo()
        print("\nRun 'python main.py --mode dashboard' to view results.")

    elif args.mode == "setup-kaggle":
        setup_kaggle_dataset()
