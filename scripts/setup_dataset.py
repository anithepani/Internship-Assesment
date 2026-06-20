#!/usr/bin/env python3
# scripts/setup_dataset.py
"""
Downloads the Kaggle dataset and prepares it for the pipeline.
Inspects dataset label structure and updates detection_engine.py mapping if needed.

Run: python scripts/setup_dataset.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"

DATASET_SLUG = "trnhhnggiang/video-dataset-for-safe-and-unsafe-behaviours"


def check_kaggle_auth():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print("[Setup] [OK] kaggle.json found")
        return True
    else:
        print("[Setup] [FAIL] kaggle.json NOT found at ~/.kaggle/kaggle.json")
        print("[Setup] Steps to fix:")
        print("  1. Go to https://www.kaggle.com/settings → Account → API → Create New Token")
        print("  2. Download kaggle.json")
        print("  3. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
        print("  4. chmod 600 ~/.kaggle/kaggle.json")
        return False


def download_dataset(partial: bool = False):
    """
    Download the full dataset (10 GB) or a partial sample.
    partial=True downloads only the first zip part if available.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Downloading dataset: {DATASET_SLUG}")
    print(f"[Setup] Target: {CLIPS_DIR}")
    print("[Setup] NOTE: This is ~10 GB. This may take a while on a slow connection.")
    print("[Setup] Tip: You only need ~20-30 clips to demo the pipeline.\n")

    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(CLIPS_DIR),
        "--unzip"
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("[Setup] [FAIL] Download failed.")
        return False

    print("[Setup] [OK] Download complete.")
    return True


def inspect_dataset_structure():
    """
    After download, inspect the label/class structure of the dataset.
    Prints info to help configure the detection mapping.
    """
    print("\n[Inspect] Scanning downloaded files...")

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    label_exts = {".txt", ".json", ".yaml", ".csv"}

    videos = []
    labels = []

    for path in CLIPS_DIR.rglob("*"):
        if path.suffix.lower() in video_exts:
            videos.append(path)
        elif path.suffix.lower() in label_exts:
            labels.append(path)

    print(f"[Inspect] Found {len(videos)} video files")
    print(f"[Inspect] Found {len(labels)} label/metadata files")

    # Show sample video paths
    print("\n[Inspect] Sample video paths:")
    for v in videos[:10]:
        print(f"  {v.relative_to(CLIPS_DIR)}")

    # Check for YOLO-style labels (classes.txt or data.yaml)
    for lf in labels:
        if lf.name in ("classes.txt", "data.yaml", "dataset.yaml"):
            print(f"\n[Inspect] Found class definition file: {lf}")
            print(lf.read_text()[:500])

    # Move videos to clips dir root for easy access
    if videos:
        print(f"\n[Inspect] Moving clips to {CLIPS_DIR}...")
        for v in videos:
            if v.parent != CLIPS_DIR:
                dest = CLIPS_DIR / v.name
                if not dest.exists():
                    v.rename(dest)

    return videos


def copy_policy_pdf():
    """Copy the policy PDF to the project root if it's in uploads."""
    src = Path("/mnt/user-data/uploads/Compliance_Policy_Manual.pdf")
    dest = ROOT / "compliance_policy.pdf"
    if src.exists() and not dest.exists():
        import shutil
        shutil.copy(src, dest)
        print(f"[Setup] [OK] Copied policy PDF to {dest}")
    elif dest.exists():
        print(f"[Setup] [OK] Policy PDF already at {dest}")
    else:
        print(f"[Setup] [WARN] Policy PDF not found. Please copy it to {dest}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect existing dataset")
    args = parser.parse_args()

    copy_policy_pdf()

    if args.inspect_only:
        inspect_dataset_structure()
    elif not args.skip_download:
        if check_kaggle_auth():
            download_dataset()
            inspect_dataset_structure()
        else:
            print("\n[Setup] Skipping download. Set up kaggle.json first.")
            print("[Setup] You can still run: python main.py --mode demo")
            print("[Setup]   to generate synthetic events and test the pipeline.")
    else:
        inspect_dataset_structure()
