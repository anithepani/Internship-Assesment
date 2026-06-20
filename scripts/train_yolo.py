#!/usr/bin/env python3
# scripts/train_yolo.py
"""
Fine-tune YOLOv8 on the Kaggle factory safety dataset.
Run AFTER downloading the dataset with scripts/setup_dataset.py.

This script:
1. Prepares a data.yaml for the factory classes
2. Fine-tunes YOLOv8n on the dataset (fast, good baseline)
3. Saves weights to models/factory_yolo.pt

Usage:
    python scripts/train_yolo.py --epochs 30 --img 640 --batch 16
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def prepare_data_yaml(clips_dir: Path) -> Path:
    """
    Create a YOLO-compatible data.yaml for the factory dataset.
    Adjust class names after inspecting the actual dataset structure.
    """
    import yaml

    # These class names correspond to the 4 unsafe behaviors in the policy.
    # If the dataset uses different names, update this list.
    class_names = [
        "walkway_violation",         # class 0
        "unauthorized_intervention", # class 1
        "opened_panel_cover",        # class 2
        "forklift_overload",         # class 3
    ]

    # Look for existing train/val splits
    train_dir = clips_dir / "train" / "images"
    val_dir = clips_dir / "valid" / "images"

    if not train_dir.exists():
        train_dir = clips_dir / "images" / "train"
        val_dir = clips_dir / "images" / "val"

    if not train_dir.exists():
        # Fall back to single directory
        train_dir = clips_dir
        val_dir = clips_dir

    data = {
        "path": str(clips_dir),
        "train": str(train_dir),
        "val": str(val_dir),
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = DATA_DIR / "factory_data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[Train] data.yaml written to {yaml_path}")
    print(f"[Train] Classes: {class_names}")
    return yaml_path


def train(epochs: int = 30, img_size: int = 640, batch: int = 16, device: str = "auto"):
    """Fine-tune YOLOv8n on factory dataset."""
    from ultralytics import YOLO

    clips_dir = DATA_DIR / "clips"
    if not clips_dir.exists() or not any(clips_dir.iterdir()):
        print("[Train] [FAIL] No dataset found. Run scripts/setup_dataset.py first.")
        return

    yaml_path = prepare_data_yaml(clips_dir)

    print(f"[Train] Starting YOLOv8n fine-tuning — {epochs} epochs, img={img_size}, batch={batch}")
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=device,
        project=str(ROOT / "models"),
        name="factory_yolo",
        save=True,
        patience=10,
        workers=4,
        pretrained=True,
        verbose=True,
    )

    best_weights = ROOT / "models" / "factory_yolo" / "weights" / "best.pt"
    if best_weights.exists():
        import shutil
        dest = MODELS_DIR / "factory_best.pt"
        shutil.copy(best_weights, dest)
        print(f"\n[Train] [OK] Best weights saved to {dest}")
        print(f"[Train] Use with: python main.py --mode detect --weights {dest}")
    else:
        print("[Train] [WARN] Could not find best.pt after training.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    args = parser.parse_args()
    train(args.epochs, args.img, args.batch, args.device)
