# src/detection/detection_engine.py
"""
Module 1 — Detection Engine
Ingests video clips, detects behavioral violations, produces structured records.

Detection approach:
  Primary:   YOLOv8 (ultralytics) — pre-trained on COCO, fine-tuned on factory dataset if available.
             Falls back to zero-shot CLIP-based classification per frame.
  Secondary: Anthropic Vision API for ambiguous/borderline cases (optional, costs API calls).

The 4 compliance classes (from policy_rules.py):
  0 — Safe Walkway Violation  (person outside green floor markings)
  1 — Unauthorized Intervention (person + equipment, no green vest)
  2 — Opened Panel Cover
  3 — Forklift Overload (3+ blocks)
"""

import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Generator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.policy_rules import COMPLIANCE_RULES, CLASS_ID_TO_RULE
from src.severity.severity_matrix import assign_severity_by_class_id
from src.escalation.escalation_pipeline import escalate
from src.reports.report_generator import generate_report

FRAMES_DIR = ROOT / "outputs" / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# ── Model loader ─────────────────────────────────────────────────────────────

def load_yolo_model(weights_path: str | None = None):
    """
    Load YOLOv8 model.
    If a custom weights path is provided (fine-tuned on factory dataset), use it.
    Otherwise falls back to yolov8n.pt (COCO pre-trained) + class remapping.
    """
    try:
        from ultralytics import YOLO
        if weights_path and Path(weights_path).exists():
            print(f"[Detection] Loading custom weights: {weights_path}")
            return YOLO(weights_path)
        else:
            print("[Detection] Loading YOLOv8n (COCO pre-trained)")
            return YOLO("yolov8n.pt")
    except Exception as e:
        print(f"[Detection] YOLO load failed: {e}")
        return None


# ── Frame-level detection ─────────────────────────────────────────────────────

def detect_frame_yolo(model, frame: np.ndarray, clip_id: str, frame_idx: int) -> list[dict]:
    """
    Run YOLO detection on a single frame.
    Maps COCO classes + heuristics to our 4 compliance classes.
    
    Returns list of violation dicts (may be empty).
    """
    results = model(frame, verbose=False, conf=0.35)
    violations = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        detected_labels = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
            detected_labels.append({
                "label": label,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "cls_id": cls_id,
            })

        # ── Rule-based violation mapping ──────────────────────────────────────
        # Uses COCO label names + spatial heuristics as proxies.

        person_boxes = [d for d in detected_labels if d["label"] == "person"]
        forklift_boxes = [d for d in detected_labels if d["label"] in ("forklift", "truck", "car")]

        h, w = frame.shape[:2]

        # Class 0 — Walkway violation: person in non-walkway zone
        # Heuristic: persons in left/right 20% of frame (outside center walkway)
        for p in person_boxes:
            px_center = (p["bbox"][0] + p["bbox"][2]) / 2
            if px_center < w * 0.15 or px_center > w * 0.85:
                violations.append(_make_violation(
                    class_id=0, conf=p["conf"],
                    clip_id=clip_id, frame_idx=frame_idx,
                    description=(
                        f"Person detected outside Designated Safe Walkway boundary "
                        f"(frame position x={px_center:.0f}/{w}). "
                        f"Observable indicator: position beyond green floor marking boundary."
                    ),
                    frame=frame, bbox=p["bbox"],
                ))

        # Class 2 — Opened panel cover
        # If no YOLO model with panel class, we use a specialized detector below
        # Placeholder: if dataset model detects class 'panel'
        for d in detected_labels:
            if "panel" in d["label"].lower() or "cabinet" in d["label"].lower():
                violations.append(_make_violation(
                    class_id=2, conf=d["conf"],
                    clip_id=clip_id, frame_idx=frame_idx,
                    description=(
                        f"Electrical panel cover detected in open position during production operations. "
                        f"Observable indicator: panel cover open state. Policy Section 5.2.2."
                    ),
                    frame=frame, bbox=d["bbox"],
                ))

        # Class 3 — Forklift overload: forklift detected with high load
        for f in forklift_boxes:
            if f["conf"] > 0.5:
                violations.append(_make_violation(
                    class_id=3, conf=f["conf"] * 0.7,  # uncertain without block count
                    clip_id=clip_id, frame_idx=frame_idx,
                    description=(
                        f"Forklift detected. Unable to confirm exact block count from COCO model — "
                        f"flagging for review. Observable indicator: block count on forks."
                    ),
                    frame=frame, bbox=f["bbox"],
                ))

    return violations


def detect_frame_clip(frame: np.ndarray, clip_id: str, frame_idx: int) -> list[dict]:
    """
    Zero-shot CLIP-based detection fallback.
    Classifies each frame against the 4 unsafe behavior descriptions.
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        model = _get_clip_model()
        if model is None:
            return []

        clip_model, clip_processor = model

        # Text prompts derived directly from policy observable indicators
        text_prompts = [
            "a person walking outside the green marked pedestrian walkway on a factory floor",  # 0
            "a person touching factory equipment without wearing a green safety vest",             # 1
            "an open electrical panel cover on a factory machine",                                # 2
            "a forklift carrying three or more large blocks as an overloaded stack",              # 3
            "normal safe factory floor activity with no violations",                              # safe
        ]

        pil_frame = _cv2_to_pil(frame)
        inputs = clip_processor(
            text=text_prompts,
            images=pil_frame,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).tolist()

        violations = []
        threshold = 0.30
        for class_id, prob in enumerate(probs[:4]):  # skip 'safe' class
            if prob >= threshold:
                rule_key = CLASS_ID_TO_RULE[class_id]
                rule = COMPLIANCE_RULES[rule_key]
                violations.append(_make_violation(
                    class_id=class_id, conf=prob,
                    clip_id=clip_id, frame_idx=frame_idx,
                    description=(
                        f"{rule['behavior_class']} detected via zero-shot classification "
                        f"(confidence {prob:.2%}). {rule['observable_indicator']}."
                    ),
                    frame=frame, bbox=None,
                ))

        return violations

    except Exception as e:
        print(f"[Detection] CLIP failed: {e}")
        return []


# ── Dataset-trained model detection ──────────────────────────────────────────

def detect_frame_dataset_model(model, frame: np.ndarray, clip_id: str, frame_idx: int) -> list[dict]:
    """
    Detection using a model trained/fine-tuned on the Kaggle factory dataset.
    Dataset classes match our 4 compliance classes directly.
    
    Expected dataset class mapping:
      0: walkway_violation OR safe_walkway
      1: unauthorized_intervention OR authorized_intervention
      2: opened_panel_cover OR closed_panel_cover
      3: forklift_overload OR safe_carrying
    
    NOTE: The Kaggle dataset label structure must be confirmed after download.
    This function handles both binary (safe/unsafe per class) and 4-class setups.
    """
    results = model(frame, verbose=False, conf=0.35)
    violations = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names.get(cls_id, str(cls_id)).lower()
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]

            # Map dataset labels to compliance class IDs
            compliance_class = _map_dataset_label_to_class(label, cls_id)
            if compliance_class is None:
                continue  # safe behavior — no violation

            rule_key = CLASS_ID_TO_RULE.get(compliance_class)
            if not rule_key:
                continue

            rule = COMPLIANCE_RULES[rule_key]
            violations.append(_make_violation(
                class_id=compliance_class,
                conf=conf,
                clip_id=clip_id,
                frame_idx=frame_idx,
                description=(
                    f"{rule['behavior_class']} detected (label='{label}', conf={conf:.2%}). "
                    f"{rule['observable_indicator']}. Policy: {rule['policy_section']}."
                ),
                frame=frame,
                bbox=(x1, y1, x2, y2),
            ))

    return violations


def _map_dataset_label_to_class(label: str, cls_id: int) -> int | None:
    """
    Map dataset label string to compliance class ID (0-3), or None if safe.
    Adjust this mapping after inspecting the actual Kaggle dataset labels.
    """
    UNSAFE_KEYWORDS = {
        "walkway_violation": 0,
        "safe_walkway_violation": 0,
        "violation": 0,
        "unauthorized": 1,
        "unauthorized_intervention": 1,
        "intervention": 1,
        "open_panel": 2,
        "opened_panel": 2,
        "panel_open": 2,
        "overload": 3,
        "forklift_overload": 3,
        "carrying_overload": 3,
    }

    for keyword, class_id in UNSAFE_KEYWORDS.items():
        if keyword in label:
            return class_id

    # If label is a raw integer from a 4-class dataset (unsafe classes only)
    if cls_id in (0, 1, 2, 3) and "safe" not in label:
        return cls_id

    return None  # safe behavior


# ── Main clip processing pipeline ─────────────────────────────────────────────

def process_clip(
    video_path: str,
    model=None,
    use_clip_fallback: bool = True,
    sample_every_n_frames: int = 15,
    zone: str = "Zone-1",
) -> list[dict]:
    """
    Process a single video clip end-to-end.
    Returns list of violation event dicts (fully processed through severity + escalation + report).
    """
    video_path = Path(video_path)
    clip_id = video_path.stem
    print(f"\n[Detection] Processing clip: {clip_id}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Detection] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Detection] {total_frames} frames @ {fps:.1f} fps")

    all_violations = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            # Run detection
            violations = []

            if model is not None:
                violations = detect_frame_dataset_model(model, frame, clip_id, frame_idx)
                if not violations:
                    violations = detect_frame_yolo(model, frame, clip_id, frame_idx)

            if not violations and use_clip_fallback:
                violations = detect_frame_clip(frame, clip_id, frame_idx)

            for v in violations:
                v["zone"] = zone
                # Assign severity
                sev = assign_severity_by_class_id(
                    v["class_id"],
                    context={"confidence": v["confidence"]},
                )
                v.update(sev)

                # Generate report
                report = generate_report(
                    clip_id=clip_id,
                    behavior_class=v["behavior_class"],
                    policy_rule_ref=v["policy_rule_ref"],
                    event_description=v["event_description"],
                    severity=v["severity"],
                    escalation_action=v["escalation_action"],
                    zone=zone,
                    confidence=v["confidence"],
                    frame_number=frame_idx,
                    frame_path=v.get("frame_path", ""),
                )
                v["event_id"] = report["event_id"]

                # Escalation
                escalate(v)
                all_violations.append(v)

        frame_idx += 1

    cap.release()
    print(f"[Detection] {len(all_violations)} violations found in {clip_id}")
    return all_violations


def process_directory(
    clips_dir: str,
    model=None,
    zone: str = "Zone-1",
    extensions: tuple = (".mp4", ".avi", ".mov", ".mkv"),
) -> list[dict]:
    """Process all video clips in a directory."""
    clips_dir = Path(clips_dir)
    clips = [p for p in clips_dir.iterdir() if p.suffix.lower() in extensions]
    print(f"[Detection] Found {len(clips)} clips in {clips_dir}")

    all_events = []
    for clip_path in sorted(clips):
        events = process_clip(str(clip_path), model=model, zone=zone)
        all_events.extend(events)

    print(f"\n[Detection] Total violations detected: {len(all_events)}")
    return all_events


# ── Helpers ───────────────────────────────────────────────────────────────────

_clip_model_cache = None

def _get_clip_model():
    global _clip_model_cache
    if _clip_model_cache is not None:
        return _clip_model_cache
    try:
        from transformers import CLIPProcessor, CLIPModel
        print("[Detection] Loading CLIP model (openai/clip-vit-base-patch32)...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model_cache = (clip_model, clip_processor)
        return _clip_model_cache
    except Exception as e:
        print(f"[Detection] CLIP load failed: {e}")
        return None


def _cv2_to_pil(frame: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _make_violation(
    class_id: int,
    conf: float,
    clip_id: str,
    frame_idx: int,
    description: str,
    frame: np.ndarray,
    bbox: tuple | None,
) -> dict:
    """Build a violation dict and save annotated frame image."""
    rule_key = CLASS_ID_TO_RULE.get(class_id, "walkway_violation")
    rule = COMPLIANCE_RULES[rule_key]

    # Save annotated frame
    frame_path = _save_frame(frame, clip_id, frame_idx, class_id, bbox)

    return {
        "event_id": str(uuid.uuid4()),
        "class_id": class_id,
        "behavior_class": rule["behavior_class"],
        "policy_rule_ref": rule["policy_section"],
        "event_description": description,
        "confidence": round(conf, 4),
        "frame_number": frame_idx,
        "frame_path": str(frame_path),
        "zone": "Zone-1",
        # severity/escalation filled in by caller
    }


def _save_frame(
    frame: np.ndarray,
    clip_id: str,
    frame_idx: int,
    class_id: int,
    bbox: tuple | None,
) -> Path:
    """Save annotated frame to outputs/frames/."""
    annotated = frame.copy()

    if bbox:
        x1, y1, x2, y2 = bbox
        rule_key = CLASS_ID_TO_RULE.get(class_id, "walkway_violation")
        label = COMPLIANCE_RULES[rule_key]["behavior_class"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, label[:30], (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    filename = f"{clip_id}_frame{frame_idx:05d}_class{class_id}.jpg"
    path = FRAMES_DIR / filename
    cv2.imwrite(str(path), annotated)
    return path
