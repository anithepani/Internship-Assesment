# Factory Compliance & Alert Escalation System
> **KMP-OHS-POL-001** | Intern Assessment Submission

End-to-end automated factory safety compliance system: video ingestion → behavioral violation detection → severity classification → escalation routing → audit reports → live dashboard.

---

## Quick Start (< 5 minutes to running dashboard)

```bash
# 1. Clone / unzip to your project directory
cd factory-compliance-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run demo mode (no videos needed — synthetic events)
python main.py --mode demo

# 5. Launch dashboard
python main.py --mode dashboard
# Open http://localhost:8501
```

---

## Getting Real Video Data (Kaggle Dataset)

```bash
# Set up Kaggle API key first:
#   https://www.kaggle.com/settings → API → Create New Token → download kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset (10 GB — takes time)
python scripts/setup_dataset.py

# Run detection on downloaded clips
python main.py --mode detect --clips data/clips/

# Or run detection + dashboard in one command
python main.py --mode full
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                  │
│  Video Clips (.mp4/.avi)    +    Policy PDF (KMP-OHS-POL-001)  │
└─────────────────┬──────────────────────┬───────────────────────┘
                  │                      │
                  ▼                      ▼
     ┌────────────────────┐   ┌──────────────────────┐
     │  Module 1          │   │  Policy Parser        │
     │  Detection Engine  │◄──│  (LLM extraction)     │
     │  (YOLOv8 + CLIP)  │   │  config/policy_rules  │
     └────────────┬───────┘   └──────────────────────┘
                  │ violation events
                  ▼
     ┌────────────────────┐
     │  Module 2          │
     │  Severity Matrix   │  LOW / MEDIUM / HIGH / CRITICAL
     └────────────┬───────┘
                  │
                  ▼
     ┌────────────────────┐
     │  Module 3          │
     │  Escalation        │  LOW/MED → DB log
     │  Pipeline          │  HIGH/CRIT → alert + DB log
     └──────┬──────┬──────┘
            │      │
            ▼      ▼
  ┌──────────┐  ┌──────────────────┐
  │ Module 4 │  │  Alert Queue     │
  │ Reports  │  │  (in-process     │
  │ JSON/CSV │  │   pub/sub)       │
  │ SQLite   │  └──────────┬───────┘
  └──────────┘             │
                            ▼
              ┌─────────────────────────┐
              │  Module 5: Dashboard    │
              │  View A: Live Feed      │
              │  View B: Alert Timeline │
              │  View C: Log + Export   │
              └─────────────────────────┘
```

---

## Module Overview

| Module | File | Function |
|--------|------|----------|
| 1 — Detection Engine | `src/detection/detection_engine.py` | Frame-by-frame violation detection (YOLOv8 + CLIP zero-shot) |
| 1 — Policy Parser | `src/detection/policy_parser.py` | Extracts compliance rules from PDF via LLM |
| 2 — Severity Matrix | `src/severity/severity_matrix.py` | Assigns LOW/MED/HIGH/CRIT to each violation |
| 3 — Escalation Pipeline | `src/escalation/escalation_pipeline.py` | Routes events to DB log and/or real-time alert |
| 4 — Report Generator | `src/reports/report_generator.py` | Writes JSON, CSV, and SQLite records |
| 4 — Database | `src/reports/database.py` | SQLite ORM via SQLAlchemy |
| 5 — Dashboard | `src/dashboard/app.py` | Streamlit GUI (live feed, alert stream, export) |

---

## Policy Parsing Approach

Rules are extracted from `compliance_policy.pdf` (KMP-OHS-POL-001) using **Anthropic Claude** (claude-sonnet-4-20250514) via the Anthropic API. The parser sends the full PDF text and asks Claude to extract structured rule objects — behavior class, observable indicator, hazard context, and severity.

A **static fallback** (`config/policy_rules.py`) is pre-populated from manual review of the policy document so the system works without API access.

**Verification approach:** The extracted rules are cross-checked against the static config at runtime. Any field discrepancy is logged for human review.

---

## Severity Mapping Rationale

Derived from the policy document's language and callout boxes:

| Behavior Class | Severity | Policy Signal |
|---|---|---|
| Safe Walkway Violation | **CRITICAL** | Section 3.3.2 WARNING — "highest-frequency unsafe behavior"; explicitly called out for recurrence risk |
| Unauthorized Intervention | **HIGH** | Section 4.3.2 CRITICAL SAFETY NOTICE — direct active personnel-equipment contact |
| Opened Panel Cover | **MEDIUM** | Section 5.2.2 WARNING — state-based condition; no confirmed concurrent personnel exposure |
| Forklift Overload | **HIGH** | Section 6.3.2 CRITICAL SAFETY NOTICE — vehicle instability risk with potential personnel in vicinity |

---

## Detection Approach

**Primary (with dataset model):** YOLOv8 fine-tuned on the Kaggle factory dataset. Class labels map directly to the 4 policy compliance categories.

**Primary (without fine-tuned model):** YOLOv8n (COCO pre-trained) + heuristic rule engine. COCO classes (person, truck, etc.) are mapped to compliance classes via spatial reasoning and label proximity.

**Fallback:** OpenAI CLIP zero-shot classification. Frame is classified against natural-language descriptions of each unsafe behavior derived from the policy's observable indicators.

**Optional (ambiguous cases):** Anthropic Vision API for borderline detections (e.g., uncertain block count on forklift).

**Known limitations:**
- COCO-pretrained YOLO cannot distinguish green vs. red-black safety vests → Unauthorized Intervention recall is lower without fine-tuning
- Block counting for forklift overload requires fine-tuned model or CLIP
- Panel cover state detection works best with fine-tuned model
- CLIP zero-shot runs ~3-5 fps on CPU; use GPU for real-time performance

---

## Training Your Own Model

```bash
# After downloading the dataset:
python scripts/train_yolo.py --epochs 50 --img 640 --batch 16 --device cuda

# Use trained model for detection:
python main.py --mode detect --weights models/factory_best.pt
```

---

## File Structure

```
factory-compliance-system/
├── README.md
├── main.py                         # Entry point
├── requirements.txt
├── compliance_policy.pdf           # KMP-OHS-POL-001 (copy here)
├── config/
│   └── policy_rules.py             # Extracted compliance rules (static fallback)
├── data/
│   └── clips/                      # Video clips go here
├── src/
│   ├── detection/
│   │   ├── detection_engine.py     # Module 1
│   │   └── policy_parser.py        # LLM-based rule extraction
│   ├── severity/
│   │   └── severity_matrix.py      # Module 2
│   ├── escalation/
│   │   └── escalation_pipeline.py  # Module 3
│   ├── reports/
│   │   ├── report_generator.py     # Module 4
│   │   └── database.py             # SQLite ORM
│   └── dashboard/
│       └── app.py                  # Module 5 (Streamlit)
├── scripts/
│   ├── setup_dataset.py            # Kaggle dataset download
│   └── train_yolo.py               # YOLOv8 fine-tuning
├── models/                         # Trained weights (created on train)
└── outputs/
    ├── compliance.db               # SQLite database
    ├── compliance_audit.csv        # Append-only audit log
    ├── compliance_audit.jsonl      # Append-only JSON log
    ├── alerts.jsonl                # Real-time alert log
    ├── frames/                     # Annotated violation frame images
    └── reports/                    # Per-event JSON reports
```

---

## Environment Variables (optional)

```bash
# For LLM-based policy parsing
export ANTHROPIC_API_KEY=sk-ant-...

# For Kaggle download
# Uses ~/.kaggle/kaggle.json automatically
```

---

## Running Modes

```bash
python main.py --mode demo           # Generate synthetic events (no video needed)
python main.py --mode dashboard      # Launch Streamlit dashboard only
python main.py --mode detect         # Run detection on data/clips/
python main.py --mode detect --clips /path/to/clips --weights models/factory_best.pt
python main.py --mode full           # detect + launch dashboard
python main.py --mode setup-kaggle   # Download Kaggle dataset
```
