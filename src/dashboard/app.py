# src/dashboard/app.py
"""
Module 5 — Operations Dashboard
Streamlit app providing:
  View A — Live Feed Monitor (video clips + compliance overlays + alerts)
  View B — Alert Timeline Stream (real-time event feed)
  View C — Historical Log & Export (filterable, CSV/JSON download)
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.policy_rules import SEVERITY_COLORS, COMPLIANCE_RULES
from src.reports.database import get_all_events, init_db
from src.reports.report_generator import load_audit_csv
from src.escalation.escalation_pipeline import get_pending_alerts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Factory Compliance System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 3 seconds for live feed
st_autorefresh(interval=3000, key="refresh")

init_db()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/factory.png", width=80)
    st.title("Factory Compliance")
    st.caption("KMP-OHS-POL-001 Monitor")
    st.divider()

    view = st.radio(
        "Navigation",
        ["📹 Live Feed Monitor", "🚨 Alert Timeline", "📋 Historical Log & Export"],
        index=0,
    )
    st.divider()

    clips_dir = st.text_input("Clips directory", value=str(ROOT / "data" / "clips"))
    run_detection = st.button(
        "▶ Run Detection on Clips", type="primary", use_container_width=True
    )

    st.divider()
    st.markdown("**Policy Reference**")
    for key, rule in COMPLIANCE_RULES.items():
        color = SEVERITY_COLORS[rule["severity"]]
        st.markdown(
            f'<span style="color:{color}; font-size:12px">●</span> '
            f"**{rule['behavior_class']}** — `{rule['severity']}`",
            unsafe_allow_html=True,
        )

# ── Detection trigger ──────────────────────────────────────────────────────────
if run_detection:
    st.toast("Starting detection pipeline...", icon="🔄")
    with st.spinner("Running detection on clips..."):
        try:
            from src.detection.detection_engine import (
                process_directory,
                load_yolo_model,
            )

            model = load_yolo_model()
            clips_path = Path(clips_dir)
            if clips_path.exists():
                process_directory(str(clips_path), model=model)
                st.toast("Detection complete!", icon="✅")
            else:
                st.error(f"Clips directory not found: {clips_dir}")
        except Exception as e:
            st.error(f"Detection failed: {e}")

# ── View A — Live Feed Monitor ────────────────────────────────────────────────
if view == "📹 Live Feed Monitor":
    st.header("📹 Live Feed Monitor")
    st.caption("Compliance status indicators updated every 3 seconds")

    # Check for pending real-time alerts
    pending_alerts = get_pending_alerts()
    for alert in pending_alerts:
        color = alert.get("color", "#ef4444")
        sev = alert.get("severity", "HIGH")
        behavior = alert.get("behavior_class", "Violation")
        clip = alert.get("clip_id", "")
        zone = alert.get("zone", "Zone-1")

        if sev in ("HIGH", "CRITICAL"):
            # Animated alert banner
            st.markdown(
                f"""
                <div style="
                    background:{color}22;
                    border:2px solid {color};
                    border-radius:8px;
                    padding:16px;
                    margin:8px 0;
                    animation: pulse 1s infinite;
                ">
                    <h3 style="color:{color}; margin:0">
                        ⚠️ {sev} ALERT — {behavior}
                    </h3>
                    <p style="margin:4px 0">
                        Clip: <b>{clip}</b> | Zone: <b>{zone}</b>
                    </p>
                    <p style="margin:4px 0; color:#888">
                        {alert.get("event_description", "")[:200]}
                    </p>
                </div>
                <style>
                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.6; }}
                    100% {{ opacity: 1; }}
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

    # Latest clip viewer
    clips_path = Path(clips_dir)
    clip_files = (
        sorted(clips_path.glob("*.mp4"))
        + sorted(clips_path.glob("*.avi"))
        + sorted(clips_path.glob("*.mov"))
        if clips_path.exists()
        else []
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Video Feed")
        if clip_files:
            selected_clip = st.selectbox(
                "Select clip",
                options=clip_files,
                format_func=lambda p: p.name,
            )
            st.video(str(selected_clip))
        else:
            st.info("No video clips found. Add clips to the data/clips directory.")
            st.markdown("""
            **Quick start:**
            ```bash
            # Download Kaggle dataset (requires kaggle CLI setup)
            kaggle datasets download -d trnhhnggiang/video-dataset-for-safe-and-unsafe-behaviours
            unzip video-dataset-for-safe-and-unsafe-behaviours.zip -d data/clips/
            ```
            """)

    with col2:
        st.subheader("Compliance Status")
        events = get_all_events(limit=10)
        if not events:
            st.success("✅ No violations detected")
        else:
            latest = events[0]
            sev = latest.get("severity", "LOW")
            color = SEVERITY_COLORS.get(sev, "#888")
            st.markdown(
                f'<div style="background:{color}22; border:2px solid {color}; '
                f'border-radius:8px; padding:12px; text-align:center;">'
                f'<h2 style="color:{color}">{sev}</h2>'
                f"<p>{latest['behavior_class']}</p>"
                f'<p style="font-size:12px; color:#888">{latest["timestamp"][:19]}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.divider()
            st.caption("Recent violations:")
            for e in events[:5]:
                c = SEVERITY_COLORS.get(e["severity"], "#888")
                st.markdown(
                    f'<p style="font-size:13px">'
                    f'<span style="color:{c}">●</span> '
                    f"<b>{e['behavior_class']}</b><br>"
                    f'<span style="color:#888; font-size:11px">{e["clip_id"]} · {e["timestamp"][11:19]}</span>'
                    f"</p>",
                    unsafe_allow_html=True,
                )

    # Recent violation frames
    st.divider()
    st.subheader("Recent Violation Frames")
    frames_dir = ROOT / "outputs" / "frames"
    frame_files = (
        sorted(frames_dir.glob("*.jpg"), reverse=True)[:6]
        if frames_dir.exists()
        else []
    )

    if frame_files:
        cols = st.columns(min(len(frame_files), 3))
        for i, fp in enumerate(frame_files[:6]):
            with cols[i % 3]:
                st.image(str(fp), caption=fp.stem[:40], use_container_width=True)
    else:
        st.info("No violation frames yet. Run detection to populate.")


# ── View B — Alert Timeline ────────────────────────────────────────────────────
elif view == "🚨 Alert Timeline":
    st.header("🚨 Alert Timeline Stream")
    st.caption("Real-time chronological stream of compliance events")

    events = get_all_events(limit=200)

    if not events:
        st.info("No compliance events recorded yet.")
    else:
        # Summary metrics
        df = pd.DataFrame(events)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total = len(df)
            st.metric("Total Events", total)
        with c2:
            critical = len(df[df["severity"] == "CRITICAL"]) if "severity" in df else 0
            st.metric("Critical", critical, delta=None)
        with c3:
            high = len(df[df["severity"] == "HIGH"]) if "severity" in df else 0
            st.metric("High", high)
        with c4:
            unique_clips = df["clip_id"].nunique() if "clip_id" in df else 0
            st.metric("Clips Processed", unique_clips)

        st.divider()

        # Timeline stream
        for event in events:
            sev = event.get("severity", "LOW")
            color = SEVERITY_COLORS.get(sev, "#888")
            ts = event.get("timestamp", "")[:19].replace("T", " ")
            behavior = event.get("behavior_class", "")
            clip = event.get("clip_id", "")
            zone = event.get("zone", "")
            desc = event.get("event_description", "")[:200]
            esc = event.get("escalation_action", "")
            conf = float(event.get("confidence", 0))
            rule_ref = event.get("policy_rule_ref", "")

            with st.container():
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {color}; padding: 8px 12px; margin: 6px 0; background: {color}11; border-radius: 0 6px 6px 0;">
                        <div style="display:flex; justify-content:space-between; align-items:center">
                            <span style="color:{color}; font-weight:bold; font-size:14px">⬤ {sev}</span>
                            <span style="color:#888; font-size:12px">{ts}</span>
                        </div>
                        <b style="font-size:15px">{behavior}</b><br>
                        <span style="color:#888; font-size:12px">
                            {clip} · {zone} · {rule_ref} · conf {conf:.0%} · {esc}
                        </span><br>
                        <span style="font-size:13px">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ── View C — Historical Log & Export ─────────────────────────────────────────
elif view == "📋 Historical Log & Export":
    st.header("📋 Historical Compliance Log")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        sev_filter = st.multiselect(
            "Severity",
            options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            default=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        )
    with col2:
        behavior_options = ["All"] + [
            r["behavior_class"] for r in COMPLIANCE_RULES.values()
        ]
        behavior_filter = st.selectbox("Behavior Class", behavior_options)
    with col3:
        limit = st.slider("Max records", 50, 1000, 200)

    # Load filtered events
    events = get_all_events(
        severity_filter=sev_filter if sev_filter else None,
        behavior_filter=None if behavior_filter == "All" else behavior_filter,
        limit=limit,
    )

    if not events:
        st.info("No records match the current filters.")
    else:
        df = pd.DataFrame(events)

        # Color-coded table
        def color_severity(val):
            colors_map = {
                "LOW": "background-color: #22c55e33",
                "MEDIUM": "background-color: #f59e0b33",
                "HIGH": "background-color: #ef444433",
                "CRITICAL": "background-color: #7c3aed33",
            }
            return colors_map.get(val, "")

        display_cols = [
            "timestamp",
            "clip_id",
            "zone",
            "behavior_class",
            "severity",
            "policy_rule_ref",
            "escalation_action",
            "confidence",
        ]
        display_df = df[[c for c in display_cols if c in df.columns]]

        st.dataframe(
            display_df.style.map(color_severity, subset=["severity"]),
            use_container_width=True,
            height=400,
        )

        st.caption(f"Showing {len(df)} records")
        st.divider()

        # Export
        st.subheader("Export Audit Log")
        ecol1, ecol2 = st.columns(2)

        with ecol1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "⬇ Download CSV",
                data=csv_data,
                file_name=f"compliance_audit_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with ecol2:
            json_data = df.to_json(orient="records", indent=2)
            st.download_button(
                "⬇ Download JSON",
                data=json_data,
                file_name=f"compliance_audit_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        # Charts
        st.divider()
        st.subheader("Analytics")
        acol1, acol2 = st.columns(2)

        with acol1:
            import plotly.express as px

            if "severity" in df.columns:
                sev_counts = df["severity"].value_counts().reset_index()
                sev_counts.columns = ["severity", "count"]
                color_map = {k: v for k, v in SEVERITY_COLORS.items()}
                fig = px.pie(
                    sev_counts,
                    values="count",
                    names="severity",
                    color="severity",
                    color_discrete_map=color_map,
                    title="Violations by Severity",
                )
                st.plotly_chart(fig, use_container_width=True)

        with acol2:
            if "behavior_class" in df.columns:
                bc_counts = df["behavior_class"].value_counts().reset_index()
                bc_counts.columns = ["behavior_class", "count"]
                fig2 = px.bar(
                    bc_counts,
                    x="behavior_class",
                    y="count",
                    title="Violations by Behavior Class",
                    color="count",
                    color_continuous_scale="reds",
                )
                fig2.update_xaxes(tickangle=15)
                st.plotly_chart(fig2, use_container_width=True)
