"""
dashboard/app.py — Streamlit Decision-Support Dashboard.

Run with:
    streamlit run dashboard/app.py

Visualises:
  - Live / pre-recorded video with bounding boxes coloured by anomaly tier
  - Per-track trajectories plotted on a top-down frame overlay
  - Anomaly score time-series per track
  - Component score breakdown (visual / LSTM / lingering)
  - Human-readable alert explanations
"""

from __future__ import annotations

import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import cv2
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CCTV Behaviour Analyser",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .alert-high   { background: #ff4b4b22; border-left: 4px solid #ff4b4b; padding: 0.5rem 1rem; border-radius: 4px; }
    .alert-medium { background: #ffa50022; border-left: 4px solid #ffa500; padding: 0.5rem 1rem; border-radius: 4px; }
    .alert-low    { background: #00ff0022; border-left: 4px solid #00cc00; padding: 0.5rem 1rem; border-radius: 4px; }
    .metric-box   { background: #1e1e2e; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Sidebar controls
# ─────────────────────────────────────────────────────────────

st.sidebar.title("🎥 CCTV Behaviour Analyser")
st.sidebar.caption("Subtle anomaly detection · Decision support only")

cfg = load_config()

mode = st.sidebar.radio("Mode", ["Upload Video", "Load Results JSON"])
threshold = st.sidebar.slider(
    "Anomaly threshold",
    min_value=0.0, max_value=1.0,
    value=cfg["fusion"]["anomaly_threshold"],
    step=0.05,
)
show_traj = st.sidebar.checkbox("Show trajectories", value=True)
show_bbox = st.sidebar.checkbox("Show bounding boxes", value=True)

degradation_mode = st.sidebar.selectbox(
    "Apply degradation (preview)",
    ["none", "motion_blur", "gaussian_noise", "low_light", "compression"],
)


# ─────────────────────────────────────────────────────────────
#  Main layout
# ─────────────────────────────────────────────────────────────

st.title("🔍 Subtle Behaviour Detection")
st.caption("Detects loitering, pacing, and irregular movement in CCTV footage")

col_video, col_panel = st.columns([3, 2])


# ─────────────────────────────────────────────────────────────
#  Video / Results loading
# ─────────────────────────────────────────────────────────────

def load_demo_results() -> list[dict]:
    """Generate plausible demo fusion results for UI testing."""
    results = []
    rng = np.random.default_rng(42)
    for tid in range(1, 6):
        n_frames = rng.integers(80, 200)
        base_score = rng.uniform(0.2, 0.9)
        for f in range(n_frames):
            drift = rng.normal(0, 0.05)
            score = float(np.clip(base_score + drift, 0, 1))
            tier = "high" if score >= 0.75 else ("medium" if score >= 0.5 else "low")
            results.append({
                "track_id": tid,
                "frame_idx": f,
                "anomaly_score": round(score, 3),
                "visual_score_norm": round(rng.uniform(0.1, 0.9), 3),
                "lstm_score_norm":   round(rng.uniform(0.1, 0.9), 3),
                "lingering_score_norm": round(rng.uniform(0.1, 0.9), 3),
                "alert_tier": tier,
                "explanation": (
                    "⚠️ Possible Loitering — sustained, confined, slow movement" if tier == "high"
                    else "🔍 Irregular Movement observed" if tier == "medium"
                    else "✅ Normal Behaviour"
                ),
            })
    return results


with col_video:
    st.subheader("Video Feed")
    if mode == "Upload Video":
        uploaded = st.file_uploader("Upload surveillance video", type=["mp4", "avi", "mov"])
        if uploaded:
            st.video(uploaded)
        else:
            st.info("Upload a video to begin analysis.  Results panel will use demo data.")
    else:
        results_file = st.file_uploader("Upload results JSON", type=["json"])


with col_panel:
    st.subheader("Live Alerts")

    # Load results
    if mode == "Load Results JSON" and 'results_file' in dir() and results_file:
        results = json.load(results_file)
    else:
        results = load_demo_results()

    df = pd.DataFrame(results)

    # Filter by threshold
    df["flagged"] = df["anomaly_score"] >= threshold

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tracks", df["track_id"].nunique())
    m2.metric("Alerts", int(df.groupby("track_id")["flagged"].any().sum()))
    m3.metric("Max Score", f"{df['anomaly_score'].max():.2f}")
    m4.metric("Avg Score", f"{df['anomaly_score'].mean():.2f}")

    st.divider()

    # Per-track alert cards
    for tid, group in df.groupby("track_id"):
        peak = group["anomaly_score"].max()
        tier = group.loc[group["anomaly_score"].idxmax(), "alert_tier"]
        label = group.loc[group["anomaly_score"].idxmax(), "explanation"]
        css_class = f"alert-{tier}"

        st.markdown(f"""
        <div class="{css_class}">
            <strong>Track {tid}</strong> &nbsp; Score: <b>{peak:.2f}</b><br>
            <small>{label}</small>
        </div><br>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Anomaly score time-series
# ─────────────────────────────────────────────────────────────

st.subheader("Anomaly Score Time-Series")

selected_tracks = st.multiselect(
    "Select tracks to display",
    options=sorted(df["track_id"].unique()),
    default=sorted(df["track_id"].unique())[:3],
)

fig_ts = go.Figure()
colours = px.colors.qualitative.Plotly
for i, tid in enumerate(selected_tracks):
    sub = df[df["track_id"] == tid].sort_values("frame_idx")
    fig_ts.add_trace(go.Scatter(
        x=sub["frame_idx"], y=sub["anomaly_score"],
        mode="lines", name=f"Track {tid}",
        line=dict(color=colours[i % len(colours)], width=2),
    ))

fig_ts.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
fig_ts.update_layout(
    xaxis_title="Frame Index",
    yaxis_title="Anomaly Score",
    yaxis=dict(range=[0, 1]),
    height=300,
    margin=dict(t=10),
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_ts, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  Component score breakdown
# ─────────────────────────────────────────────────────────────

st.subheader("Component Score Breakdown (peak frame per track)")

peak_rows = df.loc[df.groupby("track_id")["anomaly_score"].idxmax()]
bar_df = peak_rows[["track_id", "visual_score_norm", "lstm_score_norm", "lingering_score_norm"]].copy()
bar_df = bar_df.melt(id_vars="track_id", var_name="Component", value_name="Score")
bar_df["Component"] = bar_df["Component"].map({
    "visual_score_norm":    "Visual (Conv-AE)",
    "lstm_score_norm":      "Temporal (LSTM-AE)",
    "lingering_score_norm": "Lingering Heuristic",
})

fig_bar = px.bar(
    bar_df, x="track_id", y="Score", color="Component",
    barmode="group", range_y=[0, 1],
    template="plotly_dark", height=280,
    labels={"track_id": "Track ID"},
)
fig_bar.update_layout(margin=dict(t=10))
st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  Evaluation results (if available)
# ─────────────────────────────────────────────────────────────

eval_path = Path("outputs/eval_report.json")
if eval_path.exists():
    st.subheader("Robustness Evaluation Results")
    with open(eval_path) as f:
        report = json.load(f)
    eval_df = pd.DataFrame(report["results"])
    fig_eval = px.bar(
        eval_df, x="condition", y="roc_auc",
        color="roc_auc", color_continuous_scale="RdYlGn",
        range_y=[0, 1], template="plotly_dark", height=280,
        labels={"condition": "Condition", "roc_auc": "ROC-AUC"},
        title="ROC-AUC by Degradation Condition",
    )
    st.plotly_chart(fig_eval, use_container_width=True)
    st.json(report["summary"])

st.divider()
st.caption("⚠️ This system is a decision-support tool. No facial recognition is performed. Human review required before action.")
