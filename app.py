"""
JEPA Exhibit Defect Detection — Main Streamlit Application
Self-supervised spatio-temporal anomaly detection using JEPA architecture.

Run: streamlit run jepa/app.py
  (from project root: d:/agentic ai/exhibit_defect_detection/)
"""
import sys, os
from pathlib import Path

# Ensure jepa/ is importable regardless of cwd
JEPA_DIR = Path(__file__).parent
sys.path.insert(0, str(JEPA_DIR))

import streamlit as st
import numpy as np
import cv2
import torch
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import json
import time

import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JEPA Exhibit Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS Theming
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* White background */
  .stApp { background: #ffffff; }

  /* Sidebar — light grey */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0f4fa 0%, #e8eef8 100%);
    border-right: 1px solid #d0dcea;
  }

  /* Cards */
  .metric-card {
    background: #f5f8ff;
    border: 1px solid #c8d8f0;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(80,120,200,0.08);
  }
  .metric-card .value { font-size: 2.2rem; font-weight: 700; }
  .metric-card .label { font-size: 0.85rem; color: #4a6a9a; margin-top: 4px; }

  /* Alert banners */
  .alert-normal {
    background: linear-gradient(90deg, rgba(0,180,90,0.12), rgba(0,180,90,0.04));
    border-left: 4px solid #00a854;
    border-radius: 8px;
    padding: 14px 20px;
    color: #007a3d;
    font-weight: 600;
    font-size: 1.1rem;
  }
  .alert-anomaly {
    background: linear-gradient(90deg, rgba(220,50,50,0.12), rgba(220,50,50,0.04));
    border-left: 4px solid #e03030;
    border-radius: 8px;
    padding: 14px 20px;
    color: #c02020;
    font-weight: 600;
    font-size: 1.1rem;
    animation: pulse 1s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.7; }
  }

  /* Phase pill */
  .phase-pill {
    display: inline-block;
    background: linear-gradient(90deg, #1a6ab5, #2a85d5);
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Section headers */
  h1 { color: #1a2a4a !important; font-weight: 700 !important; }
  h2 { color: #2a4a7a !important; font-weight: 600 !important; }
  h3 { color: #3a5a8a !important; }

  /* Progress bar */
  .stProgress > div > div { background: linear-gradient(90deg, #1a6ab5, #00a8e8) !important; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; }
  .stTabs [data-baseweb="tab"] {
    background: #eef3fb;
    border: 1px solid #c8d8f0;
    border-radius: 10px;
    color: #3a5a8a;
    font-weight: 600;
    padding: 10px 24px;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a6ab5, #0d4d8a);
    border-color: #1a6ab5;
    color: #ffffff !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #1a6ab5, #0d4d8a);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 12px 28px;
    font-size: 1rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2280d5, #1260aa);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(26,106,181,0.25);
  }

  /* Labels */
  label { color: #3a5a8a !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_model_status():
    """Check which components are trained and calibrated."""
    return {
        "temporal":    (cfg.CHECKPOINTS_DIR / "temporal.pt").exists(),
        "spatial":     (cfg.CHECKPOINTS_DIR / "spatial.pt").exists(),
        "mahal":       cfg.MAHAL_FILE.exists(),
        "calibration": cfg.CALIBRATION_FILE.exists(),
    }


def load_normal_frames_sample(max_frames: int = 80) -> list:
    """
    Load a representative sample of normal training frames from disk.
    Returns list of (original_frame_idx, np.ndarray RGB) tuples.
    """
    frames_dir = cfg.FRAMES_DIR / cfg.NORMAL_VIDEO.stem
    if not frames_dir.exists():
        return []
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        return []
    step = max(1, len(paths) // max_frames)
    sampled = paths[::step][:max_frames]
    out = []
    for i, p in enumerate(sampled):
        img = cv2.imread(str(p))
        if img is not None:
            out.append((i * step, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return out


def annotate_frame(frame_rgb: np.ndarray, label: str,
                   score=None, is_anomaly: bool = False) -> np.ndarray:
    """Draw a coloured border + label overlay on a frame copy."""
    bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    border_color = (0, 60, 255) if is_anomaly else (0, 180, 80)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, bgr.shape[0] - 1), border_color, 4)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (bgr.shape[1], 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0, bgr)
    cv2.putText(bgr, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    if score is not None:
        cv2.putText(bgr, f"Score: {score:.3f}", (6, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, border_color, 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def score_to_color(score: float, threshold: float) -> str:
    ratio = score / max(threshold, 1e-6)
    if ratio < 0.6:   return "#00c864"
    if ratio < 0.85:  return "#ffc107"
    if ratio < 1.0:   return "#ff6b35"
    return "#ff3c3c"


def make_score_gauge(score: float, threshold: float) -> go.Figure:
    color = score_to_color(score, threshold)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 4),
        number={"font": {"size": 32, "color": color}},
        gauge={
            "axis": {"range": [0, threshold * 2], "tickcolor": "#8ab4d4",
                     "tickfont": {"color": "#8ab4d4"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,           threshold * 0.6],  "color": "rgba(0,200,100,0.12)"},
                {"range": [threshold * 0.6, threshold],    "color": "rgba(255,193,7,0.12)"},
                {"range": [threshold,   threshold * 2],    "color": "rgba(255,60,60,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.8,
                "value": threshold,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=200,
        font={"color": "#8ab4d4"},
    )
    return fig


def make_score_timeline(results: list, threshold: float) -> go.Figure:
    frames = [r["frame_idx"] for r in results]
    scores = [r["score"]     for r in results]
    colors = [score_to_color(s, threshold) for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=scores,
        mode="lines+markers",
        line=dict(color="#1a6ab5", width=2),
        marker=dict(color=colors, size=6, line=dict(color="#0d1b2e", width=1)),
        name="Anomaly Score",
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#ff3c3c", line_width=2,
        annotation_text="Threshold", annotation_font_color="#ff3c3c",
    )
    # Shade anomaly regions
    anomaly_frames = [r for r in results if r["is_anomaly"]]
    for r in anomaly_frames:
        fig.add_vrect(
            x0=r["frame_idx"] - 0.5, x1=r["frame_idx"] + 0.5,
            fillcolor="rgba(255,60,60,0.12)", line_width=0,
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,46,0.6)",
        xaxis=dict(title="Frame Index", color="#8ab4d4", gridcolor="rgba(100,200,255,0.1)"),
        yaxis=dict(title="Anomaly Score", color="#8ab4d4", gridcolor="rgba(100,200,255,0.1)"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        showlegend=False,
        font=dict(color="#8ab4d4"),
    )
    return fig


def make_component_bar(components: dict) -> go.Figure:
    keys   = ["Temporal-S", "Temporal-L", "Spatial", "Energy", "Uncertainty"]
    vals   = [
        components.get("temporal",       0),
        components.get("temporal_long",  0),
        components.get("spatial",        0),
        components.get("energy",         0),
        components.get("uncertainty",    0),
    ]
    colors = ["#1a6ab5", "#00a8e8", "#7c4fff", "#ff6b35", "#00c864"]

    fig = go.Figure(go.Bar(
        x=keys, y=vals, marker_color=colors,
        marker_line_color="rgba(0,0,0,0)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f5f8ff",
        xaxis=dict(color="#3a5a8a"),
        yaxis=dict(color="#3a5a8a", gridcolor="#dde8f5"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=200,
        font=dict(color="#3a5a8a"),
    )
    return fig


def make_loss_curve(loss_history: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=loss_history, mode="lines",
        line=dict(color="#00c8ff", width=2),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,46,0.6)",
        xaxis=dict(title="Epoch", color="#8ab4d4", gridcolor="rgba(100,200,255,0.1)"),
        yaxis=dict(title="Loss", color="#8ab4d4", gridcolor="rgba(100,200,255,0.1)"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        font=dict(color="#8ab4d4"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 JEPA Defect Detection")
    st.markdown('<span class="phase-pill">Self-Supervised · Spatio-Temporal</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    status = get_model_status()
    st.markdown("### 🎛 System Status")
    for name, ok in [
        ("Temporal Transformer (Short)", status["temporal"]),
        ("Temporal Transformer (Long)",  (cfg.CHECKPOINTS_DIR / "temporal_long.pt").exists()),
        ("Spatial JEPA Head",            status["spatial"]),
        ("Energy Model (SVDD)",          status.get("energy", cfg.ENERGY_FILE.exists())),
        ("Calibration",                  status["calibration"]),
    ]:
        icon = "✅" if ok else "⭕"
        st.markdown(f"{icon} **{name}**")

    st.markdown("---")
    st.markdown("### 📂 Videos")
    normal_exists = cfg.NORMAL_VIDEO.exists()
    test_exists   = cfg.TEST_VIDEO.exists()
    st.markdown(f"{'✅' if normal_exists else '❌'} Normal: `{cfg.NORMAL_VIDEO.name}`")
    st.markdown(f"{'✅' if test_exists   else '❌'} Test:   `{cfg.TEST_VIDEO.name}`")

    st.markdown("---")
    st.markdown("### ⚙ Hyperparameters")
    with st.expander("View / Edit Config"):
        fps        = st.slider("Sample FPS",       1, 10, cfg.TARGET_FPS)
        window_k   = st.slider("Window Size K",    4, 16, cfg.WINDOW_SIZE)
        epochs     = st.slider("Training Epochs",  5, 100, cfg.EPOCHS, step=5)
        batch_size = st.slider("Batch Size",       4, 64, cfg.BATCH_SIZE, step=4)
        alpha      = st.slider("α Temporal (Short)",  0.0, 1.0, cfg.SCORE_ALPHA,      step=0.05)
        alpha_long = st.slider("α Temporal (Long)",   0.0, 1.0, cfg.SCORE_ALPHA_LONG, step=0.05)
        beta_val   = st.slider("β Spatial",            0.0, 1.0, cfg.SCORE_BETA,        step=0.05)
        gamma      = st.slider("γ Energy (SVDD)",      0.0, 1.0, cfg.SCORE_GAMMA,       step=0.05)
    st.markdown(f"🖥 Device: **`{cfg.DEVICE.upper()}`**")

    st.markdown("---")
    st.caption("Built with JEPA · ViT-B/16 · 2026")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_train, tab_calibrate, tab_test = st.tabs([
    "📖 Overview",
    "🎓 Train Model",
    "🎯 Calibrate",
    "🔍 Run Test",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("## 🧠 JEPA Exhibit Defect Detection")
    st.markdown("""
    > A **self-supervised spatio-temporal predictive model** that learns the *normal behavioral 
    > manifold* of an exhibit and detects deviations in latent representation space —  
    > **no defect labels required.**
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="value" style="color:#00c8ff">ViT-B/16*</div>
          <div class="label">Partial Domain Adaptation<br>Last 4 blocks fine-tuned @ 1e-5 LR</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="value" style="color:#7c4fff">JEPA²</div>
          <div class="label">Cross-Attention Spatial JEPA<br>Multi-Scale Temporal (K=8 + K=32)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div class="value" style="color:#ff6b35">5-Score</div>
          <div class="label">Temporal(S+L) · Spatial · Energy<br>+ MC Dropout Uncertainty</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔄 Pipeline Flow")
    st.markdown("""
    ```
    Normal Video (2.mp4)
         ↓
    Frame Sampler  [3 FPS · 224×224 · CLAHE normalization]
         ↓
    Frozen ViT-B/16  [CLS token 768D + 196 patch tokens]
         ↓  (embeddings cached to disk)
    ┌──────────────────────────────────────────────────┐
    │              JEPA Training (phases 3+4)          │
    │  Temporal Transformer → predict E_t from E_{t-K}│
    │  Spatial JEPA Head    → reconstruct masked patches│
    └──────────────────────────────────────────────────┘
         ↓
    Mahalanobis Fit + Percentile Calibration
         ↓
    Test Video Inference
         ↓
    Anomaly Score = α·Temporal + β·Spatial + γ·Mahalanobis
         ↓
    🚨 Alert when Score > Threshold
    ```
    """)

    st.markdown("### 📋 Workflow Steps")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Step 1 — Train** *(Tab 2)*
        1. App samples frames from `2.mp4` at 3 FPS
        2. Encodes each frame with frozen ViT-B/16  
        3. Trains Temporal Transformer to predict next embedding  
        4. Trains Spatial JEPA head to reconstruct masked patches  
        """)
    with col_b:
        st.markdown("""
        **Step 2 — Calibrate** *(Tab 3)*
        1. Runs model on normal video embeddings
        2. Computes per-frame anomaly scores
        3. Sets threshold at 97th percentile (EVT-inspired)
        4. Fits Mahalanobis distribution model

        **Step 3 — Test** *(Tab 4)*
        1. Processes test video frame-by-frame
        2. Real-time anomaly score visualization
        3. Alert on threshold breach
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("## 🎓 Train the JEPA Model")
    st.markdown("Trains on **normal exhibit video** (`2.mp4`). No defect labels needed.")

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        if cfg.NORMAL_VIDEO.exists():
            st.info(f"📹 Normal video found: `{cfg.NORMAL_VIDEO}`")
        else:
            st.error(f"❌ Normal video not found at: `{cfg.NORMAL_VIDEO}`")

    with col_btn:
        start_training = st.button("🚀 Start Training")

    log_container   = st.empty()
    prog_container  = st.empty()
    chart_container = st.empty()

    if start_training:
        if not cfg.NORMAL_VIDEO.exists():
            st.error("Normal video not found. Cannot train.")
        else:
            log_lines = []
            loss_hist = []

            def add_log(msg: str):
                log_lines.append(msg)
                log_container.markdown(
                    "**Training Log:**\n" + "\n".join(f"• {l}" for l in log_lines[-20:])
                )

            def on_progress(epoch: int, total: int, loss: float):
                loss_hist.append(loss)
                prog_container.progress(epoch / total, text=f"Epoch {epoch}/{total} — Loss: {loss:.5f}")
                if len(loss_hist) > 1:
                    chart_container.plotly_chart(
                        make_loss_curve(loss_hist),
                        key=f"loss_live_{epoch}",
                    )

            try:
                from training.trainer import JEPATrainer

                trainer = JEPATrainer(
                    device=cfg.DEVICE,
                    log_callback=add_log,
                    progress_callback=on_progress,
                )
                cfg.TARGET_FPS = fps  # use sidebar value

                # Step 1: Extract frames
                add_log("📹 Extracting frames from normal video...")
                frame_prog = st.progress(0, text="Sampling frames...")
                paths = trainer.extract_frames(
                    video_path=cfg.NORMAL_VIDEO,
                    fps=fps,
                    frame_progress=lambda cur, tot: frame_prog.progress(
                        min(cur / max(tot, 1), 1.0), text=f"Frame {cur}/{tot}"
                    ),
                )
                frame_prog.empty()

                if len(paths) < cfg.WINDOW_SIZE + 2:
                    st.warning(f"⚠ Only {len(paths)} frames extracted. Video may be very short. Trying anyway...")

                # Step 2: Encode
                add_log("🧠 Encoding frames with frozen ViT-B/16...")
                enc_prog = st.progress(0, text="Encoding...")
                trainer.encode_frames(
                    frame_paths=paths,
                    video_stem=cfg.NORMAL_VIDEO.stem,
                    frame_progress=lambda cur, tot: enc_prog.progress(
                        min(cur / max(tot, 1), 1.0), text=f"Encoded {cur}/{tot} frames"
                    ),
                )
                enc_prog.empty()

                # Step 3a: Train Temporal (Short+Long) + Spatial Cross-Attention
                add_log("🎯 Training Short+Long Temporal + Cross-Attention Spatial JEPA…")
                cfg.WINDOW_SIZE = window_k
                loss_history = trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=cfg.LEARNING_RATE
                )
                (cfg.CHECKPOINTS_DIR / "loss_history.json").write_text(json.dumps(loss_history))
                chart_container.plotly_chart(make_loss_curve(loss_history),
                                             key="loss_final")

                # Step 3b: Deep SVDD Energy Model (Upgrade 4)
                add_log("⚡ Stage 3b — Training Deep SVDD energy model…")
                svdd_prog = st.progress(0, "SVDD training…")
                def on_svdd_progress(ep, tot, loss):
                    svdd_prog.progress(ep / tot, text=f"SVDD Epoch {ep}/{tot} — {loss:.5f}")
                trainer.on_progress = on_svdd_progress
                svdd_history = trainer.train_energy(epochs=cfg.SVDD_EPOCHS)
                svdd_prog.empty()
                add_log(f"✅ SVDD done — final loss: {svdd_history[-1]:.5f}")

                st.success(f"✅ All training complete! JEPA loss: {loss_history[-1]:.5f} | SVDD: {svdd_history[-1]:.5f}")

                # Store for calibration
                st.session_state["cls_embeddings"]   = trainer.cls_embeddings
                st.session_state["patch_embeddings"] = trainer.patch_embeddings
                st.session_state["trainer"]          = trainer

            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Show prior training if available
    loss_file = cfg.CHECKPOINTS_DIR / "loss_history.json"
    if loss_file.exists() and not start_training:
        with open(loss_file) as f:
            prior = json.load(f)
        st.markdown("**Previous Training Loss Curve:**")
        st.plotly_chart(make_loss_curve(prior), key="loss_prior")
        st.caption(f"Epochs: {len(prior)} | Final loss: {prior[-1]:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CALIBRATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_calibrate:
    st.markdown("## 🎯 Calibrate Anomaly Threshold")
    st.markdown("""
    Runs the trained model on the **normal video** to compute per-frame scores,
    then sets the threshold at the **97th percentile** — so only the top 3% of 
    *normal* frames would be flagged.
    """)

    status = get_model_status()
    if not (status["temporal"] and status["spatial"]):
        st.warning("⚠ Models not trained yet. Complete training first.")
    else:
        calib_file = cfg.CALIBRATION_FILE
        if calib_file.exists():
            with open(calib_file) as f:
                c = json.load(f)
            st.success(f"✅ Calibration already done — Threshold: **{c['threshold']:.4f}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Threshold", f"{c['threshold']:.4f}")
            col2.metric("Normal Score Mean", f"{c['mean']:.4f}")
            col3.metric("Normal Score Std",  f"{c['std']:.4f}")

        run_calib = st.button("🔄 Run Calibration (Re-calibrate)")

        if run_calib:
            try:
                from anomaly.scorer import AnomalyScorer
                from anomaly.energy_model import EnergyModel
                from training.calibration import calibrate_threshold, calibrate_component_scales
                from models.temporal_transformer import TemporalTransformer
                from models.spatial_jepa import SpatialJEPAHead

                calib_log  = st.empty()
                calib_prog = st.progress(0)
                def log_c(m): calib_log.info(m)

                # ── Load models ───────────────────────────────────────────────
                log_c("Loading Short Temporal model…")
                long_seq_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)
                temporal_s = TemporalTransformer(window_size=cfg.WINDOW_SIZE).to(cfg.DEVICE)
                temporal_s.load(cfg.CHECKPOINTS_DIR / "temporal.pt", cfg.DEVICE)

                temporal_l = TemporalTransformer(window_size=long_seq_len).to(cfg.DEVICE)
                if (cfg.CHECKPOINTS_DIR / "temporal_long.pt").exists():
                    temporal_l.load(cfg.CHECKPOINTS_DIR / "temporal_long.pt", cfg.DEVICE)

                log_c("Loading Cross-Attention Spatial JEPA…")
                spatial_m = SpatialJEPAHead().to(cfg.DEVICE)
                spatial_m.load(cfg.CHECKPOINTS_DIR / "spatial.pt", cfg.DEVICE)

                log_c("Loading Deep SVDD Energy model…")
                energy_m = EnergyModel().to(cfg.DEVICE)
                if cfg.ENERGY_FILE.exists():
                    energy_m.load(cfg.ENERGY_FILE, cfg.DEVICE)

                # ── Load embeddings ───────────────────────────────────────────
                cls_path   = cfg.EMBEDDINGS_DIR / f"{cfg.NORMAL_VIDEO.stem}_cls.npy"
                patch_path = cfg.EMBEDDINGS_DIR / f"{cfg.NORMAL_VIDEO.stem}_patches.npy"
                if not cls_path.exists():
                    st.error("Embeddings not found — run training first.")
                else:
                    log_c("Loading normal embeddings…")
                    cls_embs   = np.load(str(cls_path))
                    patch_embs = np.load(str(patch_path))

                    # ── Build scorer (no scales yet → all 1.0) ────────────────
                    scorer = AnomalyScorer(
                        temporal_short = temporal_s,
                        temporal_long  = temporal_l,
                        spatial_head   = spatial_m,
                        energy_model   = energy_m,
                        device         = cfg.DEVICE,
                    )

                    # ── Score all normal frames ───────────────────────────────
                    log_c("Scoring normal frames (this may take a minute)…")
                    normal_scores, t_errs, t_long_errs, s_errs, e_errs = [], [], [], [], []
                    N = len(cls_embs)
                    for i in range(N):
                        s, comp = scorer.push_and_score(cls_embs[i], patch_embs[i])
                        normal_scores.append(s)
                        t_errs.append(comp.get("temporal",      0.0))
                        t_long_errs.append(comp.get("temporal_long", 0.0))
                        s_errs.append(comp.get("spatial",       0.0))
                        e_errs.append(comp.get("energy",        0.0))
                        calib_prog.progress((i + 1) / N, text=f"Frame {i+1}/{N}")

                    calib_prog.empty()
                    normal_scores = np.array(normal_scores)

                    # ── Compute per-component scales ──────────────────────────
                    scales = calibrate_component_scales(
                        np.array(t_errs), np.array(t_long_errs),
                        np.array(s_errs), np.array(e_errs),
                    )

                    # ── Save threshold + scales ───────────────────────────────
                    threshold = calibrate_threshold(normal_scores, component_scales=scales)
                    log_c(f"✅ Threshold = {threshold:.4f} | scales: {scales}")

                    st.success(f"✅ Calibration complete! Threshold = **{threshold:.4f}**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Threshold",  f"{threshold:.4f}")
                    col2.metric("t_scale",    f"{scales['t_scale']:.4f}")
                    col3.metric("s_scale",    f"{scales['s_scale']:.4f}")
                    col4.metric("e_scale",    f"{scales['e_scale']:.4f}")

                    # Distribution plot
                    fig = px.histogram(
                        x=normal_scores, nbins=50,
                        labels={"x": "Anomaly Score", "y": "Count"},
                        color_discrete_sequence=["#1a6ab5"],
                    )
                    fig.add_vline(x=threshold, line_dash="dash", line_color="#e03030",
                                  annotation_text="Threshold", annotation_font_color="#e03030")
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
                        font=dict(color="#3a5a8a"),
                        margin=dict(l=10, r=10, t=30, b=10), height=250,
                        title="Normal Score Distribution", title_font_color="#2a4a7a",
                        xaxis=dict(gridcolor="#dde8f5"), yaxis=dict(gridcolor="#dde8f5"),
                    )
                    st.plotly_chart(fig, key="calib_hist")

            except Exception as e:
                st.error(f"Calibration failed: {e}")
                import traceback
                st.code(traceback.format_exc())




# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEST / INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_test:
    st.markdown("## 🔍 Run Anomaly Detection on Test Video")

    status = get_model_status()
    if not (status["temporal"] and status["spatial"] and status["calibration"]):
        st.warning("⚠ Models not trained or calibrated yet. Complete Tabs 2 & 3 first.")
    else:
        from training.calibration import load_calibration
        calib_data = load_calibration()
        threshold  = calib_data["threshold"] if calib_data else 1.0

        col_t, col_th = st.columns([3, 1])
        with col_t:
            st.info(f"🎬 Test video: `{cfg.TEST_VIDEO.name}`")
        with col_th:
            threshold = st.number_input(
                "Override Threshold", value=float(threshold),
                min_value=0.001, step=0.01, format="%.4f"
            )

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            max_frames_opt = st.slider("Max frames to process (0 = all)", 0, 500, 0, step=10)
        with col_opt2:
            show_every = st.slider("Show every Nth frame in gallery", 1, 20, 5)
        with col_opt3:
            mask_humans = st.toggle(
                "🧍 Ignore Humans (YOLO mask)",
                value=True,
                help="Black-out detected persons before scoring — human presence will NOT raise the anomaly score.",
            )

        run_test = st.button("▶ Run Detection")

        if run_test:
            if not cfg.TEST_VIDEO.exists():
                st.error(f"Test video not found: `{cfg.TEST_VIDEO}`")
            else:
                try:
                    from inference.pipeline import build_scorer, run_video_inference

                    with st.spinner("🔧 Loading models..."):
                        scorer, encoder, _ = build_scorer(cfg.DEVICE)

                    # Human masking filter
                    _masker = None
                    if mask_humans:
                        with st.spinner("🧍 Loading YOLO human masker..."):
                            from preprocessing.human_mask import HumanMaskFilter
                            _masker = HumanMaskFilter()
                        st.info("🧍 Human masking active — person regions blacked out before scoring.")

                    total_prog   = st.progress(0, text="Initializing...")
                    score_chart  = st.empty()
                    frame_disp   = st.empty()
                    alert_disp   = st.empty()

                    results = []
                    max_f   = max_frames_opt if max_frames_opt > 0 else None

                    # Count total frames first for progress
                    cap_tmp = cv2.VideoCapture(str(cfg.TEST_VIDEO))
                    total_src    = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
                    src_fps_tmp  = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
                    cap_tmp.release()
                    est_total = total_src // max(1, round(src_fps_tmp / cfg.TARGET_FPS))
                    if max_f:
                        est_total = min(est_total, max_f)

                    # Override scorer weights from sidebar
                    scorer.alpha      = alpha
                    scorer.alpha_long = alpha_long
                    scorer.beta       = beta_val
                    scorer.gamma      = gamma

                    def frame_cb(frame_rgb, result, fidx):
                        results.append(result)
                        pct = min((fidx + 1) / max(est_total, 1), 1.0)
                        total_prog.progress(pct, text=f"Frame {fidx+1}/{est_total} — Score: {result['score']:.4f}")

                        # Live score chart (update every 5 frames)
                        if len(results) % 5 == 0 and len(results) > 1:
                            score_chart.plotly_chart(
                                make_score_timeline(results, threshold),
                                key=f"timeline_live_{len(results)}",
                            )

                        # Alert status
                        if result["is_anomaly"]:
                            alert_disp.markdown(
                                '<div class="alert-anomaly">🚨 ANOMALY DETECTED — Exhibit may be malfunctioning!</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            alert_disp.markdown(
                                '<div class="alert-normal">✅ Normal Operation</div>',
                                unsafe_allow_html=True
                            )

                        # Frame gallery
                        if fidx % show_every == 0:
                            # Overlay score on frame
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            sc = result["score"]
                            color_bgr = (0, 200, 100) if not result["is_anomaly"] else (0, 60, 255)
                            cv2.putText(frame_bgr, f"Score: {sc:.3f}", (8, 24),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                            if result["is_anomaly"]:
                                cv2.putText(frame_bgr, "ANOMALY", (8, 52),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 60, 255), 2)
                                cv2.rectangle(frame_bgr, (0, 0),
                                              (frame_bgr.shape[1]-1, frame_bgr.shape[0]-1),
                                              (0, 60, 255), 4)
                            frame_disp.image(
                                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {fidx} | {'⚠ ANOMALY' if result['is_anomaly'] else '✅ Normal'}",
                            )

                    run_video_inference(
                        cfg.TEST_VIDEO, scorer, encoder, threshold,
                        frame_callback=frame_cb, max_frames=max_f,
                        human_masker=_masker,
                    )

                    total_prog.progress(1.0, text="✅ Complete!")

                    # ── Summary Stats ──────────────────────────────────────
                    st.markdown("---")
                    st.markdown("### 📊 Detection Summary")

                    if results:
                        n_total   = len(results)
                        n_anomaly = sum(1 for r in results if r["is_anomaly"])
                        max_score = max(r["score"] for r in results)
                        mean_score = np.mean([r["score"] for r in results])

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Frames",   n_total)
                        c2.metric("Anomaly Frames", n_anomaly,
                                  delta=f"{n_anomaly/n_total*100:.1f}% flagged",
                                  delta_color="inverse")
                        c3.metric("Max Score",  f"{max_score:.4f}")
                        c4.metric("Mean Score", f"{mean_score:.4f}")

                        # Final timeline
                        st.plotly_chart(
                            make_score_timeline(results, threshold),
                            key="timeline_final",
                        )

                        # Score gauge for last frame
                        last_r = results[-1]
                        st.markdown("#### Last Frame Analysis")
                        gcol, bcol = st.columns([1, 1])
                        with gcol:
                            st.plotly_chart(make_score_gauge(last_r["score"], threshold),
                                            use_container_width=True, key="gauge_last")
                        with bcol:
                            st.plotly_chart(make_component_bar(last_r),
                                            use_container_width=True, key="bar_components")

                        # ── Anomaly Frame List ────────────────────────────
                        anomaly_frames = [r for r in results if r["is_anomaly"]]
                        if anomaly_frames:
                            st.markdown(f"#### 🚨 {len(anomaly_frames)} Anomaly Frame(s) Detected")
                            for r in anomaly_frames[:20]:
                                st.write(
                                    f"\u2022 Frame **{r['frame_idx']}** \u2014 "
                                    f"Score: `{r['score']:.4f}` "
                                    f"(T-S: {r['temporal']:.3f}, T-L: {r.get('temporal_long', 0):.3f}, "
                                    f"Sp: {r['spatial']:.3f}, E: {r.get('energy', 0):.3f}, "
                                    f"Unc: {r.get('uncertainty', 0):.4f})"
                                )
                            if len(anomaly_frames) > 20:
                                st.caption(f"... and {len(anomaly_frames) - 20} more anomaly frames")

                            # ── Side-by-Side Comparison ────────────────────
                            st.markdown("---")
                            st.markdown("### \U0001f5bc Side-by-Side: Normal Training Frame vs Anomaly Frame")
                            st.markdown(
                                "Top anomaly frames (sorted by score) are shown each paired with "
                                "the nearest-index frame from the **normal training video** "
                                "so you can directly see what changed."
                            )

                            normal_sample = load_normal_frames_sample(max_frames=80)

                            if not normal_sample:
                                st.warning(
                                    "\u26A0 No cached training frames found. "
                                    "Run training first to populate frames."
                                )
                            else:
                                top_anomalies = sorted(
                                    anomaly_frames,
                                    key=lambda r: r["score"],
                                    reverse=True,
                                )[:12]

                                total_normal = len(normal_sample)

                                for pair_i, anom_r in enumerate(top_anomalies):
                                    # Find nearest normal frame by frame index
                                    best_ni = min(
                                        range(total_normal),
                                        key=lambda i: abs(normal_sample[i][0] - anom_r["frame_idx"]),
                                    )
                                    norm_idx, norm_frame = normal_sample[best_ni]

                                    norm_ann = annotate_frame(
                                        norm_frame,
                                        f"NORMAL  frame #{norm_idx}",
                                        score=None,
                                        is_anomaly=False,
                                    )
                                    anom_ann = annotate_frame(
                                        anom_r["frame_rgb"],
                                        f"ANOMALY frame #{anom_r['frame_idx']}",
                                        score=anom_r["score"],
                                        is_anomaly=True,
                                    )

                                    rank_label = f"#{pair_i + 1} by score"
                                    col_norm, col_anom = st.columns(2)
                                    with col_norm:
                                        st.image(
                                            norm_ann,
                                            caption=f"\u2705 Normal \u00b7 frame {norm_idx} \u00b7 training video",
                                            use_container_width=True,
                                        )
                                    with col_anom:
                                        st.image(
                                            anom_ann,
                                            caption=(
                                                f"\U0001f6a8 Anomaly \u00b7 frame {anom_r['frame_idx']} "
                                                f"\u00b7 score {anom_r['score']:.4f} ({rank_label})"
                                            ),
                                            use_container_width=True,
                                        )
                                    st.markdown(
                                        "<hr style='border:0;border-top:1px solid #1e3a5f;margin:4px 0'>",
                                        unsafe_allow_html=True,
                                    )

                                st.caption(
                                    f"Showing top {len(top_anomalies)} anomaly frames "
                                    f"paired with nearest normal training frame "
                                    f"(out of {len(anomaly_frames)} total anomaly frames detected)."
                                )
                        else:
                            st.success("\u2705 No anomalies detected \u2014 exhibit appears to be operating normally.")

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a6a8a; font-size:0.8rem;'>"
    "JEPA Exhibit Defect Detection · Self-Supervised · Joint Embedding Predictive Architecture · 2026"
    "</p>",
    unsafe_allow_html=True
)


