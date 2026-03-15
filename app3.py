"""
JEPA Defect Detection — Generalized App v2 (app2.py)
=====================================================
Upload ANY normal video to train on.
Then analyze anomalies via:
  • Uploading a test video
  • Live webcam stream

v2: All 5 architectural upgrades active:
  1. Partial ViT encoder fine-tuning
  2. Cross-attention Spatial JEPA
  3. Multi-scale temporal (K=8 short + K=32 long)
  4. Deep SVDD energy model
  5. MC Dropout uncertainty-aware scoring
"""

import sys, json, time, tempfile, shutil
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JEPA Anomaly Detector v2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Light-theme CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #ffffff; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0f4fa 0%, #e8eef8 100%);
    border-right: 1px solid #d0dcea;
  }

  .metric-card {
    background: #f5f8ff;
    border: 1px solid #c8d8f0;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(80,120,200,0.08);
  }
  .metric-card .value { font-size: 2rem; font-weight: 700; }
  .metric-card .label { font-size: 0.82rem; color: #4a6a9a; margin-top: 4px; }

  .alert-normal {
    background: rgba(0,168,84,0.08);
    border-left: 4px solid #00a854;
    border-radius: 8px; padding: 12px 18px;
    color: #007a3d; font-weight: 600; font-size: 1.05rem;
  }
  .alert-anomaly {
    background: rgba(220,50,50,0.1);
    border-left: 4px solid #e03030;
    border-radius: 8px; padding: 12px 18px;
    color: #c02020; font-weight: 600; font-size: 1.05rem;
    animation: pulse 1s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.65} }

  .phase-pill {
    display: inline-block;
    background: linear-gradient(90deg,#1a6ab5,#2a85d5);
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.72rem; font-weight: 600;
    color: #fff; letter-spacing: .05em; text-transform: uppercase;
  }

  h1 { color: #1a2a4a !important; font-weight: 700 !important; }
  h2 { color: #2a4a7a !important; font-weight: 600 !important; }
  h3 { color: #3a5a8a !important; }

  .stProgress > div > div { background: linear-gradient(90deg,#1a6ab5,#00a8e8) !important; }

  .stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; }
  .stTabs [data-baseweb="tab"] {
    background: #eef3fb; border: 1px solid #c8d8f0;
    border-radius: 10px; color: #3a5a8a; font-weight: 600; padding: 10px 22px;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#1a6ab5,#0d4d8a);
    border-color: #1a6ab5; color: #fff !important;
  }

  .stButton > button {
    background: linear-gradient(135deg,#1a6ab5,#0d4d8a);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 11px 26px; font-size: .95rem; transition: all .2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg,#2280d5,#1260aa);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(26,106,181,.25);
  }

  label { color: #3a5a8a !important; }
  .upload-card {
    background: #f5f8ff; border: 2px dashed #c8d8f0;
    border-radius: 16px; padding: 28px; text-align: center;
    margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("normal_video_path", None)
_ss("normal_video_stem", None)
_ss("trained", False)
_ss("calibrated", False)
_ss("threshold", 1.0)
_ss("trainer_obj", None)
_ss("webcam_running", False)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
UPLOADS_DIR = Path(__file__).parent / "user_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_video(uploaded_file) -> Path:
    dest = UPLOADS_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def score_to_color(score: float, threshold: float) -> str:
    r = score / max(threshold, 1e-6)
    if r < 0.6:  return "#00a854"
    if r < 0.85: return "#f5a623"
    if r < 1.0:  return "#e05000"
    return "#e03030"


def make_loss_curve(loss_history: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=loss_history, mode="lines", line=dict(color="#1a6ab5", width=2)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
        xaxis=dict(title="Epoch", color="#3a5a8a", gridcolor="#dde8f5"),
        yaxis=dict(title="Loss",  color="#3a5a8a", gridcolor="#dde8f5"),
        margin=dict(l=10, r=10, t=10, b=10), height=220,
        font=dict(color="#3a5a8a"),
    )
    return fig


def make_score_timeline(results: list, threshold: float) -> go.Figure:
    frames = [r["frame_idx"] for r in results]
    scores = [r["score"]     for r in results]
    colors = [score_to_color(s, threshold) for s in scores]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=scores, mode="lines+markers",
        line=dict(color="#1a6ab5", width=2),
        marker=dict(color=colors, size=6),
        name="Anomaly Score",
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e03030",
                  line_width=2, annotation_text="Threshold",
                  annotation_font_color="#e03030")
    for r in [r for r in results if r["is_anomaly"]]:
        fig.add_vrect(x0=r["frame_idx"]-.5, x1=r["frame_idx"]+.5,
                      fillcolor="rgba(224,48,48,0.08)", line_width=0)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
        xaxis=dict(title="Frame", color="#3a5a8a", gridcolor="#dde8f5"),
        yaxis=dict(title="Score", color="#3a5a8a", gridcolor="#dde8f5"),
        margin=dict(l=10, r=10, t=10, b=10), height=260,
        showlegend=False, font=dict(color="#3a5a8a"),
    )
    return fig


def make_gauge(score: float, threshold: float) -> go.Figure:
    color = score_to_color(score, threshold)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 4),
        number={"font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, threshold * 2], "tickcolor": "#3a5a8a",
                     "tickfont": {"color": "#3a5a8a"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,           threshold*.6], "color": "rgba(0,168,84,0.10)"},
                {"range": [threshold*.6, threshold],   "color": "rgba(245,166,35,0.10)"},
                {"range": [threshold,   threshold*2],  "color": "rgba(224,48,48,0.12)"},
            ],
            "threshold": {"line": {"color": "#555", "width": 3},
                          "thickness": 0.8, "value": threshold},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20),
        height=190, font={"color": "#3a5a8a"},
    )
    return fig


def make_component_bar(components: dict) -> go.Figure:
    """5-component bar: Temporal-S, Temporal-L, Spatial, Energy, Uncertainty."""
    keys   = ["T-Short", "T-Long", "Spatial", "Energy", "Uncertainty"]
    vals   = [
        components.get("temporal",      0),
        components.get("temporal_long", 0),
        components.get("spatial",       0),
        components.get("energy",        0),
        components.get("uncertainty",   0),
    ]
    colors = ["#1a6ab5", "#00a8e8", "#7c4fff", "#ff6b35", "#00a854"]
    fig = go.Figure(go.Bar(x=keys, y=vals, marker_color=colors,
                           marker_line_color="rgba(0,0,0,0)"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
        xaxis=dict(color="#3a5a8a"), yaxis=dict(color="#3a5a8a", gridcolor="#dde8f5"),
        margin=dict(l=10, r=10, t=10, b=10), height=190,
        font=dict(color="#3a5a8a"),
    )
    return fig


def annotate_frame(frame_rgb: np.ndarray, label: str,
                   score=None, is_anomaly: bool = False) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    bc  = (0, 60, 255) if is_anomaly else (0, 180, 80)
    cv2.rectangle(bgr, (0,0), (bgr.shape[1]-1, bgr.shape[0]-1), bc, 4)
    ov = bgr.copy()
    cv2.rectangle(ov, (0,0), (bgr.shape[1], 36), (0,0,0), -1)
    cv2.addWeighted(ov, 0.5, bgr, 0.5, 0, bgr)
    cv2.putText(bgr, label, (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (255,255,255), 1, cv2.LINE_AA)
    if score is not None:
        cv2.putText(bgr, f"Score: {score:.3f}", (6,33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bc, 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_normal_frames_sample(stem: str, max_frames: int = 80) -> list:
    frames_dir = cfg.FRAMES_DIR / stem
    if not frames_dir.exists():
        # fallback: flat frames dir
        frames_dir = cfg.FRAMES_DIR
    if not frames_dir.exists():
        return []
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    step  = max(1, len(paths) // max_frames)
    out   = []
    for i, p in enumerate(paths[::step][:max_frames]):
        img = cv2.imread(str(p))
        if img is not None:
            out.append((i * step, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return out


def model_ready() -> bool:
    return (
        (cfg.CHECKPOINTS_DIR / "temporal.pt").exists() and
        (cfg.CHECKPOINTS_DIR / "spatial.pt").exists()
    )


def calibration_ready() -> bool:
    return cfg.CALIBRATION_FILE.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 JEPA Anomaly Detector v2")
    st.markdown('<span class="phase-pill">5 Upgrades · Upload & Detect</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🎛 Status")
    normal_set = st.session_state["normal_video_path"] is not None
    st.markdown(f"{'✅' if normal_set   else '⭕'} Normal Video Uploaded")
    st.markdown(f"{'✅' if model_ready()          else '⭕'} Models Trained")
    st.markdown(f"{'✅' if cfg.ENERGY_FILE.exists() else '⭕'} Energy Model (SVDD)")
    st.markdown(f"{'✅' if calibration_ready()    else '⭕'} Calibrated")

    st.markdown("---")
    st.markdown("### ⚙ Config")
    with st.expander("Parameters"):
        fps        = st.slider("Sample FPS",          1, 10, cfg.TARGET_FPS)
        window_k   = st.slider("Window Size K (short)", 4, 16, cfg.WINDOW_SIZE)
        epochs     = st.slider("Epochs (JEPA)",       5, 100, cfg.EPOCHS, step=5)
        svdd_ep    = st.slider("Epochs (SVDD)",       5, 50,  cfg.SVDD_EPOCHS, step=5)
        batch_size = st.slider("Batch Size",          4, 64,  cfg.BATCH_SIZE, step=4)
        alpha      = st.slider("α Temporal-Short",    0.0, 1.0, cfg.SCORE_ALPHA,      step=0.05)
        alpha_long = st.slider("α Temporal-Long",     0.0, 1.0, cfg.SCORE_ALPHA_LONG, step=0.05)
        beta_v     = st.slider("β Spatial",           0.0, 1.0, cfg.SCORE_BETA,       step=0.05)
        gamma      = st.slider("γ Energy (SVDD)",     0.0, 1.0, cfg.SCORE_GAMMA,      step=0.05)

    st.markdown(f"🖥 Device: **`{cfg.DEVICE.upper()}`**")
    st.markdown("---")
    st.caption("JEPA · ViT-B/16* · 5 Upgrades · 2026")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_train, tab_calib, tab_test, tab_eval = st.tabs([
    "🎓 1 · Train on Normal Video",
    "🎯 2 · Calibrate Threshold",
    "🔍 3 · Detect Anomalies",
    "📊 4 · Evaluate",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD NORMAL VIDEO + TRAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("## 🎓 Upload Normal Video & Train")
    st.markdown(
        "Upload a video that shows **normal, defect-free operation**. "
        "The model learns what 'normal' looks like — no labels needed.\n\n"
        "> **Training stages:** 3a — Short+Long Temporal + Cross-Attention Spatial JEPA  "
        "|  3b — Deep SVDD Energy Model"
    )
    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("### 📂 Step 1 — Upload Normal Video")
    uploaded_normal = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="normal_uploader",
    )

    if uploaded_normal:
        saved_path = save_uploaded_video(uploaded_normal)
        st.session_state["normal_video_path"] = saved_path
        st.session_state["normal_video_stem"] = saved_path.stem
        st.success(f"✅ Video saved: `{saved_path.name}`")
        st.video(str(saved_path))

    if st.session_state["normal_video_path"] is None:
        st.info("⬆ Upload a normal video above to begin.")
    else:
        st.markdown("---")
        st.markdown(f"### 🚀 Step 2 — Train on `{st.session_state['normal_video_stem']}`")

        start_training = st.button("🚀 Start Training (Stage 3a + 3b)", use_container_width=True)

        log_container   = st.empty()
        prog_container  = st.empty()
        chart_container = st.empty()

        if start_training:
            normal_path = st.session_state["normal_video_path"]
            log_lines, loss_hist = [], []

            def add_log(msg):
                log_lines.append(msg)
                log_container.markdown(
                    "**Log:**\n" + "\n".join(f"• {l}" for l in log_lines[-18:])
                )

            def on_progress(epoch, total, loss):
                loss_hist.append(loss)
                prog_container.progress(
                    epoch / total, text=f"Epoch {epoch}/{total} — Loss: {loss:.5f}"
                )
                if len(loss_hist) > 1:
                    chart_container.plotly_chart(
                        make_loss_curve(loss_hist),
                        key=f"train_live_{epoch}",
                    )

            try:
                from training.trainer import JEPATrainer
                cfg.TARGET_FPS  = fps
                cfg.WINDOW_SIZE = window_k

                trainer = JEPATrainer(
                    device=cfg.DEVICE,
                    log_callback=add_log,
                    progress_callback=on_progress,
                )

                # Stage 1: Extract
                add_log("📹 Extracting frames…")
                fp_bar = st.progress(0, "Sampling frames…")
                paths = trainer.extract_frames(
                    video_path=normal_path, fps=fps,
                    frame_progress=lambda c, t: fp_bar.progress(
                        min(c / max(t,1), 1.0), text=f"Frame {c}/{t}"
                    ),
                )
                fp_bar.empty()
                add_log(f"   → {len(paths)} frames extracted")

                # Stage 2: Encode (with partial ViT fine-tuning, Upgrade 1)
                add_log("🧠 Encoding with ViT-B/16* (partial domain adaptation)…")
                en_bar = st.progress(0, "Encoding…")
                trainer.encode_frames(
                    frame_paths=paths,
                    video_stem=normal_path.stem,
                    frame_progress=lambda c, t: en_bar.progress(
                        min(c / max(t,1), 1.0), text=f"Encoded {c}/{t}"
                    ),
                )
                en_bar.empty()

                # Stage 3a: Joint Temporal + Spatial
                add_log("🎯 Stage 3a — Short+Long Temporal + Cross-Attention Spatial JEPA…")
                loss_history = trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=cfg.LEARNING_RATE
                )
                (cfg.CHECKPOINTS_DIR / "loss_history.json").write_text(json.dumps(loss_history))
                chart_container.plotly_chart(
                    make_loss_curve(loss_history), key="train_final"
                )

                # Stage 3b: Deep SVDD Energy Model (Upgrade 4)
                add_log("⚡ Stage 3b — Training Deep SVDD energy model…")
                svdd_bar = st.progress(0, "SVDD training…")
                def on_svdd(ep, tot, loss):
                    svdd_bar.progress(ep / tot, text=f"SVDD {ep}/{tot} — {loss:.5f}")
                trainer.on_progress = on_svdd
                svdd_hist = trainer.train_energy(epochs=svdd_ep)
                svdd_bar.empty()
                add_log(f"✅ SVDD done — final loss: {svdd_hist[-1]:.5f}")

                st.session_state["trained"]     = True
                st.session_state["trainer_obj"] = trainer
                st.success(
                    f"✅ All training complete — JEPA: {loss_history[-1]:.5f} | "
                    f"SVDD: {svdd_hist[-1]:.5f}"
                )

            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback; st.code(traceback.format_exc())

        # Previous loss curve
        lf = cfg.CHECKPOINTS_DIR / "loss_history.json"
        if lf.exists() and not start_training:
            prior = json.loads(lf.read_text())
            st.markdown("**Previous Training Loss:**")
            st.plotly_chart(make_loss_curve(prior), key="train_prior")
            st.caption(f"Epochs: {len(prior)} | Final loss: {prior[-1]:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CALIBRATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_calib:
    st.markdown("## 🎯 Calibrate Anomaly Threshold")
    st.markdown(
        "Runs all trained models on **normal video embeddings** to compute per-frame scores, "
        "then sets the threshold at the **97th percentile** and saves per-component "
        "normalisation scales."
    )

    if not model_ready():
        st.warning("⚠ Complete training in Tab 1 first.")
    else:
        calib_file = cfg.CALIBRATION_FILE
        if calib_file.exists():
            c = json.loads(calib_file.read_text())
            st.success(f"✅ Calibration done — Threshold: **{c['threshold']:.4f}**")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Threshold",  f"{c['threshold']:.4f}")
            cc2.metric("t_scale",    f"{c.get('t_scale', 1.0):.4f}")
            cc3.metric("s_scale",    f"{c.get('s_scale', 1.0):.4f}")
            cc4.metric("e_scale",    f"{c.get('e_scale', 1.0):.4f}")
            st.session_state["threshold"] = c["threshold"]

        run_calib = st.button("🔄 Run Calibration", use_container_width=False)

        if run_calib:
            stem = st.session_state.get("normal_video_stem") or cfg.NORMAL_VIDEO.stem
            cls_path   = cfg.EMBEDDINGS_DIR / f"{stem}_cls.npy"
            patch_path = cfg.EMBEDDINGS_DIR / f"{stem}_patches.npy"

            if not cls_path.exists():
                st.error("Embeddings not found — run training first.")
            else:
                try:
                    from models.temporal_transformer import TemporalTransformer
                    from models.spatial_jepa import SpatialJEPAHead
                    from anomaly.energy_model import EnergyModel
                    from anomaly.scorer import AnomalyScorer
                    from training.calibration import calibrate_threshold, calibrate_component_scales

                    long_seq_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)

                    clog = st.empty(); cbar = st.progress(0)

                    clog.info("Loading Short Temporal model…")
                    ts = TemporalTransformer(window_size=cfg.WINDOW_SIZE).to(cfg.DEVICE)
                    ts.load(cfg.CHECKPOINTS_DIR / "temporal.pt", cfg.DEVICE)

                    tl = TemporalTransformer(window_size=long_seq_len).to(cfg.DEVICE)
                    if (cfg.CHECKPOINTS_DIR / "temporal_long.pt").exists():
                        tl.load(cfg.CHECKPOINTS_DIR / "temporal_long.pt", cfg.DEVICE)

                    clog.info("Loading Cross-Attention Spatial JEPA…")
                    sm = SpatialJEPAHead().to(cfg.DEVICE)
                    sm.load(cfg.CHECKPOINTS_DIR / "spatial.pt", cfg.DEVICE)

                    clog.info("Loading Deep SVDD Energy model…")
                    em = EnergyModel().to(cfg.DEVICE)
                    if cfg.ENERGY_FILE.exists():
                        em.load(cfg.ENERGY_FILE, cfg.DEVICE)

                    clog.info("Loading embeddings…")
                    cls_embs   = np.load(str(cls_path))
                    patch_embs = np.load(str(patch_path))

                    scorer = AnomalyScorer(
                        temporal_short = ts,
                        temporal_long  = tl,
                        spatial_head   = sm,
                        energy_model   = em,
                        device         = cfg.DEVICE,
                    )

                    clog.info("Scoring normal frames (may take a minute)…")
                    normal_scores, t_errs, tl_errs, s_errs, e_errs = [], [], [], [], []
                    N = len(cls_embs)
                    for i in range(N):
                        s, comp = scorer.push_and_score(cls_embs[i], patch_embs[i])
                        normal_scores.append(s)
                        t_errs.append(comp.get("temporal",      0.0))
                        tl_errs.append(comp.get("temporal_long", 0.0))
                        s_errs.append(comp.get("spatial",       0.0))
                        e_errs.append(comp.get("energy",        0.0))
                        cbar.progress((i+1)/N, text=f"Frame {i+1}/{N}")

                    cbar.empty(); clog.empty()
                    normal_scores = np.array(normal_scores)

                    scales = calibrate_component_scales(
                        np.array(t_errs), np.array(tl_errs),
                        np.array(s_errs), np.array(e_errs),
                    )
                    thr = calibrate_threshold(normal_scores, component_scales=scales)
                    st.session_state["threshold"] = thr
                    st.session_state["calibrated"] = True

                    st.success(f"✅ Threshold set to **{thr:.4f}**")
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Threshold", f"{thr:.4f}")
                    cc2.metric("t_scale",   f"{scales['t_scale']:.4f}")
                    cc3.metric("s_scale",   f"{scales['s_scale']:.4f}")
                    cc4.metric("e_scale",   f"{scales['e_scale']:.4f}")

                    fig = px.histogram(
                        x=normal_scores, nbins=50,
                        labels={"x": "Anomaly Score", "y": "Count"},
                        color_discrete_sequence=["#1a6ab5"],
                    )
                    fig.add_vline(x=thr, line_dash="dash", line_color="#e03030",
                                  annotation_text="Threshold",
                                  annotation_font_color="#e03030")
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
                        font=dict(color="#3a5a8a"),
                        xaxis=dict(gridcolor="#dde8f5"), yaxis=dict(gridcolor="#dde8f5"),
                        margin=dict(l=10, r=10, t=30, b=10), height=240,
                        title="Normal Score Distribution", title_font_color="#2a4a7a",
                    )
                    st.plotly_chart(fig, key="calib_hist")

                except Exception as e:
                    st.error(f"Calibration failed: {e}")
                    import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DETECT (Upload OR Webcam)
# ══════════════════════════════════════════════════════════════════════════════
with tab_test:
    st.markdown("## 🔍 Detect Anomalies")

    if not model_ready() or not calibration_ready():
        st.warning("⚠ Complete training (Tab 1) and calibration (Tab 2) first.")
    else:
        from training.calibration import load_calibration
        calib_data = load_calibration()
        threshold  = calib_data["threshold"] if calib_data else st.session_state["threshold"]

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("### Choose Analysis Mode")
        with col_right:
            threshold = st.number_input(
                "Override Threshold", value=float(threshold),
                min_value=0.001, step=0.01, format="%.4f",
                key="thr_override",
            )

        mode = st.radio(
            "Analysis mode",
            ["📁 Upload Test Video", "📹 Live Webcam Stream"],
            horizontal=True, label_visibility="collapsed",
        )
        st.markdown("---")

        # ── Shared model loader ───────────────────────────────────────────────
        @st.cache_resource
        def get_scorer_and_encoder():
            from inference.pipeline import build_scorer
            scorer, encoder, _ = build_scorer(cfg.DEVICE)
            return scorer, encoder

        # ════════════════════════════════════════════════════════════════════
        # MODE A — UPLOAD TEST VIDEO
        # ════════════════════════════════════════════════════════════════════
        if mode == "📁 Upload Test Video":
            st.markdown("### 📂 Upload Test / Suspect Video")

            uploaded_test = st.file_uploader(
                "Choose test video", type=["mp4","avi","mov","mkv"],
                key="test_uploader",
            )

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                max_frames_n = st.slider("Max frames (0 = all)", 0, 600, 0, step=10)
            with col_opt2:
                show_every = st.slider("Show every Nth frame", 1, 20, 5)

            run_test = st.button("▶ Run Detection", use_container_width=True,
                                  disabled=(uploaded_test is None))

            if run_test and uploaded_test:
                test_path = save_uploaded_video(uploaded_test)
                st.info(f"Analyzing `{test_path.name}` …")

                try:
                    scorer, encoder = get_scorer_and_encoder()
                    scorer.alpha      = alpha
                    scorer.alpha_long = alpha_long
                    scorer.beta       = beta_v
                    scorer.gamma      = gamma
                    scorer.reset()

                    from inference.pipeline import run_video_inference

                    cap_tmp = cv2.VideoCapture(str(test_path))
                    total_src  = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
                    src_fps    = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
                    cap_tmp.release()
                    est_total = total_src // max(1, round(src_fps / cfg.TARGET_FPS))
                    max_f = max_frames_n if max_frames_n > 0 else None
                    if max_f: est_total = min(est_total, max_f)

                    total_prog  = st.progress(0, text="Starting…")
                    score_chart = st.empty()
                    frame_disp  = st.empty()
                    alert_disp  = st.empty()
                    results     = []

                    def frame_cb(frame_rgb, result, fidx):
                        results.append(result)
                        pct = min((fidx+1) / max(est_total,1), 1.0)
                        total_prog.progress(
                            pct, text=f"Frame {fidx+1}/{est_total} — Score: {result['score']:.4f}"
                        )
                        if len(results) % 5 == 0 and len(results) > 1:
                            score_chart.plotly_chart(
                                make_score_timeline(results, threshold),
                                key=f"tl_live_{len(results)}",
                            )
                        if result["is_anomaly"]:
                            alert_disp.markdown(
                                '<div class="alert-anomaly">🚨 ANOMALY DETECTED</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            alert_disp.markdown(
                                '<div class="alert-normal">✅ Normal</div>',
                                unsafe_allow_html=True
                            )
                        if fidx % show_every == 0:
                            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            clr = (0,180,80) if not result["is_anomaly"] else (0,60,255)
                            cv2.putText(bgr, f"Score: {result['score']:.3f}",
                                        (8,24), cv2.FONT_HERSHEY_SIMPLEX, .7, clr, 2)
                            if result["is_anomaly"]:
                                cv2.rectangle(bgr,(0,0),(bgr.shape[1]-1,bgr.shape[0]-1),
                                              (0,60,255),4)
                            frame_disp.image(
                                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {fidx}",
                                use_container_width=True,
                            )

                    run_video_inference(
                        test_path, scorer, encoder, threshold,
                        frame_callback=frame_cb, max_frames=max_f,
                    )
                    total_prog.progress(1.0, text="✅ Complete!")

                    # ── Summary ───────────────────────────────────────────────
                    if results:
                        st.markdown("---")
                        st.markdown("### 📊 Summary")
                        n_tot  = len(results)
                        n_anom = sum(1 for r in results if r["is_anomaly"])
                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric("Total Frames",   n_tot)
                        c2.metric("Anomaly Frames", n_anom,
                                  delta=f"{n_anom/n_tot*100:.1f}% flagged",
                                  delta_color="inverse")
                        c3.metric("Max Score",  f"{max(r['score'] for r in results):.4f}")
                        c4.metric("Mean Score", f"{np.mean([r['score'] for r in results]):.4f}")

                        st.plotly_chart(
                            make_score_timeline(results, threshold), key="tl_final"
                        )

                        # Gauge + component bar for last frame
                        last_r = results[-1]
                        st.markdown("#### Last Frame Score Breakdown")
                        gc, bc_col = st.columns(2)
                        with gc:
                            st.plotly_chart(make_gauge(last_r["score"], threshold),
                                            key="gauge_up")
                        with bc_col:
                            st.plotly_chart(make_component_bar(last_r), key="bar_up")

                        st.markdown(
                            "*Component legend: T-Short = Temporal (K=8, uncertainty-damped) · "
                            "T-Long = Temporal (K=32 drift) · Spatial = Cross-attn patches · "
                            "Energy = Deep SVDD · Uncertainty = MC dropout variance*"
                        )

                        # ── Anomaly frame list ───────────────────────────────
                        anomaly_frames = [r for r in results if r["is_anomaly"]]
                        if anomaly_frames:
                            st.markdown(f"#### 🚨 {len(anomaly_frames)} Anomaly Frame(s)")
                            for r in anomaly_frames[:20]:
                                st.write(
                                    f"• Frame **{r['frame_idx']}** — "
                                    f"Score: `{r['score']:.4f}` "
                                    f"(T-S: {r['temporal']:.3f}, "
                                    f"T-L: {r.get('temporal_long',0):.3f}, "
                                    f"Sp: {r['spatial']:.3f}, "
                                    f"E: {r.get('energy',0):.3f}, "
                                    f"Unc: {r.get('uncertainty',0):.4f})"
                                )
                            if len(anomaly_frames) > 20:
                                st.caption(f"… and {len(anomaly_frames) - 20} more")

                            # Side-by-side comparison
                            st.markdown("---")
                            st.markdown("### 🖼 Normal vs Anomaly — Side-by-Side")
                            stem = st.session_state.get("normal_video_stem") or cfg.NORMAL_VIDEO.stem
                            normal_sample = load_normal_frames_sample(stem)
                            top_anom = sorted(anomaly_frames,
                                              key=lambda r: r["score"], reverse=True)[:10]
                            if normal_sample:
                                for pi, ar in enumerate(top_anom):
                                    best_ni = min(range(len(normal_sample)),
                                                  key=lambda i: abs(normal_sample[i][0]-ar["frame_idx"]))
                                    norm_idx, norm_fr = normal_sample[best_ni]
                                    na = annotate_frame(norm_fr, f"NORMAL #{norm_idx}", is_anomaly=False)
                                    aa = annotate_frame(ar["frame_rgb"],
                                                        f"ANOMALY #{ar['frame_idx']}",
                                                        score=ar["score"], is_anomaly=True)
                                    col_n, col_a = st.columns(2)
                                    with col_n:
                                        st.image(na, caption=f"✅ Normal · frame {norm_idx}",
                                                 use_container_width=True)
                                    with col_a:
                                        st.image(aa,
                                                 caption=f"🚨 Anomaly · frame {ar['frame_idx']} · {ar['score']:.3f}",
                                                 use_container_width=True)
                                    st.markdown(
                                        "<hr style='border:0;border-top:1px solid #dde8f5;margin:4px 0'>",
                                        unsafe_allow_html=True)
                                st.caption(f"Top {len(top_anom)} anomaly frames.")
                            else:
                                st.warning("⚠ No cached training frames found. Re-run training first.")
                        else:
                            st.success("✅ No anomalies detected — video looks normal.")

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    import traceback; st.code(traceback.format_exc())

        # ════════════════════════════════════════════════════════════════════
        # MODE B — LIVE WEBCAM STREAM
        # ════════════════════════════════════════════════════════════════════
        else:
            st.markdown("### 📹 Live Webcam Stream")
            st.markdown(
                "The webcam feed is analyzed frame-by-frame. "
                "Each frame is scored using all 5 model components."
            )

            col_l, col_r = st.columns(2)
            with col_l:
                cam_index = st.number_input("Camera index (0 = default)", 0, 5, 0, step=1)
            with col_r:
                analyze_every = st.slider("Analyze every Nth frame", 1, 10, 2)

            start_stream = st.button("▶ Start Stream", use_container_width=True)
            stop_holder  = st.empty()

            live_frame_disp  = st.empty()
            live_alert_disp  = st.empty()
            live_score_disp  = st.empty()
            live_chart_disp  = st.empty()
            live_gauge_disp  = st.empty()

            if start_stream:
                try:
                    scorer, encoder = get_scorer_and_encoder()
                    scorer.alpha      = alpha
                    scorer.alpha_long = alpha_long
                    scorer.beta       = beta_v
                    scorer.gamma      = gamma
                    scorer.reset()

                    cap = cv2.VideoCapture(int(cam_index))
                    if not cap.isOpened():
                        st.error(f"Cannot open camera {cam_index}.")
                    else:
                        st.session_state["webcam_running"] = True
                        stop_btn = stop_holder.button("⏹ Stop Stream", key="stop_webcam")

                        frame_count    = 0
                        webcam_results = []

                        while st.session_state["webcam_running"]:
                            ret, frame_bgr = cap.read()
                            if not ret:
                                st.warning("Webcam feed ended.")
                                break

                            frame_count += 1
                            frame_rgb = cv2.cvtColor(
                                cv2.resize(frame_bgr, (cfg.FRAME_SIZE, cfg.FRAME_SIZE)),
                                cv2.COLOR_BGR2RGB
                            )

                            if frame_count % analyze_every == 0:
                                cls_emb, patch_emb = encoder.encode_frame_np(frame_rgb)
                                score, components  = scorer.push_and_score(cls_emb, patch_emb)
                                is_anomaly         = score > threshold

                                webcam_results.append({
                                    "frame_idx": frame_count,
                                    "score":     score,
                                    "is_anomaly": is_anomaly,
                                    **components,
                                })

                                # Annotate display frame
                                disp_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                                clr = (0, 60, 255) if is_anomaly else (0, 180, 80)
                                cv2.putText(disp_bgr, f"Score: {score:.3f}",
                                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, clr, 2)
                                cv2.putText(disp_bgr,
                                            f"E:{components.get('energy',0):.3f}  "
                                            f"Unc:{components.get('uncertainty',0):.4f}",
                                            (8, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, clr, 1)
                                if is_anomaly:
                                    cv2.rectangle(disp_bgr,(0,0),
                                                  (disp_bgr.shape[1]-1, disp_bgr.shape[0]-1),
                                                  (0,60,255), 4)

                                live_frame_disp.image(
                                    cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB),
                                    caption=f"Frame {frame_count} | Score {score:.3f}",
                                    use_container_width=True,
                                )
                                if is_anomaly:
                                    live_alert_disp.markdown(
                                        '<div class="alert-anomaly">🚨 ANOMALY DETECTED</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    live_alert_disp.markdown(
                                        '<div class="alert-normal">✅ Normal Operation</div>',
                                        unsafe_allow_html=True
                                    )
                                live_score_disp.metric(
                                    "Composite Score", f"{score:.4f}",
                                    delta=f"threshold={threshold:.3f}"
                                )
                                if len(webcam_results) > 2:
                                    live_chart_disp.plotly_chart(
                                        make_score_timeline(webcam_results[-100:], threshold),
                                        key=f"wcam_{frame_count}",
                                    )
                                    live_gauge_disp.plotly_chart(
                                        make_gauge(score, threshold),
                                        key=f"wcam_gauge_{frame_count}",
                                    )

                            if stop_btn:
                                st.session_state["webcam_running"] = False
                                break
                            time.sleep(0.03)

                        cap.release()
                        st.session_state["webcam_running"] = False
                        st.info("📹 Stream stopped.")

                        if webcam_results:
                            n_anom = sum(1 for r in webcam_results if r["is_anomaly"])
                            st.metric("Total Frames Analyzed", len(webcam_results))
                            st.metric("Anomaly Frames",        n_anom)

                except Exception as e:
                    st.error(f"Webcam error: {e}")
                    import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("## 📊 Model Evaluation")
    st.markdown(
        "Upload a **labelled test video** and tell the model which frames are anomalous. "
        "The app computes research-grade metrics: AUROC, Precision/Recall/F1, "
        "Detection Delay, False Positive Rate, component ablation, and robustness results."
    )

    if not model_ready() or not calibration_ready():
        st.warning("⚠ Complete training (Tab 1) and calibration (Tab 2) first.")
    else:
        # ── load threshold ────────────────────────────────────────────────────
        import json as _json
        _calib = _json.loads(cfg.CALIBRATION_FILE.read_text())
        eval_threshold = _calib.get("threshold", 1.0)

        # ─────────────────────────────────────────────────────────────────────
        # Sub-sections
        # ─────────────────────────────────────────────────────────────────────
        ev1, ev2, ev3 = st.tabs([
            "🏷 Label & Run",
            "📈 Metrics & Plots",
            "🧪 Robustness Test",
        ])

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB A — Label & Run
        # ══════════════════════════════════════════════════════════════════════
        with ev1:
            st.markdown("### Step A — Upload Labelled Test Video")
            eval_video = st.file_uploader(
                "Test video (MP4/AVI/MOV/MKV)",
                type=["mp4","avi","mov","mkv"],
                key="eval_uploader",
            )

            st.markdown("### Step B — Define Anomaly Segments")
            st.markdown(
                "Enter frame ranges where anomalies occur. "
                "Leave blank if the video is fully normal."
            )
            n_segments = st.number_input("Number of anomaly segments", 1, 10, 1, step=1, key="n_seg")
            segments = []
            cols_hdr = st.columns(2)
            cols_hdr[0].markdown("**Start frame**")
            cols_hdr[1].markdown("**End frame (inclusive)**")
            for i in range(int(n_segments)):
                c1, c2 = st.columns(2)
                s = c1.number_input(f"Segment {i+1} start", 0, 99999, 0,  key=f"seg_s_{i}")
                e = c2.number_input(f"Segment {i+1} end",   0, 99999, 100, key=f"seg_e_{i}")
                segments.append((int(s), int(e)))

            st.markdown("### Step C — Options")
            c1, c2, c3 = st.columns(3)
            with c1:
                eval_max_frames = st.slider("Max frames (0=all)", 0, 2000, 0, step=50, key="eval_maxf")
            with c2:
                eval_mask = st.toggle("🧍 Ignore Humans (YOLO)", value=True, key="eval_mask")
            with c3:
                eval_thr = st.number_input(
                    "Threshold override (0=auto)",
                    value=0.0, min_value=0.0, step=0.01, format="%.4f", key="eval_thr"
                )

            run_eval = st.button("▶ Run Evaluation", use_container_width=True,
                                 disabled=(eval_video is None))

            if run_eval and eval_video is not None:
                eval_path = save_uploaded_video(eval_video)
                threshold_used = eval_thr if eval_thr > 0 else eval_threshold

                try:
                    from inference.pipeline import build_scorer, run_video_inference
                    from sklearn.metrics import (
                        roc_auc_score, precision_score, recall_score,
                        f1_score, roc_curve,
                    )

                    _masker_eval = None
                    if eval_mask:
                        with st.spinner("🧍 Loading YOLO masker…"):
                            from preprocessing.human_mask import HumanMaskFilter
                            _masker_eval = HumanMaskFilter()

                    with st.spinner("🔧 Loading models…"):
                        scorer_eval, encoder_eval, _ = build_scorer(cfg.DEVICE)
                        scorer_eval.reset()

                    eval_prog  = st.progress(0, text="Running inference…")
                    eval_results = []

                    cap_e    = cv2.VideoCapture(str(eval_path))
                    tot_src  = int(cap_e.get(cv2.CAP_PROP_FRAME_COUNT))
                    efps     = cap_e.get(cv2.CAP_PROP_FPS) or 25.0
                    cap_e.release()
                    est_e    = tot_src // max(1, round(efps / cfg.TARGET_FPS))
                    max_fe   = eval_max_frames if eval_max_frames > 0 else None
                    if max_fe: est_e = min(est_e, max_fe)

                    # Raw component scores for ablation
                    raw_temporal, raw_temporal_long = [], []
                    raw_spatial, raw_energy         = [], []

                    def _eval_cb(frame_rgb, result, fidx):
                        eval_results.append(result)
                        raw_temporal.append(result.get("temporal",      0.0))
                        raw_temporal_long.append(result.get("temporal_long", 0.0))
                        raw_spatial.append(result.get("spatial",        0.0))
                        raw_energy.append(result.get("energy",          0.0))
                        eval_prog.progress(
                            min((fidx+1)/max(est_e,1), 1.0),
                            text=f"Frame {fidx+1}/{est_e}"
                        )

                    run_video_inference(
                        eval_path, scorer_eval, encoder_eval, threshold_used,
                        frame_callback=_eval_cb, max_frames=max_fe,
                        human_masker=_masker_eval,
                    )
                    eval_prog.progress(1.0, text="✅ Inference complete!")

                    # ── Build y_true ──────────────────────────────────────────
                    total_eval_frames = len(eval_results)
                    y_true = np.zeros(total_eval_frames, dtype=int)
                    for (seg_s, seg_e) in segments:
                        y_true[seg_s : min(seg_e+1, total_eval_frames)] = 1

                    y_scores  = np.array([r["score"] for r in eval_results])
                    y_pred    = (y_scores >= threshold_used).astype(int)

                    # ── Save to session for Metrics tab ───────────────────────
                    st.session_state["eval_y_true"]          = y_true
                    st.session_state["eval_y_scores"]        = y_scores
                    st.session_state["eval_y_pred"]          = y_pred
                    st.session_state["eval_results"]         = eval_results
                    st.session_state["eval_threshold"]       = threshold_used
                    st.session_state["eval_segments"]        = segments
                    st.session_state["eval_raw_temporal"]    = np.array(raw_temporal)
                    st.session_state["eval_raw_tl"]         = np.array(raw_temporal_long)
                    st.session_state["eval_raw_spatial"]     = np.array(raw_spatial)
                    st.session_state["eval_raw_energy"]      = np.array(raw_energy)

                    n_anom_gt  = int(y_true.sum())
                    n_pred_pos = int(y_pred.sum())
                    st.success(
                        f"✅ Done — {total_eval_frames} frames | "
                        f"{n_anom_gt} ground-truth anomaly frames | "
                        f"{n_pred_pos} predicted anomaly frames → **go to Metrics & Plots tab**"
                    )

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    import traceback; st.code(traceback.format_exc())

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB B — Metrics & Plots
        # ══════════════════════════════════════════════════════════════════════
        with ev2:
            if "eval_y_true" not in st.session_state:
                st.info("⬅ Run evaluation in the **Label & Run** tab first.")
            else:
                from sklearn.metrics import (
                    roc_auc_score, precision_score, recall_score,
                    f1_score, roc_curve, confusion_matrix,
                )

                y_true    = st.session_state["eval_y_true"]
                y_scores  = st.session_state["eval_y_scores"]
                y_pred    = st.session_state["eval_y_pred"]
                threshold_used = st.session_state["eval_threshold"]
                segments  = st.session_state["eval_segments"]
                results   = st.session_state["eval_results"]
                raw_t     = st.session_state["eval_raw_temporal"]
                raw_tl    = st.session_state["eval_raw_tl"]
                raw_s     = st.session_state["eval_raw_spatial"]
                raw_e     = st.session_state["eval_raw_energy"]

                has_pos = y_true.sum() > 0
                has_neg = (1 - y_true).sum() > 0

                # ── 1. Core metrics ───────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 1️⃣ Core Metrics")
                m1,m2,m3,m4 = st.columns(4)
                if has_pos and has_neg:
                    auroc = roc_auc_score(y_true, y_scores)
                    m1.metric("AUROC", f"{auroc:.4f}",
                              delta="🔥 Excellent" if auroc > 0.9 else ("Good" if auroc > 0.8 else "Weak"))
                else:
                    auroc = None
                    m1.metric("AUROC", "N/A", delta="Need both classes")

                if y_pred.sum() > 0:
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec  = recall_score(y_true, y_pred, zero_division=0)
                    f1   = f1_score(y_true, y_pred, zero_division=0)
                else:
                    prec = rec = f1 = 0.0
                m2.metric("Precision", f"{prec:.4f}")
                m3.metric("Recall",    f"{rec:.4f}")
                m4.metric("F1 Score",  f"{f1:.4f}")

                # ── 2. Detection Delay ────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 2️⃣ Detection Delay")
                st.caption("Frames from anomaly start → first predicted anomaly. Lower = better real-time detection.")
                delay_rows = []
                for idx, (seg_s, seg_e) in enumerate(segments):
                    # first predicted anomaly at or after seg_s
                    found = None
                    for fi in range(seg_s, min(seg_e+1, len(y_pred))):
                        if y_pred[fi] == 1:
                            found = fi
                            break
                    delay = (found - seg_s) if found is not None else None
                    delay_rows.append({
                        "Segment": f"{seg_s}→{seg_e}",
                        "First Detected Frame": found if found is not None else "❌ Missed",
                        "Delay (frames)": delay if delay is not None else "—",
                    })
                if delay_rows:
                    st.table(delay_rows)

                # ── 3. False Positive Rate ────────────────────────────────────
                st.markdown("---")
                st.markdown("### 3️⃣ False Positive Rate on Normal Frames")
                if has_neg:
                    normal_mask = (y_true == 0)
                    n_normal    = normal_mask.sum()
                    n_fp        = int(y_pred[normal_mask].sum())
                    fpr_val     = n_fp / max(n_normal, 1)
                    c1, c2, c3  = st.columns(3)
                    c1.metric("Normal Frames",      int(n_normal))
                    c2.metric("False Positives",    n_fp)
                    c3.metric("False Positive Rate", f"{fpr_val:.4f}",
                              delta="✅ Stable" if fpr_val < 0.05 else "⚠ High FPR")
                else:
                    st.warning("No normal frames in the ground truth — add segment labels.")

                # ── 4. Score vs Frame plot ────────────────────────────────────
                st.markdown("---")
                st.markdown("### 4️⃣ Score vs Frame — Visual Proof")
                frames_x = list(range(len(y_scores)))
                fig_sv = go.Figure()
                # colour by true label
                colors_sv = ["#e03030" if y_true[i] else "#1a6ab5" for i in frames_x]
                fig_sv.add_trace(go.Scatter(
                    x=frames_x, y=y_scores.tolist(), mode="markers+lines",
                    marker=dict(color=colors_sv, size=5),
                    line=dict(color="#aac4e8", width=1),
                    name="Score (red=GT anomaly)",
                ))
                fig_sv.add_hline(y=threshold_used, line_dash="dash", line_color="#e03030",
                                 line_width=2, annotation_text="Threshold",
                                 annotation_font_color="#e03030")
                for (seg_s, seg_e) in segments:
                    fig_sv.add_vrect(
                        x0=seg_s-0.5, x1=min(seg_e+0.5, len(y_scores)-1),
                        fillcolor="rgba(224,48,48,0.10)", line_width=0,
                        annotation_text="Anomaly zone", annotation_position="top left",
                        annotation_font_color="#e03030",
                    )
                fig_sv.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
                    xaxis=dict(title="Frame Index", color="#3a5a8a", gridcolor="#dde8f5"),
                    yaxis=dict(title="Anomaly Score", color="#3a5a8a", gridcolor="#dde8f5"),
                    font=dict(color="#3a5a8a"), height=300,
                    margin=dict(l=10,r=10,t=30,b=10),
                    title="Anomaly Score Timeline — Red dots = GT anomaly frames",
                    title_font_color="#2a4a7a",
                )
                st.plotly_chart(fig_sv, key="score_vs_frame")

                # ── 5. ROC Curve ──────────────────────────────────────────────
                if auroc is not None:
                    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_scores)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr_roc.tolist(), y=tpr_roc.tolist(),
                        mode="lines", line=dict(color="#1a6ab5", width=2.5),
                        name=f"Full Model (AUC={auroc:.3f})",
                        fill="tozeroy", fillcolor="rgba(26,106,181,0.08)",
                    ))
                    fig_roc.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,
                                      line=dict(dash="dot",color="#aaaaaa"))
                    fig_roc.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
                        xaxis=dict(title="False Positive Rate", color="#3a5a8a", gridcolor="#dde8f5"),
                        yaxis=dict(title="True Positive Rate",  color="#3a5a8a", gridcolor="#dde8f5"),
                        font=dict(color="#3a5a8a"), height=300,
                        margin=dict(l=10,r=10,t=30,b=10),
                        title=f"ROC Curve — AUROC = {auroc:.4f}",
                        title_font_color="#2a4a7a",
                    )
                    st.plotly_chart(fig_roc, key="roc_curve")

                # ── 6. Component Ablation ────────────────────────────────────
                st.markdown("---")
                st.markdown("### 5️⃣ Component-Level Ablation Study")
                st.caption(
                    "AUROC computed for each sub-signal individually. "
                    "Proves the composite model is better than any single component."
                )
                if has_pos and has_neg:
                    ablation_rows = []
                    components_map = {
                        "Temporal-Short only":  raw_t,
                        "Temporal-Long only":   raw_tl,
                        "Spatial only":         raw_s,
                        "Energy (SVDD) only":   raw_e,
                        "**Full Composite ★**": y_scores,
                    }
                    for label, scores_arr in components_map.items():
                        try:
                            a = roc_auc_score(y_true, scores_arr)
                        except Exception:
                            a = float("nan")
                        ablation_rows.append({"Model Variant": label, "AUROC": f"{a:.4f}"})

                    # Bar chart
                    abl_labels = [r["Model Variant"].replace("**","").replace("★","★") for r in ablation_rows]
                    abl_vals   = [float(r["AUROC"]) if r["AUROC"] != "nan" else 0 for r in ablation_rows]
                    abl_colors = ["#1a6ab5","#00a8e8","#7c4fff","#ff6b35","#00a854"]
                    fig_abl = go.Figure(go.Bar(
                        x=abl_labels, y=abl_vals,
                        marker_color=abl_colors,
                        text=[f"{v:.4f}" for v in abl_vals],
                        textposition="outside",
                    ))
                    fig_abl.add_hline(y=0.9, line_dash="dot", line_color="#888",
                                      annotation_text="0.9 target",
                                      annotation_font_color="#888")
                    fig_abl.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
                        yaxis=dict(range=[0,1.05], title="AUROC", color="#3a5a8a", gridcolor="#dde8f5"),
                        xaxis=dict(color="#3a5a8a"),
                        font=dict(color="#3a5a8a"), height=300,
                        margin=dict(l=10,r=10,t=30,b=10),
                        title="Ablation Study — per-component AUROC",
                        title_font_color="#2a4a7a",
                    )
                    st.plotly_chart(fig_abl, key="ablation_chart")
                    st.table(ablation_rows)
                else:
                    st.warning("Need at least one anomaly segment and normal frames for ablation.")

                # ── Summary box ───────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### ✅ Evaluation Checklist")
                checklist = {
                    "AUROC computed": auroc is not None,
                    "Precision / Recall / F1": True,
                    "Detection delay measured": len(delay_rows) > 0,
                    "False positive rate": has_neg,
                    "Score vs frame plot": True,
                    "Ablation study": has_pos and has_neg,
                }
                for item, done in checklist.items():
                    st.markdown(f"{'✅' if done else '⭕'} {item}")

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB C — Robustness Test
        # ══════════════════════════════════════════════════════════════════════
        with ev3:
            st.markdown("### 🧪 Robustness Test")
            st.markdown(
                "Upload a **normal** video processed under various conditions "
                "(brightness change, noise, slight resolution drop). "
                "If the mean anomaly score stays **below the threshold**, the model is robust."
            )

            rob_video = st.file_uploader(
                "Normal video for robustness test",
                type=["mp4","avi","mov","mkv"],
                key="rob_uploader",
            )

            st.markdown("**Perturbations to apply:**")
            rb1, rb2, rb3 = st.columns(3)
            with rb1:
                brightness_delta = st.slider("Brightness shift (±)", -80, 80, 30, step=5, key="rb_bright")
            with rb2:
                noise_std = st.slider("Gaussian noise σ", 0, 50, 15, step=5, key="rb_noise")
            with rb3:
                scale_factor = st.slider("Resolution scale (%)", 50, 100, 75, step=5, key="rb_scale")

            run_rob = st.button("🧪 Run Robustness Test", use_container_width=False,
                                disabled=(rob_video is None))

            if run_rob and rob_video is not None:
                rob_path = save_uploaded_video(rob_video)
                try:
                    from inference.pipeline import build_scorer

                    with st.spinner("🔧 Loading models…"):
                        scorer_rob, encoder_rob, _ = build_scorer(cfg.DEVICE)

                    def _perturb(frame_bgr: np.ndarray) -> np.ndarray:
                        """Apply brightness + noise + downscale perturbation."""
                        out = frame_bgr.astype(np.float32)
                        # Brightness
                        out = np.clip(out + brightness_delta, 0, 255)
                        # Gaussian noise
                        if noise_std > 0:
                            noise = np.random.normal(0, noise_std, out.shape).astype(np.float32)
                            out   = np.clip(out + noise, 0, 255)
                        out = out.astype(np.uint8)
                        # Resolution scale
                        if scale_factor < 100:
                            h, w = out.shape[:2]
                            nh, nw = max(1, int(h*scale_factor/100)), max(1, int(w*scale_factor/100))
                            out = cv2.resize(cv2.resize(out, (nw, nh)), (w, h))
                        return out

                    cap_rob = cv2.VideoCapture(str(rob_path))
                    src_fps_r = cap_rob.get(cv2.CAP_PROP_FPS) or 25.0
                    step_r    = max(1, round(src_fps_r / cfg.TARGET_FPS))

                    rob_prog   = st.progress(0, text="Testing perturbations…")
                    total_src_r = int(cap_rob.get(cv2.CAP_PROP_FRAME_COUNT))
                    est_r       = total_src_r // step_r

                    scorer_rob.reset()
                    rob_scores_clean    = []
                    rob_scores_perturbed= []
                    raw_idx_r = 0

                    while True:
                        ret_r, fr_bgr = cap_rob.read()
                        if not ret_r: break
                        raw_idx_r += 1
                        if (raw_idx_r - 1) % step_r != 0: continue

                        fr_r_clean = cv2.resize(fr_bgr, (cfg.FRAME_SIZE, cfg.FRAME_SIZE))
                        fr_r_perturbed = _perturb(fr_r_clean)

                        # Clean score
                        rgb_c  = cv2.cvtColor(fr_r_clean, cv2.COLOR_BGR2RGB)
                        c_emb, p_emb = encoder_rob.encode_frame_np(rgb_c)
                        s_c, _ = scorer_rob.push_and_score(c_emb, p_emb)
                        rob_scores_clean.append(s_c)

                        # Perturbed score (fresh scorer state not needed, just encode)
                        rgb_p  = cv2.cvtColor(fr_r_perturbed, cv2.COLOR_BGR2RGB)
                        c_emb2, p_emb2 = encoder_rob.encode_frame_np(rgb_p)
                        s_p, _ = scorer_rob.push_and_score(c_emb2, p_emb2)
                        rob_scores_perturbed.append(s_p)

                        done_r = len(rob_scores_clean)
                        rob_prog.progress(min(done_r/max(est_r,1), 1.0),
                                          text=f"Frame {done_r}/{est_r}")

                    cap_rob.release()
                    rob_prog.progress(1.0, text="✅ Done!")

                    mean_c = float(np.mean(rob_scores_clean))
                    mean_p = float(np.mean(rob_scores_perturbed))
                    max_p  = float(np.max(rob_scores_perturbed))
                    thr_u  = eval_threshold

                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Mean Score (clean)",     f"{mean_c:.4f}")
                    rc2.metric("Mean Score (perturbed)", f"{mean_p:.4f}",
                               delta=f"{mean_p - mean_c:+.4f}")
                    rc3.metric("Max Score (perturbed)",  f"{max_p:.4f}",
                               delta="✅ Below threshold" if max_p < thr_u else "⚠ Exceeds threshold")
                    rc4.metric("Threshold",              f"{thr_u:.4f}")

                    robust_pass = mean_p < thr_u
                    if robust_pass:
                        st.markdown(
                            '<div class="alert-normal">✅ Model is ROBUST — '
                            'perturbed normal frames stay below threshold</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="alert-anomaly">⚠ Not fully robust — '
                            'consider re-calibrating with perturbed normal data</div>',
                            unsafe_allow_html=True
                        )

                    # Overlay chart — clean vs perturbed scores
                    fig_rob = go.Figure()
                    frames_rob = list(range(len(rob_scores_clean)))
                    fig_rob.add_trace(go.Scatter(
                        x=frames_rob, y=rob_scores_clean, mode="lines",
                        line=dict(color="#00a854", width=1.5), name="Clean",
                    ))
                    fig_rob.add_trace(go.Scatter(
                        x=frames_rob, y=rob_scores_perturbed, mode="lines",
                        line=dict(color="#ff6b35", width=1.5), name="Perturbed",
                    ))
                    fig_rob.add_hline(y=thr_u, line_dash="dash", line_color="#e03030",
                                      line_width=2, annotation_text="Threshold",
                                      annotation_font_color="#e03030")
                    fig_rob.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
                        xaxis=dict(title="Frame", color="#3a5a8a", gridcolor="#dde8f5"),
                        yaxis=dict(title="Score", color="#3a5a8a", gridcolor="#dde8f5"),
                        font=dict(color="#3a5a8a"), height=280,
                        margin=dict(l=10,r=10,t=30,b=10),
                        title="Robustness: Clean vs Perturbed Normal Scores",
                        title_font_color="#2a4a7a",
                        legend=dict(orientation="h", y=1.1),
                    )
                    st.plotly_chart(fig_rob, key="rob_chart")

                    st.markdown(
                        f"**Perturbations applied:** Brightness {'+' if brightness_delta>=0 else ''}"
                        f"{brightness_delta}, Noise σ={noise_std}, "
                        f"Scale={scale_factor}%"
                    )

                except Exception as e:
                    st.error(f"Robustness test failed: {e}")
                    import traceback; st.code(traceback.format_exc())
