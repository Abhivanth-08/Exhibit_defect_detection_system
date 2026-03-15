"""
main.py — FastAPI Backend for JEPA Exhibit Defect Detection
============================================================
Endpoints
---------
POST  /api/train              — upload normal video, start training (SSE stream)
GET   /api/train/status       — poll training job status
POST  /api/calibrate          — run calibration (SSE stream of progress)
GET   /api/calibration-status — return saved threshold + component scales
POST  /api/detect             — upload test video, run inference (SSE stream)
WS    /ws/webcam              — live webcam inference stream (JSON messages)
GET   /api/system-status      — model / calibration readiness flags

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# App + CORS
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="JEPA Exhibit Defect Detector API",
    description="Self-supervised video anomaly detection with 5 architectural upgrades.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Upload directory
# ─────────────────────────────────────────────────────────────────────────────
UPLOADS_DIR = Path(__file__).parent / "user_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory job store  { job_id: {status, logs, error} }
# ─────────────────────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}


def _new_job() -> str:
    jid = str(uuid.uuid4())
    _jobs[jid] = {"status": "pending", "logs": [], "error": None}
    return jid


def _push_log(jid: str, msg: str):
    if jid in _jobs:
        _jobs[jid]["logs"].append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _frame_to_b64(frame_rgb: np.ndarray) -> str:
    """Encode an RGB numpy frame as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _save_upload(file: UploadFile) -> Path:
    dest = UPLOADS_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/system-status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/system-status")
def system_status():
    """Return readiness flags for each pipeline stage."""
    ckpt = cfg.CHECKPOINTS_DIR
    return {
        "temporal_trained":  (ckpt / "temporal.pt").exists(),
        "spatial_trained":   (ckpt / "spatial.pt").exists(),
        "energy_trained":    cfg.ENERGY_FILE.exists(),
        "calibrated":        cfg.CALIBRATION_FILE.exists(),
        "device":            cfg.DEVICE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/calibration-status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/calibration-status")
def calibration_status():
    if not cfg.CALIBRATION_FILE.exists():
        raise HTTPException(status_code=404, detail="Calibration not found — run calibration first.")
    data = json.loads(cfg.CALIBRATION_FILE.read_text())
    return {
        "threshold":    data.get("threshold",      1.0),
        "t_scale":      data.get("t_scale",        1.0),
        "t_long_scale": data.get("t_long_scale",   1.0),
        "s_scale":      data.get("s_scale",        1.0),
        "e_scale":      data.get("e_scale",        1.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/train   (SSE streaming)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/train")
async def train(
    video:      UploadFile = File(...),
    fps:        int = Form(cfg.TARGET_FPS),
    epochs:     int = Form(cfg.EPOCHS),
    svdd_epochs: int = Form(cfg.SVDD_EPOCHS),
    batch_size: int = Form(cfg.BATCH_SIZE),
    mask_humans: bool = Form(True),
):
    """
    Upload a normal video and start training.
    Returns an SSE stream of progress events.
    """
    video_path = await _save_upload(video)
    jid        = _new_job()

    async def event_stream() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run_training():
            try:
                from training.trainer import JEPATrainer

                # Human masker
                masker = None
                if mask_humans:
                    try:
                        from preprocessing.human_mask import HumanMaskFilter
                        masker = HumanMaskFilter()
                        loop.call_soon_threadsafe(queue.put_nowait, {
                            "type": "log", "msg": "🧍 YOLO human masker loaded."
                        })
                    except Exception as e:
                        loop.call_soon_threadsafe(queue.put_nowait, {
                            "type": "log", "msg": f"⚠ Human masker unavailable: {e}"
                        })

                def log_cb(msg):
                    _push_log(jid, msg)
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": msg})

                def prog_cb(epoch, total, loss):
                    loop.call_soon_threadsafe(queue.put_nowait, {
                        "type": "progress_a",
                        "epoch": epoch, "total": total, "loss": round(loss, 6),
                    })

                trainer = JEPATrainer(
                    device=cfg.DEVICE,
                    log_callback=log_cb,
                    progress_callback=prog_cb,
                )

                # Stage 0: Extract frames
                log_cb("📹 Extracting frames…")
                paths = trainer.extract_frames(
                    video_path=video_path, fps=fps,
                    human_masker=masker,
                )
                log_cb(f"   → {len(paths)} frames extracted")

                # Stage 1: Encode
                log_cb("🧠 Encoding with ViT-B/16* (partial domain adaptation)…")
                trainer.encode_frames(
                    frame_paths=paths,
                    video_stem=video_path.stem,
                )

                # Stage 2: JEPA training
                log_cb("🎯 Stage 3a — Short+Long Temporal + Cross-Attention Spatial JEPA…")
                loss_history = trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=cfg.LEARNING_RATE
                )
                (cfg.CHECKPOINTS_DIR / "loss_history.json").write_text(json.dumps(loss_history))
                loop.call_soon_threadsafe(queue.put_nowait, {
                    "type": "stage_a_done", "final_loss": round(loss_history[-1], 6),
                    "loss_history": loss_history,
                })

                # Stage 3: SVDD
                log_cb("⚡ Stage 3b — Training Deep SVDD energy model…")
                def svdd_prog(ep, tot, loss):
                    loop.call_soon_threadsafe(queue.put_nowait, {
                        "type": "progress_b",
                        "epoch": ep, "total": tot, "loss": round(loss, 6),
                    })
                trainer.on_progress = svdd_prog
                svdd_hist = trainer.train_energy(epochs=svdd_epochs)
                log_cb(f"✅ SVDD done — final loss: {svdd_hist[-1]:.5f}")

                _jobs[jid]["status"] = "done"
                loop.call_soon_threadsafe(queue.put_nowait, {
                    "type": "done",
                    "jepa_loss": round(loss_history[-1], 6),
                    "svdd_loss": round(svdd_hist[-1], 6),
                    "video_stem": video_path.stem,
                })

            except Exception as e:
                err = traceback.format_exc()
                _jobs[jid]["status"] = "error"
                _jobs[jid]["error"]  = err
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": str(e), "trace": err})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)   # sentinel

        thread = threading.Thread(target=_run_training, daemon=True)
        thread.start()

        while True:
            event = await queue.get()
            if event is None:
                yield _sse_event({"type": "stream_end"})
                break
            yield _sse_event(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/train/status  (simple poll)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/train/status")
def train_status():
    """Returns whether models exist and the saved loss history if available."""
    ckpt = cfg.CHECKPOINTS_DIR
    loss_history = []
    lf = ckpt / "loss_history.json"
    if lf.exists():
        loss_history = json.loads(lf.read_text())
    return {
        "trained":        (ckpt / "temporal.pt").exists() and (ckpt / "spatial.pt").exists(),
        "energy_trained": cfg.ENERGY_FILE.exists(),
        "loss_history":   loss_history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/calibrate   (SSE streaming)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/calibrate")
async def calibrate(
    video_stem: str = Form(""),
):
    """
    Run calibration on normal embeddings.
    Returns an SSE stream of progress + final threshold.
    """
    if not (cfg.CHECKPOINTS_DIR / "temporal.pt").exists():
        raise HTTPException(status_code=400, detail="Train the model first.")

    async def event_stream() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run_calib():
            try:
                from models.temporal_transformer import TemporalTransformer
                from models.spatial_jepa import SpatialJEPAHead
                from anomaly.energy_model import EnergyModel
                from anomaly.scorer import AnomalyScorer
                from training.calibration import calibrate_threshold, calibrate_component_scales

                stem = video_stem or cfg.NORMAL_VIDEO.stem
                cls_path   = cfg.EMBEDDINGS_DIR / f"{stem}_cls.npy"
                patch_path = cfg.EMBEDDINGS_DIR / f"{stem}_patches.npy"

                if not cls_path.exists():
                    raise FileNotFoundError(f"Embeddings not found for stem '{stem}'.")

                def push(ev): loop.call_soon_threadsafe(queue.put_nowait, ev)

                long_seq_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)

                push({"type": "log", "msg": "Loading models…"})
                ts = TemporalTransformer(window_size=cfg.WINDOW_SIZE).to(cfg.DEVICE)
                ts.load(cfg.CHECKPOINTS_DIR / "temporal.pt", cfg.DEVICE)

                tl = TemporalTransformer(window_size=long_seq_len).to(cfg.DEVICE)
                if (cfg.CHECKPOINTS_DIR / "temporal_long.pt").exists():
                    tl.load(cfg.CHECKPOINTS_DIR / "temporal_long.pt", cfg.DEVICE)

                sm = SpatialJEPAHead().to(cfg.DEVICE)
                sm.load(cfg.CHECKPOINTS_DIR / "spatial.pt", cfg.DEVICE)

                em = EnergyModel().to(cfg.DEVICE)
                if cfg.ENERGY_FILE.exists():
                    em.load(cfg.ENERGY_FILE, cfg.DEVICE)

                cls_embs   = np.load(str(cls_path))
                patch_embs = np.load(str(patch_path))
                N          = len(cls_embs)

                scorer = AnomalyScorer(
                    temporal_short=ts, temporal_long=tl,
                    spatial_head=sm, energy_model=em,
                    device=cfg.DEVICE,
                )

                push({"type": "log", "msg": f"Scoring {N} normal frames…"})
                normal_scores, t_errs, tl_errs, s_errs, e_errs = [], [], [], [], []
                for i in range(N):
                    s, comp = scorer.push_and_score(cls_embs[i], patch_embs[i])
                    normal_scores.append(s)
                    t_errs.append(comp.get("temporal",      0.0))
                    tl_errs.append(comp.get("temporal_long", 0.0))
                    s_errs.append(comp.get("spatial",       0.0))
                    e_errs.append(comp.get("energy",        0.0))
                    if i % max(N // 20, 1) == 0:
                        push({"type": "progress", "current": i+1, "total": N})

                scales = calibrate_component_scales(
                    np.array(t_errs), np.array(tl_errs),
                    np.array(s_errs), np.array(e_errs),
                )
                thr = calibrate_threshold(np.array(normal_scores), component_scales=scales)

                push({
                    "type": "done",
                    "threshold":    round(thr, 6),
                    "t_scale":      round(scales["t_scale"],      6),
                    "t_long_scale": round(scales.get("t_long_scale", 1.0), 6),
                    "s_scale":      round(scales["s_scale"],      6),
                    "e_scale":      round(scales["e_scale"],      6),
                    "scores":       [round(s, 6) for s in normal_scores],
                })
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait,
                    {"type": "error", "msg": str(e), "trace": traceback.format_exc()})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_run_calib, daemon=True).start()

        while True:
            ev = await queue.get()
            if ev is None:
                yield _sse_event({"type": "stream_end"})
                break
            yield _sse_event(ev)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/detect   (SSE streaming)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/detect")
async def detect(
    video:        UploadFile = File(...),
    max_frames:   int  = Form(0),
    show_every:   int  = Form(5),
    mask_humans:  bool = Form(True),
    threshold_override: float = Form(0.0),
):
    """
    Upload a test video and run anomaly detection.
    Returns an SSE stream of per-frame results.
    """
    if not (cfg.CHECKPOINTS_DIR / "temporal.pt").exists():
        raise HTTPException(status_code=400, detail="Train the model first.")
    if not cfg.CALIBRATION_FILE.exists():
        raise HTTPException(status_code=400, detail="Calibrate first.")

    video_path = await _save_upload(video)

    async def event_stream() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run_inference():
            try:
                from inference.pipeline import build_scorer, run_video_inference
                from training.calibration import load_calibration

                def push(ev): loop.call_soon_threadsafe(queue.put_nowait, ev)

                push({"type": "log", "msg": "🔧 Loading models…"})
                scorer, encoder, auto_threshold = build_scorer(cfg.DEVICE)
                threshold = threshold_override if threshold_override > 0 else auto_threshold

                masker = None
                if mask_humans:
                    try:
                        from preprocessing.human_mask import HumanMaskFilter
                        masker = HumanMaskFilter()
                        push({"type": "log", "msg": "🧍 Human masker active."})
                    except Exception as e:
                        push({"type": "log", "msg": f"⚠ Masker unavailable: {e}"})

                push({"type": "threshold", "value": round(threshold, 6)})

                # Count total frames for progress
                cap_tmp   = cv2.VideoCapture(str(video_path))
                total_src = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
                src_fps   = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
                cap_tmp.release()
                est_total = total_src // max(1, round(src_fps / cfg.TARGET_FPS))
                max_f     = max_frames if max_frames > 0 else None
                if max_f:
                    est_total = min(est_total, max_f)

                frame_idx_counter = [0]

                def frame_cb(frame_rgb, result, fidx):
                    frame_idx_counter[0] = fidx
                    ev = {
                        "type":         "frame",
                        "frame_idx":    fidx,
                        "score":        round(result["score"], 6),
                        "is_anomaly":   result["is_anomaly"],
                        "temporal":     round(result.get("temporal",      0), 6),
                        "temporal_long":round(result.get("temporal_long", 0), 6),
                        "spatial":      round(result.get("spatial",       0), 6),
                        "energy":       round(result.get("energy",        0), 6),
                        "uncertainty":  round(result.get("uncertainty",   0), 6),
                        "total_est":    est_total,
                    }
                    if fidx % show_every == 0:
                        ev["frame_b64"] = _frame_to_b64(frame_rgb)
                    push(ev)

                run_video_inference(
                    video_path, scorer, encoder, threshold,
                    frame_callback=frame_cb, max_frames=max_f,
                    human_masker=masker,
                )
                push({"type": "done", "total_frames": frame_idx_counter[0] + 1})

            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait,
                    {"type": "error", "msg": str(e), "trace": traceback.format_exc()})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_run_inference, daemon=True).start()

        while True:
            ev = await queue.get()
            if ev is None:
                yield _sse_event({"type": "stream_end"})
                break
            yield _sse_event(ev)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket /ws/webcam
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    """
    Live webcam inference stream.
    Client sends JSON config, then streams binary JPEG frames.
    Server replies with 'ready', then streams analysis results per frame.
    """
    await websocket.accept()

    try:
        # Wait for config message
        opts_raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        opts = json.loads(opts_raw)
    except Exception:
        await websocket.close(code=1003)
        return

    mask_humans      = bool(opts.get("mask_humans",     True))
    thr_override     = float(opts.get("threshold_override", 0.0))

    # Load threshold
    if cfg.CALIBRATION_FILE.exists():
        calib_data = json.loads(cfg.CALIBRATION_FILE.read_text())
        threshold  = calib_data.get("threshold", 1.0)
    else:
        threshold = 1.0
    if thr_override > 0:
        threshold = thr_override

    # Load models
    try:
        from inference.pipeline import build_scorer
        scorer, encoder, _ = build_scorer(cfg.DEVICE)
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "error", "msg": str(e)}))
        await websocket.close()
        return

    masker = None
    if mask_humans:
        try:
            from preprocessing.human_mask import HumanMaskFilter
            masker = HumanMaskFilter()
        except Exception:
            pass

    scorer.reset()
    frame_count = 0

    # Send ready signal
    await websocket.send_text(json.dumps({"type": "ready", "threshold": threshold}))

    loop = asyncio.get_event_loop()

    try:
        while True:
            # Receive either config/stop text or binary frame
            message = await websocket.receive()
            if "text" in message and message["text"] is not None:
                msg_text = message["text"].strip().lower()
                if msg_text == "stop":
                    break
            elif "bytes" in message and message["bytes"] is not None:
                frame_count += 1
                
                # Decode bytes to OpenCV image
                buf = np.frombuffer(message["bytes"], dtype=np.uint8)
                frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                frame_bgr_r = cv2.resize(frame_bgr, (cfg.FRAME_SIZE, cfg.FRAME_SIZE))
                frame_rgb   = cv2.cvtColor(frame_bgr_r, cv2.COLOR_BGR2RGB)

                encode_frame = frame_rgb
                if masker is not None:
                    masked_bgr   = masker.mask(frame_bgr_r)
                    encode_frame = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

                # Encode + score
                cls_emb, patch_emb = await loop.run_in_executor(
                    None, encoder.encode_frame_np, encode_frame
                )
                score, components = await loop.run_in_executor(
                    None, scorer.push_and_score, cls_emb, patch_emb
                )
                is_anomaly = float(score) > float(threshold)

                payload = {
                    "type":          "frame",
                    "frame_idx":     frame_count,
                    "score":         round(float(score), 6),
                    "is_anomaly":    is_anomaly,
                    "temporal":      round(float(components.get("temporal",       0)), 6),
                    "temporal_long": round(float(components.get("temporal_long",  0)), 6),
                    "spatial":       round(float(components.get("spatial",        0)), 6),
                    "energy":        round(float(components.get("energy",         0)), 6),
                    "uncertainty":   round(float(components.get("uncertainty",    0)), 6),
                    "frame_b64":     _frame_to_b64(frame_rgb),
                }


                await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "msg": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "JEPA Exhibit Defect Detector API", "version": "2.0.0", "status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
