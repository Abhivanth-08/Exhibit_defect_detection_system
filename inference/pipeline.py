"""
pipeline.py — Video Inference Pipeline (v2 — All 5 Upgrades)

build_scorer() loads all 4 trained components:
  • temporal_short (K=8, MC dropout)    — Upgrades 3, 5
  • temporal_long  (K=32 downsampled)   — Upgrade 3
  • spatial_head   (cross-attention)    — Upgrade 2
  • energy_model   (Deep SVDD)          — Upgrade 4
  + partially-adapted encoder           — Upgrade 1
"""
import sys
from pathlib import Path
from typing import Callable, Optional, List, Dict

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from models.encoder import ViTEncoder
from models.temporal_transformer import TemporalTransformer
from models.spatial_jepa import SpatialJEPAHead
from anomaly.energy_model import EnergyModel
from anomaly.scorer import AnomalyScorer
from training.calibration import load_calibration


# ─────────────────────────────────────────────────────────────────────────────
def build_scorer(device: str = cfg.DEVICE):
    """
    Load all trained checkpoints and build the upgraded AnomalyScorer.

    Returns:
        scorer  : AnomalyScorer
        encoder : ViTEncoder
        threshold: float
    """
    long_seq_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)

    # ── Load models ───────────────────────────────────────────────────────────
    temporal_short = TemporalTransformer(window_size=cfg.WINDOW_SIZE).to(device)
    temporal_short.load(cfg.CHECKPOINTS_DIR / "temporal.pt", device)

    temporal_long = TemporalTransformer(window_size=long_seq_len).to(device)
    if (cfg.CHECKPOINTS_DIR / "temporal_long.pt").exists():
        temporal_long.load(cfg.CHECKPOINTS_DIR / "temporal_long.pt", device)

    spatial_head = SpatialJEPAHead().to(device)
    spatial_head.load(cfg.CHECKPOINTS_DIR / "spatial.pt", device)

    energy_model = EnergyModel().to(device)
    if cfg.ENERGY_FILE.exists():
        energy_model.load(cfg.ENERGY_FILE, device)

    # ── Encoder (partially adapted) ───────────────────────────────────────────
    encoder = ViTEncoder(device=device)

    # ── Load calibration scales ───────────────────────────────────────────────
    calib = load_calibration() or {}
    threshold   = calib.get("threshold",    1.0)
    t_scale     = calib.get("t_scale",      1.0)
    t_long_sc   = calib.get("t_long_scale", 1.0)
    s_scale     = calib.get("s_scale",      1.0)
    e_scale     = calib.get("e_scale",      1.0)

    scorer = AnomalyScorer(
        temporal_short = temporal_short,
        temporal_long  = temporal_long,
        spatial_head   = spatial_head,
        energy_model   = energy_model,
        device         = device,
        t_scale        = t_scale,
        t_long_scale   = t_long_sc,
        s_scale        = s_scale,
        e_scale        = e_scale,
    )
    return scorer, encoder, threshold


# ─────────────────────────────────────────────────────────────────────────────
def run_video_inference(
    video_path:     Path,
    scorer:         AnomalyScorer,
    encoder:        ViTEncoder,
    threshold:      float,
    frame_callback: Optional[Callable] = None,
    max_frames:     Optional[int]      = None,
    human_masker=None,
) -> List[Dict]:
    """
    Process a video file frame-by-frame and return anomaly results.

    Args:
        video_path    : path to the test video
        scorer        : AnomalyScorer (pre-built from build_scorer)
        encoder       : ViTEncoder
        threshold     : anomaly decision boundary
        frame_callback: fn(frame_rgb, result_dict, frame_idx) called per frame
        max_frames    : stop after this many frames (None = full video)
        human_masker  : optional HumanMaskFilter; persons are blacked out
                        before encoding (frame_rgb passed to callback is
                        the ORIGINAL unmasked frame for visual display).

    Returns:
        List of result dicts per processed frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step       = max(1, round(src_fps / cfg.TARGET_FPS))

    scorer.reset()
    results    = []
    raw_idx    = 0
    frame_idx  = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        raw_idx += 1
        if (raw_idx - 1) % step != 0:
            continue   # skip to hit TARGET_FPS

        # Preprocess
        frame_resized = cv2.resize(frame_bgr, (cfg.FRAME_SIZE, cfg.FRAME_SIZE))
        frame_rgb     = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Human masking (applied to encode only; callback gets original display frame)
        encode_frame = frame_rgb
        if human_masker is not None:
            masked_bgr   = human_masker.mask(frame_resized)   # operates on BGR
            encode_frame = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

        # Encode
        cls_emb, patch_emb = encoder.encode_frame_np(encode_frame)

        # Score
        score, components = scorer.push_and_score(cls_emb, patch_emb)
        is_anomaly        = score > threshold

        result = {
            "frame_idx":   frame_idx,
            "score":       score,
            "is_anomaly":  is_anomaly,
            "frame_rgb":   frame_rgb,
            **components,
        }
        results.append(result)

        if frame_callback:
            frame_callback(frame_rgb, result, frame_idx)

        frame_idx += 1

    cap.release()
    return results
