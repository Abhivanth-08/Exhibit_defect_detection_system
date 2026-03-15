"""
scorer.py — Composite Anomaly Scorer (v2 — All 5 Upgrades)

Score = α·(temporal_short / (var+ε))   ← Upgrades 3 + 5 (short horizon, uncertainty-damped)
      + α_long·temporal_long             ← Upgrade 3 (long horizon for slow drift)
      + β·spatial                        ← Upgrade 2 (cross-attention spatial error)
      + γ·energy                         ← Upgrade 4 (Deep SVDD)

All components are normalised by their 95th-percentile on normal data (set at calibration).
"""
import sys
from pathlib import Path
from collections import deque
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from models.temporal_transformer import TemporalTransformer
from models.spatial_jepa import SpatialJEPAHead
from anomaly.energy_model import EnergyModel


class AnomalyScorer:
    """
    Maintains rolling frame buffers (short + long) and computes the
    composite anomaly score for each incoming frame.

    Call scorer.push_and_score(cls_emb, patch_emb) for every frame.
    Returns (score: float, components: dict).
    """

    def __init__(
        self,
        temporal_short: TemporalTransformer,
        temporal_long:  TemporalTransformer,
        spatial_head:   SpatialJEPAHead,
        energy_model:   EnergyModel,
        device:         str   = cfg.DEVICE,
        alpha:          float = cfg.SCORE_ALPHA,
        alpha_long:     float = cfg.SCORE_ALPHA_LONG,
        beta:           float = cfg.SCORE_BETA,
        gamma:          float = cfg.SCORE_GAMMA,
        # Normalisation scales (set from calibration, 95th pct on normal data)
        t_scale:        float = 1.0,
        t_long_scale:   float = 1.0,
        s_scale:        float = 1.0,
        e_scale:        float = 1.0,
        mc_passes:      int   = cfg.MC_DROPOUT_PASSES,
        mc_epsilon:     float = 1e-4,   # Upgrade 5 — uncertainty denominator
    ):
        self.device        = device
        self.temporal_s    = temporal_short.to(device)
        self.temporal_l    = temporal_long.to(device)
        self.spatial       = spatial_head.to(device)
        self.energy        = energy_model.to(device)

        self.alpha         = alpha
        self.alpha_long    = alpha_long
        self.beta          = beta
        self.gamma         = gamma

        self.t_scale       = t_scale
        self.t_long_scale  = t_long_scale
        self.s_scale       = s_scale
        self.e_scale       = e_scale

        self.mc_passes     = mc_passes
        self.mc_epsilon    = mc_epsilon

        # Rolling windows
        self.short_buf: deque = deque(maxlen=cfg.WINDOW_SIZE)
        self.long_buf:  deque = deque(maxlen=cfg.LONG_WINDOW_SIZE)

        # Put models in eval mode; temporal keeps dropout for MC
        self.temporal_s.eval()
        self.temporal_l.eval()
        self.spatial.eval()
        self.energy.eval()

    def reset(self):
        self.short_buf.clear()
        self.long_buf.clear()

    def is_ready(self) -> bool:
        return len(self.short_buf) >= cfg.WINDOW_SIZE

    # ─────────────────────────────────────────────────────────────────
    def push_and_score(
        self,
        cls_emb:   np.ndarray,   # [768]
        patch_emb: np.ndarray,   # [196, 768]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Push one frame and compute its anomaly score.

        Returns:
            score      : float — composite anomaly score
            components : dict  — individual sub-scores + uncertainty
        """
        # Push to buffers (long uses every LONG_DOWNSAMPLE-th push, handled below)
        self.short_buf.append(cls_emb.copy())
        self.long_buf.append(cls_emb.copy())

        if not self.is_ready():
            return 0.0, {"temporal": 0.0, "temporal_long": 0.0,
                         "spatial": 0.0, "energy": 0.0, "uncertainty": 0.0}

        cls_t   = torch.from_numpy(cls_emb.astype(np.float32)).unsqueeze(0).to(self.device)
        patch_t = torch.from_numpy(patch_emb.astype(np.float32)).unsqueeze(0).to(self.device)

        # ── Short temporal (MC Dropout → uncertainty) ──────────────────────────
        ctx_short = torch.from_numpy(
            np.stack(list(self.short_buf), axis=0).astype(np.float32)
        ).unsqueeze(0).to(self.device)   # [1, K_short, 768]

        with torch.no_grad():
            # mc_forward internally enables dropout, then re-evals
            mean_pred_s, variance_s = self.temporal_s.mc_forward(ctx_short, self.mc_passes)

        t_err_short = (
            F.mse_loss(mean_pred_s, cls_t).item() +
            (1.0 - F.cosine_similarity(mean_pred_s, cls_t, dim=-1).mean()).item()
        )
        unc = variance_s.item()   # scalar uncertainty (Upgrade 5)

        # Uncertainty-damped temporal score (Upgrade 5)
        t_score_short = (t_err_short / self.t_scale) / (unc + self.mc_epsilon)

        # ── Long temporal ──────────────────────────────────────────────────────
        t_score_long = 0.0
        if len(self.long_buf) >= cfg.LONG_WINDOW_SIZE:
            # Downsample: take every LONG_DOWNSAMPLE-th from the long buffer
            long_list = list(self.long_buf)
            downsampled = long_list[::cfg.LONG_DOWNSAMPLE]
            # Pad or truncate to LONG_WINDOW_SIZE // LONG_DOWNSAMPLE
            target_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 1)
            if len(downsampled) < target_len:
                pad = [downsampled[0]] * (target_len - len(downsampled))
                downsampled = pad + downsampled
            else:
                downsampled = downsampled[-target_len:]

            ctx_long = torch.from_numpy(
                np.stack(downsampled, axis=0).astype(np.float32)
            ).unsqueeze(0).to(self.device)   # [1, target_len, 768]

            with torch.no_grad():
                pred_l = self.temporal_l(ctx_long)
            t_err_long = (
                F.mse_loss(pred_l, cls_t).item() +
                (1.0 - F.cosine_similarity(pred_l, cls_t, dim=-1).mean()).item()
            )
            t_score_long = t_err_long / self.t_long_scale

        # ── Spatial reconstruction (Upgrade 2 — cross-attention head) ─────────
        P = patch_t.size(1)
        n_masked = int(P * cfg.MASK_RATIO)
        # Block mask — deterministic for scoring (use centre block)
        start = (P - n_masked) // 2
        mask = torch.zeros(1, P, dtype=torch.bool, device=self.device)
        mask[0, start : start + n_masked] = True

        masked_patches = patch_t.clone()
        masked_patches[mask.unsqueeze(-1).expand_as(masked_patches)] = 0.0

        with torch.no_grad():
            pred_masked = self.spatial(masked_patches, mask)
            true_masked = patch_t[mask.unsqueeze(-1).expand_as(patch_t)].view(1, n_masked, -1)

        s_err = F.mse_loss(pred_masked, true_masked[:, :pred_masked.size(1)]).item()
        s_score = s_err / self.s_scale

        # ── Energy score (Upgrade 4 — Deep SVDD) ──────────────────────────────
        with torch.no_grad():
            e_val = self.energy(cls_t).item()
        e_score = e_val / self.e_scale

        # ── Composite score ────────────────────────────────────────────────────
        score = (
            self.alpha      * t_score_short +
            self.alpha_long * t_score_long  +
            self.beta       * s_score       +
            self.gamma      * e_score
        )

        return score, {
            "temporal":      t_score_short,
            "temporal_long": t_score_long,
            "spatial":       s_score,
            "energy":        e_score,
            "uncertainty":   unc,
        }
