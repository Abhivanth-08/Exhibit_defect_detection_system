"""
calibration.py — Percentile calibration + per-component scale fitting (v2)

Saves calibration.json:
{
  "threshold"  : float,   — 97th pct of composite normal scores
  "mean"       : float,
  "std"        : float,
  "percentile" : int,
  "t_scale"    : float,   — 95th pct of temporal_short errors
  "t_long_scale": float,  — 95th pct of temporal_long errors
  "s_scale"    : float,   — 95th pct of spatial errors
  "e_scale"    : float,   — 95th pct of energy scores
}
"""
import json, sys
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def calibrate_threshold(
    normal_scores:   np.ndarray,
    component_scales: Optional[dict] = None,
    percentile: int = cfg.CALIBRATION_PERCENTILE,
) -> float:
    """
    Compute threshold from normal scores and save calibration JSON.

    Args:
        normal_scores    : composite anomaly scores on normal data [N]
        component_scales : dict with keys t_scale, t_long_scale, s_scale, e_scale
        percentile       : which percentile to use as threshold

    Returns:
        threshold (float)
    """
    threshold = float(np.percentile(normal_scores, percentile))

    data: dict = {
        "threshold":   threshold,
        "mean":        float(normal_scores.mean()),
        "std":         float(normal_scores.std()),
        "percentile":  percentile,
    }
    if component_scales:
        data.update(component_scales)

    cfg.CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    cfg.CALIBRATION_FILE.write_text(json.dumps(data, indent=2))
    return threshold


def calibrate_component_scales(
    t_errors:      np.ndarray,          # [N] temporal-short errors
    t_long_errors: np.ndarray,          # [N] temporal-long errors (may be shorter)
    s_errors:      np.ndarray,          # [N] spatial errors
    e_scores:      np.ndarray,          # [N] energy scores
    pct: int = 95,
) -> dict:
    """
    Compute 95th-percentile normalisation scales for each sub-score.
    Call this during calibration to make all components comparable.
    """
    def safe_pct(arr):
        v = float(np.percentile(arr[arr > 0], pct)) if (arr > 0).any() else 1.0
        return max(v, 1e-6)

    return {
        "t_scale":      safe_pct(t_errors),
        "t_long_scale": safe_pct(t_long_errors) if len(t_long_errors) > 0 else 1.0,
        "s_scale":      safe_pct(s_errors),
        "e_scale":      safe_pct(e_scores),
    }


def load_calibration() -> Optional[dict]:
    if not cfg.CALIBRATION_FILE.exists():
        return None
    return json.loads(cfg.CALIBRATION_FILE.read_text())


def load_threshold() -> float:
    c = load_calibration()
    return c["threshold"] if c else 1.0
