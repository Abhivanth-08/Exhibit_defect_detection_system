"""
losses.py — Loss functions for JEPA training (v2 — Upgrade 4 SVDD)
"""
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def temporal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE + cosine-alignment loss for temporal prediction.
    Encourages both direction and magnitude to match.
    """
    mse = F.mse_loss(pred, target)
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return mse + cos


def spatial_loss(pred_masked: torch.Tensor, true_masked: torch.Tensor) -> torch.Tensor:
    """
    MSE between predicted and true masked patch embeddings.
    """
    return F.mse_loss(pred_masked, true_masked)


def combined_loss(
    t_loss: torch.Tensor,
    s_loss: torch.Tensor,
    alpha:  float = cfg.LOSS_ALPHA,
    beta:   float = cfg.LOSS_BETA,
) -> torch.Tensor:
    """Weighted sum of temporal and spatial losses."""
    return alpha * t_loss + beta * s_loss


# ─── Upgrade 4 — Deep SVDD Loss ───────────────────────────────────────────────

def svdd_loss(
    projections: torch.Tensor,   # [B, proj_dim] — output of EnergyModel.encode()
    center:      torch.Tensor,   # [proj_dim]
    nu:          float = cfg.SVDD_NU,
) -> torch.Tensor:
    """
    Soft-boundary Deep SVDD objective.

    Minimizes the volume of the hypersphere enclosing normal data:
        L = (1/N) Σ max(0, ||f(x) - c||² - R²) + (1/nu) * R²

    In the simplified (one-class deep SVDD) variant without R:
        L = (1/N) Σ ||f(x) - c||²

    We use a hinge variant that allows a small fraction nu of outliers:

        dist = ||z - c||²
        loss = mean(dist) + (1/nu) * clamp(dist - mean(dist), min=0).mean()

    Args:
        projections : batch of encoded embeddings  [B, proj_dim]
        center      : SVDD hypersphere center       [proj_dim]
        nu          : allowed outlier fraction (0 < nu < 1)
    """
    dist = ((projections - center) ** 2).sum(dim=-1)    # [B]
    r_sq = dist.mean().detach()                          # adaptive radius (no grad)
    penalty = F.relu(dist - r_sq).mean()
    return dist.mean() + (1.0 / max(nu, 1e-6)) * penalty
