"""
energy_model.py — Deep SVDD Energy-Based Anomaly Model (Upgrade 4)

Replaces Mahalanobis distance with a learned non-linear energy function.

Deep SVDD objective:
    min  ||f(x) - c||²       (normal data → project near center c)
    The center c is fixed as the mean of projections on normal data.

At inference:
    E(x) = ||f(x) - c||²    → small for normal, large for anomalous

Advantage over Mahalanobis:
    • Non-linear manifold awareness (Mahalanobis is purely Gaussian)
    • Learns the exact shape of the normal data manifold
    • Better for exhibit data with complex structural patterns
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class EnergyModel(nn.Module):
    """
    Deep SVDD projection network.

    f: R^embed_dim → R^proj_dim
    E(x) = ||f(x) - c||²
    """

    def __init__(
        self,
        input_dim: int = cfg.EMBED_DIM,
        proj_dim:  int = cfg.SVDD_DIM,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, proj_dim),
        )

        # Center c — registered buffer so it moves with .to(device)
        self.register_buffer("center", torch.zeros(proj_dim))
        self._center_set = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → [B, proj_dim]"""
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns energy scores [B] = ||f(x) - c||²"""
        z = self.encode(x)
        return ((z - self.center) ** 2).sum(dim=-1)

    # ─────────────────────────────────────────────────────────────────
    # Center Initialisation
    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def fit_center(
        self,
        embeddings: np.ndarray,          # [N, D] normal CLS embeddings
        device: str = cfg.DEVICE,
        batch_size: int = 256,
    ) -> None:
        """
        Set c as the mean of projections on normal data.
        Called once after SVDD training (or before, for warm-start).
        """
        self.eval()
        self.to(device)
        all_z = []
        t = torch.from_numpy(embeddings.astype(np.float32)).to(device)
        for i in range(0, len(t), batch_size):
            z = self.encode(t[i : i + batch_size])
            all_z.append(z)
        c = torch.cat(all_z, 0).mean(0)
        # Avoid center collapse: if any dim is very close to 0, push away
        c[(c.abs() < 0.01) & (c > 0)] =  0.01
        c[(c.abs() < 0.01) & (c < 0)] = -0.01
        self.center.copy_(c)
        self._center_set = True
        print(f"[SVDD] Center set | norm={c.norm().item():.4f} | dim={len(c)}")

    @torch.no_grad()
    def score_numpy(
        self,
        cls_emb: np.ndarray,   # [D] or [N, D]
        device: str = cfg.DEVICE,
    ) -> float:
        """Return energy for one or a batch of embeddings as numpy float(s)."""
        self.eval()
        self.to(device)
        t = torch.from_numpy(
            cls_emb.astype(np.float32)
        ).unsqueeze(0).to(device) if cls_emb.ndim == 1 else \
            torch.from_numpy(cls_emb.astype(np.float32)).to(device)
        e = self.forward(t)
        return e.squeeze().cpu().item() if cls_emb.ndim == 1 else e.cpu().numpy()

    # ─────────────────────────────────────────────────────────────────
    def save(self, path: Path):
        torch.save({
            "state_dict": self.state_dict(),
            "center":     self.center.cpu().numpy(),
        }, str(path))

    def load(self, path: Path, device: str = cfg.DEVICE):
        ckpt = torch.load(str(path), map_location=device)
        self.load_state_dict(ckpt["state_dict"])
        self.center.copy_(
            torch.from_numpy(ckpt["center"]).to(device)
        )
        self._center_set = True
        self.to(device)
