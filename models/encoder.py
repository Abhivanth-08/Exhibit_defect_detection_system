"""
encoder.py — Frozen (or partially fine-tuned) ViT-B/16 encoder.

Upgrade 1: partial_unfreeze() exposes last 4 transformer blocks
           for domain adaptation at a low learning rate.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class ViTEncoder(nn.Module):
    """
    ViT-B/16 visual encoder.

    Default: first 8 blocks frozen, last 4 blocks trainable (Upgrade 1).
    Set ENCODER_FINETUNE=False in config to fully freeze (original behaviour).

    Returns:
      cls_token  : [B, 768]     — global frame representation
      patch_tokens: [B, 196, 768] — local spatial tokens
    """

    def __init__(
        self,
        model_name: str = cfg.ENCODER_MODEL,
        device: str     = cfg.DEVICE,
        finetune: bool  = cfg.ENCODER_FINETUNE,
        freeze_blocks: int = cfg.ENCODER_FREEZE_BLOCKS,
    ):
        super().__init__()
        self.device = device

        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0
        ).to(device)

        # Start fully frozen
        for param in self.model.parameters():
            param.requires_grad = False

        # Upgrade 1 — selectively unfreeze last N blocks
        if finetune:
            self.partial_unfreeze(freeze_blocks)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        self.eval()   # eval by default; trainer sets .train() on unfrozen subset

    # ─────────────────────────────────────────────────────────────────
    # Upgrade 1 — Partial Domain Adaptation
    # ─────────────────────────────────────────────────────────────────
    def partial_unfreeze(self, n_freeze: int = 8) -> None:
        """
        Freeze first n_freeze transformer blocks; unfreeze the rest.
        Also unfreezes the final LayerNorm so the output is adaptable.
        """
        total = len(self.model.blocks)
        for i, block in enumerate(self.model.blocks):
            for param in block.parameters():
                param.requires_grad = (i >= n_freeze)

        # Final norm
        for param in self.model.norm.parameters():
            param.requires_grad = True

        n_unfrozen = total - n_freeze
        trainable  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_p    = sum(p.numel() for p in self.model.parameters())
        print(f"[Encoder] {n_freeze}/{total} blocks frozen | "
              f"{n_unfrozen} blocks trainable "
              f"({trainable:,} / {total_p:,} params, "
              f"{100*trainable/total_p:.1f}% unfrozen)")

    def trainable_parameters(self):
        """Yield only the unfrozen encoder parameters (for optimizer)."""
        return (p for p in self.model.parameters() if p.requires_grad)

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, 3, 224, 224] normalised tensor

        Returns:
            cls_tokens:    [B, 768]
            patch_tokens:  [B, 196, 768]
        """
        features = self.model.forward_features(pixel_values)
        # timm ViT: features[:, 0] = CLS, features[:, 1:] = patches
        cls_tokens   = features[:, 0]         # [B, 768]
        patch_tokens = features[:, 1:]        # [B, 196, 768]
        return cls_tokens, patch_tokens

    # ─────────────────────────────────────────────────────────────────
    # NumPy helpers
    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_frame_np(
        self, frame_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a single frame.

        Args:
            frame_rgb: uint8 [224, 224, 3]

        Returns:
            cls_emb   : float32 [768]
            patch_emb : float32 [196, 768]
        """
        t = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        cls, patches = self.forward(t)
        return cls[0].cpu().numpy(), patches[0].cpu().numpy()

    @torch.no_grad()
    def encode_batch_numpy(
        self,
        frames: np.ndarray,        # [N, 224, 224, 3] uint8
        batch_size: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a numpy batch. Returns:
            cls   : [N, 768]
            patches: [N, 196, 768]
        """
        all_cls, all_patches = [], []
        for i in range(0, len(frames), batch_size):
            batch_np = frames[i : i + batch_size]
            tensors  = torch.stack([self.transform(f) for f in batch_np]).to(self.device)
            cls, pat = self.forward(tensors)
            all_cls.append(cls.cpu().numpy())
            all_patches.append(pat.cpu().numpy())
        return np.concatenate(all_cls, 0), np.concatenate(all_patches, 0)
