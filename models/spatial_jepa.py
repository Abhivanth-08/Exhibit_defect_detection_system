"""
spatial_jepa.py — Contextual Cross-Attention Spatial JEPA Head (Upgrade 2)

Architecture:
  1. Context Encoder  : Transformer Encoder over VISIBLE patch embeddings
                        (self-attention — patches observe each other)
  2. Mask Token       : learnable [MASK] vector for each masked position
  3. Predictor        : Transformer Decoder — masked positions cross-attend
                        to visible context to reconstruct their true embeddings

This replaces the independent per-patch MLP, making spatial modeling
relational ("structural consistency") rather than purely local.
"""
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class SpatialJEPAHead(nn.Module):
    """
    Contextual cross-attention Spatial JEPA predictor.

    forward(patches, mask) ->  pred_masked [B, n_masked, D]

    Args (constructor):
        embed_dim     : patch embedding dimension (768 for ViT-B)
        num_heads     : attention heads
        num_enc_layers: self-attention encoder depth over visible patches
        num_dec_layers: cross-attention decoder depth for mask prediction
        dropout       : dropout rate
    """

    def __init__(
        self,
        embed_dim:      int = cfg.EMBED_DIM,
        num_heads:      int = cfg.SPATIAL_NUM_HEADS,
        num_enc_layers: int = cfg.SPATIAL_NUM_ENC_LAYERS,
        num_dec_layers: int = cfg.SPATIAL_NUM_DEC_LAYERS,
        dropout:        float = cfg.T_DROPOUT,
    ):
        super().__init__()

        # ── Visible-patch context encoder (self-attention) ────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # ── Learnable [MASK] token ─────────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ── Predictor: cross-attention, query=masked positions, kv=visible context ─
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.predictor = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # ── Output norm + projection ───────────────────────────────────────────
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        patches: torch.Tensor,   # [B, P, D]  — ALL patches (masked zeroed)
        mask:    torch.Tensor,   # [B, P]  bool/float — True = masked position
    ) -> torch.Tensor:
        """
        Returns predicted embeddings for masked positions.
        Output shape: [B, n_masked, D]
        """
        B, P, D = patches.shape

        # ── Build visible subset ───────────────────────────────────────────────
        # patches already have masked positions zeroed by dataset; we extract
        # visible positions per sample using the mask.
        # For efficiency we process the full sequence but mask the keys in encoder.

        # mask is True where MASKED (hidden), False where visible
        mask_bool = mask.bool()                    # [B, P]
        vis_mask  = ~mask_bool                     # True at visible positions

        # Context encoder over ALL patches (visible carry info; masked ignored via key_padding_mask)
        # key_padding_mask: True → this key position is IGNORED
        context = self.context_encoder(
            patches, src_key_padding_mask=mask_bool
        )   # [B, P, D]  — contextualised visible embeddings

        # ── Build query: masked positions with learnable mask token ───────────
        # Gather masked positions for the queries
        # We use a fixed-size approach: replace masked positions with mask_token
        mask_tokens = self.mask_token.expand(B, P, D)           # [B, P, D]
        query = torch.where(mask_bool.unsqueeze(-1), mask_tokens, patches)  # [B, P, D]
        # We only want query at masked positions — but cross-attn to full context
        # We pass the full query sequence and later select masked outputs
        pred_all = self.predictor(
            tgt=query,
            memory=context,
            tgt_key_padding_mask=~mask_bool,   # only masked positions attend
        )   # [B, P, D]

        pred_all = self.out_proj(self.out_norm(pred_all))  # [B, P, D]

        # ── Extract only the masked positions ─────────────────────────────────
        # For batch uniformity, gather by first sample's mask count
        # (dataset ensures fixed mask count per batch - see PatchEmbeddingDataset)
        n_masked = mask_bool[0].sum().item()
        pred_masked = pred_all[mask_bool].view(B, int(n_masked), D)   # [B, n_masked, D]
        return pred_masked

    # ─────────────────────────────────────────────────────────────────
    def reconstruct_error(
        self,
        patches: torch.Tensor,
        mask:    torch.Tensor,
        true_patches: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: MSE between predicted and true masked patches."""
        pred = self.forward(patches, mask)
        n    = min(pred.size(1), true_patches.size(1))
        return torch.nn.functional.mse_loss(pred[:, :n], true_patches[:, :n])

    # ─────────────────────────────────────────────────────────────────
    def save(self, path: Path):
        torch.save(self.state_dict(), str(path))

    def load(self, path: Path, device: str = "cpu"):
        self.load_state_dict(torch.load(str(path), map_location=device))
        self.to(device)
