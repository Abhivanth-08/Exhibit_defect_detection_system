"""
trainer.py — Upgraded Training Orchestrator (v2 — All 5 Upgrades)

Training stages:
  1. Extract frames
  2. Encode with (partially trainable) ViT-B/16    [Upgrade 1]
  3a. Joint Short+Long Temporal + Spatial training   [Upgrades 2, 3]
  3b. Deep SVDD energy model training               [Upgrade 4]
  4. Save all checkpoints
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Callable, Optional
import json, sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from data.frame_sampler import extract_frames, load_frames_as_numpy
from data.dataset import EmbeddingWindowDataset, PatchEmbeddingDataset
from models.encoder import ViTEncoder
from models.temporal_transformer import TemporalTransformer
from models.spatial_jepa import SpatialJEPAHead
from anomaly.energy_model import EnergyModel
from training.losses import temporal_loss, spatial_loss, combined_loss, svdd_loss


class JEPATrainer:
    """Full training coordinator — v2 with all 5 architectural upgrades."""

    def __init__(
        self,
        device: str = cfg.DEVICE,
        log_callback:      Optional[Callable[[str], None]]       = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ):
        self.device      = device
        self.log         = log_callback or print
        self.on_progress = progress_callback

        self.encoder              : Optional[ViTEncoder]          = None
        self.temporal_short       : Optional[TemporalTransformer] = None
        self.temporal_long        : Optional[TemporalTransformer] = None
        self.spatial_model        : Optional[SpatialJEPAHead]     = None
        self.energy_model         : Optional[EnergyModel]         = None
        self.cls_embeddings       : Optional[np.ndarray]          = None
        self.patch_embeddings     : Optional[np.ndarray]          = None

        cfg.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        cfg.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        cfg.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Stage 1 — Frame Extraction
    # ──────────────────────────────────────────────────────────────
    def extract_frames(
        self,
        video_path: Path = cfg.NORMAL_VIDEO,
        fps: int = cfg.TARGET_FPS,
        frame_progress: Optional[Callable] = None,
        human_masker=None
    ) -> list:
        self.log(f"📹 Extracting frames from: {video_path.name}")
        paths = extract_frames(
            video_path, cfg.FRAMES_DIR, target_fps=fps,
            progress_callback=frame_progress,
        )
        self.log(f"✅ Extracted {len(paths)} frames")
        return paths

    # ──────────────────────────────────────────────────────────────
    # Stage 2 — Encoding (with partially trainable ViT — Upgrade 1)
    # ──────────────────────────────────────────────────────────────
    def encode_frames(
        self,
        frame_paths: list,
        video_stem: str,
        batch_size: int = 16,
        frame_progress: Optional[Callable] = None,
        force: bool = False,
    ):
        cls_path   = cfg.EMBEDDINGS_DIR / f"{video_stem}_cls.npy"
        patch_path = cfg.EMBEDDINGS_DIR / f"{video_stem}_patches.npy"

        if cls_path.exists() and patch_path.exists() and not force:
            self.log("💾 Loading cached embeddings…")
            self.cls_embeddings   = np.load(str(cls_path))
            self.patch_embeddings = np.load(str(patch_path))
            self.log(f"✅ Loaded {len(self.cls_embeddings)} embeddings")
            return

        if self.encoder is None:
            self.log("🧠 Loading ViT-B/16 (Upgrade 1: partial unfreeze)…")
            self.encoder = ViTEncoder(device=self.device)

        self.log(f"🔄 Encoding {len(frame_paths)} frames…")
        frames_np = load_frames_as_numpy(frame_paths)

        all_cls, all_patches = [], []
        total = len(frames_np)
        for i in range(0, total, batch_size):
            batch = frames_np[i : i + batch_size]
            c, p  = self.encoder.encode_batch_numpy(batch, batch_size=batch_size)
            all_cls.append(c)
            all_patches.append(p)
            if frame_progress:
                frame_progress(i + len(batch), total)

        self.cls_embeddings   = np.concatenate(all_cls,     axis=0)
        self.patch_embeddings = np.concatenate(all_patches, axis=0)
        np.save(str(cls_path),   self.cls_embeddings)
        np.save(str(patch_path), self.patch_embeddings)
        self.log(f"✅ Encoded & cached {len(self.cls_embeddings)} embeddings")

    # ──────────────────────────────────────────────────────────────
    # Stage 3a — Joint Temporal + Spatial Training (Upgrades 1-3)
    # ──────────────────────────────────────────────────────────────
    def train(
        self,
        epochs:     int   = cfg.EPOCHS,
        batch_size: int   = cfg.BATCH_SIZE,
        lr:         float = cfg.LEARNING_RATE,
        video_stem: str   = "",
    ) -> list:
        if self.cls_embeddings is None:
            raise RuntimeError("Call encode_frames first.")

        self.log("🚀 Initialising models (Short + Long Temporal, Spatial)…")

        # ── Models ────────────────────────────────────────────────
        self.temporal_short = TemporalTransformer(window_size=cfg.WINDOW_SIZE).to(self.device)
        # Long temporal uses downsampled window → shorter sequence
        long_seq_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)
        self.temporal_long  = TemporalTransformer(window_size=long_seq_len).to(self.device)
        self.spatial_model  = SpatialJEPAHead().to(self.device)

        # ── Optimizer: 3 param groups ─────────────────────────────
        # Upgrade 1: encoder gets separate low LR if loaded
        param_groups = [
            {"params": list(self.temporal_short.parameters()), "lr": lr},
            {"params": list(self.temporal_long.parameters()),  "lr": lr},
            {"params": list(self.spatial_model.parameters()),  "lr": lr},
        ]
        if self.encoder is not None and cfg.ENCODER_FINETUNE:
            enc_params = list(self.encoder.trainable_parameters())
            if enc_params:
                param_groups.append({"params": enc_params, "lr": cfg.ENCODER_LR})
                self.log(f"🎛 Encoder fine-tune group: {len(enc_params)} params @ LR={cfg.ENCODER_LR}")

        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # ── Datasets ──────────────────────────────────────────────
        t_dataset = EmbeddingWindowDataset(self.cls_embeddings, window_size=cfg.WINDOW_SIZE)
        s_dataset = PatchEmbeddingDataset(self.patch_embeddings)

        if len(t_dataset) < 2:
            raise RuntimeError("Not enough frames. Record a longer normal video.")

        t_loader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
        s_loader = DataLoader(s_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)

        # Long-window dataset (downsampled CLS — simulate long context)
        long_seq_len_actual = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)
        cls_downsampled     = self.cls_embeddings[::cfg.LONG_DOWNSAMPLE]
        long_dataset        = EmbeddingWindowDataset(cls_downsampled, window_size=long_seq_len_actual)
        use_long            = len(long_dataset) >= 1
        if use_long:
            long_loader = DataLoader(long_dataset, batch_size=min(batch_size, len(long_dataset)),
                                     shuffle=True, drop_last=False)
        else:
            long_loader = None
            self.log(f"⚠ Video too short for long temporal head (need ≥{long_seq_len_actual * cfg.LONG_DOWNSAMPLE} frames) — skipping.")

        loss_history = []
        self.log(f"🎯 Training {epochs} epochs (Short+Long Temporal + Cross-Attn Spatial)…")

        for epoch in range(1, epochs + 1):
            self.temporal_short.train()
            self.temporal_long.train()
            self.spatial_model.train()
            if self.encoder is not None and cfg.ENCODER_FINETUNE:
                self.encoder.train()

            epoch_loss = 0.0
            n_batches  = 0
            s_iter    = iter(s_loader)
            l_iter    = iter(long_loader) if long_loader is not None else None

            for context, target in t_loader:
                context = context.to(self.device)
                target  = target.to(self.device)

                # Short temporal
                pred_s = self.temporal_short(context)
                t_loss_s = temporal_loss(pred_s, target)

                # Long temporal (skipped if video too short)
                t_loss_l = torch.tensor(0.0, device=self.device)
                if l_iter is not None:
                    try:
                        ctx_l, tgt_l = next(l_iter)
                        ctx_l = ctx_l.to(self.device)
                        tgt_l = tgt_l.to(self.device)
                        pred_l    = self.temporal_long(ctx_l)
                        t_loss_l  = temporal_loss(pred_l, tgt_l)
                    except StopIteration:
                        l_iter = iter(long_loader)

                # Spatial cross-attention (Upgrade 2)
                try:
                    vis_patches, true_masked, mask = next(s_iter)
                except StopIteration:
                    s_iter = iter(s_loader)
                    vis_patches, true_masked, mask = next(s_iter)

                vis_patches = vis_patches.to(self.device)
                true_masked = true_masked.to(self.device)
                mask        = mask.to(self.device)

                pred_masked = self.spatial_model(vis_patches, mask)
                min_m = min(pred_masked.size(1), true_masked.size(1))
                s_loss_v = spatial_loss(
                    pred_masked[:, :min_m], true_masked[:, :min_m]
                )

                loss = combined_loss(t_loss_s + 0.5 * t_loss_l, s_loss_v)

                optimizer.zero_grad()
                loss.backward()
                all_params = (
                    list(self.temporal_short.parameters()) +
                    list(self.temporal_long.parameters()) +
                    list(self.spatial_model.parameters())
                )
                if self.encoder is not None and cfg.ENCODER_FINETUNE:
                    all_params += [p for p in self.encoder.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg = epoch_loss / max(n_batches, 1)
            loss_history.append(avg)

            if self.on_progress:
                self.on_progress(epoch, epochs, avg)
            if epoch % max(1, epochs // 5) == 0:
                self.log(f"  Epoch [{epoch:3d}/{epochs}] loss={avg:.5f}")

        # Save
        self.temporal_short.save(cfg.CHECKPOINTS_DIR / "temporal.pt")
        self.temporal_long.save( cfg.CHECKPOINTS_DIR / "temporal_long.pt")
        self.spatial_model.save( cfg.CHECKPOINTS_DIR / "spatial.pt")
        if self.encoder is not None and cfg.ENCODER_FINETUNE:
            torch.save(
                {p_name: p for p_name, p in self.encoder.named_parameters() if p.requires_grad},
                str(cfg.CHECKPOINTS_DIR / "encoder_finetune.pt"),
            )
        self.log("✅ Stage 3a checkpoints saved")
        return loss_history

    # ──────────────────────────────────────────────────────────────
    # Stage 3b — Deep SVDD Training (Upgrade 4)
    # ──────────────────────────────────────────────────────────────
    def train_energy(
        self,
        epochs:     int   = cfg.SVDD_EPOCHS,
        batch_size: int   = cfg.BATCH_SIZE,
        lr:         float = cfg.SVDD_LR,
    ) -> list:
        if self.cls_embeddings is None:
            raise RuntimeError("Call encode_frames first.")

        self.log("⚡ Stage 3b — Training Deep SVDD energy model…")
        self.energy_model = EnergyModel().to(self.device)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.cls_embeddings.astype(np.float32))
        )
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Initialise center from mean of projections (warm-start)
        self.energy_model.fit_center(self.cls_embeddings, device=self.device)
        center = self.energy_model.center.detach()   # fixed during SVDD

        optimizer = torch.optim.Adam(self.energy_model.parameters(), lr=lr)
        loss_history = []

        for epoch in range(1, epochs + 1):
            self.energy_model.train()
            ep_loss = 0.0; n = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                z     = self.energy_model.encode(batch)
                loss  = svdd_loss(z, center, nu=cfg.SVDD_NU)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item(); n += 1
            avg = ep_loss / max(n, 1)
            loss_history.append(avg)
            if epoch % max(1, epochs // 4) == 0:
                self.log(f"  SVDD Epoch [{epoch:3d}/{epochs}] loss={avg:.5f}")
            if self.on_progress:
                self.on_progress(epoch, epochs, avg)

        # Re-fit center on trained projections
        self.energy_model.fit_center(self.cls_embeddings, device=self.device)
        self.energy_model.save(cfg.ENERGY_FILE)
        self.log("✅ Energy model saved")
        return loss_history

    # ──────────────────────────────────────────────────────────────
    # Convenience loaders
    # ──────────────────────────────────────────────────────────────
    def load_encoder(self):
        if self.encoder is None:
            self.encoder = ViTEncoder(device=self.device)

    def load_models(self):
        long_seq_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)
        self.temporal_short = TemporalTransformer(window_size=cfg.WINDOW_SIZE).to(self.device)
        self.temporal_long  = TemporalTransformer(window_size=long_seq_len).to(self.device)
        self.spatial_model  = SpatialJEPAHead().to(self.device)
        self.energy_model   = EnergyModel().to(self.device)

        self.temporal_short.load(cfg.CHECKPOINTS_DIR / "temporal.pt",      self.device)
        self.temporal_long.load( cfg.CHECKPOINTS_DIR / "temporal_long.pt", self.device)
        self.spatial_model.load( cfg.CHECKPOINTS_DIR / "spatial.pt",       self.device)
        if cfg.ENERGY_FILE.exists():
            self.energy_model.load(cfg.ENERGY_FILE, self.device)
        self.log("✅ All models loaded")
