"""
JEPA Exhibit Defect Detection — Central Configuration (v2 — 5 Upgrades)
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
PROJECT_DIR      = BASE_DIR.parent
VIDEOS_DIR       = PROJECT_DIR / "videos"

NORMAL_VIDEO     = VIDEOS_DIR / "2.mp4"
TEST_VIDEO       = VIDEOS_DIR / "WhatsApp Video 2026-01-11 at 10.13.51 PM.mp4"
FRAMES_DIR       = BASE_DIR / "data" / "frames"
EMBEDDINGS_DIR   = BASE_DIR / "data" / "embeddings"
CHECKPOINTS_DIR  = BASE_DIR / "checkpoints"
CALIBRATION_FILE = CHECKPOINTS_DIR / "calibration.json"
MAHAL_FILE       = CHECKPOINTS_DIR / "mahal_params.npz"
ENERGY_FILE      = CHECKPOINTS_DIR / "energy_model.pt"        # Upgrade 4

# ─── Video Sampling ───────────────────────────────────────────────────────────
TARGET_FPS  = 3     # frames per second to sample
FRAME_SIZE  = 224   # resize to FRAME_SIZE × FRAME_SIZE

# ─── Encoder ──────────────────────────────────────────────────────────────────
ENCODER_MODEL = "vit_base_patch16_224"
EMBED_DIM     = 768
NUM_PATCHES   = 196   # 14×14 spatial grid

# ─── Upgrade 1 — Partial Encoder Fine-Tuning ──────────────────────────────────
ENCODER_FREEZE_BLOCKS = 8       # freeze first 8 of 12 ViT blocks
ENCODER_FINETUNE      = True    # False → fully frozen (original behaviour)
ENCODER_LR            = 1e-5   # low LR for partially unfrozen blocks

# ─── Temporal Transformer (shared by short/long heads) ────────────────────────
WINDOW_SIZE  = 8    # K — short-horizon context window
T_NUM_LAYERS = 4
T_NUM_HEADS  = 8
T_HIDDEN_DIM = 768
T_FF_DIM     = 2048
T_DROPOUT    = 0.1

# ─── Upgrade 3 — Multi-Scale Temporal ─────────────────────────────────────────
LONG_WINDOW_SIZE = 32   # K for long-horizon head
LONG_DOWNSAMPLE  = 4    # use every Nth CLS embedding for long context

# ─── Upgrade 2 — Contextual Attention Spatial JEPA ────────────────────────────
SPATIAL_NUM_HEADS      = 8
SPATIAL_NUM_ENC_LAYERS = 3
SPATIAL_NUM_DEC_LAYERS = 2
MASK_RATIO             = 0.50

# ─── Training ─────────────────────────────────────────────────────────────────
EPOCHS        = 30
BATCH_SIZE    = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4
LOSS_ALPHA    = 1.0   # temporal loss weight
LOSS_BETA     = 0.5   # spatial JEPA loss weight

# ─── Upgrade 4 — Deep SVDD Energy Model ───────────────────────────────────────
SVDD_DIM    = 64    # projection dimension
SVDD_NU     = 0.1   # soft-boundary fraction of allowed outliers
SVDD_EPOCHS = 20    # SVDD fine-tuning epochs
SVDD_LR     = 1e-4

# ─── Upgrade 5 — MC Dropout Uncertainty ───────────────────────────────────────
MC_DROPOUT_RATE   = 0.1
MC_DROPOUT_PASSES = 20   # stochastic forward passes at inference

# ─── Anomaly Scoring (weights must sum to 1.0) ────────────────────────────────
SCORE_ALPHA      = 0.40   # short-horizon temporal error
SCORE_ALPHA_LONG = 0.15   # long-horizon temporal error   (Upgrade 3)
SCORE_BETA       = 0.25   # spatial reconstruction error
SCORE_GAMMA      = 0.20   # Deep SVDD energy              (Upgrade 4)

# ─── Calibration ──────────────────────────────────────────────────────────────
CALIBRATION_PERCENTILE = 97

# ─── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
