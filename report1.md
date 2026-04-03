# JEPA Project: Custom "From Scratch" Implementation Comprehensive Report

## 1. Overview
The `jepa` directory contains a highly customized, built-from-scratch Vision Joint-Embedding Predictive Architecture (V-JEPA) designed explicitly for identifying defects or anomalies in video streams such as museum exhibits. Unlike off-the-shelf deployments, this pipeline builds several sub-components (spatial predictors, temporal transformers, energy models) from the ground up to achieve high sensitivity and structural consistency without relying on massive pre-trained foundation models. The system exposes its functionality via a FastAPI web service, allowing for asynchronous training, real-time webcam inference, and interactive threshold calibration.

## 2. Core Architectural Components ("The 5 Upgrades")

### 2.1 Upgrade 1: Partially Fine-Tuned ViT Encoder
**File:** `config.py` (Encoder section), `training/trainer.py`
Instead of using a gigantic foundation model, the system leverages a lightweight Vision Transformer (`vit_base_patch16_224`). 
- **Freezing & Fine-Tuning:** The first 8 blocks of the 12 total transformer blocks are frozen. Only the last 4 blocks are fine-tuned with a very low learning rate (`1e-5`).
- **Purpose:** This partial "domain adaptation" allows the model to learn the specific lighting, angles, and artifacts of the museum exhibit without suffering from catastrophic forgetting or overfitting given the limited training data.

### 2.2 Upgrade 2: Contextual Cross-Attention Spatial JEPA
**File:** `models/spatial_jepa.py`
Originally, JEPA implementations often reconstructed patches locally. This project replaces independent MLPs with a full-blown dual-transformer module.
- **Mechanism:** 
  1. A `TransformerEncoder` analyzes all *visible* patches via self-attention so they "communicate" with each other.
  2. A `TransformerDecoder` predicts the *masked* patches using a learnable `[MASK]` token, cross-attending against the memory of the visible patches.
- **Purpose:** Forces the model to understand the *global layout* and structural consistency of the image, rather than just guessing local colors and textures.

### 2.3 Upgrades 3 & 5: Multi-Scale Temporal Transformer & MC Dropout
**File:** `models/temporal_transformer.py`
Temporal forecasting predicts the future state of the exhibit to detect sudden or slow-moving anomalies.
- **Multi-Scale Analysis (Upgrade 3):** Two separate Causal Transformers are instantiated. The "Short-Horizon" head predicts based on the immediate past 8 frames (`K=8`). The "Long-Horizon" head downsamples the video to look across a far wider temporal window (`K=32`, `DOWNSAMPLE=4`), guarding against slow, gradual tampering that a short window might miss.
- **Monte Carlo Dropout Uncertainty (Upgrade 5):** The `mc_forward` subroutine intentionally keeps Dropout *active* during inference. It runs the model 20 times (`MC_DROPOUT_PASSES=20`) to calculate variance. If variance is high, the model is uncertain, which can be factored into anomaly scoring.

### 2.4 Upgrade 4: Deep Support Vector Data Description (SVDD) Energy Model
**File:** `anomaly/energy_model.py`
A custom neural network module (`EnergyModel`) designed to project multi-dimensional embeddings into a dense 64-dimensional hypersphere.
- **Mechanism:** Normal data is pushed toward a single fixed "center".
- **Purpose:** Creates a hard mathematical boundary for what explicitly constitutes "normal", acting as the final arbiter for anomaly generation instead of relying purely on fuzzy loss thresholds.

## 3. Pipeline Workflows & Backend Mechanics

### 3.1 FastAPI Infrastructure
**File:** `main.py`
The backend handles the orchestration via async Python. It features multiple key endpoints:
- `POST /api/train` and `POST /api/detect`: Takes uploaded mp4 files, strips them into frames, runs them through the architecture, and returns progress linearly through SSE (Server-Sent Events) streams so the frontend doesn't hang.
- `WS /ws/webcam`: A WebSocket endpoint allowing bi-directional streaming for live webcam detection. Base64 encoded JPEGs are piped in, and model scores are piped out at near-zero latency.

### 3.2 Human Masking & Preprocessing
**File:** `main.py`
As anomalies are meant for exhibits, humans (like museum staff or attendees) are false positives. The backend optionally instantiates a YOLOv8n object detection model via `HumanMaskFilter` to locate persons in the frame and explicitly mask them out before they reach the JEPA encoder.

### 3.3 Dynamic Anomaly Scoring Algorithm
**File:** `anomaly/scorer.py`
The final "Anomaly Score" is not generated from one metric, but is a weighted ensemble of five distinct signals:
1. Short-horizon temporal predicting error (`30%`)
2. Long-horizon temporal predicting error (`10%`)
3. Spatial reconstruction error (`25%`)
4. Deep SVDD Energy (`15%`)
5. Patch-level Nearest Neighbors error (`20%`)
These weights are normalized through dynamic scaling factors during the `api/calibrate` phase, calibrating the threshold against the 97th percentile of normal data.
