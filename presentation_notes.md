# JEPA Defect Detection Project - Presentation Notes
*Project for Unisys Innovation Program - Agent Unleashed Category*

This document provides a comprehensive "nook and corner" explanation of the JEPA (Joint Embedding Predictive Architecture) codebase, detailing the components from training to detection. It also includes steps for running the POC demonstration.

---

## 🏗️ 1. High-Level Architecture
The project shifts away from traditional pixel-reconstruction anomaly detection (like Autoencoders) and instead uses **JEPA**, which operates entirely in the **embedding space**. By predicting high-level semantic features rather than exact pixel values, it becomes highly robust to harmless variations (e.g., lighting changes or camera shake) while remaining highly sensitive to true structural defects.

We have implemented **5 Architectural Upgrades**:
1. **Partial ViT Encoder Fine-Tuning:** Enhances domain adaptation.
2. **Cross-Attention Spatial JEPA:** Learns relationships between visible and masked patches.
3. **Multi-Scale Temporal Transformer:** Uses both Short (K=8) and Long (K=32) context windows to understand motions at different speeds.
4. **Deep SVDD Energy Model:** Learns the explicit boundaries of "normalcy."
5. **MC Dropout Uncertainty:** Adds confidence bounds to predictions.

---

## 🎓 2. The Training Pipeline (`/api/train`)
Training relies ONLY on **normal, defect-free video**. No anomalous labels are needed.

### Stage 1: Frame Extraction & Preprocessing
* **Sampling:** Videos are sampled at `TARGET_FPS` (default 3 FPS).
* **Human Masking:** (Optional but default) A YOLO-based masker blacks out human elements so the model strictly learns assembly line/exhibit mechanics.

### Stage 2: Encoding (`ViT-B/16*`)
* The images are passed through a Vision Transformer (`vit_base_patch16_224`).
* **Outputs:** 
  1. `[CLS] token` (represents the global semantic context).
  2. `Patch embeddings` (A grid of 14x14 = 196 local patch embeddings).
* *Upgrade 1:* The first 8 blocks of ViT are frozen, while the remaining blocks are fine-tuned with a low learning rate (`1e-5`) to adapt specifically to your factory environment.

### Stage 3a: Joint Temporal & Spatial JEPA 
The system trains three distinct Transformer networks:
1. **Short Temporal Predictor (`K=8`):** Looks at the past 8 `[CLS]` embeddings and tries to predict the next future `[CLS]` embedding.
2. **Long Temporal Predictor (`K=32` downsampled to 8):** Skips frames (picks every 4th frame) to understand macro-level, slow-evolving workflows.
3. **Cross-Attention Spatial Head:** Masks out 50% of an image's patches, then uses the visible patches to predict the embeddings of the masked ones.
* **Loss Function:** `combined_loss( temporal + 0.5 * temporal_long, spatial )` using Cosine Similarity/MSE loss.

### Stage 3b: Deep SVDD Energy Model
* Support Vector Data Description (SVDD) is an anomaly detection technique.
* We freeze the learned `[CLS]` embeddings and train a narrow neural net to map all normal embeddings into a tightly packed hypersphere.
* The "center" of this sphere is calculated. The SVDD loss penalises embeddings that stray too far from this center, establishing a hard mathematical boundary for "normal."

### Stage 4: Threshold Calibration (`/api/calibrate`)
* Once trained, all normal frames are passed through the entire scoring pipeline.
* Errors across the 5 dimensions (Short Temporal, Long Temporal, Spatial, SVDD Energy, Uncertainty) are calculated.
* The framework automatically calculates normalisation scales (`t_scale`, `e_scale`, etc.) so they have equal weight.
* **Threshold Setting:** The system finds the **97th percentile** of all normal scores and sets this as the hard threshold.

---

## 🔍 3. The Detection Pipeline (`/api/detect`)
During inference (either uploaded suspect video or Live Webcam feed), the system evaluates each frame on the fly.

1. **Encode:** The frame is encoded using the trained ViT.
2. **Score Calculation (`AnomalyScorer`):**
   * **Temporal Error:** How poorly did the model predict this frame's concept from the past 8 frames?
   * **Spatial Error:** How poorly did it predict the hidden patch subsets of this frame?
   * **Energy Error:** How far is this frame's `[CLS]` embedding from the SVDD normal sphere center?
   * **Uncertainty Penalty:** *Upgrade 5* enables Monte Carlo Dropout (running the frame through the network 20 times with dropout on). If the outputs vary wildly, the model is highly uncertain, and an uncertainty penalty is added.
3. **Aggregated Score:** Combine the scores using weighted sums (`0.4*T + 0.15*T_L + 0.25*S + 0.20*E` + Uncertainty).
4. **Decision:** If the aggregated score > Threshold, it is flagged as an **Anomaly**.

### The "WOW" Factor: Localised Semantic Heatmaps (`/api/analyze-anomaly`)
When an anomaly is flagged, the frontend can call a semantic matching endpoint.
* **Semantic Match:** It calculates the cosine similarity between the anomaly's `[CLS]` embedding and ALL normal frames. It finds the "closest matching normal state".
* **Spatial Difference:** It compares the 14x14 patch embeddings of the anomalous frame vs. the matched normal frame.
* **Heatmap Generation:** It generates an L2 distance heatmap overlaid on the original image, visually pointing a localized red hotspot directly at the defect (e.g. telling the operator "this screw is missing" or "this gear is jammed").

---

## 🚀 4. POC Demonstration Guide
How to present this live to your Unisys Mentors.

### Prerequisites:
1. Make sure Python dependencies in `requirements.txt` are installed.
2. Ensure you have two videos ready on your drive:
   * **Normal Video:** e.g., `videos/2.mp4`
   * **Defect Video:** e.g., `videos/WhatsApp Video...mp4`

### Running the System
Run the backend and frontend simultaneously:

**Terminal 1 (Backend - FastAPI):**
```bash
cd jepa
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend - Streamlit):**
```bash
cd jepa
streamlit run app4.py
```

### Flow of Presentation:
1. **Tab 1 - Training (Simulated):** Explain that the system needs no labels. Show the UI where you upload the normal video. Click "Start Training". *Note: Since actual training might take a few minutes depending on your hardware, you might want to show pre-trained loss curves and mention "for the sake of time, we have pre-trained this."*
2. **Tab 2 - Calibration:** Explain the SVDD and how the system dynamically finds its own threshold (97th percentile).
3. **Tab 3 - Detection:** 
   * Upload the Defect video.
   * Watch the live graph plot the "Anomaly Score" against the dotted red threshold line.
   * Pause/Highlight when the score spikes and the red alert box pulses.
   * Describe how the temporal and spatial JEPA embeddings caught the anomaly without ever seeing one before.
4. **Architecture Mention:** Briefly summarize the 5 Upgrades (especially Semantic matching & MC Dropout) which sets this strictly in the "Agent Unleashed" advanced ML category.
