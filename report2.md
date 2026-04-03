# Meta V-JEPA 2 (Open Source): Project Implementation Report

## 1. Overview
The `jepav2/Colab_VJEPA2_Workflow.ipynb` notebook represents a highly streamlined, standalone workflow leveraging Meta's state-of-the-art open-source Video Joint-Embedding Predictive Architecture (V-JEPA 2) foundation model. Rather than attempting to train custom spatial or temporal forecasting transformers from scratch on limited localized hardware, this project treats the massive pre-trained foundation model as a fixed "zero-shot" feature extractor, bolting on a lightweight anomaly detection mechanism at the very end.

## 2. Core Architectural Setup

### 2.1 The Foundation Model: Meta V-JEPA 2
**Component:** `facebook/vjepa2-vitl-fpc16-256-ssv2` via HuggingFace Transformers
- **Extraction over Prediction:** Meta's model is used strictly in inference mode (`eval`). There is no training loop for the core model. 
- **Mechanism:** Incoming video clips are processed in chunks of 16 frames (`CLIP_DURATION=16`) representing 2 seconds of video at 8 FPS. The model outputs sprawling `1024-D` vector embeddings that inherently encapsulate both spatial relationships and temporal motion flawlessly due to its millions of hours of pre-training.

### 2.2 PCA Dimensionality Reduction
**File:** Notebook Cell 8
Because V-JEPA 2 generates 1024-D embeddings, and typical fine-tuning sets for specific exhibits are extremely small (~30 to 50 clips max), the curse of dimensionality guarantees catastrophic model collapse if those raw embeddings are fed directly into a classifier.
- **Mechanism:** Scikit-Learn's Principal Component Analysis (PCA) maps the 1024-D vectors down to a stable 64-D subspace prior to learning the anomaly boundary. This preserves the vast majority of the variance while preventing neural network collapse.

### 2.3 Deep SVDD Network
**File:** Notebook Cell 4 (`EnergyModel`)
This is the only piece of neural network architecture mathematically trained during execution.
- **Mechanism:** A lightweight feedforward Multi-Layer Perceptron (Linear → BatchNorm → GELU → Dropout). It takes the PCA-reduced 64-D vectors and trains to map them tightly around a static representation coordinate (the "center"). 
- **Anti-Collapse Mechanism:** The hypersphere center explicitly undergoes bound-checking (`c[(c.abs() < 0.01) & (c > 0)] = 0.01`) during `fit_center` to ensure it doesn't trivially drop to absolute 0.0.

## 3. Workflow & Anomaly Pipeline

### 3.1 Clip Extraction Geometry
The raw video isn't fed frame-by-frame. It's aggressively sliced overlapping segments (`video_to_clips`), striding by 4 frames at a time. The midpoint frame of each clip is taken out and stored as the visual representation of that segment to later show the user.

### 3.2 Calibration Phase
After extracting spatial-temporal tokens from the normal video and routing them through PCA and into the SVDD model, the system logs the *Energy Scores* (distances to the mathematical center) for all normal clips. 
- A hard mathematical threshold is declared by running `np.percentile(..., 97)`, cutting off the extreme 3% of regular scores to account for naturally noisy camera frames.

### 3.3 Semantic Match and Heatmap Localisation
**File:** Notebook Cell 10
When an anomaly stringently exceeds the 97th percentile threshold limit, it is necessary to not just flag it contextually, but show the user *where* in the video the problem occurred.
- **Matching:** The pipeline calculates the Cosine Similarity between the raw 1024-D extracted tensor of the anomaly and the entire database of normal vectors. Whichever normal vector has the highest similarity is crowned the "Best Match".
- **Heatmap:** It bypasses the abstract CLS token arrays and taps the underlying sequence tokens representing the physical image grid. It calculates the L2 mathematical difference between the anomaly's grid tokens and the normal matched grid tokens. Min-max normalization transforms this patch-by-patch error into a 14x14 grid, mapping standard values into a cv2 `JET` heatmap, overlaying red wherever raw pixel embeddings mismatch massively against expectation.
