# JEPA Exhibit Defect Detection System

A self-supervised spatio-temporal predictive model that learns the "normal behavioral manifold" of an exhibit and detects deviations in latent representation space without requiring defect labels.

This project uses the Joint-Embedding Predictive Architecture (JEPA) for high-accuracy anomaly detection in video streams (e.g., CCTV context, physical object defect detection, or an exhibition defect detection system).

## 🧠 Architecture Highlights
* **Feature Extraction**: ViT-B/16 (Partial Domain Adaptation, fine-tuned last 4 blocks).
* **JEPA Core**: Cross-Attention Spatial JEPA and Multi-Scale Temporal analysis (Short + Long windows).
* **Anomaly Scoring**: 5-components combined: Temporal (Short+Long), Spatial, Energy Model (Deep SVDD), and MC Dropout Uncertainty.
* **Human Masking**: YOLO-based human mask filter to ignore humans in the exhibit during scoring.

## 🚀 Features

### 1. Training (Self-Supervised)
* Extracts frames from a normal video sequence.
* Encodes frames using ViT-B/16.
* Trains Temporal Transformers and Spatial JEPA Head.
* Trains Deep SVDD Energy Model.

### 2. Calibration
* Computes per-frame anomaly scores on the normal reference video.
* Dynamically sets an anomaly threshold at the 97th percentile.

### 3. Detection & Inference
* Real-time anomaly score visualization for test videos or webcam streams.
* Visual flags when the combined anomaly score exceeds the calibrated threshold.
* Option to mask humans out of anomaly calculations.

## 🛠️ Project Structure
* `app.py`: Main Streamlit UI for training, calibrating, and testing models interactively.
* `main.py`: FastAPI backend offering endpoints for programmatic training (`/api/train`), calibration (`/api/calibrate`), and detection (`/api/detect`) via Server-Sent Events (SSE). Also supports live webcam inference via WebSockets (`/ws/webcam`).
* `config.py`: Configuration details for hardware (CUDA/CPU), thresholds, FPS constraints, etc.
* `models/`: Contains network architectures (JEPA, Temporal Transformer, Energy models).
* `anomaly/`: Scorers and analyzers mapping model errors to an anomaly metric.
* `inference/`: Pipeline setup for continuous evaluation.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "Exhibit Defect Detection System"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Running the Application

### 1. Streamlit Interactive Dashboard
Run the Streamlit frontend to interact with the training and testing phases via a GUI:
```bash
streamlit run app.py
```

### 2. FastAPI Backend
Run the backend server for API access and WebSocket streaming:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 Pipeline Flow Overview
1. **Normal Video Source**: Frame sampler (e.g., 3 FPS, 224x224, CLAHE normalization).
2. **Frozen ViT-B/16**: Extracts CLS and patch embeddings.
3. **JEPA Training**: Trains temporal transformers and spatial JEPA head to predict and reconstruct representations.
4. **Calibration**: Mahalanobis fit and percentile calibration define the threshold.
5. **Score Evaluation**: `Score = α·Temporal + β·Spatial + γ·Mahalanobis + ...`
6. **Trigger Alert**: Alert triggered if Frame Score > Threshold.
