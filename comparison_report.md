# Architectural Comparison Report: Custom Exhibit JEPA vs. Meta's V-JEPA 2
*Prepared for the Unisys Innovation Program Presentation*

This report provides a detailed, technical comparison between the custom-built **Exhibit Defect Detection JEPA** (`v1`) and Meta's official **V-JEPA 2** foundation model (`v2`). It outlines the engineering choices, architectural components, and computational efficiency of both approaches.

---

## 1. Core Philosophy & Goals
Both models operate on the core principle of **Joint Embedding Predictive Architecture (JEPA)**: they perform anomaly detection by predicting high-level semantic embeddings rather than reconstructing raw, exact pixels. This makes both models highly immune to superficial noise (like lighting changes).

* **Our Custom Approach (Exhibit JEPA):** Focuses on being **dynamic, highly adaptable, and computationally lightweight**. It builds the JEPA physics specifically around the exact mechanical workflows of *your* environment from scratch.
* **Meta's Approach (V-JEPA 2):** Focuses on being an **encyclopedic foundation model**. It relies on millions of hours of pre-trained internet video to understand general physics and motion out-of-the-box, sacrificing adaptability for raw generalization power.

---

## 2. Backbone Architecture (The "Eyes")

| Feature | Custom Exhibit JEPA | Meta V-JEPA 2 |
| :--- | :--- | :--- |
| **Model Type** | 2D Vision Transformer (`ViT-B/16`) processing single frames. | 3D Spatio-Temporal Video Transformer processing video "tubes". |
| **Parameters** | ~86 Million (Base) | ~300M to 1B+ (Large/Giant) |
| **Input Shape** | `(1, 3, 224, 224)` (Single Frame) | `(B, 3, 16, 256, 256)` (16-Frame Clip) |
| **Domain Adaptation**| **Yes.** The first 8 blocks are frozen, but the final blocks undergo partial fine-tuning on the user's specific factory video to learn niche textures. | **No.** V-JEPA is entirely frozen during deployment and acts as a generic "black-box" feature extractor. |

---

## 3. Temporal Modeling (Predicting Motion)

How does the pipeline understand when a machine "jams" or moves incorrectly?

### Custom Exhibit JEPA (Explicit Predictive Mapping)
We engineered a highly specific **Multi-Scale Temporal Transformer**. 
* **Short Context Head (K=8):** Looks at the previous 8 frames (about 2 seconds of video) to predict sudden, rapid defects (e.g., a belt snapping or a sudden jolt).
* **Long Context Head (K=32, downsampled):** Analyzes frames over 10 seconds to detect macro-level process failures (e.g., an assembly line moving 10% too slow over a long period).
* *Efficiency:* Highly interpretable. It mathematically outputs a distinct "Temporal Error" score when its motion prediction fails.

### Meta V-JEPA 2 (Implicit Attention)
It uses **Deep Self-Supervision**.
* It does not have separate "motion" heads. Instead, the entire sequence of 16 frames is flattened into 3D space-time patches. 
* All patches attend to each other simultaneously. The "motion" is inherently captured within the massive `last_hidden_state` embedding.
* *Efficiency:* It has a world-class understanding of physics, but it is a "black box." You cannot easily separate *why* an anomaly occurred (motion vs. spatial).

---

## 4. Spatial Modeling (Finding Missing Parts)

How do the pipelines find physical defects, like a missing screw or misplaced component?

### Custom Exhibit JEPA (Cross-Attention Masking)
We built a custom **Spatial JEPA Head**. 
* During training, we randomly black out 50% of the normal image. A Cross-Attention Decoder is forced to look at the visible patches and guess the embeddings of the hidden patches. 
* When a test image has a missing gear, the model predicts the gear should be there. The mathematical difference between the physical space and the mental space immediately flags the anomaly!
* Furthermore, this allows us to generate a **localized L2 heatmap** to circle the exact pixel location of the defect.

### Meta V-JEPA 2 (Dense Predictive Loss)
* V-JEPA 2 was pre-trained using "Dense Predictive Loss" where it had to predict masked spatial and temporal tokens across internet videos. 
* However, during *inference* (deployment), this capability isn't accessible to us without complex feature inversion. It simply outputs a single massive embedding summarizing the frame. Heatmap generation requires advanced grad-CAM techniques rather than direct spatial prediction.

---

## 5. Anomaly Scoring & Uncertainty Handling

| Feature | Custom Exhibit JEPA | Meta V-JEPA 2 |
| :--- | :--- | :--- |
| **Scoring Mechanism** | Multi-Component Weighted Sum: <br> `0.4*Short + 0.15*Long + 0.25*Spatial + 0.20*SVDD Energy` | Unified representation passed natively to an SVDD Energy Model. |
| **Uncertainty Penalty** | Uses **Monte Carlo (MC) Dropout Inference**. It runs 20 stochastic passes per frame; if the variance is high, it automatically lowers confidence, preventing false positives. | None inherently available without hacking the HuggingFace transformer pipeline. |
| **Threshold Calibration** | Computes components via Deep SVDD, Auto-calibrating to the 97th percentile of normal data dynamically. | Computes exactly the same way: SVDD over the 16-frame embeddings. |

---

## 6. Computational & Operational Efficiency (The Verdict)

### The Custom Baseline (`jepa/`) Is Vastly Superior for Edge/IoT Devices.
By building custom temporal and spatial heads on top of a lightweight 86M parameter image model, the custom pipeline can be run on **consumer laptops or edge factory IoT devices** (even fast CPUs). While it *requires* a 2-5 minute local pre-training phase on a normal video to "learn the physics" of the room, it is vastly more agile, explainable, and deployable. 

### The Meta Foundation Baseline (`jepav2/`) Is Vastly Superior for Zero-Shot Power (Assuming Infinite Hardware).
Meta's approach utilizes hundreds of millions of parameters and natively understands moving physical space-time. Because of this, it **requires zero temporal or spatial training locally**. It skips entirely to the final SVDD phase. However, processing a massive 3D transformer requires **datacenter-grade hardware (NVIDIA A100s / 24GB+ VRAM)**, making it wildly expensive to deploy locally on a factory floor. 

---

### Mentor Presentation Takeaway
**"We built two architectures to prove scale. Our custom Exhibit JEPA proves that highly capable, interpretable spatial and temporal anomaly detection can be run dynamically on edge devices for cheap. The V-JEPA 2 foundation pipeline proves that as hardware scales, our software can plug immediately into frontier models to leverage internet-scale physics knowledge without changing our unsupervised goal."**
