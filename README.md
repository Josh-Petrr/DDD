# Driver Drowsiness Detection (DDD)

> **Real-time driver drowsiness detection using multi-modal feature fusion — EfficientNet-B0 CNN embeddings + MediaPipe facial geometry.**

---

## 🧠 Overview

This project builds a drowsiness detection system that generalizes to **completely unseen subjects** by fusing two complementary signal streams:

| Signal | What it captures |
|---|---|
| **CNN Visual Features** (EfficientNet-B0, 1280-d) | Appearance — eyelid heaviness, facial slack |
| **Geometric Landmarks** (MediaPipe, 4 features) | Kinematics — EAR, MAR, eyebrow drop, head tilt |

Three model variants are trained and compared:

| Model | Input | Purpose |
|---|---|---|
| **BaselineCNN** | Image only | "Before" comparison |
| **GeometricOnlyMLP** | 4 landmarks only | Ablation — what geometry alone achieves |
| **FusionModel** | Image + 4 landmarks | Main model |

---

## 📐 Architecture

```
Input Image (224×224)               MediaPipe FaceMesh
        │                                   │
        ▼                                   ▼
  EfficientNet-B0                  EAR · MAR · Eyebrow Dist · Head Tilt
  (1280-d embedding)                   (4 geometric features)
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
             Concatenate [1280 + 4 = 1284]
                       ▼
              Fusion MLP (256 → 64 → 2)
              (BatchNorm + ReLU + Dropout)
                       ▼
              Drowsy  /  Non-Drowsy
```

---

## 📁 Project Structure

```text
DDD/
├── backend/                    # FastAPI REST API
│   ├── main.py                 # Endpoints: /health, /predict/image, /predict/frame
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── services/
│   │   └── inference_service.py  # Model loading & prediction wrapper
│   └── requirements.txt
│
├── ml_pipeline/                # Machine Learning Pipeline
│   ├── config.py               # Paths, hyperparameters, constants
│   ├── data_split.py           # Subject-disjoint train/val/test splitting
│   ├── dataset.py              # PyTorch Dataset + data augmentation
│   ├── extract_features.py     # Batch MediaPipe feature extraction
│   ├── landmark_features.py    # EAR, MAR, eyebrow distance, head tilt
│   ├── models.py               # BaselineCNN, GeometricOnlyMLP, FusionModel
│   ├── train.py                # 2-stage training loop (AMP + early stopping)
│   ├── train_all.py            # Orchestrator — trains all 3 models
│   ├── evaluate.py             # Accuracy, F1, APCER/BPCER/ACER, ROC, PR
│   ├── gradcam.py              # Grad-CAM visual explanations
│   └── inference.py            # CLI inference on unseen images/folders
│
├── assets/
│   ├── checkpoints/            # Trained .pth model weights (download separately)
│   └── face_landmarker.task    # MediaPipe FaceLandmarker task file
│
├── data/                       # ⚠ NOT in Git (see below)
│   ├── Driver Drowsiness Dataset (DDD)/
│   ├── geometric_features.csv
│   └── splits.json
│
├── docs/
│   ├── MDS505.pdf              # Project report
│   ├── drowsiness_feature_fusion_plan.md
│   └── generate_report.py      # Produces all plots & evaluation tables
│
├── results/                    # Generated plots & CSVs (after training)
├── requirements.txt
└── README.md
```

---

## 🗃️ Dataset

**Driver Drowsiness Dataset (DDD)**

| Class | Images |
|---|---|
| Drowsy | 22,348 PNG face crops |
| Non Drowsy | 19,445 PNG face crops |
| **Total** | **41,793** |

> ⚠️ The dataset is **not included in this repository** (≈2.6 GB). Place it at `data/Driver Drowsiness Dataset (DDD)/` before running.

Subject-disjoint splits (no identity leakage between train/test):
| Split | Images | Notes |
|---|---|---|
| Train | 29,901 (71.5%) | 674 subject groups |
| Val | 5,918 (14.2%) | |
| Test | 5,974 (14.3%) | Fully unseen subjects |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/DDD.git
cd DDD
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 2. Add the Dataset
Place the dataset at:
```
data/Driver Drowsiness Dataset (DDD)/Drowsy/
data/Driver Drowsiness Dataset (DDD)/Non Drowsy/
```

### 3. Extract Geometric Features *(~15–20 min)*
```bash
python ml_pipeline/extract_features.py
```
Outputs: `data/geometric_features.csv`

### 4. Create Data Splits
```bash
python ml_pipeline/data_split.py
```
Outputs: `data/splits.json`

### 5. Train All Models *(~1–1.5 hrs on GPU)*
```bash
python ml_pipeline/train_all.py
```
Outputs: `assets/checkpoints/baseline_best.pth`, `geometric_best.pth`, `fusion_best.pth`

### 6. Generate Evaluation Report
```bash
python docs/generate_report.py
```
Outputs: `results/` — plots, confusion matrices, ROC curves, Grad-CAM maps

### 7. Run CLI Inference
```bash
python ml_pipeline/inference.py --input path/to/image.jpg --model fusion
```

### 8. Start the REST API Backend
```bash
uvicorn backend.main:app --reload --port 8000
```
Interactive docs: **http://127.0.0.1:8000/docs**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check model load status, device, checkpoint |
| `POST` | `/api/predict/image` | Upload image file → drowsiness prediction |
| `POST` | `/api/predict/frame` | Send base64 webcam frame → prediction |

**Sample response:**
```json
{
  "success": true,
  "label": "Drowsy",
  "confidence": 0.94,
  "is_drowsy": true,
  "drowsy_probability": 0.94,
  "non_drowsy_probability": 0.06,
  "geometric_features": {
    "ear": 0.18,
    "mar": 0.42,
    "eyebrow_dist": 0.12,
    "head_tilt": -8.3
  },
  "model_used": "fusion"
}
```

---

## 📊 Evaluation Metrics

- **Accuracy, Precision, Recall, F1-Score**
- **ROC AUC, PR AUC**
- **APCER** — Drowsy missed as Non-Drowsy *(safety-critical)*
- **BPCER** — Non-Drowsy misclassified as Drowsy *(false alarm)*
- **ACER** — (APCER + BPCER) / 2
- **Confidence-tiered analysis** — High / Medium / Low confidence buckets
- **Grad-CAM heatmaps** — Visual explainability

---

## ⚙️ Training Config

| Parameter | Value |
|---|---|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Batch size | 32 |
| Optimizer | AdamW |
| LR (Stage 1) | 1e-3 (frozen backbone) |
| LR (Stage 2) | 1e-4 head / 1e-5 backbone |
| Scheduler | Cosine Annealing |
| Mixed Precision | AMP (float16) |
| Early stopping | Patience = 5 |

---

## 🌱 SDG Alignment

- **SDG 3 — Good Health & Well-Being** (Target 3.6): Reducing road traffic fatalities
- **SDG 11 — Sustainable Cities & Communities** (Target 11.2): Safe transport systems

---

## 🛠️ Environment

| Package | Version |
|---|---|
| Python | 3.13.14 |
| PyTorch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| OpenCV | 4.12.0 |
| MediaPipe | 0.10.35 |
| FastAPI | ≥0.100.0 |
| NumPy | 2.2.6 |
| Pandas | 2.2.3 |
| scikit-learn | 1.7.2 |
