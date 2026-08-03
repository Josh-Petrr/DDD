# Driver Drowsiness Detection (DDD)

> **Real-time driver drowsiness detection using multi-modal feature fusion — EfficientNet-B0 CNN embeddings + MediaPipe facial geometry.**

---

## 🧠 Overview

This project builds a **real-time drowsiness detection system** that generalizes to **completely unseen subjects** by combining two complementary signal streams and adding temporal sequence modeling:

| Signal | What it captures |
|---|---|
| **CNN Visual Features** (EfficientNet-B0, 1280-d) | Appearance — eyelid heaviness, facial slack |
| **Geometric Landmarks** (MediaPipe, 4 features) | Kinematics — EAR, MAR, eyebrow drop, head tilt |

The final pipeline is a two-stage **CNN-LSTM** system:

| Stage | Model | Role |
|---|---|---|
| **Stage 1** | FusionGRL_V4 (EfficientNet-B0 + GRL) | Per-frame feature extraction |
| **Stage 2** | DrowsinessLSTM | 30-frame temporal sequence classification |

The LSTM looks at a **rolling 1-second window** of 30 fused feature vectors to detect drowsiness patterns over time (sustained eye closure, head nodding, prolonged yawning), achieving **87.88% Recall** and **12.12% APCER** on completely unseen drivers.

---

## 📐 Architecture

```
Input Frame (224×224)              MediaPipe FaceLandmarker
        │                                   │
        ▼                                   ▼
  EfficientNet-B0              EAR · MAR · Eyebrow Dist · Head Tilt
  (1280-d embedding)               (4 geometric features)
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
             Concatenate [1280 + 4 = 1284]
                       │
              (per frame, repeated 30 times)
                       │
                       ▼
          Rolling Buffer [30 frames]
                       │
                       ▼
           DrowsinessLSTM
      (LSTM 64 hidden + Dropout 0.6)
                       ▼
              Drowsy  /  Awake
```

The CNN is trained with a **Gradient Reversal Layer (GRL)** domain classifier that forces it to learn identity-invariant (driver-agnostic) features.

---

## 📁 Project Structure

```text
DDD/
├── app.py                      # 🚀 FastAPI + WebSocket live inference server
│
├── static/                     # Live Webcam Dashboard (Frontend)
│   ├── index.html              # Dark-mode glassmorphic UI
│   ├── styles.css              # Animations, gauges, threat level cards
│   └── script.js               # WebSocket client, webcam capture, UI updates
│
├── ml_pipeline/                # Machine Learning Pipeline
│   ├── config.py               # Paths, hyperparameters, constants
│   ├── data_split.py           # Subject-disjoint train/val/test splitting
│   ├── dataset.py              # PyTorch Dataset + data augmentation (CNN)
│   ├── dataset_lstm.py         # Sequence Dataset — sliding window over .npy arrays
│   ├── landmark_features.py    # EAR, MAR, eyebrow distance, head tilt (MediaPipe)
│   ├── models.py               # FusionGRL_V4 (CNN) + DrowsinessLSTM
│   ├── train.py                # 2-stage CNN training loop (AMP + GRL)
│   ├── train_all.py            # Orchestrator — trains all CNN variants
│   ├── train_lstm.py           # LSTM training loop (sliding window sequences)
│   ├── evaluate.py             # Accuracy, F1, APCER/BPCER/ACER, ROC, PR
│   ├── gradcam.py              # Grad-CAM visual explanations
│   └── inference.py            # CLI inference on unseen images/folders
│
├── assets/
│   ├── checkpoints/
│   │   ├── V4/
│   │   │   └── fusion_grl_4_best.pth   # Best CNN checkpoint
│   │   └── lstm_best.pth               # Best LSTM checkpoint
│   └── face_landmarker.task    # MediaPipe FaceLandmarker task file
│
├── data/                       # ⚠ NOT in Git — place locally
│   ├── Driver Drowsiness Dataset (DDD)/   # Raw images (~41K)
│   ├── geometric_features.csv             # MediaPipe features for all frames
│   ├── splits.json                        # Subject-disjoint split assignments
│   └── sequence_features/                 # Pre-extracted CNN embeddings (.npy)
│
├── docs/
│   ├── MDS505.pdf                  # Project report
│   ├── generate_report.py          # Evaluation report for CNN models
│   ├── generate_lstm_report.py     # Evaluation report for LSTM model
│   └── task/                       # Detailed task logs (task_1.md … task_7.md), brief process of how the project was developed
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

### 5. Train the CNN Models *(~1–1.5 hrs on GPU)*
```bash
python ml_pipeline/train_all.py
```
Outputs: `assets/checkpoints/V4/fusion_grl_4_best.pth` (and other variants)

### 6. Pre-extract CNN Features for LSTM *(~20–30 min)*
This runs all 41K images through the frozen CNN and saves the embeddings:
```bash
python ml_pipeline/extract_features.py --mode sequence
```
Outputs: `data/sequence_features/*.npy`

### 7. Train the LSTM *(~5–10 min)*
```bash
python ml_pipeline/train_lstm.py
```
Outputs: `assets/checkpoints/lstm_best.pth`

### 8. Generate Evaluation Reports

**For the CNN models** (Baseline, Geometric, FusionGRL variants):
```bash
python docs/generate_report.py
```
Outputs: `results/result_V4/` — confusion matrices, ROC curves, confidence analysis, Grad-CAM maps

**For the LSTM sequence model:**
```bash
python docs/generate_lstm_report.py
```
Outputs: `results/lstm_1/` — training curves, confusion matrix, ROC curve, confidence tiers

### 9. Launch the Live Webcam App
```bash
python app.py
```
Open **http://127.0.0.1:8000** in your browser. Allow webcam access and click **Start Detection**.

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

## 📊 Results

> **Note on terminology:** "CNN-LSTM" and "FusionGRL_V4-LSTM" refer to the **exact same pipeline**. The "CNN" used inside the LSTM pipeline is the `FusionGRL_V4` model. The LSTM then classifies sequences of 30 frames of its embeddings.

### Full Experiment Results (sourced from `results/` folder)

| # | Folder | Model | Split Type | Acc | Recall | F1 | APCER | BPCER | ROC AUC |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | `results_joint` | Baseline CNN | Joint (⚠️ Cheating) | 97.04% | 99.97% | 97.21% | 0.03% | 6.10% | 0.9985 |
| 2 | `results_joint` | Geometric MLP | Joint (⚠️ Cheating) | 65.94% | 62.61% | 65.53% | 37.39% | 30.50% | 0.7302 |
| 3 | `results_joint` | Fusion CNN+Geo | Joint (⚠️ Cheating) | 97.19% | 100.0% | 97.35% | 0.00% | 5.82% | 0.9995 |
| 4 | `results_2` | Baseline CNN v2 | Disjoint (Bug) | 61.31% | 81.79% | 64.89% | 18.21% | 54.60% | 0.7607 |
| 5 | `results_2` | Geometric MLP v2 | Disjoint (Bug) | 46.85% | 63.48% | 51.08% | 36.52% | 66.07% | 0.5840 |
| 6 | `results_2` | Fusion v2 | Disjoint (Bug) | 46.04% | 78.36% | 55.94% | 21.64% | 79.06% | 0.6453 |
| 7 | `results_3` | Baseline CNN v2 | Disjoint (Fixed) | 76.12% | 88.46% | 78.74% | 11.54% | 36.22% | 0.6447 |
| 8 | `results_3` | Geometric MLP v2 | Disjoint (Fixed) | 62.64% | 53.83% | 59.03% | 46.17% | 28.56% | 0.7305 |
| 9 | `results_3` | Fusion v2 | Disjoint (Fixed) | 78.21% | 92.64% | 80.96% | 7.36% | 36.22% | 0.6661 |
| 10 | `result_4` | Baseline CNN v3 | Disjoint + GRL | 71.49% | 79.20% | 73.53% | 20.80% | 36.22% | 0.8273 |
| 11 | `result_4` | Geometric MLP v3 | Disjoint + GRL | 60.80% | 58.21% | 59.75% | 41.79% | 36.62% | 0.6643 |
| 12 | `result_4` | Fusion v3 | Disjoint + GRL | 63.16% | 65.92% | 64.15% | 34.08% | 39.60% | 0.6466 |
| 13 | `result_4` | **FusionGRL v3** | Disjoint + GRL | 71.22% | 81.04% | 73.79% | 18.96% | 38.61% | 0.6674 |
| 14 | `result_V4` | **FusionGRL_V4** | Disjoint + Higher Dropout | 70.87% | 77.91% | 72.79% | 22.09% | 36.17% | 0.7757 |
| 15 | `final_results` ⭐ | **FusionGRL_V4 → LSTM** | Disjoint + Temporal | **72.52%** | **87.88%** | **76.32%** | **12.12%** | 43.08% | 0.6744 |

> ⭐ = **Production model** running in the live webcam app (`app.py`)
>
> ⚠️ Experiments 1–3 used a **joint (non-disjoint) split** — the model memorised faces. Those results are invalid for real-world use.
>
> **APCER** = % of drowsy events missed (lower = safer). **BPCER** = % of false alarms (lower = fewer interruptions).

### Metric Definitions

| Metric | Meaning |
|---|---|
| **APCER** | Drowsy missed as Non-Drowsy — *the safety-critical metric* |
| **BPCER** | Non-Drowsy misclassified as Drowsy — *false alarm rate* |
| **ACER** | (APCER + BPCER) / 2 — balanced error |
| **ROC AUC** | Ability to rank drowsy above awake across all thresholds |

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
