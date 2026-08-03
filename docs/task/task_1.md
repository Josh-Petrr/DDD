# Task 1: Driver Drowsiness Detection — Feature Fusion System

**Date:** 2026-07-28  
**Status:** In Progress  
**Owner:** Josh

---

## 1. Objective

Build a **driver drowsiness detection system** that generalizes to completely unseen subjects by fusing CNN visual features with geometric facial landmarks. This is not a basic classifier — it uses:
- Subject-disjoint evaluation (no identity leakage between train/test)
- Multi-modal feature fusion (CNN embeddings + facial geometry)
- Confidence-tiered analysis to show exactly where the fusion adds value
- Grad-CAM explainability to verify the model attends to physiologically meaningful regions

**SDG Alignment:**
- **SDG 3 — Good Health & Well-Being:** Target 3.6 (reducing road traffic deaths/injuries)
- **SDG 11 — Sustainable Cities & Communities:** Target 11.2 (safe, accessible transport systems)

---

## 2. Environment

| Resource | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| PyTorch | 2.7.1+cu118 |
| Python | 3.13.14 |
| torchvision | 0.22.1+cu118 |
| OpenCV | 4.12.0 |
| NumPy | 2.2.6 |
| Pandas | 2.2.3 |
| scikit-learn | 1.7.2 |
| MediaPipe | 0.10.35 (newly installed) |
| Matplotlib | 3.11.1 (newly installed) |
| Seaborn | 0.13.2 (newly installed) |

---

## 3. Dataset

**Source:** Driver Drowsiness Dataset (DDD)  
**Location:** `Driver Drowsiness Dataset (DDD)/`

| Class | Image Count | Format |
|---|---|---|
| Drowsy | 22,348 | PNG (~64KB each, face crops) |
| Non Drowsy | 19,445 | PNG (~64KB each, face crops) |
| **Total** | **41,793** | |

- Images are face crops from multiple subjects
- Filenames follow pattern: `A0001.png` (Drowsy), `B0001.png` (Non Drowsy)
- Images are approximately 224×224 pixels

---

## 4. Architecture Design

### Architecture Choice: EfficientNet-B0

Chosen over the original plan's YOLOv11x-cls because:
- EfficientNet-B0 is a purpose-built classification backbone (5.3M params)
- Fits comfortably in 6GB VRAM with mixed precision training
- torchvision already installed; Ultralytics was not available
- Strong ImageNet pretrained weights for transfer learning

### Three Model Variants (for comparison)

| Model | Input | Architecture | Purpose |
|---|---|---|---|
| **BaselineCNN** | Image only | EfficientNet-B0 + 2-class head | "Before" comparison |
| **GeometricOnlyMLP** | 4 landmarks only | Small MLP (32→16→2) | Ablation study |
| **FusionModel** | Image + 4 landmarks | EfficientNet-B0 (1280-d) + geometric (4-d) → Fusion MLP (256→64→2) | Main model |

### Fusion Architecture Diagram

```
Input Image (224×224)
    │                              │
    ▼                              ▼
┌──────────────┐          ┌──────────────────┐
│ EfficientNet │          │ MediaPipe         │
│ B0 Backbone  │          │ FaceMesh          │
│ (1280-d emb) │          │ (EAR,MAR,EB,HT)  │
└──────┬───────┘          └────────┬─────────┘
       │                           │
       └───────────┬───────────────┘
                   ▼
          ┌────────────────┐
          │ Concatenate    │
          │ [1280 + 4]     │
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │ Fusion MLP     │
          │ 1284→256→64→2  │
          │ (BN+ReLU+Drop) │
          └────────┬───────┘
                   ▼
          Drowsy / Non-Drowsy
```

### Geometric Features Extracted (via MediaPipe FaceMesh)

| Feature | Description | Drowsiness Signal |
|---|---|---|
| **EAR** (Eye Aspect Ratio) | Vertical/horizontal eye distance ratio | Drops when eyes close |
| **MAR** (Mouth Aspect Ratio) | Vertical/horizontal mouth distance ratio | Increases during yawning |
| **Eyebrow-to-eye distance** | Normalized eyebrow-eye gap | Drops with drooping/fatigue |
| **Head tilt angle** | Nose-to-chin vector angle | Captures head nodding |

---

## 5. Data Splitting Strategy

**Method:** Subject-disjoint splitting using filename sequence gap analysis

- Consecutive numbered images (e.g., A0001–A0080) are grouped as one subject/session
- A gap of ≥3 in numbering signals a new subject boundary
- Subjects are split at the group level — no subject appears in more than one partition

### Split Results

| Split | Total Images | Drowsy | Non-Drowsy | Subject Groups |
|---|---|---|---|---|
| **Train** | 29,901 (71.5%) | 16,074 (53.8%) | 13,827 (46.2%) | — |
| **Val** | 5,918 (14.2%) | 3,185 (53.8%) | 2,733 (46.2%) | — |
| **Test** | 5,974 (14.3%) | 3,089 (51.7%) | 2,885 (48.3%) | — |
| **Total** | 41,793 | 22,348 | 19,445 | 674 (362 drowsy + 312 non-drowsy) |

Split saved to: `splits.json`

---

## 6. Training Configuration

### 2-Stage Training Schedule

| Stage | Epochs | Backbone | Learning Rate | Purpose |
|---|---|---|---|---|
| **Stage 1** | 5 | Frozen | 1e-3 (head only) | Fast convergence of fusion head |
| **Stage 2** | 15 | Unfrozen | 1e-5 (backbone) / 1e-4 (head) | End-to-end fine-tuning |

### Hyperparameters

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| LR scheduler | Cosine Annealing |
| Early stopping patience | 5 epochs |
| Mixed precision | AMP (float16) |
| Dropout rate | 0.3 |
| Image size | 224×224 |
| ImageNet normalization | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### Data Augmentation (training only)

- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
- Random affine translation (±5%)

---

## 7. Evaluation Metrics

### Core Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC AUC, PR AUC

### Biometric Metrics (from original plan)
- **APCER** (Attack Presentation Classification Error Rate) — Drowsy classified as Non-Drowsy (missed drowsiness — dangerous!)
- **BPCER** (Bonafide Presentation Classification Error Rate) — Non-Drowsy classified as Drowsy (false alarm)
- **ACER** (Average Classification Error Rate) — (APCER + BPCER) / 2

### Confidence-Tiered Analysis
Predictions bucketed by confidence:
- **High (>0.9)** — how many and how accurate
- **Medium (0.7–0.9)** — the uncertain zone
- **Low (<0.7)** — where fusion should help most

### Explainability
- **Grad-CAM** heatmaps showing CNN attention regions for both classes

---

## 8. Files Created

| File | Purpose | Lines |
|---|---|---|
| `config.py` | Central configuration — paths, hyperparameters, constants | ~75 |
| `requirements.txt` | Dependencies list | 6 |
| `data_split.py` | Subject-disjoint train/val/test splitting | ~130 |
| `landmark_features.py` | MediaPipe geometric feature extraction (EAR, MAR, eyebrow, tilt) | ~250 |
| `extract_features.py` | Batch feature extraction with multiprocessing | ~100 |
| `dataset.py` | PyTorch Dataset with augmentation + geometric feature integration | ~150 |
| `models.py` | 3 model architectures (BaselineCNN, GeometricOnlyMLP, FusionModel) | ~190 |
| `train.py` | Training loop with 2-stage schedule, AMP, early stopping | ~210 |
| `train_all.py` | Orchestrator to train all 3 models sequentially | ~110 |
| `evaluate.py` | Comprehensive evaluation with APCER/BPCER/ACER | ~190 |
| `gradcam.py` | Grad-CAM visualization for CNN explainability | ~200 |
| `generate_report.py` | Plot generation and summary report | ~250 |
| `inference.py` | Standalone inference on unseen images | ~240 |

---

## 9. Dependencies Installed

```
pip install mediapipe matplotlib seaborn tqdm Pillow
```

All installed successfully:
- mediapipe 0.10.35
- matplotlib 3.11.1
- seaborn 0.13.2
- absl-py 2.5.0
- sounddevice 0.5.5
- opencv-contrib-python 5.0.0.93
- contourpy 1.3.3
- cycler 0.12.1
- kiwisolver 1.5.0

---

## 10. Current Progress

| Step | Status | Details |
|---|---|---|
| ✅ Environment setup | Complete | All dependencies installed |
| ✅ Code written | Complete | All 13 files created |
| ✅ Data splitting | Complete | 674 subject groups, disjoint split saved to `splits.json` |
| 🔄 Feature extraction | Running | MediaPipe processing 41,793 images (multiprocessed, 4 workers) |
| ⏳ Model training | Pending | Will train baseline → geometric → fusion sequentially |
| ⏳ Evaluation | Pending | Will run after training completes |
| ⏳ Report generation | Pending | Plots, tables, Grad-CAM visualizations |
| ⏳ Inference testing | Pending | Test on unseen images |

---

## 11. Expected Outputs (after training completes)

```
nndl project/
├── splits.json                  # Data split metadata
├── geometric_features.csv       # 41,793 rows of extracted landmarks
├── checkpoints/
│   ├── baseline_best.pth        # Best baseline model weights
│   ├── geometric_best.pth       # Best geometric model weights
│   └── fusion_best.pth          # Best fusion model weights
├── results/
│   ├── baseline_history.csv     # Training logs
│   ├── geometric_history.csv
│   ├── fusion_history.csv
│   ├── training_curves.png      # Loss & accuracy plots
│   ├── confusion_matrices.png   # All 3 models side-by-side
│   ├── roc_curves.png           # ROC comparison
│   ├── confidence_analysis.png  # Tier breakdown
│   ├── confidence_distributions.png
│   ├── gradcam_baseline.png     # Attention maps
│   ├── gradcam_fusion.png
│   ├── evaluation_summary.csv   # Metrics comparison table
│   └── evaluation_results.json  # Raw results
└── task/
    └── task_1.md                # This file
```

---

## 12. How to Run (after all steps complete)

```bash
# Step 1: Create data splits
python data_split.py

# Step 2: Extract geometric features (takes ~15-20 min)
python extract_features.py

# Step 3: Train all 3 models (takes ~1-1.5 hours)
python train_all.py

# Step 4: Generate evaluation report
python generate_report.py

# Step 5: Run inference on new images
python inference.py --input "path/to/unseen/folder" --model fusion
```
