# Task 2: Project Restructuring, Web Backend API, Training Execution & Unseen Data Verification

**Date:** 2026-07-28  
**Status:** Complete  
**Owner:** Josh & AI Pair Programmer  

---

## 1. Executive Summary

Task 2 focused on transforming the flat machine learning codebase into an enterprise-ready, modular project architecture. This included:
- Restructuring all source files into modular packages (`ml_pipeline/`, `backend/`, `assets/`, `data/`, `docs/`).
- Building a **FastAPI REST API** server for real-time web inference.
- Fixing all cross-platform compatibility bugs (Windows DataLoader, PyTorch AMP deprecations, encoding errors).
- Executing model training across 3 architectures and verifying accuracy (**Fusion Model achieved 90.23%**).
- Verifying subject-disjoint data split integrity and exporting **5,974 completely unseen test images**.
- Configuring `.gitignore` and pushing the repository to GitHub (`Josh-Petrr/DDD`).

---

## 2. Directory Architecture & Modularization

The project root was reorganized into a clean directory layout:

```text
DDD/
├── backend/                       # FastAPI REST API Backend
│   ├── main.py                    # Endpoints: /health, /api/predict/image, /api/predict/frame
│   ├── schemas.py                 # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   └── inference_service.py   # Model loading & prediction wrapper
│   └── requirements.txt           # Backend-specific dependencies
│
├── ml_pipeline/                   # Machine Learning Pipeline Package
│   ├── __init__.py
│   ├── config.py                  # Global path & hyperparameter configuration
│   ├── data_split.py              # Subject-disjoint train/val/test splitting
│   ├── dataset.py                 # PyTorch Dataset & augmentation pipeline
│   ├── extract_features.py        # MediaPipe landmark feature extraction
│   ├── extract_filenames.py       # Dataset filename indexer
│   ├── export_unseen_data.py      # Unseen test set exporter
│   ├── landmark_features.py       # EAR, MAR, Eyebrow Dist & Head Tilt calculator
│   ├── models.py                  # BaselineCNN, GeometricOnlyMLP, FusionModel
│   ├── train.py                   # 2-stage training engine with AMP & Early Stopping
│   ├── train_all.py               # Sequential model training orchestrator
│   ├── evaluate.py                # Comprehensive metrics (APCER/BPCER/ACER, ROC, PR)
│   ├── gradcam.py                 # Grad-CAM explainability visualizer
│   └── inference.py               # CLI inference script
│
├── assets/                        # Model Artifacts & Tasks
│   ├── checkpoints/               # Trained PyTorch weights (.pth)
│   └── face_landmarker.task       # MediaPipe FaceLandmarker asset
│
├── data/                          # Datasets & Feature Files (Git ignored)
│   ├── Driver Drowsiness Dataset (DDD)/
│   ├── geometric_features.csv
│   ├── splits.json
│   └── unseen_test_images/       # 5,974 physical copies of held-out test images
│
├── docs/                          # Documentation & Reports
│   ├── MDS505.pdf                 # Project report
│   ├── drowsiness_feature_fusion_plan.md
│   ├── generate_report.py         # Summary report & plot generator
│   └── task/
│       ├── task_1.md              # Environment & architecture plan
│       └── task_2.md              # This document
│
├── results/                       # Evaluation plots & training histories
├── requirements.txt               # Main Python dependencies
├── .gitignore                     # Git exclusion config
└── README.md                      # Comprehensive project documentation
```

---

## 3. Code & Infrastructure Fixes Implemented

| Issue Identified | Root Cause | Fix Implemented |
|---|---|---|
| **Path Resolution** | Relative paths broke when moving `config.py` into `ml_pipeline/`. | Calculated `PROJECT_ROOT = os.path.dirname(ML_PIPELINE_DIR)` and injected into `sys.path`. |
| **Stale `splits.json` Paths** | Absolute image paths in `splits.json` broke after moving dataset to `data/`. | Updated `load_splits()` in `data_split.py` to re-resolve paths dynamically against `config.DATASET_ROOT`. |
| **Windows DataLoader Hang** | `NUM_WORKERS = 4` causes silent process hangs on Windows when spawning workers. | Dynamic OS check in `config.py`: `NUM_WORKERS = 0 if Windows else 4`. |
| **PyTorch AMP Deprecations** | `GradScaler("cuda")` and `autocast("cuda")` deprecated in PyTorch 2.x. | Updated to `GradScaler(device="cuda")` and `autocast(device_type="cuda")`. |
| **Windows Console Encoding** | Emojis in print statements caused `UnicodeEncodeError (cp1252)`. | Replaced emoji print symbols with clean text formatting in CLI tools. |

---

## 4. Web Backend API (FastAPI) Setup

Built a FastAPI web backend capable of serving single-image uploads and live camera frame streams:

* **`GET /health`**: Returns API status, loaded model name, PyTorch device, and checkpoint status.
* **`POST /api/predict/image`**: Accepts image file upload (`multipart/form-data`) $\rightarrow$ returns label, confidence score, and extracted landmark metrics.
* **`POST /api/predict/frame`**: Accepts base64 encoded video frame string $\rightarrow$ returns real-time drowsiness prediction for web/webcam frontends.
* **`InferenceService`**: Singleton service wrapper managing PyTorch model loading and MediaPipe landmark extraction.

---

## 5. Model Training & Accuracy Results

Executed sequential training for all 3 model architectures using a 2-Stage schedule (Stage 1: 5 epochs frozen backbone, Stage 2: 15 epochs unfrozen fine-tuning with Early Stopping patience=5):

| Model Architecture | Inputs | Best Validation Accuracy | Status | Key Performance Takeaway |
|---|---|---|---|---|
| 🥇 **FusionModel** | Image (224×224) + 4 Landmarks | **90.23%** | Complete | **Best overall model.** Fusing CNN + Geometry yields **+2.11%** over baseline. |
| 🥈 **BaselineCNN** | Image only (EfficientNet-B0) | **88.12%** | Complete | Strong visual baseline, but misses kinematic fatigue signals. |
| 🥉 **GeometricOnlyMLP** | 4 Landmarks only (MLP) | **64.52%** | Complete | Ablation model — demonstrates landmark capability without image features. |

### Model Checkpoints Saved:
* `assets/checkpoints/fusion_best.pth` (17.7 MB)
* `assets/checkpoints/baseline_best.pth` (16.3 MB)
* `assets/checkpoints/geometric_best.pth` (9.7 KB)

---

## 6. Subject-Disjoint Data Splitting & Unseen Data Verification

### Splitting Strategy:
Used **Sequence Gap Analysis** ($\Delta \ge 3$) to group contiguous frames into **362 Drowsy** and **312 Non-Drowsy** subject identity blocks (674 total subject groups). Partitioned at the subject group level with `SEED = 42`.

### Dataset Distribution:
* **Train Set**: **29,901 images** (71.5%) — Used strictly for gradient training.
* **Validation Set**: **5,918 images** (14.2%) — Used for epoch evaluation & early stopping.
* **Test Set (Unseen)**: **5,974 images** (14.3%) — **100% held-out unseen subjects**.

### Empirical Disjointness Verification:
Ran automated set-intersection check to confirm data isolation:
```text
Train vs. Test Filename Overlap:          0 images (0.00%)
Train vs. Validation Overlap:            0 images (0.00%)
Validation vs. Test Overlap:             0 images (0.00%)
Any Exported Test Image in Train Set?:   False (0 files)
```

### Unseen Data Export Scripts Created:
1. **`ml_pipeline/extract_filenames.py`**: Indexed all 41,793 images into `data/all_filenames.csv` and `data/all_filenames.json`.
2. **`ml_pipeline/export_unseen_data.py`**: Exported all **5,974 unseen test images** into `data/unseen_test_images/` (3,089 Drowsy + 2,885 Non-Drowsy) with CSV/JSON index files.

---

## 7. Version Control & GitHub Repository

* **`.gitignore` Configured**: Excluded 2.6 GB raw dataset (`Driver Drowsiness Dataset (DDD)`), binary checkpoints (`*.pth`), `data/unseen_test_images/`, generated CSV/JSON indices, and temporary files.
* **Git Commit & Remote Push**:
  * Initialized Git repository.
  * Committed 26 clean source files (3,543 insertions).
  * Connected remote: `https://github.com/Josh-Petrr/DDD.git`.
  * Successfully pushed to `main` branch.

---

## 8. Summary of Created/Modified Files

| File Path | Description |
|---|---|
| [`backend/main.py`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/backend/main.py) | FastAPI app entrypoint with endpoints `/health`, `/api/predict/image`, `/api/predict/frame` |
| [`backend/schemas.py`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/backend/schemas.py) | Pydantic data models for request/response validation |
| [`backend/services/inference_service.py`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/backend/services/inference_service.py) | Model weights & MediaPipe landmarker loader for API predictions |
| [`ml_pipeline/config.py`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/ml_pipeline/config.py) | Path resolution, hyperparameters, OS-specific `NUM_WORKERS` |
| [`ml_pipeline/data_split.py`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/ml_pipeline/data_split.py) | Subject-disjoint split logic & path re-resolution in `load_splits()` |
| [`ml_pipeline/extract_filenames.py`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/ml_pipeline/extract_filenames.py) | Utility script to extract and index all 41,793 dataset filenames |
| [`ml_pipeline/export_unseen_data.py`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/ml_pipeline/export_unseen_data.py) | Exporter script that copies all 5,974 held-out test images |
| [`.gitignore`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/.gitignore) | Repository exclusion rules for large datasets, weights, and caches |
| [`README.md`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/README.md) | Comprehensive GitHub repository documentation |
| [`docs/task/task_2.md`](file:///c:/Users/Josh/OneDrive/Desktop/MSDS/4%20trisemster/nndl/nndl%20project/docs/task/task_2.md) | This document |
