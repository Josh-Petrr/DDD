# Project Plan: Geometric Feature Fusion Module for Drowsiness Detection System

**System:** Driver Drowsiness Classifier (YOLOv11x-cls)
**Project:** Add a landmark-based geometric feature fusion layer to improve/extend classification
**Duration:** 10 working days
**Owner:** Darren

---

## 1. Project Goals

Build and integrate a new module into the existing drowsiness detection pipeline that:
1. Extracts facial geometry (eye closure ratio, mouth opening ratio, eyebrow position, head tilt) from each frame using a pretrained landmark model.
2. Fuses this with the existing CNN's learned features to produce a combined prediction.
3. Ships with a clear before/after performance comparison and a breakdown of exactly where the new module helps.

**SDG alignment:**
- **SDG 3 — Good Health & Well-Being:** contributes toward Target 3.6 (reducing road traffic deaths/injuries) by improving a driver-safety detection system.
- **SDG 11 — Sustainable Cities & Communities:** supports Target 11.2 (safe, accessible transport systems).

---

## 2. What's Being Built

### Existing system (already in place)
- YOLOv11x-cls binary classifier (Drowsy / Non-Drowsy) trained on DDD + Roboflow datasets.
- Grad-CAM explainability layer.
- Standard eval metrics: accuracy, APCER, BPCER, ACER.

### New module (this project)
**Geometric Feature Extractor**
- Landmark detection (MediaPipe FaceMesh) on each input frame.
- Computes 4 features per frame: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), eyebrow-to-eye distance, head-tilt angle.

**Fusion Head**
- Small MLP that takes [CNN embedding + geometric features] and outputs the final Drowsy/Non-Drowsy prediction.
- Sits on top of the existing trained YOLO backbone — no retraining of the base model required.

**Evaluation Dashboard**
- Side-by-side metrics: baseline model vs. fusion model.
- Breakdown by prediction-confidence tier, showing exactly where the new module adds value.

---

## 3. Build Plan — Step by Step

### Step 1: Landmark Extraction Module
- Add `mediapipe` to the project environment.
- Build `landmark_features.py`: takes an image in, returns EAR, MAR, eyebrow distance, head-tilt angle.
- Test on a sample batch (~50 images) across both classes to confirm sane outputs.

### Step 2: Batch Feature Pipeline
- Run the extractor across the full train/val/test sets using multiprocessing (same pattern already used in the existing pipeline for image verification).
- Save output as `features_train_val_test.csv` — one row per image, feature values + detection-success flag.
- Log and flag any frames where landmark detection fails (occlusion, extreme angle, etc.) rather than silently dropping them.

### Step 3: Confidence-Tagged Baseline Predictions
- Re-run the existing trained YOLO model on the test set.
- Save prediction + softmax confidence per image (not just the label) — needed for the tiered comparison in Step 6.

### Step 4: Embedding Export
- Extract the pooled feature embedding (pre-classification-head) from the trained YOLO backbone for every image — frozen, no gradient, reused as-is.

### Step 5: Fusion Head — Build & Train
- Concatenate YOLO embedding + 4 geometric features.
- Train a small 2-layer MLP (128 → 32 → 2) on this combined vector using the existing train/val split.
- Lightweight — should train in minutes, not hours.

### Step 6: Evaluation Build-Out
- Compute accuracy, APCER, BPCER, ACER for the fusion model vs. baseline — headline comparison table.
- Bucket predictions into confidence tiers (High / Medium / Low) and compare fusion vs. baseline performance within each tier — this is the key deliverable showing *where* the new module earns its keep.
- Package results into plots + a summary table.

### Step 7: Packaging & Documentation
- Write up module usage instructions (how to run the feature extractor + fusion head on new images).
- Document known limitations (landmark-detection failure cases, what conditions the module struggles with).
- Prepare a short results summary for presentation/demo purposes.

---

## 4. Tools & Software

| Tool | Purpose | Status |
|---|---|---|
| `mediapipe` | Facial landmark detection (EAR/MAR/eyebrow/tilt) | New — pip install, CPU-friendly |
| OpenCV | Image I/O, preprocessing | Already in use |
| NumPy / Pandas | Feature computation, data logging | Already in use |
| PyTorch | Fusion MLP training | Already in use (Ultralytics dependency) |
| Ultralytics (YOLO) | Backbone reuse, embedding extraction | Already in use |
| scikit-learn | Metrics, quick prototyping utilities | Already in use |
| Matplotlib / Seaborn | Result visualizations | Already in use |

**Only new install: `mediapipe`.** Everything else already exists in the current project environment.

---

## 5. 10-Day Build Schedule

| Day | Deliverable |
|---|---|
| 1 | Landmark extractor working on sample batch, outputs validated |
| 2 | Full-dataset feature pipeline run, `features_train_val_test.csv` produced |
| 3 | **Checkpoint:** review landmark-detection failure rate; adjust detector settings if failures exceed ~15% |
| 4 | Baseline model re-run with confidence scores saved |
| 5 | YOLO embeddings exported for train/val/test |
| 6 | Fusion MLP head built and trained |
| 7 | Fusion model evaluated on test set, metrics table produced |
| 8 | Confidence-tiered breakdown built, comparison plots generated |
| 9 | Buffer — bug fixes, edge case handling, limitation write-up |
| 10 | Final packaging: documentation, results summary, demo-ready output |

---

## 6. Feasibility

**On track for 10 days.** No retraining of the base YOLO model, no new heavy infrastructure — the fusion head is a small MLP on precomputed embeddings, and landmark extraction runs comfortably on CPU.

**Main risk:** landmark detection failing on non-frontal/occluded frames. Handled via the Day 3 checkpoint so it's caught early rather than discovered near the deadline.

**Fallback plan if fusion doesn't outperform the baseline overall:** the confidence-tiered breakdown still gives a usable result — if the module improves accuracy specifically on low-confidence frames, that's a real, shippable enhancement (e.g., use fusion as a "second opinion" only when the base model is uncertain, rather than replacing it outright). Either outcome produces a complete, working deliverable.

---

## 7. Project Deliverables

1. `landmark_features.py` — geometric feature extraction module
2. `features_train_val_test.csv` — extracted features dataset
3. `fusion_model.py` — fusion head training + inference script
4. `evaluation_report` (notebook/script + plots) — baseline vs. fusion comparison, confidence-tiered breakdown
5. `README` / usage docs for the new module
6. Final results summary for demo/presentation
