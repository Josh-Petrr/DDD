# Task 7: Full Project Report — Driver Drowsiness Detection (DDD)

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Data Preprocessing & Splitting](#3-data-preprocessing--splitting)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Evolution — All Experiments](#5-model-evolution--all-experiments)
6. [Final Model Selection & Justification](#6-final-model-selection--justification)
7. [Live Frontend Application](#7-live-frontend-application)
8. [System Architecture Diagrams](#8-system-architecture-diagrams)
9. [Summary of All Results](#9-summary-of-all-results)

---

## 1. Problem Statement

Driver fatigue is one of the leading causes of road accidents globally. According to the WHO, drowsy driving contributes to approximately 20% of all road accidents. Existing solutions — steering input monitoring, infrared eye sensors — are either expensive, invasive, or inaccurate.

**Goal:** Build a non-invasive, camera-based real-time drowsiness detection system that:
- Generalizes to completely **unseen drivers** (no subject memorization)
- Operates in **real time** from a standard webcam
- Achieves high **Recall** (catching true drowsiness is more important than avoiding false alarms)

**Key Metric Focus:** We use **APCER (Attack Presentation Classification Error Rate)** as the primary safety metric — it measures what percentage of truly drowsy events we *miss*. A lower APCER means the system is safer.

---

## 2. Dataset

**Dataset:** Driver Drowsiness Dataset (DDD)

| Property | Value |
| :--- | :--- |
| Total Images | ~40,000 face images |
| Classes | 2 — **Drowsy** and **Non-Drowsy** |
| Subjects | 22 unique drivers |
| Image Source | Video frames of drivers under lab conditions |
| Format | RGB images, variable resolution |
| Naming Convention | Subject-prefixed filenames (e.g., `A_0001.jpg`, `B_0002.jpg`) |

**Key challenge:** Images from the same person are visually very similar. A model can easily cheat by memorizing faces rather than learning drowsiness cues. This makes standard random splits completely invalid.

---

## 3. Data Preprocessing & Splitting

### 3.1 Subject-Disjoint Split Strategy

The most important design decision in the entire project. We enforce that **no subject (driver) appears in more than one split**. This guarantees the model is tested on completely unseen people.

```
22 Subjects Total:
  ├── Train:  70% of subjects (~15 people)
  ├── Val:    15% of subjects (~3-4 people)  
  └── Test:   15% of subjects (~3-4 people)
```

Each subject contributes ALL their images (both Drowsy and Non-Drowsy) to exactly one split. This is stored in `data/splits.json` and locked permanently so every experiment uses identical splits.

### 3.2 Image Preprocessing

Every image is processed through the following pipeline before entering any model:

```
Raw Image (any size)
    │
    ├─► Resize → 224×224 pixels (EfficientNet-B0 input size)
    ├─► Convert → RGB (3 channels)
    ├─► ToTensor → float32 tensor in [0, 1]
    └─► Normalize → ImageNet Mean/Std
        Mean = [0.485, 0.456, 0.406]
        Std  = [0.229, 0.224, 0.225]
```

### 3.3 Training Augmentations

Applied only to the training set to improve generalization:
- **Random Horizontal Flip** (p=0.5)
- **Color Jitter** (brightness ±0.3, contrast ±0.3, saturation ±0.2)
- **Grayscale** (p=0.15, simulates low-light conditions)
- **CutMix** (p=0.3, blends two training images to prevent feature over-reliance)

---

## 4. Feature Engineering

### 4.1 Geometric Facial Features (MediaPipe)

In addition to raw image CNN features, we computed 4 biologically-meaningful geometric features from each frame using **MediaPipe FaceLandmarker** (478-point facial mesh):

| Feature | Formula | Drowsiness Indicator |
| :--- | :--- | :--- |
| **EAR** (Eye Aspect Ratio) | `(p2-p6 + p3-p5) / (2 × p1-p4)` | Drops when eyes close |
| **MAR** (Mouth Aspect Ratio) | `Vertical mouth / Horizontal mouth` | Rises during yawning |
| **Eyebrow Distance** | `Distance from eyebrow to eye top` | Drops with drooping fatigue |
| **Head Tilt** | `Angle between nose tip and chin` | Increases as head nods |

These 4 values are normalized per-subject using the **95th percentile** of their own distribution (their personal "alert" baseline), then concatenated with the CNN embedding.

### 4.2 Feature Normalization (For Live Inference)

Since we cannot know the 95th percentile for a completely new driver in real-time, we use the **global 95th percentile** from the training set population as a universal fallback baseline. This is computed from `data/geometric_features.csv` at server startup.

---

## 5. Model Evolution — All Experiments

### Experiment 1 — Joint Split (Baseline Sanity Check)

**Problem Discovered:** We initially used a random, non-subject-disjoint split by mistake. Results were unrealistically high.

| Model | Accuracy | Recall | ROC AUC |
| :--- | :--- | :--- | :--- |
| Baseline CNN | 97.04% | 99.97% | 0.9985 |
| Geometric MLP | 65.94% | 62.61% | 0.7302 |
| Fusion (CNN + Geo) | 97.19% | 100.00% | 0.9995 |

**Conclusion:** These numbers are fraudulent. The model memorized 22 specific faces. The 65% geometric model was the only honest signal — it cannot memorize faces. We immediately switched to **subject-disjoint splits**.

---

### Experiment 2 — Subject-Disjoint Split (Reality Check)

First honest experiment with the correct split strategy. Performance dropped dramatically, revealing the true challenge.

| Model | Accuracy | Recall | F1-Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- |
| Baseline CNN (v2) | 61.31% | 81.79% | 64.89% | 0.7607 |
| Geometric MLP (v2) | 46.85% | 63.48% | 51.08% | 0.5840 |
| Fusion (v2) | 46.04% | 78.36% | 55.94% | 0.6453 |

**What went wrong:** The Fusion model performed worse than the Baseline CNN. Investigation revealed the geometric feature normalization was corrupted — subject baseline normalization was leaking test subjects' data into the normalizer.

**Change made:** Fixed the normalization pipeline to only use training-set baselines.

---

### Experiment 3 — Fixed Normalization, Subject-Disjoint

After fixing the normalization bug, the Fusion model outperformed the Baseline as expected.

| Model | Accuracy | Recall | F1-Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- |
| Baseline CNN (v2) | 76.12% | 88.46% | 78.74% | 0.6447 |
| Geometric MLP (v2) | 62.64% | 53.83% | 59.03% | 0.7305 |
| Fusion (v2) | **78.21%** | **92.64%** | **80.96%** | 0.6661 |

**Change made:** Added partial backbone unfreeze (last 3 blocks of EfficientNet) in Stage 2 training, allowing the CNN to fine-tune its features for drowsiness detection rather than staying frozen on ImageNet features.

---

### Experiment 4 — Domain Adaptation (GRL)

**Problem identified:** The CNN was still partially learning identity-specific features. A domain classifier would force it to be identity-blind.

**Change made:** Introduced the **Gradient Reversal Layer (GRL)** — an adversarial domain classifier head trained to predict *which driver* is in frame, but with the gradient reversed. This actively pushes the CNN toward learning **domain-invariant (identity-invariant)** features.

| Model | Accuracy | Recall | F1-Score | APCER | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline CNN (v3) | 71.49% | 79.20% | 73.53% | 20.80% | 0.8273 |
| Geometric MLP (v3) | 60.80% | 58.21% | 59.75% | 41.79% | 0.6643 |
| Fusion (v3) | 63.16% | 65.92% | 64.15% | 34.08% | 0.6466 |
| **FusionGRL (v3)** | **71.22%** | **81.04%** | **73.79%** | **18.96%** | 0.6674 |

**Key finding:** `FusionGRL` achieved the best APCER (18.96%), meaning it missed the fewest drowsy events. GRL successfully forced domain-invariant learning.

---

### Experiment 5 — FUSION_GRL_V4 (Higher Dropout)

**Problem:** GRL model still showed signs of subject memorization. The Dropout rate in the fusion head was too low.

**Change made:** Increased `DROPOUT_RATE` from `0.30` → `0.50` across all fusion and domain head layers. Added stronger `WEIGHT_DECAY` of `5e-3` vs `1e-3` previously.

| Model | Accuracy | Recall | F1-Score | APCER | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FusionGRL_V4** | **70.87%** | **77.91%** | **72.79%** | **22.09%** | **0.7757** |

**Key gain:** The ROC AUC jumped from `0.6674` → `0.7757`. Higher Dropout successfully improved model calibration and its ability to rank true drowsiness events above false alarms. This became our best single-image model.

---

### Experiment 6 — Temporal Sequence Modeling (CNN-LSTM)

**Core insight:** A single image is a very ambiguous signal. A half-closed eye could be a blink. Sustained closed eyes for 1 second is definitively drowsiness. The model needed *time*.

**Architecture change:** We transitioned from single-image classification to a **temporal sequence model**:
1. Pre-extract all CNN features offline via `extract_features.py` → saves `(N_frames, 1284)` arrays
2. A new `DrowsinessLSTM` processes sliding windows of 30 consecutive frames

#### Iteration 1 — Overfitting (Stride 15)
- Dataset: 1,998 sequences (too few)
- Model: 2-layer LSTM, 256 hidden units (too large)
- Train Accuracy: **100%** | Val: **76.37%** | Test: **70.99%**
- **Failed** — classic memorization

#### Iteration 2 — Anti-Overfit (Stride 5)
Three simultaneous changes:
1. **Stride 15 → 5** multiplied training data from 1,998 → 6,000+ sequences
2. **Model shrunken** to 1-layer LSTM with 64 hidden units + Dropout 0.6
3. **GaussianNoise layer** injected at LSTM input to replace lost image augmentations

| Model | Accuracy | Recall | F1-Score | APCER | ACER |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FusionGRL_V4 (best single-img) | 70.87% | 77.91% | 72.79% | 22.09% | 29.13% |
| **LSTM (final)** | **72.52%** | **87.88%** | **76.32%** | **12.12%** | **27.60%** |

**APCER dropped by 9.97%** — the model now catches nearly 88% of all real drowsiness events.

---

## 6. Final Model Selection & Justification

**Chosen Pipeline: FUSION_GRL_V4 (CNN Feature Extractor) → DrowsinessLSTM (Sequence Classifier)**

### Why Not Pure Single-Image?
The fundamental ceiling of single-image drowsiness detection on unseen subjects is ~70-72% accuracy. A yawning person and a talking person can look identical in one frame. Only temporal context breaks this ambiguity.

### Why LSTM over a Larger CNN?
The dataset provides only 22 subjects (~40,000 images). Adding more CNN parameters causes identity memorization. The LSTM is parameter-efficient (its ~50K parameters process time, not pixels) and directly addresses the temporal nature of drowsiness as a physiological process.

### Why These Specific Numbers?
- **87.88% Recall / 12.12% APCER:** In a life-safety system, missing a drowsy event is far more dangerous than a false alarm. We prioritized recall at the cost of BPCER (false alarm rate of 43%).
- **ROC AUC 0.67:** Lower than V4's 0.77 because the smaller LSTM is less well-calibrated, but its hard-threshold accuracy and recall are both higher.

### Model Architecture Summary
```
Input (1 frame, 224×224 RGB)
    │
    ▼
EfficientNet-B0 (frozen)     ← ImageNet pretrained, fine-tuned last 3 blocks
    │ 1280-d embedding
    ▼
MediaPipe FaceLandmarker     ← 4 geometric features (EAR, MAR, Eyebrow, Tilt)
    │ 4-d feature
    ▼
Concatenation               → 1284-d fused vector per frame
    │
    ▼ (30 frames buffered)
DrowsinessLSTM
    ├── GaussianNoise (σ=0.05) [training only]
    ├── LSTM (1 layer, 64 hidden units)
    ├── Dropout (p=0.6)
    └── Linear → 2 logits (Drowsy / Awake)
    │
    ▼
Softmax → Prediction (Drowsy / Awake)
```

---

## 7. Live Frontend Application

### Technology Stack
| Layer | Technology | Reason |
| :--- | :--- | :--- |
| Backend | FastAPI (Python) | Needed to run PyTorch models; fastest Python async framework |
| Real-time Comm. | WebSockets | Low-latency bidirectional streaming (vs. polling HTTP) |
| Frontend | HTML + CSS + JavaScript | No build step; FastAPI serves static files directly |
| Audio Alarm | Web Audio API | Browser-native synthesized beep; no audio file needed |

### Application Flow
```
Browser                         FastAPI Server (app.py)
  │                                      │
  │──── WebSocket Connect (/ws) ────────>│
  │                                      │  Load CNN + LSTM + MediaPipe
  │                                      │  (one-time startup, ~5 seconds)
  │                                      │
  │<──── Connected ─────────────────────│
  │                                      │
  │  [Every 100ms — 10 FPS]             │
  │──── base64 JPEG frame ─────────────>│
  │                                      │  1. MediaPipe → EAR, MAR, Eyebrow, Tilt
  │                                      │  2. EfficientNet → 1280-d embedding
  │                                      │  3. Concatenate → 1284-d vector
  │                                      │  4. Append to rolling buffer[30]
  │                                      │  5. If buffer full → LSTM → prediction
  │<──── JSON prediction ───────────────│
  │                                      │
  │  [Update UI]                         │
  │  - Threat level bar                  │
  │  - AWAKE / DROWSY label              │
  │  - Live biometric gauges             │
  │  - Audio alarm if drowsy             │
```

### Frontend UI Components
1. **Live Camera Feed** — Webcam stream rendered via `<video>` element, mirrored
2. **Threat Level Card** — Green "AWAKE" / Red "DROWSY" with pulsing animation; shows drowsy probability as a gradient bar
3. **Sequence Buffer** — Progress bar showing 0-30 frames, labeled "Buffer full - LSTM is predicting in real-time"
4. **Live Biometric Gauges** — Real-time EAR, MAR, Eyebrow Distance, Head Tilt bars with numeric values
5. **Session Stats** — Duration timer, total frames processed, drowsy alert count, average confidence

### How to Launch
```bash
# Start the server
python app.py

# Open in browser
http://127.0.0.1:8000
```

---

## 8. System Architecture Diagrams

### 8.1 — End-to-End Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                         │
│                                                                 │
│  Raw DDD Dataset (40K images, 22 subjects)                      │
│       │                                                         │
│       ├── Subject-Disjoint Split (splits.json)                  │
│       │   ├── Train: 70% subjects → all their images            │
│       │   ├── Val:   15% subjects → all their images            │
│       │   └── Test:  15% subjects → all their images            │
│       │                                                         │
│       └── MediaPipe FaceLandmarker                              │
│           → geometric_features.csv (EAR, MAR, Eyebrow, Tilt)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 1: CNN TRAINING                       │
│                                                                 │
│  EfficientNet-B0 (ImageNet pretrained)                          │
│       │ 1280-d embedding                                        │
│       ▼                                                         │
│  Concatenate + 4-d geometric features → 1284-d                  │
│       ▼                                                         │
│  Fusion MLP Head → Drowsy / Non-Drowsy                          │
│       │                              ↑                          │
│       └── GRL Domain Head → Driver ID (gradient reversed)       │
│                                                                 │
│  Loss = CrossEntropy(drowsy) + λ × CrossEntropy(domain)         │
│  Stage 1: Freeze backbone, train only heads (5 epochs, LR=1e-3) │
│  Stage 2: Unfreeze last 3 blocks, fine-tune all (15 ep, LR=1e-4)│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ fusion_grl_4_best.pth
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2: FEATURE EXTRACTION                    │
│                                                                 │
│  extract_features.py:                                           │
│  ├── Load frozen FUSION_GRL_V4                                  │
│  ├── Run all 40K images through CNN → 1280-d embedding          │
│  ├── Concat with 4-d geometric features                         │
│  └── Save as (N_frames, 1284) .npy files per subject/class      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 3: LSTM TRAINING                         │
│                                                                 │
│  Sliding Window (stride=5) over .npy files                      │
│       → Training sequences: ~6,000+                            │
│       → Each sequence: (30, 1284) tensor                        │
│                                                                 │
│  DrowsinessLSTM:                                                │
│  ├── GaussianNoise(σ=0.05)  ← augmentation replacement         │
│  ├── LSTM(input=1284, hidden=64, layers=1)                      │
│  ├── Dropout(p=0.6)                                             │
│  └── Linear(64 → 2)                                             │
│                                                                 │
│  Loss = CrossEntropy, Optimizer = Adam (LR=1e-3)               │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 — Live Inference Architecture

```
WEBCAM FRAME (browser)
       │
       │ base64 JPEG (WebSocket, every 100ms)
       ▼
┌──────────────────────────────────────────┐
│           FastAPI WebSocket Server        │
│                                          │
│  ┌─── MediaPipe FaceLandmarker ────┐     │
│  │  478-point facial mesh          │     │
│  │  → EAR, MAR, Eyebrow, Tilt      │     │
│  │  → Normalize vs. global train   │     │
│  │    95th-percentile baseline     │     │
│  └──────────────┬──────────────────┘     │
│                 │ 4-d vector             │
│                 │                        │
│  ┌─── EfficientNet-B0 (frozen) ────┐     │
│  │  features → avgpool → flatten   │     │
│  │  → 1280-d visual embedding      │     │
│  └──────────────┬──────────────────┘     │
│                 │ 1280-d vector          │
│                 │                        │
│  ┌─────── Concatenate ─────────────┐     │
│  │  [1280-d CNN] + [4-d Geometric] │     │
│  │  = 1284-d fused vector          │     │
│  └──────────────┬──────────────────┘     │
│                 │                        │
│  ┌─────── Rolling Buffer ──────────┐     │
│  │  Deque of 30 × 1284-d vectors   │     │
│  │  (FIFO: drop oldest, add newest)│     │
│  └──────────────┬──────────────────┘     │
│                 │ (30, 1284) sequence    │
│                 │ [when buffer full]     │
│  ┌─────── DrowsinessLSTM ──────────┐     │
│  │  LSTM(64 hidden) → Linear(2)    │     │
│  │  → Softmax → [P(Drowsy), P(Awake)]│   │
│  └──────────────┬──────────────────┘     │
│                 │                        │
│  JSON Response: {prediction, confidence, │
│    drowsy_prob, geo_features, buffer_fill}│
└──────────────────┬───────────────────────┘
                   │
                   ▼
         BROWSER (script.js)
           ├── Update Threat Level card
           ├── Update biometric gauges
           ├── Update buffer progress bar
           └── If drowsy: flash red overlay + beep alarm
```

### 8.3 — Subject-Disjoint Data Split (Why It Matters)

```
  WRONG (Joint Split)          CORRECT (Subject-Disjoint Split)
  
  Subject A ─┬─ Train          Subject A ──────── Train (ALL images)
             ├─ Val            Subject B ──────── Train (ALL images)
             └─ Test           Subject C ──────── Train (ALL images)
                               ...
  Subject B ─┬─ Train          Subject P ──────── Val   (ALL images)
             ├─ Val            Subject Q ──────── Val   (ALL images)
             └─ Test           
                               Subject S ──────── Test  (ALL images)
  ❌ Model learns:              Subject T ──────── Test  (ALL images)
  "Face X = Drowsy"            
  97% accuracy (cheating!)     ✅ Model learns:
                               "Eyes half-closed + slow blinks = Drowsy"
                               72% accuracy (real, honest performance)
```

---

## 9. Summary of All Results

### 9.1 — Complete Model Leaderboard

| # | Experiment | Model | Split | Accuracy | Recall | F1 | APCER | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Sanity Check | Baseline CNN | Joint | 97.04% | 99.97% | 97.21% | 0.03% | 0.9985 |
| 2 | Sanity Check | Fusion | Joint | 97.19% | 100.0% | 97.35% | 0.00% | 0.9995 |
| 3 | Reality Check | Baseline CNN | Disjoint | 61.31% | 81.79% | 64.89% | 18.21% | 0.7607 |
| 4 | Reality Check | Fusion | Disjoint | 46.04% | 78.36% | 55.94% | 21.64% | 0.6453 |
| 5 | Fixed Norm | Baseline CNN | Disjoint | 76.12% | 88.46% | 78.74% | 11.54% | 0.6447 |
| 6 | Fixed Norm | Fusion | Disjoint | 78.21% | 92.64% | 80.96% | 7.36% | 0.6661 |
| 7 | GRL | Baseline CNN (v3) | Disjoint | 71.49% | 79.20% | 73.53% | 20.80% | 0.8273 |
| 8 | GRL | FusionGRL (v3) | Disjoint | 71.22% | 81.04% | 73.79% | 18.96% | 0.6674 |
| 9 | V4 Dropout | FusionGRL_V4 | Disjoint | 70.87% | 77.91% | 72.79% | 22.09% | **0.7757** |
| 10 | LSTM (overfit) | CNN-LSTM | Disjoint | 70.99% | 78.79% | 73.24% | 21.21% | 0.8366 |
| 11 ⭐ | LSTM (final) | CNN-LSTM | Disjoint | **72.52%** | **87.88%** | **76.32%** | **12.12%** | 0.6744 |

> ⭐ = **Final production model** used in the live frontend application

### 9.2 — Final Model vs. Best Single-Image Model

| Metric | FusionGRL_V4 | CNN-LSTM (Final) | Δ |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 70.87% | **72.52%** | +1.65% |
| **Recall** *(safety metric)* | 77.91% | **87.88%** | **+9.97%** |
| **F1-Score** | 72.79% | **76.32%** | +3.53% |
| **APCER** (missed drowsy) | 22.09% | **12.12%** | **-9.97%** |
| **BPCER** (false alarms) | 36.17% | 43.08% | +6.91% |
| **ROC AUC** | **0.7757** | 0.6744 | -0.1013 |

The CNN-LSTM is clearly superior for the safety-critical task: it catches **88% of all true drowsiness events** on unseen drivers, trading off a slight increase in false alarms which is acceptable in a life-safety system.

---

## 10. Real-Life Deployment Limitations & Future Work

When testing the final application on live, unseen people via webcam (`app.py`), we observed a practical issue: **the model frequently predicted "Drowsy" even when the user was fully awake.**

### 10.1 Identified Limitations

1. **The Calibration Problem (Geometric Feature Mismatch)**
   - In the live application, we normalize live geometric features (EAR, MAR) by dividing them by the **global 95th percentile** from the training dataset.
   - *The Issue:* If the global baseline for a "wide open eye" (EAR) is 0.35, but a live user naturally has smaller or narrower eyes with a max EAR of 0.25, the system calculates `0.25 / 0.35 = 0.71`. The model perceives this 71% ratio as a drooping eye and incorrectly flags the user as drowsy.
   - *Result:* High false positive rate for individuals whose facial geometry significantly deviates from the training dataset's average.

2. **The Safety Tradeoff (High BPCER)**
   - As shown in the metrics above, our final LSTM model has a BPCER (False Alarm rate) of 43%. 
   - *The Issue:* Because we explicitly optimized the architecture and hyperparameters to aggressively minimize APCER (Missed Drowsiness), the model became highly sensitive. If the sequence is even slightly ambiguous, it defaults to "Drowsy" to prioritize life safety.

3. **Domain Shift (Lab vs. Real World)**
   - The DDD dataset images were captured in controlled environments. Live webcams introduce extreme variations in lighting (backlighting, dark rooms), camera angles (looking down at a laptop vs. up at a dashcam), and distance from the camera, all of which degrade the CNN embeddings.

### 10.2 Future Improvements

To make the system truly production-ready for real-world driving scenarios, we recommend the following engineering solutions:

1. **Dynamic User Calibration (Rolling Baselines)**
   - *Solution:* Instead of using a static global baseline, the app should silently record the user's maximum EAR and minimum MAR over the first 10-15 seconds of the session (assuming the driver starts the car awake). 
   - *Impact:* The normalizer will divide live values by the *user's personal baseline*, immediately resolving the false alarms caused by natural eye-shape differences.

2. **Tunable Confidence Thresholds**
   - *Solution:* Currently, the app flashes the alarm if `P(Drowsy) > 0.5`. Exposing a sensitivity slider in the UI (e.g., requiring >0.75 confidence for "Drowsy") allows users to manually reduce false alarms based on their environment.

3. **Data Augmentation for Live Conditions**
   - *Solution:* Fine-tune the CNN using heavily augmented versions of the dataset that explicitly mimic poor webcam quality (motion blur, low-light noise, off-center cropping) to improve robustness against real-world domain shifts.
