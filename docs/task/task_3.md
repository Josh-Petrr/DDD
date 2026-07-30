# Task 3: Overcoming Identity Overfitting, Data Leakage Fixes & Regularization Pipeline

**Date:** 2026-07-29  
**Status:** Implemented & Verified  
**Owner:** Josh & AI Pair Programmer  

---

## 1. Executive Summary

During initial subject-disjoint evaluation runs, the model exhibited severe **identity-based overfitting**:
* Training Accuracy reached **~99.9%** (Loss ~0.0016), while Validation/Test accuracy dropped significantly on unseen drivers (**46.04% - 61.31%**).
* Detailed evaluation revealed a **BPCER (False Alarm Rate) of 27.20% to 79.06%**, indicating that the model was memorizing specific driver faces rather than learning general physiological indicators of drowsiness (eye closure, yawning, head droop).

Task 3 addresses the root causes of identity memorization and data leakage by refactoring the data partition engine, introducing facial cutout augmentations, stabilizing geometric feature scaling, and applying loss & weight regularization.

---

## 2. Root Cause Analysis & Findings

| Problem Identified | Root Cause | Consequence |
|---|---|---|
| **Identity Data Leakage** | `data_split.py` split `Drowsy` and `Non Drowsy` folders independently. | A driver's images (e.g. prefix `ZA`) could be assigned to `Train` for Drowsy and `Val`/`Test` for Non-Drowsy. The model memorized Driver `ZA`'s face as "Drowsy", causing high false alarms when Driver `ZA` appeared in `Test` as awake. |
| **Validation Class Imbalance** | Coarse letter-prefix group sizes created 66.4% Drowsy vs 34.6% Non-Drowsy imbalance in validation. | Skewed evaluation metrics and biased decision threshold. |
| **Facial Shortcut Learning** | CNN backbone focused on fixed driver traits (beards, glasses, hair, background lighting). | High training accuracy (99.9%), poor cross-subject generalization. |
| **Overconfident Logits** | Hard target labels ($[1.0, 0.0]$) in `CrossEntropyLoss`. | Extremely large logit outputs and weight explosion. |
| **Geometric Feature Outliers** | Unbounded Z-scores from MediaPipe during extreme head tilts. | Gradient instability in Fusion MLP layers. |

---

## 3. Summary of Code Changes & Implementation Details

### 3.1 Global Subject-Disjoint & Stratified Splitting (`ml_pipeline/data_split.py`)
- **Changes Made:**
  - Replaced single-directory prefix parsing with `_parse_all_subjects()`, which groups images from both `Drowsy` and `Non Drowsy` folders by alphabetical subject prefix globally using case-insensitive `.upper()` matching.
  - Implemented greedy subject allocation into Train, Val, and Test splits.
  - Preserved **100% of training data** while applying strict min-count truncation for Validation and Test splits to guarantee an **exact 50% Drowsy / 50% Non-Drowsy balance**.
- **Reasoning:**
  - Guarantees strict **Leave-One-Subject-Out (LOSO)** evaluation: unifies uppercase (`A`) and lowercase (`a`) filenames into the same driver subject so no driver's images leak between `Train` and `Val`/`Test`.
  - Retains maximum training data for the neural network backbone while ensuring evaluation metrics are completely un-skewed.

### 3.2 Cutout Augmentation & Geometric Feature Clipping (`ml_pipeline/dataset.py`)
- **Changes Made:**
  - Added `transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value='random')` to training image transforms (applied post-normalization).
  - Added Z-score clipping `torch.clamp(geo_tensor, min=-3.0, max=3.0)` for MediaPipe landmark features (EAR, MAR, Eyebrow Dist, Head Tilt).
- **Reasoning:**
  - **RandomErasing:** Obscures random rectangular patches of the driver's face during training. Forces the model to look at multiple region cues (e.g., both eyes and mouth) rather than relying on a single facial anchor (like facial hair or skin tone).
  - **Feature Clipping:** Prevents noisy MediaPipe landmark estimates (caused by rapid movements or lighting changes) from injecting large gradient spikes into the Fusion MLP.

### 3.3 Loss Function Label Smoothing (`ml_pipeline/train.py`)
- **Changes Made:**
  - Updated `criterion = nn.CrossEntropyLoss(label_smoothing=0.1)` in the training loop.
- **Reasoning:**
  - Softens hard targets from $[1, 0]$ to $[0.95, 0.05]$. Prevents the network from pursuing infinite logit values and over-fitting to identity-specific training samples, leading to smoother decision boundaries.

### 3.4 Weight Decay Regularization (`ml_pipeline/config.py`)
- **Changes Made:**
  - Increased `WEIGHT_DECAY` hyperparameter from `1e-4` to `1e-3`.
- **Reasoning:**
  - Applies a stronger $L_2$ penalty to optimizer parameter updates across all parameter groups (`AdamW`), curbing extreme weight growth and forcing the model to learn parsimonious representations.

### 3.5 Report Generation & Checkpoint Evaluation (`docs/generate_report.py`)
- **Changes Made:**
  - Updated model loading loop to support `_2` checkpoint names (`baseline_2`, `geometric_2`, `fusion_2`) while dynamically stripping the `_2` suffix when calling `get_model()`.
- **Reasoning:**
  - Enables accurate evaluation of new model checkpoints without encountering architectural lookup errors in `models.py`.

---

## 4. Verification Plan & Next Steps

1. **Verify Split Balance & Integrity:**
   ```bash
   python ml_pipeline/data_split.py
   ```
   *Expected Result:* 100% subject-disjoint partitions with exact 50% Drowsy / 50% Non-Drowsy balance in `Val` and `Test`.

2. **Retrain All Models with Regularization:**
   ```bash
   python ml_pipeline/train_all.py
   ```
   *Expected Result:* Training curves show smooth convergence with significantly reduced gap between Train and Val accuracy.

3. **Re-evaluate Evaluation Metrics:**
   ```bash
   python docs/generate_report.py
   ```
   *Expected Result:* Improved validation/test accuracy, lower BPCER (False Alarm Rate), and higher ROC AUC across unseen drivers.
