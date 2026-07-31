# Driver Drowsiness Detection — Complete Model Training & Evaluation Report

This report documents the full training progression (both Stage 1 and Stage 2 across training and validation) and final test evaluation results for all 4 trained model variants:
1. **Baseline Model (`baseline_3`)** — EfficientNet-B0 backbone
2. **Geometric Model (`geometric_3`)** — MLP trained on 4 extracted facial landmark features
3. **Fusion Model (`fusion_3`)** — Multimodal fusion of EfficientNet-B0 embeddings + geometric features
4. **Fusion GRL Model (`fusion_grl_3`)** — Multimodal fusion with Gradient Reversal Layer for subject-domain adaptation

---

## Executive Summary & Comparison

| Model Name | Best Val Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1-Score | APCER | BPCER | ACER | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`baseline_3`)** | 60.42% | 71.47% | 68.61% | 79.15% | 73.50% | 0.2085 | 0.3622 | 0.2853 | 0.6783 |
| **Geometric (`geometric_3`)** | 48.60% | 59.93% | 62.91% | 48.36% | 54.68% | 0.5164 | 0.2851 | 0.4007 | 0.6601 |
| **Fusion (`fusion_3`)** | 56.61% | 74.18% | 68.04% | **91.19%** | **77.93%** | **0.0881** | 0.4284 | 0.2582 | **0.7935** |
| **Fusion GRL (`fusion_grl_3`)** | 47.41% | **74.23%** | **69.54%** | 86.22% | 76.99% | 0.1378 | **0.3776** | **0.2577** | 0.7661 |

---

## 1. Baseline Model (`baseline_3`)

### Model Overview
- **Architecture**: Pretrained EfficientNet-B0 + Classification Head
- **Total Parameters**: 4,010,110 (Trainable: 4,010,110)

### Training & Validation Progression

| Epoch | Stage | Training Loss | Training Acc (%) | Validation Loss | Validation Acc (%) | Learning Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Stage 1 (Head Only) | 0.4324 | 85.26% | 1.0813 | 24.99% | 9.05e-04 |
| 2 | Stage 1 (Head Only) | 0.3797 | 89.07% | 1.2373 | 23.43% | 6.55e-04 |
| 3 | Stage 1 (Head Only) | 0.3711 | 89.67% | 1.2482 | 23.36% | 3.45e-04 |
| 4 | Stage 1 (Head Only) | 0.3696 | 89.83% | 1.2185 | 22.45% | 9.55e-05 |
| 5 | Stage 1 (Head Only) | 0.3695 | 89.81% | 1.2783 | 22.76% | 0.00e+00 |
| 6 | Stage 2 (Fine-tuning) | 0.2436 | 98.82% | 1.0401 | 33.91% | 9.89e-05 |
| 7 | Stage 2 (Fine-tuning) | 0.2131 | 99.95% | 0.9340 | 42.86% | 9.57e-05 |
| 8 | Stage 2 (Fine-tuning) | 0.2077 | 99.99% | 0.9063 | 42.23% | 9.05e-05 |
| 9 | Stage 2 (Fine-tuning) | 0.2049 | 99.99% | 0.8254 | 46.21% | 8.35e-05 |
| 10 | Stage 2 (Fine-tuning) | 0.2034 | 100.00% | 0.8473 | 51.27% | 7.50e-05 |
| 11 | Stage 2 (Fine-tuning) | 0.2024 | 100.00% | 0.8723 | 47.70% | 6.55e-05 |
| 12 | Stage 2 (Fine-tuning) | 0.2019 | 100.00% | 0.7865 | 57.83% | 5.52e-05 |
| 13 | Stage 2 (Fine-tuning) | 0.2015 | 100.00% | 0.8237 | 52.74% | 4.48e-05 |
| 14 | Stage 2 (Fine-tuning) | **0.2011** | **100.00%** | **0.7742** | **60.42%** | 3.45e-05 |
| 15 | Stage 2 (Fine-tuning) | 0.2008 | 100.00% | 0.7693 | 59.51% | 2.50e-05 |
| 16 | Stage 2 (Fine-tuning) | 0.2005 | 100.00% | 0.8058 | 56.51% | 1.65e-05 |
| 17 | Stage 2 (Fine-tuning) | 0.2003 | 100.00% | 0.8198 | 56.78% | 9.55e-06 |
| 18 | Stage 2 (Fine-tuning) | 0.2002 | 100.00% | 0.9126 | 52.14% | 4.32e-06 |
| 19 | Stage 2 (Fine-tuning) | 0.2001 | 100.00% | 0.8084 | 59.19% | 1.09e-06 |

*Early stopping triggered at Epoch 19 (Best Validation Accuracy: 60.42% at Epoch 14).*

### Test Set Performance Evaluation
- **Accuracy**: 71.47%
- **Precision**: 68.61%
- **Recall**: 79.15%
- **F1-Score**: 73.50%
- **ROC-AUC**: 0.6783 | **PR-AUC**: 0.5695
- **Error Rates**: APCER = 20.85% | BPCER = 36.22% | ACER = 28.53%
- **Confusion Matrix**:
  - True Drowsy: 1591 | False Non-Drowsy (FN): 419
  - False Drowsy (FP): 728 | True Non-Drowsy: 1282

---

## 2. Geometric Model (`geometric_3`)

### Model Overview
- **Architecture**: 2-Layer MLP on 4 extracted facial landmark features (EAR, MAR, Eyebrow Distance, Head Tilt)
- **Total Parameters**: 818 (Trainable: 818)

### Training & Validation Progression

| Epoch | Stage | Training Loss | Training Acc (%) | Validation Loss | Validation Acc (%) | Learning Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Stage 1 (Head Only) | **0.5349** | **77.03%** | **0.9070** | **48.60%** | 9.05e-04 |
| 2 | Stage 1 (Head Only) | 0.4897 | 81.50% | 0.9711 | 47.90% | 6.55e-04 |
| 3 | Stage 1 (Head Only) | 0.4810 | 82.20% | 0.9708 | 47.77% | 3.45e-04 |
| 4 | Stage 1 (Head Only) | 0.4792 | 82.33% | 0.9904 | 47.59% | 9.55e-05 |
| 5 | Stage 1 (Head Only) | 0.4752 | 82.63% | 1.0019 | 47.34% | 0.00e+00 |
| 6 | Stage 2 (Fine-tuning) | 0.4753 | 82.59% | 0.9633 | 47.59% | 9.89e-05 |
| 7 | Stage 2 (Fine-tuning) | 0.4754 | 82.65% | 0.9988 | 47.54% | 9.57e-05 |
| 8 | Stage 2 (Fine-tuning) | 0.4737 | 82.67% | 0.9875 | 47.44% | 9.05e-05 |
| 9 | Stage 2 (Fine-tuning) | 0.4752 | 82.87% | 0.9934 | 47.48% | 8.35e-05 |
| 10 | Stage 2 (Fine-tuning) | 0.4748 | 82.43% | 1.0015 | 47.12% | 7.50e-05 |

*Early stopping triggered at Epoch 10 (Best Validation Accuracy: 48.60% at Epoch 1).*

### Test Set Performance Evaluation
- **Accuracy**: 59.93%
- **Precision**: 62.91%
- **Recall**: 48.36%
- **F1-Score**: 54.68%
- **ROC-AUC**: 0.6601 | **PR-AUC**: 0.6475
- **Error Rates**: APCER = 51.64% | BPCER = 28.51% | ACER = 40.07%
- **Confusion Matrix**:
  - True Drowsy: 972 | False Non-Drowsy (FN): 1038
  - False Drowsy (FP): 573 | True Non-Drowsy: 1437

---

## 3. Fusion Model (`fusion_3`)

### Model Overview
- **Architecture**: Multimodal Fusion (EfficientNet-B0 1280-dim feature vector concatenated with 4-dim geometric vector)
- **Total Parameters**: 4,353,726 (Trainable: 4,353,726)

### Training & Validation Progression

| Epoch | Stage | Training Loss | Training Acc (%) | Validation Loss | Validation Acc (%) | Learning Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Stage 1 (Head Only) | 0.2936 | 95.17% | 0.9933 | 42.06% | 9.05e-04 |
| 2 | Stage 1 (Head Only) | 0.2447 | 98.29% | 0.8651 | 44.00% | 6.55e-04 |
| 3 | Stage 1 (Head Only) | 0.2346 | 98.84% | 0.8134 | 50.38% | 3.45e-04 |
| 4 | Stage 1 (Head Only) | 0.2298 | 98.99% | 0.8161 | 49.96% | 9.55e-05 |
| 5 | Stage 1 (Head Only) | 0.2272 | 99.10% | 0.8053 | 47.73% | 0.00e+00 |
| 6 | Stage 2 (Fine-tuning) | 0.2190 | 99.57% | 0.7298 | 54.40% | 9.89e-06 |
| 7 | Stage 2 (Fine-tuning) | **0.2133** | **99.82%** | **0.7486** | **56.61%** | 9.57e-06 |
| 8 | Stage 2 (Fine-tuning) | 0.2112 | 99.89% | 0.7056 | 56.23% | 9.05e-06 |
| 9 | Stage 2 (Fine-tuning) | 0.2095 | 99.92% | 0.7783 | 52.39% | 8.35e-06 |
| 10 | Stage 2 (Fine-tuning) | 0.2082 | 99.94% | 0.7646 | 51.01% | 7.50e-06 |
| 11 | Stage 2 (Fine-tuning) | 0.2078 | 99.93% | 0.7540 | 50.83% | 6.55e-06 |
| 12 | Stage 2 (Fine-tuning) | 0.2076 | 99.95% | 0.7427 | 53.26% | 5.52e-06 |

*Early stopping triggered at Epoch 12 (Best Validation Accuracy: 56.61% at Epoch 7).*

### Test Set Performance Evaluation
- **Accuracy**: 74.18%
- **Precision**: 68.04%
- **Recall**: **91.19%**
- **F1-Score**: **77.93%**
- **ROC-AUC**: **0.7935** | **PR-AUC**: 0.7746
- **Error Rates**: **APCER = 8.81%** | BPCER = 42.84% | ACER = 25.82%
- **Confusion Matrix**:
  - True Drowsy: 1833 | False Non-Drowsy (FN): 177
  - False Drowsy (FP): 861 | True Non-Drowsy: 1149

---

## 4. Fusion GRL Model (`fusion_grl_3`)

### Model Overview
- **Architecture**: Multimodal Fusion + Gradient Reversal Layer for subject domain adversarial classification
- **Total Parameters**: 4,701,204 (Trainable: 4,701,204)

### Training & Validation Progression

| Epoch | Stage | Training Loss | Training Acc (%) | Validation Loss | Validation Acc (%) | Learning Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Stage 1 (Head Only) | 0.2915 | 95.18% | 1.2654 | 21.71% | 9.05e-04 |
| 2 | Stage 1 (Head Only) | 0.2457 | 98.22% | 1.0611 | 33.56% | 6.55e-04 |
| 3 | Stage 1 (Head Only) | 0.2355 | 98.76% | 0.9729 | 40.83% | 3.45e-04 |
| 4 | Stage 1 (Head Only) | 0.2291 | 99.00% | 0.9847 | 38.31% | 9.55e-05 |
| 5 | Stage 1 (Head Only) | 0.2270 | 99.12% | 0.9918 | 39.13% | 0.00e+00 |
| 6 | Stage 2 (Fine-tuning) | 0.2188 | 99.57% | 0.8906 | 39.96% | 9.89e-06 |
| 7 | Stage 2 (Fine-tuning) | 0.2133 | 99.81% | 0.9500 | 45.60% | 9.57e-06 |
| 8 | Stage 2 (Fine-tuning) | 0.2113 | 99.86% | 0.8787 | 42.88% | 9.05e-06 |
| 9 | Stage 2 (Fine-tuning) | 0.2090 | 99.92% | 0.9061 | 43.29% | 8.35e-06 |
| 10 | Stage 2 (Fine-tuning) | **0.2084** | **99.93%** | **0.9256** | **47.41%** | 7.50e-06 |
| 11 | Stage 2 (Fine-tuning) | 0.2073 | 99.95% | 0.9311 | 44.67% | 6.55e-05 |
| 12 | Stage 2 (Fine-tuning) | 0.2069 | 99.94% | 0.9482 | 44.85% | 5.52e-06 |
| 13 | Stage 2 (Fine-tuning) | 0.2069 | 99.95% | 0.9348 | 43.75% | 4.48e-06 |
| 14 | Stage 2 (Fine-tuning) | 0.2061 | 99.96% | 0.9098 | 46.16% | 3.45e-06 |
| 15 | Stage 2 (Fine-tuning) | 0.2059 | 99.97% | 0.9790 | 46.12% | 2.50e-06 |

*Early stopping triggered at Epoch 15 (Best Validation Accuracy: 47.41% at Epoch 10).*

### Test Set Performance Evaluation
- **Accuracy**: **74.23%**
- **Precision**: **69.54%**
- **Recall**: 86.22%
- **F1-Score**: 76.99%
- **ROC-AUC**: 0.7661 | **PR-AUC**: 0.7282
- **Error Rates**: APCER = 13.78% | **BPCER = 37.76%** | **ACER = 25.77%**
- **Confusion Matrix**:
  - True Drowsy: 1733 | False Non-Drowsy (FN): 277
  - False Drowsy (FP): 759 | True Non-Drowsy: 1251

---

## Key Takeaways & Observations

1. **Multimodal Fusion Superiority**: Both `fusion_3` (74.18% accuracy, 77.93% F1) and `fusion_grl_3` (74.23% accuracy, 76.99% F1) significantly outperformed the visual-only `baseline_3` (71.47% accuracy) and geometric-only `geometric_3` (59.93% accuracy).
2. **High Safety Recall**: `fusion_3` achieved an exceptional **91.19% recall on drowsiness detection** with an **APCER of only 8.81%**, making it the safest model for minimizing undetected drowsiness instances.
3. **Domain Generalization with GRL**: `fusion_grl_3` achieved the highest overall test accuracy (74.23%) and lowest average classification error rate (ACER = 25.77%), demonstrating that adversarial domain adaptation helps balance error rates across subject splits.
