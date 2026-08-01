"""
config.py — Central configuration for the Drowsiness Detection project.
All paths, hyperparameters, and constants in one place.
"""

import os
import sys
import torch

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ML_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ML_PIPELINE_DIR)

# Ensure PROJECT_ROOT and ML_PIPELINE_DIR are in sys.path for smooth imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ML_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, ML_PIPELINE_DIR)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "Driver Drowsiness Dataset (DDD)")
DROWSY_DIR = os.path.join(DATASET_ROOT, "Drowsy")
NON_DROWSY_DIR = os.path.join(DATASET_ROOT, "Non Drowsy")

SPLITS_FILE = os.path.join(PROJECT_ROOT, "data", "splits.json")
FEATURES_FILE = os.path.join(PROJECT_ROOT, "data", "geometric_features.csv")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "assets", "checkpoints")
LANDMARK_TASK_FILE = os.path.join(PROJECT_ROOT, "assets", "face_landmarker.task")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
IMG_SIZE = 224                 # EfficientNet-B0 input size
NUM_CLASSES = 2
CLASS_NAMES = ["Drowsy", "Non Drowsy"]
NUM_GEOMETRIC_FEATURES = 4    # EAR, MAR, eyebrow dist, head tilt

# Split ratios (by subject group, not by image)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 32
# Since train_all.py uses if __name__ == "__main__":, we can safely use workers on Windows
NUM_WORKERS = 6

# Stage 1: Frozen backbone — train only fusion head
STAGE1_EPOCHS = 5
STAGE1_LR = 1e-3

# Stage 2: Unfreeze backbone — fine-tune everything
STAGE2_EPOCHS = 15
STAGE2_LR = 1e-4
STAGE2_BACKBONE_LR = 1e-5     # Lower LR for pretrained layers

WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 5

# ──────────────────────────────────────────────
# Fusion MLP
# ──────────────────────────────────────────────
EMBEDDING_DIM = 1280           # EfficientNet-B0 output dim
FUSION_HIDDEN_1 = 256
FUSION_HIDDEN_2 = 64
DROPOUT_RATE = 0.3

# ──────────────────────────────────────────────
# Geometric-only MLP (for ablation)
# ──────────────────────────────────────────────
GEO_HIDDEN_1 = 32
GEO_HIDDEN_2 = 16

# ──────────────────────────────────────────────
# ImageNet normalization (for pretrained backbone)
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
