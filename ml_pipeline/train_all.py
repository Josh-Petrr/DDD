"""
train_all.py — Orchestrator to train all model variants.

Runs sequentially:
  1. Baseline CNN (EfficientNet-B0 only)
  2. Geometric-only MLP (landmarks only)
  3. Fusion Model (CNN + landmarks)
  4. Fusion GRL Model (Adversarial Domain Adaptation)

Usage:
    python train_all.py
"""

import os
import sys
import time
import json

import torch
import pandas as pd
import numpy as np
import random

import config
from data_split import create_subject_disjoint_split, save_splits, \
    load_splits, print_split_summary
from dataset import create_dataloaders
from models import get_model, count_parameters
from train import train_model


def set_seed(seed: int = config.SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    start_time = time.time()
    set_seed()
    
    print("=" * 60)
    print("DRIVER DROWSINESS DETECTION — TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ── Step 1: Create or load splits ──
    print("\n" + "-" * 60)
    print("Step 1: Data Splitting")
    print("-" * 60)
    
    if os.path.exists(config.SPLITS_FILE):
        print(f"Loading existing splits from {config.SPLITS_FILE}")
        splits = load_splits()
    else:
        print("Creating new subject-disjoint splits...")
        splits = create_subject_disjoint_split()
        save_splits(splits)
    
    print_split_summary(splits)
    
    # ── Step 2: Load geometric features ──
    print("\n" + "─" * 60)
    print("Step 2: Loading Geometric Features")
    print("─" * 60)
    
    if not os.path.exists(config.FEATURES_FILE):
        print(f"ERROR: {config.FEATURES_FILE} not found!")
        print("Run 'python extract_features.py' first.")
        sys.exit(1)
    
    geo_df = pd.read_csv(config.FEATURES_FILE)
    success_rate = geo_df["success"].mean() * 100
    print(f"Loaded {len(geo_df)} feature records "
          f"(detection success rate: {success_rate:.1f}%)")
    
    # ── Step 3: Create DataLoaders ──
    print("\n" + "─" * 60)
    print("Step 3: Creating DataLoaders")
    print("─" * 60)
    
    loaders, num_domains = create_dataloaders(splits, geo_df)
    
    # ── Step 4: Train all models ──
    # (architecture_type, save_name) — save_name gets _3 suffix for new iteration
    model_configs = [
        ("fusion_grl", "fusion_grl_3")
    ]
    histories = {}
    
    for arch_type, save_name in model_configs:
        print(f"\n{'#' * 60}")
        print(f"# Training: {save_name.upper()}")
        print(f"{'#' * 60}")
        
        model = get_model(arch_type, pretrained=True, num_domains=num_domains)
        params = count_parameters(model)
        print(f"Parameters -- Total: {params['total']:,} | "
              f"Trainable: {params['trainable']:,}")
        
        history = train_model(model, loaders, save_name)
        histories[save_name] = history
        
        # Clear GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # -- Summary --
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"ALL TRAINING COMPLETE -- Total time: {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
    
    print(f"\nBest validation accuracies:")
    for _, save_name in model_configs:
        best_val = max(histories[save_name]["val_acc"])
        print(f"  {save_name:>12}: {100*best_val:.2f}%")
    
    print(f"\nCheckpoints saved in: {config.CHECKPOINTS_DIR}")
    print(f"Training logs saved in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
