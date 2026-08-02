"""
dataset.py — PyTorch Dataset for drowsiness detection.

Handles image loading, augmentation, and geometric feature integration.
Applies Subject Baseline Normalization and returns Domain Labels for GRL.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import config


def get_subject_prefix(filename: str) -> str:
    """Extract subject prefix from filename."""
    name = os.path.splitext(filename)[0]
    prefix = "".join(c for c in name if c.isalpha()).upper()
    return prefix if prefix else "UNKNOWN"


class DrowsinessDataset(Dataset):
    """
    Dataset that combines images with precomputed geometric features.
    Applies Subject-Level Baseline Normalization.
    
    Args:
        split_items: list of dicts with 'filename', 'path', 'label'
        geometric_df: DataFrame with columns [filename, ear, mar, eyebrow_dist, 
                      head_tilt, success]
        augment: whether to apply training augmentations
        subject_to_idx: mapping from subject string to integer domain label
    """
    
    def __init__(self, split_items: list, geometric_df: pd.DataFrame,
                 augment: bool = False, subject_to_idx: dict = None):
        self.items = split_items
        self.augment = augment
        self.subject_to_idx = subject_to_idx or {}
        
        # Build a lookup from filename → geometric features
        self.geo_lookup = {}
        self.subject_stats = {}
        
        if geometric_df is not None:
            # Group features by subject to compute baselines
            subject_values = {}
            for _, row in geometric_df.iterrows():
                fname = row["filename"]
                subj = get_subject_prefix(fname)
                
                geo = {
                    "ear": float(row["ear"]),
                    "mar": float(row["mar"]),
                    "eyebrow_dist": float(row["eyebrow_dist"]),
                    "head_tilt": float(row["head_tilt"]),
                    "success": bool(row["success"]),
                    "subject": subj
                }
                self.geo_lookup[fname] = geo
                
                if geo["success"]:
                    if subj not in subject_values:
                        subject_values[subj] = {"ear": [], "mar": [], "eyebrow_dist": [], "head_tilt": []}
                    subject_values[subj]["ear"].append(geo["ear"])
                    subject_values[subj]["mar"].append(geo["mar"])
                    subject_values[subj]["eyebrow_dist"].append(geo["eyebrow_dist"])
                    subject_values[subj]["head_tilt"].append(geo["head_tilt"])

            # Compute 95th percentile (max baseline) for each subject
            for subj, vals in subject_values.items():
                self.subject_stats[subj] = {
                    "ear_max": np.percentile(vals["ear"], 95) if vals["ear"] else 1e-6,
                    "mar_max": np.percentile(vals["mar"], 95) if vals["mar"] else 1e-6,
                    "eyebrow_max": np.percentile(vals["eyebrow_dist"], 95) if vals["eyebrow_dist"] else 1e-6,
                    "head_tilt_max": np.percentile(vals["head_tilt"], 95) if vals["head_tilt"] else 1e-6,
                }
        
        # Global fallback stats if a subject is completely missing
        if self.subject_stats:
            self.global_ear_max = np.mean([s["ear_max"] for s in self.subject_stats.values()])
            self.global_mar_max = np.mean([s["mar_max"] for s in self.subject_stats.values()])
            self.global_eb_max = np.mean([s["eyebrow_max"] for s in self.subject_stats.values()])
            self.global_tilt_max = np.mean([s["head_tilt_max"] for s in self.subject_stats.values()])
        else:
            self.global_ear_max = 1.0
            self.global_mar_max = 1.0
            self.global_eb_max = 1.0
            self.global_tilt_max = 1.0

        # Image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                       saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
                transforms.RandomGrayscale(p=0.3),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), value='random'),
                transforms.Normalize(mean=config.IMAGENET_MEAN, 
                                     std=config.IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, 
                                     std=config.IMAGENET_STD),
            ])
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load image
        image = Image.open(item["path"]).convert("RGB")
        image_tensor = self.transform(image)
        
        # Get geometric features
        filename = item["filename"]
        subj = get_subject_prefix(filename)
        domain_label = self.subject_to_idx.get(subj, 0)  # Default 0 if unseen in val/test
        
        if filename in self.geo_lookup and self.geo_lookup[filename]["success"]:
            geo = self.geo_lookup[filename]
            raw_features = [geo["ear"], geo["mar"], 
                           geo["eyebrow_dist"], geo["head_tilt"]]
        else:
            # Use max baselines for failed detections (imputation)
            stats = self.subject_stats.get(subj, None)
            if stats is None:
                raw_features = [self.global_ear_max, self.global_mar_max, 
                                self.global_eb_max, self.global_tilt_max]
            else:
                raw_features = [stats["ear_max"], stats["mar_max"], 
                                stats["eyebrow_max"], stats["head_tilt_max"]]
        
        # Subject Baseline Normalization
        stats = self.subject_stats.get(subj, None)
        if stats:
            norm_features = [
                raw_features[0] / stats["ear_max"],
                raw_features[1] / stats["mar_max"],
                raw_features[2] / stats["eyebrow_max"],
                raw_features[3] / stats["head_tilt_max"],
            ]
        else:
            norm_features = [
                raw_features[0] / self.global_ear_max,
                raw_features[1] / self.global_mar_max,
                raw_features[2] / self.global_eb_max,
                raw_features[3] / self.global_tilt_max,
            ]
            
        geo_tensor = torch.tensor(norm_features, dtype=torch.float32)
        geo_tensor = torch.clamp(geo_tensor, min=0.0, max=2.0) # Normal scale around 0-1.0
        
        label = torch.tensor(item["label"], dtype=torch.long)
        domain_label = torch.tensor(domain_label, dtype=torch.long)
        
        return image_tensor, geo_tensor, label, domain_label


def create_dataloaders(splits: dict, geometric_df: pd.DataFrame,
                       batch_size: int = config.BATCH_SIZE) -> dict:
    """
    Create train/val/test DataLoaders from splits and geometric features.
    Builds the subject-to-domain mapping from the training split.
    
    Returns dict with 'train', 'val', 'test' DataLoaders and 'num_domains' integer.
    """
    # 1. Extract all unique subjects in the training set for Domain Classification
    train_subjects = set()
    for item in splits["train"]:
        subj = get_subject_prefix(item["filename"])
        train_subjects.add(subj)
        
    train_subjects = sorted(list(train_subjects))
    subject_to_idx = {subj: idx for idx, subj in enumerate(train_subjects)}
    num_domains = len(train_subjects)
    
    print(f"\nExtracted {num_domains} unique subjects in Training set for GRL Domain Classification.")
    
    # 2. Create datasets
    train_ds = DrowsinessDataset(splits["train"], geometric_df, augment=True, subject_to_idx=subject_to_idx)
    val_ds = DrowsinessDataset(splits["val"], geometric_df, augment=False, subject_to_idx=subject_to_idx)
    test_ds = DrowsinessDataset(splits["test"], geometric_df, augment=False, subject_to_idx=subject_to_idx)
    
    # 3. Create DataLoaders
    loaders = {
        "train": torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
        ),
        "val": torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=True
        ),
        "test": torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=True
        ),
    }
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_ds)} images, {len(loaders['train'])} batches")
    print(f"  Val:   {len(val_ds)} images, {len(loaders['val'])} batches")
    print(f"  Test:  {len(test_ds)} images, {len(loaders['test'])} batches")
    
    return loaders, num_domains
