"""
dataset.py — PyTorch Dataset for drowsiness detection.

Handles image loading, augmentation, and geometric feature integration.
Returns (image_tensor, geometric_features_tensor, label) tuples.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import config


class DrowsinessDataset(Dataset):
    """
    Dataset that combines images with precomputed geometric features.
    
    Args:
        split_items: list of dicts with 'filename', 'path', 'label'
        geometric_df: DataFrame with columns [filename, ear, mar, eyebrow_dist, 
                      head_tilt, success]
        augment: whether to apply training augmentations
    """
    
    def __init__(self, split_items: list, geometric_df: pd.DataFrame,
                 augment: bool = False):
        self.items = split_items
        self.augment = augment
        
        # Build a lookup from filename → geometric features
        self.geo_lookup = {}
        if geometric_df is not None:
            for _, row in geometric_df.iterrows():
                self.geo_lookup[row["filename"]] = {
                    "ear": float(row["ear"]),
                    "mar": float(row["mar"]),
                    "eyebrow_dist": float(row["eyebrow_dist"]),
                    "head_tilt": float(row["head_tilt"]),
                    "success": bool(row["success"]),
                }
        
        # Image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                       saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
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
        
        # Compute geometric feature statistics for normalization
        if self.geo_lookup:
            ears = [v["ear"] for v in self.geo_lookup.values() if v["success"]]
            mars = [v["mar"] for v in self.geo_lookup.values() if v["success"]]
            ebds = [v["eyebrow_dist"] for v in self.geo_lookup.values() if v["success"]]
            tilts = [v["head_tilt"] for v in self.geo_lookup.values() if v["success"]]
            
            self.geo_stats = {
                "ear": (np.mean(ears), np.std(ears) + 1e-8),
                "mar": (np.mean(mars), np.std(mars) + 1e-8),
                "eyebrow_dist": (np.mean(ebds), np.std(ebds) + 1e-8),
                "head_tilt": (np.mean(tilts), np.std(tilts) + 1e-8),
            }
        else:
            self.geo_stats = {
                "ear": (0.0, 1.0), "mar": (0.0, 1.0),
                "eyebrow_dist": (0.0, 1.0), "head_tilt": (0.0, 1.0),
            }
    
    def set_geo_stats(self, stats: dict):
        """Set normalization stats (use training set stats for val/test)."""
        self.geo_stats = stats
    
    def get_geo_stats(self) -> dict:
        """Get computed normalization stats."""
        return self.geo_stats
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load image
        image = Image.open(item["path"]).convert("RGB")
        image_tensor = self.transform(image)
        
        # Get geometric features
        filename = item["filename"]
        if filename in self.geo_lookup and self.geo_lookup[filename]["success"]:
            geo = self.geo_lookup[filename]
            raw_features = [geo["ear"], geo["mar"], 
                           geo["eyebrow_dist"], geo["head_tilt"]]
        else:
            # Use mean values for failed detections (imputation)
            raw_features = [
                self.geo_stats["ear"][0],
                self.geo_stats["mar"][0],
                self.geo_stats["eyebrow_dist"][0],
                self.geo_stats["head_tilt"][0],
            ]
        
        # Z-score normalize geometric features
        normalized = [
            (raw_features[0] - self.geo_stats["ear"][0]) / self.geo_stats["ear"][1],
            (raw_features[1] - self.geo_stats["mar"][0]) / self.geo_stats["mar"][1],
            (raw_features[2] - self.geo_stats["eyebrow_dist"][0]) / self.geo_stats["eyebrow_dist"][1],
            (raw_features[3] - self.geo_stats["head_tilt"][0]) / self.geo_stats["head_tilt"][1],
        ]
        
        geo_tensor = torch.tensor(normalized, dtype=torch.float32)
        label = torch.tensor(item["label"], dtype=torch.long)
        
        return image_tensor, geo_tensor, label


def create_dataloaders(splits: dict, geometric_df: pd.DataFrame,
                       batch_size: int = config.BATCH_SIZE) -> dict:
    """
    Create train/val/test DataLoaders from splits and geometric features.
    
    Returns dict with 'train', 'val', 'test' DataLoaders.
    """
    # Create datasets
    train_ds = DrowsinessDataset(splits["train"], geometric_df, augment=True)
    val_ds = DrowsinessDataset(splits["val"], geometric_df, augment=False)
    test_ds = DrowsinessDataset(splits["test"], geometric_df, augment=False)
    
    # Use training set stats for all splits
    train_stats = train_ds.get_geo_stats()
    val_ds.set_geo_stats(train_stats)
    test_ds.set_geo_stats(train_stats)
    
    # Create DataLoaders
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
    
    return loaders
