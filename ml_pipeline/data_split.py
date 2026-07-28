"""
data_split.py — Subject-disjoint train/val/test splitting.

Groups images by subject identity (using filename sequence gaps as subject
boundaries) and splits at the group level so no subject appears in more
than one partition.

Usage:
    python data_split.py
"""

import os
import json
import random
import numpy as np
from collections import defaultdict

import config


def _parse_subject_groups(directory: str, label: int) -> dict[str, list[dict]]:
    """
    Group images into subject blocks based on alphabetical filename prefixes.
    
    Strategy: Files with the same alphabetical prefix (e.g., 'A', 'ZA', 'a', 'zc')
    are considered to belong to the same subject/session.
    """
    files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if not files:
        return {}
    
    groups = defaultdict(list)
    
    for fname in files:
        # Extract alphabetical prefix from filename (e.g., "ZA0007.png" -> "ZA")
        name = os.path.splitext(fname)[0]
        prefix = "".join(c for c in name if c.isalpha())
        
        # Fallback if no letters are found
        if not prefix:
            prefix = "UNKNOWN"
            
        groups[prefix].append({
            "filename": fname,
            "path": os.path.join(directory, fname),
            "label": label
        })
        
    return dict(groups)


def create_subject_disjoint_split(seed: int = config.SEED) -> dict:
    """
    Create train/val/test splits where subject groups don't overlap.
    
    Returns dict with keys 'train', 'val', 'test', each containing a list
    of {"filename", "path", "label"} dicts.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get subject groups for both classes
    drowsy_groups = _parse_subject_groups(config.DROWSY_DIR, label=0)
    non_drowsy_groups = _parse_subject_groups(config.NON_DROWSY_DIR, label=1)
    
    print(f"Found {len(drowsy_groups)} subject groups in Drowsy "
          f"({sum(len(v) for v in drowsy_groups.values())} images)")
    print(f"Found {len(non_drowsy_groups)} subject groups in Non Drowsy "
          f"({sum(len(v) for v in non_drowsy_groups.values())} images)")
    
    splits = {"train": [], "val": [], "test": []}
    
    # Split each class independently to maintain balance
    for class_name, groups in [("Drowsy", drowsy_groups), 
                                ("Non Drowsy", non_drowsy_groups)]:
        group_keys = list(groups.keys())
        random.shuffle(group_keys)
        
        # Count total images to compute split points
        total_images = sum(len(groups[k]) for k in group_keys)
        train_target = int(total_images * config.TRAIN_RATIO)
        val_target = int(total_images * config.VAL_RATIO)
        
        # Assign groups to splits
        count = 0
        for key in group_keys:
            group_images = groups[key]
            if count < train_target:
                splits["train"].extend(group_images)
            elif count < train_target + val_target:
                splits["val"].extend(group_images)
            else:
                splits["test"].extend(group_images)
            count += len(group_images)
    
    # Shuffle within each split
    for key in splits:
        random.shuffle(splits[key])
    
    return splits


def save_splits(splits: dict, path: str = config.SPLITS_FILE):
    """Save splits to JSON for reproducibility."""
    # Convert to serializable format
    serializable = {}
    for key, items in splits.items():
        serializable[key] = [
            {"filename": item["filename"], "path": item["path"], "label": item["label"]}
            for item in items
        ]
    
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nSplits saved to {path}")


def load_splits(path: str = config.SPLITS_FILE) -> dict:
    """
    Load previously saved splits.

    Re-resolves image paths against the current config.DATASET_ROOT so that
    splits.json remains portable even if the project folder is moved/renamed.
    """
    with open(path, 'r') as f:
        raw = json.load(f)

    # Map label index → directory (0=Drowsy, 1=Non Drowsy)
    label_to_dir = {
        0: config.DROWSY_DIR,
        1: config.NON_DROWSY_DIR,
    }

    resolved = {}
    for split_name, items in raw.items():
        resolved_items = []
        for item in items:
            correct_dir = label_to_dir[item["label"]]
            correct_path = os.path.join(correct_dir, item["filename"])
            resolved_items.append({
                "filename": item["filename"],
                "path": correct_path,
                "label": item["label"],
            })
        resolved[split_name] = resolved_items

    return resolved


def print_split_summary(splits: dict):
    """Print a summary table of the splits."""
    print("\n" + "=" * 60)
    print("SUBJECT-DISJOINT SPLIT SUMMARY")
    print("=" * 60)
    
    for split_name in ["train", "val", "test"]:
        items = splits[split_name]
        drowsy = sum(1 for x in items if x["label"] == 0)
        non_drowsy = sum(1 for x in items if x["label"] == 1)
        total = len(items)
        print(f"\n{split_name.upper():>6}: {total:>6} images "
              f"| Drowsy: {drowsy:>5} ({100*drowsy/total:.1f}%) "
              f"| Non-Drowsy: {non_drowsy:>5} ({100*non_drowsy/total:.1f}%)")
    
    total_all = sum(len(splits[k]) for k in splits)
    print(f"\n{'TOTAL':>6}: {total_all:>6} images")
    print("=" * 60)


if __name__ == "__main__":
    print("Creating subject-disjoint data splits...")
    splits = create_subject_disjoint_split()
    print_split_summary(splits)
    save_splits(splits)
