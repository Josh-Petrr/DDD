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
    Group images into subject blocks.
    
    Strategy: consecutive numbered images (e.g., A0001–A0080) are assumed to
    be from the same session/subject.  A gap of ≥3 in the numbering signals a
    new subject boundary.  This is a practical heuristic for datasets that
    don't include explicit subject IDs.
    """
    files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if not files:
        return {}
    
    # Extract numeric part from filename
    def get_number(fname):
        # "A0001.png" → 1, "B1234.png" → 1234
        name = os.path.splitext(fname)[0]
        num_str = ''.join(c for c in name if c.isdigit())
        return int(num_str) if num_str else 0
    
    prefix = files[0][0]  # 'A' for Drowsy, 'B' for Non Drowsy
    GAP_THRESHOLD = 3     # gap of ≥3 = new subject
    
    groups = {}
    current_group = []
    current_group_id = 0
    prev_num = None
    
    for f in files:
        num = get_number(f)
        if prev_num is not None and (num - prev_num) >= GAP_THRESHOLD:
            # Save current group
            group_key = f"{prefix}_group_{current_group_id}"
            groups[group_key] = [
                {
                    "filename": fname,
                    "path": os.path.join(directory, fname),
                    "label": label
                }
                for fname in current_group
            ]
            current_group = []
            current_group_id += 1
        
        current_group.append(f)
        prev_num = num
    
    # Save last group
    if current_group:
        group_key = f"{prefix}_group_{current_group_id}"
        groups[group_key] = [
            {
                "filename": fname,
                "path": os.path.join(directory, fname),
                "label": label
            }
            for fname in current_group
        ]
    
    return groups


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
