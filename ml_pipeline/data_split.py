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

from ml_pipeline import config


def _parse_all_subjects() -> dict:
    """
    Globally group all images from both classes by subject prefix.
    Returns: dict[prefix] = {"drowsy": [...], "non_drowsy": [...]}
    """
    global_groups = defaultdict(lambda: {"drowsy": [], "non_drowsy": []})
    
    def add_directory(directory, label, label_key):
        if not os.path.exists(directory): return
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for fname in files:
            name = os.path.splitext(fname)[0]
            prefix = "".join(c for c in name if c.isalpha()).upper()
            if not prefix: prefix = "UNKNOWN"
            
            global_groups[prefix][label_key].append({
                "filename": fname,
                "path": os.path.join(directory, fname),
                "label": label
            })
            
    add_directory(config.DROWSY_DIR, 0, "drowsy")
    add_directory(config.NON_DROWSY_DIR, 1, "non_drowsy")
    return dict(global_groups)


def create_subject_disjoint_split(seed: int = config.SEED) -> dict:
    """
    Create train/val/test splits where subject groups don't overlap globally.
    Ensures an exact 50/50 class balance in Val and Test sets by trimming.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    groups = _parse_all_subjects()
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    
    # Calculate target sizes based on TOTAL images
    total_images = sum(len(v["drowsy"]) + len(v["non_drowsy"]) for v in groups.values())
    train_target = int(total_images * config.TRAIN_RATIO)
    val_target = int(total_images * config.VAL_RATIO)
    
    raw_splits = {"train": [], "val": [], "test": []}
    count = 0
    
    # Greedily assign subjects to splits
    for key in group_keys:
        subj = groups[key]
        all_subj_imgs = subj["drowsy"] + subj["non_drowsy"]
        
        if count < train_target:
            raw_splits["train"].extend(all_subj_imgs)
        elif count < train_target + val_target:
            raw_splits["val"].extend(all_subj_imgs)
        else:
            raw_splits["test"].extend(all_subj_imgs)
            
        count += len(all_subj_imgs)
        
    # Apply strict 50/50 class balance for Val and Test sets only
    balanced_splits = {"train": raw_splits["train"]}
    
    for split_name in ["val", "test"]:
        drowsy = [x for x in raw_splits[split_name] if x["label"] == 0]
        non_drowsy = [x for x in raw_splits[split_name] if x["label"] == 1]
        
        # Balance by taking the minimum count
        min_count = min(len(drowsy), len(non_drowsy))
        
        if min_count > 0:
            drowsy = random.sample(drowsy, min_count)
            non_drowsy = random.sample(non_drowsy, min_count)
            
        balanced_splits[split_name] = drowsy + non_drowsy

    for key in balanced_splits:
        random.shuffle(balanced_splits[key])
    
    return balanced_splits


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
