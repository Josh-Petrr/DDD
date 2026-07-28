"""
export_unseen_data.py — Export and copy all images NOT used in training (Test set).

Outputs:
  - data/unseen_test_filenames.csv (CSV file listing all unseen test images)
  - data/unseen_test_filenames.json (JSON metadata of unseen test images)
  - data/unseen_test_images/ (Folder containing physical copies of unseen test images)
"""

import os
import sys
import csv
import json
import shutil

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config
from ml_pipeline.data_split import load_splits


def export_unseen_data():
    """Extract and copy all test images that were NEVER used during model training."""
    
    splits_path = config.SPLITS_FILE
    if not os.path.exists(splits_path):
        print(f"Error: {splits_path} not found. Run data_split.py first.")
        return
    
    splits = load_splits(splits_path)
    test_items = splits.get("test", [])
    
    if not test_items:
        print("Error: No test split items found in splits.json.")
        return
    
    print("=" * 60)
    print("EXPORTING UNSEEN TEST DATA (NOT USED IN TRAINING)")
    print("=" * 60)
    print(f"Total Unseen Test Images: {len(test_items):,}")
    
    # Destination paths
    output_dir = os.path.join(config.PROJECT_ROOT, "data", "unseen_test_images")
    drowsy_dest = os.path.join(output_dir, "Drowsy")
    non_drowsy_dest = os.path.join(output_dir, "Non_Drowsy")
    
    os.makedirs(drowsy_dest, exist_ok=True)
    os.makedirs(non_drowsy_dest, exist_ok=True)
    
    csv_path = os.path.join(config.PROJECT_ROOT, "data", "unseen_test_filenames.csv")
    json_path = os.path.join(config.PROJECT_ROOT, "data", "unseen_test_filenames.json")
    
    records = []
    copied_count = 0
    
    for item in test_items:
        filename = item["filename"]
        label = item["label"]
        src_path = item["path"]
        category = "Drowsy" if label == 0 else "Non Drowsy"
        
        dest_folder = drowsy_dest if label == 0 else non_drowsy_dest
        dest_path = os.path.join(dest_folder, filename)
        
        records.append({
            "filename": filename,
            "category": category,
            "label": label,
            "original_path": src_path,
            "copied_path": dest_path
        })
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            copied_count += 1
    
    # Save CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "label", "original_path", "copied_path"])
        writer.writeheader()
        writer.writerows(records)
    
    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    
    drowsy_count = sum(1 for r in records if r["label"] == 0)
    non_drowsy_count = sum(1 for r in records if r["label"] == 1)
    
    print(f"\nUnseen Data Breakdown:")
    print(f"   - Drowsy (Label 0):     {drowsy_count:,} images")
    print(f"   - Non Drowsy (Label 1): {non_drowsy_count:,} images")
    print(f"   - Total Copied:          {copied_count:,} images")
    
    print(f"\nSaved CSV index to:  {csv_path}")
    print(f"Saved JSON index to: {json_path}")
    print(f"Copied images to:     {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    export_unseen_data()
