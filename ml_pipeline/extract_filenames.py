"""
extract_filenames.py — Extract all filenames from Drowsy and Non Drowsy folders.

Outputs:
  - data/all_filenames.csv  (CSV format with columns: filename, category, label, full_path)
  - data/all_filenames.json (JSON format)
"""

import os
import sys
import csv
import json

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config


def extract_all_filenames():
    """Scan Drowsy and Non Drowsy folders and extract all filenames."""
    
    categories = [
        ("Drowsy", config.DROWSY_DIR, 0),
        ("Non Drowsy", config.NON_DROWSY_DIR, 1)
    ]
    
    all_records = []
    summary = {}
    
    for cat_name, directory, label in categories:
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            continue
        
        files = sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ])
        
        prefixes = set(''.join(c for c in os.path.splitext(f)[0] if not c.isdigit()) for f in files)
        
        summary[cat_name] = {
            "total_files": len(files),
            "prefixes": sorted(list(prefixes)),
            "directory": directory
        }
        
        for fname in files:
            full_path = os.path.join(directory, fname)
            all_records.append({
                "filename": fname,
                "category": cat_name,
                "label": label,
                "full_path": full_path
            })
    
    # Define output paths
    csv_path = os.path.join(config.PROJECT_ROOT, "data", "all_filenames.csv")
    json_path = os.path.join(config.PROJECT_ROOT, "data", "all_filenames.json")
    
    # Save CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "label", "full_path"])
        writer.writeheader()
        writer.writerows(all_records)
    
    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2)
    
    # Print Summary
    print("=" * 60)
    print("FILENAME EXTRACTION COMPLETE")
    print("=" * 60)
    total_images = len(all_records)
    for cat_name, info in summary.items():
        print(f"\nCategory: {cat_name}")
        print(f"   - File Count: {info['total_files']:,} images")
        print(f"   - Prefixes Found: {info['prefixes']}")
        print(f"   - Directory: {info['directory']}")
    
    print(f"\nTotal Images Extracted: {total_images:,}")
    print(f"Saved CSV  to: {csv_path}")
    print(f"Saved JSON to: {json_path}")
    print("=" * 60)
    
    return all_records


if __name__ == "__main__":
    extract_all_filenames()
