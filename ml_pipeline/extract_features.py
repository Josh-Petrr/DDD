"""
extract_features.py — Batch geometric feature extraction pipeline.

Runs landmark_features.py across the entire dataset and saves results
to geometric_features.csv.

Uses a single LandmarkExtractor (not multiprocessed, since the MediaPipe
Tasks API objects can't be pickled). Processing is still fast because
FaceLandmarker runs efficiently on CPU.

Usage:
    python extract_features.py
"""

import os
import csv
import time

import cv2
from tqdm import tqdm

from ml_pipeline import config
from landmark_features import LandmarkExtractor


def extract_all_features():
    """Extract geometric features for every image in the dataset."""
    
    # Gather all image paths
    tasks = []
    
    for label, (label_name, directory) in enumerate([
        ("Drowsy", config.DROWSY_DIR),
        ("Non Drowsy", config.NON_DROWSY_DIR)
    ]):
        files = sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        for fname in files:
            filepath = os.path.join(directory, fname)
            tasks.append((filepath, fname, label))
    
    print(f"Extracting geometric features from {len(tasks)} images...")
    
    start_time = time.time()
    results = []
    success_count = 0
    fail_count = 0
    
    # Create a single extractor instance (reused for all images)
    extractor = LandmarkExtractor()
    
    for filepath, filename, label in tqdm(tasks, desc="Extracting features"):
        image = cv2.imread(filepath)
        
        if image is None:
            result = {
                "filename": filename, "label": label,
                "ear": 0.0, "mar": 0.0, "eyebrow_dist": 0.0,
                "head_tilt": 0.0, "success": False
            }
        else:
            features = extractor.extract(image)
            result = {
                "filename": filename,
                "label": label,
                "ear": features["ear"],
                "mar": features["mar"],
                "eyebrow_dist": features["eyebrow_dist"],
                "head_tilt": features["head_tilt"],
                "success": features["success"],
            }
        
        results.append(result)
        if result["success"]:
            success_count += 1
        else:
            fail_count += 1
    
    del extractor
    elapsed = time.time() - start_time
    
    # Save to CSV
    fieldnames = ["filename", "label", "ear", "mar", "eyebrow_dist",
                  "head_tilt", "success"]
    
    # Sort by filename for consistency
    results.sort(key=lambda x: x["filename"])
    
    with open(config.FEATURES_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    total = success_count + fail_count
    print(f"\n{'=' * 50}")
    print(f"Feature Extraction Complete")
    print(f"{'=' * 50}")
    print(f"Total images:      {total}")
    print(f"Successful:        {success_count} ({100*success_count/total:.1f}%)")
    print(f"Failed (no face):  {fail_count} ({100*fail_count/total:.1f}%)")
    print(f"Time elapsed:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Speed:             {total/elapsed:.1f} images/sec")
    print(f"Output saved to:   {config.FEATURES_FILE}")
    
    if fail_count / total > 0.15:
        print(f"\n⚠ WARNING: Failure rate ({100*fail_count/total:.1f}%) exceeds 15%.")
        print("  Consider lowering min_detection_confidence in LandmarkExtractor.")
    
    return results


if __name__ == "__main__":
    extract_all_features()
