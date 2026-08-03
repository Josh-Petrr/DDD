"""
extract_features.py — Offline feature extraction for Sequence Modeling.

Runs the entire dataset through the pre-trained FUSION_GRL_V4 feature extractor.
Groups the frames by subject and class, sorts them chronologically,
and saves them as (N_frames, 1284) numpy arrays.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from dataset import get_subject_prefix, DrowsinessDataset
from data_split import load_splits
from models import get_model


def extract():
    os.makedirs(config.SEQUENCE_FEATURES_DIR, exist_ok=True)
    
    # 1. Load trained feature extractor
    print("Loading FUSION_GRL_V4 feature extractor...")
    splits = load_splits()
    
    # Determine num_domains just to satisfy model initialization
    train_subjects = set(get_subject_prefix(item["filename"]) for item in splits["train"])
    num_domains = len(train_subjects)
    
    model = get_model("fusion_grl_v4", pretrained=False, num_domains=num_domains)
    checkpoint = os.path.join(config.CHECKPOINTS_DIR, "fusion_grl_4_best.pth")
    checkpoint_v4 = os.path.join(config.CHECKPOINTS_DIR, "V4", "fusion_grl_4_best.pth")
    
    if os.path.exists(checkpoint):
        target_checkpoint = checkpoint
    elif os.path.exists(checkpoint_v4):
        target_checkpoint = checkpoint_v4
    else:
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint} or {checkpoint_v4}")
        
    model.load_state_dict(torch.load(target_checkpoint, map_location=config.DEVICE, weights_only=True))
    model = model.to(config.DEVICE)
    model.eval()
    
    # 2. Prepare dataset helper (for baselining and transformations)
    geo_df = pd.read_csv(config.FEATURES_FILE)
    ds = DrowsinessDataset([], geo_df, augment=False, subject_to_idx={})
    
    # Process all splits
    for split_name, items in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Group by subject AND label so we don't mix drowsy and non-drowsy frames
        sequences = {}
        for item in items:
            subj = get_subject_prefix(item["filename"])
            key = f"{subj}_{item['label']}"
            if key not in sequences:
                sequences[key] = []
            sequences[key].append(item)
            
        for key, seq_items in tqdm(sequences.items()):
            # Sort chronologically by filename (e.g., A0001, A0002)
            seq_items = sorted(seq_items, key=lambda x: x["filename"])
            
            all_features = []
            
            with torch.no_grad():
                for item in seq_items:
                    # Temporarily inject item into dataset to reuse exact normalization logic
                    ds.items = [item]
                    img_tensor, geo_tensor, _, _ = ds[0]
                    
                    img_tensor = img_tensor.unsqueeze(0).to(config.DEVICE)
                    geo_tensor = geo_tensor.unsqueeze(0).to(config.DEVICE)
                    
                    # Extract 1280-d visual feature from EfficientNet
                    cnn_emb = model.get_embedding(img_tensor)
                    
                    # Concat with 4-d geometric feature
                    fused = torch.cat([cnn_emb, geo_tensor], dim=1).cpu().numpy()[0]
                    all_features.append(fused)
            
            # Save numpy array of shape (N_frames, 1284)
            features_array = np.stack(all_features)
            out_path = os.path.join(config.SEQUENCE_FEATURES_DIR, f"{split_name}_{key}.npy")
            np.save(out_path, features_array)
            
    print(f"\nFeature extraction complete! Saved to {config.SEQUENCE_FEATURES_DIR}")

if __name__ == "__main__":
    extract()
