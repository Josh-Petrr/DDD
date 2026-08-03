"""
dataset_lstm.py — PyTorch Dataset for Sequence Modeling.

Reads the pre-extracted (N_frames, 1284) numpy arrays using memory mapping
and yields sliding windows of shape (seq_len, 1284) for LSTM training.
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

import config


class LSTMSequenceDataset(Dataset):
    """
    Loads pre-extracted (N_frames, 1284) arrays and yields 
    sliding windows of shape (seq_len, 1284).
    """
    def __init__(self, split_name: str, seq_len: int = 30, stride: int = 15):
        self.seq_len = seq_len
        self.stride = stride
        self.samples = []  # List of tuples: (feature_array_path, start_idx, label)
        
        # Find all numpy files for this split
        # Files are named like: train_A_0.npy
        pattern = os.path.join(config.SEQUENCE_FEATURES_DIR, f"{split_name}_*.npy")
        files = glob.glob(pattern)
        
        for file_path in files:
            # Extract label from filename (e.g. train_A_0.npy -> 0)
            basename = os.path.basename(file_path)
            label_str = basename.split("_")[-1].split(".")[0]
            label = int(label_str)
            
            # Read shape using memory mapping to build the index without loading to RAM
            arr = np.load(file_path, mmap_mode='r')
            num_frames = arr.shape[0]
            
            # Slide window
            for start_idx in range(0, num_frames - seq_len + 1, stride):
                self.samples.append((file_path, start_idx, label))
                
        print(f"[{split_name.upper()}] LSTMSequenceDataset created {len(self.samples)} windows (len={seq_len}, stride={stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, start_idx, label = self.samples[idx]
        
        # Memory map the file to grab just the chunk we need
        arr = np.load(file_path, mmap_mode='r')
        chunk = arr[start_idx : start_idx + self.seq_len]
        
        # Convert to tensor
        x = torch.tensor(chunk, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y


def create_lstm_dataloaders(seq_len: int = 30, stride: int = 15, batch_size: int = config.BATCH_SIZE):
    """Creates Train, Val, Test DataLoaders for sequences."""
    loaders = {}
    
    # Train: uses stride to augment data
    train_ds = LSTMSequenceDataset("train", seq_len, stride)
    loaders["train"] = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    
    # Val/Test: use no overlap (stride=seq_len) for strict evaluation
    val_ds = LSTMSequenceDataset("val", seq_len, stride=seq_len)
    loaders["val"] = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    test_ds = LSTMSequenceDataset("test", seq_len, stride=seq_len)
    loaders["test"] = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    return loaders
