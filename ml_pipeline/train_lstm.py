"""
train_lstm.py — Training pipeline for the LSTM sequence model.
"""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset_lstm import create_lstm_dataloaders
from models import get_model


def train_lstm():
    # 1. Check if features exist
    if not os.path.exists(config.SEQUENCE_FEATURES_DIR) or not os.listdir(config.SEQUENCE_FEATURES_DIR):
        print("ERROR: Sequence features not found!")
        print("Please run `python ml_pipeline/extract_features.py` first.")
        return
        
    print(f"============================================================")
    print(f"LSTM SEQUENCE MODEL TRAINING")
    print(f"============================================================")
    
    # 2. Create DataLoaders
    # We'll use 30 frames (1 second) and stride of 5 (massive overlap for data augmentation)
    loaders = create_lstm_dataloaders(seq_len=30, stride=5)
    
    # 3. Initialize Model
    model = get_model("lstm")
    model = model.to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 4. Training Loop
    num_epochs = 20
    best_val_acc = 0.0
    patience_counter = 0
    save_path = os.path.join(config.CHECKPOINTS_DIR, "lstm_best.pth")
    
    print("\nStarting Training...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x, y in tqdm(loaders["train"], desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            _, predicted = torch.max(logits, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in tqdm(loaders["val"], desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                
                logits = model(x)
                loss = criterion(logits, y)
                
                val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc*100:.2f}% | Val Loss={val_loss:.4f} Acc={val_acc*100:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved! (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    total_time = (time.time() - start_time) / 60
    print(f"\n============================================================")
    print(f"TRAINING COMPLETE -- Total time: {total_time:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"============================================================")

if __name__ == "__main__":
    train_lstm()
