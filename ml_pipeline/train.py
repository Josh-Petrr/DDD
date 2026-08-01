"""
train.py — Training loop for drowsiness detection models.

Supports 2-stage training:
  Stage 1: Frozen backbone — train only the head (fast convergence)
  Stage 2: Unfreeze backbone — fine-tune everything with lower LR

Supports FusionGRLModel with Adversarial Domain Adaptation.
"""

import os
import csv
import time
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import config


def train_model(model: nn.Module, loaders: dict, model_name: str,
                device: torch.device = config.DEVICE) -> dict:
    """
    Train a model with 2-stage schedule.
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    domain_criterion = nn.CrossEntropyLoss()
    
    scaler = GradScaler(device="cuda") if device.type == "cuda" else None
    
    history = {
        "epoch": [], "stage": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_best.pth")
    
    # ──────────────────────────────────────────
    # Stage 1: Frozen backbone
    # ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 1: Frozen backbone — training head only")
    print(f"{'='*60}")
    
    model.freeze_backbone()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.STAGE1_LR,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.STAGE1_EPOCHS
    )
    
    for epoch in range(1, config.STAGE1_EPOCHS + 1):
        # Progressively increase alpha for GRL from 0.0 to 1.0, but scale down by 0.05
        p = float(epoch) / (config.STAGE1_EPOCHS + config.STAGE2_EPOCHS)
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * 0.05
        
        metrics = _run_epoch(model, loaders, optimizer, criterion, domain_criterion, scaler,
                             scheduler, device, epoch, "Stage1", model_name, alpha)
        _update_history(history, epoch, "stage1", metrics, optimizer)
        
        if metrics["val_acc"] > best_val_acc:
            best_val_acc = metrics["val_acc"]
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
    
    # ──────────────────────────────────────────
    # Stage 2: Unfreeze backbone — fine-tune all
    # ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 2: Unfrozen backbone — fine-tuning all layers")
    print(f"{'='*60}")
    
    model.unfreeze_backbone()
    
    param_groups = _get_param_groups(model, model_name)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.STAGE2_EPOCHS
    )
    
    patience_counter = 0  # Reset patience for stage 2
    
    for epoch in range(1, config.STAGE2_EPOCHS + 1):
        global_epoch = config.STAGE1_EPOCHS + epoch
        
        p = float(global_epoch) / (config.STAGE1_EPOCHS + config.STAGE2_EPOCHS)
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * 0.05
        
        metrics = _run_epoch(model, loaders, optimizer, criterion, domain_criterion, scaler,
                             scheduler, device, epoch, "Stage2", model_name, alpha)
        _update_history(history, global_epoch, "stage2", metrics, optimizer)
        
        if metrics["val_acc"] > best_val_acc:
            best_val_acc = metrics["val_acc"]
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚡ Early stopping triggered at epoch {global_epoch} "
                      f"(patience={config.EARLY_STOPPING_PATIENCE})")
                break
    
    if best_model_state is not None:
        torch.save(best_model_state, checkpoint_path)
        print(f"\n✅ Best model saved: {checkpoint_path}")
        print(f"   Best val accuracy: {best_val_acc:.4f}")
        model.load_state_dict(best_model_state)
    
    _save_history_csv(history, model_name)
    return history


def _run_epoch(model, loaders, optimizer, criterion, domain_criterion, scaler,
               scheduler, device, epoch, stage_name, model_name, alpha):
    """Run one training + validation epoch."""
    
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(loaders["train"], desc=f"  {stage_name} Epoch {epoch} [Train]", leave=False)
    
    for images, geo_features, labels, domain_labels in pbar:
        images = images.to(device, non_blocking=True)
        geo_features = geo_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domain_labels = domain_labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast(device_type="cuda"):
                if model_name.startswith("fusion_grl"):
                    drowsy_logits, domain_logits = model(images, geo_features, alpha)
                    loss_drowsy = criterion(drowsy_logits, labels)
                    loss_domain = domain_criterion(domain_logits, domain_labels)
                    loss = loss_drowsy + loss_domain
                    outputs = drowsy_logits
                else:
                    outputs = model(images, geo_features)
                    loss = criterion(outputs, labels)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if model_name.startswith("fusion_grl"):
                drowsy_logits, domain_logits = model(images, geo_features, alpha)
                loss_drowsy = criterion(drowsy_logits, labels)
                loss_domain = domain_criterion(domain_logits, domain_labels)
                loss = loss_drowsy + loss_domain
                outputs = drowsy_logits
            else:
                outputs = model(images, geo_features)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*train_correct/train_total:.1f}%")
    
    scheduler.step()
    
    # ── Validation ──
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, geo_features, labels, domain_labels in loaders["val"]:
            images = images.to(device, non_blocking=True)
            geo_features = geo_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            domain_labels = domain_labels.to(device, non_blocking=True)
            
            if scaler is not None:
                with autocast(device_type="cuda"):
                    if model_name.startswith("fusion_grl"):
                        drowsy_logits, domain_logits = model(images, geo_features, alpha)
                        loss_drowsy = criterion(drowsy_logits, labels)
                        loss_domain = domain_criterion(domain_logits, domain_labels)
                        loss = loss_drowsy + loss_domain
                        outputs = drowsy_logits
                    else:
                        outputs = model(images, geo_features)
                        loss = criterion(outputs, labels)
            else:
                if model_name.startswith("fusion_grl"):
                    drowsy_logits, domain_logits = model(images, geo_features, alpha)
                    loss_drowsy = criterion(drowsy_logits, labels)
                    loss_domain = domain_criterion(domain_logits, domain_labels)
                    loss = loss_drowsy + loss_domain
                    outputs = drowsy_logits
                else:
                    outputs = model(images, geo_features)
                    loss = criterion(outputs, labels)
            
            val_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    train_loss /= train_total
    train_acc = train_correct / train_total
    val_loss /= val_total
    val_acc = val_correct / val_total
    
    print(f"  {stage_name} Epoch {epoch}: "
          f"Train Loss={train_loss:.4f} Acc={100*train_acc:.2f}% | "
          f"Val Loss={val_loss:.4f} Acc={100*val_acc:.2f}%")
    
    return {
        "train_loss": train_loss, "train_acc": train_acc,
        "val_loss": val_loss, "val_acc": val_acc
    }


def _get_param_groups(model, model_name):
    if model_name == "geometric":
        return [{"params": model.parameters(), "lr": config.STAGE2_LR}]
    
    if model_name == "baseline":
        backbone_params = list(model.backbone.features.parameters())
        head_params = list(model.backbone.classifier.parameters())
    elif model_name.startswith("fusion"):
        backbone_params = list(model.features.parameters())
        
        # Collect parameters from all heads
        head_params = list(model.fusion_head.parameters())
        if hasattr(model, "domain_head"):
            head_params += list(model.domain_head.parameters())
    else:
        return [{"params": model.parameters(), "lr": config.STAGE2_LR}]
    
    return [
        {"params": backbone_params, "lr": config.STAGE2_BACKBONE_LR},
        {"params": head_params, "lr": config.STAGE2_LR},
    ]


def _update_history(history, epoch, stage, metrics, optimizer):
    history["epoch"].append(epoch)
    history["stage"].append(stage)
    history["train_loss"].append(metrics["train_loss"])
    history["train_acc"].append(metrics["train_acc"])
    history["val_loss"].append(metrics["val_loss"])
    history["val_acc"].append(metrics["val_acc"])
    history["lr"].append(optimizer.param_groups[0]["lr"])


def _save_history_csv(history, model_name):
    csv_path = os.path.join(config.RESULTS_DIR, f"{model_name}_history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "stage", "train_loss", "train_acc",
                          "val_loss", "val_acc", "lr"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i], history["stage"][i],
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['val_acc'][i]:.6f}",
                f"{history['lr'][i]:.8f}",
            ])
    print(f"  Training history saved: {csv_path}")
