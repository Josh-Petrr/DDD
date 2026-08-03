"""
evaluate.py — Comprehensive evaluation on the subject-disjoint test set.

Computes:
  - Accuracy, Precision, Recall, F1
  - APCER, BPCER, ACER (biometric metrics)
  - Confusion matrix
  - Confidence-tiered analysis (High/Medium/Low)
  - Per-image predictions with confidence scores
"""

import os
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score)

import config


def evaluate_model(model: torch.nn.Module, test_loader,
                   model_name: str,
                   device: torch.device = config.DEVICE) -> dict:
    """
    Run comprehensive evaluation on the test set.
    """
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []  # softmax probabilities
    
    with torch.no_grad():
        for images, geo_features, labels, domain_labels in test_loader:
            images = images.to(device, non_blocking=True)
            geo_features = geo_features.to(device, non_blocking=True)
            
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    if model_name.startswith("fusion_grl"):
                        outputs, _ = model(images, geo_features)
                    else:
                        outputs = model(images, geo_features)
            else:
                if model_name.startswith("fusion_grl"):
                    outputs, _ = model(images, geo_features)
                else:
                    outputs = model(images, geo_features)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # ── Core metrics ──
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary", pos_label=0)
    rec = recall_score(all_labels, all_preds, average="binary", pos_label=0)
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # ── APCER / BPCER / ACER ──
    drowsy_mask = all_labels == 0
    if drowsy_mask.sum() > 0:
        apcer = (all_preds[drowsy_mask] != 0).sum() / drowsy_mask.sum()
    else:
        apcer = 0.0
    
    non_drowsy_mask = all_labels == 1
    if non_drowsy_mask.sum() > 0:
        bpcer = (all_preds[non_drowsy_mask] != 1).sum() / non_drowsy_mask.sum()
    else:
        bpcer = 0.0
    
    acer = (apcer + bpcer) / 2.0
    
    # ── Confidence-tiered analysis ──
    max_probs = np.max(all_probs, axis=1)
    tiers = _compute_confidence_tiers(all_labels, all_preds, max_probs)
    
    # ── ROC curve data ──
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs[:, 0], pos_label=0)
    roc_auc = auc(fpr, tpr)
    
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        all_labels, all_probs[:, 0], pos_label=0
    )
    pr_auc = average_precision_score(all_labels, all_probs[:, 0], pos_label=0)
    
    results = {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
        "confusion_matrix": cm.tolist(),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confidence_tiers": tiers,
        "_roc": {"fpr": fpr, "tpr": tpr},
        "_pr": {"precision": pr_precision, "recall": pr_recall},
        "_all_probs": all_probs,
        "_all_labels": all_labels,
        "_all_preds": all_preds,
    }
    
    _print_results(results)
    
    return results


def evaluate_lstm_model(model: torch.nn.Module, test_loader,
                        model_name: str,
                        device: torch.device = config.DEVICE) -> dict:
    """
    Run comprehensive evaluation on the test set for the LSTM Sequence model.
    """
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    outputs = model(x)
            else:
                outputs = model(x)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # ── Core metrics ──
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary", pos_label=0)
    rec = recall_score(all_labels, all_preds, average="binary", pos_label=0)
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # ── APCER / BPCER / ACER ──
    drowsy_mask = all_labels == 0
    if drowsy_mask.sum() > 0:
        apcer = (all_preds[drowsy_mask] != 0).sum() / drowsy_mask.sum()
    else:
        apcer = 0.0
    
    non_drowsy_mask = all_labels == 1
    if non_drowsy_mask.sum() > 0:
        bpcer = (all_preds[non_drowsy_mask] != 1).sum() / non_drowsy_mask.sum()
    else:
        bpcer = 0.0
    
    acer = (apcer + bpcer) / 2.0
    
    max_probs = np.max(all_probs, axis=1)
    tiers = _compute_confidence_tiers(all_labels, all_preds, max_probs)
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 0], pos_label=0)
    roc_auc = auc(fpr, tpr)
    
    pr_precision, pr_recall, _ = precision_recall_curve(all_labels, all_probs[:, 0], pos_label=0)
    pr_auc = average_precision_score(all_labels, all_probs[:, 0], pos_label=0)
    
    results = {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
        "confusion_matrix": cm.tolist(),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confidence_tiers": tiers,
        "_roc": {"fpr": fpr, "tpr": tpr},
        "_pr": {"precision": pr_precision, "recall": pr_recall},
        "_all_probs": all_probs,
        "_all_labels": all_labels,
        "_all_preds": all_preds,
    }
    
    _print_results(results)
    return results


def _compute_confidence_tiers(labels, preds, max_probs):
    tiers = {}
    boundaries = [
        ("High (>0.9)", max_probs > 0.9),
        ("Medium (0.7–0.9)", (max_probs >= 0.7) & (max_probs <= 0.9)),
        ("Low (<0.7)", max_probs < 0.7),
    ]
    for tier_name, mask in boundaries:
        n = mask.sum()
        if n > 0:
            tier_acc = accuracy_score(labels[mask], preds[mask])
            tier_correct = (labels[mask] == preds[mask]).sum()
        else:
            tier_acc = 0.0
            tier_correct = 0
        
        tiers[tier_name] = {
            "count": int(n),
            "accuracy": float(tier_acc),
            "correct": int(tier_correct),
            "pct_of_total": float(n / len(labels)) if len(labels) > 0 else 0.0,
        }
    return tiers


def _print_results(results):
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {results['model_name'].upper()}")
    print(f"{'='*60}")
    print(f"\n  Accuracy:   {100*results['accuracy']:.2f}%")
    print(f"  Precision:  {100*results['precision']:.2f}%")
    print(f"  Recall:     {100*results['recall']:.2f}%")
    print(f"  F1-Score:   {100*results['f1_score']:.2f}%")
    print(f"  ROC AUC:    {results['roc_auc']:.4f}")
    print(f"  PR AUC:     {results['pr_auc']:.4f}")
    
    print(f"\n  Biometric Metrics:")
    print(f"    APCER (missed drowsiness): {100*results['apcer']:.2f}%")
    print(f"    BPCER (false alarm):       {100*results['bpcer']:.2f}%")
    print(f"    ACER (average):            {100*results['acer']:.2f}%")
    
    print(f"\n  Confusion Matrix:")
    cm = results['confusion_matrix']
    print(f"    {'':>15} Pred Drowsy  Pred Non-Drowsy")
    print(f"    {'True Drowsy':>15}  {cm[0][0]:>8}    {cm[0][1]:>8}")
    print(f"    {'True Non-Drowsy':>15}  {cm[1][0]:>8}    {cm[1][1]:>8}")
    
    print(f"\n  Confidence-Tiered Analysis:")
    for tier_name, tier_data in results["confidence_tiers"].items():
        print(f"    {tier_name:>20}: "
              f"{tier_data['count']:>5} images "
              f"({100*tier_data['pct_of_total']:.1f}%) — "
              f"Accuracy: {100*tier_data['accuracy']:.1f}%")


def save_results(all_results: dict, path: str = None):
    if path is None:
        path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    
    serializable = {}
    for model_name, results in all_results.items():
        serializable[model_name] = {
            k: v for k, v in results.items() 
            if not k.startswith("_")
        }
    
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to {path}")
