"""
generate_report.py — Generate all evaluation plots and summary report.

Runs evaluation on all 4 models, produces comparison tables and plots,
and saves everything to the results/ directory.

Usage:
    python generate_report.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DOCS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_pipeline import config
from ml_pipeline.dataset import create_dataloaders, DrowsinessDataset
from ml_pipeline.data_split import load_splits
from ml_pipeline.models import get_model, count_parameters
from ml_pipeline.evaluate import evaluate_model, save_results
from ml_pipeline.gradcam import generate_gradcam_grid


# ── Plot styling ──
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
COLORS = {"baseline": "#2196F3", "geometric": "#FF9800", "fusion": "#4CAF50", "fusion_grl": "#9C27B0"}


def plot_training_curves(model_names: list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name in model_names:
        csv_path = os.path.join(config.RESULTS_DIR, f"{name}_history.csv")
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path)
        base_name = name.rsplit("_", 1)[0]
        color = COLORS.get(base_name, "gray")
        
        # Loss
        axes[0].plot(df["epoch"], df["train_loss"], 
                     label=f"{name} (train)", color=color, linestyle="-")
        axes[0].plot(df["epoch"], df["val_loss"],
                     label=f"{name} (val)", color=color, linestyle="--")
        
        # Accuracy
        axes[1].plot(df["epoch"], df["train_acc"] * 100,
                     label=f"{name} (train)", color=color, linestyle="-")
        axes[1].plot(df["epoch"], df["val_acc"] * 100,
                     label=f"{name} (val)", color=color, linestyle="--")
    
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(fontsize=8)
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend(fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved: {path}")


def plot_confusion_matrices(all_results: dict):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, (name, results) in zip(axes, all_results.items()):
        cm = np.array(results["confusion_matrix"])
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=config.CLASS_NAMES,
                    yticklabels=config.CLASS_NAMES,
                    cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name.upper()}\nAcc: {100*results['accuracy']:.1f}%")
    
    plt.suptitle("Confusion Matrices — Subject-Disjoint Test Set",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    path = os.path.join(config.RESULTS_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrices saved: {path}")


def plot_roc_curves(all_results: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for name, results in all_results.items():
        if "_roc" not in results:
            continue
        fpr = results["_roc"]["fpr"]
        tpr = results["_roc"]["tpr"]
        auc_score = results["roc_auc"]
        base_name = name.rsplit("_", 1)[0]
        
        ax.plot(fpr, tpr, color=COLORS.get(base_name, "gray"), linewidth=2,
                label=f"{name} (AUC = {auc_score:.4f})")
    
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Subject-Disjoint Test Set")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curves saved: {path}")


def plot_confidence_analysis(all_results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    tier_names = ["High (>0.9)", "Medium (0.7–0.9)", "Low (<0.7)"]
    x = np.arange(len(tier_names))
    width = 0.2
    
    for i, (name, results) in enumerate(all_results.items()):
        tiers = results["confidence_tiers"]
        counts = [tiers[t]["pct_of_total"] * 100 for t in tier_names]
        base_name = name.rsplit("_", 1)[0]
        axes[0].bar(x + i * width, counts, width, 
                    label=name, color=COLORS.get(base_name, "gray"))
    
    axes[0].set_xlabel("Confidence Tier")
    axes[0].set_ylabel("% of Predictions")
    axes[0].set_title("Prediction Distribution by Confidence")
    axes[0].set_xticks(x + width*1.5)
    axes[0].set_xticklabels(tier_names, fontsize=9)
    axes[0].legend()
    
    for i, (name, results) in enumerate(all_results.items()):
        tiers = results["confidence_tiers"]
        accs = [tiers[t]["accuracy"] * 100 for t in tier_names]
        base_name = name.rsplit("_", 1)[0]
        axes[1].bar(x + i * width, accs, width,
                    label=name, color=COLORS.get(base_name, "gray"))
    
    axes[1].set_xlabel("Confidence Tier")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy by Confidence Tier")
    axes[1].set_xticks(x + width*1.5)
    axes[1].set_xticklabels(tier_names, fontsize=9)
    axes[1].legend()
    axes[1].set_ylim(0, 105)
    
    plt.suptitle("Confidence-Tiered Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    path = os.path.join(config.RESULTS_DIR, "confidence_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confidence analysis saved: {path}")


def plot_confidence_distributions(all_results: dict):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, (name, results) in zip(axes, all_results.items()):
        if "_all_probs" not in results:
            continue
        
        probs = results["_all_probs"]
        labels = results["_all_labels"]
        max_probs = np.max(probs, axis=1)
        
        drowsy_conf = max_probs[labels == 0]
        non_drowsy_conf = max_probs[labels == 1]
        
        ax.hist(drowsy_conf, bins=30, alpha=0.6, label="Drowsy", 
                color="#F44336", density=True)
        ax.hist(non_drowsy_conf, bins=30, alpha=0.6, label="Non Drowsy",
                color="#2196F3", density=True)
        
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        ax.set_title(f"{name.upper()}")
        ax.legend()
    
    plt.suptitle("Prediction Confidence Distribution by Class",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    path = os.path.join(config.RESULTS_DIR, "confidence_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confidence distributions saved: {path}")


def create_summary_table(all_results: dict):
    rows = []
    for name, results in all_results.items():
        rows.append({
            "Model": name.upper(),
            "Accuracy (%)": f"{100*results['accuracy']:.2f}",
            "Precision (%)": f"{100*results['precision']:.2f}",
            "Recall (%)": f"{100*results['recall']:.2f}",
            "F1-Score (%)": f"{100*results['f1_score']:.2f}",
            "ROC AUC": f"{results['roc_auc']:.4f}",
            "APCER (%)": f"{100*results['apcer']:.2f}",
            "BPCER (%)": f"{100*results['bpcer']:.2f}",
            "ACER (%)": f"{100*results['acer']:.2f}",
        })
    
    df = pd.DataFrame(rows)
    path = os.path.join(config.RESULTS_DIR, "evaluation_summary.csv")
    df.to_csv(path, index=False)
    
    print(f"\n  Summary table saved: {path}")
    print(f"\n{df.to_string(index=False)}")


def main():
    print("=" * 60)
    print("GENERATING EVALUATION REPORT")
    print("=" * 60)
    
    if not os.path.exists(config.SPLITS_FILE):
        print("ERROR: splits.json not found. Run train_all.py first.")
        sys.exit(1)
    
    splits = load_splits()
    geo_df = pd.read_csv(config.FEATURES_FILE)
    loaders, num_domains = create_dataloaders(splits, geo_df)
    
    model_names = ["fusion_grl_4"]
    all_results = {}
    
    arch_mapping = {
        "baseline_3": "baseline",
        "geometric_3": "geometric",
        "fusion_3": "fusion",
        "fusion_grl_3": "fusion_grl",
        "fusion_grl_4": "fusion_grl_v4"
    }
    
    for name in model_names:
        checkpoint = os.path.join(config.CHECKPOINTS_DIR, f"{name}_best.pth")
        if not os.path.exists(checkpoint):
            print(f"\n⚠ Skipping {name}: checkpoint not found at {checkpoint}")
            continue
        
        print(f"\nEvaluating {name.upper()}...")
        arch_name = arch_mapping.get(name, name)
        model = get_model(arch_name, pretrained=False, num_domains=num_domains)
        model.load_state_dict(torch.load(checkpoint, map_location=config.DEVICE,
                                          weights_only=True))
        
        results = evaluate_model(model, loaders["test"], name)
        all_results[name] = results
        
        if arch_name in ["baseline", "fusion", "fusion_grl"]:
            print(f"  Generating Grad-CAM for {name}...")
            # Note: For fusion_grl, generate_gradcam_grid will pass (images, geo_features) 
            # and may need alpha=1.0 depending on how gradcam is implemented.
            try:
                generate_gradcam_grid(model, loaders["test"], name)
            except Exception as e:
                print(f"  Grad-CAM failed for {name}: {e}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if not all_results:
        print("No models evaluated. Exiting.")
        return
    
    print("\n" + "─" * 40)
    print("Generating plots...")
    print("─" * 40)
    
    plot_training_curves(list(all_results.keys()))
    plot_confusion_matrices(all_results)
    plot_roc_curves(all_results)
    plot_confidence_analysis(all_results)
    plot_confidence_distributions(all_results)
    create_summary_table(all_results)
    
    save_results(all_results)
    
    print(f"\n{'='*60}")
    print(f"REPORT COMPLETE — All outputs in: {config.RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
