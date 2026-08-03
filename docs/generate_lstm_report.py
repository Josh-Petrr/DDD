"""
generate_lstm_report.py — Generate evaluation report for the LSTM Sequence Model.
"""

import os
import sys
import json
import torch
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DOCS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_pipeline import config
from ml_pipeline.dataset_lstm import create_lstm_dataloaders
from ml_pipeline.models import get_model
from ml_pipeline.evaluate import evaluate_lstm_model, save_results

# Reuse plot functions from generate_report
from generate_report import plot_training_curves, plot_confusion_matrices, plot_roc_curves, plot_confidence_analysis

def main():
    print(f"============================================================")
    print(f"GENERATING EVALUATION REPORT FOR LSTM")
    print(f"============================================================")
    
    # 1. Create DataLoaders
    # Validation/Test use stride=seq_len (30) for zero overlap
    loaders = create_lstm_dataloaders(seq_len=30, stride=30)
    
    # 2. Evaluate model
    arch_name = "lstm"
    model_name = "lstm"
    all_results = {}
    
    print(f"\nEvaluating {model_name.upper()}...")
    
    checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, "lstm_best.pth")
    if not os.path.exists(checkpoint_path):
        print(f"⚠ ERROR: checkpoint not found at {checkpoint_path}")
        return
        
    model = get_model(arch_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True))
    
    results = evaluate_lstm_model(model, loaders["test"], model_name)
    all_results[model_name] = results
    
    # 3. Generate plots and outputs
    print("\n" + "-"*40)
    print("Generating plots...")
    print("-"*40)
    
    # Pass as lists/dicts matching what generate_report expects
    plot_training_curves([model_name])
    plot_confusion_matrices(all_results)
    plot_roc_curves(all_results)
    plot_confidence_analysis(all_results)
    
    # 4. Save tabular summary
    summary_data = []
    for m_name, res in all_results.items():
        summary_data.append({
            "Model": m_name.upper(),
            "Accuracy (%)": round(res["accuracy"] * 100, 2),
            "Precision (%)": round(res["precision"] * 100, 2),
            "Recall (%)": round(res["recall"] * 100, 2),
            "F1-Score (%)": round(res["f1_score"] * 100, 2),
            "ROC AUC": round(res["roc_auc"], 4),
            "APCER (%)": round(res["apcer"] * 100, 2),
            "BPCER (%)": round(res["bpcer"] * 100, 2),
            "ACER (%)": round(res["acer"] * 100, 2),
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(config.RESULTS_DIR, "lstm_evaluation_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Summary table saved: {summary_path}\n")
    print(df_summary.to_string(index=False))
    
    save_results(all_results, os.path.join(config.RESULTS_DIR, "lstm_evaluation_results.json"))
    
    print(f"\n============================================================")
    print(f"REPORT COMPLETE — All outputs in: {config.RESULTS_DIR}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
