import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from fastai.vision.all import Learner, ClassificationInterpretation
from sklearn.metrics import classification_report, confusion_matrix
from .config import get_base_paths

def evaluate_and_diagnose(learn: Learner, save_prefix: str = "eval") -> dict:
    """Computes error analysis, confusion matrix, and top losses."""
    paths = get_base_paths()
    eval_dir = paths["output_dir"] / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    interp = ClassificationInterpretation.from_learner(learn)
    
    # Classification Report
    val_probs, val_targets = learn.get_preds()
    val_preds = torch.argmax(val_probs, dim=1).numpy()
    val_targets = val_targets.numpy()
    vocab = list(learn.dls.vocab)
    
    report_dict = classification_report(
        val_targets,
        val_preds,
        target_names=vocab,
        output_dict=True,
        zero_division=0
    )
    
    report_text = classification_report(
        val_targets,
        val_preds,
        target_names=vocab,
        zero_division=0
    )
    print("=== Classification Report ===")
    print(report_text)
    
    # Save Report
    with open(eval_dir / f"{save_prefix}_report.txt", "w") as f:
        f.write(report_text)
        
    # Confusion Matrix
    cm = confusion_matrix(val_targets, val_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=vocab, yticklabels=vocab, ax=ax)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {save_prefix}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = eval_dir / f"{save_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    print(f"Confusion matrix saved to {cm_path}")
    
    # Most Confused Pairs
    most_confused = interp.most_confused(min_val=2)
    print("\n=== Most Confused Class Pairs (min 2 errors) ===")
    for true_cls, pred_cls, count in most_confused:
        print(f"True: {true_cls:<25} | Predicted: {pred_cls:<25} | Count: {count}")
        
    return {
        "report_dict": report_dict,
        "most_confused": most_confused,
        "cm_path": str(cm_path)
    }
