from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd
import torch
from fastai.vision.all import Learner, get_image_files
from .config import get_base_paths

def predict_test_dataset(
    learner: Learner,
    use_tta: bool = True,
    n_tta: int = 4
) -> (np.ndarray, List[str], List[str]):
    """Generates predictions and probabilities on test_images using the given learner."""
    paths = get_base_paths()
    test_files = get_image_files(paths["test_images_dir"]).sorted()
    
    test_dl = learner.dls.test_dl(test_files)
    
    if use_tta:
        probs, _ = learner.tta(dl=test_dl, n=n_tta, beta=0.25)
    else:
        probs, _ = learner.get_preds(dl=test_dl)
        
    class_vocab = list(learner.dls.vocab)
    image_ids = [f.name for f in test_files]
    
    return probs.numpy(), class_vocab, image_ids

def create_and_validate_submission(
    probs: np.ndarray,
    vocab: List[str],
    image_ids: List[str],
    submission_name: str = "submission.csv"
) -> Path:
    """Formats predictions, aligns with sample_submission.csv, verifies integrity, and saves."""
    paths = get_base_paths()
    sample_sub = pd.read_csv(paths["sample_submission_csv"])
    
    pred_indices = np.argmax(probs, axis=1)
    pred_labels = [vocab[idx] for idx in pred_indices]
    
    pred_df = pd.DataFrame({
        "image_id": image_ids,
        "label": pred_labels
    })
    
    # Merge with sample_sub to ensure identical ordering
    final_sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
    
    # Verification checks
    assert len(final_sub) == len(sample_sub), f"Row count mismatch: {len(final_sub)} vs {len(sample_sub)}"
    assert final_sub["label"].isnull().sum() == 0, f"Found {final_sub['label'].isnull().sum()} missing predictions!"
    assert list(final_sub.columns) == ["image_id", "label"], f"Invalid columns: {final_sub.columns}"
    
    valid_classes = set(vocab)
    invalid_preds = set(final_sub["label"].unique()) - valid_classes
    assert len(invalid_preds) == 0, f"Found invalid predicted class names: {invalid_preds}"
    
    sub_path = paths["submissions_dir"] / submission_name
    final_sub.to_csv(sub_path, index=False)
    print(f"Verified submission successfully saved to: {sub_path}")
    print(f"Sample predictions:\n{final_sub.head()}")
    return sub_path

def ensemble_predictions(
    learners_and_weights: List[tuple],
    use_tta: bool = True,
    submission_name: str = "ensemble_submission.csv"
) -> Path:
    """Averages predicted probabilities across multiple models with optional weighting."""
    combined_probs = None
    ref_vocab = None
    ref_image_ids = None
    
    total_weight = sum(w for _, w in learners_and_weights)
    
    for learn, weight in learners_and_weights:
        probs, vocab, image_ids = predict_test_dataset(learn, use_tta=use_tta)
        
        if ref_vocab is None:
            ref_vocab = vocab
            ref_image_ids = image_ids
            combined_probs = np.zeros_like(probs)
        else:
            assert ref_vocab == vocab, "Learners must have identical vocabulary order!"
            assert ref_image_ids == image_ids, "Learners must process identical test order!"
            
        combined_probs += (weight / total_weight) * probs
        
    return create_and_validate_submission(combined_probs, ref_vocab, ref_image_ids, submission_name)
