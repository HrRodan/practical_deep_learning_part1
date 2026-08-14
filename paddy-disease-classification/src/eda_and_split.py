import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from config import get_base_paths

def run_eda_and_splits():
    paths = get_base_paths()
    train_csv = paths["train_csv"]
    sample_sub_csv = paths["sample_submission_csv"]
    
    df = pd.read_csv(train_csv)
    sub_df = pd.read_csv(sample_sub_csv)
    
    print("=== DATASET OVERVIEW ===")
    print(f"Total training samples: {len(df)}")
    print(f"Total test samples: {len(sub_df)}")
    print(f"Number of classes: {df['label'].nunique()}")
    print("\nClass distribution in train set:")
    class_counts = df['label'].value_counts()
    for cls, cnt in class_counts.items():
        print(f"  {cls:<26}: {cnt:>5} ({cnt/len(df)*100:5.2f}%)")
        
    print("\nMetadata summary:")
    print(f"Unique varieties ({df['variety'].nunique()}): {sorted(df['variety'].unique().tolist())}")
    print(f"Age range: min={df['age'].min()}, max={df['age'].max()}, mean={df['age'].mean():.1f}, median={df['age'].median()}")
    
    # Check sample image resolutions
    sample_images = list(paths["train_images_dir"].glob("*/*.jpg"))[:50]
    sizes = set()
    for img_p in sample_images:
        with Image.open(img_p) as img:
            sizes.add(img.size)
    print(f"\nSample image dimensions (width, height): {sizes}")
    
    # Generate 5-Fold Stratified Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold
        
    # Check fold class balance
    print("\nFold distribution check (samples per fold):")
    print(df["fold"].value_counts().sort_index())
    
    # Save split
    splits_csv = paths["splits_csv"]
    df.to_csv(splits_csv, index=False)
    print(f"\nSaved 5-fold stratified split to: {splits_csv}")
    
    # Verify zero overlap between train and test IDs
    train_ids = set(df["image_id"])
    test_ids = set(sub_df["image_id"])
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage detected: IDs overlap between train and test!"
    print("Leakage check PASSED: No ID overlap between train and test.")

if __name__ == "__main__":
    run_eda_and_splits()
