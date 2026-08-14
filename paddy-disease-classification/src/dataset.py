import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, ColReader,
    Resize, aug_transforms, RandomResizedCrop, get_image_files
)
from .config import get_base_paths, TrainConfig

def create_or_load_splits(seed: int = 42, n_splits: int = 5) -> pd.DataFrame:
    """Loads train.csv, performs Stratified K-Fold split, saves/loads splits_5fold.csv."""
    paths = get_base_paths()
    splits_csv = paths["splits_csv"]

    if splits_csv.exists():
        df = pd.read_csv(splits_csv)
        return df

    # If not existing, read train.csv and generate splits
    train_csv = paths["train_csv"]
    df = pd.read_csv(train_csv)

    # Full relative image path inside train_images
    # Note: in train_images, images are categorized into label subdirectories
    # e.g., train_images/<label>/<image_id>
    df["image_path"] = df.apply(
        lambda row: str(paths["train_images_dir"] / row["label"] / row["image_id"]),
        axis=1
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    # Try saving locally if writable
    try:
        df.to_csv(splits_csv, index=False)
    except Exception:
        pass

    return df

def get_dataloaders(cfg: TrainConfig, custom_item_tfms=None, custom_batch_tfms=None):
    """Builds FastAI DataLoaders for a specific fold and image size."""
    paths = get_base_paths()
    df = create_or_load_splits(seed=cfg.seed, n_splits=cfg.n_splits)

    # Validation split mask for the requested fold
    val_fold = cfg.fold
    val_filter = lambda r: df.loc[r.name, "fold"] == val_fold if hasattr(r, 'name') and r.name in df.index else False

    # Default item and batch transforms
    item_tfms = custom_item_tfms or Resize(cfg.img_size, method='squish')
    batch_tfms = custom_batch_tfms or aug_transforms(
        mult=1.0,
        do_flip=True,
        flip_vert=False,
        max_rotate=10.0,
        min_zoom=0.9,
        max_zoom=1.1,
        max_lighting=0.2,
        max_warp=0.1
    )

    def get_x(row):
        return str(paths["train_images_dir"] / row["label"] / row["image_id"])

    def get_y(row):
        return row["label"]

    def splitter(df_input):
        train_idx = df_input[df_input["fold"] != val_fold].index.tolist()
        val_idx = df_input[df_input["fold"] == val_fold].index.tolist()
        return train_idx, val_idx

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=get_x,
        get_y=get_y,
        splitter=splitter,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )

    dls = dblock.dataloaders(df, bs=cfg.bs)
    return dls, df
