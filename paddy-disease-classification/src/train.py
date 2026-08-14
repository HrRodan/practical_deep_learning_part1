import os
from pathlib import Path
import torch
from fastai.vision.all import (
    vision_learner, accuracy, error_rate, SaveModelCallback,
    EarlyStoppingCallback, CSVLogger, ShowGraphCallback,
    set_seed
)
from .config import get_base_paths, TrainConfig
from .dataset import get_dataloaders

def train_model(cfg: TrainConfig, custom_item_tfms=None, custom_batch_tfms=None):
    """Trains a model with specified config and returns the trained learner and metrics."""
    set_seed(cfg.seed, reproducible=True)
    paths = get_base_paths()
    
    print(f"--- Training {cfg.arch} on Fold {cfg.fold} (Resolution: {cfg.img_size}, Batch Size: {cfg.bs}) ---")
    dls, df = get_dataloaders(cfg, custom_item_tfms=custom_item_tfms, custom_batch_tfms=custom_batch_tfms)
    
    # FastAI vision learner with timm backbone support
    learn = vision_learner(
        dls,
        cfg.arch,
        metrics=[accuracy, error_rate],
        path=str(paths["output_dir"]),
        model_dir="models"
    )

    if cfg.use_fp16 and torch.cuda.is_available():
        learn = learn.to_fp16()

    save_name = cfg.save_name or f"{cfg.arch}_f{cfg.fold}_sz{cfg.img_size}"
    cbs = [
        SaveModelCallback(monitor='accuracy', fname=save_name, with_opt=True),
        CSVLogger(fname=str(paths["output_dir"] / f"history_{save_name}.csv"))
    ]

    # Fine-tune the backbone
    learn.fine_tune(cfg.epochs, base_lr=cfg.lr, cbs=cbs)

    # Load best model weights
    learn.load(save_name)

    # Validate best model
    val_loss, val_acc, val_err = learn.validate()
    print(f"Validation Results for {save_name} -> Accuracy: {val_acc:.4f}, Error Rate: {val_err:.4f}, Loss: {val_loss:.4f}")

    return learn, {"val_acc": val_acc, "val_err": val_err, "val_loss": val_loss, "save_name": save_name}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Paddy Disease Classifier")
    parser.add_argument("--arch", type=str, default="convnext_nano")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig(
        arch=args.arch,
        img_size=args.img_size,
        bs=args.bs,
        epochs=args.epochs,
        lr=args.lr,
        fold=args.fold
    )
    train_model(cfg)
