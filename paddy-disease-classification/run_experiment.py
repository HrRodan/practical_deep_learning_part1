import argparse
import sys
from pathlib import Path
import torch

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_base_paths, TrainConfig
from src.dataset import get_dataloaders
from src.train import train_model
from src.evaluate import evaluate_and_diagnose
from src.inference import predict_test_dataset, create_and_validate_submission
from fastai.vision.all import Resize, aug_transforms

def main():
    parser = argparse.ArgumentParser(description="Run Paddy Disease Classification Experiment")
    parser.add_argument("--arch", type=str, default="convnext_nano", help="Model architecture")
    parser.add_argument("--img_size", type=int, default=224, help="Square image size or resize dimension")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Base learning rate")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold index (0-4)")
    parser.add_argument("--tta", action="store_true", default=True, help="Use test-time augmentation")
    parser.add_argument("--exp_id", type=str, default="baseline_convnext_nano", help="Experiment ID for tracking")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Starting Experiment: {args.exp_id}")
    print(f"Architecture: {args.arch} | Resolution: {args.img_size} | Epochs: {args.epochs} | Fold: {args.fold}")
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    cfg = TrainConfig(
        arch=args.arch,
        img_size=args.img_size,
        bs=args.bs,
        epochs=args.epochs,
        lr=args.lr,
        fold=args.fold,
        save_name=f"{args.exp_id}_f{args.fold}"
    )

    # Train model
    learn, metrics = train_model(cfg)

    # Evaluate & Error analysis
    print("\n--- Running Validation Error Analysis ---")
    eval_results = evaluate_and_diagnose(learn, save_prefix=f"{args.exp_id}_f{args.fold}")

    # Test inference & Submission generation
    print("\n--- Generating Test Predictions & Submission ---")
    probs, vocab, image_ids = predict_test_dataset(learn, use_tta=args.tta)
    submission_path = create_and_validate_submission(
        probs, vocab, image_ids,
        submission_name=f"sub_{args.exp_id}_valacc_{metrics['val_acc']:.4f}.csv"
    )

    print("\n" + "=" * 60)
    print(f"EXPERIMENT {args.exp_id} COMPLETED SUCCESSFULLY")
    print(f"Validation Accuracy: {metrics['val_acc']:.4f}")
    print(f"Validation Error Rate: {metrics['val_err']:.4f}")
    print(f"Submission File: {submission_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
