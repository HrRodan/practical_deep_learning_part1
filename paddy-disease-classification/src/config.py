from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch

def ensure_data_downloaded(data_dir: Path):
    """Downloads competition data via Kaggle API if not already present."""
    if (data_dir / "train.csv").exists() and (data_dir / "train_images").exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset missing at {data_dir}. Downloading from Kaggle...")
    try:
        import zipfile
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("paddy-disease-classification", path=str(data_dir))
        zip_path = data_dir / "paddy-disease-classification.zip"
        if zip_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            zip_path.unlink()
            print("Dataset ready!")
    except Exception as e:
        print(f"Notice: Automatic download failed ({e}). Please ensure data is placed in {data_dir}")

def get_base_paths():
    """Detects whether running locally or on Kaggle and sets paths accordingly."""
    if Path("/kaggle/input").exists():
        output_dir = Path("/kaggle/working")
        found_train = list(Path("/kaggle/input").rglob("train.csv"))
        if found_train:
            data_dir = found_train[0].parent
        else:
            data_dir = Path("/kaggle/input/paddy-disease-classification")
    else:
        # Local path relative to project
        current_file_dir = Path(__file__).resolve().parent
        data_dir = current_file_dir.parent / "data"
        output_dir = current_file_dir.parent / "outputs"
        ensure_data_downloaded(data_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = output_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": data_dir,
        "train_csv": data_dir / "train.csv",
        "sample_submission_csv": data_dir / "sample_submission.csv",
        "train_images_dir": data_dir / "train_images",
        "test_images_dir": data_dir / "test_images",
        "splits_csv": data_dir / "splits_5fold.csv",
        "output_dir": output_dir,
        "submissions_dir": submissions_dir,
        "models_dir": models_dir,
    }

@dataclass
class TrainConfig:
    arch: str = "convnext_nano"  # timm or fastai model architecture
    img_size: int = 224  # image resolution (or tuple for rectangular)
    bs: int = 32  # batch size
    epochs: int = 5  # epochs for fine_tuning
    lr: float = 2e-3  # base learning rate
    fold: int = 0  # validation fold index (0 to 4)
    n_splits: int = 5  # number of k-fold splits
    seed: int = 42  # random seed
    use_fp16: bool = True  # mixed precision
    pretrained: bool = True
    save_name: Optional[str] = None
    tta: bool = True  # test-time augmentation
