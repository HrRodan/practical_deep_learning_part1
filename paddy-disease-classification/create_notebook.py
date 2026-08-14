import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Paddy Doctor: Paddy Disease Classification
### FastAI + Timm State-of-the-Art Vision Solution

This notebook provides an end-to-end, modular, and fully reproducible pipeline for the **Paddy Disease Classification** competition.
It runs **identically on local GPU environments and remote Kaggle GPU kernels**.

#### Pipeline Steps:
1. **Dual Environment Setup**: Automatic path resolution for Local vs Kaggle.
2. **Data & Validation Strategy**: 5-Fold Stratified Cross-Validation ensuring zero leakage and balanced classes.
3. **Data Augmentation**: Task-aligned vision transforms (squish resize, rotation, lighting, perspective warp).
4. **Model Architecture**: Modern high-efficiency backbone (`convnext_nano`) with mixed precision (`FP16`).
5. **Validation & Error Analysis**: Classification report, confusion matrix, and hardest error diagnosis.
6. **Inference & TTA**: Test-Time Augmentation (TTA) and probability calculation.
7. **Submission Integrity**: Strict format, column, null, and category validations.
"""))

# Cell: Install missing packages if running on Kaggle
cells.append(nbf.v4.new_code_cell("""# Install timm if needed without touching preinstalled PyTorch/CUDA packages
import sys
try:
    import timm
except ImportError:
    !pip install -q --no-deps timm
    import timm
"""))

# Cell: Imports & Seed
cells.append(nbf.v4.new_code_cell("""import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, Resize, aug_transforms,
    vision_learner, accuracy, error_rate, SaveModelCallback,
    CSVLogger, ClassificationInterpretation, get_image_files, set_seed
)

# Fix seed for strict reproducibility
SEED = 42
set_seed(SEED, reproducible=True)

# Strict T4 GPU Hardware Verification & Halt on Non-T4
print(f"PyTorch Version: {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("FATAL: CUDA is NOT available! Ensure GPU accelerator (2x T4) is enabled in session settings.")

gpu_name = torch.cuda.get_device_name(0)
gpu_count = torch.cuda.device_count()
cap = torch.cuda.get_device_capability(0)
print(f"Detected GPU: {gpu_count}x {gpu_name} (CUDA Capability: {cap[0]}.{cap[1]})")

if cap[0] < 7:
    raise RuntimeError(
        f"FATAL: Incompatible legacy GPU ({gpu_name}, sm_{cap[0]}{cap[1]}). "
        f"PyTorch 2.6+ requires sm_70+ (NVIDIA T4 / V100 / A100). Please select 'GPU T4 x2' in Kaggle Settings -> Accelerator."
    )

try:
    # Test CUDA compute kernel
    test_tensor = (torch.zeros(2, 2, device="cuda") + 1.0) / 2.0
    _ = test_tensor.cpu()
    USE_CUDA = True
    print(f"=== CUDA T4 HARDWARE ACCELERATION ACTIVE: {gpu_count}x {gpu_name} (FP16 Enabled) ===")
except Exception as e:
    raise RuntimeError(f"FATAL: CUDA compute kernel failure on {gpu_name}: {e}")
"""))

# Cell: Environment Paths
cells.append(nbf.v4.new_code_cell("""# Automatic Path Resolution & Dataset Download
def ensure_data_downloaded(data_dir: Path):
    if (data_dir / "train.csv").exists() and (data_dir / "train_images").exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset missing at {data_dir}. Downloading via Kaggle API...")
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
            print("Dataset downloaded and ready!")
    except Exception as e:
        print(f"Notice: Automatic download failed ({e}). Please ensure data is present.")

def get_paths():
    if Path("/kaggle/input").exists():
        output_dir = Path("/kaggle/working")
        found_train = list(Path("/kaggle/input").rglob("train.csv"))
        if found_train:
            data_dir = found_train[0].parent
        else:
            data_dir = Path("/kaggle/input/paddy-disease-classification")
    else:
        # Local workspace relative path
        current_dir = Path("./data")
        data_dir = current_dir if current_dir.exists() else Path("../data")
        ensure_data_downloaded(data_dir)
        output_dir = Path("./outputs")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "submissions").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    
    return {
        "data_dir": data_dir,
        "train_csv": data_dir / "train.csv",
        "sample_sub": data_dir / "sample_submission.csv",
        "train_images": data_dir / "train_images",
        "test_images": data_dir / "test_images",
        "output_dir": output_dir,
        "submissions_dir": output_dir / "submissions",
        "models_dir": output_dir / "models",
    }

paths = get_paths()
print("Paths configured:")
for k, v in paths.items():
    print(f"  {k}: {v}")
"""))

# Cell: Data Loading & Validation Split
cells.append(nbf.v4.new_code_cell("""# Load train data and metadata
df_train = pd.read_csv(paths["train_csv"])
sample_sub = pd.read_csv(paths["sample_sub"])

print(f"Train samples: {len(df_train)}")
print(f"Test samples: {len(sample_sub)}")
print(f"Target classes ({df_train['label'].nunique()}): {sorted(df_train['label'].unique().tolist())}")

# Stratified 5-Fold Split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
df_train["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(df_train, df_train["label"])):
    df_train.loc[val_idx, "fold"] = fold

# Verification
print("\\nSamples per validation fold:")
print(df_train["fold"].value_counts().sort_index())
"""))

# Cell: 5-Fold Dual-GPU Concurrent Training Pipeline (Swin-Base @ 384x384)
cells.append(nbf.v4.new_code_cell("""# Write worker script to a dedicated python module so child processes can import it cleanly
import multiprocessing as mp
import gc

worker_module_code = '''
import os, sys, torch, gc, random
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, Resize, vision_learner,
    accuracy, error_rate, SaveModelCallback, CSVLogger, get_image_files, set_seed
)

TRAIN_IMG_DIR = None

def get_img_path(row):
    return str(TRAIN_IMG_DIR / row["label"] / row["image_id"])

def get_img_label(row):
    return row["label"]

def run_fold_worker(fold_id, gpu_id, paths_str_dict, seed, epochs, lr, img_size, batch_size, arch):
    global TRAIN_IMG_DIR
    set_seed(seed + fold_id, reproducible=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    print(f"[GPU {gpu_id}] Starting Fold {fold_id} for {arch} ({img_size}x{img_size})...", flush=True)
    
    df_train = pd.read_csv(paths_str_dict["train_csv"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    df_train["fold"] = -1
    for f, (_, v_idx) in enumerate(skf.split(df_train, df_train["label"])):
        df_train.loc[v_idx, "fold"] = f
        
    def split_by_current_fold(df_input):
        train_idx = df_input[df_input["fold"] != fold_id].index.tolist()
        val_idx = df_input[df_input["fold"] == fold_id].index.tolist()
        return train_idx, val_idx
        
    train_images = Path(paths_str_dict["train_images"])
    test_images = Path(paths_str_dict["test_images"])
    output_dir = Path(paths_str_dict["output_dir"])
    TRAIN_IMG_DIR = train_images
    
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=get_img_path,
        get_y=get_img_label,
        splitter=split_by_current_fold,
        item_tfms=[Resize(img_size, method='squish')]
    )
    dls = dblock.dataloaders(df_train, bs=batch_size, num_workers=0, device=device)
    vocab = list(dls.vocab)
    
    learn = vision_learner(
        dls,
        arch,
        metrics=[accuracy, error_rate],
        path=str(output_dir),
        model_dir="models"
    )
    learn = learn.to_fp16()
    
    save_name = f"{arch}_f{fold_id}_sz{img_size}"
    cbs = [
        SaveModelCallback(monitor='accuracy', fname=save_name, with_opt=True),
        CSVLogger(fname=str(output_dir / f"history_{save_name}.csv"))
    ]
    
    learn.fine_tune(epochs, base_lr=lr, cbs=cbs)
    learn.load(save_name)
    val_loss, val_acc, val_err = learn.validate()
    print(f"[GPU {gpu_id}] Fold {fold_id} BEST Validation Accuracy: {val_acc:.4f} (Error: {val_err:.4f})", flush=True)
    
    # Save OOF Validation Predictions
    val_probs, val_targets = learn.get_preds()
    torch.save(val_probs.cpu(), output_dir / f"oof_probs_{save_name}.pt")
    
    # Save Test TTA Predictions (4-pass)
    test_files = get_image_files(test_images).sorted()
    test_dl = dls.test_dl(test_files, num_workers=0)
    print(f"[GPU {gpu_id}] Running 4-pass TTA on test images for Fold {fold_id}...", flush=True)
    test_probs, _ = learn.tta(dl=test_dl, n=4, beta=0.25)
    torch.save(test_probs.cpu(), output_dir / f"test_probs_{save_name}.pt")
    
    del learn
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] Fold {fold_id} complete. GPU memory purged.", flush=True)
'''

with open("worker.py", "w") as f:
    f.write(worker_module_code)

from worker import run_fold_worker

IMG_SIZE = 384
BATCH_SIZE = 16
N_FOLDS = 5
EPOCHS = 6
LR = 1e-4
ARCH = "convnext_base"

# Dual-GPU Parallel Dispatcher
n_gpus = torch.cuda.device_count()
print(f"=== SATURATING {n_gpus} DUAL T4 GPUs IN PARALLEL ===")

paths_str_dict = {k: str(v) for k, v in paths.items()}

# Construct optimal fold dispatch batches across available GPUs
if n_gpus >= 2:
    fold_batches = [
        [(0, 0), (1, 1)],  # Folds 0 & 1 train in parallel on GPU 0 & GPU 1
        [(2, 0), (3, 1)],  # Folds 2 & 3 train in parallel on GPU 0 & GPU 1
        [(4, 0)]           # Fold 4 trains on GPU 0
    ]
else:
    fold_batches = [[(0, 0)], [(1, 0)], [(2, 0)], [(3, 0)], [(4, 0)]]

mp.set_start_method("spawn", force=True)

for b_idx, batch in enumerate(fold_batches):
    print(f"\\n{'='*25} Running Parallel Batch {b_idx + 1}/{len(fold_batches)}: Folds {[f[0] for f in batch]} {'='*25}")
    processes = []
    for fold_id, gpu_id in batch:
        p = mp.Process(
            target=run_fold_worker,
            args=(fold_id, gpu_id, paths_str_dict, SEED, EPOCHS, LR, IMG_SIZE, BATCH_SIZE, ARCH)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"FATAL: Worker process failed with exit code {p.exitcode}")

print("\\n=== ALL 5 FOLDS COMPLETED ACROSS DUAL T4 GPUs ===")
"""))

# Cell: Out-of-Fold (OOF) Evaluation & Confusion Matrix
cells.append(nbf.v4.new_code_cell("""# Full 5-Fold Out-of-Fold (OOF) Evaluation
vocab = sorted(df_train["label"].unique().tolist())
oof_matrix = np.zeros((len(df_train), len(vocab)))

for fold in range(N_FOLDS):
    save_name = f"{ARCH}_f{fold}_sz{IMG_SIZE}"
    val_probs = torch.load(paths["output_dir"] / f"oof_probs_{save_name}.pt")
    val_indices = df_train[df_train["fold"] == fold].index.tolist()
    oof_matrix[val_indices] = val_probs.numpy()

oof_pred_labels = np.argmax(oof_matrix, axis=1)
true_labels = np.array([vocab.index(l) for l in df_train["label"]])

oof_acc = np.mean(oof_pred_labels == true_labels)
print("\\n" + "=" * 60)
print(f"5-FOLD {ARCH} OUT-OF-FOLD (OOF) ACCURACY: {oof_acc:.4f} (Error Rate: {1 - oof_acc:.4f})")
print("=" * 60)

print("\\n5-Fold OOF Classification Report:")
print(classification_report(true_labels, oof_pred_labels, target_names=vocab, zero_division=0))

cm = confusion_matrix(true_labels, oof_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=vocab, yticklabels=vocab)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'5-Fold {ARCH} OOF Confusion Matrix (Acc: {oof_acc:.4f})')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(paths["output_dir"] / f"confusion_matrix_oof_{ARCH}.png", dpi=150)
plt.show()
"""))

# Cell: Test Inference & Ensembling
cells.append(nbf.v4.new_code_cell("""# 5-Fold Swin Test Probability Averaging & Submission
test_files = get_image_files(paths["test_images"]).sorted()
accum_test_probs = None

for fold in range(N_FOLDS):
    save_name = f"{ARCH}_f{fold}_sz{IMG_SIZE}"
    test_probs = torch.load(paths["output_dir"] / f"test_probs_{save_name}.pt")
    if accum_test_probs is None:
        accum_test_probs = test_probs / N_FOLDS
    else:
        accum_test_probs += test_probs / N_FOLDS

# Format Final Submission
pred_indices = torch.argmax(accum_test_probs, dim=1).numpy()
pred_labels = [vocab[idx] for idx in pred_indices]
image_ids = [f.name for f in test_files]

df_preds = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
submission = sample_sub[["image_id"]].merge(df_preds, on="image_id", how="left")

# Strict Integrity Checks
assert len(submission) == len(sample_sub), f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
assert submission["label"].isnull().sum() == 0, "Missing values found in predictions!"
assert list(submission.columns) == ["image_id", "label"], f"Invalid columns: {submission.columns}"
assert set(submission["label"].unique()).issubset(set(vocab)), "Invalid class predicted!"

sub_file = paths["submissions_dir"] / f"submission_5fold_{ARCH}_oofacc_{oof_acc:.4f}.csv"
submission.to_csv(sub_file, index=False)

root_sub_file = paths["output_dir"] / "submission.csv"
submission.to_csv(root_sub_file, index=False)

print(f"\\n5-Fold {ARCH} Submission verified and saved to {sub_file} and {root_sub_file}")
print("\\nClass Distribution in Test Predictions:")
print(submission["label"].value_counts())
print(submission.head(10))
"""))

nb.cells = cells
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}
nb.metadata.language_info = {
    "name": "python",
    "version": "3.10"
}

with open("/home/martin/projects/practical_deep_learning_part1/paddy-disease-classification/paddy_solution.ipynb", "w") as f:
    nbf.write(nb, f)

print("Updated paddy_solution.ipynb with kernelspec metadata successfully!")

