# Working with the Kaggle CLI: Agent Guide

This document provides guidelines, workflows, and common pitfall solutions for AI agents and developers interacting with Kaggle via `uv run kaggle`.

---

## 1. Overview & Setup

- **Invocation**: Always execute Kaggle CLI commands through `uv run kaggle`.
- **Authentication**: Kaggle API credentials are automatically sourced from `~/.kaggle/kaggle.json` or `~/.kaggle` configuration.
- **Config View**: Verify user credentials with:
  ```bash
  uv run kaggle config view
  ```

---

## 2. Kaggle Kernels Workflow

### Step 1: Initialize Kernel Metadata
Generate a metadata template in the target folder:
```bash
uv run kaggle kernels init -p .
```
This creates `kernel-metadata.json`.

### Step 2: Configure `kernel-metadata.json`
Example production schema:
```json
{
  "id": "<username>/<kernel-slug>",
  "title": "Notebook Title",
  "code_file": "notebook-name.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_tpu": "false",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

> **Note**: Boolean fields in Kaggle kernel metadata (`is_private`, `enable_gpu`, `enable_internet`) should be formatted as string booleans (`"true"` / `"false"`).

### Step 3: Push Notebook to Kaggle
```bash
uv run kaggle kernels push -p .
```

### Step 4: Monitor Execution Status
Poll the status until `KernelWorkerStatus.COMPLETE` or `KernelWorkerStatus.ERROR`:
```bash
uv run kaggle kernels status <username>/<kernel-slug>
```

### Step 5: Debugging & Reading Logs
If the status reports `KernelWorkerStatus.ERROR`, inspect the remote execution stack trace immediately:
```bash
uv run kaggle kernels logs <username>/<kernel-slug>
```

### Step 6: Download Remote Outputs
To inspect generated files/artifacts:
```bash
uv run kaggle kernels output <username>/<kernel-slug> -p /tmp/kaggle_output
```

---

## 3. Common Kaggle Pitfalls & Solutions

### A. IPython Magic Syntax (`SyntaxError`)
- **Problem**: Placing IPython magics on the same line as a Python `if` statement (e.g., `if iskaggle: !pip install ...`) breaks Papermill (Kaggle's headless runner) with `SyntaxError: invalid syntax`.
- **Fix**: Place `!` magics on their own indented line:
  ```python
  if iskaggle:
      !pip install -Uqq 'ddgs>=6.2'
  ```

### B. PyTorch & CUDA Driver Collisions
- **Problem**: Running `!pip install -Uqq fastai` or upgrading `torch` inside a Kaggle notebook installs PyPI CUDA wheels that shadow Kaggle's system-installed PyTorch drivers, resulting in `AcceleratorError: CUDA error: no kernel image is available for execution on the device`.
- **Fix**: Kaggle environments pre-install hardware-matched `torch` and `fastai`. Only install specific missing packages (e.g., `ddgs`), and avoid `--upgrade` or reinstalling core ML frameworks unless strictly required.

### C. Resource Selection (CPU vs GPU)
- **Rule of Thumb**: For small models/datasets (e.g. ResNet18 fine-tuning on <100 small images), set `"enable_gpu": "false"`. CPU execution runs in seconds, avoids GPU queue times, and eliminates CUDA driver architecture mismatches.

### D. Flaky Image Downloads (`UnidentifiedImageError`)
- **Problem**: Single-image queries (`max_images=1`) can return hotlink-protected or dead URLs.
- **Fix**: Fetch `max_images=10` and use a retry loop:
  ```python
  urls = search_images('bird photos', max_images=10)
  for url in urls:
      try:
          download_url(url, dest, show_progress=False)
          im = Image.open(dest)
          im.to_thumb(256, 256)
          break
      except Exception:
          continue
  ```

---

## 4. Useful Kaggle CLI Reference Commands

| Action | Command |
| :--- | :--- |
| **List Kernels** | `uv run kaggle kernels list --user <username>` |
| **Pull Kernel** | `uv run kaggle kernels pull <username>/<kernel-slug> -p . -m` |
| **List Datasets** | `uv run kaggle datasets list -s <query>` |
| **Download Dataset** | `uv run kaggle datasets download -d <owner>/<dataset-name> --unzip` |
| **List Competitions**| `uv run kaggle competitions list` |
