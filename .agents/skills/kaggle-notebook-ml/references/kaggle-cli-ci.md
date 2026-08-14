# Kaggle CLI and Remote CI

Use this reference whenever creating, pushing, monitoring, pulling, or deleting a Kaggle notebook. Run every Kaggle CLI command through `uv run kaggle`.

## Preflight

```bash
uv run kaggle config view
```

Credentials are expected in `~/.kaggle/kaggle.json`. Do not print, copy, commit, or expose credentials. Stop and ask the user if authentication, competition acceptance, or data permission is missing.

## Notebook package

Keep each remote notebook in one folder:

```text
<notebook-folder>/
  notebook.ipynb
  kernel-metadata.json
```

Create metadata once:

```bash
uv run kaggle kernels init -p <notebook-folder>
```

Use string booleans in `kernel-metadata.json`. For a T4 GPU notebook, use this shape and fill in the actual owner, slug, title, code filename, and data sources:

```json
{
  "id": "<username>/<kernel-slug>",
  "title": "Notebook Title",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "machine_shape": "nvidiaTeslaT4x2",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

Use only the minimum required data and model sources. Keep the notebook private unless the user explicitly requests otherwise.

## GPU contract

For a GPU workload, set `enable_gpu` and `machine_shape`, then pass the matching accelerator explicitly when pushing:

```bash
uv run kaggle kernels push -p <notebook-folder> --accelerator nvidiaTeslaT4x2
```

Use `nvidiaTeslaT4` only when a single T4 is sufficient. The notebook must print `torch.cuda.device_count()` and `torch.cuda.get_device_name(i)` for every allocated GPU, and fail immediately when CUDA is unavailable or the allocated GPU is not an NVIDIA T4. Do not silently fall back to CPU or a legacy P100 for a T4-required run.

Never upgrade or overwrite Kaggle's `torch` or `torchvision`. Install only missing libraries, preferably with no dependency resolution, for example:

```python
!pip install -q --no-deps timm
```

Put an IPython `!` magic on its own indented line; never combine it with a Python statement on the same line.

## Remote CI loop

Treat each Kaggle run as a remote integration test:

1. Create or update the notebook with the scaffold and validate its structure locally.
2. Run a local `DEBUG` smoke test only; do not treat it as proof of remote compatibility.
3. Push with the explicit accelerator when the workload requires one.
4. Poll the exact remote kernel ID until it is `COMPLETE` or `ERROR`.
5. On `ERROR`, retrieve logs; fix the smallest confirmed issue and repeat from step 1.
6. On `COMPLETE`, list outputs and download required artifacts, then validate expected files, metrics, and submission contents.
7. Record the run URL/ID, metadata, accelerator, hardware report, status, artifacts, and conclusion in `decision_log.md`.

Use these commands:

```bash
uv run kaggle kernels status <username>/<kernel-slug>
uv run kaggle kernels logs <username>/<kernel-slug>
uv run kaggle kernels files <username>/<kernel-slug>
uv run kaggle kernels output <username>/<kernel-slug> -p <output-folder>
uv run kaggle kernels output <username>/<kernel-slug> --file-pattern ".*\\.png$"
```

To pull the user's own remote notebook and metadata, optionally at a specific version:

```bash
uv run kaggle kernels pull <username>/<kernel-slug>[/version] -p <notebook-folder> -m
```

Delete a remote kernel only after explicit user authorization and after confirming its exact owner and slug:

```bash
uv run kaggle kernels delete <username>/<kernel-slug> -y
```

## Inputs and secrets

Resolve mounted inputs dynamically because Kaggle mount paths vary:

```python
from pathlib import Path

matches = list(Path("/kaggle/input").rglob("train.csv"))
if not matches:
    raise FileNotFoundError("Could not find train.csv under /kaggle/input")
data_dir = matches[0].parent
```

When running outside Kaggle and required competition data is absent, use the authenticated Kaggle API as a guarded fallback, for example `KaggleApi().competition_download_files(...)`. Do this only for authorized competition data and only if the expected files are missing; avoid needless copies of large data.

Store runtime secrets in Kaggle's **Add-ons → Secrets** and retrieve them with `kaggle_secrets.UserSecretsClient`. Never put secrets in notebook cells, metadata, outputs, or the repository.
