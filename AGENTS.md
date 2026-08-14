

# Use UV as package manager

The package and project manager is uv. Run scripts with "uv run script.py" Important Commands:
| Command | Example | Action |
| :--- | :--- | :--- |
| **`uv run`** | `uv run main.py` | Run a script/command in the managed environment. |
| **`uv init`** | `uv init` | Create a new project with `pyproject.toml`. |
| **`uv add`** | `uv add pandas` | Add a dependency to the project and install it. |
| **`uv sync`** | `uv sync` | Ensure virtual env matches the lockfile exactly. |
| **`uv tool install`**| `uv tool install ruff`| Install a global CLI tool (replaces `pipx`). |
| **`uv python install`**| `uv python install 3.12`| Download and install a specific Python version. |
| **`uv lock`** | `uv lock` | Resolve dependencies and update `uv.lock`. |
| **`uv remove`** | `uv remove requests` | Remove a dependency from project and environment. |
| **`uv tree`** | `uv tree` | Display the dependency graph visually. |
| **`uv pip install`** | `uv pip install numpy` | Low-level, fast package install (classic pip style). |

# Actuality

Current month is August in 2026

# Kaggle and Machine Learning Work

For Kaggle competitions or machine-learning problems, always use the
`$kaggle-notebook-ml` skill.

- Treat Kaggle as the primary execution environment. Use local execution only
  for small inspections, unit checks, and debug smoke tests.
- Create a Kaggle-compatible, top-to-bottom runnable Jupyter notebook as the
  primary deliverable. Keep the user informed through visual data exploration,
  Markdown explanations, and concise progress updates.
- Never search for competition solutions, winning notebooks, leaked labels, or
  competition-specific code. Research only official documentation and
  general-purpose methods.
- Use Context7 for current library and API documentation.
- Run all Kaggle CLI commands through `uv run kaggle`.
- Design and justify validation before model selection; actively check for
  target leakage, duplicates, groups, temporal dependencies, and train/test
  distribution shift.
- Prefer robust validation over public-leaderboard optimization. Record every
  material experiment, decision, runtime, and submission in `decision_log.md`.
- Use remote Kaggle execution for meaningful training, inference, and
  submission generation. Verify every final notebook in a fresh Kaggle session.
- Stop and ask the user when authentication, competition acceptance, Kaggle
  resources, data permissions, or a material modeling choice requires their
  action or decision.
- Never delete remote Kaggle notebooks or submit competition predictions unless
  the user has explicitly authorized the action.

## Kaggle Remote GPU Execution & Dual T4 (2x T4) Optimization

1. **Explicit GPU Accelerator Flag & Metadata**:
   - Always push GPU notebooks using `--accelerator nvidiaTeslaT4x2` (or `--accelerator nvidiaTeslaT4`).
   - Always specify `"enable_gpu": "true"` and `"machine_shape": "nvidiaTeslaT4x2"` in `kernel-metadata.json`.
2. **Strict Hardware Verification**:
   - Notebooks must strictly verify GPU hardware at startup (`torch.cuda.device_count() >= 1` and `sm_70+` architecture like T4).
   - If GPU is unavailable or legacy P100 (`sm_60`) is detected, halt immediately with a descriptive `RuntimeError`.
3. **Dual T4 Parallelism Optimization**:
   - For multi-fold CV or multi-model ensembles on Dual T4:
     - Use process-level fold/model parallelism (`multiprocessing` assigning Fold $i$ to GPU 0 and Fold $i+1$ to GPU 1) to achieve linear $2\times$ training speedup without PCIe communication bottlenecks.
     - Alternatively, use `torch.nn.DataParallel` or FastAI distributed dataloaders for per-batch dual-GPU data parallelism.
4. **Notebook `kernelspec` Metadata**:
   - Ensure notebook metadata contains `kernelspec` (`{"name": "python3", "display_name": "Python 3", "language": "python"}`) to prevent Papermill crashes.
5. **No CUDA/Torch Overwrites**:
   - Install missing packages strictly with `--no-deps` (`!pip install -q --no-deps timm`). Never upgrade or overwrite `torch` or `torchvision` in remote kernels.