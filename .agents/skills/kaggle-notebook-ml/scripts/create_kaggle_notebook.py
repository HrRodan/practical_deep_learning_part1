#!/usr/bin/env python3
"""Create a clean, Kaggle-compatible ML notebook scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str, *, tags: list[str] | None = None):
    cell = nbf.v4.new_code_cell(source.strip())
    if tags:
        cell.metadata["tags"] = tags
    return cell


def build_notebook(title: str):
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook.cells = [
        markdown(f"# {title}\n\n> Kaggle-first, reproducible machine-learning notebook."),
        markdown("## Problem and approach\n\nDescribe the target, competition metric, constraints, and the current hypothesis."),
        markdown("## Setup and configuration\n\nKeep all settings here. Change `DEBUG` before expensive work."),
        code(
            """from pathlib import Path
import os
import random

SEED = 42
DEBUG = True
REQUIRE_T4_GPU = False
IS_KAGGLE = bool(os.getenv(\"KAGGLE_KERNEL_RUN_TYPE\")) or Path(\"/kaggle\").exists()
INPUT_ROOT = Path(\"/kaggle/input\") if IS_KAGGLE else Path(\"data\")
WORK_ROOT = Path(\"/kaggle/working\") if IS_KAGGLE else Path(\"artifacts\")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)

print({\"is_kaggle\": IS_KAGGLE, \"input_root\": INPUT_ROOT, \"work_root\": WORK_ROOT, \"debug\": DEBUG, \"require_t4_gpu\": REQUIRE_T4_GPU})""",
            tags=["parameters"],
        ),
        markdown("## Accelerator check\n\nEnable `REQUIRE_T4_GPU` for GPU workloads. Crash early rather than silently training on CPU or an unsupported accelerator."),
        code(
            """if REQUIRE_T4_GPU:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(\"A T4 GPU is required but CUDA is unavailable.\")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print({\"gpu_count\": len(gpu_names), \"gpus\": gpu_names})
    if not gpu_names or any(\"T4\" not in name for name in gpu_names):
        raise RuntimeError(f\"Expected NVIDIA T4 GPU(s), received: {gpu_names}\")"""
        ),
        markdown("## Data audit\n\nShow schema, quality checks, target distribution, representative samples, and train/test differences. Interpret every meaningful plot."),
        code("# Resolve inputs dynamically under INPUT_ROOT; then create concise, visual data checks."),
        markdown("## Validation design\n\nJustify the split, identify leakage risks, and implement the exact competition metric."),
        code("# Create deterministic split assignments and metric helpers."),
        markdown("## Baseline\n\nTrain a fast end-to-end baseline in debug mode, then run the comparable full Kaggle version."),
        code("# Train, validate, record runtime, and save validation predictions."),
        markdown("## Error analysis\n\nShow worst predictions, subgroup metrics, and plots that justify the next hypothesis."),
        code("# Analyze errors and state the next experiment."),
        markdown("## Experiments\n\nAdd one subsection per hypothesis. Change one material factor at a time and compare to the baseline."),
        code("# Experiment 001 — hypothesis: <state it here>."),
        markdown("## Final training and inference\n\nRerun the selected configuration and write artifacts to `WORK_ROOT`."),
        code("# Train final model and generate test predictions."),
        markdown("## Submission checks\n\nValidate columns, IDs, row count, types, missing values, and prediction range."),
        code("# Write WORK_ROOT / 'submission.csv' after validation."),
        markdown("## Conclusions and next steps\n\nSummarize executed results, caveats, and the next evidence-based action."),
    ]
    nbf.validate(notebook)
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination .ipynb file")
    parser.add_argument("--title", default="Kaggle ML Solution", help="Notebook title")
    parser.add_argument("--force", action="store_true", help="Replace an existing notebook")
    args = parser.parse_args()

    if args.output.suffix != ".ipynb":
        parser.error("output must end in .ipynb")
    if args.output.exists() and not args.force:
        parser.error(f"refusing to overwrite {args.output}; pass --force to replace it")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(args.title), args.output)
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
