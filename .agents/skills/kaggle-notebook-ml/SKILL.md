---
name: kaggle-notebook-ml
description: Build, refactor, or run a Kaggle-compatible Jupyter notebook for a machine-learning problem or competition. Use when you must perform Kaggle-first data exploration, validation design, model experimentation, visual error analysis, reproducible notebook execution, or competition submission preparation; local execution is limited to small smoke tests.
---

# Kaggle-First ML Notebooks

Create a reader-facing, top-to-bottom runnable `.ipynb` notebook as the primary artifact. Use Kaggle for meaningful preprocessing, training, cross-validation, inference, and submission creation; use local resources only for small data inspection, unit checks, and fast smoke tests.

## Operating rules

- Work independently. Never search for competition solutions, leaked labels, winning notebooks, or competition-specific code.
- Use current official documentation for libraries and APIs; use Context7 before using unfamiliar or version-sensitive library APIs.
- Optimize the stated competition metric, but select models by leakage-safe validation rather than public leaderboard feedback.
- Keep the user involved: show important evidence, explain conclusions, and state the next proposed experiment before expensive work.
- Make every material experiment reproducible, comparable, and recorded in `decision_log.md`.
- Run Kaggle CLI commands through `uv run kaggle`, and use the remote-CI procedure in [references/kaggle-cli-ci.md](references/kaggle-cli-ci.md) for every remote run.

## Frame the problem before building the notebook

Identify the task type, target, exact evaluation metric, resource/time limits, competition rules, and submission format. Inspect the sample submission and verify its IDs, prediction columns, row order, shape, types, and expected value range.

Create a trivial reference prediction or heuristic before training. Use it to validate the metric and the submission pipeline; do not advance to modeling until it produces a valid submission artifact.

## Create the notebook before adding analysis

Do not hand-edit raw `.ipynb` JSON. Create new notebooks with the bundled scaffold, JupyterLab, or `nbformat`; use the scaffold for the standard Kaggle ML workflow:

```bash
python3 scripts/create_kaggle_notebook.py notebooks/<problem>.ipynb --title "<Competition> — ML solution"
```

The scaffold produces a valid Python-3 notebook with one ordered Markdown/code-cell skeleton, a single environment configuration cell, a `DEBUG` switch, and Kaggle/local paths. It is intentionally executable before any project-specific code is added. Use notebook-safe tools (`nbformat`, `nbclient`, or JupyterLab) to add, edit, inspect, and validate cells; never compose or mutate notebook JSON as plain text.

Choose one creation path:

1. **Kaggle editor** — use for a new interactive notebook, attaching competition datasets, choosing CPU/GPU/TPU, and visually reviewing outputs. Create the notebook there, then apply the same section order and configuration cell as the scaffold.
2. **Local scaffold + Kaggle CLI** — use when the agent needs a reviewable local artifact and repeatable remote execution. Put the `.ipynb` and `kernel-metadata.json` in one dedicated notebook folder, initialize metadata with `uv run kaggle kernels init -p <folder>`, attach only required inputs, then push, monitor, and retrieve outputs using the remote-CI loop in [references/kaggle-cli-ci.md](references/kaggle-cli-ci.md). Do not pull or reuse other competitors' notebooks; only pull the user's own existing notebook.

Before implementing modeling logic, open the generated notebook, confirm the kernel metadata and input paths, run the setup cell, and ask the user to resolve any missing competition access, data attachment, accelerator, or privacy setting.

## Create the notebook structure

Use this order, adapting titles to the problem:

1. `# Problem and approach` — task, metric, constraints, and current plan.
2. `## Setup and configuration` — seeds, paths, debug/full mode, model and validation parameters.
3. `## Data audit` — schema, quality checks, representative samples, and train/test comparison.
4. `## Validation design` — split choice, leakage risks, and metric implementation.
5. `## Baseline` — end-to-end, fast model and results.
6. `## Error analysis` — failure cases and evidence-driven hypotheses.
7. `## Experiments` — one clearly named hypothesis per subsection.
8. `## Final training and inference` — selected configuration and test predictions.
9. `## Submission checks` — sample-submission compatibility and output validation.
10. `## Conclusions and next steps` — observed outcomes, caveats, and recommended work.

Put concise Markdown before each significant code cell: state the purpose, relevant assumptions, and expected output. Interpret each material table, chart, and score immediately below it. Use attractive, consistently labeled charts and bounded preview tables; never leave unexplained debug dumps in the notebook.

## Guarantee Kaggle compatibility

Keep all environment-specific settings in one configuration cell. Detect Kaggle from `KAGGLE_KERNEL_RUN_TYPE` or the `/kaggle` directory. Use `/kaggle/input` for immutable data and `/kaggle/working` for generated artifacts. Never assume a local GPU, local paths, or locally installed packages.

Include:

- A `DEBUG` flag that reduces rows, folds, epochs, image resolution, or samples without changing the data flow.
- Explicit seeds and deterministic settings where practical.
- A package-install cell only for required packages unavailable in Kaggle; pin versions when reproducibility requires it.
- Relative project imports or notebook-local helpers that work in Kaggle after any required source upload/attachment.
- Clear artifact paths for checkpoints, out-of-fold predictions, plots, logs, and `submission.csv` under `/kaggle/working`.
- CPU-safe fallback settings and optional GPU/mixed-precision settings that activate only when supported.

For a GPU-required workload, enable an explicit `REQUIRE_T4_GPU` configuration, print every allocated GPU name and count, and raise a fatal error unless NVIDIA T4 hardware is available. Push with a matching explicit accelerator and metadata. Follow the package, input-path, fallback-download, and secret-handling rules in [references/kaggle-cli-ci.md](references/kaggle-cli-ci.md).

Prefer reusable functions or small Python modules for data loading, preprocessing, metrics, training, and inference. Keep the notebook as the clear narrative and orchestration layer; do not copy large implementations across cells.

## Select tools deliberately

Use current stable versions and verify APIs through Context7. Select the smallest suitable toolset:

- Use FastAI or PyTorch for training; use Hugging Face for pretrained models, datasets, and tokenizers when appropriate.
- Use Pandas for small tabular data and Polars for larger tabular workloads.
- Use Altair for polished, interactive visualizations when it is supported by the execution environment.
- Use Optuna or an equivalent controlled search tool only after baseline, validation, and data flow are trustworthy.
- Prefer Kaggle's existing compatible packages. Never upgrade `torch` or `torchvision` in a Kaggle notebook.

## Work in evidence-based stages

### 1. Audit the data

Inspect schema, row counts, IDs, targets, missing values, duplicates, data types, class balance, invalid records, and outliers. For images, text, audio, or time series, show representative and difficult examples. Compare train and test feature distributions without using unavailable test labels.

Actively check for leakage, duplicated entities, group structure, temporal ordering, and overlap between train, validation, and test. Show the user the compact visual evidence that affects the modeling plan.

### 2. Choose validation before training

Match the split to the expected hidden-test construction: use stratification for class balance, grouping for related records, temporal splits for time-dependent data, and cross-validation when a single split is unreliable. Never let duplicates, related groups, or augmented versions cross folds.

Save split assignments or the deterministic split definition. Explain why the validation design is representative and name its remaining uncertainty.

### 3. Build an end-to-end baseline

Create a fast, complete pipeline that loads data, preprocesses it, trains, evaluates the exact metric, predicts test data, and writes a valid submission. First run it in `DEBUG` mode, then run the comparable full version on Kaggle.

Verify that the model can overfit a tiny subset when feasible. Treat failure as a pipeline or data bug before changing models. Report score, fold variance, runtime, hardware, and resource constraints to the user.

### 4. Diagnose errors, then improve

Inspect worst predictions, confident mistakes, per-class or subgroup metrics, confusion matrices, residuals, learning curves, and feature/model explanations appropriate to the task. Use errors to form the next hypothesis.

Change one major factor per experiment: preprocessing, augmentation, features, loss, sampling, thresholds, pretrained backbone, hyperparameters, or architecture. Consider pretrained transfer learning, task-appropriate test-time augmentation, and threshold optimization when validation evidence supports them. Start with efficient models and only increase compute when results support it. Keep different modeling paths in separate, clearly labeled sections; create another notebook only for a genuinely different approach.

Test several credible approaches before finalizing. Ensemble only individually strong and materially diverse models.

## Communicate and document

At each stage, tell the user:

- What was executed and where.
- What the notebook visibly shows.
- What was learned and what remains uncertain.
- The next experiment, why it is justified, and expected cost.

Ask the user for direction when the data permits materially different validation or modeling interpretations, when a high-cost choice has no clear evidence-based winner, or when Kaggle access, quota, accelerator selection, or private data requires user action.

Maintain `decision_log.md` with the experiment ID, hypothesis, configuration, data and split version, validation score and variance, runtime/hardware, conclusion, and next action. Record every Kaggle submission and its leaderboard score, but do not repeatedly tune against the public leaderboard.

## Verify before handoff or submission

Run the notebook from a fresh Kaggle session, top to bottom, in full mode. For automated local verification, use `nbclient` or `jupyter nbconvert --execute` against a small, representative fixture; treat this as a smoke test, not proof that Kaggle execution works.

Before submitting, validate sample-submission columns, ID order, row count, data types, missing or infinite values, prediction range, and output path. Preserve the executed notebook, final configuration, generated submission, and any strong checkpoint or out-of-fold predictions.

Do not claim completion unless the notebook has a clear execution status. If Kaggle execution cannot be performed, identify the exact blocker, what was verified locally, and the user action required.

## Definition of done

Complete the work only when all applicable conditions hold:

- The notebook runs top-to-bottom in a fresh Kaggle session and produces a valid submission.
- Data findings are shown and interpreted; validation is justified and leakage-safe.
- A complete baseline and several credible, documented improvements were evaluated.
- Model behavior and important errors were analyzed.
- Failed ideas, final configuration, artifacts, and submission history are recorded in `decision_log.md`.
- The user receives the final score evidence, caveats, and evidence-based next steps.

## Remote CI and cleanup

For every full Kaggle run, follow the remote-CI loop in [references/kaggle-cli-ci.md](references/kaggle-cli-ci.md): validate the notebook, push it, monitor its exact kernel ID, inspect logs after errors, download and verify artifacts after completion, and log the result. Treat a successful local smoke test as insufficient.

Delete a remote notebook only with explicit user approval after resolving the exact owner and slug.
