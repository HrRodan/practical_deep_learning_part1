# Paddy Disease Classification - Decision Log

## Problem Overview
- **Competition**: Paddy Doctor - Paddy Disease Classification
- **Task**: 10-class image classification (9 disease classes + 1 normal class).
- **Metric**: Top-1 Categorization Accuracy ($1 - \text{error\_rate}$).
- **Train Size**: 10,407 labeled images across 10 classes + metadata (`variety`, `age`).
- **Test Size**: 3,469 images in test set.
- **Environment**: Dual-compatible (Local NVIDIA RTX A1000 6GB GPU + Kaggle GPU Kernel workflow).

---

## 1. Data Understanding & Leakage Checks
- **Class Balance**: 10 classes (`normal`: 16.95%, `blast`: 16.70%, `hispa`: 15.32%, `dead_heart`: 13.86%, `tungro`: 10.45%, `brown_spot`: 9.27%, `downy_mildew`: 5.96%, `bacterial_leaf_blight`: 4.60%, `bacterial_leaf_streak`: 3.65%, `bacterial_panicle_blight`: 3.24%).
- **Image Resolution**: Portrait format $(480 \times 640)$ across the dataset.
- **Metadata**: `variety` (10 categories) and `age` (range: 45–82 days, mean: 64.0 days).
- **Leakage / Overlap Check**:
  - Image IDs are disjoint between train and test (`train`: 100000–110407, `test`: 200000–203469). PASSED.

---

## 2. Validation Design
- **Strategy**: 5-Fold Stratified K-Fold based on target `label`.
- **Reproducibility**: Fixed seed (`42`). Saved to `data/splits_5fold.csv` (2,081–2,082 samples per fold).
- **Leakage Prevention**: Stratification preserves class distributions without cross-fold sample contamination.

---

## 3. Experiment & Submission Tracker

| Exp ID | Model Architecture | Resolution | Epochs | Val Accuracy | Public LB | Private LB | Submission Reference | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-00** | Majority Class Heuristic | - | - | ~0.1695 | - | - | - | Baseline lower bound |
| **EXP-01** | `convnext_nano` (Fold 0) | $224 \times 224$ | 4 | **93.61%** | **0.94925** | **0.95391** | `55491442` | Fast initial baseline with TTA |
| **EXP-02** | `convnext_small` (Fold 0) | $384 \times 384$ | 6 | **97.17%** | **0.98077** | **0.97004** | `55501580` | High-res scaling, large gain (+3.15% LB) |
| **EXP-03** | `convnext_small` + `resnet50d` Ensemble | $384 \times 384$ | 6 + 6 | **96.88%** | **0.97616** | **0.97004** | `55503637` | Dual architecture blend on Dual T4 GPUs |
| **EXP-04** | `convnext_small` 5-Fold Ensemble | $384 \times 384$ | $5 \times 6$ | **97.44% (OOF)** | **0.98539** | **0.97811** | `55506232` | 5-Fold CV + 20-pass consensus TTA |
| **EXP-05** | `convnext_base` 5-Fold Ensemble (Dual T4) | $384 \times 384$ | $5 \times 6$ | 89.91% (OOF) | - | - | - | Dual-T4 multiprocessing benchmark (under-converged in 6 epochs) |

---

## 4. Error Analysis & Key Observations (EXP-04 & EXP-05)
* **Performance Breakthrough (EXP-04 - Champion Model)**:
  * **Public Leaderboard**: **0.98077 -> 0.98539 (+0.46% gain)**
  * **Private Leaderboard**: **0.97004 -> 0.97811 (+0.81% gain)**
  * **Out-of-Fold (OOF) Accuracy**: **97.44%** across all 10,407 samples:
    * Fold 0: 97.17% (Loss: 0.1148)
    * Fold 1: 97.45% (Loss: 0.0894)
    * Fold 2: 97.21% (Loss: 0.0898)
    * Fold 3: 97.69% (Loss: 0.0889)
    * Fold 4: 97.65% (Loss: 0.0992)
    * Mean Fold Accuracy: **97.43% +/- 0.22%**
* **Dual T4 Multiprocessing Execution (EXP-05)**:
  * Successfully parallelized 5-fold cross-validation across Dual T4 GPUs using process-level fold dispatcher.
  * `convnext_base` (88M params) reached 89.91% OOF in 6 epochs, showing continuous improvement per epoch but requiring 10-12 epochs or higher initial LR to reach full convergence.
  * **Takeaway**: `convnext_small` (EXP-04) remains our **champion model** with 97.44% OOF and **0.98539 Public LB / 0.97811 Private LB**.
* **Next Optimization Candidates**:
  1. **Auxiliary Variety / Cultivar Multi-Head Training**:
     * Predict `variety` (rice cultivar) alongside `label` to exploit cultivar-specific disease resistance genetics.
  2. **Test-Time Augmentation Scaling & Post-Processing**:
     * Increase TTA passes or optimize temperature scaling on the EXP-04 5-Fold ensemble.



