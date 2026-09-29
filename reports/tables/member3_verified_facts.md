# Member 3 Verified Facts: EfficientNet-B0 & Evaluation Infrastructure

All parameter numbers, tensor dimensions, and protocol configurations below are directly verified from the PyTorch model implementation (`src/models/efficientnet_b0.py`), experiment configuration (`configs/config.yaml`), and standalone evaluation runner (`src/evaluation/runner.py`).

## 1. Architectural Dimensions and Parameter Allocations

- **Backbone:** EfficientNet-B0 (Compound scaled architecture with MBConv blocks)
- **Pretrained Weights:** ImageNet-1K (`EfficientNet_B0_Weights.DEFAULT`)
- **Input Dimensions:** 3 × 224 × 224 RGB
- **Total Parameters:** 4,056,226 (100.0%)
- **Phase 1 Trainable Parameters:** 48,678 (1.20%) — Head only (`Dropout(p=0.3)` + `Linear(1280, 38)`)
- **Phase 1 Frozen Parameters:** 4,007,548 (98.80%)
- **Phase 2 Trainable Parameters:** 1,178,070 (29.04%) — `features.7` (MBConv6 block), `features.8` (1×1 head conv), and Classifier Head
- **Phase 2 Frozen Parameters:** 2,878,156 (70.96%) — `features.0` through `features.6`

## 2. Stage Breakdown Table

| Stage | Submodule / Operator | Kernel / Stride | Output Tensor Shape | Parameter Count | Phase 1 Status | Phase 2 Status |
|---|---|---|---|---:|---|---|
| Input | Image Tensor | - | B × 3 × 224 × 224 | 0 | Input | Input |
| Stem | `features.0` | Conv2d 3×3, s=2 | B × 32 × 112 × 112 | 928 | Frozen | Frozen |
| Stage 1 | `features.1` (MBConv1) | DW 3×3 + SE, s=1 | B × 16 × 112 × 112 | 3,138 | Frozen | Frozen |
| Stage 2 | `features.2` (MBConv6) | DW 3×3 + SE, s=2 (2 blocks) | B × 24 × 56 × 56 | 24,142 | Frozen | Frozen |
| Stage 3 | `features.3` (MBConv6) | DW 5×5 + SE, s=2 (2 blocks) | B × 40 × 28 × 28 | 72,504 | Frozen | Frozen |
| Stage 4 | `features.4` (MBConv6) | DW 3×3 + SE, s=2 (3 blocks) | B × 80 × 14 × 14 | 227,616 | Frozen | Frozen |
| Stage 5 | `features.5` (MBConv6) | DW 5×5 + SE, s=1 (3 blocks) | B × 112 × 14 × 14 | 436,160 | Frozen | Frozen |
| Stage 6 | `features.6` (MBConv6) | DW 5×5 + SE, s=2 (4 blocks) | B × 192 × 7 × 7 | 1,211,808 | Frozen | Frozen |
| Stage 7 | `features.7` (MBConv6) | DW 3×3 + SE, s=1 (1 block) | B × 320 × 7 × 7 | 1,118,592 | Frozen | **Trainable** |
| Stage 8 | `features.8` (Head Conv) | Conv2d 1×1, s=1 | B × 1280 × 7 × 7 | 412,160 | Frozen | **Trainable** |
| Pooling | `avgpool` | AdaptiveAvgPool2d | B × 1280 × 1 × 1 | 0 | Static | Static |
| Regularizer | `classifier.0` | Dropout (p=0.3) | B × 1280 | 0 | Train-only | Train-only |
| Classifier | `classifier.1` | Linear 1280→38 | B × 38 | 48,678 | **Trainable** | **Trainable** |
| **Total** | | | | **4,056,226** | **48,678 trainable** | **1,178,070 trainable** |

## 3. Evaluation Protocol Specifications

- **Metric Suite:**
  - Overall: Top-1 Accuracy, Top-3 Accuracy
  - Balanced Performance: Macro-averaged Precision, Recall, and F1 score
  - Class-Weighted: Weighted F1 score
  - Threshold-Independent: Multiclass One-vs-Rest (OvR) Macro ROC-AUC
  - Diagnostic: Confusion Matrix (raw counts and row-normalized recall rates)
- **Model Selection Rule:** Validation Macro F1 score on frozen `validation.csv` partition
- **Locked Test Rule:** Evaluated exactly once on `test.csv` using `--confirm-locked-test`
