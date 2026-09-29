# Member 2 Verified Facts: ResNet-50 & Training Infrastructure

All parameter numbers, tensor dimensions, and protocol configurations below are directly verified from the PyTorch model implementation (`src/models/resnet50.py`), experiment configuration (`configs/config.yaml`), and frozen split definitions.

## 1. Architectural Dimensions and Parameter Allocations

- **Backbone:** ResNet-50 (50-layer deep residual network with bottleneck building blocks)
- **Pretrained Weights:** ImageNet-1K (`ResNet50_Weights.DEFAULT`)
- **Input Dimensions:** 3 × 224 × 224 RGB
- **Total Parameters:** 23,585,894 (100.0%)
- **Phase 1 Trainable Parameters:** 77,862 (0.33%) — Head only (`Dropout(p=0.4)` + `Linear(2048, 38)`)
- **Phase 1 Frozen Parameters:** 23,508,032 (99.67%)
- **Phase 2 Trainable Parameters:** 15,042,598 (63.78%) — `layer4` (3 bottleneck blocks) + Head
- **Phase 2 Frozen Parameters:** 8,543,296 (36.22%) — `conv1`, `bn1`, `layer1`, `layer2`, `layer3`

## 2. Layer Breakdown Table

| Stage | Submodule / Layer | Output Tensor Shape | Parameter Count | Phase 1 Status | Phase 2 Status |
|---|---|---|---:|---|---|
| Input | Image Batch | B × 3 × 224 × 224 | 0 | Input | Input |
| Stem Conv | `conv1` | B × 64 × 112 × 112 | 9,408 | Frozen | Frozen |
| Stem BN | `bn1` | B × 64 × 112 × 112 | 128 | Frozen | Frozen |
| Stem Pool | `maxpool` | B × 64 × 56 × 56 | 0 | Static | Static |
| Stage 1 | `layer1` (3 blocks) | B × 256 × 56 × 56 | 215,808 | Frozen | Frozen |
| Stage 2 | `layer2` (4 blocks) | B × 512 × 28 × 28 | 1,219,584 | Frozen | Frozen |
| Stage 3 | `layer3` (6 blocks) | B × 1024 × 14 × 14 | 7,098,368 | Frozen | Frozen |
| Stage 4 | `layer4` (3 blocks) | B × 2048 × 7 × 7 | 14,964,736 | Frozen | **Trainable** |
| Pooling | `avgpool` | B × 2048 × 1 × 1 | 0 | Static | Static |
| Regularizer | `fc.0` (Dropout p=0.4) | B × 2048 | 0 | Train-only | Train-only |
| Classifier | `fc.1` (Linear 2048→38) | B × 38 | 77,862 | **Trainable** | **Trainable** |
| **Total** | | | **23,585,894** | **77,862 trainable** | **15,042,598 trainable** |

## 3. Training Protocol Specifications

- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, weight decay $1\times 10^{-4}$)
- **Phase 1 Learning Rate:** $1\times 10^{-3}$ (15 epochs max)
- **Phase 2 Learning Rate:** $1\times 10^{-4}$ (10 epochs max)
- **Batch Size:** 32
- **Loss Function:** Categorical Cross-Entropy (`nn.CrossEntropyLoss`)
- **Model Selection Metric:** Validation Macro F1 score
- **Early Stopping Patience:** 5 epochs without improvement on validation macro F1
- **Learning Rate Reduction:** Factor 0.5 with patience 2 epochs, min LR $1\times 10^{-5}$
