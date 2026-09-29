# Member 2 Report Draft: Experimental Design, Training Protocol, and ResNet-50 Architecture

> **Draft Status:** This section provides verified technical documentation for the training engine, reproducibility mechanisms, and ResNet-50 transfer learning architecture. Parameter counts and tensor dimensions match repository assertions in `tests/test_resnet50.py`. Citations resolve through `references/member2_references.bib`.

## 5. Experimental Design and Training Infrastructure

### 5.1 Training Engine Architecture and Command-Line Interface

To ensure fair and strictly comparable experimental conditions across all four models, the project refactored disparate training routines into a unified, deterministic training engine orchestrated through `run_pipeline.py`. Each model is executed individually via command-line flags rather than monolithic scripts:

```bash
python run_pipeline.py train --model resnet50 --seed 42 --config configs/config.yaml
```

The training engine enforces four strict integrity guarantees:
1. **Authoritative Configuration:** All hyperparameters (learning rate, weight decay, batch size, patience, image resolution) are parsed from `configs/config.yaml`. Hardcoded parameters are prohibited.
2. **Deterministic Seeding:** The random number generator states for Python `random`, NumPy, and PyTorch (CPU and CUDA backends) are initialized with explicit seeds (`42`, `123`, `2026`). CUDA convolution benchmarking is deterministic (`torch.backends.cudnn.deterministic = True`).
3. **Partition Isolation:** The training pipeline accepts only `train.csv` and `validation.csv` manifests. The locked test manifest is physically inaccessible during training and hyperparameter selection.
4. **Resumable State Checkpointing:** Training writes two distinct artifacts per epoch: `best_inference.pt` (pruned weights, class mappings, and preprocessing metadata for deployment) and `last_resume.pt` (complete state including model weights, optimizer momentum buffers, learning rate scheduler state, epoch counter, and early stopping history).

### 5.2 Two-Phase Transfer Learning Protocol

Direct end-to-end backpropagation through pretrained convolutional backbones with randomly initialized classification heads frequently triggers destructive gradient updates that erase useful low-level visual features [@yosinski2014transferable]. To prevent representation collapse, the training pipeline executes a phased transfer learning protocol:

- **Phase 1: Feature Extraction (15 Epochs Maximum):**
  The backbone convolutional weights are frozen (`requires_grad = False`). Only the custom classification head (`Dropout(p=0.4)` + `Linear(2048, 38)`) is updated using the primary learning rate $\eta_1 = 1\times 10^{-3}$ and Adam optimization [@kingma2014adam]. Frozen batch normalization layers maintain fixed running mean and variance statistics rather than batch statistics.
- **Phase 2: Fine-Tuning (10 Epochs Maximum):**
  The best model checkpoint from Phase 1 (selected strictly by validation macro F1) is reloaded. The highest-level residual block (`layer4`, comprising three bottleneck blocks) is unfrozen, while lower stages (`conv1`, `layer1`, `layer2`, `layer3`) remain frozen to retain generic edge and texture primitives. The optimizer and learning rate scheduler are re-instantiated with a reduced fine-tuning rate $\eta_2 = 1\times 10^{-4}$ to enable conservative gradient adjustments.

### 5.3 Optimization, Regularization, and Early Stopping

- **Loss Function:** Multi-class cross-entropy loss with softmax activation over 38 mutually exclusive crop-disease classes.
- **Weight Decay:** $L_2$ regularization with coefficient $1\times 10^{-4}$ applied to weight matrices (excluding bias terms and batch normalization affine parameters) to penalize excessive parameter magnitudes [@loshchilov2017decoupled].
- **Adaptive Scheduling:** `ReduceLROnPlateau` monitors validation macro F1. If the metric fails to improve for 2 consecutive epochs, the learning rate is scaled by a factor of 0.5 down to a floor of $1\times 10^{-5}$.
- **Early Stopping:** Training terminates early if validation macro F1 fails to exceed the historical maximum for 5 consecutive epochs, preventing overfitting to training-set artifacts.

---

## 6. Model Architecture: ResNet-50 Deep Residual Learning

### 6.1 Architectural Principles and Mathematical Formulation

As neural network depth increases, standard convolutional networks exhibit a degradation problem: beyond a threshold, adding layers leads to higher training error, which is not attributable to overfitting but to vanishing and exploding gradient dynamics during backpropagation [@he2016deep].

ResNet-50 resolves degradation by introducing identity shortcut connections. Instead of requiring stacked non-linear layers to directly fit an underlying mapping $\mathcal{H}(x)$, the network explicitly reformulates the residual mapping:

$$\mathcal{F}(x) = \mathcal{H}(x) - x \implies \mathcal{H}(x) = \mathcal{F}(x) + x$$

When identity mappings are optimal, the training optimizer can push residual weights $\mathcal{F}(x) \to 0$ far more readily than fitting identity transformations through non-linear convolutions [@he2016identity].

### 6.2 Bottleneck Building Block Design

To control computational cost across 50 layers, ResNet-50 employs a three-layer bottleneck architecture instead of standard two-layer 3 × 3 blocks:
1. **1 × 1 Convolution:** Compresses feature dimensions from 256 to 64 channels, reducing input volume for the subsequent spatial operation.
2. **3 × 3 Convolution:** Extracts spatial features at reduced channel depth.
3. **1 × 1 Convolution:** Expands feature channels by a factor of 4 (from 64 to 256 in Stage 1, up to 2048 in Stage 4).

Batch normalization is placed immediately after each convolution and before the ReLU non-linearity. In shortcut paths where spatial dimensions or channel counts change (e.g., transition between `layer1` and `layer2`), a 1 × 1 projection convolution with stride 2 matches tensor dimensions.

### 6.3 Layer Allocations and Parameter Breakdown

ResNet-50 comprises 23,585,894 total parameters across its stem, four residual stages, and classifier:
- **Stem:** 7 × 7 convolution (stride 2, 64 filters), batch normalization, and 3 × 3 max pooling reduces spatial resolution from 224 × 224 to 56 × 56 (9,536 parameters).
- **Stage 1 (`layer1`):** 3 bottleneck blocks, output 256 channels at 56 × 56 (215,808 parameters).
- **Stage 2 (`layer2`):** 4 bottleneck blocks, output 512 channels at 28 × 28 (1,219,584 parameters).
- **Stage 3 (`layer3`):** 6 bottleneck blocks, output 1024 channels at 14 × 14 (7,098,368 parameters).
- **Stage 4 (`layer4`):** 3 bottleneck blocks, output 2048 channels at 7 × 7 (14,964,736 parameters).
- **Classification Head:** Adaptive global average pooling produces a 2048-dimensional embedding, followed by `Dropout(p=0.4)` and a single linear transformation yielding 38 logits (77,862 parameters).

In Phase 1, only 77,862 parameters (0.33%) are updated. In Phase 2, `layer4` and the head are trained simultaneously, activating 15,042,598 parameters (63.78%) while 8,543,296 parameters (36.22%) in stages 1–3 remain frozen.
