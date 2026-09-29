# Member 3 Report Draft: Model Evaluation Protocol and EfficientNet-B0 Architecture

> **Draft Status:** This section provides verified technical documentation for the evaluation and benchmarking protocol, multi-class metric formulations, and EfficientNet-B0 compound scaling architecture. All facts are verified by automated tests in `tests/test_efficientnet_b0.py`. Citations resolve through `references/member3_references.bib`.

## 4. Model Evaluation and Fair Comparison Protocol

### 4.1 Multi-Metric Assessment Framework

Relying on overall accuracy alone in medical or agricultural diagnosis can be deeply misleading [@powers2011evaluation]. In the locked PlantVillage benchmark, the training set contains a 37:1 ratio between the majority class (`Orange___Haunglongbing`) and minority class (`Potato___healthy`). A trivial classifier predicting only frequent classes could achieve deceptively high accuracy while failing entirely on vulnerable agricultural pathogens.

To provide an evidence-based assessment, Member 3 designed a multi-metric evaluation protocol implemented in `src/evaluation/metrics.py`:

1. **Top-1 and Top-3 Accuracy:** Evaluates the proportion of samples where the ground truth matches the highest-probability class and is within the top three predicted classes, respectively.
2. **Macro-Averaged Precision, Recall, and F1 Score:** Calculates unweighted arithmetic means across all 38 classes [@grandini2020metrics]:
   $$\text{Macro F1} = \frac{1}{C}\sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
   Treating each disease category with equal weight guarantees that catastrophic misclassifications on rare diseases penalize the reported score proportionally.
3. **Multiclass One-vs-Rest (OvR) ROC-AUC:** Measures discriminative capability across decision thresholds by averaging binary One-vs-Rest ROC integrals [@fawcett2006introduction].
4. **Normalized Confusion Matrix Analysis:** Row-normalized confusion matrices reveal per-class recall rates and pinpoint inter-class morphological confusions (e.g., distinguishing early blight from late blight on tomato foliage).

### 4.2 Standalone Evaluation and Locked Test Integrity

The evaluation routine operates independently of the training module via `run_pipeline.py evaluate`. Before processing batches, the runner validates:
- Checkpoint architectural signature and weight shapes;
- Contiguous alphabetical class mapping (38 classes);
- Exact SHA-256 hash of the target partition manifest.

The locked test set (`test.csv`, 10,709 images) cannot be evaluated without explicitly passing the `--confirm-locked-test` safeguard. This architectural constraint prevents data leakage and preserves test-set virginity throughout model selection.

---

## 6. Model Architecture: EfficientNet-B0 Compound Scaling

### 6.1 Compound Scaling Principle

Prior convolutional architectures typically scaled networks arbitrarily along a single dimension: depth (e.g., ResNet-50 to ResNet-152), width (e.g., WideResNet), or image resolution. Tan & Le (2019) demonstrated that scaling depth, width, and resolution simultaneously yields superior accuracy and parameter efficiency [@tan2019efficientnet].

EfficientNet-B0 is founded on a principled compound scaling method:
$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$
$$\text{subject to } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \ge 1, \beta \ge 1, \gamma \ge 1$$

Here, $\phi$ is a user-specified compound coefficient governing available computational resources, while $\alpha=1.2, \beta=1.1, \gamma=1.15$ are fixed scaling coefficients discovered via grid search on the baseline architecture.

### 6.2 Mobile Inverted Bottleneck (MBConv) and Squeeze-and-Excitation

EfficientNet-B0 employs MBConv blocks consisting of:
1. **1 × 1 Expansion Convolution:** Expands low-dimensional input channels by an expansion factor of 6 (except the first block where expansion is 1), projecting features into a higher-dimensional space where non-linear representations are richer.
2. **Depthwise Convolution:** Performs spatial filtering independently per channel using 3 × 3 or 5 × 5 kernels, reducing FLOPs by approximately $k^2$ relative to standard convolutions.
3. **Squeeze-and-Excitation (SE) Optimization:** Dynamically recalibrates channel-wise feature responses [@hu2018squeeze]. A global average pooling operation squeezes spatial dimensions into a channel descriptor vector, followed by two fully connected layers with reduction ratio 4 and sigmoid activation that reweight channel significance.
4. **1 × 1 Projection Convolution:** Compresses channels back to the target bottleneck dimension without non-linear activation (linear bottleneck), preserving information integrity.
5. **Residual Connections:** Added whenever input and output tensor dimensions match.

### 6.3 Layer Allocations and Parameter Breakdown

EfficientNet-B0 contains 4,056,226 parameters—an 82.8% reduction in parameter footprint compared to ResNet-50 (23,585,894):
- **Stem (`features.0`):** 3 × 3 convolution (stride 2, 32 channels) + BatchNorm + SiLU (928 parameters).
- **Stages 1–6 (`features.1` through `features.6`):** 15 MBConv blocks progressing from 16 to 192 channels (2,393,372 parameters).
- **Stage 7 (`features.7`):** Final MBConv6 block expanding to 320 channels (1,118,592 parameters).
- **Stage 8 (`features.8`):** 1 × 1 pointwise convolution projecting to 1280 feature channels (412,160 parameters).
- **Classifier Head:** Global average pooling, `Dropout(p=0.3)`, and linear projection to 38 classes (48,678 parameters).

In Phase 1, only 48,678 parameters (1.20%) are trained. In Phase 2, unfreezing `features.7` and `features.8` unlocks 1,178,070 trainable parameters (29.04%), allowing high-level disease pattern recognition while keeping low-level edge extractors fixed.
