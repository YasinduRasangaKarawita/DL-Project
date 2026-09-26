# Member 1 report draft: introduction, data, preprocessing, and Custom CNN

> **Draft status:** The dataset, EDA, preprocessing, and Custom CNN design statements below are supported by committed repository evidence. Training outcomes are intentionally absent because the shared training/checkpoint/evaluation interface and `training-v1.0` tag have not yet been approved. Citation keys use Pandoc-style syntax and resolve through `references/member1_references.bib`.

## 1. Introduction and problem definition

Plant diseases can reduce crop quality and yield, while visual diagnosis may require expertise that is not consistently available. Image classification offers a way to investigate whether visible leaf symptoms can be mapped to crop–condition categories using reproducible deep-learning methods. PlantVillage provides a widely used controlled-image benchmark for this problem: the original study describes healthy and diseased leaves photographed under controlled conditions and demonstrates the feasibility of convolutional classification on this domain [@mohanty2016plant]. The controlled setting makes the collection useful for comparing model architectures, but it does not represent the full variability of field imagery.

This project asks: **when data, preprocessing, model selection, and evaluation are held constant, how do a from-scratch Custom CNN and three ImageNet transfer-learning architectures compare in predictive quality, stability, and computational efficiency on the locked PlantVillage benchmark?** The four models are a Custom CNN, ResNet-50, EfficientNet-B0, and MobileNetV3-Large. The Custom CNN supplies a compact, non-pretrained baseline against which the benefits and costs of the larger pretrained networks can later be assessed.

The project objectives are to:

1. acquire and validate a fixed public dataset revision without committing the images;
2. prevent path, exact-duplicate, and physical-leaf leakage across partitions;
3. apply identical input resolution and normalization to all models, with stochastic augmentation restricted to training;
4. train and evaluate every architecture under one frozen protocol and class order;
5. compare macro and per-class predictive metrics together with parameter count, checkpoint size, latency, throughput, and memory usage; and
6. interpret the results without presenting a controlled-background classifier as a field-ready diagnostic system.

Only the first three objectives and the Custom CNN implementation are complete at this draft stage. Model-performance and efficiency claims must be inserted later from traceable experiment records.

## 3. Dataset and exploratory data analysis

### 3.1 Dataset identity and provenance

The project uses the **color** configuration of the maintained `mohanty/PlantVillage` dataset [@plantvillage_huggingface]. Acquisition is locked to repository revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7`. The source card declares the dataset under CC BY-SA 3.0; attribution and share-alike requirements must therefore be reviewed before redistributing any derivative data bundle.

The locked source contains 54,305 images in 38 crop–condition classes. This observed total is one image lower than the 54,306 images reported in the original publication [@mohanty2016plant], so all project calculations use the locked distribution rather than copying the publication total. The archive and supporting split/grouping files are protected by SHA-256 hashes in `data/plantvillage_source.lock.json`. The images remain outside Git history.

### 3.2 Integrity and leakage audit

All 54,305 source images were fully decoded successfully: 54,304 are RGB and one is RGBA. The RGBA image is retained because it is readable and the dataset loader deterministically converts every input to RGB. There were no empty or corrupted files, and all source images were 256 × 256 pixels.

The audit found 21 groups containing 42 byte-identical images. Five exact-duplicate groups crossed the upstream train/test boundary. Difference hashing also produced 249 visually similar candidate pairs, including 41 cross-boundary candidates. These perceptual matches were treated as screening evidence rather than automatic duplicates. Human review classified 15 cross-boundary candidates as related and 26 as false positives. Combining the related perceptual evidence with the five exact cross-boundary copies produced 19 unique official-training images for exclusion. Source files were not deleted or altered; the decisions are preserved in review and exclusion ledgers.

The source audit deliberately reports a failing leakage status before remediation because the five exact groups violate the project criterion. This is an expected finding, not an image-readability failure. The downstream frozen manifests are the remediated dataset used for modelling.

### 3.3 Frozen partition design

The source provides 43,596 official-training images and 10,709 official-test images. After excluding 19 leakage-risk training images, the cleaned official-training pool contains 43,577 images. Validation data were then selected only from this cleaned pool using a 15% target, seed 42, class-stratified stable hashing, and physical-leaf grouping. The official test set was preserved in full.

| Partition | Images | Role |
|---|---:|---|
| Training | 37,037 | Parameter fitting and training-only augmentation |
| Validation | 6,540 | Model selection, early stopping, and threshold-independent review |
| Test | 10,709 | Locked final evaluation only |
| **Total used** | **54,286** | Source total minus 19 excluded training images |

Upstream metadata mapped 41,111 images (75.70% of the source) to an unambiguous physical-leaf group. Unresolved images received unique fallback group identifiers rather than invented shared identities. This conservative rule prevents known groups from crossing partitions, although incomplete upstream grouping remains a threat to validity because relatedness among unresolved images cannot be proven absent. Automated validation confirms that image paths and resolved groups do not overlap among training, validation, and test manifests.

The frozen bundle contains a contiguous alphabetical mapping for all 38 classes and is identified by manifest-bundle SHA-256 `19ca82ed9a1832dbeca228bbcb35274cd3362758e918818e64587f4d0bd1306c`. Training must fail closed if this bundle, its class map, or the locked source revision changes unexpectedly.

### 3.4 Class distribution and image characteristics

The training partition is substantially imbalanced. `Orange___Haunglongbing_(Citrus_greening)` is the largest class with 3,851 images (10.40% of training), whereas `Potato___healthy` is the smallest with 104 images (0.28%). The resulting largest-to-smallest ratio is 37.03:1. The mean and median class sizes are 974.7 and 756.5 images, respectively. Consequently, accuracy alone could obscure weak performance on minority classes; macro F1 and per-class precision, recall, and F1 are required alongside accuracy in the final evaluation.

All 54,286 images referenced by the frozen manifests are square 256 × 256 images with an aspect ratio of 1.0. A deterministic training-only sample grid selects one image per class using seed 42. Neither sample selection nor EDA displays locked test images, preventing visual inspection of test examples from influencing model design.

**Figure insertion points for the consolidated report:**

- class-distribution chart generated by `notebooks/01_dataset_exploration.ipynb`;
- deterministic 38-class training sample grid generated by the same notebook; and
- concise data-quality table based on `docs/DATA_VALIDATION.md`.

These assets should be copied into `reports/figures/` only after reviewer approval; generated working figures are not themselves evidence of model performance.

## 4. Data preprocessing and augmentation

Every image is opened through Pillow and converted to three-channel RGB. The common model input is 3 × 224 × 224. The training pipeline applies the following operations in order:

1. resize to 224 × 224;
2. random horizontal flip with probability 0.5;
3. random rotation within ±20 degrees;
4. random affine scaling between 0.8 and 1.2;
5. brightness jitter of 0.2 and contrast jitter of 0.2;
6. conversion to a floating-point tensor; and
7. channel-wise normalization using mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

Validation and test images use deterministic resize, centre crop, tensor conversion, and the same normalization. They receive no flip, rotation, affine scaling, or colour jitter. Keeping augmentation training-only avoids stochastic evaluation and prevents validation or test information from shaping synthetic training views. Using one normalization contract for all architectures also removes a preprocessing difference that could otherwise confound the model comparison.

The transform builder validates image dimensions, probabilities, rotation and colour-jitter magnitudes, zoom bounds, and normalization vectors before constructing a pipeline. Automated tests demonstrate that all stochastic transforms are absent from validation/test, repeated evaluation transforms produce identical tensors, and custom normalization values are honoured. `notebooks/02_preprocessing.ipynb` additionally verifies the contract against the checksum-validated frozen manifests and visualizes seeded augmentation examples drawn only from training.

Augmentation should be understood as limited robustness training, not as evidence of field generalization. The chosen operations vary orientation, scale, brightness, and contrast, but they do not reproduce cluttered backgrounds, occlusion, multiple leaves, severe viewpoint changes, unseen cultivars, different cameras, or novel diseases.

## 6. Model architecture: Custom CNN baseline

### 6.1 Purpose and configuration

The Custom CNN is initialized from scratch and contains no pretrained backbone. It is designed as a compact baseline: if a transfer-learning model performs better under the same data and evaluation contract, that gain can be considered relative to a reproducible non-pretrained reference rather than to an undocumented network.

The authoritative configuration specifies convolution widths `[32, 64, 128]`, 256 dense units, and dropout probability 0.5. The number of output units is derived from the frozen 38-class mapping rather than hard-coded. Invalid class counts, channel specifications, dense widths, and dropout values are rejected during construction.

### 6.2 Layer structure and tensor shapes

Each feature block uses a bias-free 3 × 3 convolution with padding one, followed by batch normalization, ReLU activation, and 2 × 2 max pooling. Batch normalization is included as an explicit part of the model architecture [@ioffe2015batch], and ReLU supplies the hidden non-linearity [@glorot2011rectifier]. Adaptive global average pooling reduces every final feature map to one scalar, limiting the dense head's parameter count and removing dependence on a fixed post-convolution spatial size [@lin2013network]. Dropout regularizes the 256-unit hidden representation during training [@srivastava2014dropout].

| Stage | Operation | Output shape (batch = 1) | Parameters |
|---|---|---:|---:|
| Input | RGB tensor | 1 × 3 × 224 × 224 | 0 |
| Conv block 1 | Conv 3→32, BatchNorm, ReLU, MaxPool | 1 × 32 × 112 × 112 | 928 |
| Conv block 2 | Conv 32→64, BatchNorm, ReLU, MaxPool | 1 × 64 × 56 × 56 | 18,560 |
| Conv block 3 | Conv 64→128, BatchNorm, ReLU, MaxPool | 1 × 128 × 28 × 28 | 73,984 |
| Global pool | Adaptive average pooling to 1 × 1 | 1 × 128 × 1 × 1 | 0 |
| Hidden layer | Flatten, Linear 128→256, ReLU | 1 × 256 | 33,024 |
| Regularization | Dropout, p = 0.5 | 1 × 256 | 0 |
| Output | Linear 256→38 logits | 1 × 38 | 9,766 |
| **Total** |  |  | **136,262** |

The output is a vector of 38 unnormalized logits suitable for multiclass cross-entropy loss. The architecture notebook obtains the shapes with forward hooks and asserts that the stage-level parameter counts sum to 136,262. Unit tests cover configurable widths, output shape, the exact default parameter count, invalid configurations, and one optimizer step. The optimizer-step smoke test confirms a finite cross-entropy loss, finite gradients for every trainable parameter, and an actual output-layer weight update. These checks establish implementation viability, not predictive performance.

### 6.3 Pending experiment evidence

The following statements are intentionally **not yet available** and must not be inferred from the architecture tests:

- pilot or final training loss;
- validation or test accuracy, macro F1, per-class metrics, ROC-AUC, or confusion matrices;
- convergence, underfitting, overfitting, or seed stability;
- checkpoint size, latency, throughput, or memory usage; and
- comparison or ranking against ResNet-50, EfficientNet-B0, or MobileNetV3-Large.

These results require the approved shared single-model training, checkpoint/resume, and standalone evaluation interfaces. Official training must start from the team-reviewed `training-v1.0` tag. Each accepted result must identify the run ID, seeds, Git SHA, configuration, manifest checksum, hardware/software environment, selected validation checkpoint, and model checksum. The locked test set must be evaluated only after model selection is complete.

## Threats to validity and practical limitations

Several limitations should constrain interpretation of the eventual results. First, PlantVillage images use centred leaves and controlled backgrounds [@mohanty2016plant], so high in-distribution performance would not demonstrate reliable diagnosis in farms or gardens. Second, the 37.03:1 training imbalance may produce uneven class behaviour even when aggregate accuracy is high. Third, upstream physical-leaf grouping is unambiguous for 75.70% of source images; unique fallback groups prevent invented relationships but cannot prove that all unresolved images are independent. Fourth, crop and condition are encoded jointly as 38 labels, so the task does not directly test recognition of unseen crops or diseases. Finally, predictions must be presented as research outputs rather than agricultural advice: incorrect classification could lead to inappropriate treatment, and neither confidence calibration nor out-of-distribution rejection has yet been established.

## Reviewer and consolidation notes

- Member 4 should review the introduction/problem framing.
- Member 3 should review Dataset/EDA numbers, split language, and metric rationale.
- Member 2 should review preprocessing and consolidate the architecture/training-method description.
- Replace section numbering only when merging into the team report; preserve the citation keys.
- Insert result paragraphs only from the accepted experiment record. Do not overwrite the explicit pending-evidence boundary with provisional smoke-test output.
