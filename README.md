# Plant Disease Classification Benchmark — EfficientNet-B0 Focus

> **Branch / Component:** `feature/efficientnet-b0`  
> **Model Owner:** Member 3 — Evaluation Lead & EfficientNet-B0 Architecture Lead  
> **Primary Deliverable:** EfficientNet-B0 compound scaling transfer learning, multi-metric standalone evaluation suite, confusion matrix analysis, and multi-seed statistical comparison.

A reproducible PyTorch project for comparing four deep-learning architectures on the public PlantVillage plant-disease dataset:

1. Custom CNN (baseline trained from scratch)
2. ResNet-50 (ImageNet transfer learning)
3. **EfficientNet-B0 (ImageNet transfer learning — THIS WORKSPACE)**
4. MobileNetV3-Large (ImageNet transfer learning)

The goal is a fair, reproducible comparison of predictive quality, generalization, computational cost, and deployment suitability under the shared experimental protocol.

> [!IMPORTANT]
> The locked PlantVillage color dataset is acquired under `data/raw/plantvillage/color/` and split using frozen manifests in `data/splits/`. Checkpoints and metrics are traceable to configuration, random seed, and Git commit.

## Workspace Focus: EfficientNet-B0 Compound Scaling

- **Architecture:** EfficientNet-B0 balancing network depth, channel width, and input resolution ($\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$).
- **Key Modules:** MBConv inverted bottlenecks with depthwise separable convolutions and Squeeze-and-Excitation (SE) channel attention.
- **Pretrained Weights:** ImageNet-1K (`EfficientNet_B0_Weights.DEFAULT`).
- **Phase 1 (Feature Extraction):** Backbone frozen, only custom dropout ($p=0.3$) and linear classification head (1280→38) trained (48,678 trainable parameters).
- **Phase 2 (Fine-Tuning):** `features.7` and `features.8` unfrozen with reduced learning rate $1\times 10^{-4}$ (1,178,070 trainable parameters).
- **Verification:** Unit tests in `tests/test_efficientnet_b0.py`, interactive analysis in `notebooks/05_efficientnet.ipynb`, and report draft in `reports/member3_report_draft.md`.

## Quick start

### Local development

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
```

Train EfficientNet-B0 with configured seed (locked test set is never evaluated during training):

```powershell
python run_pipeline.py train --model efficientnet_b0 --seed 42
```

For a one-epoch provisional pilot from an uncommitted development branch:

```powershell
python run_pipeline.py train --model efficientnet_b0 --seed 42 --epochs 1 --allow-dirty
```

Resume the same run from its durable checkpoint:

```powershell
python run_pipeline.py train --model efficientnet_b0 --seed 42 --resume models/efficientnet/<run-id>/last_resume.pt
```

Evaluate the selected checkpoint on validation data:

```powershell
python run_pipeline.py evaluate --checkpoint models/efficientnet/<run-id>/best_inference.pt --split validation
```


Locked-test evaluation is deliberately separate and additionally requires
`--confirm-locked-test`. Use it only once, after the team has completed model selection.

Acquire the locked PlantVillage color dataset:

```powershell
python scripts/prepare_data.py download
```

Validate the acquired images and locked official split:

```powershell
python scripts/prepare_data.py validate
```

Validate the reviewed exclusions and frozen split bundle before training:

```powershell
python scripts/prepare_data.py validate-review
python scripts/prepare_data.py validate-exclusions
python scripts/prepare_data.py validate-manifests
```

The committed source lock at `data/plantvillage_source.lock.json` makes the download command
fetch the reviewed revision rather than a moving `main`. Only the Data Lead should
deliberately refresh that lock. The frozen split contains 37,037 training, 6,540 validation,
and 10,709 test images across 38 classes. Its bundle checksum is
`19ca82ed9a1832dbeca228bbcb35274cd3362758e918818e64587f4d0bd1306c`.
Do not run final training until the shared interface has passed a GPU pilot and resume test,
received the required review, and the reviewed commit is tagged `training-v1.0`.

Run the Streamlit prototype:

```powershell
streamlit run app/streamlit_app.py
```

### Google Colab training

GPU training should be performed in Colab while GitHub remains the source of truth for code and configuration. Checkpoints must be written to Google Drive after every epoch because Colab runtimes are temporary.

See `docs/COLAB.md` for the complete clone, dataset, checkpoint, resume, and GitHub Release workflow.

## Project structure

```text
.
├── .github/                  # Pull-request template and CI checks
├── app/                      # Streamlit inference prototype
├── configs/                  # Version-controlled experiment configuration
├── data/
│   ├── raw/                  # Local source images; ignored by Git
│   ├── processed/            # Generated caches; ignored by Git
│   └── splits/               # Final reproducible split manifests
├── docs/                     # Colab, protocol, structure, and team guides
├── figures/                  # Generated exploratory figures; ignored by default
├── models/                   # Local checkpoints; ignored by Git
├── notebooks/                # Thin, reviewable experiment notebooks
├── reports/                  # Report source and final curated report assets
├── results/                  # Generated metrics and predictions; ignored by default
├── scripts/                  # Utility and orchestration entry points
├── src/
│   ├── data/                 # Dataset loading, validation, and transforms
│   ├── evaluation/           # Metrics, plots, comparisons, error analysis
│   ├── models/               # Model definitions
│   ├── training/             # Trainer, callbacks, and tracking
│   └── utils/                # Configuration, logging, device, and seed helpers
├── tests/                    # Fast automated tests
├── run_pipeline.py           # Current end-to-end pipeline entry point
├── requirements.txt          # Runtime dependencies
└── requirements-dev.txt      # Development/test dependencies
```

The ownership and purpose of every generated directory is documented in `docs/PROJECT_STRUCTURE.md`.

## Experiment rules

All four models must use the same:

- source dataset and immutable split manifests;
- input resolution and normalization;
- allowed training augmentations;
- model-selection metric (validation macro F1 recommended);
- test protocol and evaluation code;
- hardware session for final latency/throughput comparison.

Use at least three fixed seeds (`42`, `123`, `2026`) if compute permits. Report mean and standard deviation, per-class metrics, confusion matrices, one-vs-rest ROC-AUC, parameter count, checkpoint size, latency, throughput, memory usage, learning curves, and qualitative failure cases. See `docs/EXPERIMENT_PROTOCOL.md` before final training.

## Model files

`.pt`, `.pth`, `.ckpt`, and `.onnx` files are intentionally excluded from normal Git history. During training, save resumable checkpoints to Google Drive. Publish only the four final inference checkpoints as assets on a versioned GitHub Release, together with a model manifest and SHA-256 checksums.

The repository should contain links to the release—not the binaries themselves. See `models/README.md`.

## Team workflow

Every member should own one model plus one cross-cutting responsibility, work through issues and feature branches, and open small pull requests reviewed by another member. Each member must make meaningful, traceable contributions every week; do not manufacture or backdate commits.

The four-person ownership plan and naming conventions are in `docs/TEAM_WORKFLOW.md`. General contribution rules are in `CONTRIBUTING.md`.

## Reproducibility checklist

Before accepting a final run, preserve:

- Git commit SHA and configuration file;
- dataset source/version, checksum, class map, and split manifest;
- random seed and package versions;
- GPU model, CUDA version, and Colab runtime details;
- best validation checkpoint and last resumable checkpoint;
- training history, test predictions, metrics, and timing methodology;
- SHA-256 checksums for published model files.

## Academic and safety notes

- Cite the dataset, original dataset paper, and every model architecture/source.
- Do not claim synthetic smoke-test artifacts are real experimental results.
- The app is a research demonstration, not a substitute for agricultural diagnosis. Confidence scores must be calibrated and out-of-distribution inputs must be handled before any real-world use.
- No software or dataset license has been selected for redistribution yet. Confirm the dataset license and add a project license only after all group members agree.
