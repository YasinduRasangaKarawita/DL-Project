# Four-Member Project Execution Plan

This is the working agreement and completion plan for the SE4050 plant-disease classification project. Replace `Member 1`–`Member 4` with names, registration numbers, GitHub usernames, and contact details before work begins.

## 1. Project objective

Build and fairly compare four deep-learning image classifiers on one real, publicly available plant-disease dataset:

1. Custom CNN trained from scratch
2. ResNet-50 using transfer learning
3. EfficientNet-B0 using transfer learning
4. MobileNetV3-Large using transfer learning

The final comparison must cover predictive performance, generalization, stability, computational cost, practical limitations, and deployment suitability. The project is not complete merely because four models can train.

## 2. Team roster

Complete this table in the first team meeting.

| Role | Name | Registration number | GitHub username | Primary model |
|---|---|---|---|---|
| Member 1 — Data Lead | TBD | TBD | TBD | Custom CNN |
| Member 2 — Training Lead | TBD | TBD | TBD | ResNet-50 |
| Member 3 — Evaluation Lead | TBD | TBD | TBD | EfficientNet-B0 |
| Member 4 — Integration Lead | TBD | TBD | TBD | MobileNetV3-Large |

Primary ownership means leading and documenting the work. It does not mean working alone. Every member must understand the complete data pipeline, all four architectures, experimental design, results, and limitations for the viva.

## 3. Non-negotiable rules

- Use a real, cited public dataset. The images currently in `data/raw/` are synthetic smoke-test data and cannot support the final report.
- Confirm that the chosen dataset was not already used in a course lab or tutorial.
- Never commit images, credentials, Google Drive tokens, or model binaries to ordinary Git history.
- Use one frozen class mapping and one frozen train/validation/test split for every model.
- Use validation data for model selection. Do not use the test set while tuning.
- Train from the same tagged Git commit and declared configuration.
- Evaluate all final checkpoints through the same evaluation code.
- Benchmark all models on the same hardware session for a fair efficiency comparison.
- Keep every result traceable to its config, seed, Git SHA, split hash, runtime, and checkpoint hash.
- Every member must contribute meaningful, reviewable work regularly. Do not create artificial or backdated commits.
- All claims in the report must be supported by preserved results.

## 4. Work that must happen before parallel model training

The group must finish these shared foundations first.

### 4.1 Repository administration

The group leader should:

- [ ] Add all four members as repository collaborators.
- [ ] Confirm the default branch is `main`.
- [ ] Require pull requests before merging into `main` if repository settings allow it.
- [ ] Require passing CI and at least one reviewer.
- [ ] Create GitHub labels such as `data`, `training`, `evaluation`, `model`, `app`, `report`, and `blocked`.
- [ ] Create one GitHub issue for every deliverable described in this plan.
- [ ] Create a simple board with `Backlog`, `Ready`, `In progress`, `Review`, and `Done` columns.
- [ ] Add the real team roster to this document.
- [ ] Connect the GitHub remote and make the initial scaffold commit.

Suggested initial commit:

```bash
git add .
git commit -m "chore: initialize reproducible project structure"
git remote add origin https://github.com/ORGANIZATION/REPOSITORY.git
git push -u origin main
```

### 4.2 Individual onboarding

Every member should complete the following independently:

```bash
git clone https://github.com/ORGANIZATION/REPOSITORY.git
cd REPOSITORY
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest
```

Then each member should:

- [ ] Configure their own Git name and university email.
- [ ] Read `README.md`, `CONTRIBUTING.md`, `docs/EXPERIMENT_PROTOCOL.md`, and `docs/COLAB.md`.
- [ ] Run the tests and record any setup problem in a GitHub issue.
- [ ] Run a declared one-epoch pilot with `python run_pipeline.py train --model <name> --seed 42 --epochs 1 --allow-dirty`.
- [ ] Confirm that synthetic results are not used in the report.
- [ ] Create a small documentation pull request to prove the branch/review workflow works.

### 4.3 Freeze the research contract

All members must agree on and document:

- exact dataset source, version or commit, license, citation, and checksum;
- included classes and any excluded/corrupted samples;
- group identifier used to prevent leakage, such as a physical leaf ID;
- train/validation/test split manifests and hashes;
- image size, normalization, and train-only augmentation policy;
- primary validation metric;
- epoch budget, early stopping rule, and allowed hyperparameter search;
- final seeds, preferably `42`, `123`, and `2026` if compute permits;
- checkpoint and result schemas;
- final evaluation and hardware benchmarking procedure.

After review, tag the agreed implementation before expensive training:

```bash
git tag -a training-v1.0 -m "Freeze data and training protocol"
git push origin training-v1.0
```

No member should launch a final run before this tag exists.

## 5. Shared interfaces

The shared implementation is backed by reusable functions in `src/`. `run_pipeline.py`
provides the current train/resume and standalone evaluation commands:

| Interface | Responsibility |
|---|---|
| `scripts/prepare_data.py` | Validate data, detect duplicates/leakage, and create/freeze split manifests |
| `run_pipeline.py train` | Train or resume one selected architecture and seed |
| `run_pipeline.py evaluate` | Evaluate one frozen checkpoint without training |
| `scripts/benchmark.py` | Measure latency, throughput, memory, parameters, and model size consistently |
| `scripts/download_models.py` | Download release assets and verify SHA-256 checksums |

Target training interface:

```bash
python run_pipeline.py train \
  --config configs/config.yaml \
  --model resnet50 \
  --seed 42 \
  --artifact-root /content/drive/MyDrive/SE4050
```

Target evaluation interface:

```bash
python run_pipeline.py evaluate \
  --checkpoint models/release/resnet50_best.pt \
  --split test \
  --confirm-locked-test
```

### 5.1 Checkpoint contract

Every inference checkpoint must contain:

- architecture and model state;
- number and ordered names of classes;
- `class_to_idx` mapping;
- input size, color mode, normalization, and preprocessing version;
- seed and best validation metric;
- dataset version and split-manifest hash;
- Git commit SHA and relevant library versions.

Every resumable checkpoint must additionally contain optimizer, scheduler, gradient scaler, epoch, early-stopping state, and random-number-generator state.

### 5.2 Result contract

Each run must write to a unique run directory:

```text
results/<run_id>/<model_name>/
├── run_metadata.json
├── history.csv
├── validation/
│   ├── metrics.json
│   └── predictions.npz
└── test/
    ├── metrics.json
    └── predictions.npz
```

Use a run ID such as:

```text
20260927-1430_resnet50_seed42_a1b2c3d
```

## 6. Member 1 — Data Lead and Custom CNN owner

### Primary responsibilities

Member 1 owns the integrity of the dataset and the baseline model.

### Tasks

- [ ] Confirm the final public dataset with the team and verify its license.
- [ ] Implement a reproducible download/access procedure without committing the dataset.
- [ ] Record source URL, version, date, archive checksum, sample count, and class counts in `data/README.md`.
- [ ] Detect unreadable files, exact duplicates, perceptual near-duplicates, and class-folder mistakes.
- [ ] Determine the correct grouping key for leakage prevention.
- [ ] Produce one immutable grouped split and tests proving groups do not cross partitions.
- [ ] Generate EDA: class distribution, image dimensions, sample grid, imbalance, and data-quality findings.
- [ ] Implement or verify training-only augmentations and validation/test transforms.
- [ ] Implement the Custom CNN baseline with documented layer shapes, activations, normalization, dropout, and parameter count.
- [ ] Train the Custom CNN using every agreed seed.
- [ ] Preserve checkpoints, histories, metadata, and hashes.
- [ ] Draft the Dataset/EDA, Preprocessing, and Custom CNN report content.

### Required pull requests

1. Dataset provenance and acquisition documentation
2. Data validation and leakage tests
3. Frozen split manifests and class mapping
4. Custom CNN implementation/tests
5. Custom CNN experiment records and report material

### Acceptance criteria

- Another member can reproduce the data preparation from the documentation.
- The same sample/group cannot appear in multiple partitions.
- Class order is explicit and stable.
- Custom CNN trains from scratch and produces a valid standardized checkpoint.
- No synthetic data claim appears in final material.

## 7. Member 2 — Training Lead and ResNet-50 owner

### Primary responsibilities

Member 2 owns the common training engine, reproducibility, checkpoint/resume behavior, and ResNet-50 experiments.

### Tasks

- [ ] Refactor training so one model and seed can be selected from the command line.
- [ ] Make configuration values authoritative; remove conflicting hard-coded settings.
- [ ] Support CPU/CUDA device selection, mixed precision, deterministic seeding, and clear logs.
- [ ] Implement separate feature-extraction and fine-tuning phases.
- [ ] Keep frozen BatchNorm statistics fixed during feature extraction.
- [ ] Restore the best phase-one checkpoint before fine-tuning.
- [ ] Rebuild the optimizer and scheduler correctly when parameters are unfrozen.
- [ ] Monitor validation macro F1 and implement early stopping consistently.
- [ ] Save both `best_inference.pt` and `last_resume.pt` after every epoch.
- [ ] Test interruption and resume equivalence on a short run.
- [ ] Implement ResNet-50 transfer learning and document what is frozen/unfrozen.
- [ ] Train ResNet-50 using every agreed seed.
- [ ] Preserve checkpoints, histories, metadata, and hashes.
- [ ] Draft the Experimental Design, Training Procedure, and ResNet-50 report content.

### Required pull requests

1. Single-model training CLI and model registry
2. Standard checkpoint/resume format
3. Fine-tuning, scheduler, and BatchNorm corrections
4. Training reproducibility tests
5. ResNet-50 experiments and report material

### Acceptance criteria

- Any model can be trained using the same CLI and configuration system.
- An interrupted run can resume without losing optimizer/scheduler state.
- The checkpoint contains every field in the shared contract.
- ResNet-50 training uses only the agreed data and split.
- Another member can reproduce the run from its metadata.

## 8. Member 3 — Evaluation Lead and EfficientNet-B0 owner

### Primary responsibilities

Member 3 owns metrics, statistical comparison, error analysis, efficiency benchmarking, and EfficientNet-B0 experiments.

### Tasks

- [ ] Implement evaluation that never calls training code.
- [ ] Validate architecture, class order, preprocessing, dataset version, and split hash before loading a checkpoint.
- [ ] Compute accuracy, top-3 accuracy, macro/weighted precision, recall, and F1.
- [ ] Compute per-class metrics and multiclass one-vs-rest ROC-AUC where statistically valid.
- [ ] Produce raw and normalized confusion matrices.
- [ ] Save sample-level predictions and probabilities for reproducible error analysis.
- [ ] Aggregate multiple seeds and report mean ± standard deviation.
- [ ] Implement warm-up and repeated hardware benchmarks with CUDA synchronization.
- [ ] Record parameter count, checkpoint size, latency, throughput, peak memory, and FLOPs/MACs methodology.
- [ ] Analyze common confusions, low-confidence predictions, class imbalance, and failure cases.
- [ ] Implement EfficientNet-B0 transfer learning and train every agreed seed.
- [ ] Preserve checkpoints, histories, metadata, and hashes.
- [ ] Lead the Results/Comparison and Critical Analysis report sections.

### Required pull requests

1. Standalone evaluation CLI and checkpoint validation
2. Metrics, predictions, confusion matrices, and ROC-AUC
3. Multi-seed aggregation and comparison tables
4. Fair performance benchmarking
5. EfficientNet-B0 experiments and critical analysis material

### Acceptance criteria

- One command can evaluate all four checkpoints on the same test manifest.
- Metrics can be regenerated from the saved predictions.
- Hardware results are measured in one consistent session.
- Statistical summaries identify the seed count and variation.
- Conclusions distinguish evidence from speculation.

## 9. Member 4 — Integration Lead and MobileNetV3-Large owner

### Primary responsibilities

Member 4 owns the shared Colab workflow, model collection/release, demonstration app, video coordination, and MobileNetV3-Large experiments.

### Tasks

- [ ] Create one Colab entry notebook that clones a tag rather than duplicating source code.
- [ ] Mount the current member's Drive and stage the dataset under fast `/content` storage.
- [ ] Support one-model training, resume, evaluation, and artifact export through repository scripts.
- [ ] Record GPU, CUDA, package versions, Git SHA, configuration, and split hash.
- [ ] Test checkpoint persistence by disconnecting and resuming a smoke run.
- [ ] Implement MobileNetV3-Large transfer learning and train every agreed seed.
- [ ] Collect selected inference checkpoints from all four members.
- [ ] Generate `model_manifest.json` and `checksums.sha256`.
- [ ] Publish versioned final inference files as GitHub Release assets.
- [ ] Implement model downloading and checksum verification.
- [ ] Update Streamlit to load the release checkpoint and metadata safely.
- [ ] Make the app fail closed when weights or class metadata are missing.
- [ ] Add input validation, uncertainty language, and a research-only disclaimer.
- [ ] Coordinate the 10-minute video script, recording, timing, and upload.
- [ ] Draft the MobileNetV3, deployment discussion, demonstration, and conclusion material.

### Required pull requests

1. Reproducible Colab training/resume notebook
2. Model manifest, release, download, and checksum workflow
3. MobileNetV3-Large experiments and report material
4. Safe Streamlit inference integration
5. Video plan and final demonstration checklist

### Acceptance criteria

- A teammate can train/resume from a clean Colab runtime.
- The final evaluator can retrieve all four models without accessing four Drives.
- Every downloaded model is checksum-verified before loading.
- The app uses exactly the same class map and preprocessing as evaluation.
- The video remains within the assignment time limit and covers contributions/results.

## 10. Cross-review rotation

Use this default review rotation so no component has only one knowledgeable owner:

| Author | Primary reviewer |
|---|---|
| Member 1 | Member 2 |
| Member 2 | Member 3 |
| Member 3 | Member 4 |
| Member 4 | Member 1 |

For high-risk changes involving split generation, checkpoint formats, test evaluation, or final report numbers, require a second reviewer.

Reviewers should verify behavior and evidence, not only style. A reviewer must be able to explain what changed and why.

## 11. Git and communication workflow

### Branch naming

```text
feature/<issue-number>-<short-description>
fix/<issue-number>-<short-description>
docs/<issue-number>-<short-description>
experiment/<issue-number>-<model>-<seed>
```

### Commit examples

```text
feat(data): add grouped split leakage validation
feat(training): save resumable scheduler state
fix(training): attach fine-tune scheduler to new optimizer
test(models): verify output shapes for all architectures
docs(report): add dataset provenance and limitations
```

### Pull-request requirements

Every pull request must include:

- linked issue and acceptance criteria;
- explanation of the change;
- tests or manual verification;
- screenshots/figures only when relevant;
- experiment metadata if a result changed;
- no unrelated generated files.

### Communication rhythm

- Daily: ten-minute check-in covering completed work, next action, and blocker.
- Every two days: merge reviewed work and update the GitHub board.
- After every experiment: immediately upload durable artifacts and metadata.
- Before reporting a number: have a second member trace it to the raw result.
- Important decisions: record them in an issue or pull request, not only in chat.

## 12. Colab and model handoff workflow

Each member mounts only their own Drive while training:

```text
MyDrive/SE4050_PlantDisease/
├── datasets/
├── checkpoints/<model>/<run_id>/
└── experiment_backups/<run_id>/
```

When a run finishes:

1. Select the checkpoint using validation macro F1 only.
2. Verify its metadata and SHA-256 hash.
3. Send the inference checkpoint, manifest record, history, and validation metrics to Member 4.
4. Keep the resumable checkpoint in the owner's Drive.
5. Member 4 uploads selected inference files to a draft GitHub Release.
6. Member 3 downloads all models into one clean Colab session.
7. Member 3 verifies checksums and runs the locked test evaluation once.

No final comparison should combine independently measured latency values from different Colab GPU types.

## 13. Report ownership

| Report section | Lead | Required reviewers |
|---|---|---|
| Introduction and problem definition | Member 1 | Member 4 |
| Related Work | All members, one model family each | Member 3 consolidates |
| Dataset and EDA | Member 1 | Member 3 |
| Preprocessing | Member 1 | Member 2 |
| Experimental Design | Member 2 | Member 3 |
| Model Architectures | Each model owner | Member 2 consolidates |
| Results and Comparison | Member 3 | All members |
| Critical Analysis | Member 3 leads | All members contribute |
| Application/practical limitations | Member 4 | Member 1 |
| Conclusion | Member 4 | All members |
| References and formatting | All members | Group leader final audit |

Report figures and tables must be copied into `reports/figures/` and `reports/tables/` only after verification. Every result must identify the source run, seed count, and evaluation conditions.

Critical analysis should discuss:

- why architectures behaved differently;
- underfitting, overfitting, convergence, and stability;
- class-specific errors and likely dataset biases;
- accuracy versus size, latency, memory, and practical constraints;
- limits of controlled-background datasets and expected field-domain shift;
- confidence calibration and consequences of incorrect predictions;
- threats to validity and what further work is required.

## 14. Six-day completion schedule

This schedule is aligned to the stated September 30, 2026 deadline. Finish by September 29 and keep September 30 as submission/emergency buffer.

### September 24 — Repository and decision freeze

- Add collaborators, issues, board, and protections.
- Assign real names to the four roles.
- Confirm dataset eligibility, source, license, and citation.
- Agree on metrics, seeds, split strategy, and model protocol.
- Each member completes onboarding and one genuine pull request.

### September 25 — Shared infrastructure

- Member 1: real dataset validation and grouped split proposal.
- Member 2: single-model training/checkpoint/resume interface.
- Member 3: standalone evaluation and result schema.
- Member 4: Colab and Drive workflow.
- Integrate and run a tiny end-to-end smoke test for all four models.

### September 26 — Freeze and pilot

- Review leakage checks, class order, preprocessing, and checkpoint compatibility.
- Fix all protocol-critical defects.
- Freeze the split and create tag `training-v1.0`.
- Run one short pilot per model and verify resume/evaluation.
- Start writing report sections from verified design facts.

### September 27–28 — Final training

- Each member trains their assigned model using agreed seeds.
- Save durable checkpoints and metadata after every epoch.
- Review validation curves and select by validation metric only.
- Deliver selected inference checkpoints to Member 4.
- Continue report writing without inventing results.

### September 29 — Locked evaluation and report completion

- Member 3 evaluates all frozen models on the locked test set.
- Benchmark all models in one hardware session.
- Generate final tables, plots, confusion matrices, and error analysis.
- All members review the critical discussion and verify every number.
- Member 4 publishes the model release and records the demo.
- Complete the report, references, repository instructions, and contribution record.

### September 30 — Submission buffer

- Run a clean-clone reproducibility check.
- Verify links, permissions, video access, file names, and ZIP contents.
- Generate and inspect the final report PDF.
- Submit early enough to correct upload or permission problems.
- Do not plan new experiments unless a critical result is invalid.

## 15. Definition of done

### Code and reproducibility

- [ ] Clean clone can install dependencies and run tests.
- [ ] Real dataset access/provenance is documented.
- [ ] Split manifests and hashes are frozen.
- [ ] Four distinct deep-learning models train through one shared interface.
- [ ] Checkpoint resume is tested.
- [ ] Configuration controls actual behavior.
- [ ] All final runs record seed, Git SHA, environment, data version, and split hash.

### Experiments and evaluation

- [ ] All models use the same split and evaluation protocol.
- [ ] Checkpoints are selected only using validation data.
- [ ] Final test evaluation is performed by the shared evaluator.
- [ ] Accuracy, precision, recall, F1, ROC-AUC, and confusion matrices are reported as applicable.
- [ ] Efficiency, stability, complexity, and practical limitations are compared.
- [ ] Failure cases and domain-shift limitations are discussed.
- [ ] Final model files are versioned, checksummed, and downloadable.

### Report and presentation

- [ ] Required report headings are used.
- [ ] Dataset, papers, libraries, and architectures are cited correctly.
- [ ] Every table/figure has a caption and traceable source.
- [ ] Every member's contribution is accurate and evidenced by Git activity.
- [ ] Every member can explain the full pipeline and answer viva questions.
- [ ] The demonstration video is no longer than 10 minutes.

### Submission package

- [ ] Turnitin report is submitted using the group leader's required naming convention.
- [ ] Code is submitted through Gradescope as instructed.
- [ ] CourseWeb ZIP is named with the group leader's registration number.
- [ ] ZIP contains `Members.txt`.
- [ ] ZIP contains the final `Report.pdf`.
- [ ] ZIP contains the Turnitin similarity report.
- [ ] ZIP contains `Submission.txt` with the GitHub and YouTube links.
- [ ] GitHub repository permissions allow markers to access it.
- [ ] YouTube link works in an incognito/private browser.

## 16. Experiment handoff template

Each member should send this completed record with every candidate model:

```text
Model:
Owner:
Run ID:
Git commit/tag:
Configuration file:
Dataset version/checksum:
Split-manifest checksum:
Seed:
Colab GPU/runtime:
Best epoch:
Best validation macro F1:
Inference checkpoint path:
Inference checkpoint SHA-256:
Resume checkpoint path:
History path:
Known warnings or deviations:
Reviewer:
```

## 17. Viva preparation questions

Every member should be able to answer:

1. Why was this dataset selected, and what are its limitations?
2. How was leakage prevented?
3. Why is validation separate from the final test set?
4. How does the Custom CNN differ from each transfer-learning model?
5. What was frozen and later unfrozen in each model?
6. Why was macro F1 used for model selection?
7. Which classes were confused most often and why?
8. Is the apparent performance difference stable across seeds?
9. How were latency and throughput measured fairly?
10. Which model should be deployed under different resource constraints?
11. Why might performance fall on real field photographs?
12. What exactly did each team member contribute?
