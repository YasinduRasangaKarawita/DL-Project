# Experiment protocol

This document is the pre-training checklist. The shared one-model interface implements the
required training and evaluation contracts, but it remains provisional until the GPU pilot,
independent review, and `training-v1.0` tag are complete.

## Completed data foundations

- The real PlantVillage color dataset has an immutable, checksum-verified acquisition path.
- Human-reviewed exclusions and immutable, leaf-grouped train/validation/test manifests are
  committed and independently validated.
- The data loader consumes those checksum-verified manifests rather than discovering
  classes or creating a random split at runtime.

## Training-interface status

Implemented and covered by automated tests:

1. Declared model, optimizer, scheduler, seed, and preprocessing configuration drives the run.
2. Fine-tuning restores the best phase-one checkpoint and creates a new optimizer and callbacks.
3. Frozen-backbone BatchNorm running statistics stay fixed during feature extraction.
4. Each epoch writes a complete resume checkpoint and macro-F1-selected inference checkpoint.
5. Resume restores model, optimizer, callbacks, RNG, data-shuffle generator, history, and phase.
6. Standalone evaluation validates the class map and manifest hash, records ROC-AUC, latency,
   throughput, peak GPU memory, hardware, raw predictions, and confusion matrices.
7. Locked-test evaluation is separate from training and requires explicit confirmation.

Still required before creating `training-v1.0`:

1. Run a GPU pilot from a clean commit and verify interruption/resume in a fresh runtime.
2. Have another member review the checkpoint schema, result schema, and test-set safeguard.
3. Record the accepted commit and create the annotated tag.

The Streamlit app still needs safe release-checkpoint loading, input/OOD handling, and
calibrated confidence before the final demonstration. That work does not block training.

## Dataset controls

- Record source URL, version/commit, license, download date, archive checksum, sample count, class counts, and exclusions.
- Detect unreadable files, exact duplicates, perceptual near-duplicates, and group leakage.
- Fit data-dependent processing only on training data.
- Apply augmentation only to training samples.
- Do not inspect test results during hyperparameter selection.

## Fair model comparison

Keep the dataset split, input resolution, normalization, augmentation policy, epoch budget, early-stopping rule, model-selection metric, test code, and performance benchmark method fixed. Architecture-specific learning rates are acceptable if their search space and selection rule are declared.

Recommended baseline settings are 224×224 RGB, batch size 32, AdamW, weight decay `1e-4`, head learning rate `1e-3`, fine-tuning learning rate `1e-4`, early-stopping patience 5, and validation macro F1 for model selection. Treat these as a starting point, not hidden facts.

## Evaluation package

For each model and seed, preserve:

- accuracy, top-3 accuracy, macro/weighted precision, recall, and F1;
- multiclass one-vs-rest ROC-AUC and per-class metrics;
- raw and normalized confusion matrices;
- learning curves and best-epoch selection;
- parameter count, checkpoint size, latency, throughput, peak memory, and FLOPs/MACs methodology;
- test predictions and confidence values for repeatable error analysis;
- representative correct, incorrect, low-confidence, and domain-shift examples.

Run seeds `42`, `123`, and `2026` if feasible and report mean ± standard deviation. Perform model selection using validation data, then evaluate each frozen selected model on the unseen test set once.
