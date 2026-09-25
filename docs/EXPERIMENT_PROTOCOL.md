# Experiment protocol

This document is the pre-training checklist. The existing pipeline is suitable for a smoke test, not yet for final assignment results.

## Completed data foundations

- The real PlantVillage color dataset has an immutable, checksum-verified acquisition path.
- Human-reviewed exclusions and immutable, leaf-grouped train/validation/test manifests are
  committed and independently validated.
- The data loader consumes those checksum-verified manifests rather than discovering
  classes or creating a random split at runtime.

## Required fixes before final runs

1. Make every declared configuration field authoritative; remove hard-coded pretrained, freeze, unfreeze, optimizer, and scheduler values.
2. Re-create callbacks after replacing the optimizer during fine-tuning so the scheduler controls the new optimizer.
3. Prevent frozen backbone BatchNorm statistics from changing during feature extraction.
4. Restore the best phase-one checkpoint before fine-tuning rather than continuing from the final phase-one epoch.
5. Save complete resumable checkpoints and self-describing inference checkpoints.
6. Add warm-up, synchronization, repeated batches, and hardware metadata to performance measurement.
7. Make the app fail safely when weights or metadata are missing; add input/OOD handling and calibrated confidence.

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
