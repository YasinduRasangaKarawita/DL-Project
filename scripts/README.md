# Scripts

This directory is reserved for thin command-line entry points that call reusable code from `src/`.

Implemented interface:

- `prepare_data.py`: lock, acquire, audit, review, and validate the pinned PlantVillage
  source and frozen split manifests;
- root `run_pipeline.py train`: train or resume one selected model and seed;
- root `run_pipeline.py evaluate`: evaluate one selected inference checkpoint without
  training, with an explicit locked-test safeguard.

Planned interfaces:

- `benchmark.py`: compare final models on one device;
- `download_models.py`: download release assets and verify checksums.

The current executable remains `run_pipeline.py` at the repository root. Do not create wrapper scripts that duplicate its internal implementation.

## PlantVillage acquisition

Run the following commands from the repository root after installing the requirements:

```powershell
python scripts/prepare_data.py download
```

The command reads the committed `data/plantvillage_source.lock.json`, downloads the locked
source files, verifies their SHA-256 hashes, and extracts only the original color images to
`data/raw/plantvillage/color/`.

Only the Data Lead should intentionally refresh the source version. To resolve the current
upstream state to an immutable revision, review the diff, and commit a new lock:

```powershell
python scripts/prepare_data.py lock
```

Downloaded images, the Hugging Face cache, copied grouping metadata, and the local
provenance report are ignored by Git. Do not commit or manually move them into a tracked
directory.

## PlantVillage validation

After acquisition, run the reproducible data-quality and leakage audit:

```powershell
python scripts/prepare_data.py validate
```

The command fully decodes every image, checks RGB mode and class folders, detects exact
SHA-256 duplicates, screens perceptual near-duplicates with a 64-bit difference hash, and
checks leaf-group and exact-duplicate leakage across the locked official train/test split.
It writes the machine-readable report to the ignored local path
`data/processed/plantvillage/validation_report.json`.

Perceptual matches are review candidates rather than proof of duplication. The command
fails for objective integrity or leakage errors and reports incomplete or historically
inconsistent upstream leaf metadata as warnings.

## Frozen split verification

The reviewed ledgers and manifests are already committed under `data/splits/`. Verify them
without modifying or regenerating any file:

```powershell
python scripts/prepare_data.py validate-review
python scripts/prepare_data.py validate-exclusions
python scripts/prepare_data.py validate-manifests
```

For the approved 15% grouped validation split, the frozen counts are 37,037 train, 6,540
validation, and 10,709 test across 38 classes. See `data/splits/README.md` for the generation
command, review-decision meanings, and bundle checksum.
