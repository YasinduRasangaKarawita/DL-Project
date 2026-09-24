# Scripts

This directory is reserved for thin command-line entry points that call reusable code from `src/`.

Implemented interface:

- `prepare_data.py`: acquire the pinned PlantVillage source dataset;

Planned interfaces:

- `train.py`: train or resume one selected model;
- `evaluate.py`: evaluate a frozen checkpoint without training;
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
