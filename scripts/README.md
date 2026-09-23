# Scripts

This directory is reserved for thin command-line entry points that call reusable code from `src/`.

Planned interfaces:

- `prepare_data.py`: verify/acquire data and freeze manifests;
- `train.py`: train or resume one selected model;
- `evaluate.py`: evaluate a frozen checkpoint without training;
- `benchmark.py`: compare final models on one device;
- `download_models.py`: download release assets and verify checksums.

The current executable remains `run_pipeline.py` at the repository root. Do not create wrapper scripts that duplicate its internal implementation.

