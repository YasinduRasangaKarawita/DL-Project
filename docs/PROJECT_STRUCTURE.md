# Project structure and file ownership

## Source-controlled inputs

- `configs/`: experiment settings. A result is not reproducible without its exact config.
- `data/splits/`: final train/validation/test manifests and their hashes. These contain relative paths or stable sample IDs, not images.
- `src/`: reusable implementation. Training logic belongs here, not in notebooks.
- `tests/`: fast checks for data invariants, tensor shapes, model outputs, and metrics.
- `notebooks/`: thin narrative interfaces for EDA and experiments.
- `app/`: inference demonstration. It consumes published model metadata and weights.
- `reports/`: report source plus deliberately selected tables/figures used in the submission.

## Local or generated content

- `data/raw/`: downloaded source images. Never commit.
- `data/processed/`: generated caches and temporary indexes. Rebuild from source data.
- `models/`: downloaded or locally trained weights. Publish final files through GitHub Releases.
- `results/`: histories, predictions, raw metrics, profiler output, and experiment logs.
- `figures/`: generated plots before report curation.
- `tmp/`: disposable local scratch space.

These directories are ignored because they are large, machine-specific, reproducible, or currently contain smoke-test artifacts. Copy only verified final tables and figures into `reports/`.

## Naming conventions

Use lowercase snake case for files and model identifiers:

```text
results/<run_id>/<model_name>/metrics.json
results/<run_id>/<model_name>/history.csv
models/<model_name>/<run_id>/best_inference.pt
models/<model_name>/<run_id>/last_resume.pt
```

A recommended run ID is:

```text
YYYYMMDD-HHMM_<model>_seed<seed>_<short-git-sha>
```

## Boundary between code, notebooks, and results

`src/` defines behavior, `configs/` defines an experiment, notebooks explain or invoke it, `results/` records raw outputs, and `reports/` contains the reviewed evidence presented to markers. Keeping those roles separate prevents hidden notebook state and accidental result cherry-picking.

