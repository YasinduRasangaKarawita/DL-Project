# Member 1 verified design facts

This table is an audit aid for `reports/member1_report_draft.md`. It distinguishes source-level counts from the final manifest counts and identifies the repository evidence for every numerical claim. The evidence baseline is `main` commit `d94b51e`, which includes the merged Custom CNN implementation.

| Claim | Verified value | Primary repository evidence |
|---|---|---|
| Dataset source | `mohanty/PlantVillage`, color configuration | `data/README.md`; `data/plantvillage_source.lock.json` |
| Locked revision | `9e97599868962bd0079b8db4b7f1efa9185fa1e7` | `data/plantvillage_source.lock.json` |
| Declared source license | CC BY-SA 3.0 | `data/plantvillage_source.lock.json`; source dataset card |
| Source images/classes | 54,305 / 38 | `data/plantvillage_source.lock.json`; `docs/DATA_VALIDATION.md` |
| Source readability | 54,305 readable; 0 empty; 0 corrupted | `docs/DATA_VALIDATION.md`; regenerated validation report |
| Source modes | 54,304 RGB; 1 RGBA | `docs/DATA_VALIDATION.md`; regenerated validation report |
| Source dimensions | all 54,305 are 256 × 256 | regenerated validation report |
| Exact duplicates | 21 groups / 42 images | `docs/DATA_VALIDATION.md` |
| Exact groups crossing official train/test | 5 | `docs/DATA_VALIDATION.md` |
| Perceptual candidates | 249 total; 41 crossing official train/test | `docs/DATA_VALIDATION.md` |
| Human review | 15 related; 26 false positives | `data/splits/perceptual_duplicate_review.csv`; `docs/DATA_VALIDATION.md` |
| Unique training images excluded | 19 | `data/splits/training_exclusions.csv`; `data/splits/split_metadata.json` |
| Unambiguous leaf mapping | 41,111 images (75.70%) | `docs/DATA_VALIDATION.md` |
| Frozen splits | 37,037 train; 6,540 validation; 10,709 test | `data/splits/split_metadata.json` |
| Frozen images/classes | 54,286 / 38 | `src/evaluation/dataset_analysis.py`; frozen manifests |
| Manifest bundle SHA-256 | `19ca82ed9a1832dbeca228bbcb35274cd3362758e918818e64587f4d0bd1306c` | `data/splits/checksums.sha256` validation |
| Largest training class | Orange HLB, 3,851 (10.40%) | `data/splits/split_metadata.json`; `analyze_class_distribution` |
| Smallest training class | Potato healthy, 104 (0.28%) | `data/splits/split_metadata.json`; `analyze_class_distribution` |
| Training imbalance | 37.03:1; mean 974.7; median 756.5 | `analyze_class_distribution` |
| Model input | RGB, 3 × 224 × 224 | `configs/config.yaml`; `src/data/preprocessing.py` |
| Normalization | mean `[0.485, 0.456, 0.406]`; std `[0.229, 0.224, 0.225]` | `configs/config.yaml` |
| Custom CNN widths/head | `[32, 64, 128]`; 256 dense units; dropout 0.5 | `configs/config.yaml`; `src/models/custom_cnn.py` |
| Custom CNN outputs | 38 logits in frozen class order | `data/splits/class_mapping.json`; `notebooks/03_custom_cnn.ipynb` |
| Custom CNN parameters | 136,262 total/trainable | `tests/test_custom_cnn.py`; `notebooks/03_custom_cnn.ipynb` |

## Reproduction commands

Run these commands from the repository root after acquiring the locked source:

```powershell
python scripts/prepare_data.py validate
python scripts/prepare_data.py validate-review
python scripts/prepare_data.py validate-exclusions
python scripts/prepare_data.py validate-manifests
python -m pytest tests/test_dataset_analysis.py tests/test_preprocessing.py tests/test_custom_cnn.py
```

The first `validate` command is expected to return a non-zero status for this locked source because it detects the five exact cross-boundary duplicate groups. The subsequent reviewed-exclusion and manifest checks must pass; together they verify the documented remediation.

## Claims deliberately unavailable

No accepted run ID, official checkpoint, model checksum, training history, predictive metric, timing measurement, or test prediction exists yet. Do not add values for these fields until the shared protocol is frozen and the corresponding artifacts pass review.
