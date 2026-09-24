# Dataset documentation

## Selected dataset

The project uses the maintained PlantVillage color distribution from
`mohanty/PlantVillage` on Hugging Face. The exact repository commit and upstream file
hashes are stored in `data/plantvillage_source.lock.json`; local acquisition evidence is
written to the ignored file `data/processed/plantvillage/provenance.json`.

| Field | Value |
|---|---|
| Dataset name | PlantVillage, color configuration |
| Source URL | https://huggingface.co/datasets/mohanty/PlantVillage |
| Exact version/commit | See `data/plantvillage_source.lock.json` |
| License | Source dataset card declares CC BY-SA 3.0; preserve attribution and verify redistribution requirements before release |
| Download date | Recorded automatically in local provenance |
| Archive SHA-256 | Locked upstream hash and locally verified digest |
| Expected images | 54,305 |
| Expected classes | 38 crop–disease pairs |
| Exclusions/repairs | TBD |
| Source split | Leaf-grouped 43,596 train / 10,709 locked test |
| Validation split | To be created only from the source training partition, grouped by `leaf_id` |
| Split manifest SHA-256 | TBD |

The original paper reports 54,306 images, while this maintained color distribution contains
54,305. Results and documentation in this repository must use the observed count from the
locked distribution.

## Acquisition

From the repository root:

```powershell
python scripts/prepare_data.py download
```

The downloader keeps the official test partition locked, copies the upstream leaf-grouping
metadata, and refuses a source archive whose hash or expected image/class counts do not
match. Do not use the test partition for model selection or hyperparameter tuning.

The `lock` subcommand deliberately refreshes the reviewed source revision and must not be
run as part of ordinary setup. If the team decides to update the source, run it explicitly,
review every lock-file change, and repeat data validation before merging.

After acquisition, the local layout is:

```text
data/raw/plantvillage/
├── color/       # ignored original RGB class folders
└── metadata/    # ignored upstream split and leaf-grouping files
```

Run `python scripts/prepare_data.py validate` after acquisition. The audit checks image
readability, RGB mode, expected class folders, exact and perceptual duplicates, upstream
leaf-map coverage, and leakage across the official train/test boundary. Its detailed JSON
report is reproducible local output under `data/processed/plantvillage/` and is ignored by
Git.

The upstream leaf map does not unambiguously cover every image and contains historical
class-name inconsistencies. The validator therefore records coverage explicitly and gives
unresolved images unique fallback groups; it never invents shared leaf IDs. Perceptual-hash
matches are treated as review candidates, not definitive duplicates.

The reviewed findings for the locked source are documented in
`docs/DATA_VALIDATION.md`. In particular, five byte-identical groups cross the official
train/test boundary, so the official lists must not be used unchanged for final training.

Create the final grouped training/validation manifests from the official training
partition only after this validation report has been reviewed. That work belongs in the
separate frozen-manifest pull request.

## Current local development data

`data/raw/` currently contains 300 programmatically generated images: 20 examples in each of 15 classes. They exist only to exercise the pipeline quickly. They are not downloaded PlantVillage photographs and must never be described as such in the report.

The generated `data/processed/split_indices.json` currently represents a 210/45/45 split of that synthetic data. Delete/regenerate the processed cache after installing the final dataset.

## Directory policy

- `raw/`: local public dataset images; ignored.
- `processed/`: rebuildable caches/indexes; ignored.
- `splits/`: reviewed final split manifests; tracked.
- `external/`: optional domain-shift datasets; ignored.

Never redistribute data unless the exact dataset license permits it. Cite the dataset and original publication in the report.
