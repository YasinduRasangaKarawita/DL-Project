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
| Validation split | 6,540 images (15% target from cleaned official training, seed 42, grouped by physical leaf) |
| Final train/test counts | 37,037 train / 10,709 locked test |
| Excluded training images | 19 leakage-risk images with evidence in `data/splits/training_exclusions.csv` |
| Manifest bundle SHA-256 | `19ca82ed9a1832dbeca228bbcb35274cd3362758e918818e64587f4d0bd1306c` (`checksums.sha256`) |

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

The final grouped manifests and class mapping are tracked under `data/splits/`. Validate
their provenance, coverage, grouping, and checksums with
`python scripts/prepare_data.py validate-manifests` before training.

## Current local development data

`data/raw/` currently contains 300 programmatically generated images: 20 examples in each of 15 classes. They exist only to exercise the pipeline quickly. They are not downloaded PlantVillage photographs and must never be described as such in the report.

The training loader does not discover classes or create random splits. It reads the
checksum-verified manifests and class mapping tracked under `data/splits/`.

## Directory policy

- `raw/`: local public dataset images; ignored.
- `processed/`: rebuildable caches/indexes; ignored.
- `splits/`: reviewed final split manifests; tracked.
- `external/`: optional domain-shift datasets; ignored.

Never redistribute data unless the exact dataset license permits it. Cite the dataset and original publication in the report.
