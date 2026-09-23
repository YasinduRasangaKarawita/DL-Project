# Dataset documentation

## Final dataset record

Complete this table before final training.

| Field | Value |
|---|---|
| Dataset name | TBD |
| Source URL | TBD |
| Exact version/commit | TBD |
| License | TBD—verify for the exact distribution used |
| Download date | TBD |
| Archive SHA-256 | TBD |
| Total accepted images | TBD |
| Number of classes | TBD |
| Exclusions/repairs | TBD |
| Split method | Leaf-grouped train/validation/test recommended |
| Split manifest SHA-256 | TBD |

## Current local development data

`data/raw/` currently contains 300 programmatically generated images: 20 examples in each of 15 classes. They exist only to exercise the pipeline quickly. They are not downloaded PlantVillage photographs and must never be described as such in the report.

The generated `data/processed/split_indices.json` currently represents a 210/45/45 split of that synthetic data. Delete/regenerate the processed cache after installing the final dataset.

## Directory policy

- `raw/`: local public dataset images; ignored.
- `processed/`: rebuildable caches/indexes; ignored.
- `splits/`: reviewed final split manifests; tracked.
- `external/`: optional domain-shift datasets; ignored.

Never redistribute data unless the exact dataset license permits it. Cite the dataset and original publication in the report.

