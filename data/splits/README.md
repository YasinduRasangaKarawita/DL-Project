# Dataset split manifests

Store the reviewed final split manifests here. Use stable sample IDs or paths relative to the dataset root, include the random seed and split method, and record a checksum.

For PlantVillage, preserve leaf groups so related images of the same physical leaf cannot appear in different partitions. Once frozen, do not regenerate the test partition during experimentation.

Before generating manifests, export every cross-split perceptual-hash candidate from the
local validation report:

```powershell
python scripts/prepare_data.py review-candidates
```

This creates `perceptual_duplicate_review.csv` with a `pending` decision for each candidate.
Review every pair before using the ledger as split-generation input. The command refuses to
overwrite an existing ledger unless `--overwrite` is supplied, because overwriting could
erase human review decisions.

Record exactly one of these decisions for every pair:

- `exclude_train_related`: visual evidence indicates the same physical leaf or image;
  preserve the locked test image and exclude the related training image.
- `keep_both_false_positive`: the images are visibly unrelated and the perceptual-hash
  match is only a screening false positive.

Every decision requires a short review note. Validate the completed ledger against the
local source report before generating any split manifest:

```powershell
python scripts/prepare_data.py validate-review
```

After the review ledger is complete, consolidate its approved training removals with the
byte-identical files that cross the official train/test boundary:

```powershell
python scripts/prepare_data.py build-exclusions
python scripts/prepare_data.py validate-exclusions
```

The resulting `training_exclusions.csv` contains evidence records rather than deleting or
moving source images. A training image can appear more than once when multiple test images
support the same exclusion; split generation must treat `train_path` as a set.
