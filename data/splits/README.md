# Dataset split manifests

This directory contains the reviewed, frozen PlantVillage split manifests. They use paths
relative to the dataset root, preserve grouping evidence, and record the source revision,
split method, random seed, class mapping, and checksums.

The manifests preserve leaf groups so related images of the same physical leaf cannot
appear in different partitions. Do not regenerate the test partition during experimentation.

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

The grouped manifest generator requires an explicit validation fraction because the value
means a fraction of the cleaned official training partition—not a fraction of the complete
dataset. The reviewed bundle uses 15% and seed 42:

```powershell
python scripts/prepare_data.py build-manifests --validation-fraction 0.15 --seed 42
```

The generator stratifies by class approximately while keeping every resolved physical-leaf
group wholly in either training or validation. The official test partition is preserved.

After generation, independently validate the committed bundle without regenerating it:

```powershell
python scripts/prepare_data.py validate-manifests
```

This verifies file checksums, source revision and counts, exclusion coverage, class mapping,
manifest metadata, exact official-test preservation, and path/group isolation across all
three partitions.

## Frozen bundle

| Partition | Images |
|---|---:|
| Train | 37,037 |
| Validation | 6,540 |
| Test | 10,709 |

The cleaned official training source contains 43,577 images after excluding 19 unique
training images supported by the review and exact-duplicate evidence. All 10,709 official
test images are preserved. The bundle contains 38 classes and its SHA-256 checksum (the
digest of `checksums.sha256`) is
`19ca82ed9a1832dbeca228bbcb35274cd3362758e918818e64587f4d0bd1306c`.

Do not regenerate these files during ordinary experiments. Use `--overwrite` only when the
team has deliberately approved a new source revision or split protocol, then review every
manifest and checksum change before merging.
