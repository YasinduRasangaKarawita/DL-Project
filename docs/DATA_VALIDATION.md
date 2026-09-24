# PlantVillage data validation

This audit applies to the locked `mohanty/PlantVillage` color dataset at revision
`9e97599868962bd0079b8db4b7f1efa9185fa1e7`. Run it after acquisition with:

```powershell
python scripts/prepare_data.py validate
```

The detailed machine-readable report is regenerated locally at
`data/processed/plantvillage/validation_report.json`. It is ignored because it is derived
from the committed source lock, code, and upstream metadata.

## Checks performed

- Fully decode every image and record its dimensions and color mode.
- Compare local image and class counts with the committed source lock.
- Compare class folders and relative paths with the official split lists.
- Group byte-identical files by SHA-256.
- Screen visually similar, non-identical files using a 64-bit difference hash with a
  maximum Hamming distance of four.
- Resolve upstream physical-leaf groups using the source repository's filename
  normalization and class disambiguation rules.
- Test official train/test partitions for path, exact-duplicate, and resolved-leaf-group
  leakage.

Perceptual-hash results are candidates for human review. They can contain visually similar
but unrelated images and must not be treated as proven duplicates without inspection.

## Findings on the locked source

| Check | Result |
|---|---:|
| Images scanned/readable | 54,305 / 54,305 |
| Class folders | 38 expected, 38 observed |
| Empty or unreadable images | 0 |
| RGB images | 54,304 |
| Readable RGBA images | 1 |
| Official train/test paths | 43,596 / 10,709 |
| Path overlap or coverage gaps | 0 |
| Images with an unambiguous upstream leaf group | 41,111 (75.70%) |
| Images without an unambiguous upstream leaf group | 13,194 |
| Resolved leaf groups crossing train/test | 0 |
| Exact duplicate groups | 21 groups / 42 images |
| Exact duplicate groups crossing train/test | 5 |
| Difference-hash candidate pairs | 249 |
| Difference-hash candidates crossing train/test | 41 |

The one RGBA PNG is readable and the dataset loader already converts all inputs to RGB, so
it is retained and reported as a warning. The 621 metadata class-name differences are the
upstream historical `Apple_Frogeye Spot` label attached to files now stored under
`Apple___Black_rot`; they are recorded rather than silently rewritten.

The five byte-identical train/test duplicate groups are:

| Class | Source identifier | Test/train copies |
|---|---|---|
| `Tomato___healthy` | `GH_HL Leaf 466.1` | 1 test, 1 train |
| `Tomato___healthy` | `GH_HL Leaf 220` | 1 test, 1 train |
| `Tomato___Late_blight` | `GHLB_PS Leaf 24 Day 16` | 1 test, 1 train |
| `Tomato___Late_blight` | `GHLB_PS Leaf 23.7 Day 13` | 1 test, 1 train |
| `Tomato___Late_blight` | `GHLB2 Leaf 8999` | 1 test, 1 train |

## Decision for the next data pull request

The official split files are useful source partitions, but they are not ready to use
unchanged for final experiments because exact image content crosses the boundary. Do not
delete or edit the downloaded source files.

When freezing project manifests, keep the official test copy fixed and exclude each
byte-identical training copy. Review the 41 cross-split perceptual candidates, merge
confirmed related images into the same grouping constraint where possible, and record any
additional exclusions with reasons. Create validation only from the cleaned official
training partition and keep every resolved leaf group intact.
