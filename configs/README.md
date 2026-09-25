# Experiment configuration

Configuration files are version-controlled experiment inputs. A final run must record the exact config path and Git commit SHA.

`config.yaml` is the current baseline, but parts of the existing pipeline still hard-code values instead of honoring it. Resolve every item in `docs/EXPERIMENT_PROTOCOL.md` before relying on the configuration for final runs.

The dataset paths point to the acquired PlantVillage color folders and the committed frozen
manifests:

```yaml
dataset:
  raw_dir: "data/raw/plantvillage/color"
  manifest_dir: "data/splits"
```

Changing a random seed must not regenerate the data partitions. The split seed and 15%
validation decision are frozen in `data/splits/split_metadata.json`; the training seed only
controls runtime stochastic behavior.
