# Google Colab workflow

## 1. Start from a versioned commit

In Colab, select a GPU runtime, then verify it:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
!nvidia-smi
```

Clone the repository and record the exact commit:

```python
!git clone https://github.com/OWNER/REPOSITORY.git
%cd REPOSITORY
!git rev-parse HEAD
!pip install -r requirements.txt
```

Do not edit the only copy of important code inside Colab. Make code/config changes through normal branches and pull requests, then pull the reviewed commit into Colab.

## 2. Mount Drive for durable outputs

```python
from google.colab import drive
drive.mount('/content/drive')
```

Use a folder such as:

```text
/content/drive/MyDrive/SE4050_PlantDisease/
├── datasets/
├── checkpoints/
└── experiment_backups/
```

Colab's `/content` storage disappears when the runtime ends. Save `last_resume.pt`, `best_inference.pt`, the configuration, and training history to Drive at every epoch.

## 3. Stage the dataset locally

Keep the archived dataset in Drive, but copy and extract it to `/content/data` at the beginning of a session. Reading thousands of small files directly from mounted Drive is slow. Never commit the archive or extracted images to GitHub.

Verify the dataset version/checksum and the final split-manifest checksum before training.

## 4. Train and resume

The target interface for training should be a script backed by `src/`, for example:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --model efficientnet_b0 \
  --seed 42 \
  --data-root /content/data/plant_village \
  --output-dir /content/drive/MyDrive/SE4050_PlantDisease/checkpoints
```

`scripts/train.py` is a target interface and still needs to be implemented; the current executable is `python run_pipeline.py`, which trains all models. Do not duplicate the training implementation in four notebooks.

A resumable checkpoint should contain model, optimizer, scheduler/scaler states, epoch, best metric, class map, preprocessing metadata, config, seed, Git SHA, and library versions. The smaller inference checkpoint needs model weights, class map, architecture, preprocessing, and validation score.

## 5. Compare fairly

Colab GPU types can change between sessions. Training duration may be reported with its hardware context, but final latency, throughput, and memory comparisons must be measured for all four models in the same session, on the same device, with warm-up and repeated runs.

## 6. Publish final models

Keep intermediate and resumable checkpoints in Drive. For the four final inference checkpoints:

1. Compute SHA-256 hashes.
2. Fill in `models/model_manifest.example.json` and rename the release copy to `model_manifest.json`.
3. Create a GitHub Release tagged `models-v1.0.0`.
4. Attach the four checkpoints, `model_manifest.json`, and `checksums.sha256`.
5. Link the release and manifest from the root README.

Ordinary GitHub repositories reject files larger than 100 MiB, and binary checkpoints make history expensive. GitHub Releases are preferred here; Git LFS is an acceptable alternative only if the team understands its storage and bandwidth limits.

