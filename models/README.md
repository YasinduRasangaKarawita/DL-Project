# Model artifacts

Checkpoint binaries are ignored by Git. The local files currently present were trained on the synthetic smoke-test dataset and are not final project models.

Recommended local layout:

```text
models/<model_name>/<run_id>/
├── best_inference.pt
└── last_resume.pt
```

Keep intermediate/resume files in Google Drive during Colab training. Publish only the selected inference checkpoints through a versioned GitHub Release. Attach:

- `custom_cnn_best.pt`
- `resnet50_best.pt`
- `efficientnet_b0_best.pt`
- `mobilenet_v3_best.pt`
- `model_manifest.json`
- `checksums.sha256`

The app and README should point to that release. Consumers must verify SHA-256 before loading a checkpoint.

