"""Standalone evaluation for frozen inference checkpoints."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ..data.dataset_loader import get_dataloaders
from ..models.registry import build_model
from ..training.checkpoints import load_checkpoint, sha256_file
from ..training.experiment import environment_record, utc_timestamp, write_json
from ..utils.helpers import get_device
from ..utils.seed import set_seed
from .confusion_matrix import plot_confusion_matrix
from .error_analysis import run_error_analysis
from .metrics import evaluate_model_metrics


def evaluate_checkpoint(
    *,
    checkpoint_path: str,
    split: str = "validation",
    device_preference: str | None = None,
    output_root: str = ".",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    device = get_device(device_preference or "auto")
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("checkpoint_type") != "inference":
        raise ValueError("Evaluation requires a best_inference checkpoint")

    metadata = checkpoint["run_metadata"]
    config = deepcopy(metadata["config"])
    model_name = str(metadata["model_name"])
    seed = int(metadata["seed"])
    set_seed(seed)
    manifest_dir = config["dataset"].get("manifest_dir", "data/splits")
    current_manifest_checksum = sha256_file(Path(manifest_dir) / "checksums.sha256")
    if current_manifest_checksum != metadata["manifest_bundle_sha256"]:
        raise ValueError("Current manifest bundle does not match the checkpoint")

    train_loader, validation_loader, test_loader, class_names, class_to_index = get_dataloaders(
        raw_dir=config["dataset"]["raw_dir"],
        manifest_dir=manifest_dir,
        batch_size=int(config["training"]["batch_size"]),
        image_size=tuple(config["dataset"].get("image_size", [224, 224])),
        random_seed=seed,
        num_workers=int(config["dataset"].get("num_workers", 0)),
        augmentation_config=config.get("augmentation", {}),
    )
    del train_loader
    if checkpoint["class_names"] != class_names or checkpoint["class_to_index"] != class_to_index:
        raise ValueError("Checkpoint class mapping does not match the frozen manifests")

    # Evaluation never downloads pretrained weights; the complete learned state follows.
    if model_name != "custom_cnn":
        config["models"][model_name]["pretrained"] = False
    model = build_model(
        model_name,
        num_classes=len(class_names),
        models_config=config["models"],
    ).model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = validation_loader if split == "validation" else test_loader
    metrics, targets, predictions, probabilities, _ = evaluate_model_metrics(
        model,
        loader,
        class_names,
        device,
    )
    run_id = str(metadata["run_id"])
    output_dir = Path(output_root) / "results" / run_id / model_name / split
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.npz"
    image_paths = np.asarray(loader.dataset.image_paths, dtype=str)
    np.savez_compressed(
        prediction_path,
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        image_paths=image_paths,
        class_names=np.asarray(class_names, dtype=str),
    )
    confusion_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        targets,
        predictions,
        class_names,
        model_name=model_name,
        save_path=str(confusion_path),
    )
    confusion_counts_path = output_dir / "confusion_matrix_counts.png"
    plot_confusion_matrix(
        targets,
        predictions,
        class_names,
        model_name=model_name,
        save_path=str(confusion_counts_path),
        normalize=False,
    )
    run_error_analysis(
        targets,
        predictions,
        probabilities,
        list(loader.dataset.image_paths),
        class_names,
        model_name=model_name,
        output_dir=str(output_dir),
    )
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "model_name": model_name,
        "seed": seed,
        "split": split,
        "evaluated_at_utc": utc_timestamp(),
        "git_commit": metadata["git_commit"],
        "manifest_bundle_sha256": current_manifest_checksum,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_mb": round(Path(checkpoint_path).stat().st_size / (1024 * 1024), 2),
            "selected_epoch": checkpoint["epoch"],
            "selection_metric": checkpoint["monitor"],
        },
        "environment": environment_record(device),
        "metrics": metrics,
        "artifacts": {
            "predictions": str(prediction_path),
            "confusion_matrix": str(confusion_path),
            "confusion_matrix_counts": str(confusion_counts_path),
        },
    }
    result_path = output_dir / "metrics.json"
    write_json(result_path, result)
    result["result_path"] = str(result_path)
    return result
