"""Model-agnostic training orchestration with safe resume semantics."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ..data.dataset_loader import get_dataloaders
from ..models.registry import build_model
from ..utils.helpers import count_parameters, get_device, load_yaml_config
from ..utils.logger import setup_logger
from ..utils.seed import set_seed
from .callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from .checkpoints import (
    load_checkpoint,
    restore_rng_state,
    save_inference_checkpoint,
    save_resume_checkpoint,
    sha256_file,
)
from .experiment import (
    build_run_metadata,
    current_git_commit,
    generate_run_id,
    git_worktree_is_dirty,
    utc_timestamp,
    write_json,
)
from .trainer import ModelTrainer

logger = setup_logger("training_runner")


def build_optimizer(
    model: nn.Module,
    training_config: Mapping[str, Any],
    learning_rate: float,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The selected model has no trainable parameters")
    optimizer_name = str(training_config.get("optimizer", "Adam"))
    arguments = {
        "lr": float(learning_rate),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
    }
    if optimizer_name == "Adam":
        return torch.optim.Adam(parameters, **arguments)
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(parameters, **arguments)
    raise ValueError("training.optimizer must be 'Adam' or 'AdamW'")


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    model_name: str,
    seed: int,
    config: Mapping[str, Any],
    manifest_checksum: str,
    initial_epochs: int,
    fine_tune_epochs: int,
) -> None:
    if checkpoint.get("checkpoint_type") != "resume":
        raise ValueError("--resume requires a resumable checkpoint")
    metadata = checkpoint["run_metadata"]
    expected = {
        "model_name": model_name,
        "seed": seed,
        "manifest_bundle_sha256": manifest_checksum,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Resume checkpoint {key} does not match the requested run")
    if metadata.get("config") != dict(config):
        raise ValueError("Resume checkpoint configuration does not match the current configuration")
    if not metadata.get("git_worktree_dirty") and metadata.get("git_commit") != current_git_commit():
        raise ValueError("Resume checkpoint Git commit does not match the current checkout")
    expected_schedule = {
        "initial_epochs": initial_epochs,
        "fine_tune_epochs": fine_tune_epochs,
    }
    if metadata.get("schedule") != expected_schedule:
        raise ValueError("Resume checkpoint epoch schedule does not match the requested run")


def train_experiment(
    *,
    model_name: str,
    seed: int,
    config_path: str = "configs/config.yaml",
    resume_path: str | None = None,
    run_id: str | None = None,
    device_preference: str | None = None,
    initial_epochs_override: int | None = None,
    allow_dirty: bool = False,
    artifact_root: str = ".",
) -> dict[str, Any]:
    """Train one model and never evaluate the locked test partition."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if git_worktree_is_dirty() and not allow_dirty:
        raise RuntimeError(
            "Refusing a non-reproducible run from a dirty worktree; commit the reviewed code "
            "or use allow_dirty only for a declared pilot"
        )
    config = load_yaml_config(config_path)
    training_config = config["training"]
    configured_seeds = training_config.get("seeds")
    if not isinstance(configured_seeds, list) or seed not in configured_seeds:
        raise ValueError(f"seed must be one of the configured seeds: {configured_seeds}")
    if training_config.get("loss_function") != "CrossEntropyLoss":
        raise ValueError("Only the configured CrossEntropyLoss contract is currently supported")
    if training_config.get("model_selection_metric") != "validation_macro_f1":
        raise ValueError("model_selection_metric must be validation_macro_f1")
    initial_epochs = (
        int(initial_epochs_override)
        if initial_epochs_override is not None
        else int(training_config["initial_epochs"])
    )
    if initial_epochs < 1:
        raise ValueError("initial epochs must be positive")
    fine_tune_epochs = 0 if model_name == "custom_cnn" else int(
        training_config.get("fine_tune_epochs", 0)
    )

    set_seed(seed)
    device = get_device(device_preference or config["project"].get("device", "auto"))
    raw_dir = config["dataset"]["raw_dir"]
    manifest_dir = config["dataset"].get("manifest_dir", "data/splits")
    image_size = tuple(config["dataset"].get("image_size", [224, 224]))
    train_loader, validation_loader, _, class_names, class_to_index = get_dataloaders(
        raw_dir=raw_dir,
        manifest_dir=manifest_dir,
        batch_size=int(training_config["batch_size"]),
        image_size=image_size,
        random_seed=seed,
        num_workers=int(config["dataset"].get("num_workers", 0)),
        augmentation_config=config.get("augmentation", {}),
    )

    manifest_checksum = sha256_file(Path(manifest_dir) / "checksums.sha256")
    checkpoint = load_checkpoint(resume_path, device) if resume_path else None
    if checkpoint:
        _validate_resume(
            checkpoint,
            model_name=model_name,
            seed=seed,
            config=config,
            manifest_checksum=manifest_checksum,
            initial_epochs=initial_epochs,
            fine_tune_epochs=fine_tune_epochs,
        )
        if checkpoint.get("class_names") != class_names or checkpoint.get(
            "class_to_index"
        ) != class_to_index:
            raise ValueError("Resume checkpoint class mapping does not match the frozen manifests")
        metadata = checkpoint["run_metadata"]
        resolved_run_id = str(metadata["run_id"])
        if run_id and run_id != resolved_run_id:
            raise ValueError("--run-id does not match the resume checkpoint")
    else:
        resolved_run_id = run_id or generate_run_id(model_name, seed)
        metadata = build_run_metadata(
            run_id=resolved_run_id,
            model_name=model_name,
            seed=seed,
            config=config,
            manifest_dir=manifest_dir,
            device=device,
        )
        metadata["schedule"] = {
            "initial_epochs": initial_epochs,
            "fine_tune_epochs": fine_tune_epochs,
        }

    artifact_base = Path(artifact_root)
    model_dir = artifact_base / "models" / model_name / resolved_run_id
    result_dir = artifact_base / "results" / resolved_run_id / model_name
    best_path = model_dir / "best_inference.pt"
    last_path = model_dir / "last_resume.pt"
    history_path = result_dir / "history.csv"
    record_path = result_dir / "run_metadata.json"
    if not checkpoint and any(path.exists() for path in (best_path, last_path, record_path)):
        raise FileExistsError(f"Run ID already exists: {resolved_run_id}")
    logger.info("Run ID: %s", resolved_run_id)
    logger.info("Durable resume checkpoint: %s", last_path)

    construction_models_config = deepcopy(config["models"])
    if checkpoint and model_name != "custom_cnn":
        # A resume checkpoint contains the complete state and must not depend on a download.
        construction_models_config[model_name]["pretrained"] = False
    model_spec = build_model(
        model_name,
        num_classes=len(class_names),
        models_config=construction_models_config,
    )
    model = model_spec.model.to(device)
    total_parameters, initially_trainable = count_parameters(model)
    metadata.setdefault(
        "parameters",
        {
            "total": total_parameters,
            "initially_trainable": initially_trainable,
        },
    )
    resumed_phase = str(checkpoint["phase"]) if checkpoint else "initial"
    if resumed_phase == "fine_tune":
        if model_spec.unfreeze is None:
            raise ValueError("Custom CNN checkpoints cannot use the fine_tune phase")
        model_spec.unfreeze(model)

    phase_lr = (
        float(training_config["fine_tune_learning_rate"])
        if resumed_phase == "fine_tune"
        else float(training_config["learning_rate"])
    )
    optimizer = build_optimizer(model, training_config, phase_lr)
    trainer = ModelTrainer(model, optimizer, nn.CrossEntropyLoss(), device=device)
    checkpoint_tracker = ModelCheckpoint(str(best_path), monitor="val_macro_f1", mode="max")
    early_stopping = EarlyStopping(
        patience=int(training_config["early_stopping_patience"]), mode="max"
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=int(training_config["reduce_lr_patience"]),
        factor=float(training_config["reduce_lr_factor"]),
        min_lr=float(training_config["min_lr"]),
        mode="max",
    )

    completed_epoch = 0
    completed_phase_epoch = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.load_history(checkpoint["history"])
        callback_states = checkpoint["callbacks"]
        checkpoint_tracker.load_state_dict(callback_states["checkpoint"])
        early_stopping.load_state_dict(callback_states["early_stopping"])
        scheduler.load_state_dict(callback_states["scheduler"])
        restore_rng_state(checkpoint["rng_state"])
        if checkpoint.get("dataloader_generator_state") is not None:
            train_loader.generator.set_state(checkpoint["dataloader_generator_state"])
        completed_epoch = int(checkpoint["epoch"])
        completed_phase_epoch = int(checkpoint["phase_epoch"])

    csv_logger = CSVLogger(str(history_path), resume=bool(checkpoint))
    preprocessing = {
        "image_size": list(image_size),
        "normalization": deepcopy(config.get("augmentation", {}).get("normalization", {})),
        "validation_transform": "resize_center_crop_tensor_normalize",
    }
    started_at = time.perf_counter()

    def run_phase(
        phase: str,
        *,
        phase_epochs: int,
        start_global_epoch: int,
        already_completed: int = 0,
    ) -> int:
        nonlocal early_stopping, scheduler
        remaining = phase_epochs - already_completed
        if remaining <= 0 or early_stopping.early_stop:
            return start_global_epoch - 1
        phase_origin = start_global_epoch - already_completed

        def epoch_end(epoch: int, metrics: dict[str, float]) -> None:
            metric = metrics["validation_macro_f1"]
            if metric > checkpoint_tracker.best_val:
                checkpoint_tracker.best_val = metric
                save_inference_checkpoint(
                    best_path,
                    model=model,
                    epoch=epoch,
                    monitor_name="validation_macro_f1",
                    monitor_value=metric,
                    run_metadata=metadata,
                    class_names=class_names,
                    class_to_index=class_to_index,
                    preprocessing=preprocessing,
                )
            save_resume_checkpoint(
                last_path,
                model=model,
                optimizer=trainer.optimizer,
                epoch=epoch,
                phase=phase,
                phase_epoch=epoch - phase_origin + 1,
                history=trainer.history,
                run_metadata=metadata,
                callback_states={
                    "checkpoint": checkpoint_tracker.state_dict(),
                    "early_stopping": early_stopping.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                class_names=class_names,
                class_to_index=class_to_index,
                preprocessing=preprocessing,
                dataloader_generator_state=train_loader.generator.get_state(),
            )
            logger.info("Epoch %d checkpoint saved: %s", epoch, last_path)

        before = len(trainer.history["train_loss"])
        trainer.fit(
            train_loader,
            validation_loader,
            epochs=remaining,
            early_stopping_cb=early_stopping,
            reduce_lr_cb=scheduler,
            csv_logger_cb=csv_logger,
            start_epoch=start_global_epoch,
            phase=phase,
            epoch_end_cb=epoch_end,
        )
        epochs_ran = len(trainer.history["train_loss"]) - before
        return start_global_epoch + epochs_ran - 1

    if resumed_phase == "initial":
        completed_epoch = run_phase(
            "initial",
            phase_epochs=initial_epochs,
            start_global_epoch=completed_epoch + 1,
            already_completed=completed_phase_epoch,
        )
        completed_phase_epoch = 0

        if model_spec.unfreeze is not None and fine_tune_epochs > 0:
            best_checkpoint = load_checkpoint(best_path, device)
            model.load_state_dict(best_checkpoint["model_state_dict"])
            model_spec.unfreeze(model)
            optimizer = build_optimizer(
                model,
                training_config,
                float(training_config["fine_tune_learning_rate"]),
            )
            trainer.optimizer = optimizer
            early_stopping = EarlyStopping(
                patience=int(training_config["early_stopping_patience"]), mode="max"
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=int(training_config["reduce_lr_patience"]),
                factor=float(training_config["reduce_lr_factor"]),
                min_lr=float(training_config["min_lr"]),
                mode="max",
            )
            completed_epoch = run_phase(
                "fine_tune",
                phase_epochs=fine_tune_epochs,
                start_global_epoch=completed_epoch + 1,
            )
    else:
        completed_epoch = run_phase(
            "fine_tune",
            phase_epochs=fine_tune_epochs,
            start_global_epoch=completed_epoch + 1,
            already_completed=completed_phase_epoch,
        )

    if not best_path.exists() or not last_path.exists():
        raise RuntimeError("Training ended without producing the required checkpoints")
    record = {
        **metadata,
        "status": "completed",
        "completed_at_utc": utc_timestamp(),
        "epochs_completed": len(trainer.history["train_loss"]),
        "last_global_epoch": completed_epoch,
        "best_validation_macro_f1": checkpoint_tracker.best_val,
        "parameters": {
            **metadata["parameters"],
            "finally_trainable": count_parameters(model)[1],
        },
        "training_elapsed_seconds_this_session": round(time.perf_counter() - started_at, 2),
        "artifacts": {
            "best_inference": {
                "path": str(best_path),
                "sha256": sha256_file(best_path),
            },
            "last_resume": {"path": str(last_path), "sha256": sha256_file(last_path)},
            "history": {"path": str(history_path), "sha256": sha256_file(history_path)},
        },
    }
    write_json(record_path, record)
    logger.info("Completed run %s; best validation macro F1 %.4f", resolved_run_id, checkpoint_tracker.best_val)
    return record
