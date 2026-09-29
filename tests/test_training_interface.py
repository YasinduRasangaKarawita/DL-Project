import csv
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_pipeline import build_parser
from src.evaluation.metrics import evaluate_model_metrics
from src.training.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.training.checkpoints import (
    load_checkpoint,
    restore_rng_state,
    save_inference_checkpoint,
    save_resume_checkpoint,
)
from src.training.runner import build_optimizer, train_experiment
from src.training.trainer import ModelTrainer
from src.utils.seed import set_seed


def _history():
    return {
        "train_loss": [1.0],
        "train_acc": [0.5],
        "val_loss": [0.9],
        "val_acc": [0.5],
        "val_macro_f1": [0.5],
        "epoch_times": [0.1],
    }


def test_cli_selects_one_model_and_seed():
    args = build_parser().parse_args(
        ["train", "--model", "custom_cnn", "--seed", "42"]
    )
    assert args.command == "train"
    assert args.model == "custom_cnn"
    assert args.seed == 42


def test_optimizer_factory_honors_configuration():
    model = nn.Linear(2, 2)
    optimizer = build_optimizer(
        model,
        {"optimizer": "AdamW", "weight_decay": 0.01},
        learning_rate=0.002,
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)


def test_resume_checkpoint_round_trip_restores_training_state(tmp_path):
    torch.manual_seed(7)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = torch.tensor([[1.0, 2.0]])
    loss = model(inputs).sum()
    loss.backward()
    optimizer.step()

    early = EarlyStopping(patience=2, mode="max")
    scheduler = ReduceLROnPlateau(optimizer, mode="max")
    best = ModelCheckpoint(str(tmp_path / "best.pt"), monitor="val_macro_f1", mode="max")
    early(0.5)
    scheduler.step(0.5)
    best.best_val = 0.5
    generator = torch.Generator().manual_seed(42)
    path = tmp_path / "last_resume.pt"
    save_resume_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        phase="initial",
        phase_epoch=1,
        history=_history(),
        run_metadata={"run_id": "test"},
        callback_states={
            "checkpoint": best.state_dict(),
            "early_stopping": early.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        class_names=["a", "b"],
        class_to_index={"a": 0, "b": 1},
        preprocessing={"image_size": [224, 224]},
        dataloader_generator_state=generator.get_state(),
    )

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    loaded = load_checkpoint(path, "cpu")
    restore_rng_state(loaded["rng_state"])

    assert loaded["checkpoint_type"] == "resume"
    assert loaded["phase_epoch"] == 1
    assert loaded["history"] == _history()
    assert loaded["gradient_scaler_state_dict"] is None
    assert loaded["class_to_index"] == {"a": 0, "b": 1}
    assert random.random() == expected_python
    assert np.random.random() == pytest.approx(expected_numpy)
    assert torch.equal(torch.rand(1), expected_torch)
    assert torch.equal(loaded["dataloader_generator_state"], generator.get_state())


def test_rng_restore_normalizes_serialized_tensor_to_cpu(monkeypatch):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    observed = {}
    original_set_rng_state = torch.set_rng_state

    def recording_set_rng_state(value):
        observed["device"] = value.device.type
        observed["dtype"] = value.dtype
        original_set_rng_state(value)

    monkeypatch.setattr(torch, "set_rng_state", recording_set_rng_state)
    restore_rng_state(state)

    assert observed == {"device": "cpu", "dtype": torch.uint8}


def test_inference_checkpoint_is_self_describing(tmp_path):
    path = tmp_path / "best_inference.pt"
    model = nn.Linear(2, 2)
    save_inference_checkpoint(
        path,
        model=model,
        epoch=3,
        monitor_name="validation_macro_f1",
        monitor_value=0.75,
        run_metadata={"run_id": "test", "model_name": "custom_cnn"},
        class_names=["a", "b"],
        class_to_index={"a": 0, "b": 1},
        preprocessing={"image_size": [224, 224]},
    )
    loaded = load_checkpoint(path, "cpu")
    assert loaded["checkpoint_type"] == "inference"
    assert loaded["monitor"] == {"name": "validation_macro_f1", "value": 0.75}
    assert loaded["class_to_index"] == {"a": 0, "b": 1}


def test_trainer_uses_macro_f1_and_keeps_frozen_batchnorm_fixed(tmp_path):
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.1)
    trainer = ModelTrainer(model, optimizer, device=torch.device("cpu"))
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]]),
            torch.tensor([0, 1, 1, 0]),
        ),
        batch_size=2,
    )
    original_mean = model[0].running_mean.clone()
    csv_path = tmp_path / "history.csv"
    trainer.fit(loader, loader, epochs=1, csv_logger_cb=CSVLogger(str(csv_path)))

    assert torch.equal(model[0].running_mean, original_mean)
    assert len(trainer.history["val_macro_f1"]) == 1
    with csv_path.open(encoding="utf-8", newline="") as file_handle:
        row = next(csv.DictReader(file_handle))
    assert row["phase"] == "initial"
    assert row["val_macro_f1"]


def test_evaluation_reports_roc_auc_and_benchmark_metadata():
    model = nn.Linear(2, 2)
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1, 0, 1]),
        ),
        batch_size=2,
    )
    metrics, targets, predictions, probabilities, latency = evaluate_model_metrics(
        model,
        loader,
        ["a", "b"],
        torch.device("cpu"),
        warmup_batches=0,
    )
    assert len(targets) == len(predictions) == len(probabilities) == 4
    assert "macro_roc_auc_ovr" in metrics
    assert metrics["benchmark"]["timed_images"] == 4
    assert metrics["throughput_images_per_second"] > 0
    assert latency > 0


def test_training_runner_creates_reusable_artifacts_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest_dir = tmp_path / "splits"
    manifest_dir.mkdir()
    (manifest_dir / "checksums.sha256").write_text("fixture\n", encoding="utf-8")
    (manifest_dir / "split_metadata.json").write_text(
        json.dumps({"dataset_revision": "fixture"}), encoding="utf-8"
    )
    config = {
        "project": {"device": "cpu"},
        "dataset": {
            "raw_dir": "unused",
            "manifest_dir": str(manifest_dir),
            "image_size": [16, 16],
            "num_workers": 0,
        },
        "augmentation": {
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
        },
        "training": {
            "seeds": [42],
            "batch_size": 2,
            "initial_epochs": 1,
            "fine_tune_epochs": 0,
            "learning_rate": 0.001,
            "fine_tune_learning_rate": 0.0001,
            "weight_decay": 0.0,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "model_selection_metric": "validation_macro_f1",
            "early_stopping_patience": 2,
            "reduce_lr_patience": 1,
            "reduce_lr_factor": 0.5,
            "min_lr": 0.00001,
        },
        "models": {
            "custom_cnn": {
                "conv_channels": [4, 8, 16],
                "dense_units": 8,
                "dropout_rate": 0.1,
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    images = torch.randn(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1])

    def fake_loaders(**_kwargs):
        generator = torch.Generator().manual_seed(42)
        train = DataLoader(
            TensorDataset(images, labels), batch_size=2, shuffle=True, generator=generator
        )
        validation = DataLoader(TensorDataset(images, labels), batch_size=2)
        test = DataLoader(TensorDataset(images, labels), batch_size=2)
        return train, validation, test, ["a", "b"], {"a": 0, "b": 1}

    monkeypatch.setattr("src.training.runner.get_dataloaders", fake_loaders)
    record = train_experiment(
        model_name="custom_cnn",
        seed=42,
        config_path=str(config_path),
        run_id="fixture-run",
        device_preference="cpu",
        allow_dirty=True,
    )
    best_path = tmp_path / record["artifacts"]["best_inference"]["path"]
    resume_path = tmp_path / record["artifacts"]["last_resume"]["path"]
    assert best_path.is_file()
    assert resume_path.is_file()
    assert record["epochs_completed"] == 1

    resumed = train_experiment(
        model_name="custom_cnn",
        seed=42,
        config_path=str(config_path),
        resume_path=str(resume_path),
        device_preference="cpu",
        allow_dirty=True,
    )
    assert resumed["run_id"] == "fixture-run"
    assert resumed["epochs_completed"] == 1


def test_interrupted_resume_matches_uninterrupted_training(tmp_path):
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]
    )
    labels = torch.tensor([0, 1, 0, 1, 0, 1])

    def make_model():
        return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Dropout(0.25), nn.Linear(4, 2))

    def make_loader(generator):
        return DataLoader(
            TensorDataset(inputs, labels),
            batch_size=2,
            shuffle=True,
            generator=generator,
        )

    set_seed(42)
    uninterrupted_model = make_model()
    uninterrupted_optimizer = torch.optim.Adam(uninterrupted_model.parameters(), lr=0.01)
    uninterrupted_trainer = ModelTrainer(
        uninterrupted_model, uninterrupted_optimizer, device=torch.device("cpu")
    )
    uninterrupted_loader = make_loader(torch.Generator().manual_seed(42))
    uninterrupted_trainer.fit(uninterrupted_loader, uninterrupted_loader, epochs=2)

    set_seed(42)
    interrupted_model = make_model()
    interrupted_optimizer = torch.optim.Adam(interrupted_model.parameters(), lr=0.01)
    interrupted_trainer = ModelTrainer(
        interrupted_model, interrupted_optimizer, device=torch.device("cpu")
    )
    interrupted_loader = make_loader(torch.Generator().manual_seed(42))
    interrupted_trainer.fit(interrupted_loader, interrupted_loader, epochs=1)
    resume_path = tmp_path / "resume.pt"
    save_resume_checkpoint(
        resume_path,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        epoch=1,
        phase="initial",
        phase_epoch=1,
        history=interrupted_trainer.history,
        run_metadata={"run_id": "equivalence"},
        callback_states={},
        class_names=["a", "b"],
        class_to_index={"a": 0, "b": 1},
        preprocessing={},
        dataloader_generator_state=interrupted_loader.generator.get_state(),
    )

    resumed_model = make_model()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    resumed_trainer = ModelTrainer(
        resumed_model, resumed_optimizer, device=torch.device("cpu")
    )
    resumed_loader = make_loader(torch.Generator().manual_seed(42))
    checkpoint = load_checkpoint(resume_path, "cpu")
    resumed_model.load_state_dict(checkpoint["model_state_dict"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resumed_trainer.load_history(checkpoint["history"])
    resumed_loader.generator.set_state(checkpoint["dataloader_generator_state"])
    restore_rng_state(checkpoint["rng_state"])
    resumed_trainer.fit(resumed_loader, resumed_loader, epochs=1, start_epoch=2)

    for expected, actual in zip(
        uninterrupted_model.parameters(), resumed_model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
    for key in ("train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1"):
        assert resumed_trainer.history[key] == pytest.approx(uninterrupted_trainer.history[key])
