"""Versioned, resumable and inference checkpoint helpers."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(map_location="cuda")`` also maps serialized CPU RNG tensors
    # to CUDA. PyTorch's RNG restoration APIs require CPU ByteTensors.
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [generator_state.detach().cpu() for generator_state in state["torch_cuda"]]
        )


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing checkpoint schema_version")
    return checkpoint


def save_resume_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    phase: str,
    phase_epoch: int,
    history: dict[str, list[Any]],
    run_metadata: dict[str, Any],
    callback_states: dict[str, dict[str, Any]],
    class_names: list[str],
    class_to_index: dict[str, int],
    preprocessing: dict[str, Any],
    dataloader_generator_state: torch.Tensor | None = None,
) -> str:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": "resume",
        "epoch": epoch,
        "phase": phase,
        "phase_epoch": phase_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gradient_scaler_state_dict": None,
        "history": history,
        "callbacks": callback_states,
        "rng_state": capture_rng_state(),
        "dataloader_generator_state": dataloader_generator_state,
        "run_metadata": run_metadata,
        "class_names": class_names,
        "class_to_index": class_to_index,
        "preprocessing": preprocessing,
    }
    _atomic_torch_save(payload, path)
    return sha256_file(path)


def save_inference_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    monitor_name: str,
    monitor_value: float,
    run_metadata: dict[str, Any],
    class_names: list[str],
    class_to_index: dict[str, int],
    preprocessing: dict[str, Any],
) -> str:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": "inference",
        "epoch": epoch,
        "monitor": {"name": monitor_name, "value": monitor_value},
        "model_state_dict": model.state_dict(),
        "run_metadata": run_metadata,
        "class_names": class_names,
        "class_to_index": class_to_index,
        "preprocessing": preprocessing,
    }
    _atomic_torch_save(payload, path)
    return sha256_file(path)
