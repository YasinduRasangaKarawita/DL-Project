"""Shared experiment identity, environment and result-record helpers."""

from __future__ import annotations

import json
import platform
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sklearn
import torch
import torchvision

from .checkpoints import sha256_file

RESULT_SCHEMA_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def generate_run_id(model_name: str, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_commit = current_git_commit()[:7]
    return f"{timestamp}_{model_name}_seed{seed}_{short_commit}"


def current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_worktree_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def environment_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if device.type == "cuda":
        record["gpu_name"] = torch.cuda.get_device_name(device)
    return record


def build_run_metadata(
    *,
    run_id: str,
    model_name: str,
    seed: int,
    config: Mapping[str, Any],
    manifest_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    manifest_path = Path(manifest_dir)
    split_metadata = json.loads(
        (manifest_path / "split_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "model_name": model_name,
        "seed": seed,
        "git_commit": current_git_commit(),
        "git_worktree_dirty": git_worktree_is_dirty(),
        "created_at_utc": utc_timestamp(),
        "manifest_bundle_sha256": sha256_file(manifest_path / "checksums.sha256"),
        "dataset_revision": split_metadata.get("dataset_revision"),
        "config": deepcopy(dict(config)),
        "environment": environment_record(device),
    }


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
