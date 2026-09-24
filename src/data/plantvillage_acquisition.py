"""Reproducible acquisition helpers for the PlantVillage color dataset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

DATASET_NAME = "PlantVillage"
DATASET_REPO_ID = "mohanty/PlantVillage"
DATASET_CONFIG = "color"
DATASET_LICENSE = "CC BY-SA 3.0"
DATASET_DOI = "10.3389/fpls.2016.01419"

EXPECTED_IMAGE_COUNT = 54_305
EXPECTED_CLASS_COUNT = 38
EXPECTED_TRAIN_COUNT = 43_596
EXPECTED_TEST_COUNT = 10_709

SOURCE_FILES = (
    "data.zip",
    "leaf_grouping/leaf-map.json",
    "splits/color_train.txt",
    "splits/color_test.txt",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
COLOR_ARCHIVE_PREFIX = PurePosixPath("raw/color")


def configure_system_trust_store() -> None:
    """Use the Windows certificate store without weakening TLS verification."""
    if os.name != "nt":
        return
    try:
        import truststore
    except ImportError as exc:  # pragma: no cover - depends on the host OS
        raise RuntimeError(
            "truststore is required for verified HTTPS on Windows. "
            "Install the project requirements first."
        ) from exc
    truststore.inject_into_ssl()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_split_paths(path: Path) -> set[str]:
    relative_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            value = line.strip()
            if not value:
                continue
            source_path = PurePosixPath(value)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise ValueError(f"Unsafe path in {path}:{line_number}: {value}")
            if source_path.parts[:2] != COLOR_ARCHIVE_PREFIX.parts:
                raise ValueError(f"Unexpected path in {path}:{line_number}: {value}")
            relative = PurePosixPath(*source_path.parts[2:])
            if len(relative.parts) != 2 or relative.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unexpected image path in {path}:{line_number}: {value}")
            normalized = relative.as_posix()
            if normalized in relative_paths:
                raise ValueError(f"Duplicate path in {path}:{line_number}: {value}")
            relative_paths.add(normalized)
    return relative_paths


def validate_official_splits(
    color_dir: Path,
    train_path: Path,
    test_path: Path,
    expected_train: int = EXPECTED_TRAIN_COUNT,
    expected_test: int = EXPECTED_TEST_COUNT,
) -> dict[str, Any]:
    """Prove the official split lists are disjoint and cover the local color dataset."""
    train_paths = _read_split_paths(train_path)
    test_paths = _read_split_paths(test_path)
    overlap = train_paths & test_paths
    if overlap:
        raise RuntimeError(f"Official train/test split overlap detected: {len(overlap)} images")
    if len(train_paths) != expected_train:
        raise RuntimeError(f"Expected {expected_train} train images, found {len(train_paths)}")
    if len(test_paths) != expected_test:
        raise RuntimeError(f"Expected {expected_test} test images, found {len(test_paths)}")

    local_paths = {
        path.relative_to(color_dir).as_posix()
        for path in color_dir.glob("*/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    listed_paths = train_paths | test_paths
    missing = listed_paths - local_paths
    unlisted = local_paths - listed_paths
    if missing or unlisted:
        raise RuntimeError(
            "Official split coverage mismatch: "
            f"{len(missing)} referenced files missing; {len(unlisted)} local files unlisted"
        )
    return {
        "train": len(train_paths),
        "test": len(test_paths),
        "overlap": 0,
        "covered_images": len(listed_paths),
    }


def _lfs_metadata(sibling: Any) -> tuple[str | None, int | None]:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None, getattr(sibling, "size", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256"), lfs.get("size")
    return getattr(lfs, "sha256", None), getattr(lfs, "size", None)


def resolve_source_lock(
    revision: str = "main",
    repo_id: str = DATASET_REPO_ID,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve a moving revision to an immutable commit and source-file metadata."""
    configure_system_trust_store()
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised by CLI installation check
        raise RuntimeError(
            "huggingface-hub is required. Install the project requirements first."
        ) from exc

    info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
    siblings = {sibling.rfilename: sibling for sibling in info.siblings or []}
    missing = sorted(set(SOURCE_FILES) - set(siblings))
    if missing:
        raise RuntimeError(f"Dataset repository is missing required files: {missing}")

    source_files: dict[str, dict[str, Any]] = {}
    for filename in SOURCE_FILES:
        sha256, size = _lfs_metadata(siblings[filename])
        if not sha256:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=info.sha,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
            )
            sha256 = sha256_file(local_path)
        source_files[filename] = {
            "sha256": sha256,
            "size_bytes": size,
        }

    return {
        "schema_version": 1,
        "dataset": {
            "name": DATASET_NAME,
            "repository": repo_id,
            "repository_type": "dataset",
            "configuration": DATASET_CONFIG,
            "resolved_revision": info.sha,
            "license_declared_by_source": DATASET_LICENSE,
            "citation_doi": DATASET_DOI,
        },
        "expected": {
            "images": EXPECTED_IMAGE_COUNT,
            "classes": EXPECTED_CLASS_COUNT,
            "official_train_images": EXPECTED_TRAIN_COUNT,
            "official_test_images": EXPECTED_TEST_COUNT,
        },
        "source_files": source_files,
    }


def write_source_lock(lock: dict[str, Any], path: Path) -> None:
    """Write stable, reviewable source metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_source_lock(path: Path) -> dict[str, Any]:
    """Load and minimally validate a PlantVillage source lock."""
    lock = json.loads(path.read_text(encoding="utf-8"))
    dataset = lock.get("dataset", {})
    if dataset.get("repository") != DATASET_REPO_ID:
        raise ValueError(f"Unexpected dataset repository in {path}")
    if not dataset.get("resolved_revision"):
        raise ValueError(f"Missing resolved_revision in {path}")
    return lock


def download_locked_files(
    lock: dict[str, Any],
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """Download the locked source files and verify declared SHA-256 hashes."""
    configure_system_trust_store()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised by CLI installation check
        raise RuntimeError(
            "huggingface-hub is required. Install the project requirements first."
        ) from exc

    dataset = lock["dataset"]
    revision = dataset["resolved_revision"]
    repo_id = dataset["repository"]
    downloaded: dict[str, Path] = {}

    for filename, metadata in lock["source_files"].items():
        local_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
        expected_hash = metadata.get("sha256")
        if expected_hash:
            actual_hash = sha256_file(local_path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"SHA-256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
                )
        downloaded[filename] = local_path

    return downloaded


def _safe_color_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir() or path.is_absolute() or ".." in path.parts:
            continue
        if path.parts[:2] != COLOR_ARCHIVE_PREFIX.parts:
            continue
        relative = PurePosixPath(*path.parts[2:])
        if len(relative.parts) != 2 or relative.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        yield member


def inspect_extracted_dataset(color_dir: Path) -> dict[str, Any]:
    """Return class and image counts for an extracted class-folder dataset."""
    class_counts: dict[str, int] = {}
    if color_dir.exists():
        for class_dir in sorted(path for path in color_dir.iterdir() if path.is_dir()):
            class_counts[class_dir.name] = sum(
                1
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
    return {
        "class_count": len(class_counts),
        "image_count": sum(class_counts.values()),
        "class_counts": class_counts,
    }


def extract_color_dataset(
    archive_path: Path,
    destination: Path,
    expected_images: int = EXPECTED_IMAGE_COUNT,
    expected_classes: int = EXPECTED_CLASS_COUNT,
) -> dict[str, Any]:
    """Safely extract only the original color images using an atomic staging directory."""
    if destination.exists():
        existing = inspect_extracted_dataset(destination)
        if (
            existing["image_count"] == expected_images
            and existing["class_count"] == expected_classes
        ):
            return existing
        raise FileExistsError(
            f"Destination already exists but is incomplete or unexpected: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".plantvillage-color-", dir=destination.parent))
    staging_color = staging_root / "color"
    staging_color.mkdir()

    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = list(_safe_color_members(archive))
            for member in members:
                archive_path_parts = PurePosixPath(member.filename).parts
                relative = Path(*archive_path_parts[2:])
                target = staging_color / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)

        summary = inspect_extracted_dataset(staging_color)
        if summary["image_count"] != expected_images:
            raise RuntimeError(
                f"Expected {expected_images} color images, found {summary['image_count']}"
            )
        if summary["class_count"] != expected_classes:
            raise RuntimeError(
                f"Expected {expected_classes} classes, found {summary['class_count']}"
            )
        os.replace(staging_color, destination)
        return summary
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def copy_source_metadata(downloaded: dict[str, Path], metadata_dir: Path) -> None:
    """Copy the locked grouping and split metadata beside the local dataset."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        if filename == "data.zip":
            continue
        target = metadata_dir / Path(filename).name
        shutil.copy2(downloaded[filename], target)


def build_local_provenance(
    lock: dict[str, Any],
    downloaded: dict[str, Path],
    summary: dict[str, Any],
    color_dir: Path,
) -> dict[str, Any]:
    """Create machine-readable local acquisition evidence."""
    train_path = downloaded["splits/color_train.txt"]
    test_path = downloaded["splits/color_test.txt"]
    official_splits = validate_official_splits(
        color_dir=color_dir,
        train_path=train_path,
        test_path=test_path,
        expected_train=lock["expected"]["official_train_images"],
        expected_test=lock["expected"]["official_test_images"],
    )
    return {
        "schema_version": 1,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": lock["dataset"],
        "archive_sha256": sha256_file(downloaded["data.zip"]),
        "observed": summary,
        "official_splits": official_splits,
    }
