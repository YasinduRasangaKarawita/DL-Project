"""Generate deterministic leaf-grouped PlantVillage split manifests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from .plantvillage_validation import resolve_leaf_group
from .split_exclusions import EXCLUSION_FIELDNAMES

MANIFEST_FIELDNAMES = (
    "relative_path",
    "class_name",
    "class_index",
    "group_id",
    "source_partition",
)
GENERATED_FILENAMES = (
    "train.csv",
    "validation.csv",
    "test.csv",
    "class_mapping.json",
    "split_metadata.json",
    "checksums.sha256",
)


def _read_official_paths(path: Path) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            value = line.strip()
            if not value:
                continue
            source_path = PurePosixPath(value)
            if source_path.parts[:2] != ("raw", "color") or len(source_path.parts) != 4:
                raise ValueError(f"Unexpected path in {path}:{line_number}: {value}")
            relative_path = PurePosixPath(*source_path.parts[2:]).as_posix()
            if relative_path in seen:
                raise ValueError(f"Duplicate path in {path}:{line_number}: {value}")
            seen.add(relative_path)
            values.append(relative_path)
    return sorted(values)


def _read_excluded_paths(path: Path, expected_revision: str) -> set[str]:
    with path.open(encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != EXCLUSION_FIELDNAMES:
            raise ValueError("Training-exclusion ledger columns do not match the schema")
        rows = list(reader)
    revisions = {row["dataset_revision"] for row in rows}
    if revisions != {expected_revision}:
        raise ValueError("Training-exclusion ledger dataset revision does not match source lock")
    return {row["train_path"] for row in rows}


def _stable_group_order(seed: int, class_name: str, group_id: str) -> str:
    value = f"{seed}\0{class_name}\0{group_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _select_validation_groups(
    grouped_paths: dict[str, list[str]],
    target_images: int,
    seed: int,
    class_name: str,
) -> set[str]:
    if len(grouped_paths) < 2:
        raise ValueError(f"Class {class_name} needs at least two groups for train/validation")
    ordered = sorted(
        grouped_paths,
        key=lambda group_id: _stable_group_order(seed, class_name, group_id),
    )
    prefix_counts = []
    running_count = 0
    for group_id in ordered[:-1]:
        running_count += len(grouped_paths[group_id])
        prefix_counts.append(running_count)
    best_count = min(
        prefix_counts,
        key=lambda count: (abs(count - target_images), count > target_images, count),
    )
    selected: set[str] = set()
    running_count = 0
    for group_id in ordered:
        if running_count == best_count:
            break
        selected.add(group_id)
        running_count += len(grouped_paths[group_id])
    return selected


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=MANIFEST_FIELDNAMES,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _build_rows(
    paths: list[str],
    source_partition: str,
    class_to_index: dict[str, int],
    leaf_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    rows = []
    groups: dict[str, list[str]] = defaultdict(list)
    for relative_path in paths:
        class_name = PurePosixPath(relative_path).parts[0]
        group_id, _, _ = resolve_leaf_group(relative_path, class_name, leaf_map)
        groups[group_id].append(relative_path)
        rows.append(
            {
                "relative_path": relative_path,
                "class_name": class_name,
                "class_index": class_to_index[class_name],
                "group_id": group_id,
                "source_partition": source_partition,
            }
        )
    return rows, groups


def build_grouped_split_manifests(
    metadata_dir: Path,
    lock_path: Path,
    exclusion_path: Path,
    output_dir: Path,
    *,
    validation_fraction: float,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create train/validation/test manifests while preserving resolved leaf groups."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    revision = str(lock["dataset"]["resolved_revision"])
    official_train = _read_official_paths(metadata_dir / "color_train.txt")
    official_test = _read_official_paths(metadata_dir / "color_test.txt")
    if set(official_train) & set(official_test):
        raise ValueError("Official train and test paths overlap")
    expected = lock.get("expected", {})
    if expected.get("official_train_images") not in (None, len(official_train)):
        raise ValueError("Official training count does not match source lock")
    if expected.get("official_test_images") not in (None, len(official_test)):
        raise ValueError("Official test count does not match source lock")

    excluded_paths = _read_excluded_paths(exclusion_path, revision)
    unknown_exclusions = excluded_paths - set(official_train)
    if unknown_exclusions:
        raise ValueError(f"Training exclusions contain {len(unknown_exclusions)} unknown paths")
    cleaned_train = sorted(set(official_train) - excluded_paths)
    all_classes = sorted(
        {PurePosixPath(path).parts[0] for path in cleaned_train + official_test}
    )
    if expected.get("classes") not in (None, len(all_classes)):
        raise ValueError("Class count does not match source lock")
    class_to_index = {class_name: index for index, class_name in enumerate(all_classes)}
    leaf_map = json.loads((metadata_dir / "leaf-map.json").read_text(encoding="utf-8"))

    candidate_rows, groups = _build_rows(
        cleaned_train, "official_train", class_to_index, leaf_map
    )
    groups_by_class: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for group_id, group_paths in groups.items():
        group_classes = {PurePosixPath(path).parts[0] for path in group_paths}
        if len(group_classes) != 1:
            raise ValueError(f"Leaf group spans classes: {group_id}")
        groups_by_class[next(iter(group_classes))][group_id] = group_paths

    validation_groups: set[str] = set()
    for class_name in all_classes:
        class_groups = groups_by_class[class_name]
        class_image_count = sum(len(paths) for paths in class_groups.values())
        target = max(1, min(class_image_count - 1, round(class_image_count * validation_fraction)))
        validation_groups.update(
            _select_validation_groups(class_groups, target, seed, class_name)
        )

    train_rows = sorted(
        (row for row in candidate_rows if row["group_id"] not in validation_groups),
        key=lambda row: row["relative_path"],
    )
    validation_rows = sorted(
        (row for row in candidate_rows if row["group_id"] in validation_groups),
        key=lambda row: row["relative_path"],
    )
    test_rows, test_groups = _build_rows(
        official_test, "official_test", class_to_index, leaf_map
    )
    test_rows.sort(key=lambda row: row["relative_path"])

    split_rows = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }
    split_paths = {
        split: {row["relative_path"] for row in rows}
        for split, rows in split_rows.items()
    }
    if split_paths["train"] & split_paths["validation"]:
        raise ValueError("Train and validation paths overlap")
    if (split_paths["train"] | split_paths["validation"]) != set(cleaned_train):
        raise ValueError("Train/validation manifests do not cover cleaned official training data")
    if split_paths["test"] != set(official_test):
        raise ValueError("Test manifest does not preserve the official test partition")
    train_groups = {row["group_id"] for row in train_rows}
    validation_group_ids = {row["group_id"] for row in validation_rows}
    if train_groups & validation_group_ids:
        raise ValueError("Leaf groups cross train and validation")
    if (train_groups | validation_group_ids) & set(test_groups):
        raise ValueError("Leaf groups cross development data and the official test partition")

    class_counts = {
        split: dict(sorted(Counter(row["class_name"] for row in rows).items()))
        for split, rows in split_rows.items()
    }
    metadata = {
        "schema_version": 1,
        "dataset_revision": revision,
        "method": "class_stratified_stable_hash_group_prefix_v1",
        "seed": seed,
        "validation_fraction_of_cleaned_official_train": validation_fraction,
        "source_counts": {
            "official_train": len(official_train),
            "official_test": len(official_test),
            "excluded_unique_training_images": len(excluded_paths),
            "cleaned_official_train": len(cleaned_train),
        },
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "class_counts": class_counts,
    }
    class_mapping = {
        "schema_version": 1,
        "classes": all_classes,
        "class_to_index": class_to_index,
    }
    contents = {
        "train.csv": _csv_bytes(train_rows),
        "validation.csv": _csv_bytes(validation_rows),
        "test.csv": _csv_bytes(test_rows),
        "class_mapping.json": _json_bytes(class_mapping),
        "split_metadata.json": _json_bytes(metadata),
    }
    checksums = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {filename}\n"
        for filename, content in sorted(contents.items())
    ).encode("utf-8")
    contents["checksums.sha256"] = checksums

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / filename for filename in contents if (output_dir / filename).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Generated split files already exist: " + ", ".join(path.name for path in existing)
        )
    for filename, content in contents.items():
        _write_bytes_atomic(output_dir / filename, content)
    return metadata
