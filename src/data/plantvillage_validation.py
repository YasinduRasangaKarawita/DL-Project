"""Data-quality and leakage auditing for the locked PlantVillage dataset."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image

from .plantvillage_acquisition import IMAGE_SUFFIXES, load_source_lock, sha256_file

PERCEPTUAL_HASH_NAME = "dhash-64"
DEFAULT_NEAR_DUPLICATE_DISTANCE = 4


def normalize_image_identifier(filename: str) -> str:
    """Return the lookup key used by PlantVillage's upstream leaf-map logic."""
    identifier = Path(filename.replace("_final_masked", "")).stem
    if "___" in identifier:
        identifier = identifier.rsplit("___", 1)[-1]
    return identifier.casefold().split("copy", 1)[0].strip()


def resolve_leaf_group(
    relative_path: str,
    class_name: str,
    leaf_map: dict[str, list[str]],
) -> tuple[str, str, str | None]:
    """Resolve a stable leaf group, safely falling back to one group per image.

    Returns ``(group_id, status, declared_class)``. A per-image fallback prevents
    accidental grouping when the upstream metadata has no unambiguous match.
    """
    lookup_key = normalize_image_identifier(PurePosixPath(relative_path).name)
    suggestions = leaf_map.get(lookup_key, [])
    if len(suggestions) == 1:
        suggestion = suggestions[0]
        return suggestion, "mapped", suggestion.split(":::", 1)[0]

    matches = [
        suggestion
        for suggestion in suggestions
        if suggestion.split(":::", 1)[0] == class_name
    ]
    if len(matches) == 1:
        suggestion = matches[0]
        return suggestion, "mapped", class_name

    status = "unmapped" if not suggestions else "ambiguous"
    return f"fallback:{relative_path}", status, None


def difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    """Calculate a deterministic 64-bit difference hash for visual screening."""
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    value = 0
    for row in range(hash_size):
        for column in range(hash_size):
            left = grayscale.getpixel((column, row))
            right = grayscale.getpixel((column + 1, row))
            value = (value << 1) | int(left > right)
    return value


def _read_official_split(path: Path) -> set[str]:
    entries: set[str] = set()
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            value = line.strip()
            if not value:
                continue
            source_path = PurePosixPath(value)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise ValueError(f"Unsafe path in {path}:{line_number}: {value}")
            if source_path.parts[:2] != ("raw", "color"):
                raise ValueError(f"Unexpected path in {path}:{line_number}: {value}")
            relative = PurePosixPath(*source_path.parts[2:])
            if len(relative.parts) != 2 or relative.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unexpected image path in {path}:{line_number}: {value}")
            normalized = relative.as_posix()
            if normalized in entries:
                raise ValueError(f"Duplicate path in {path}:{line_number}: {value}")
            entries.add(normalized)
    return entries


def _iter_image_paths(color_dir: Path) -> Iterable[Path]:
    for class_dir in sorted(path for path in color_dir.iterdir() if path.is_dir()):
        yield from sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )


def _limited(items: Iterable[Any], limit: int) -> list[Any]:
    result = []
    for item in items:
        if len(result) == limit:
            break
        result.append(item)
    return result


def _split_for(relative_path: str, train_paths: set[str], test_paths: set[str]) -> str:
    if relative_path in train_paths:
        return "train"
    if relative_path in test_paths:
        return "test"
    return "unlisted"


def _near_duplicate_summary(
    records: list[dict[str, Any]],
    maximum_distance: int,
    max_examples: int,
) -> dict[str, Any]:
    if not 0 <= maximum_distance <= 16:
        raise ValueError("near-duplicate distance must be between 0 and 16")

    block_count = maximum_distance + 1
    block_widths = [64 // block_count] * block_count
    for index in range(64 % block_count):
        block_widths[index] += 1

    indexes: list[dict[int, list[int]]] = [defaultdict(list) for _ in block_widths]
    pair_count = 0
    cross_split_pair_count = 0
    examples: list[dict[str, Any]] = []
    cross_split_examples: list[dict[str, Any]] = []

    for current_index, current in enumerate(records):
        candidates: set[int] = set()
        shift = 64
        block_values: list[int] = []
        for block_index, width in enumerate(block_widths):
            shift -= width
            value = (current["dhash"] >> shift) & ((1 << width) - 1)
            block_values.append(value)
            candidates.update(indexes[block_index].get(value, []))

        for candidate_index in sorted(candidates):
            candidate = records[candidate_index]
            if candidate["sha256"] == current["sha256"]:
                continue
            distance = (candidate["dhash"] ^ current["dhash"]).bit_count()
            if distance > maximum_distance:
                continue
            pair_count += 1
            example = {
                "left": candidate["path"],
                "right": current["path"],
                "hamming_distance": distance,
                "left_split": candidate["split"],
                "right_split": current["split"],
            }
            if len(examples) < max_examples:
                examples.append(example)
            if {candidate["split"], current["split"]} == {"train", "test"}:
                cross_split_pair_count += 1
                if len(cross_split_examples) < max_examples:
                    cross_split_examples.append(example)

        for block_index, value in enumerate(block_values):
            indexes[block_index][value].append(current_index)

    return {
        "algorithm": PERCEPTUAL_HASH_NAME,
        "maximum_hamming_distance": maximum_distance,
        "pairs": pair_count,
        "cross_split_pairs": cross_split_pair_count,
        "examples": examples,
        "cross_split_examples": cross_split_examples,
        "note": "Perceptual matches are heuristic review candidates, not proven duplicates.",
    }


def _write_json_atomic(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_handle:
            json.dump(report, file_handle, indent=2, sort_keys=True)
            file_handle.write("\n")
        os.replace(temporary_name, report_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def validate_plantvillage_dataset(
    color_dir: Path,
    metadata_dir: Path,
    lock_path: Path,
    report_path: Path,
    near_duplicate_distance: int = DEFAULT_NEAR_DUPLICATE_DISTANCE,
    max_examples: int = 100,
) -> dict[str, Any]:
    """Audit image integrity, duplicates, metadata coverage, and split leakage."""
    if not color_dir.is_dir():
        raise FileNotFoundError(f"PlantVillage color directory does not exist: {color_dir}")
    if max_examples < 0:
        raise ValueError("max_examples cannot be negative")

    lock = load_source_lock(lock_path)
    expected = lock.get("expected", {})
    train_paths = _read_official_split(metadata_dir / "color_train.txt")
    test_paths = _read_official_split(metadata_dir / "color_test.txt")
    split_overlap = train_paths & test_paths
    listed_paths = train_paths | test_paths
    leaf_map = json.loads((metadata_dir / "leaf-map.json").read_text(encoding="utf-8"))

    expected_classes = {PurePosixPath(path).parts[0] for path in listed_paths}
    actual_classes = {path.name for path in color_dir.iterdir() if path.is_dir()}
    image_paths = list(_iter_image_paths(color_dir))
    local_paths = {path.relative_to(color_dir).as_posix() for path in image_paths}

    corrupted: list[dict[str, str]] = []
    empty_files: list[str] = []
    unsupported_modes: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    mapping_statuses: Counter[str] = Counter()
    metadata_class_mismatches: list[dict[str, str]] = []
    exact_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    leaf_groups: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for path in image_paths:
        relative_path = path.relative_to(color_dir).as_posix()
        class_name = path.parent.name
        split = _split_for(relative_path, train_paths, test_paths)
        class_counts[class_name] += 1

        if path.stat().st_size == 0:
            empty_files.append(relative_path)
            continue

        digest = sha256_file(path)
        exact_hashes[digest].append({"path": relative_path, "split": split})
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                modes[image.mode] += 1
                dimensions[f"{image.width}x{image.height}"] += 1
                if image.mode != "RGB":
                    unsupported_modes.append({"path": relative_path, "mode": image.mode})
                perceptual_hash = difference_hash(image)
        except Exception as exc:  # Pillow exposes several format-specific exception types
            corrupted.append({"path": relative_path, "error": str(exc)})
            continue

        group_id, mapping_status, declared_class = resolve_leaf_group(
            relative_path, class_name, leaf_map
        )
        mapping_statuses[mapping_status] += 1
        if declared_class is not None and declared_class != class_name:
            metadata_class_mismatches.append(
                {
                    "path": relative_path,
                    "folder_class": class_name,
                    "metadata_class": declared_class,
                }
            )
        group = leaf_groups.setdefault(group_id, {"splits": set(), "paths": []})
        group["splits"].add(split)
        group["paths"].append(relative_path)
        records.append(
            {
                "path": relative_path,
                "split": split,
                "sha256": digest,
                "dhash": perceptual_hash,
            }
        )

    exact_groups = [
        {"sha256": digest, "members": members}
        for digest, members in exact_hashes.items()
        if len(members) > 1
    ]
    exact_cross_split = [
        group
        for group in exact_groups
        if {member["split"] for member in group["members"]} >= {"train", "test"}
    ]
    leaking_leaf_groups = [
        {"group_id": group_id, "paths": group["paths"]}
        for group_id, group in leaf_groups.items()
        if group["splits"] >= {"train", "test"} and not group_id.startswith("fallback:")
    ]
    near_duplicates = _near_duplicate_summary(
        records, maximum_distance=near_duplicate_distance, max_examples=max_examples
    )

    hard_failures = []
    if split_overlap:
        hard_failures.append(f"official split path overlap: {len(split_overlap)}")
    if listed_paths - local_paths:
        hard_failures.append(f"official split paths missing locally: {len(listed_paths - local_paths)}")
    if local_paths - listed_paths:
        hard_failures.append(f"local images absent from official splits: {len(local_paths - listed_paths)}")
    if expected_classes - actual_classes:
        hard_failures.append(f"missing class folders: {len(expected_classes - actual_classes)}")
    if actual_classes - expected_classes:
        hard_failures.append(f"unexpected class folders: {len(actual_classes - expected_classes)}")
    if expected.get("images") is not None and len(image_paths) != expected["images"]:
        hard_failures.append(
            f"locked image count mismatch: expected {expected['images']}, found {len(image_paths)}"
        )
    if expected.get("classes") is not None and len(actual_classes) != expected["classes"]:
        hard_failures.append(
            f"locked class count mismatch: expected {expected['classes']}, found {len(actual_classes)}"
        )
    if (
        expected.get("official_train_images") is not None
        and len(train_paths) != expected["official_train_images"]
    ):
        hard_failures.append(
            "locked train count mismatch: "
            f"expected {expected['official_train_images']}, found {len(train_paths)}"
        )
    if (
        expected.get("official_test_images") is not None
        and len(test_paths) != expected["official_test_images"]
    ):
        hard_failures.append(
            "locked test count mismatch: "
            f"expected {expected['official_test_images']}, found {len(test_paths)}"
        )
    if empty_files:
        hard_failures.append(f"empty image files: {len(empty_files)}")
    if corrupted:
        hard_failures.append(f"unreadable images: {len(corrupted)}")
    if exact_cross_split:
        hard_failures.append(f"exact duplicate groups crossing train/test: {len(exact_cross_split)}")
    if leaking_leaf_groups:
        hard_failures.append(f"leaf groups crossing train/test: {len(leaking_leaf_groups)}")

    warnings = []
    if mapping_statuses["unmapped"] or mapping_statuses["ambiguous"]:
        warnings.append(
            "upstream leaf metadata does not unambiguously cover "
            f"{mapping_statuses['unmapped'] + mapping_statuses['ambiguous']} images"
        )
    if metadata_class_mismatches:
        warnings.append(
            f"upstream metadata labels differ from {len(metadata_class_mismatches)} image records"
        )
    if unsupported_modes:
        warnings.append(
            f"readable non-RGB images require deterministic RGB conversion: {len(unsupported_modes)}"
        )
    if exact_groups:
        warnings.append(f"exact duplicate groups require review: {len(exact_groups)}")
    if near_duplicates["pairs"]:
        warnings.append(f"perceptual near-duplicate pairs require review: {near_duplicates['pairs']}")
    if near_duplicates["cross_split_pairs"]:
        warnings.append(
            "perceptual near-duplicate pairs cross the official train/test boundary: "
            f"{near_duplicates['cross_split_pairs']}"
        )

    status = "fail" if hard_failures else "warning" if warnings else "pass"
    report = {
        "schema_version": 1,
        "dataset": {
            "name": lock["dataset"].get("name", "PlantVillage"),
            "configuration": lock["dataset"].get("configuration", "color"),
            "source_revision": lock["dataset"]["resolved_revision"],
            "expected_images": expected.get("images"),
            "expected_classes": expected.get("classes"),
        },
        "parameters": {
            "perceptual_hash": PERCEPTUAL_HASH_NAME,
            "near_duplicate_maximum_hamming_distance": near_duplicate_distance,
            "maximum_examples_per_category": max_examples,
        },
        "summary": {
            "status": status,
            "hard_failures": hard_failures,
            "warnings": warnings,
        },
        "official_splits": {
            "train_images": len(train_paths),
            "test_images": len(test_paths),
            "path_overlap": len(split_overlap),
            "missing_local_images": len(listed_paths - local_paths),
            "unlisted_local_images": len(local_paths - listed_paths),
        },
        "integrity": {
            "images_scanned": len(image_paths),
            "readable_images": len(records),
            "empty_files": empty_files[:max_examples],
            "empty_file_count": len(empty_files),
            "corrupted_files": corrupted[:max_examples],
            "corrupted_file_count": len(corrupted),
            "image_modes": dict(sorted(modes.items())),
            "non_rgb_images": unsupported_modes[:max_examples],
            "non_rgb_image_count": len(unsupported_modes),
            "dimensions": dict(sorted(dimensions.items())),
        },
        "classes": {
            "actual_count": len(actual_classes),
            "expected_count": len(expected_classes),
            "counts": dict(sorted(class_counts.items())),
            "missing_folders": sorted(expected_classes - actual_classes),
            "unexpected_folders": sorted(actual_classes - expected_classes),
        },
        "leaf_grouping": {
            "mapped_images": mapping_statuses["mapped"],
            "unmapped_images": mapping_statuses["unmapped"],
            "ambiguous_images": mapping_statuses["ambiguous"],
            "metadata_coverage_fraction": (
                mapping_statuses["mapped"] / len(records) if records else 0.0
            ),
            "resolved_groups": len(leaf_groups),
            "train_test_group_overlap": len(leaking_leaf_groups),
            "train_test_group_overlap_examples": _limited(leaking_leaf_groups, max_examples),
            "metadata_class_mismatch_count": len(metadata_class_mismatches),
            "metadata_class_mismatch_examples": metadata_class_mismatches[:max_examples],
        },
        "duplicates": {
            "exact": {
                "groups": len(exact_groups),
                "images": sum(len(group["members"]) for group in exact_groups),
                "cross_split_groups": len(exact_cross_split),
                "examples": exact_groups[:max_examples],
                "cross_split_examples": exact_cross_split[:max_examples],
            },
            "perceptual": near_duplicates,
        },
    }
    _write_json_atomic(report, report_path)
    return report
