import json
import zipfile
from pathlib import Path

import pytest

from src.data.plantvillage_acquisition import (
    extract_color_dataset,
    inspect_extracted_dataset,
    load_source_lock,
    sha256_file,
    validate_official_splits,
    write_source_lock,
)


def _write_test_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("raw/color/Apple___healthy/apple.jpg", b"apple")
        archive.writestr("raw/color/Tomato___healthy/tomato.png", b"tomato")
        archive.writestr("raw/grayscale/Apple___healthy/apple.jpg", b"gray")
        archive.writestr("../escape.jpg", b"unsafe")


def test_sha256_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"plantvillage")
    assert sha256_file(source) == "5ae48623f225565a0b6f5d18491d23ba7f6a0a3bf3af2b02fe1e3c276f5701b2"


def test_extracts_only_safe_color_images(tmp_path):
    archive_path = tmp_path / "data.zip"
    destination = tmp_path / "plantvillage" / "color"
    _write_test_archive(archive_path)

    summary = extract_color_dataset(
        archive_path,
        destination,
        expected_images=2,
        expected_classes=2,
    )

    assert summary["image_count"] == 2
    assert summary["class_count"] == 2
    assert (destination / "Apple___healthy" / "apple.jpg").read_bytes() == b"apple"
    assert not (tmp_path / "escape.jpg").exists()
    assert inspect_extracted_dataset(destination) == summary


def test_existing_incomplete_destination_is_rejected(tmp_path):
    archive_path = tmp_path / "data.zip"
    destination = tmp_path / "color"
    destination.mkdir()
    _write_test_archive(archive_path)

    with pytest.raises(FileExistsError):
        extract_color_dataset(
            archive_path,
            destination,
            expected_images=2,
            expected_classes=2,
        )


def test_source_lock_round_trip(tmp_path):
    lock_path = tmp_path / "source.lock.json"
    lock = {
        "schema_version": 1,
        "dataset": {
            "repository": "mohanty/PlantVillage",
            "resolved_revision": "abc123",
        },
        "expected": {},
        "source_files": {},
    }

    write_source_lock(lock, lock_path)

    assert load_source_lock(lock_path) == lock
    assert json.loads(lock_path.read_text(encoding="utf-8")) == lock


def test_official_splits_are_disjoint_and_cover_images(tmp_path):
    color_dir = tmp_path / "color"
    (color_dir / "Apple___healthy").mkdir(parents=True)
    (color_dir / "Tomato___healthy").mkdir(parents=True)
    (color_dir / "Apple___healthy" / "apple.jpg").write_bytes(b"apple")
    (color_dir / "Tomato___healthy" / "tomato.png").write_bytes(b"tomato")
    train_path = tmp_path / "color_train.txt"
    test_path = tmp_path / "color_test.txt"
    train_path.write_text("raw/color/Apple___healthy/apple.jpg\n", encoding="utf-8")
    test_path.write_text("raw/color/Tomato___healthy/tomato.png\n", encoding="utf-8")

    summary = validate_official_splits(
        color_dir,
        train_path,
        test_path,
        expected_train=1,
        expected_test=1,
    )

    assert summary == {"train": 1, "test": 1, "overlap": 0, "covered_images": 2}


def test_official_split_overlap_is_rejected(tmp_path):
    color_dir = tmp_path / "color" / "Apple___healthy"
    color_dir.mkdir(parents=True)
    (color_dir / "apple.jpg").write_bytes(b"apple")
    train_path = tmp_path / "color_train.txt"
    test_path = tmp_path / "color_test.txt"
    split_line = "raw/color/Apple___healthy/apple.jpg\n"
    train_path.write_text(split_line, encoding="utf-8")
    test_path.write_text(split_line, encoding="utf-8")

    with pytest.raises(RuntimeError, match="overlap"):
        validate_official_splits(
            color_dir.parent,
            train_path,
            test_path,
            expected_train=1,
            expected_test=1,
        )
