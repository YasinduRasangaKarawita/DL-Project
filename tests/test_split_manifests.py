import csv
import json

from src.data.split_exclusions import EXCLUSION_FIELDNAMES
from src.data.split_manifests import build_grouped_split_manifests


def _write_fixture(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    train_paths = [
        "Class_A/a1.JPG",
        "Class_A/a2.JPG",
        "Class_A/a3.JPG",
        "Class_A/a4.JPG",
        "Class_B/b1.JPG",
        "Class_B/b2.JPG",
        "Class_B/b3.JPG",
        "Class_B/b4.JPG",
    ]
    test_paths = ["Class_A/a-test.JPG", "Class_B/b-test.JPG"]
    (metadata_dir / "color_train.txt").write_text(
        "".join(f"raw/color/{path}\n" for path in train_paths), encoding="utf-8"
    )
    (metadata_dir / "color_test.txt").write_text(
        "".join(f"raw/color/{path}\n" for path in test_paths), encoding="utf-8"
    )
    (metadata_dir / "leaf-map.json").write_text(
        json.dumps(
            {
                "a1": ["Class_A:::leaf-1"],
                "a2": ["Class_A:::leaf-1"],
                "a3": ["Class_A:::leaf-2"],
                "a4": ["Class_A:::leaf-3"],
                "a-test": ["Class_A:::leaf-test"],
                "b1": ["Class_B:::leaf-1"],
                "b2": ["Class_B:::leaf-1"],
                "b3": ["Class_B:::leaf-2"],
                "b4": ["Class_B:::leaf-3"],
                "b-test": ["Class_B:::leaf-test"],
            }
        ),
        encoding="utf-8",
    )
    lock_path = tmp_path / "source.lock.json"
    lock_path.write_text(
        json.dumps({"dataset": {"resolved_revision": "locked-revision"}}),
        encoding="utf-8",
    )
    exclusion_path = tmp_path / "exclusions.csv"
    with exclusion_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=EXCLUSION_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "exclusion_id": "exclude-fixture",
                "dataset_revision": "locked-revision",
                "reason": "exact_duplicate_cross_split",
                "train_path": "Class_A/a4.JPG",
                "related_test_path": "Class_A/a-test.JPG",
                "evidence": "fixture-sha256",
            }
        )
    return metadata_dir, lock_path, exclusion_path


def _read_manifest(path):
    with path.open(encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def test_builds_deterministic_grouped_manifests(tmp_path):
    metadata_dir, lock_path, exclusion_path = _write_fixture(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    metadata = build_grouped_split_manifests(
        metadata_dir,
        lock_path,
        exclusion_path,
        first_output,
        validation_fraction=0.4,
        seed=42,
    )
    build_grouped_split_manifests(
        metadata_dir,
        lock_path,
        exclusion_path,
        second_output,
        validation_fraction=0.4,
        seed=42,
    )

    for filename in (
        "train.csv",
        "validation.csv",
        "test.csv",
        "class_mapping.json",
        "split_metadata.json",
        "checksums.sha256",
    ):
        assert (first_output / filename).read_bytes() == (second_output / filename).read_bytes()

    train_rows = _read_manifest(first_output / "train.csv")
    validation_rows = _read_manifest(first_output / "validation.csv")
    test_rows = _read_manifest(first_output / "test.csv")
    train_paths = {row["relative_path"] for row in train_rows}
    validation_paths = {row["relative_path"] for row in validation_rows}
    assert "Class_A/a4.JPG" not in train_paths | validation_paths
    assert {row["relative_path"] for row in test_rows} == {
        "Class_A/a-test.JPG",
        "Class_B/b-test.JPG",
    }
    assert {row["group_id"] for row in train_rows}.isdisjoint(
        {row["group_id"] for row in validation_rows}
    )
    assert metadata["source_counts"]["excluded_unique_training_images"] == 1
    assert set(json.loads((first_output / "class_mapping.json").read_text())["classes"]) == {
        "Class_A",
        "Class_B",
    }


def test_rejects_invalid_validation_fraction(tmp_path):
    metadata_dir, lock_path, exclusion_path = _write_fixture(tmp_path)

    for invalid_fraction in (0.0, 1.0):
        try:
            build_grouped_split_manifests(
                metadata_dir,
                lock_path,
                exclusion_path,
                tmp_path / f"invalid-{invalid_fraction}",
                validation_fraction=invalid_fraction,
                seed=42,
            )
        except ValueError as exc:
            assert "between 0 and 1" in str(exc)
        else:
            raise AssertionError("Invalid validation fraction was accepted")
