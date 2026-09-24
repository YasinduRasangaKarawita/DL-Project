import json
from pathlib import Path

from PIL import Image

from src.data.plantvillage_validation import (
    difference_hash,
    normalize_image_identifier,
    resolve_leaf_group,
    validate_plantvillage_dataset,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    color_dir = tmp_path / "color"
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    _write_image(color_dir / "Apple___healthy" / "a___leaf_a.JPG", (0, 80, 0))
    _write_image(color_dir / "Apple___healthy" / "b___leaf_b.JPG", (0, 120, 0))
    train_line = "raw/color/Apple___healthy/a___leaf_a.JPG\n"
    test_line = "raw/color/Apple___healthy/b___leaf_b.JPG\n"
    (metadata_dir / "color_train.txt").write_text(train_line, encoding="utf-8")
    (metadata_dir / "color_test.txt").write_text(test_line, encoding="utf-8")
    (metadata_dir / "leaf-map.json").write_text(
        json.dumps(
            {
                "leaf_a": ["Apple___healthy:::1.0"],
                "leaf_b": ["Apple___healthy:::2.0"],
            }
        ),
        encoding="utf-8",
    )
    lock_path = tmp_path / "source.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "name": "PlantVillage",
                    "repository": "mohanty/PlantVillage",
                    "configuration": "color",
                    "resolved_revision": "test-revision",
                },
                "expected": {"images": 2, "classes": 1},
            }
        ),
        encoding="utf-8",
    )
    return color_dir, metadata_dir, lock_path, tmp_path / "report.json"


def test_normalizes_identifier_using_upstream_copy_rule():
    assert normalize_image_identifier("uuid___RS_GLSp 4378 copy 2.JPG") == "rs_glsp 4378"


def test_resolves_ambiguous_leaf_group_by_class():
    leaf_map = {
        "rs_hl 5120": [
            "Blueberry___healthy:::135.0",
            "Soybean___healthy:::269.0",
        ]
    }
    group, status, declared_class = resolve_leaf_group(
        "Soybean___healthy/uuid___RS_HL 5120.JPG", "Soybean___healthy", leaf_map
    )
    assert group == "Soybean___healthy:::269.0"
    assert status == "mapped"
    assert declared_class == "Soybean___healthy"


def test_difference_hash_is_stable_for_same_pixels():
    first = Image.new("RGB", (16, 16), (10, 20, 30))
    second = Image.new("RGB", (32, 32), (10, 20, 30))
    assert difference_hash(first) == difference_hash(second)


def test_validation_reports_clean_split_and_writes_json(tmp_path):
    color_dir, metadata_dir, lock_path, report_path = _write_fixture(tmp_path)

    report = validate_plantvillage_dataset(
        color_dir,
        metadata_dir,
        lock_path,
        report_path,
        near_duplicate_distance=0,
    )

    assert report["summary"]["status"] == "warning"
    assert report["integrity"]["readable_images"] == 2
    assert report["leaf_grouping"]["train_test_group_overlap"] == 0
    assert report["duplicates"]["exact"]["cross_split_groups"] == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_validation_fails_for_exact_duplicate_across_split(tmp_path):
    color_dir, metadata_dir, lock_path, report_path = _write_fixture(tmp_path)
    source = color_dir / "Apple___healthy" / "a___leaf_a.JPG"
    duplicate = color_dir / "Apple___healthy" / "b___leaf_b.JPG"
    duplicate.write_bytes(source.read_bytes())

    report = validate_plantvillage_dataset(
        color_dir,
        metadata_dir,
        lock_path,
        report_path,
        near_duplicate_distance=0,
    )

    assert report["summary"]["status"] == "fail"
    assert report["duplicates"]["exact"]["cross_split_groups"] == 1


def test_validation_fails_for_leaf_group_leakage(tmp_path):
    color_dir, metadata_dir, lock_path, report_path = _write_fixture(tmp_path)
    (metadata_dir / "leaf-map.json").write_text(
        json.dumps(
            {
                "leaf_a": ["Apple___healthy:::1.0"],
                "leaf_b": ["Apple___healthy:::1.0"],
            }
        ),
        encoding="utf-8",
    )

    report = validate_plantvillage_dataset(
        color_dir,
        metadata_dir,
        lock_path,
        report_path,
        near_duplicate_distance=0,
    )

    assert report["summary"]["status"] == "fail"
    assert report["leaf_grouping"]["train_test_group_overlap"] == 1
