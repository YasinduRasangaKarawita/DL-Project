import csv
import hashlib
import json

import pytest
from PIL import Image

from src.data.split_exclusions import (
    EXACT_REASON,
    EXCLUSION_FIELDNAMES,
    PERCEPTUAL_REASON,
)
from src.data.split_manifests import MANIFEST_FIELDNAMES
from src.data.split_review import REVIEW_FIELDNAMES
from src.evaluation.dataset_analysis import (
    analyze_class_distribution,
    analyze_image_dimensions,
    plot_class_distribution,
    plot_sample_grid,
    select_sample_images,
    summarize_data_quality,
    summarize_frozen_splits,
)


def _write_dimension_fixture(tmp_path):
    raw_dir = tmp_path / "color"
    manifest_dir = tmp_path / "splits"
    manifest_dir.mkdir()
    class_name = "Plant___healthy"
    generated = {}

    for split, size in {
        "train": (20, 10),
        "validation": (10, 20),
        "test": (12, 12),
    }.items():
        relative_path = f"{class_name}/{split}.JPG"
        image_path = raw_dir / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color=(40, 120, 40)).save(image_path)

        manifest_file = manifest_dir / f"{split}.csv"
        with manifest_file.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=MANIFEST_FIELDNAMES,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "class_index": 0,
                    "group_id": f"group-{split}",
                    "source_partition": (
                        "official_test" if split == "test" else "official_train"
                    ),
                }
            )
        generated[manifest_file.name] = manifest_file

    class_mapping_file = manifest_dir / "class_mapping.json"
    class_mapping_file.write_text(
        json.dumps({"classes": [class_name], "class_to_index": {class_name: 0}}),
        encoding="utf-8",
    )
    generated[class_mapping_file.name] = class_mapping_file

    metadata_file = manifest_dir / "split_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "dataset_revision": "fixture-revision",
                "split_counts": {"train": 1, "validation": 1, "test": 1},
                "class_counts": {
                    split: {class_name: 1}
                    for split in ("train", "validation", "test")
                },
                "source_counts": {"excluded_unique_training_images": 2},
            }
        ),
        encoding="utf-8",
    )
    generated[metadata_file.name] = metadata_file

    (manifest_dir / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {filename}\n"
            for filename, path in sorted(generated.items())
        ),
        encoding="utf-8",
    )
    return raw_dir, manifest_dir


def _write_quality_fixture(tmp_path):
    _, manifest_dir = _write_dimension_fixture(tmp_path)
    source_lock_path = tmp_path / "source-lock.json"
    source_lock_path.write_text(
        json.dumps(
            {
                "dataset": {"resolved_revision": "fixture-revision"},
                "expected": {"images": 3},
            }
        ),
        encoding="utf-8",
    )

    validation_report_path = tmp_path / "validation-report.json"
    validation_report_path.write_text(
        json.dumps(
            {
                "dataset": {"source_revision": "fixture-revision"},
                "summary": {"status": "fail"},
                "integrity": {
                    "images_scanned": 3,
                    "readable_images": 3,
                    "empty_file_count": 0,
                    "corrupted_file_count": 0,
                    "image_modes": {"RGB": 3},
                    "non_rgb_image_count": 0,
                    "dimensions": {"10x10": 3},
                },
                "duplicates": {
                    "exact": {"groups": 1, "images": 2, "cross_split_groups": 1},
                    "perceptual": {
                        "pairs": 4,
                        "cross_split_pairs": 2,
                    },
                },
                "leaf_grouping": {
                    "mapped_images": 3,
                    "unmapped_images": 0,
                    "ambiguous_images": 0,
                    "metadata_coverage_fraction": 1.0,
                    "resolved_groups": 3,
                    "train_test_group_overlap": 0,
                },
                "official_splits": {
                    "train_images": 2,
                    "test_images": 1,
                    "path_overlap": 0,
                    "missing_local_images": 0,
                    "unlisted_local_images": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    review_rows = [
        {
            "candidate_id": "candidate-1",
            "dataset_revision": "fixture-revision",
            "hash_algorithm": "dhash-64",
            "maximum_hamming_distance": 4,
            "hamming_distance": 2,
            "train_path": "Plant___healthy/train-1.JPG",
            "test_path": "Plant___healthy/test.JPG",
            "decision": "exclude_train_related",
            "review_notes": "Related fixture pair.",
        },
        {
            "candidate_id": "candidate-2",
            "dataset_revision": "fixture-revision",
            "hash_algorithm": "dhash-64",
            "maximum_hamming_distance": 4,
            "hamming_distance": 4,
            "train_path": "Plant___healthy/train-2.JPG",
            "test_path": "Plant___healthy/test.JPG",
            "decision": "keep_both_false_positive",
            "review_notes": "Unrelated fixture pair.",
        },
    ]
    with (manifest_dir / "perceptual_duplicate_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(review_rows)

    exclusion_rows = [
        {
            "exclusion_id": "exact-1",
            "dataset_revision": "fixture-revision",
            "reason": EXACT_REASON,
            "train_path": "Plant___healthy/train-1.JPG",
            "related_test_path": "Plant___healthy/test.JPG",
            "evidence": "exact-hash",
        },
        {
            "exclusion_id": "near-1",
            "dataset_revision": "fixture-revision",
            "reason": PERCEPTUAL_REASON,
            "train_path": "Plant___healthy/train-2.JPG",
            "related_test_path": "Plant___healthy/test.JPG",
            "evidence": "candidate-1",
        },
    ]
    with (manifest_dir / "training_exclusions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=EXCLUSION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(exclusion_rows)

    return validation_report_path, source_lock_path, manifest_dir


def test_summarizes_frozen_plantvillage_splits():
    summary = summarize_frozen_splits()

    assert summary["num_classes"] == 38
    assert summary["split_counts"] == {
        "train": 37037,
        "validation": 6540,
        "test": 10709,
    }
    assert summary["total_images"] == 54286
    assert set(summary["class_counts"]) == {"train", "validation", "test"}
    assert summary["manifest_bundle_sha256"] == (
        "19ca82ed9a1832dbeca228bbcb35274cd3362758e918818e64587f4d0bd1306c"
    )


def test_analyzes_training_class_distribution():
    summary = summarize_frozen_splits()
    analysis = analyze_class_distribution(summary, split="train")

    assert analysis["split"] == "train"
    assert analysis["total_images"] == 37037
    assert len(analysis["distribution"]) == 38
    assert sum(row["percentage"] for row in analysis["distribution"]) == pytest.approx(100)
    assert analysis["smallest_class"]["class_name"] == "Potato___healthy"
    assert analysis["smallest_class"]["count"] == 104
    assert analysis["largest_class"]["class_name"] == "Orange___Haunglongbing_(Citrus_greening)"
    assert analysis["largest_class"]["count"] == 3851
    assert analysis["imbalance_ratio"] == pytest.approx(3851 / 104)


def test_rejects_unknown_split_for_class_distribution():
    summary = summarize_frozen_splits()

    with pytest.raises(ValueError, match="Unknown split"):
        analyze_class_distribution(summary, split="development")


def test_plots_training_class_distribution(tmp_path):
    summary = summarize_frozen_splits()
    analysis = analyze_class_distribution(summary, split="train")
    output_path = tmp_path / "training_class_distribution.png"

    result = plot_class_distribution(analysis, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.width > image.height / 2


def test_analyzes_manifest_image_dimensions(tmp_path):
    raw_dir, manifest_dir = _write_dimension_fixture(tmp_path)

    analysis = analyze_image_dimensions(raw_dir, manifest_dir)

    assert analysis["image_count"] == 3
    assert analysis["width"] == {
        "minimum": 10,
        "maximum": 20,
        "mean": 14,
        "median": 12,
    }
    assert analysis["height"] == {
        "minimum": 10,
        "maximum": 20,
        "mean": 14,
        "median": 12,
    }
    assert analysis["orientation_counts"] == {
        "landscape": 1,
        "portrait": 1,
        "square": 1,
    }
    assert len(analysis["records"]) == 3


def test_selects_and_plots_manifest_sample_grid(tmp_path):
    raw_dir, manifest_dir = _write_dimension_fixture(tmp_path)
    selection = select_sample_images(raw_dir, manifest_dir, split="train", seed=42)

    assert selection["split"] == "train"
    assert selection["seed"] == 42
    assert len(selection["samples"]) == 1
    assert selection["samples"][0]["relative_path"] == "Plant___healthy/train.JPG"

    output_path = tmp_path / "sample_grid.png"
    result = plot_sample_grid(selection, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as image:
        assert image.format == "PNG"


def test_summarizes_cross_checked_data_quality_evidence(tmp_path):
    report_path, source_lock_path, manifest_dir = _write_quality_fixture(tmp_path)

    summary = summarize_data_quality(report_path, source_lock_path, manifest_dir)

    assert summary["dataset_revision"] == "fixture-revision"
    assert summary["integrity"]["readable_images"] == 3
    assert summary["duplicates"]["review_decisions"] == {
        "exclude_train_related": 1,
        "keep_both_false_positive": 1,
    }
    assert summary["duplicates"]["unique_training_images_excluded"] == 2
    assert summary["leaf_grouping"]["official_train_test_overlap"] == 0
    assert summary["frozen_splits"]["counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
