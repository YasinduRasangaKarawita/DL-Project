import csv
import hashlib
import json

import pytest
from PIL import Image

from src.data.split_manifests import MANIFEST_FIELDNAMES
from src.evaluation.dataset_analysis import (
    analyze_class_distribution,
    analyze_image_dimensions,
    plot_class_distribution,
    plot_sample_grid,
    select_sample_images,
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
    metadata_file.write_text("{}", encoding="utf-8")
    generated[metadata_file.name] = metadata_file

    (manifest_dir / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {filename}\n"
            for filename, path in sorted(generated.items())
        ),
        encoding="utf-8",
    )
    return raw_dir, manifest_dir


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
    assert len(summary["manifest_bundle_sha256"]) == 64


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
