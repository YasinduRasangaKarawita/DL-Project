import pytest
from PIL import Image

from src.evaluation.dataset_analysis import (
    analyze_class_distribution,
    plot_class_distribution,
    summarize_frozen_splits,
)


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
