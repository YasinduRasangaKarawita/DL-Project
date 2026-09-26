from src.evaluation.dataset_analysis import summarize_frozen_splits


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
