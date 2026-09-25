import csv
import json

import pytest

from src.data.split_exclusions import (
    EXACT_REASON,
    PERCEPTUAL_REASON,
    build_training_exclusion_ledger,
    validate_training_exclusion_ledger,
)
from src.data.split_review import export_perceptual_review_ledger


def _write_inputs(tmp_path):
    report_path = tmp_path / "report.json"
    review_path = tmp_path / "review.csv"
    report_path.write_text(
        json.dumps(
            {
                "dataset": {"source_revision": "locked-revision"},
                "duplicates": {
                    "exact": {
                        "cross_split_groups": 1,
                        "cross_split_examples": [
                            {
                                "sha256": "abc123",
                                "members": [
                                    {"path": "Class/exact-test.JPG", "split": "test"},
                                    {"path": "Class/exact-train.JPG", "split": "train"},
                                ],
                            }
                        ],
                    },
                    "perceptual": {
                        "algorithm": "dhash-64",
                        "maximum_hamming_distance": 4,
                        "cross_split_pairs": 2,
                        "cross_split_examples": [
                            {
                                "left": "Class/related-train.JPG",
                                "left_split": "train",
                                "right": "Class/related-test.JPG",
                                "right_split": "test",
                                "hamming_distance": 2,
                            },
                            {
                                "left": "Class/distinct-test.JPG",
                                "left_split": "test",
                                "right": "Class/distinct-train.JPG",
                                "right_split": "train",
                                "hamming_distance": 4,
                            },
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    export_perceptual_review_ledger(report_path, review_path)
    with review_path.open(encoding="utf-8", newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    for row in rows:
        is_related = row["train_path"] == "Class/related-train.JPG"
        row["decision"] = (
            "exclude_train_related" if is_related else "keep_both_false_positive"
        )
        row["review_notes"] = "Reviewed."
    with review_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return report_path, review_path


def test_builds_and_validates_training_exclusions(tmp_path):
    report_path, review_path = _write_inputs(tmp_path)
    output_path = tmp_path / "exclusions.csv"

    summary = build_training_exclusion_ledger(report_path, review_path, output_path)

    with output_path.open(encoding="utf-8", newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    assert summary == {
        "records": 2,
        "unique_training_images": 2,
        EXACT_REASON: 1,
        PERCEPTUAL_REASON: 1,
    }
    assert {row["reason"] for row in rows} == {EXACT_REASON, PERCEPTUAL_REASON}
    assert validate_training_exclusion_ledger(
        report_path, review_path, output_path
    ) == summary


def test_refuses_to_overwrite_exclusion_ledger(tmp_path):
    report_path, review_path = _write_inputs(tmp_path)
    output_path = tmp_path / "exclusions.csv"
    output_path.write_text("existing decisions\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="intentionally rebuilding"):
        build_training_exclusion_ledger(report_path, review_path, output_path)


def test_validation_rejects_modified_exclusion_ledger(tmp_path):
    report_path, review_path = _write_inputs(tmp_path)
    output_path = tmp_path / "exclusions.csv"
    build_training_exclusion_ledger(report_path, review_path, output_path)
    content = output_path.read_text(encoding="utf-8")
    output_path.write_text(content.replace("exact-train", "wrong-train"), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        validate_training_exclusion_ledger(report_path, review_path, output_path)
