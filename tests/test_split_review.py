import csv
import json

import pytest

from src.data.split_review import export_perceptual_review_ledger


def _write_report(path, *, expected_count=2, candidates=None):
    if candidates is None:
        candidates = [
            {
                "left": "Class/z-train.JPG",
                "left_split": "train",
                "right": "Class/a-test.JPG",
                "right_split": "test",
                "hamming_distance": 4,
            },
            {
                "left": "Class/b-test.JPG",
                "left_split": "test",
                "right": "Class/a-train.JPG",
                "right_split": "train",
                "hamming_distance": 2,
            },
        ]
    path.write_text(
        json.dumps(
            {
                "dataset": {"source_revision": "locked-revision"},
                "duplicates": {
                    "perceptual": {
                        "algorithm": "dhash-64",
                        "maximum_hamming_distance": 4,
                        "cross_split_pairs": expected_count,
                        "cross_split_examples": candidates,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_exports_complete_deterministic_review_ledger(tmp_path):
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "review.csv"
    _write_report(report_path)

    count = export_perceptual_review_ledger(report_path, output_path)

    with output_path.open(encoding="utf-8", newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    assert count == 2
    assert [row["train_path"] for row in rows] == [
        "Class/a-train.JPG",
        "Class/z-train.JPG",
    ]
    assert {row["decision"] for row in rows} == {"pending"}
    assert len({row["candidate_id"] for row in rows}) == 2
    assert {row["dataset_revision"] for row in rows} == {"locked-revision"}


def test_refuses_truncated_validation_report(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, expected_count=3)

    with pytest.raises(ValueError, match="does not contain every"):
        export_perceptual_review_ledger(report_path, tmp_path / "review.csv")


def test_refuses_to_overwrite_existing_review_decisions(tmp_path):
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "review.csv"
    _write_report(report_path)
    output_path.write_text("reviewed data\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="discarding existing review decisions"):
        export_perceptual_review_ledger(report_path, output_path)
