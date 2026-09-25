"""Create the review ledger used before freezing PlantVillage split manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REVIEW_FIELDNAMES = (
    "candidate_id",
    "dataset_revision",
    "hash_algorithm",
    "maximum_hamming_distance",
    "hamming_distance",
    "train_path",
    "test_path",
    "decision",
    "review_notes",
)
VALID_REVIEW_DECISIONS = {
    "exclude_train_related",
    "keep_both_false_positive",
}


def _candidate_id(train_path: str, test_path: str) -> str:
    value = f"{train_path}\0{test_path}".encode("utf-8")
    return f"near-{hashlib.sha256(value).hexdigest()[:12]}"


def _review_rows(report: dict[str, Any]) -> list[dict[str, str | int]]:
    try:
        revision = report["dataset"]["source_revision"]
        perceptual = report["duplicates"]["perceptual"]
        algorithm = perceptual["algorithm"]
        maximum_distance = perceptual["maximum_hamming_distance"]
        expected_count = perceptual["cross_split_pairs"]
        candidates = perceptual["cross_split_examples"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Validation report does not contain perceptual candidate data") from exc

    if not isinstance(candidates, list) or len(candidates) != expected_count:
        raise ValueError(
            "Validation report does not contain every cross-split candidate; rerun "
            "validation with --max-examples at least equal to cross_split_pairs"
        )

    rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        try:
            left_path = candidate["left"]
            right_path = candidate["right"]
            left_split = candidate["left_split"]
            right_split = candidate["right_split"]
            distance = candidate["hamming_distance"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed perceptual candidate in validation report") from exc

        if {left_split, right_split} != {"train", "test"}:
            raise ValueError("Review ledger accepts only train/test cross-split candidates")

        train_path = left_path if left_split == "train" else right_path
        test_path = left_path if left_split == "test" else right_path
        pair = (train_path, test_path)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate perceptual candidate: {train_path} / {test_path}")
        seen_pairs.add(pair)

        rows.append(
            {
                "candidate_id": _candidate_id(train_path, test_path),
                "dataset_revision": revision,
                "hash_algorithm": algorithm,
                "maximum_hamming_distance": maximum_distance,
                "hamming_distance": distance,
                "train_path": train_path,
                "test_path": test_path,
                "decision": "pending",
                "review_notes": "",
            }
        )

    return sorted(rows, key=lambda row: (str(row["train_path"]), str(row["test_path"])))


def export_perceptual_review_ledger(
    report_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    """Export every cross-split perceptual candidate to a deterministic CSV ledger."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Review ledger already exists: {output_path}. Use --overwrite only if "
            "discarding existing review decisions is intentional."
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _review_rows(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=REVIEW_FIELDNAMES,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, output_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    return len(rows)


def validate_perceptual_review_ledger(
    report_path: Path,
    ledger_path: Path,
) -> dict[str, int]:
    """Verify that a completed ledger exactly covers the report's candidates."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_rows = _review_rows(report)
    expected_by_id = {str(row["candidate_id"]): row for row in expected_rows}

    with ledger_path.open(encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDNAMES:
            raise ValueError("Review ledger columns do not match the required schema")
        actual_rows = list(reader)

    if len(actual_rows) != len(expected_rows):
        raise ValueError(
            f"Review ledger has {len(actual_rows)} rows; expected {len(expected_rows)}"
        )

    counts = {decision: 0 for decision in sorted(VALID_REVIEW_DECISIONS)}
    seen_ids: set[str] = set()
    immutable_fields = REVIEW_FIELDNAMES[:-2]
    for row in actual_rows:
        candidate_id = row["candidate_id"]
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate review candidate ID: {candidate_id}")
        seen_ids.add(candidate_id)

        expected = expected_by_id.get(candidate_id)
        if expected is None:
            raise ValueError(f"Unknown review candidate ID: {candidate_id}")
        for field in immutable_fields:
            if row[field] != str(expected[field]):
                raise ValueError(f"Candidate {candidate_id} has changed {field}")

        decision = row["decision"]
        if decision not in VALID_REVIEW_DECISIONS:
            raise ValueError(f"Candidate {candidate_id} has invalid decision: {decision}")
        if not row["review_notes"].strip():
            raise ValueError(f"Candidate {candidate_id} requires review notes")
        counts[decision] += 1

    missing_ids = set(expected_by_id) - seen_ids
    if missing_ids:
        raise ValueError(f"Review ledger is missing {len(missing_ids)} candidates")
    return counts
