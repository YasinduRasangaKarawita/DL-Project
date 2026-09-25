"""Build the reviewed training-exclusion ledger used by split generation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .split_review import validate_perceptual_review_ledger

EXCLUSION_FIELDNAMES = (
    "exclusion_id",
    "dataset_revision",
    "reason",
    "train_path",
    "related_test_path",
    "evidence",
)
EXACT_REASON = "exact_duplicate_cross_split"
PERCEPTUAL_REASON = "perceptual_related_cross_split"


def _exclusion_id(reason: str, train_path: str, test_path: str, evidence: str) -> str:
    value = f"{reason}\0{train_path}\0{test_path}\0{evidence}".encode("utf-8")
    return f"exclude-{hashlib.sha256(value).hexdigest()[:12]}"


def _expected_exclusion_rows(
    report_path: Path,
    review_path: Path,
) -> list[dict[str, str]]:
    validate_perceptual_review_ledger(report_path, review_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    revision = str(report["dataset"]["source_revision"])

    exact = report["duplicates"]["exact"]
    exact_groups = exact["cross_split_examples"]
    if len(exact_groups) != exact["cross_split_groups"]:
        raise ValueError(
            "Validation report does not contain every cross-split exact-duplicate group"
        )

    rows: list[dict[str, str]] = []
    for group in exact_groups:
        digest = str(group["sha256"])
        train_paths = sorted(
            str(member["path"])
            for member in group["members"]
            if member["split"] == "train"
        )
        test_paths = sorted(
            str(member["path"])
            for member in group["members"]
            if member["split"] == "test"
        )
        if not train_paths or not test_paths:
            raise ValueError("Malformed cross-split exact-duplicate group")
        for train_path in train_paths:
            for test_path in test_paths:
                rows.append(
                    {
                        "exclusion_id": _exclusion_id(
                            EXACT_REASON, train_path, test_path, digest
                        ),
                        "dataset_revision": revision,
                        "reason": EXACT_REASON,
                        "train_path": train_path,
                        "related_test_path": test_path,
                        "evidence": digest,
                    }
                )

    with review_path.open(encoding="utf-8", newline="") as file_handle:
        review_rows = list(csv.DictReader(file_handle))
    for review in review_rows:
        if review["decision"] != "exclude_train_related":
            continue
        train_path = review["train_path"]
        test_path = review["test_path"]
        evidence = review["candidate_id"]
        rows.append(
            {
                "exclusion_id": _exclusion_id(
                    PERCEPTUAL_REASON, train_path, test_path, evidence
                ),
                "dataset_revision": revision,
                "reason": PERCEPTUAL_REASON,
                "train_path": train_path,
                "related_test_path": test_path,
                "evidence": evidence,
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row["train_path"],
            row["reason"],
            row["related_test_path"],
        ),
    )


def _write_csv_atomic(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=EXCLUSION_FIELDNAMES,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, output_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_training_exclusion_ledger(
    report_path: Path,
    review_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Combine exact duplicates and approved visual-review exclusions."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Training-exclusion ledger already exists: {output_path}. Use --overwrite "
            "only when intentionally rebuilding it from reviewed inputs."
        )
    rows = _expected_exclusion_rows(report_path, review_path)
    _write_csv_atomic(rows, output_path)
    return _summarize(rows)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "records": len(rows),
        "unique_training_images": len({str(row["train_path"]) for row in rows}),
        EXACT_REASON: sum(row["reason"] == EXACT_REASON for row in rows),
        PERCEPTUAL_REASON: sum(row["reason"] == PERCEPTUAL_REASON for row in rows),
    }


def validate_training_exclusion_ledger(
    report_path: Path,
    review_path: Path,
    ledger_path: Path,
) -> dict[str, int]:
    """Verify that the exclusion ledger is the exact derivation of its inputs."""
    expected_rows = _expected_exclusion_rows(report_path, review_path)
    with ledger_path.open(encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != EXCLUSION_FIELDNAMES:
            raise ValueError("Training-exclusion ledger columns do not match the schema")
        actual_rows = list(reader)

    if actual_rows != expected_rows:
        raise ValueError("Training-exclusion ledger does not match its reviewed inputs")
    return _summarize(actual_rows)
