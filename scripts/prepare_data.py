"""Acquire and validate the locked PlantVillage source dataset.

Run from the repository root:
    python scripts/prepare_data.py lock
    python scripts/prepare_data.py download
    python scripts/prepare_data.py validate
    python scripts/prepare_data.py review-candidates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.plantvillage_acquisition import (  # noqa: E402
    build_local_provenance,
    copy_source_metadata,
    download_locked_files,
    extract_color_dataset,
    load_source_lock,
    resolve_source_lock,
    write_source_lock,
)
from src.data.plantvillage_validation import validate_plantvillage_dataset  # noqa: E402
from src.data.split_review import export_perceptual_review_ledger  # noqa: E402

DEFAULT_LOCK = PROJECT_ROOT / "data" / "plantvillage_source.lock.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "color"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "metadata"
DEFAULT_PROVENANCE = (
    PROJECT_ROOT / "data" / "processed" / "plantvillage" / "provenance.json"
)
DEFAULT_VALIDATION_REPORT = (
    PROJECT_ROOT / "data" / "processed" / "plantvillage" / "validation_report.json"
)
DEFAULT_REVIEW_LEDGER = PROJECT_ROOT / "data" / "splits" / "perceptual_duplicate_review.csv"
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "huggingface"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock_parser = subparsers.add_parser(
        "lock", help="Resolve the dataset source to an immutable Hugging Face commit"
    )
    lock_parser.add_argument("--revision", default="main")
    lock_parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    lock_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)

    download_parser = subparsers.add_parser(
        "download", help="Download, verify, and extract the locked color dataset"
    )
    download_parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    download_parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    download_parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    download_parser.add_argument("--provenance-file", type=Path, default=DEFAULT_PROVENANCE)
    download_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)

    validate_parser = subparsers.add_parser(
        "validate", help="Audit image integrity, duplicates, leaf groups, and split leakage"
    )
    validate_parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    validate_parser.add_argument("--color-dir", type=Path, default=DEFAULT_DESTINATION)
    validate_parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    validate_parser.add_argument("--report-file", type=Path, default=DEFAULT_VALIDATION_REPORT)
    validate_parser.add_argument("--near-duplicate-distance", type=int, default=4)
    validate_parser.add_argument("--max-examples", type=int, default=100)

    review_parser = subparsers.add_parser(
        "review-candidates",
        help="Export cross-split perceptual matches to a review ledger",
    )
    review_parser.add_argument("--report-file", type=Path, default=DEFAULT_VALIDATION_REPORT)
    review_parser.add_argument("--output-file", type=Path, default=DEFAULT_REVIEW_LEDGER)
    review_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing ledger and discard any recorded decisions",
    )
    return parser


def command_lock(args: argparse.Namespace) -> None:
    lock = resolve_source_lock(revision=args.revision, cache_dir=args.cache_dir)
    write_source_lock(lock, args.lock_file)
    print(f"Locked PlantVillage revision: {lock['dataset']['resolved_revision']}")
    print(f"Wrote source lock: {args.lock_file}")


def command_download(args: argparse.Namespace) -> None:
    lock = load_source_lock(args.lock_file)
    downloaded = download_locked_files(lock, cache_dir=args.cache_dir)
    summary = extract_color_dataset(
        archive_path=downloaded["data.zip"],
        destination=args.destination,
        expected_images=lock["expected"]["images"],
        expected_classes=lock["expected"]["classes"],
    )
    copy_source_metadata(downloaded, args.metadata_dir)
    provenance = build_local_provenance(lock, downloaded, summary, args.destination)
    args.provenance_file.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_file.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PlantVillage color dataset ready: {args.destination}")
    print(f"Images: {summary['image_count']}; classes: {summary['class_count']}")
    print(f"Local provenance: {args.provenance_file}")


def command_validate(args: argparse.Namespace) -> None:
    report = validate_plantvillage_dataset(
        color_dir=args.color_dir,
        metadata_dir=args.metadata_dir,
        lock_path=args.lock_file,
        report_path=args.report_file,
        near_duplicate_distance=args.near_duplicate_distance,
        max_examples=args.max_examples,
    )
    summary = report["summary"]
    print(f"Validation status: {summary['status'].upper()}")
    print(
        f"Images: {report['integrity']['readable_images']}; "
        f"classes: {report['classes']['actual_count']}"
    )
    print(
        "Official train/test leakage: "
        f"{report['leaf_grouping']['train_test_group_overlap']} leaf groups, "
        f"{report['duplicates']['exact']['cross_split_groups']} exact duplicate groups"
    )
    print(f"Validation report: {args.report_file}")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    for failure in summary["hard_failures"]:
        print(f"ERROR: {failure}", file=sys.stderr)
    if summary["status"] == "fail":
        raise SystemExit(1)


def command_review_candidates(args: argparse.Namespace) -> None:
    count = export_perceptual_review_ledger(
        report_path=args.report_file,
        output_path=args.output_file,
        overwrite=args.overwrite,
    )
    print(f"Review candidates: {count}")
    print(f"Review ledger: {args.output_file}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "lock":
        command_lock(args)
    elif args.command == "download":
        command_download(args)
    elif args.command == "validate":
        command_validate(args)
    elif args.command == "review-candidates":
        command_review_candidates(args)


if __name__ == "__main__":
    main()
