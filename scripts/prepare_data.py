"""Acquire the locked PlantVillage source dataset.

Run from the repository root:
    python scripts/prepare_data.py lock
    python scripts/prepare_data.py download
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

DEFAULT_LOCK = PROJECT_ROOT / "data" / "plantvillage_source.lock.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "color"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "metadata"
DEFAULT_PROVENANCE = (
    PROJECT_ROOT / "data" / "processed" / "plantvillage" / "provenance.json"
)
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


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "lock":
        command_lock(args)
    elif args.command == "download":
        command_download(args)


if __name__ == "__main__":
    main()
