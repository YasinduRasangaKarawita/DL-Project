import csv
import hashlib
import json
import math
import os
import textwrap
from collections import Counter
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.data.split_exclusions import (
    EXACT_REASON,
    EXCLUSION_FIELDNAMES,
    PERCEPTUAL_REASON,
)
from src.data.split_manifests import MANIFEST_FIELDNAMES, validate_manifest_checksums
from src.data.split_review import REVIEW_FIELDNAMES, VALID_REVIEW_DECISIONS


def summarize_frozen_splits(
    manifest_dir: str | Path = "data/splits",
) -> dict[str, Any]:
    """Load the checksum-verified counts and class order used by EDA."""
    manifest_path = Path(manifest_dir)
    validate_manifest_checksums(manifest_path)
    canonical_checksums = (manifest_path / "checksums.sha256").read_text(
        encoding="utf-8"
    )
    bundle_checksum = hashlib.sha256(canonical_checksums.encode("utf-8")).hexdigest()

    class_mapping = json.loads(
        (manifest_path / "class_mapping.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (manifest_path / "split_metadata.json").read_text(encoding="utf-8")
    )

    classes = class_mapping["classes"]
    expected_mapping = {class_name: index for index, class_name in enumerate(classes)}
    if class_mapping.get("class_to_index") != expected_mapping:
        raise ValueError("Class mapping is not contiguous or does not match the class list")

    split_counts = metadata["split_counts"]
    class_counts = metadata["class_counts"]
    for split, expected_count in split_counts.items():
        observed_count = sum(class_counts[split].values())
        if observed_count != expected_count:
            raise ValueError(f"Class counts do not add up to the {split} split count")

    return {
        "dataset_revision": metadata["dataset_revision"],
        "manifest_bundle_sha256": bundle_checksum,
        "classes": classes,
        "num_classes": len(classes),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "total_images": sum(split_counts.values()),
    }


def analyze_class_distribution(
    summary: dict[str, Any],
    split: str = "train",
) -> dict[str, Any]:
    """Calculate per-class proportions and imbalance statistics for one split."""
    class_counts = summary["class_counts"]
    if split not in class_counts:
        available = ", ".join(sorted(class_counts))
        raise ValueError(f"Unknown split {split!r}; expected one of: {available}")

    classes = summary["classes"]
    counts = class_counts[split]
    if set(counts) != set(classes):
        raise ValueError(f"Class counts for {split} do not match the frozen class mapping")
    if any(counts[class_name] <= 0 for class_name in classes):
        raise ValueError(f"Every class must have at least one image in {split}")

    total_images = sum(counts.values())
    distribution = [
        {
            "class_name": class_name,
            "count": counts[class_name],
            "percentage": counts[class_name] / total_images * 100,
        }
        for class_name in classes
    ]
    smallest = min(distribution, key=lambda row: (row["count"], row["class_name"]))
    largest = max(distribution, key=lambda row: (row["count"], row["class_name"]))
    count_values = [row["count"] for row in distribution]

    return {
        "split": split,
        "total_images": total_images,
        "distribution": distribution,
        "smallest_class": smallest,
        "largest_class": largest,
        "imbalance_ratio": largest["count"] / smallest["count"],
        "mean_images_per_class": mean(count_values),
        "median_images_per_class": median(count_values),
    }


def plot_class_distribution(
    analysis: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a readable horizontal class-distribution chart for one split."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        analysis["distribution"],
        key=lambda row: (row["count"], row["class_name"]),
    )
    labels = [
        row["class_name"].replace("___", " — ").replace("_", " ")
        for row in ordered
    ]
    counts = [row["count"] for row in ordered]
    colors = sns.color_palette("mako", n_colors=len(ordered))

    fig, ax = plt.subplots(figsize=(12, 14))
    try:
        bars = ax.barh(labels, counts, color=colors)
        ax.bar_label(bars, padding=3, fontsize=8)
        ax.axvline(
            analysis["mean_images_per_class"],
            color="#c44e52",
            linestyle="--",
            linewidth=1.5,
            label=f'Mean: {analysis["mean_images_per_class"]:.1f}',
        )
        ax.set_title(
            f'PlantVillage {analysis["split"].title()} Class Distribution\n'
            f'Imbalance ratio: {analysis["imbalance_ratio"]:.2f}:1',
            fontsize=14,
            weight="bold",
            pad=12,
        )
        ax.set_xlabel("Number of images")
        ax.set_ylabel("Class")
        ax.legend(loc="lower right")
        ax.margins(x=0.12)
        fig.tight_layout()
        fig.savefig(output, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)

    return output


def analyze_image_dimensions(
    raw_dir: str | Path,
    manifest_dir: str | Path = "data/splits",
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """Inspect image headers referenced by the frozen manifests and summarize dimensions."""
    raw_path = Path(raw_dir).resolve()
    manifest_path = Path(manifest_dir)
    validate_manifest_checksums(manifest_path)

    allowed_splits = {"train", "validation", "test"}
    if not splits or set(splits) - allowed_splits:
        expected = ", ".join(sorted(allowed_splits))
        raise ValueError(f"Splits must contain one or more of: {expected}")

    records: list[dict[str, Any]] = []
    dimension_frequencies: Counter[tuple[int, int]] = Counter()
    orientation_counts: Counter[str] = Counter()

    for split in splits:
        with (manifest_path / f"{split}.csv").open(
            encoding="utf-8", newline=""
        ) as file_handle:
            reader = csv.DictReader(file_handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDNAMES:
                raise ValueError(f"Manifest columns do not match the schema: {split}.csv")

            for row in reader:
                relative = PurePosixPath(row["relative_path"])
                if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
                    raise ValueError(f"Unsafe or malformed manifest path: {relative}")

                image_path = raw_path / Path(*relative.parts)
                if not image_path.is_file():
                    raise FileNotFoundError(f"Manifest image does not exist: {image_path}")

                with Image.open(image_path) as image:
                    width, height = image.size
                aspect_ratio = width / height
                orientation = "square" if width == height else "landscape" if width > height else "portrait"

                records.append(
                    {
                        "split": split,
                        "class_name": row["class_name"],
                        "relative_path": relative.as_posix(),
                        "width": width,
                        "height": height,
                        "aspect_ratio": aspect_ratio,
                        "orientation": orientation,
                    }
                )
                dimension_frequencies[(width, height)] += 1
                orientation_counts[orientation] += 1

    if not records:
        raise ValueError("No images were found in the requested manifest splits")

    widths = [record["width"] for record in records]
    heights = [record["height"] for record in records]
    aspect_ratios = [record["aspect_ratio"] for record in records]

    return {
        "splits": list(splits),
        "image_count": len(records),
        "records": records,
        "width": {
            "minimum": min(widths),
            "maximum": max(widths),
            "mean": mean(widths),
            "median": median(widths),
        },
        "height": {
            "minimum": min(heights),
            "maximum": max(heights),
            "mean": mean(heights),
            "median": median(heights),
        },
        "aspect_ratio": {
            "minimum": min(aspect_ratios),
            "maximum": max(aspect_ratios),
            "mean": mean(aspect_ratios),
            "median": median(aspect_ratios),
        },
        "orientation_counts": {
            orientation: orientation_counts.get(orientation, 0)
            for orientation in ("landscape", "portrait", "square")
        },
        "dimension_frequencies": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in dimension_frequencies.most_common()
        ],
    }


def select_sample_images(
    raw_dir: str | Path,
    manifest_dir: str | Path = "data/splits",
    split: str = "train",
    seed: int = 42,
) -> dict[str, Any]:
    """Select one deterministic, non-cherry-picked manifest image per class."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("Sample split must be train, validation, or test")

    raw_path = Path(raw_dir).resolve()
    manifest_path = Path(manifest_dir)
    validate_manifest_checksums(manifest_path)
    class_mapping = json.loads(
        (manifest_path / "class_mapping.json").read_text(encoding="utf-8")
    )
    classes = class_mapping["classes"]
    paths_by_class: dict[str, list[str]] = {class_name: [] for class_name in classes}

    with (manifest_path / f"{split}.csv").open(
        encoding="utf-8", newline=""
    ) as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDNAMES:
            raise ValueError(f"Manifest columns do not match the schema: {split}.csv")
        for row in reader:
            class_name = row["class_name"]
            if class_name not in paths_by_class:
                raise ValueError(f"Unknown class in {split} manifest: {class_name}")
            paths_by_class[class_name].append(row["relative_path"])

    samples = []
    for class_name in classes:
        candidates = paths_by_class[class_name]
        if not candidates:
            raise ValueError(f"No {split} images are available for class {class_name}")
        relative_path = min(
            candidates,
            key=lambda value: hashlib.sha256(
                f"{seed}\0{class_name}\0{value}".encode("utf-8")
            ).hexdigest(),
        )
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise ValueError(f"Unsafe or malformed manifest path: {relative}")
        image_path = (raw_path / Path(*relative.parts)).resolve()
        if not image_path.is_relative_to(raw_path):
            raise ValueError(f"Manifest image escapes the dataset directory: {relative}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
        samples.append(
            {
                "class_name": class_name,
                "relative_path": relative.as_posix(),
                "image_path": image_path,
            }
        )

    return {"split": split, "seed": seed, "samples": samples}


def plot_sample_grid(
    selection: dict[str, Any],
    output_path: str | Path,
    columns: int = 5,
) -> Path:
    """Save a grid containing one selected image for every class."""
    if columns < 1:
        raise ValueError("Grid columns must be positive")
    samples = selection["samples"]
    if not samples:
        raise ValueError("At least one sample is required to create a grid")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(samples) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.5, rows * 2.5),
        squeeze=False,
    )
    try:
        for axis, sample in zip(axes.flat, samples):
            with Image.open(sample["image_path"]) as image:
                axis.imshow(image.convert("RGB"))
            label = sample["class_name"].replace("___", " — ").replace("_", " ")
            axis.set_title(textwrap.fill(label, width=26), fontsize=8)
            axis.axis("off")

        for axis in list(axes.flat)[len(samples):]:
            axis.axis("off")

        fig.suptitle(
            f'Deterministic {selection["split"].title()} Sample per Class '
            f'(seed {selection["seed"]})',
            fontsize=14,
            weight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(output, dpi=120, bbox_inches="tight")
    finally:
        plt.close(fig)

    return output


def summarize_data_quality(
    validation_report_path: str | Path = (
        "data/processed/plantvillage/validation_report.json"
    ),
    source_lock_path: str | Path = "data/plantvillage_source.lock.json",
    manifest_dir: str | Path = "data/splits",
) -> dict[str, Any]:
    """Cross-check validation evidence and return concise data-quality findings."""
    report = json.loads(Path(validation_report_path).read_text(encoding="utf-8"))
    source_lock = json.loads(Path(source_lock_path).read_text(encoding="utf-8"))
    manifest_path = Path(manifest_dir)
    frozen_summary = summarize_frozen_splits(manifest_path)
    split_metadata = json.loads(
        (manifest_path / "split_metadata.json").read_text(encoding="utf-8")
    )

    revision = source_lock["dataset"]["resolved_revision"]
    observed_revisions = {
        report["dataset"]["source_revision"],
        frozen_summary["dataset_revision"],
    }
    if observed_revisions != {revision}:
        raise ValueError("Data-quality evidence does not use one source revision")
    if report["integrity"]["images_scanned"] != source_lock["expected"]["images"]:
        raise ValueError("Validation image count does not match the source lock")

    review_path = manifest_path / "perceptual_duplicate_review.csv"
    with review_path.open(encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDNAMES:
            raise ValueError("Perceptual-review ledger columns do not match the schema")
        review_rows = list(reader)

    exclusion_path = manifest_path / "training_exclusions.csv"
    with exclusion_path.open(encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != EXCLUSION_FIELDNAMES:
            raise ValueError("Training-exclusion ledger columns do not match the schema")
        exclusion_rows = list(reader)

    ledger_revisions = {
        row["dataset_revision"] for row in review_rows + exclusion_rows
    }
    if ledger_revisions != {revision}:
        raise ValueError("Review or exclusion evidence uses a different source revision")

    review_counts = Counter(row["decision"] for row in review_rows)
    if set(review_counts) - VALID_REVIEW_DECISIONS:
        raise ValueError("Perceptual-review ledger contains an invalid decision")
    perceptual = report["duplicates"]["perceptual"]
    if len(review_rows) != perceptual["cross_split_pairs"]:
        raise ValueError("Review ledger does not cover every cross-split candidate")

    exclusion_counts = Counter(row["reason"] for row in exclusion_rows)
    if set(exclusion_counts) - {EXACT_REASON, PERCEPTUAL_REASON}:
        raise ValueError("Training-exclusion ledger contains an invalid reason")
    if exclusion_counts[EXACT_REASON] != report["duplicates"]["exact"]["cross_split_groups"]:
        raise ValueError("Exact-duplicate exclusions do not match the validation report")
    if exclusion_counts[PERCEPTUAL_REASON] != review_counts["exclude_train_related"]:
        raise ValueError("Perceptual exclusions do not match the completed reviews")

    excluded_training_images = {row["train_path"] for row in exclusion_rows}
    expected_exclusions = split_metadata["source_counts"][
        "excluded_unique_training_images"
    ]
    if len(excluded_training_images) != expected_exclusions:
        raise ValueError("Unique exclusions do not match the frozen split metadata")

    integrity = report["integrity"]
    exact = report["duplicates"]["exact"]
    leaf_grouping = report["leaf_grouping"]
    official_splits = report["official_splits"]

    return {
        "dataset_revision": revision,
        "source_audit_status": report["summary"]["status"],
        "integrity": {
            "images_scanned": integrity["images_scanned"],
            "readable_images": integrity["readable_images"],
            "empty_files": integrity["empty_file_count"],
            "corrupted_files": integrity["corrupted_file_count"],
            "image_modes": integrity["image_modes"],
            "non_rgb_images": integrity["non_rgb_image_count"],
            "dimensions": integrity["dimensions"],
        },
        "duplicates": {
            "exact_groups": exact["groups"],
            "exact_images": exact["images"],
            "exact_cross_split_groups": exact["cross_split_groups"],
            "perceptual_candidate_pairs": perceptual["pairs"],
            "perceptual_cross_split_pairs": perceptual["cross_split_pairs"],
            "review_decisions": dict(sorted(review_counts.items())),
            "exclusion_records": len(exclusion_rows),
            "unique_training_images_excluded": len(excluded_training_images),
            "exclusions_by_reason": dict(sorted(exclusion_counts.items())),
        },
        "leaf_grouping": {
            "mapped_images": leaf_grouping["mapped_images"],
            "unmapped_images": leaf_grouping["unmapped_images"],
            "ambiguous_images": leaf_grouping["ambiguous_images"],
            "coverage_fraction": leaf_grouping["metadata_coverage_fraction"],
            "resolved_groups": leaf_grouping["resolved_groups"],
            "official_train_test_overlap": leaf_grouping["train_test_group_overlap"],
        },
        "official_splits": {
            "train_images": official_splits["train_images"],
            "test_images": official_splits["test_images"],
            "path_overlap": official_splits["path_overlap"],
            "coverage_gaps": (
                official_splits["missing_local_images"]
                + official_splits["unlisted_local_images"]
            ),
        },
        "frozen_splits": {
            "counts": frozen_summary["split_counts"],
            "classes": frozen_summary["num_classes"],
            "bundle_sha256": frozen_summary["manifest_bundle_sha256"],
        },
    }


def generate_dataset_figures(
    raw_dir: str = "data/raw",
    figures_dir: str = "figures/dataset"
) -> None:
    """
    Generate Exploratory Data Analysis (EDA) charts:
    - Class distribution bar plot
    - Grid of representative sample leaf images across classes
    """
    os.makedirs(figures_dir, exist_ok=True)
    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])

    counts = {}
    sample_images = {}
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        counts[cls] = len(imgs)
        if imgs:
            sample_images[cls] = os.path.join(cls_dir, imgs[0])

    # 1. Class Distribution Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [c.replace("___", "\n") for c in counts.keys()]
    values = list(counts.values())
    sns.barplot(x=names, y=values, palette="mako", ax=ax)
    ax.set_title("PlantVillage Dataset — Class Sample Distribution", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Image Count", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "class_distribution.png"), dpi=200)
    plt.close(fig)

    # 2. Sample Images Grid (up to 12 classes)
    display_classes = classes[:12]
    rows = (len(display_classes) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, cls in enumerate(display_classes):
        img_path = sample_images.get(cls)
        if img_path and os.path.exists(img_path):
            with Image.open(img_path) as img:
                axes[idx].imshow(img)
        display_label = cls.replace("___", "\n")
        axes[idx].set_title(display_label, fontsize=10, weight="semibold")
        axes[idx].axis("off")

    for idx in range(len(display_classes), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Representative Leaf Images per Disease Category", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sample_images.png"), dpi=200)
    plt.close(fig)
