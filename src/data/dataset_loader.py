"""PyTorch datasets and loaders backed by the frozen split manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.logger import setup_logger
from .preprocessing import get_transforms
from .split_manifests import MANIFEST_FIELDNAMES, validate_manifest_checksums

logger = setup_logger("dataset_loader")


class PlantDiseaseDataset(Dataset):
    """Load RGB PlantVillage images from one frozen manifest partition."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: Any = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]
        label = self.labels[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def _load_class_mapping(manifest_dir: Path) -> tuple[list[str], dict[str, int]]:
    value = json.loads((manifest_dir / "class_mapping.json").read_text(encoding="utf-8"))
    classes = value.get("classes")
    class_to_index = value.get("class_to_index")
    if not isinstance(classes, list) or classes != sorted(classes):
        raise ValueError("Frozen class list must be sorted")
    expected_mapping = {class_name: index for index, class_name in enumerate(classes)}
    if class_to_index != expected_mapping:
        raise ValueError("Frozen class mapping is not contiguous or does not match its list")
    return classes, class_to_index


def _load_manifest_partition(
    dataset_dir: Path,
    manifest_path: Path,
    class_to_index: dict[str, int],
    expected_source_partition: str,
) -> tuple[list[str], list[int], set[str]]:
    image_paths: list[str] = []
    labels: list[int] = []
    group_ids: set[str] = set()
    seen_paths: set[str] = set()
    dataset_root = dataset_dir.resolve()

    with manifest_path.open(encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDNAMES:
            raise ValueError(f"Manifest columns do not match the schema: {manifest_path.name}")
        for row in reader:
            relative_path = row["relative_path"]
            relative = PurePosixPath(relative_path)
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
                raise ValueError(f"Unsafe or malformed manifest path: {relative_path}")
            class_name = relative.parts[0]
            if row["class_name"] != class_name:
                raise ValueError(f"Class/path mismatch: {relative_path}")
            expected_index = class_to_index.get(class_name)
            if expected_index is None or row["class_index"] != str(expected_index):
                raise ValueError(f"Class index mismatch: {relative_path}")
            if row["source_partition"] != expected_source_partition:
                raise ValueError(f"Source partition mismatch: {relative_path}")
            if relative_path in seen_paths:
                raise ValueError(f"Duplicate path in {manifest_path.name}: {relative_path}")

            image_path = (dataset_root / Path(*relative.parts)).resolve()
            if not image_path.is_relative_to(dataset_root):
                raise ValueError(f"Manifest path escapes dataset root: {relative_path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {image_path}")

            seen_paths.add(relative_path)
            group_ids.add(row["group_id"])
            image_paths.append(str(image_path))
            labels.append(expected_index)

    return image_paths, labels, group_ids


def get_dataloaders(
    raw_dir: str | Path = "data/raw/plantvillage/color",
    manifest_dir: str | Path = "data/splits",
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    random_seed: int = 42,
    num_workers: int = 0,
    augmentation_config: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], dict[str, int]]:
    """Create loaders from checksum-verified immutable split manifests."""
    dataset_dir = Path(raw_dir)
    frozen_dir = Path(manifest_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"PlantVillage color directory does not exist: {dataset_dir}")
    if not frozen_dir.is_dir():
        raise FileNotFoundError(f"Frozen manifest directory does not exist: {frozen_dir}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    validate_manifest_checksums(frozen_dir)
    classes, class_to_index = _load_class_mapping(frozen_dir)
    train_paths, train_labels, train_groups = _load_manifest_partition(
        dataset_dir,
        frozen_dir / "train.csv",
        class_to_index,
        "official_train",
    )
    validation_paths, validation_labels, validation_groups = _load_manifest_partition(
        dataset_dir,
        frozen_dir / "validation.csv",
        class_to_index,
        "official_train",
    )
    test_paths, test_labels, test_groups = _load_manifest_partition(
        dataset_dir,
        frozen_dir / "test.csv",
        class_to_index,
        "official_test",
    )

    path_sets = [set(train_paths), set(validation_paths), set(test_paths)]
    if any(path_sets[left] & path_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Frozen manifest image paths overlap")
    group_sets = [train_groups, validation_groups, test_groups]
    if any(group_sets[left] & group_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Frozen manifest leaf groups overlap")

    metadata = json.loads((frozen_dir / "split_metadata.json").read_text(encoding="utf-8"))
    observed_counts = {
        "train": len(train_paths),
        "validation": len(validation_paths),
        "test": len(test_paths),
    }
    if metadata.get("split_counts") != observed_counts:
        raise ValueError("Frozen split counts do not match split_metadata.json")

    transforms = get_transforms(
        image_size=image_size,
        augmentation_config=augmentation_config,
    )
    train_dataset = PlantDiseaseDataset(train_paths, train_labels, transforms["train"])
    validation_dataset = PlantDiseaseDataset(
        validation_paths, validation_labels, transforms["val"]
    )
    test_dataset = PlantDiseaseDataset(test_paths, test_labels, transforms["test"])
    generator = torch.Generator().manual_seed(random_seed)

    logger.info(
        "Loaded frozen splits: train=%d validation=%d test=%d",
        len(train_dataset),
        len(validation_dataset),
        len(test_dataset),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader, test_loader, classes, class_to_index
