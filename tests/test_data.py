import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from src.data.dataset_loader import get_dataloaders
from src.data.split_manifests import MANIFEST_FIELDNAMES


def _write_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def _write_loader_fixture(tmp_path):
    dataset_dir = tmp_path / "color"
    manifest_dir = tmp_path / "splits"
    manifest_dir.mkdir()
    classes = ["Apple___healthy", "Tomato___healthy"]
    mapping = {class_name: index for index, class_name in enumerate(classes)}
    colors = {
        "Apple___healthy": (20, 120, 20),
        "Tomato___healthy": (40, 160, 40),
    }
    partitions = {"train": [], "validation": [], "test": []}
    for class_name in classes:
        for split, source_partition in (
            ("train", "official_train"),
            ("validation", "official_train"),
            ("test", "official_test"),
        ):
            count = 2 if split == "train" else 1
            for index in range(count):
                relative_path = f"{class_name}/{split}-{index}.JPG"
                _write_image(dataset_dir / relative_path, colors[class_name])
                partitions[split].append(
                    {
                        "relative_path": relative_path,
                        "class_name": class_name,
                        "class_index": mapping[class_name],
                        "group_id": f"{class_name}:::{split}-{index}",
                        "source_partition": source_partition,
                    }
                )

    generated = {}
    for split, rows in partitions.items():
        path = manifest_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=MANIFEST_FIELDNAMES,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        generated[path.name] = path

    class_mapping_path = manifest_dir / "class_mapping.json"
    class_mapping_path.write_text(
        json.dumps(
            {"schema_version": 1, "classes": classes, "class_to_index": mapping},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated[class_mapping_path.name] = class_mapping_path
    metadata_path = manifest_dir / "split_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {"split_counts": {split: len(rows) for split, rows in partitions.items()}},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated[metadata_path.name] = metadata_path
    (manifest_dir / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {filename}\n"
            for filename, path in sorted(generated.items())
        ),
        encoding="utf-8",
    )
    return dataset_dir, manifest_dir


def test_loads_frozen_manifests_without_resplitting(tmp_path):
    dataset_dir, manifest_dir = _write_loader_fixture(tmp_path)

    train_loader, validation_loader, test_loader, classes, class_to_index = get_dataloaders(
        raw_dir=dataset_dir,
        manifest_dir=manifest_dir,
        batch_size=2,
        image_size=(16, 16),
        random_seed=42,
    )

    assert classes == ["Apple___healthy", "Tomato___healthy"]
    assert class_to_index == {"Apple___healthy": 0, "Tomato___healthy": 1}
    assert len(train_loader.dataset) == 4
    assert len(validation_loader.dataset) == 2
    assert len(test_loader.dataset) == 2
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(validation_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    images, labels = next(iter(validation_loader))
    assert images.shape == (2, 3, 16, 16)
    assert labels.dtype == torch.int64
    assert all(Path(path).is_absolute() for path in test_loader.dataset.image_paths)


def test_rejects_manifest_with_invalid_checksum(tmp_path):
    dataset_dir, manifest_dir = _write_loader_fixture(tmp_path)
    with (manifest_dir / "train.csv").open("a", encoding="utf-8") as file_handle:
        file_handle.write("\n")

    with pytest.raises(ValueError, match="Checksum mismatch: train.csv"):
        get_dataloaders(raw_dir=dataset_dir, manifest_dir=manifest_dir)


def test_rejects_missing_manifest_image(tmp_path):
    dataset_dir, manifest_dir = _write_loader_fixture(tmp_path)
    missing_path = dataset_dir / "Apple___healthy" / "train-0.JPG"
    missing_path.unlink()

    with pytest.raises(FileNotFoundError, match="Manifest image does not exist"):
        get_dataloaders(raw_dir=dataset_dir, manifest_dir=manifest_dir)
