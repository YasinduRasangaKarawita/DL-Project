import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any
from .preprocessing import get_transforms
from ..utils.logger import setup_logger

logger = setup_logger("dataset_loader")

class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset for PlantVillage image classification.
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        return img, label

def get_dataloaders(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0,
    augmentation_config: Dict[str, Any] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    """
    Creates stratified 70% Train, 15% Val, 15% Unseen Test split.
    Saves split information to data/processed for 100% reproducible cross-model evaluation.
    """
    os.makedirs(processed_dir, exist_ok=True)
    split_cache_path = os.path.join(processed_dir, "split_indices.json")
    class_map_path = os.path.join(processed_dir, "class_to_idx.json")

    # Discover classes
    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    # Check if split cache exists
    if os.path.exists(split_cache_path):
        logger.info(f"Loading cached reproducible data split from {split_cache_path}")
        with open(split_cache_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        
        train_paths = split_data["train_paths"]
        train_labels = split_data["train_labels"]
        val_paths = split_data["val_paths"]
        val_labels = split_data["val_labels"]
        test_paths = split_data["test_paths"]
        test_labels = split_data["test_labels"]
    else:
        logger.info("Computing new stratified Train/Validation/Test split...")
        all_paths = []
        all_labels = []

        for cls_name in classes:
            cls_dir = os.path.join(raw_dir, cls_name)
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_paths.append(os.path.join(cls_dir, f))
                    all_labels.append(class_to_idx[cls_name])

        # Step 1: Split train vs (val + test)
        eval_ratio = val_split + test_split
        train_paths, eval_paths, train_labels, eval_labels = train_test_split(
            all_paths, all_labels,
            test_size=eval_ratio,
            stratify=all_labels,
            random_state=random_seed
        )

        # Step 2: Split val vs test equally
        val_sub_ratio = val_split / eval_ratio
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            eval_paths, eval_labels,
            train_size=val_sub_ratio,
            stratify=eval_labels,
            random_state=random_seed
        )

        # Save split for reproducibility
        split_dict = {
            "random_seed": random_seed,
            "train_count": len(train_paths),
            "val_count": len(val_paths),
            "test_count": len(test_paths),
            "train_paths": train_paths,
            "train_labels": train_labels,
            "val_paths": val_paths,
            "val_labels": val_labels,
            "test_paths": test_paths,
            "test_labels": test_labels
        }
        with open(split_cache_path, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2)

        logger.info(f"Split completed: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    transforms_dict = get_transforms(image_size=image_size, augmentation_config=augmentation_config)

    train_dataset = PlantDiseaseDataset(train_paths, train_labels, transform=transforms_dict["train"])
    val_dataset = PlantDiseaseDataset(val_paths, val_labels, transform=transforms_dict["val"])
    test_dataset = PlantDiseaseDataset(test_paths, test_labels, transform=transforms_dict["test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes, class_to_idx
