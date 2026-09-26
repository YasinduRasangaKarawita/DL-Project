from typing import Any, Dict, Tuple

import numpy as np
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Dict[str, Any] = None
) -> Dict[str, transforms.Compose]:
    """
    Construct torchvision transform pipelines.
    - 'train': includes data augmentation (flips, rotation, color jitter, affine)
    - 'val' & 'test': deterministic preprocessing ONLY (resize, crop, normalize)
    """
    if augmentation_config is None:
        augmentation_config = {}

    rot_deg = augmentation_config.get("rotation_degrees", 20)
    flip_prob = augmentation_config.get("horizontal_flip_prob", 0.5)
    zoom_range = tuple(augmentation_config.get("zoom_range", (0.8, 1.2)))
    contrast = augmentation_config.get("contrast_factor", 0.2)
    brightness = augmentation_config.get("brightness_factor", 0.2)
    mean = augmentation_config.get("normalization", {}).get("mean", IMAGENET_MEAN)
    std = augmentation_config.get("normalization", {}).get("std", IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=flip_prob),
        transforms.RandomRotation(degrees=rot_deg),
        transforms.RandomAffine(degrees=0, scale=zoom_range),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform
    }

def denormalize_image(tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """
    Denormalize a float PyTorch image tensor (C, H, W) to a uint8 numpy array (H, W, C) for plotting.
    """
    if tensor.dim() == 4:
        tensor = tensor[0] # take first sample if batched

    tensor = tensor.clone().detach().cpu()
    for c in range(3):
        tensor[c] = tensor[c] * std[c] + mean[c]

    tensor = torch.clamp(tensor, 0.0, 1.0)
    np_img = tensor.permute(1, 2, 0).numpy()
    return (np_img * 255).astype(np.uint8)
