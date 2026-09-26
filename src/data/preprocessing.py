import math
from numbers import Real
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _validate_transform_config(
    image_size: Tuple[int, int],
    rotation_degrees: Any,
    horizontal_flip_prob: Any,
    zoom_range: Any,
    brightness_factor: Any,
    contrast_factor: Any,
    mean: Any,
    std: Any,
) -> None:
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
        or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
            for dimension in image_size
        )
    ):
        raise ValueError("image_size must contain two positive integers")

    if not _is_finite_number(horizontal_flip_prob) or not 0 <= horizontal_flip_prob <= 1:
        raise ValueError("horizontal_flip_prob must be between 0 and 1")

    for name, value in (
        ("rotation_degrees", rotation_degrees),
        ("brightness_factor", brightness_factor),
        ("contrast_factor", contrast_factor),
    ):
        if not _is_finite_number(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number")

    if (
        not isinstance(zoom_range, (list, tuple))
        or len(zoom_range) != 2
        or any(not _is_finite_number(value) or value <= 0 for value in zoom_range)
        or zoom_range[0] > zoom_range[1]
    ):
        raise ValueError("zoom_range must contain two positive values in ascending order")

    if (
        not isinstance(mean, (list, tuple))
        or len(mean) != 3
        or any(not _is_finite_number(value) for value in mean)
    ):
        raise ValueError("normalization mean must contain three finite values")

    if (
        not isinstance(std, (list, tuple))
        or len(std) != 3
        or any(not _is_finite_number(value) or value <= 0 for value in std)
    ):
        raise ValueError("normalization std must contain three positive finite values")


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Dict[str, Any] = None,
) -> Dict[str, transforms.Compose]:
    """
    Construct torchvision transform pipelines.
    - 'train': includes data augmentation (flips, rotation, color jitter, affine)
    - 'val' & 'test': deterministic preprocessing ONLY (resize, crop, normalize)
    """
    if augmentation_config is None:
        augmentation_config = {}
    if not isinstance(augmentation_config, dict):
        raise ValueError("augmentation_config must be a dictionary")

    rot_deg = augmentation_config.get("rotation_degrees", 20)
    flip_prob = augmentation_config.get("horizontal_flip_prob", 0.5)
    zoom_range = augmentation_config.get("zoom_range", (0.8, 1.2))
    contrast = augmentation_config.get("contrast_factor", 0.2)
    brightness = augmentation_config.get("brightness_factor", 0.2)
    normalization = augmentation_config.get("normalization", {})
    if not isinstance(normalization, dict):
        raise ValueError("normalization must be a dictionary")
    mean = normalization.get("mean", IMAGENET_MEAN)
    std = normalization.get("std", IMAGENET_STD)

    _validate_transform_config(
        image_size,
        rot_deg,
        flip_prob,
        zoom_range,
        brightness,
        contrast,
        mean,
        std,
    )
    zoom_range = tuple(zoom_range)

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
        tensor = tensor[0]  # take first sample if batched

    tensor = tensor.clone().detach().cpu()
    for c in range(3):
        tensor[c] = tensor[c] * std[c] + mean[c]

    tensor = torch.clamp(tensor, 0.0, 1.0)
    np_img = tensor.permute(1, 2, 0).numpy()
    return (np_img * 255).astype(np.uint8)
