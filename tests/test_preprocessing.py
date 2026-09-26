import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.data.preprocessing import denormalize_image, get_transforms


def test_preprocessing_transforms():
    transforms_dict = get_transforms(image_size=(224, 224))
    assert "train" in transforms_dict
    assert "val" in transforms_dict
    assert "test" in transforms_dict

    # Dummy PIL image
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    tensor_train = transforms_dict["train"](img)
    tensor_test = transforms_dict["test"](img)

    assert tensor_train.shape == (3, 224, 224)
    assert tensor_test.shape == (3, 224, 224)

    # Denormalize
    np_img = denormalize_image(tensor_test)
    assert np_img.shape == (224, 224, 3)
    assert np_img.dtype == np.uint8


def test_zoom_augmentation_is_training_only():
    transforms_dict = get_transforms(
        augmentation_config={"zoom_range": [0.9, 1.1]},
    )

    train_zoom = [
        transform
        for transform in transforms_dict["train"].transforms
        if isinstance(transform, transforms.RandomAffine)
    ]

    assert len(train_zoom) == 1
    assert train_zoom[0].scale == (0.9, 1.1)
    assert not any(
        isinstance(transform, transforms.RandomAffine)
        for transform in transforms_dict["val"].transforms
    )
    assert transforms_dict["val"] is transforms_dict["test"]


def test_all_random_augmentations_are_training_only():
    transforms_dict = get_transforms()
    random_transform_types = (
        transforms.RandomHorizontalFlip,
        transforms.RandomRotation,
        transforms.RandomAffine,
        transforms.ColorJitter,
    )

    for transform_type in random_transform_types:
        assert any(
            isinstance(transform, transform_type)
            for transform in transforms_dict["train"].transforms
        )
        assert not any(
            isinstance(transform, transform_type)
            for transform in transforms_dict["val"].transforms
        )
        assert not any(
            isinstance(transform, transform_type)
            for transform in transforms_dict["test"].transforms
        )


def test_validation_and_test_transforms_are_deterministic():
    transforms_dict = get_transforms(image_size=(32, 32))
    pixel_values = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    image = Image.fromarray(pixel_values, mode="RGB")

    first_validation = transforms_dict["val"](image)
    second_validation = transforms_dict["val"](image)
    test_result = transforms_dict["test"](image)

    assert torch.equal(first_validation, second_validation)
    assert torch.equal(first_validation, test_result)


def test_custom_normalization_is_applied():
    transforms_dict = get_transforms(
        image_size=(16, 16),
        augmentation_config={
            "normalization": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            }
        },
    )
    white_image = Image.new("RGB", (16, 16), color=(255, 255, 255))

    normalized = transforms_dict["val"](white_image)

    assert torch.equal(normalized, torch.ones((3, 16, 16)))


@pytest.mark.parametrize(
    ("image_size", "augmentation_config", "expected_message"),
    [
        ((0, 224), {}, "image_size"),
        ((224, 224), {"horizontal_flip_prob": 1.1}, "horizontal_flip_prob"),
        ((224, 224), {"rotation_degrees": -1}, "rotation_degrees"),
        ((224, 224), {"brightness_factor": -0.1}, "brightness_factor"),
        ((224, 224), {"contrast_factor": float("inf")}, "contrast_factor"),
        ((224, 224), {"zoom_range": [1.2, 0.8]}, "zoom_range"),
        (
            (224, 224),
            {"normalization": {"mean": [0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
            "normalization mean",
        ),
        (
            (224, 224),
            {"normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0, 0.5]}},
            "normalization std",
        ),
    ],
)
def test_invalid_transform_config_is_rejected(
    image_size,
    augmentation_config,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        get_transforms(image_size=image_size, augmentation_config=augmentation_config)
