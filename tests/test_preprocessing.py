import numpy as np
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
