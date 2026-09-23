import torch
import numpy as np
from PIL import Image
from src.data.preprocessing import get_transforms, denormalize_image

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
