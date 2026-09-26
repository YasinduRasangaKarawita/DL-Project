"""Focused tests for the Custom CNN baseline architecture."""

import pytest
import torch
from torch import nn

from src.models.custom_cnn import CustomCNN, build_custom_cnn
from src.utils.helpers import count_parameters


def test_custom_cnn_uses_configurable_architecture():
    model = CustomCNN(
        num_classes=38,
        conv_channels=[16, 32, 64],
        dense_units=128,
        dropout_rate=0.25,
    )

    assert model.block1[0].in_channels == 3
    assert model.block1[0].out_channels == 16
    assert model.block2[0].in_channels == 16
    assert model.block2[0].out_channels == 32
    assert model.block3[0].in_channels == 32
    assert model.block3[0].out_channels == 64
    assert model.classifier[1].in_features == 64
    assert model.classifier[1].out_features == 128
    assert isinstance(model.classifier[3], nn.Dropout)
    assert model.classifier[3].p == 0.25
    assert model.classifier[4].out_features == 38

    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 38)


def test_default_custom_cnn_parameter_count_is_stable():
    model = CustomCNN(num_classes=38)

    total_parameters, trainable_parameters = count_parameters(model)

    assert total_parameters == 136_262
    assert trainable_parameters == total_parameters


def test_build_custom_cnn_uses_experiment_config():
    model = build_custom_cnn(
        num_classes=38,
        model_config={
            "conv_channels": [12, 24, 48],
            "dense_units": 96,
            "dropout_rate": 0.3,
        },
    )

    assert model.block1[0].out_channels == 12
    assert model.block2[0].out_channels == 24
    assert model.block3[0].out_channels == 48
    assert model.classifier[1].in_features == 48
    assert model.classifier[1].out_features == 96
    assert model.classifier[3].p == 0.3
    assert model.classifier[4].out_features == 38


@pytest.mark.parametrize(
    ("model_kwargs", "expected_message"),
    [
        ({"num_classes": 0}, "num_classes"),
        ({"num_classes": 38, "conv_channels": [32, 64]}, "conv_channels"),
        ({"num_classes": 38, "conv_channels": [32, 0, 128]}, "conv_channels"),
        ({"num_classes": 38, "dense_units": 0}, "dense_units"),
        ({"num_classes": 38, "dropout_rate": -0.1}, "dropout_rate"),
        ({"num_classes": 38, "dropout_rate": 1.0}, "dropout_rate"),
    ],
)
def test_custom_cnn_rejects_invalid_architecture(model_kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        CustomCNN(**model_kwargs)
