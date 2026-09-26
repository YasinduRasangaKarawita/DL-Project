"""Configuration-driven Custom CNN baseline."""

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch
from torch import nn


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class CustomCNN(nn.Module):
    """Three-block convolutional baseline for PlantVillage classification.

    The default architecture is::

        RGB input
          -> Conv(3, 32)   -> BatchNorm -> ReLU -> MaxPool(2)
          -> Conv(32, 64)  -> BatchNorm -> ReLU -> MaxPool(2)
          -> Conv(64, 128) -> BatchNorm -> ReLU -> MaxPool(2)
          -> AdaptiveAvgPool(1, 1)
          -> Linear(128, 256) -> ReLU -> Dropout(0.5)
          -> Linear(256, num_classes)

    ``conv_channels``, ``dense_units``, and ``dropout_rate`` are exposed so
    the values declared in the experiment configuration can be authoritative.
    """

    def __init__(
        self,
        num_classes: int,
        conv_channels: Sequence[int] = (32, 64, 128),
        dense_units: int = 256,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        if not _is_positive_integer(num_classes):
            raise ValueError("num_classes must be a positive integer")
        if (
            not isinstance(conv_channels, Sequence)
            or isinstance(conv_channels, (str, bytes))
            or len(conv_channels) != 3
            or any(not _is_positive_integer(channel) for channel in conv_channels)
        ):
            raise ValueError("conv_channels must contain three positive integers")
        if not _is_positive_integer(dense_units):
            raise ValueError("dense_units must be a positive integer")
        if (
            not isinstance(dropout_rate, Real)
            or isinstance(dropout_rate, bool)
            or not math.isfinite(dropout_rate)
            or not 0 <= dropout_rate < 1
        ):
            raise ValueError("dropout_rate must be a finite value in [0, 1)")

        channel_1, channel_2, channel_3 = conv_channels
        self.block1 = self._make_conv_block(3, channel_1)
        self.block2 = self._make_conv_block(channel_1, channel_2)
        self.block3 = self._make_conv_block(channel_2, channel_3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_3, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout_rate)),
            nn.Linear(dense_units, num_classes),
        )

    @staticmethod
    def _make_conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.block1(inputs)
        features = self.block2(features)
        features = self.block3(features)
        features = self.global_pool(features)
        return self.classifier(features)


def build_custom_cnn(num_classes: int, model_config: Mapping[str, Any]) -> CustomCNN:
    """Build the baseline from the authoritative experiment configuration."""
    return CustomCNN(
        num_classes=num_classes,
        conv_channels=model_config["conv_channels"],
        dense_units=model_config["dense_units"],
        dropout_rate=model_config["dropout_rate"],
    )
