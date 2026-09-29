"""Configuration-driven model construction for shared experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import nn

from .custom_cnn import build_custom_cnn
from .efficientnet_b0 import get_efficientnet_b0, unfreeze_efficientnet_layers
from .mobilenet_v3 import get_mobilenet_v3, unfreeze_mobilenet_layers
from .resnet50 import get_resnet50, unfreeze_resnet50_layers

MODEL_NAMES = ("custom_cnn", "resnet50", "efficientnet_b0", "mobilenet_v3")


@dataclass(frozen=True)
class ModelSpec:
    """A built model and the optional phase-two unfreeze operation."""

    model: nn.Module
    unfreeze: Callable[[nn.Module], None] | None


def _required_bool(config: Mapping[str, Any], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"models.*.{key} must be true or false")
    return value


def build_model(
    model_name: str,
    *,
    num_classes: int,
    models_config: Mapping[str, Mapping[str, Any]],
) -> ModelSpec:
    """Build one named model using only its authoritative configuration."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose one of {MODEL_NAMES}")
    config = models_config[model_name]

    if model_name == "custom_cnn":
        return ModelSpec(build_custom_cnn(num_classes, config), None)

    pretrained = _required_bool(config, "pretrained")
    freeze_base = _required_bool(config, "freeze_base")
    dropout_rate = float(config["dropout_rate"])
    configured_layers = config.get("fine_tune_unfreeze_layers", [])
    if not isinstance(configured_layers, list) or not all(
        isinstance(layer, str) and layer for layer in configured_layers
    ):
        raise ValueError("fine_tune_unfreeze_layers must be a list of non-empty strings")

    if model_name == "resnet50":
        model = get_resnet50(num_classes, pretrained, freeze_base, dropout_rate)

        def unfreeze(instance: nn.Module) -> None:
            unfreeze_resnet50_layers(instance, configured_layers)

        return ModelSpec(model, unfreeze)

    layer_numbers = [layer.rsplit(".", 1)[-1] for layer in configured_layers]
    if model_name == "efficientnet_b0":
        model = get_efficientnet_b0(num_classes, pretrained, freeze_base, dropout_rate)

        def unfreeze(instance: nn.Module) -> None:
            unfreeze_efficientnet_layers(instance, layer_numbers)

        return ModelSpec(model, unfreeze)

    model = get_mobilenet_v3(num_classes, pretrained, freeze_base, dropout_rate)

    def unfreeze(instance: nn.Module) -> None:
        unfreeze_mobilenet_layers(instance, layer_numbers)

    return ModelSpec(model, unfreeze)
