import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

def get_mobilenet_v3(num_classes: int = 15, pretrained: bool = True, freeze_base: bool = True, dropout_rate: float = 0.2) -> nn.Module:
    """
    Constructs MobileNetV3-Large model for efficient edge / mobile plant disease detection.
    """
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # MobileNetV3 classifier: Linear -> Hardswish -> Dropout -> Linear
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(1024, num_classes)
    )

    return model

def unfreeze_mobilenet_layers(model: nn.Module, layers_to_unfreeze: list = None) -> None:
    """
    Unfreeze the last inverted-residual stage (features.13-15) plus the final
    Conv2dNormActivation (features.16) for fine-tuning.
    """
    if layers_to_unfreeze is None:
        layers_to_unfreeze = ["13", "14", "15", "16"]

    for name, param in model.named_parameters():
        if any(f"features.{layer}" in name for layer in layers_to_unfreeze) or "classifier" in name:
            param.requires_grad = True
