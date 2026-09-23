import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_efficientnet_b0(num_classes: int = 15, pretrained: bool = True, freeze_base: bool = True, dropout_rate: float = 0.3) -> nn.Module:
    """
    Constructs EfficientNetB0 model with transfer learning.
    - Compound scaling architecture
    - Replaces classifier with Dropout + Linear layer
    - Frozen base for Phase 1 feature extraction
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # EfficientNet classifier has in_features in classifier[1]
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    return model

def unfreeze_efficientnet_layers(model: nn.Module, layers_to_unfreeze: list = None) -> None:
    """
    Unfreeze top MBConv stages for fine-tuning.
    """
    if layers_to_unfreeze is None:
        layers_to_unfreeze = ["7", "8"]

    for name, param in model.named_parameters():
        if any(f"features.{layer}" in name for layer in layers_to_unfreeze) or "classifier" in name:
            param.requires_grad = True
