import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50(num_classes: int = 15, pretrained: bool = True, freeze_base: bool = True, dropout_rate: float = 0.4) -> nn.Module:
    """
    Constructs ResNet50 model with transfer learning.
    - Uses ImageNet-1K pre-trained weights
    - Replaces final fully connected layer with Dropout + Linear classifier
    - If freeze_base is True, freezes feature extractor parameters for Phase 1
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Replace fc layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    return model

def unfreeze_resnet50_layers(model: nn.Module, layers_to_unfreeze: list = None) -> None:
    """
    Unfreeze upper residual blocks (e.g. 'layer4') for Phase 2 fine-tuning.
    """
    if layers_to_unfreeze is None:
        layers_to_unfreeze = ["layer4"]

    for name, child in model.named_children():
        if any(target in name for target in layers_to_unfreeze) or name == "fc":
            for param in child.parameters():
                param.requires_grad = True
