import torch
import pytest
from src.models.custom_cnn import CustomCNN
from src.models.resnet50 import get_resnet50
from src.models.efficientnet_b0 import get_efficientnet_b0
from src.models.mobilenet_v3 import get_mobilenet_v3, unfreeze_mobilenet_layers
from src.utils.helpers import count_parameters

def test_custom_cnn_forward():
    model = CustomCNN(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 10)
    total, trainable = count_parameters(model)
    assert total > 0 and trainable == total

def test_resnet50_forward():
    model = get_resnet50(num_classes=10, pretrained=False, freeze_base=True)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 10)

def test_efficientnet_forward():
    model = get_efficientnet_b0(num_classes=10, pretrained=False, freeze_base=True)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 10)

def test_mobilenet_forward():
    model = get_mobilenet_v3(num_classes=10, pretrained=False, freeze_base=True)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 10)

def test_mobilenet_freeze_base_only_trains_classifier():
    model = get_mobilenet_v3(num_classes=10, pretrained=False, freeze_base=True)
    total, trainable = count_parameters(model)
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    assert 0 < trainable < total
    assert trainable == classifier_params
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            assert not param.requires_grad

def test_mobilenet_unfreeze_last_stage_and_final_conv():
    model = get_mobilenet_v3(num_classes=10, pretrained=False, freeze_base=True)
    unfreeze_mobilenet_layers(model)
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            assert param.requires_grad
        elif any(f"features.{layer}." in name for layer in ("13", "14", "15", "16")):
            assert param.requires_grad
        else:
            assert not param.requires_grad
