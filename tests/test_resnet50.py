"""Comprehensive tests for ResNet-50 architecture and transfer learning protocol."""

import pytest
import torch
from torch import nn

from src.models.registry import build_model
from src.models.resnet50 import get_resnet50, unfreeze_resnet50_layers
from src.utils.helpers import count_parameters


def test_resnet50_initialization_frozen():
    """Verify ResNet-50 in Phase 1 has frozen base and trainable classification head."""
    model = get_resnet50(num_classes=38, pretrained=False, freeze_base=True, dropout_rate=0.4)

    total_params, trainable_params = count_parameters(model)
    assert total_params == 23_585_894
    assert trainable_params == 77_862

    # Check classifier structure
    assert isinstance(model.fc, nn.Sequential)
    assert isinstance(model.fc[0], nn.Dropout)
    assert model.fc[0].p == 0.4
    assert isinstance(model.fc[1], nn.Linear)
    assert model.fc[1].in_features == 2048
    assert model.fc[1].out_features == 38

    # Check that feature extractor layers are frozen
    for name, param in model.named_parameters():
        if "fc" not in name:
            assert not param.requires_grad, f"Parameter {name} should be frozen in Phase 1"
        else:
            assert param.requires_grad, f"Head parameter {name} should be trainable"


def test_resnet50_unfreeze_layer4():
    """Verify Phase 2 unfreezes layer4 residual blocks while keeping lower layers frozen."""
    model = get_resnet50(num_classes=38, pretrained=False, freeze_base=True, dropout_rate=0.4)
    unfreeze_resnet50_layers(model, ["layer4"])

    total_params, trainable_params = count_parameters(model)
    assert total_params == 23_585_894
    assert trainable_params == 15_042_598

    # Verify layer-specific gradient requirements
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            assert param.requires_grad, f"Parameter {name} should be trainable in Phase 2"
        else:
            assert not param.requires_grad, f"Parameter {name} should remain frozen in Phase 2"


def test_resnet50_forward_shape():
    """Verify ResNet-50 forward pass output shape with standard image dimensions."""
    model = get_resnet50(num_classes=38, pretrained=False, freeze_base=True)
    model.eval()

    inputs = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 38)
    assert torch.isfinite(outputs).all()


def test_resnet50_phase1_training_step():
    """Verify Phase 1 only updates the head weights and leaves the backbone unchanged."""
    torch.manual_seed(42)
    model = get_resnet50(num_classes=4, pretrained=False, freeze_base=True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
    )
    criterion = nn.CrossEntropyLoss()

    initial_conv1_weight = model.conv1.weight.detach().clone()
    initial_layer4_weight = model.layer4[0].conv1.weight.detach().clone()
    initial_fc_weight = model.fc[1].weight.detach().clone()

    inputs = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])

    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    # Backbone parameters should have no gradients
    assert model.conv1.weight.grad is None
    assert model.layer4[0].conv1.weight.grad is None
    assert model.fc[1].weight.grad is not None
    assert torch.isfinite(loss)

    optimizer.step()

    # Backbone weights must remain identical; fc weights must be updated
    assert torch.equal(model.conv1.weight, initial_conv1_weight)
    assert torch.equal(model.layer4[0].conv1.weight, initial_layer4_weight)
    assert not torch.equal(model.fc[1].weight, initial_fc_weight)


def test_resnet50_phase2_finetune_step():
    """Verify Phase 2 updates both layer4 and head weights while preserving layer1-3."""
    torch.manual_seed(42)
    model = get_resnet50(num_classes=4, pretrained=False, freeze_base=True)
    unfreeze_resnet50_layers(model, ["layer4"])

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0001,
    )
    criterion = nn.CrossEntropyLoss()

    initial_layer1_weight = model.layer1[0].conv1.weight.detach().clone()
    initial_layer4_weight = model.layer4[0].conv1.weight.detach().clone()
    initial_fc_weight = model.fc[1].weight.detach().clone()

    inputs = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])

    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    assert model.layer1[0].conv1.weight.grad is None
    assert model.layer4[0].conv1.weight.grad is not None
    assert model.fc[1].weight.grad is not None

    optimizer.step()

    assert torch.equal(model.layer1[0].conv1.weight, initial_layer1_weight)
    assert not torch.equal(model.layer4[0].conv1.weight, initial_layer4_weight)
    assert not torch.equal(model.fc[1].weight, initial_fc_weight)


def test_resnet50_registry_integration():
    """Verify build_model builds ResNet-50 and provides a functional unfreeze callback."""
    models_config = {
        "resnet50": {
            "name": "ResNet50",
            "pretrained": False,
            "freeze_base": True,
            "fine_tune_unfreeze_layers": ["layer4"],
            "dropout_rate": 0.4,
            "save_path": "models/resnet50/best_model.pt",
        }
    }

    spec = build_model("resnet50", num_classes=38, models_config=models_config)
    assert callable(spec.unfreeze)

    # Initially frozen
    _, trainable_before = count_parameters(spec.model)
    assert trainable_before == 77_862

    # Execute unfreeze callback
    spec.unfreeze(spec.model)
    _, trainable_after = count_parameters(spec.model)
    assert trainable_after == 15_042_598
