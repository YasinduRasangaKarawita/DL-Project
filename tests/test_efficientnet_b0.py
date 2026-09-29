"""Comprehensive tests for EfficientNet-B0 architecture and compound scaling transfer learning."""

import pytest
import torch
from torch import nn

from src.models.efficientnet_b0 import get_efficientnet_b0, unfreeze_efficientnet_layers
from src.models.registry import build_model
from src.utils.helpers import count_parameters


def test_efficientnet_initialization_frozen():
    """Verify EfficientNet-B0 in Phase 1 has frozen backbone and trainable classifier."""
    model = get_efficientnet_b0(num_classes=38, pretrained=False, freeze_base=True, dropout_rate=0.3)

    total_params, trainable_params = count_parameters(model)
    assert total_params == 4_056_226
    assert trainable_params == 48_678

    # Verify classifier structure
    assert isinstance(model.classifier, nn.Sequential)
    assert isinstance(model.classifier[0], nn.Dropout)
    assert model.classifier[0].p == 0.3
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].in_features == 1280
    assert model.classifier[1].out_features == 38

    # Verify base layers are frozen
    for name, param in model.named_parameters():
        if "classifier" not in name:
            assert not param.requires_grad, f"Parameter {name} should be frozen in Phase 1"
        else:
            assert param.requires_grad, f"Classifier parameter {name} should be trainable"


def test_efficientnet_unfreeze_top_stages():
    """Verify Phase 2 unfreezes features.7 and features.8 while keeping lower stages frozen."""
    model = get_efficientnet_b0(num_classes=38, pretrained=False, freeze_base=True, dropout_rate=0.3)
    unfreeze_efficientnet_layers(model, ["7", "8"])

    total_params, trainable_params = count_parameters(model)
    assert total_params == 4_056_226
    assert trainable_params == 1_178_070

    for name, param in model.named_parameters():
        if "features.7" in name or "features.8" in name or "classifier" in name:
            assert param.requires_grad, f"Parameter {name} should be trainable in Phase 2"
        else:
            assert not param.requires_grad, f"Parameter {name} should remain frozen in Phase 2"


def test_efficientnet_forward_shape():
    """Verify forward pass output shape for standard input dimensions."""
    model = get_efficientnet_b0(num_classes=38, pretrained=False, freeze_base=True)
    model.eval()

    inputs = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 38)
    assert torch.isfinite(outputs).all()


def test_efficientnet_phase1_training_step():
    """Verify Phase 1 updates only the classification head."""
    torch.manual_seed(42)
    model = get_efficientnet_b0(num_classes=4, pretrained=False, freeze_base=True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
    )
    criterion = nn.CrossEntropyLoss()

    initial_stem_weight = model.features[0][0].weight.detach().clone()
    initial_classifier_weight = model.classifier[1].weight.detach().clone()

    inputs = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])

    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    assert model.features[0][0].weight.grad is None
    assert model.classifier[1].weight.grad is not None
    assert torch.isfinite(loss)

    optimizer.step()

    assert torch.equal(model.features[0][0].weight, initial_stem_weight)
    assert not torch.equal(model.classifier[1].weight, initial_classifier_weight)


def test_efficientnet_phase2_finetune_step():
    """Verify Phase 2 updates features.7, features.8, and classifier while keeping features.0-6 frozen."""
    torch.manual_seed(42)
    model = get_efficientnet_b0(num_classes=4, pretrained=False, freeze_base=True)
    unfreeze_efficientnet_layers(model, ["7", "8"])

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0001,
    )
    criterion = nn.CrossEntropyLoss()

    initial_feat0_weight = model.features[0][0].weight.detach().clone()
    initial_feat7_weight = model.features[7][0].block[0][0].weight.detach().clone()
    initial_classifier_weight = model.classifier[1].weight.detach().clone()

    inputs = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])

    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    assert model.features[0][0].weight.grad is None
    assert model.features[7][0].block[0][0].weight.grad is not None
    assert model.classifier[1].weight.grad is not None

    optimizer.step()

    assert torch.equal(model.features[0][0].weight, initial_feat0_weight)
    assert not torch.equal(model.features[7][0].block[0][0].weight, initial_feat7_weight)
    assert not torch.equal(model.classifier[1].weight, initial_classifier_weight)


def test_efficientnet_registry_integration():
    """Verify build_model constructs EfficientNet-B0 and supplies a working unfreeze callback."""
    models_config = {
        "efficientnet_b0": {
            "name": "EfficientNet_B0",
            "pretrained": False,
            "freeze_base": True,
            "fine_tune_unfreeze_layers": ["features.7", "features.8"],
            "dropout_rate": 0.3,
            "save_path": "models/efficientnet/best_model.pt",
        }
    }

    spec = build_model("efficientnet_b0", num_classes=38, models_config=models_config)
    assert callable(spec.unfreeze)

    _, trainable_before = count_parameters(spec.model)
    assert trainable_before == 48_678

    spec.unfreeze(spec.model)
    _, trainable_after = count_parameters(spec.model)
    assert trainable_after == 1_178_070
