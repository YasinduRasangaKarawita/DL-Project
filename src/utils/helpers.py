import os
import yaml
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

def get_device(preference: str = "auto") -> torch.device:
    """
    Detect and return best available computing device.
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_parameters, trainable_parameters).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_mb(model_path: str) -> float:
    """
    Get file size of saved model checkpoint in Megabytes.
    """
    if not os.path.exists(model_path):
        return 0.0
    return os.path.getsize(model_path) / (1024 * 1024)

def load_yaml_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load project YAML configuration file safely.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
