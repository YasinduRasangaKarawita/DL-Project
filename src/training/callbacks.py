import os
import csv
import torch
from typing import Dict, Any
from ..utils.logger import setup_logger

logger = setup_logger("callbacks")

class EarlyStopping:
    """
    Early stops training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric: float) -> bool:
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

class ModelCheckpoint:
    """
    Save best model weights when validation score improves.
    """
    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_val = float("inf") if mode == "min" else -float("inf")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def __call__(self, current_val: float, model: torch.nn.Module, epoch: int, extra_state: Dict[str, Any] = None) -> bool:
        improved = (current_val < self.best_val) if self.mode == "min" else (current_val > self.best_val)
        if improved:
            logger.info(f"Epoch {epoch}: {self.monitor} improved from {self.best_val:.4f} to {current_val:.4f}. Saving checkpoint to {self.filepath}")
            self.best_val = current_val
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "monitor_value": current_val,
            }
            if extra_state:
                state.update(extra_state)
            torch.save(state, self.filepath)
            return True
        return False

class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, factor: float = 0.5, patience: int = 2, min_lr: float = 1e-6, mode: str = "min"):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def step(self, val_metric: float) -> None:
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    if new_lr < old_lr:
                        param_group['lr'] = new_lr
                        logger.info(f"ReduceLROnPlateau: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                self.counter = 0
        else:
            self.best_score = score
            self.counter = 0

class CSVLogger:
    """
    Stream epoch training and validation metrics to CSV.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_seconds"]
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)
