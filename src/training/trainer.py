import time
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ..utils.logger import setup_logger
from .callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau

logger = setup_logger("trainer")

class ModelTrainer:
    """
    Unified trainer for plant disease classification models.
    Handles device management, evaluation loops, timing, and callback orchestration.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None,
        device: torch.device = None,
        callbacks: Optional[List[Any]] = None
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.callbacks = callbacks if callbacks is not None else []
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_macro_f1": [],
            "epoch_times": []
        }

    def load_history(self, history: Dict[str, List[float]]) -> None:
        expected = set(self.history)
        if set(history) != expected:
            raise ValueError("Checkpoint training history has an incompatible schema")
        self.history = history

    def _set_frozen_batchnorm_eval(self) -> None:
        """Keep frozen backbone running statistics fixed during feature extraction."""
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                parameters = list(module.parameters(recurse=False))
                if parameters and all(not parameter.requires_grad for parameter in parameters):
                    module.eval()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        self._set_frozen_batchnorm_eval()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_time = time.time() - start_time
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = (correct / total) if total > 0 else 0.0

        return {"loss": epoch_loss, "accuracy": epoch_acc, "time": epoch_time}

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        targets: list[int] = []
        predictions: list[int] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)
                targets.extend(labels.cpu().tolist())
                predictions.extend(preds.cpu().tolist())

        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = (correct / total) if total > 0 else 0.0
        macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
        return {"loss": val_loss, "accuracy": val_acc, "macro_f1": macro_f1}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        early_stopping_cb: Optional[EarlyStopping] = None,
        reduce_lr_cb: Optional[ReduceLROnPlateau] = None,
        csv_logger_cb: Optional[CSVLogger] = None,
        start_epoch: int = 1,
        phase: str = "initial",
        epoch_end_cb: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, List[float]]:
        logger.info(f"Starting training on device: {self.device} for {epochs} epochs...")
        total_start = time.time()

        for epoch in range(start_epoch, start_epoch + epochs):
            train_res = self.train_epoch(train_loader)
            val_res = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history["train_loss"].append(train_res["loss"])
            self.history["train_acc"].append(train_res["accuracy"])
            self.history["val_loss"].append(val_res["loss"])
            self.history["val_acc"].append(val_res["accuracy"])
            self.history["val_macro_f1"].append(val_res["macro_f1"])
            self.history["epoch_times"].append(train_res["time"])

            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {train_res['loss']:.4f} - Acc: {train_res['accuracy']*100:.2f}% | "
                f"Val Loss: {val_res['loss']:.4f} - Acc: {val_res['accuracy']*100:.2f}% - "
                f"Macro F1: {val_res['macro_f1']:.4f} | "
                f"LR: {current_lr:.6f} | Time: {train_res['time']:.2f}s"
            )

            # Callbacks
            if csv_logger_cb:
                csv_logger_cb.log({
                    "epoch": epoch,
                    "phase": phase,
                    "train_loss": round(train_res["loss"], 4),
                    "train_acc": round(train_res["accuracy"], 4),
                    "val_loss": round(val_res["loss"], 4),
                    "val_acc": round(val_res["accuracy"], 4),
                    "val_macro_f1": round(val_res["macro_f1"], 4),
                    "lr": current_lr,
                    "epoch_time_seconds": round(train_res["time"], 2)
                })

            epoch_metrics = {
                "train_loss": train_res["loss"],
                "train_accuracy": train_res["accuracy"],
                "validation_loss": val_res["loss"],
                "validation_accuracy": val_res["accuracy"],
                "validation_macro_f1": val_res["macro_f1"],
                "learning_rate": current_lr,
                "epoch_time_seconds": train_res["time"],
            }

            if reduce_lr_cb:
                reduce_lr_cb.step(val_res["macro_f1"])

            should_stop = bool(
                early_stopping_cb and early_stopping_cb(val_res["macro_f1"])
            )

            if epoch_end_cb:
                epoch_end_cb(epoch, epoch_metrics)

            if should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        total_duration = time.time() - total_start
        logger.info(f"Training completed in {total_duration:.2f} seconds.")
        return self.history
