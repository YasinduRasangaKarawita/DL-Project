import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from .callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from ..utils.logger import setup_logger

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
            "val_loss": [], "val_acc": [],
            "epoch_times": []
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
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

        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = (correct / total) if total > 0 else 0.0
        return {"loss": val_loss, "accuracy": val_acc}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        checkpoint_cb: Optional[ModelCheckpoint] = None,
        early_stopping_cb: Optional[EarlyStopping] = None,
        reduce_lr_cb: Optional[ReduceLROnPlateau] = None,
        csv_logger_cb: Optional[CSVLogger] = None,
        start_epoch: int = 1
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
            self.history["epoch_times"].append(train_res["time"])

            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {train_res['loss']:.4f} - Acc: {train_res['accuracy']*100:.2f}% | "
                f"Val Loss: {val_res['loss']:.4f} - Acc: {val_res['accuracy']*100:.2f}% | "
                f"LR: {current_lr:.6f} | Time: {train_res['time']:.2f}s"
            )

            # Callbacks
            if csv_logger_cb:
                csv_logger_cb.log({
                    "epoch": epoch,
                    "train_loss": round(train_res["loss"], 4),
                    "train_acc": round(train_res["accuracy"], 4),
                    "val_loss": round(val_res["loss"], 4),
                    "val_acc": round(val_res["accuracy"], 4),
                    "lr": current_lr,
                    "epoch_time_seconds": round(train_res["time"], 2)
                })

            if checkpoint_cb:
                checkpoint_cb(val_res["loss"], self.model, epoch)

            if reduce_lr_cb:
                reduce_lr_cb.step(val_res["loss"])

            if early_stopping_cb and early_stopping_cb(val_res["loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        total_duration = time.time() - total_start
        logger.info(f"Training completed in {total_duration:.2f} seconds.")
        return self.history
