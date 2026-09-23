import os
import json
import time
import torch
from typing import Dict, Any
from ..utils.helpers import count_parameters, get_model_size_mb
from ..utils.logger import setup_logger

logger = setup_logger("experiment_tracker")

class ExperimentTracker:
    """
    Tracks and records experiment hyperparameters, architecture complexity,
    training durations, and test evaluation metrics.
    """
    def __init__(self, experiment_name: str, base_dir: str = "results/experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.base_dir, f"{experiment_name}_meta.json")

    def record_run(
        self,
        model: torch.nn.Module,
        model_save_path: str,
        hyperparameters: Dict[str, Any],
        training_time_seconds: float,
        inference_time_ms: float,
        test_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        total_params, trainable_params = count_parameters(model)
        model_size_mb = get_model_size_mb(model_save_path)

        record = {
            "experiment_name": self.experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": hyperparameters,
            "parameters": {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": total_params - trainable_params,
                "model_file_size_mb": round(model_size_mb, 2)
            },
            "computational_efficiency": {
                "total_training_time_sec": round(training_time_seconds, 2),
                "inference_latency_ms_per_image": round(inference_time_ms, 2)
            },
            "test_metrics": test_metrics
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        logger.info(f"Experiment metadata logged successfully to {self.metadata_path}")
        return record
