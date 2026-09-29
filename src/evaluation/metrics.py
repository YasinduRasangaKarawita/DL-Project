import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def evaluate_model_metrics(
    model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    warmup_batches: int = 5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluate trained model on unseen test set.
    Returns:
      - metrics_dict (overall + per-class metrics)
      - all_targets (ground truth array)
      - all_preds (predicted class array)
      - all_probs (softmax probabilities array: N x C)
      - inference_latency_ms (mean inference time per image in milliseconds)
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    timed_seconds = 0.0
    timed_images = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            if batch_index >= warmup_batches:
                timed_seconds += elapsed
                timed_images += images.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)
    mean_latency_ms = (timed_seconds * 1000 / timed_images) if timed_images else 0.0
    throughput = (timed_images / timed_seconds) if timed_seconds else 0.0

    acc = float(accuracy_score(targets_arr, preds_arr))
    macro_p = float(precision_score(targets_arr, preds_arr, average="macro", zero_division=0))
    macro_r = float(recall_score(targets_arr, preds_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))

    weighted_p = float(precision_score(targets_arr, preds_arr, average="weighted", zero_division=0))
    weighted_r = float(recall_score(targets_arr, preds_arr, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(targets_arr, preds_arr, average="weighted", zero_division=0))

    labels = list(range(len(class_names)))
    try:
        macro_roc_auc = float(
            roc_auc_score(
                targets_arr,
                probs_arr,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        weighted_roc_auc = float(
            roc_auc_score(
                targets_arr,
                probs_arr,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        macro_roc_auc = float("nan")
        weighted_roc_auc = float("nan")

    # Top-3 Accuracy
    top3_correct = 0
    for i, target in enumerate(targets_arr):
        top3_indices = np.argsort(probs_arr[i])[-3:]
        if target in top3_indices:
            top3_correct += 1
    top3_acc = float(top3_correct / len(targets_arr)) if len(targets_arr) > 0 else 0.0

    # Per-class report
    clf_report = classification_report(
        targets_arr, preds_arr,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    for class_index, class_name in enumerate(class_names):
        binary_targets = (targets_arr == class_index).astype(int)
        try:
            class_auc = float(roc_auc_score(binary_targets, probs_arr[:, class_index]))
        except ValueError:
            class_auc = float("nan")
        clf_report[class_name]["roc_auc_ovr"] = round(class_auc, 4)

    metrics_dict = {
        "accuracy": round(acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_roc_auc_ovr": round(macro_roc_auc, 4),
        "weighted_roc_auc_ovr": round(weighted_roc_auc, 4),
        "inference_latency_ms_per_image": round(mean_latency_ms, 2),
        "throughput_images_per_second": round(throughput, 2),
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
            if device.type == "cuda"
            else None
        ),
        "benchmark": {
            "warmup_batches": warmup_batches,
            "timed_images": timed_images,
            "device": str(device),
        },
        "per_class_report": clf_report
    }

    return metrics_dict, targets_arr, preds_arr, probs_arr, mean_latency_ms
