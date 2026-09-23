import torch
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader

def evaluate_model_metrics(
    model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device
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

    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            t0 = time.time()
            outputs = model(images)
            t1 = time.time()

            batch_latency_ms = ((t1 - t0) * 1000) / images.size(0)
            inference_times.append(batch_latency_ms)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)
    mean_latency_ms = float(np.mean(inference_times)) if inference_times else 0.0

    acc = float(accuracy_score(targets_arr, preds_arr))
    macro_p = float(precision_score(targets_arr, preds_arr, average="macro", zero_division=0))
    macro_r = float(recall_score(targets_arr, preds_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    
    weighted_p = float(precision_score(targets_arr, preds_arr, average="weighted", zero_division=0))
    weighted_r = float(recall_score(targets_arr, preds_arr, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(targets_arr, preds_arr, average="weighted", zero_division=0))

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
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    metrics_dict = {
        "accuracy": round(acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "inference_latency_ms_per_image": round(mean_latency_ms, 2),
        "per_class_report": clf_report
    }

    return metrics_dict, targets_arr, preds_arr, probs_arr, mean_latency_ms
