import os
import json
import numpy as np
from typing import List, Dict, Any

def run_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    test_paths: List[str],
    class_names: List[str],
    model_name: str = "Model",
    output_dir: str = "results/predictions"
) -> Dict[str, Any]:
    """
    Perform deep qualitative and quantitative error analysis.
    Identifies top confused class pairs, hard examples, and lowest performing classes.
    """
    os.makedirs(output_dir, exist_ok=True)
    misclassifications_path = os.path.join(output_dir, f"{model_name.lower()}_misclassifications.json")

    # Find misclassified sample indices
    misclassified_indices = np.where(y_true != y_pred)[0]

    misclassified_samples = []
    confusion_pairs: Dict[str, int] = {}

    for idx in misclassified_indices:
        true_cls = class_names[y_true[idx]]
        pred_cls = class_names[y_pred[idx]]
        confidence = float(y_probs[idx][y_pred[idx]])

        pair_key = f"{true_cls} -> {pred_cls}"
        confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1

        sample_info = {
            "index": int(idx),
            "image_path": test_paths[idx] if idx < len(test_paths) else f"sample_{idx}",
            "actual_class": true_cls,
            "predicted_class": pred_cls,
            "confidence_score": round(confidence, 4)
        }
        misclassified_samples.append(sample_info)

    # Sort top confused pairs
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    summary = {
        "model_name": model_name,
        "total_test_samples": len(y_true),
        "total_misclassifications": len(misclassified_indices),
        "error_rate": round(len(misclassified_indices) / len(y_true), 4) if len(y_true) > 0 else 0.0,
        "top_confused_pairs": [{"pair": pair, "count": count} for pair, count in sorted_pairs[:10]],
        "sample_misclassifications": misclassified_samples[:50]
    }

    with open(misclassifications_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
