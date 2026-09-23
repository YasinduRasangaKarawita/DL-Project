import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str = "Model",
    save_path: str = "figures/evaluation/confusion_matrix.png",
    normalize: bool = True
) -> str:
    """
    Generate and save high-resolution Confusion Matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = ".2f"
        title = f"{model_name} — Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title = f"{model_name} — Confusion Matrix (Counts)"

    # Shorten class names for readability if needed
    display_names = [name.replace("___", "\n") for name in class_names]

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        cbar=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(title, fontsize=14, weight="bold", pad=15)
    ax.set_ylabel("True Ground Truth Class", fontsize=11, labelpad=10)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
