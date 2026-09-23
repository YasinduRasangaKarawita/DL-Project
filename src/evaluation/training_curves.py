import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_learning_curves(
    csv_history_path: str,
    model_name: str,
    save_path: str
) -> None:
    """
    Plot Training and Validation Loss and Accuracy vs Epochs.
    Identifies overfitting, convergence point, and stability.
    """
    if not os.path.exists(csv_history_path):
        return

    df = pd.read_csv(csv_history_path)
    if df.empty or "epoch" not in df.columns:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss Curve
    ax1.plot(df["epoch"], df["train_loss"], label="Training Loss", color="#2563eb", linewidth=2)
    ax1.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="#dc2626", linewidth=2, linestyle="--")
    ax1.set_title(f"{model_name} — Loss vs Epoch", fontsize=12, weight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy Curve
    ax2.plot(df["epoch"], df["train_acc"] * 100, label="Training Accuracy", color="#16a34a", linewidth=2)
    ax2.plot(df["epoch"], df["val_acc"] * 100, label="Validation Accuracy", color="#ea580c", linewidth=2, linestyle="--")
    ax2.set_title(f"{model_name} — Accuracy vs Epoch", fontsize=12, weight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
