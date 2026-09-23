import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import List, Dict

def generate_dataset_figures(
    raw_dir: str = "data/raw",
    figures_dir: str = "figures/dataset"
) -> None:
    """
    Generate Exploratory Data Analysis (EDA) charts:
    - Class distribution bar plot
    - Grid of representative sample leaf images across classes
    """
    os.makedirs(figures_dir, exist_ok=True)
    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    
    counts = {}
    sample_images = {}
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        counts[cls] = len(imgs)
        if imgs:
            sample_images[cls] = os.path.join(cls_dir, imgs[0])

    # 1. Class Distribution Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [c.replace("___", "\n") for c in counts.keys()]
    values = list(counts.values())
    sns.barplot(x=names, y=values, palette="mako", ax=ax)
    ax.set_title("PlantVillage Dataset — Class Sample Distribution", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Image Count", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "class_distribution.png"), dpi=200)
    plt.close(fig)

    # 2. Sample Images Grid (up to 12 classes)
    display_classes = classes[:12]
    rows = (len(display_classes) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, cls in enumerate(display_classes):
        img_path = sample_images.get(cls)
        if img_path and os.path.exists(img_path):
            with Image.open(img_path) as img:
                axes[idx].imshow(img)
        display_label = cls.replace("___", "\n")
        axes[idx].set_title(display_label, fontsize=10, weight="semibold")
        axes[idx].axis("off")

    for idx in range(len(display_classes), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Representative Leaf Images per Disease Category", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sample_images.png"), dpi=200)
    plt.close(fig)
