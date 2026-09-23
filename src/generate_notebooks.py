import os
import json

notebooks_dir = "notebooks"
os.makedirs(notebooks_dir, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

# 1. Dataset Exploration
nb1 = make_notebook([
    md_cell("# 🌿 01. PlantVillage Dataset Exploration & Quality Audit\nThis notebook performs Exploratory Data Analysis (EDA) on plant leaf images across disease and healthy categories."),
    code_cell("""import os, sys
sys.path.append('..')
from src.data.download_data import prepare_dataset
from src.data.validate_data import validate_dataset_integrity
from src.evaluation.dataset_analysis import generate_dataset_figures

# Ensure dataset exists and audit integrity
classes = prepare_dataset(raw_dir='../data/raw')
report = validate_dataset_integrity(raw_dir='../data/raw', report_path='../results/experiments/data_validation_report.json')
print(f"Total valid images: {report['valid_images']} across {report['total_classes']} classes.")"""),
    code_cell("""# Generate and display EDA visualizations
generate_dataset_figures(raw_dir='../data/raw', figures_dir='../figures/dataset')
print("EDA figures successfully generated in figures/dataset/")""")
])

# 2. Preprocessing
nb2 = make_notebook([
    md_cell("# 🔄 02. Preprocessing, Data Augmentation & Stratified Splitting\nDemonstrates data leakage prevention, stratified 70/15/15 train/val/test splitting, and torchvision transform pipelines."),
    code_cell("""import os, sys
sys.path.append('..')
from src.data.dataset_loader import get_dataloaders
from src.data.preprocessing import get_transforms

train_loader, val_loader, test_loader, classes, class_to_idx = get_dataloaders(
    raw_dir='../data/raw',
    processed_dir='../data/processed',
    batch_size=32,
    image_size=(224, 224),
    train_split=0.70, val_split=0.15, test_split=0.15,
    random_seed=42
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
print("Class categories:", classes)""")
])

# 3. Custom CNN
nb3 = make_notebook([
    md_cell("# 🤖 03. Custom CNN Baseline Model\nImplements 3-block convolutional baseline (Conv2D -> BatchNorm -> ReLU -> MaxPool) with Global Average Pooling."),
    code_cell("""import os, sys, torch
sys.path.append('..')
from src.models.custom_cnn import CustomCNN
from src.utils.helpers import count_parameters

model = CustomCNN(num_classes=15)
total_p, train_p = count_parameters(model)
print(f"Custom CNN Total Parameters: {total_p:,} | Trainable: {train_p:,}")

dummy_x = torch.randn(2, 3, 224, 224)
out = model(dummy_x)
print("Output logits shape:", out.shape)""")
])

# 4. ResNet50
nb4 = make_notebook([
    md_cell("# 🧠 04. ResNet50 Transfer Learning & Fine-Tuning\nTransfer learning with residual skip-connections. Evaluates frozen feature extraction vs fine-tuning layer4."),
    code_cell("""import os, sys, torch
sys.path.append('..')
from src.models.resnet50 import get_resnet50, unfreeze_resnet50_layers
from src.utils.helpers import count_parameters

model = get_resnet50(num_classes=15, pretrained=True, freeze_base=True)
total_p, train_p = count_parameters(model)
print(f"Phase 1 (Frozen) - Total: {total_p:,} | Trainable: {train_p:,}")

unfreeze_resnet50_layers(model, ['layer4'])
total_p, train_p = count_parameters(model)
print(f"Phase 2 (Fine-tuning layer4) - Total: {total_p:,} | Trainable: {train_p:,}")""")
])

# 5. EfficientNetB0
nb5 = make_notebook([
    md_cell("# ⚡ 05. EfficientNetB0 Compound Scaling\nEvaluates EfficientNetB0 balance between computational depth, width, resolution scaling and classification accuracy."),
    code_cell("""import os, sys, torch
sys.path.append('..')
from src.models.efficientnet_b0 import get_efficientnet_b0, unfreeze_efficientnet_layers
from src.utils.helpers import count_parameters

model = get_efficientnet_b0(num_classes=15, pretrained=True, freeze_base=True)
total_p, train_p = count_parameters(model)
print(f"EfficientNetB0 - Total: {total_p:,} | Trainable: {train_p:,}")""")
])

# 6. MobileNetV3
nb6 = make_notebook([
    md_cell("# 📱 06. MobileNetV3 Lightweight Edge Model\nEvaluates inverted residual bottlenecks for resource-constrained edge / mobile deployment."),
    code_cell("""import os, sys, torch
sys.path.append('..')
from src.models.mobilenet_v3 import get_mobilenet_v3
from src.utils.helpers import count_parameters

model = get_mobilenet_v3(num_classes=15, pretrained=True, freeze_base=True)
total_p, train_p = count_parameters(model)
print(f"MobileNetV3 - Total: {total_p:,} | Trainable: {train_p:,}")""")
])

# 7. Model Comparison
nb7 = make_notebook([
    md_cell("# 📊 07. Cross-Model Comparative Evaluation & Error Analysis\nGenerates final comparative metrics table, confusion matrices, error analysis reports, and Pareto efficiency trade-off charts."),
    code_cell("""import os, sys, pandas as pd
sys.path.append('..')
from src.evaluation.model_comparison import generate_model_comparison

df = generate_model_comparison(
    experiments_dir='../results/experiments',
    output_dir='../results/model_comparison',
    figures_dir='../figures/comparison'
)
print(df.to_string(index=False) if not df.empty else "Run pipeline to generate complete comparison records.")""")
])

notebook_map = {
    "01_dataset_exploration.ipynb": nb1,
    "02_preprocessing.ipynb": nb2,
    "03_custom_cnn.ipynb": nb3,
    "04_resnet50.ipynb": nb4,
    "05_efficientnet.ipynb": nb5,
    "06_mobilenet.ipynb": nb6,
    "07_model_comparison.ipynb": nb7
}

for fname, nb in notebook_map.items():
    path = os.path.join(notebooks_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {path}")
