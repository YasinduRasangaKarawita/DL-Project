import json
import os

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
    md_cell("""# 🌿 01. PlantVillage Dataset Exploration & Quality Audit

This notebook uses the locked PlantVillage source, reviewed exclusion evidence, and checksum-verified frozen manifests. It reports class balance, image dimensions, representative **training-only** samples, and the data-quality decisions that prevent leakage. The locked test images are never displayed or used to make preprocessing or model-design decisions."""),
    code_cell("""import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image as NotebookImage
from IPython.display import Markdown, display

project_root = Path.cwd().resolve()
if not (project_root / 'src').is_dir():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.evaluation.dataset_analysis import (
    analyze_class_distribution,
    analyze_image_dimensions,
    generate_dataset_figures,
    summarize_data_quality,
    summarize_frozen_splits,
)

raw_dir = project_root / 'data/raw/plantvillage/color'
manifest_dir = project_root / 'data/splits'
figures_dir = project_root / 'figures/dataset'
validation_report = project_root / 'data/processed/plantvillage/validation_report.json'"""),
    md_cell("""## 1. Revalidate the locked dataset evidence

These commands fully decode the acquired images, recheck the human-review and exclusion ledgers, and independently validate the frozen manifest bundle. They may take several minutes."""),
    code_cell("""for command in ('validate', 'validate-review', 'validate-exclusions', 'validate-manifests'):
    print(f'Running data check: {command}')
    subprocess.run(
        [sys.executable, 'scripts/prepare_data.py', command],
        cwd=project_root,
        check=True,
    )"""),
    md_cell("## 2. Frozen split and class-distribution summary"),
    code_cell("""split_summary = summarize_frozen_splits(manifest_dir)
training_distribution = analyze_class_distribution(split_summary, split='train')

display(pd.DataFrame([{
    'classes': split_summary['num_classes'],
    'train_images': split_summary['split_counts']['train'],
    'validation_images': split_summary['split_counts']['validation'],
    'test_images': split_summary['split_counts']['test'],
    'total_images': split_summary['total_images'],
    'manifest_bundle_sha256': split_summary['manifest_bundle_sha256'],
}]))

class_table = pd.DataFrame(training_distribution['distribution']).sort_values(
    'count', ascending=False
)
display(class_table.reset_index(drop=True))
print(f"Largest class: {training_distribution['largest_class']['class_name']} "
      f"({training_distribution['largest_class']['count']:,})")
print(f"Smallest class: {training_distribution['smallest_class']['class_name']} "
      f"({training_distribution['smallest_class']['count']:,})")
print(f"Training imbalance ratio: {training_distribution['imbalance_ratio']:.2f}:1")"""),
    md_cell("""## 3. Manifest-backed EDA figures

The distribution chart uses only the frozen training counts. The sample grid selects one training image per class deterministically with seed 42; it does not cherry-pick examples or display locked test images."""),
    code_cell("""figure_paths = generate_dataset_figures(
    raw_dir=raw_dir,
    manifest_dir=manifest_dir,
    figures_dir=figures_dir,
    seed=42,
)
for label, path in figure_paths.items():
    display(Markdown(f'### {label.replace("_", " ").title()}'))
    display(NotebookImage(filename=str(path)))"""),
    md_cell("## 4. Image dimensions and aspect ratios"),
    code_cell("""dimension_analysis = analyze_image_dimensions(raw_dir, manifest_dir)
dimension_summary = {
    key: value
    for key, value in dimension_analysis.items()
    if key not in {'records'}
}
print(json.dumps(dimension_summary, indent=2))"""),
    md_cell("## 5. Data-quality and leakage findings"),
    code_cell("""quality = summarize_data_quality(
    validation_report_path=validation_report,
    source_lock_path=project_root / 'data/plantvillage_source.lock.json',
    manifest_dir=manifest_dir,
)
print(json.dumps(quality, indent=2))"""),
    code_cell("""duplicates = quality['duplicates']
integrity = quality['integrity']
grouping = quality['leaf_grouping']

display(Markdown(f'''## Evidence-based findings

- All **{integrity['readable_images']:,}** source images are readable; there are **{integrity['empty_files']}** empty and **{integrity['corrupted_files']}** corrupted files.
- Every source image is 256×256. One readable RGBA image is converted deterministically to RGB by the loader.
- The training split is imbalanced: **{training_distribution['largest_class']['class_name']}** has **{training_distribution['largest_class']['count']:,}** images, while **{training_distribution['smallest_class']['class_name']}** has **{training_distribution['smallest_class']['count']:,}** ({training_distribution['imbalance_ratio']:.2f}:1).
- The source audit found **{duplicates['exact_cross_split_groups']}** exact duplicate groups crossing the official train/test boundary and **{duplicates['perceptual_cross_split_pairs']}** perceptual candidates requiring review.
- Human review classified **{duplicates['review_decisions']['exclude_train_related']}** candidates as related and **{duplicates['review_decisions']['keep_both_false_positive']}** as false positives. The resulting evidence excludes **{duplicates['unique_training_images_excluded']}** unique training images while preserving the official test set.
- Upstream metadata maps **{grouping['mapped_images']:,}** images to resolved physical-leaf groups ({grouping['coverage_fraction']:.2%} coverage). Frozen manifests keep resolved groups within one partition.
- PlantVillage uses controlled backgrounds and centered leaves. Results may not generalize to cluttered field photographs, varied lighting, occlusion, multiple leaves, or unseen diseases.
'''))""")
])

# 2. Preprocessing
nb2 = make_notebook([
    md_cell("# 🔄 02. Preprocessing, Data Augmentation & Frozen Splits\nLoads the checksum-verified grouped train/validation/test manifests and demonstrates the torchvision transform pipelines."),
    code_cell("""import os, sys
sys.path.append('..')
from src.data.dataset_loader import get_dataloaders
from src.data.preprocessing import get_transforms

train_loader, val_loader, test_loader, classes, class_to_idx = get_dataloaders(
    raw_dir='../data/raw/plantvillage/color',
    manifest_dir='../data/splits',
    batch_size=32,
    image_size=(224, 224),
    random_seed=42
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
print("Class categories:", classes)""")
])

# 3. Custom CNN
nb3 = make_notebook([
    md_cell("# 🤖 03. Custom CNN Baseline Model\nImplements 3-block convolutional baseline (Conv2D -> BatchNorm -> ReLU -> MaxPool) with Global Average Pooling."),
    code_cell("""import json, os, sys, torch
from pathlib import Path

sys.path.append('..')
from src.models.custom_cnn import CustomCNN
from src.utils.helpers import count_parameters

mapping = json.loads(Path('../data/splits/class_mapping.json').read_text(encoding='utf-8'))
model = CustomCNN(num_classes=len(mapping['classes']))
total_p, train_p = count_parameters(model)
print(f"Custom CNN Total Parameters: {total_p:,} | Trainable: {train_p:,}")

dummy_x = torch.randn(2, 3, 224, 224)
out = model(dummy_x)
print("Output logits shape:", out.shape)""")
])

# 4. ResNet50
nb4 = make_notebook([
    md_cell("# 🧠 04. ResNet50 Transfer Learning & Fine-Tuning\nTransfer learning with residual skip-connections. Evaluates frozen feature extraction vs fine-tuning layer4."),
    code_cell("""import json, os, sys, torch
from pathlib import Path

sys.path.append('..')
from src.models.resnet50 import get_resnet50, unfreeze_resnet50_layers
from src.utils.helpers import count_parameters

mapping = json.loads(Path('../data/splits/class_mapping.json').read_text(encoding='utf-8'))
model = get_resnet50(num_classes=len(mapping['classes']), pretrained=True, freeze_base=True)
total_p, train_p = count_parameters(model)
print(f"Phase 1 (Frozen) - Total: {total_p:,} | Trainable: {train_p:,}")

unfreeze_resnet50_layers(model, ['layer4'])
total_p, train_p = count_parameters(model)
print(f"Phase 2 (Fine-tuning layer4) - Total: {total_p:,} | Trainable: {train_p:,}")""")
])

# 5. EfficientNetB0
nb5 = make_notebook([
    md_cell("# ⚡ 05. EfficientNetB0 Compound Scaling\nEvaluates EfficientNetB0 balance between computational depth, width, resolution scaling and classification accuracy."),
    code_cell("""import json, os, sys, torch
from pathlib import Path

sys.path.append('..')
from src.models.efficientnet_b0 import get_efficientnet_b0, unfreeze_efficientnet_layers
from src.utils.helpers import count_parameters

mapping = json.loads(Path('../data/splits/class_mapping.json').read_text(encoding='utf-8'))
model = get_efficientnet_b0(num_classes=len(mapping['classes']), pretrained=True, freeze_base=True)
total_p, train_p = count_parameters(model)
print(f"EfficientNetB0 - Total: {total_p:,} | Trainable: {train_p:,}")""")
])

# 6. MobileNetV3
nb6 = make_notebook([
    md_cell("# 📱 06. MobileNetV3 Lightweight Edge Model\nEvaluates inverted residual bottlenecks for resource-constrained edge / mobile deployment."),
    code_cell("""import json, os, sys, torch
from pathlib import Path

sys.path.append('..')
from src.models.mobilenet_v3 import get_mobilenet_v3
from src.utils.helpers import count_parameters

mapping = json.loads(Path('../data/splits/class_mapping.json').read_text(encoding='utf-8'))
model = get_mobilenet_v3(num_classes=len(mapping['classes']), pretrained=True, freeze_base=True)
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
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {path}")
