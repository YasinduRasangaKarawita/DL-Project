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
    code_cell("""print('Running source audit: validate')
previous_report_mtime = (
    validation_report.stat().st_mtime_ns if validation_report.exists() else None
)
source_audit = subprocess.run(
    [sys.executable, 'scripts/prepare_data.py', 'validate'],
    cwd=project_root,
    check=False,
)
if not validation_report.is_file():
    raise RuntimeError('Source audit did not create its validation report')
if validation_report.stat().st_mtime_ns == previous_report_mtime:
    raise RuntimeError('Source audit did not refresh its validation report')

source_report = json.loads(validation_report.read_text(encoding='utf-8'))
expected_source_failures = {'exact duplicate groups crossing train/test: 5'}
observed_source_failures = set(source_report['summary']['hard_failures'])
if source_audit.returncode != 1 or observed_source_failures != expected_source_failures:
    raise RuntimeError(
        'Source audit produced an unexpected result: '
        f'exit={source_audit.returncode}, failures={sorted(observed_source_failures)}'
    )
print('Expected source issue confirmed; checking reviewed remediation evidence.')

for command in ('validate-review', 'validate-exclusions', 'validate-manifests'):
    print(f'Running remediation check: {command}')
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
    md_cell("""# 🔄 02. Preprocessing, Data Augmentation & Frozen Splits

This notebook verifies the configuration-driven preprocessing contract against the checksum-verified frozen manifests. Random augmentation is applied only to training images; validation and test preprocessing remains deterministic. Only a training image is displayed, so the locked test set is not inspected."""),
    code_cell("""import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from IPython.display import Markdown, display
from PIL import Image
from torchvision import transforms

project_root = Path.cwd().resolve()
if not (project_root / 'src').is_dir():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import get_dataloaders
from src.data.preprocessing import denormalize_image, get_transforms
from src.utils.helpers import load_yaml_config

config = load_yaml_config(str(project_root / 'configs/config.yaml'))
dataset_config = config['dataset']
augmentation_config = config['augmentation']
image_size = tuple(dataset_config['image_size'])

transform_pipelines = get_transforms(
    image_size=image_size,
    augmentation_config=augmentation_config,
)

print('Training pipeline:')
print(transform_pipelines['train'])
print('\\nValidation/test pipeline:')
print(transform_pipelines['val'])"""),
    md_cell("""## 1. Load the frozen splits

The loader independently verifies the manifest checksum bundle, class mapping, split counts, path separation, and physical-leaf group separation before constructing datasets."""),
    code_cell("""train_loader, val_loader, test_loader, classes, class_to_idx = get_dataloaders(
    raw_dir=project_root / dataset_config['raw_dir'],
    manifest_dir=project_root / dataset_config['manifest_dir'],
    batch_size=config['training']['batch_size'],
    image_size=image_size,
    random_seed=config['project']['random_seed'],
    num_workers=dataset_config['num_workers'],
    augmentation_config=augmentation_config,
)

print(f'Classes: {len(classes)}')
print(
    f'Images — train: {len(train_loader.dataset):,}, '
    f'validation: {len(val_loader.dataset):,}, test: {len(test_loader.dataset):,}'
)
print(
    f'Batches — train: {len(train_loader):,}, '
    f'validation: {len(val_loader):,}, test: {len(test_loader):,}'
)"""),
    md_cell("""## 2. Verify the transform contract

These assertions prove that all stochastic operations are restricted to training and that repeated validation/test preprocessing produces identical tensors."""),
    code_cell("""random_transform_types = (
    transforms.RandomHorizontalFlip,
    transforms.RandomRotation,
    transforms.RandomAffine,
    transforms.ColorJitter,
)

for transform_type in random_transform_types:
    assert any(
        isinstance(transform, transform_type)
        for transform in transform_pipelines['train'].transforms
    )
    assert not any(
        isinstance(transform, transform_type)
        for transform in transform_pipelines['val'].transforms
    )
    assert not any(
        isinstance(transform, transform_type)
        for transform in transform_pipelines['test'].transforms
    )

training_image_path = Path(train_loader.dataset.image_paths[0])
with Image.open(training_image_path) as image_file:
    training_image = image_file.convert('RGB')

validation_once = transform_pipelines['val'](training_image)
validation_twice = transform_pipelines['val'](training_image)
test_once = transform_pipelines['test'](training_image)

assert torch.equal(validation_once, validation_twice)
assert torch.equal(validation_once, test_once)
assert validation_once.shape == (3, *image_size)
print('Passed: augmentation is training-only and evaluation preprocessing is deterministic.')"""),
    md_cell("""## 3. Visualize training-only augmentation

The examples below use fixed seeds for reproducibility. They demonstrate possible training views of one source image alongside its deterministic evaluation view."""),
    code_cell("""mean = augmentation_config['normalization']['mean']
std = augmentation_config['normalization']['std']

figure, axes = plt.subplots(1, 6, figsize=(18, 4))
axes[0].imshow(training_image)
axes[0].set_title('Original training image')

for index in range(4):
    torch.manual_seed(config['project']['random_seed'] + index)
    augmented = transform_pipelines['train'](training_image)
    axes[index + 1].imshow(denormalize_image(augmented, mean=mean, std=std))
    axes[index + 1].set_title(f'Augmented {index + 1}')

axes[5].imshow(denormalize_image(validation_once, mean=mean, std=std))
axes[5].set_title('Validation/test')

for axis in axes:
    axis.axis('off')
figure.tight_layout()
plt.show()"""),
    code_cell("""display(Markdown(f'''## Verified preprocessing summary

- Input shape: **3 × {image_size[0]} × {image_size[1]}** RGB.
- Training augmentation: horizontal flip, rotation, zoom, brightness, and contrast jitter.
- Validation/test: deterministic resize, center crop, tensor conversion, and normalization only.
- Normalization mean: **{augmentation_config['normalization']['mean']}**.
- Normalization standard deviation: **{augmentation_config['normalization']['std']}**.
- Frozen classes: **{len(classes)}**, with one shared class-to-index mapping across all partitions.
'''))""")
])

# 3. Custom CNN
nb3 = make_notebook([
    md_cell("""# 🤖 03. Custom CNN Baseline Architecture

This notebook documents and verifies the from-scratch Custom CNN baseline. It builds the model from the authoritative experiment configuration and frozen 38-class mapping, then records every stage's operation, output shape, and parameter count. No training or test-set evaluation is performed here."""),
    code_cell("""import json
import sys
from pathlib import Path

import pandas as pd
import torch
from IPython.display import Markdown, display

project_root = Path.cwd().resolve()
if not (project_root / 'src').is_dir():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.models.custom_cnn import build_custom_cnn
from src.utils.helpers import count_parameters, load_yaml_config

config = load_yaml_config(str(project_root / 'configs/config.yaml'))
model_config = config['models']['custom_cnn']
image_size = tuple(config['dataset']['image_size'])

mapping = json.loads(
    (project_root / 'data/splits/class_mapping.json').read_text(encoding='utf-8')
)
classes = mapping['classes']
class_to_index = mapping['class_to_index']
assert class_to_index == {class_name: index for index, class_name in enumerate(classes)}

model = build_custom_cnn(
    num_classes=len(classes),
    model_config=model_config,
)
model.eval()

total_parameters, trainable_parameters = count_parameters(model)
assert len(classes) == 38
assert total_parameters == 136_262
assert trainable_parameters == total_parameters

print(f'Classes: {len(classes)}')
print(f'Total parameters: {total_parameters:,}')
print(f'Trainable parameters: {trainable_parameters:,}')"""),
    md_cell("""## Verified layer-by-layer architecture

Forward hooks capture the actual tensor produced by each stage. This prevents the reported shapes from drifting away from the implementation."""),
    code_cell("""stage_records = []

def capture_stage(stage_name, operation):
    def hook(module, inputs, output):
        stage_records.append({
            'Stage': stage_name,
            'Operation': operation,
            'Output shape': ' × '.join(str(value) for value in output.shape),
            'Parameters': sum(parameter.numel() for parameter in module.parameters()),
        })
    return hook

channels = model_config['conv_channels']
dense_units = model_config['dense_units']
dropout_rate = model_config['dropout_rate']
stage_specs = [
    (model.block1, 'Conv block 1', f'Conv 3→{channels[0]}, BatchNorm, ReLU, MaxPool'),
    (model.block2, 'Conv block 2', f'Conv {channels[0]}→{channels[1]}, BatchNorm, ReLU, MaxPool'),
    (model.block3, 'Conv block 3', f'Conv {channels[1]}→{channels[2]}, BatchNorm, ReLU, MaxPool'),
    (model.global_pool, 'Global pool', 'Adaptive average pooling to 1×1'),
    (model.classifier[1], 'Dense', f'Linear {channels[2]}→{dense_units}'),
    (model.classifier[2], 'Dense activation', 'ReLU'),
    (model.classifier[3], 'Regularization', f'Dropout p={dropout_rate}'),
    (model.classifier[4], 'Output', f'Linear {dense_units}→{len(classes)} logits'),
]
handles = [
    module.register_forward_hook(capture_stage(stage_name, operation))
    for module, stage_name, operation in stage_specs
]

dummy_input = torch.zeros(1, 3, *image_size)
with torch.no_grad():
    logits = model(dummy_input)

for handle in handles:
    handle.remove()

architecture = pd.DataFrame([
    {
        'Stage': 'Input',
        'Operation': 'RGB image tensor',
        'Output shape': ' × '.join(str(value) for value in dummy_input.shape),
        'Parameters': 0,
    },
    *stage_records,
])

assert logits.shape == (1, len(classes))
assert architecture['Parameters'].sum() == total_parameters
display(architecture)"""),
    code_cell("""display(Markdown(f'''## Verified baseline facts

- The network is initialized **from scratch**; it has no pretrained backbone.
- Input: **3 × {image_size[0]} × {image_size[1]}** RGB tensor.
- Feature extractor: three Conv2d → BatchNorm → ReLU → MaxPool blocks with channels **{channels}**.
- Head: adaptive global average pooling, **{dense_units}** dense units, ReLU, and dropout **p={dropout_rate}**.
- Output: **{len(classes)}** unnormalized class logits in the frozen class order.
- Parameters: **{total_parameters:,} total**, all trainable.
'''))""")
])

# 4. ResNet50
nb4 = make_notebook([
    md_cell("""# 🧠 04. ResNet-50 Transfer Learning & Residual Bottlenecks

This notebook documents and verifies the **ResNet-50** deep residual network architecture for plant disease classification.
It evaluates ImageNet-1K transfer learning under the frozen experimental protocol:
- **Phase 1 (Feature Extraction):** Backbone convolutional weights are frozen; only the custom classification head is trained.
- **Phase 2 (Fine-Tuning):** The top residual stage (`layer4`) is unfrozen with a reduced learning rate to adapt high-level domain representations.

The notebook verifies exact parameter allocations, tensor shapes through forward hooks, and training interface commands."""),
    code_cell("""import json
import sys
from pathlib import Path
import pandas as pd
import torch
from IPython.display import Markdown, display

project_root = Path.cwd().resolve()
if not (project_root / 'src').is_dir():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.models.resnet50 import get_resnet50, unfreeze_resnet50_layers
from src.utils.helpers import count_parameters, load_yaml_config

config = load_yaml_config(str(project_root / 'configs/config.yaml'))
model_config = config['models']['resnet50']
image_size = tuple(config['dataset']['image_size'])

mapping = json.loads((project_root / 'data/splits/class_mapping.json').read_text(encoding='utf-8'))
classes = mapping['classes']
print(f'Total Classes: {len(classes)}')
print(f'Configured Dropout: {model_config["dropout_rate"]}')
print(f'Fine-tune Unfreeze Layers: {model_config["fine_tune_unfreeze_layers"]}')"""),
    md_cell("""## 1. Phase 1 Model Construction (Frozen Feature Extractor)

We initialize ResNet-50 with pretrained ImageNet weights. The base layers are frozen, and the final fully connected layer is replaced with a custom dropout regularizer and 38-class linear projection."""),
    code_cell("""model = get_resnet50(
    num_classes=len(classes),
    pretrained=False,
    freeze_base=True,
    dropout_rate=model_config['dropout_rate'],
)
model.eval()

total_p, trainable_p = count_parameters(model)
print(f'Phase 1 — Total Parameters:     {total_p:,}')
print(f'Phase 1 — Trainable Parameters: {trainable_p:,} ({trainable_p / total_p * 100:.2f}%)')
print(f'Phase 1 — Frozen Parameters:    {total_p - trainable_p:,}')

assert total_p == 23_585_894
assert trainable_p == 77_862"""),
    md_cell("""## 2. Layer-by-Layer Architectural Audit

Using forward hooks, we record the exact output dimensions and parameter counts of each major stage in the ResNet-50 backbone and classifier."""),
    code_cell("""stage_records = []

def capture_stage(stage_name, operation):
    def hook(module, inputs, output):
        stage_records.append({
            'Stage': stage_name,
            'Operation': operation,
            'Output Shape': ' × '.join(str(v) for v in output.shape),
            'Parameters': sum(p.numel() for p in module.parameters()),
        })
    return hook

stage_specs = [
    (model.conv1, 'Stem Conv', 'Conv2d 7×7, stride=2, padding=3 (3→64)'),
    (model.bn1, 'Stem BatchNorm', 'BatchNorm2d(64) + ReLU'),
    (model.maxpool, 'Stem MaxPool', 'MaxPool2d 3×3, stride=2, padding=1'),
    (model.layer1, 'ResNet Layer 1', '3 Bottleneck Blocks (64→256 ch)'),
    (model.layer2, 'ResNet Layer 2', '4 Bottleneck Blocks (256→512 ch, stride=2)'),
    (model.layer3, 'ResNet Layer 3', '6 Bottleneck Blocks (512→1024 ch, stride=2)'),
    (model.layer4, 'ResNet Layer 4', '3 Bottleneck Blocks (1024→2048 ch, stride=2)'),
    (model.avgpool, 'Global Average Pool', 'AdaptiveAvgPool2d((1, 1))'),
    (model.fc[0], 'Regularization', f'Dropout(p={model_config["dropout_rate"]})'),
    (model.fc[1], 'Classification Head', f'Linear(2048 → {len(classes)})'),
]

handles = [module.register_forward_hook(capture_stage(name, op)) for module, name, op in stage_specs]

dummy_input = torch.zeros(1, 3, *image_size)
with torch.no_grad():
    logits = model(dummy_input)

for handle in handles:
    handle.remove()

architecture = pd.DataFrame([
    {
        'Stage': 'Input',
        'Operation': 'RGB Image Tensor',
        'Output Shape': ' × '.join(str(v) for v in dummy_input.shape),
        'Parameters': 0,
    },
    *stage_records,
])

assert logits.shape == (1, len(classes))
display(architecture)"""),
    md_cell("""## 3. Phase 2 Fine-Tuning: Unfreezing Layer 4

In Phase 2, we unfreeze `layer4` (the final 3 bottleneck residual blocks) to allow high-level visual representations to adapt to plant pathology morphology."""),
    code_cell("""unfreeze_resnet50_layers(model, model_config['fine_tune_unfreeze_layers'])

total_p_ft, trainable_p_ft = count_parameters(model)
print(f'Phase 2 — Total Parameters:     {total_p_ft:,}')
print(f'Phase 2 — Trainable Parameters: {trainable_p_ft:,} ({trainable_p_ft / total_p_ft * 100:.2f}%)')
print(f'Phase 2 — Frozen Parameters:    {total_p_ft - trainable_p_ft:,}')

assert total_p_ft == 23_585_894
assert trainable_p_ft == 15_042_598"""),
    md_cell("""## 4. Training Command Reference

To execute official ResNet-50 training under the reproducible protocol:

```bash
# Full two-phase training (Phase 1: 15 epochs, Phase 2: 10 epochs fine-tune)
python run_pipeline.py train --model resnet50 --seed 42

# Single-epoch pilot check
python run_pipeline.py train --model resnet50 --seed 42 --epochs 1 --allow-dirty

# Evaluate best validation checkpoint
python run_pipeline.py evaluate --checkpoint models/resnet50/<run-id>/best_inference.pt --split validation
```""")
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
