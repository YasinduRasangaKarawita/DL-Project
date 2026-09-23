# 🌿 Deep Learning-Based Plant Disease Classification

> **A Comparative Evaluation of Deep Learning Architectures for Plant Disease Classification Using the PlantVillage Dataset**

---

## 📌 Project Overview

This project develops and evaluates multiple Deep Learning architectures for automatically identifying plant diseases from leaf images.

The main objective is not simply to achieve the highest classification accuracy. The project will conduct a **fair comparative evaluation** of different Deep Learning architectures based on:

- Classification performance
- Generalization ability
- Training stability
- Computational efficiency
- Model complexity
- Training time
- Error patterns and class confusion
- Practical suitability for real-world deployment

The system will classify plant leaf images into multiple disease and healthy categories using the **PlantVillage dataset**.

The project will implement and compare the following four Deep Learning models:

1. Custom Convolutional Neural Network (CNN)
2. ResNet50
3. EfficientNetB0
4. MobileNetV3

The final system will include a simple demonstration interface where a user can upload a plant leaf image and receive a predicted disease class and confidence score.

---

# 🎯 Research Problem

Plant diseases can significantly affect agricultural productivity. Early identification of plant diseases can help farmers and agricultural stakeholders take appropriate action.

Manual disease identification can require expert knowledge and may not always be easily accessible.

This project investigates whether different Deep Learning architectures can accurately classify plant diseases from leaf images and examines the trade-offs between predictive performance and computational efficiency.

---

# ❓ Research Question

> **How does the choice of Deep Learning architecture affect classification performance, generalization, computational efficiency, and model complexity in plant disease recognition?**

---

# 🎯 Project Objectives

## Primary Objective

Develop and compare multiple Deep Learning architectures for multi-class plant disease classification.

## Specific Objectives

1. Explore and analyze the PlantVillage dataset.
2. Identify dataset quality issues and class distributions.
3. Apply appropriate preprocessing and image augmentation techniques.
4. Implement a Custom CNN as a Deep Learning baseline.
5. Implement ResNet50 using transfer learning.
6. Implement EfficientNetB0 using transfer learning.
7. Implement MobileNetV3 as a lightweight architecture.
8. Evaluate all models using the same experimental framework.
9. Compare models based on performance, generalization, efficiency, and complexity.
10. Perform detailed error analysis using confusion matrices and class-level metrics.
11. Identify the most suitable architecture for practical plant disease classification.

---

# 📊 Dataset

## Dataset Name

**PlantVillage Dataset**

The dataset contains images of healthy and diseased plant leaves.

The dataset includes multiple plant species and disease categories.

Each image belongs to a specific plant-disease class.

Example classes may include:

```text
Apple___Apple_scab
Apple___Black_rot
Apple___healthy

Corn___Common_rust
Corn___Northern_Leaf_Blight
Corn___healthy

Grape___Black_rot
Grape___healthy

Potato___Early_blight
Potato___Late_blight
Potato___healthy

Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___healthy
```

> The exact class list must be generated programmatically from the downloaded dataset to ensure consistency.

---

# 🧠 Project Architecture

```text
                         ┌─────────────────────┐
                         │ PlantVillage Dataset │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Validation     │
                         │ • Missing images    │
                         │ • Corrupted files   │
                         │ • Class validation  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Exploratory Data    │
                         │ Analysis            │
                         │                     │
                         │ • Class distribution│
                         │ • Image dimensions  │
                         │ • Sample images     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Preprocessing  │
                         │                     │
                         │ • Resize            │
                         │ • Normalization     │
                         │ • Encoding          │
                         │ • Augmentation      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Train / Validation / Test Split│
                    └───────────────┬───────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
      ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
      │ Training Data │     │ Validation    │     │ Test Data     │
      │               │     │ Data          │     │ UNSEEN        │
      └───────────────┘     └───────────────┘     └───────────────┘
              │                     │                     │
              │                     │                     │
              ▼                     ▼                     │
     ┌───────────────────────────────────────────┐       │
     │       DEEP LEARNING MODEL COMPARISON      │       │
     └───────────────────────────────────────────┘       │
              │
     ┌────────┼─────────┬──────────────┬─────────┐
     │        │         │              │         │
     ▼        ▼         ▼              ▼
 ┌────────┐ ┌────────┐ ┌────────────┐ ┌────────────┐
 │Custom  │ │ResNet50│ │EfficientNet│ │MobileNetV3 │
 │ CNN    │ │        │ │    B0      │ │            │
 └────┬───┘ └────┬───┘ └─────┬──────┘ └─────┬──────┘
      │          │           │              │
      └──────────┴───────────┴──────────────┘
                         │
                         ▼
              ┌────────────────────────┐
              │ Final Test Evaluation  │
              └────────────┬───────────┘
                           │
                           ▼
          ┌─────────────────────────────────┐
          │ Comparative Evaluation          │
          │                                 │
          │ • Accuracy                      │
          │ • Precision                     │
          │ • Recall                        │
          │ • F1-score                      │
          │ • ROC-AUC                       │
          │ • Confusion Matrix              │
          │ • Training Time                 │
          │ • Parameter Count               │
          │ • Model Size                    │
          └───────────────┬─────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Critical Analysis   │
               │                     │
               │ • Generalization    │
               │ • Overfitting       │
               │ • Convergence       │
               │ • Efficiency        │
               │ • Error Analysis    │
               │ • Limitations       │
               └─────────────────────┘
```

---

# 🧪 Experimental Design

All models must be evaluated under fair and comparable experimental conditions.

## Data Split

Use a stratified split.

Recommended:

```text
Training Set:     70%
Validation Set:   15%
Test Set:         15%
```

Requirements:

- The test dataset must remain completely unseen during training.
- Hyperparameter tuning must use only training and validation data.
- The same dataset split must be used for all models.
- Use a fixed random seed.
- Save split information for reproducibility.

Example:

```python
RANDOM_SEED = 42
```

---

# 🖼 Image Preprocessing

All models should receive comparable input data.

## Image Size

Recommended initial size:

```text
224 × 224 × 3
```

This is compatible with:

- ResNet50
- EfficientNetB0
- MobileNetV3

## Preprocessing Steps

```text
Raw Image
    │
    ▼
Validate Image
    │
    ▼
Resize to 224 × 224
    │
    ▼
Convert to RGB
    │
    ▼
Normalize Pixels
    │
    ▼
Training Augmentation
```

---

# 🔄 Data Augmentation

Apply augmentation only to the training dataset.

Recommended transformations:

- Random horizontal flip
- Random rotation
- Random zoom
- Random translation
- Random contrast adjustment

Example configuration:

```text
Rotation: ±20°
Zoom: 10–20%
Horizontal Flip: Enabled
Translation: Small random shift
```

Do not apply random augmentation to validation or test data.

---

# 🤖 Model 1 — Custom CNN

The Custom CNN will serve as a Deep Learning baseline.

## Proposed Architecture

```text
Input Image
     │
Conv2D → BatchNorm → ReLU
     │
MaxPooling
     │
Conv2D → BatchNorm → ReLU
     │
MaxPooling
     │
Conv2D → BatchNorm → ReLU
     │
MaxPooling
     │
GlobalAveragePooling
     │
Dense
     │
Dropout
     │
Softmax Output
```

## Suggested Initial Configuration

```text
Optimizer: Adam
Learning Rate: 0.001
Loss: Categorical Crossentropy
Batch Size: 32
Epochs: 30–50
Early Stopping: Enabled
```

---

# 🧠 Model 2 — ResNet50

ResNet50 will be implemented using transfer learning.

## Phase 1

```text
Pre-trained ResNet50
        │
Freeze Base Layers
        │
Global Average Pooling
        │
Dropout
        │
Dense Output Layer
```

## Phase 2

Fine-tune selected upper layers.

Compare:

```text
Frozen Feature Extraction
        vs
Fine-Tuning
```

Record whether fine-tuning improves validation performance or causes overfitting.

---

# ⚡ Model 3 — EfficientNetB0

EfficientNetB0 is selected because it provides a balance between:

- Model complexity
- Computational efficiency
- Classification accuracy

Architecture:

```text
Input
  │
EfficientNetB0
  │
Global Average Pooling
  │
Dropout
  │
Dense Softmax
```

Experiment with:

```text
Frozen Base Model
        ↓
Partial Fine-Tuning
```

---

# 📱 Model 4 — MobileNetV3

MobileNetV3 represents a lightweight architecture suitable for resource-constrained environments.

Architecture:

```text
Input
  │
MobileNetV3
  │
Global Average Pooling
  │
Dropout
  │
Dense Softmax
```

Important analysis:

```text
Does MobileNetV3 sacrifice accuracy
in exchange for lower computational cost?
```

Compare:

- Accuracy
- F1-score
- Parameter count
- Model size
- Training time
- Inference time

---

# ⚙️ Training Configuration

Create one central configuration file.

Example:

```python
RANDOM_SEED = 42

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 20

LEARNING_RATE = 0.001

NUM_CLASSES = "AUTO_DETECT"
```

Each experiment should save:

```text
Model name
Random seed
Dataset split
Image size
Batch size
Optimizer
Learning rate
Epochs
Trainable parameters
Total parameters
Training time
Validation results
Test results
```

---

# 🛑 Training Callbacks

Use appropriate callbacks:

```text
EarlyStopping
ReduceLROnPlateau
ModelCheckpoint
CSVLogger
```

Recommended monitoring metric:

```text
Validation Loss
```

Save the best model based on validation performance.

---

# 📈 Evaluation Metrics

Because this is a multi-class classification problem, do not rely only on accuracy.

Calculate:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC (where appropriate)
Confusion Matrix
Per-class Precision
Per-class Recall
Per-class F1-score
```

Also measure:

```text
Training Time
Inference Time
Parameter Count
Model File Size
```

---

# 📊 Required Visualizations

Generate and save the following visualizations.

## Dataset Analysis

```text
Class Distribution
Sample Images
Image Dimension Distribution
Class Balance Analysis
```

## Training Analysis

For each model:

```text
Training Accuracy vs Epoch
Validation Accuracy vs Epoch

Training Loss vs Epoch
Validation Loss vs Epoch
```

## Model Evaluation

```text
Confusion Matrix
Per-Class F1-score
Model Comparison Bar Chart
Training Time Comparison
Parameter Count Comparison
Accuracy vs Model Complexity
```

---

# 🔍 Error Analysis

Do not stop after reporting accuracy.

Perform error analysis.

## Required Questions

For the best and worst models:

1. Which classes are frequently confused?
2. Are visually similar diseases being confused?
3. Which classes have low recall?
4. Which classes have low precision?
5. Are there specific plant species that are harder to classify?
6. Does class imbalance affect performance?
7. Does augmentation improve generalization?
8. Does fine-tuning improve performance?
9. Which model overfits the most?
10. Which model provides the best accuracy-efficiency trade-off?

Save example misclassified images.

Suggested output:

```text
Actual Class
Predicted Class
Confidence Score
Image
```

---

# 🧠 Critical Analysis

The discussion section must go beyond:

> "Model A achieved higher accuracy than Model B."

Instead, investigate:

## Generalization

```text
Training Accuracy
        vs
Validation Accuracy
```

Identify:

- Overfitting
- Underfitting
- Stable learning
- Validation instability

---

## Model Complexity

Compare:

```text
Number of Parameters
Model Size
Training Time
Inference Time
```

Possible discussion:

> A model may achieve the highest classification accuracy but require significantly more computational resources.

---

## Accuracy vs Efficiency

Create a comparison such as:

| Model | Accuracy | F1 | Parameters | Training Time | Model Size |
|---|---:|---:|---:|---:|---:|
| Custom CNN | | | | | |
| ResNet50 | | | | | |
| EfficientNetB0 | | | | | |
| MobileNetV3 | | | | | |

Investigate:

> Which model provides the best practical trade-off between predictive performance and computational efficiency?

---

# 🧪 Recommended Experiments

## Experiment 1 — Baseline

Train Custom CNN.

Goal:

```text
Establish Deep Learning baseline performance.
```

---

## Experiment 2 — Transfer Learning

Train:

```text
ResNet50
EfficientNetB0
MobileNetV3
```

with frozen base layers.

Compare against Custom CNN.

---

## Experiment 3 — Fine-Tuning

Fine-tune selected layers of transfer learning models.

Investigate:

```text
Does fine-tuning improve performance?
Does it increase overfitting?
Does training time increase?
```

---

## Experiment 4 — Data Augmentation

Compare selected models:

```text
Without Augmentation
        vs
With Augmentation
```

Measure:

- Validation accuracy
- Validation loss
- Generalization gap

---

## Experiment 5 — Computational Efficiency

Compare:

```text
Parameter Count
Training Time
Inference Time
Model Size
```

---

# 🖥 Demo Application

Create a lightweight web interface.

Recommended:

```text
Streamlit
```

Alternative:

```text
Flask
FastAPI + React
```

For the assignment deadline, prefer:

> **Streamlit**

## Required Features

### Image Upload

```text
Upload Plant Leaf Image
```

### Prediction

Display:

```text
Predicted Disease
Confidence Score
```

### Top Predictions

Example:

```text
1. Tomato Early Blight — 92%
2. Tomato Late Blight — 6%
3. Tomato Healthy — 2%
```

### Model Selection

Allow:

```text
Select Model

○ Custom CNN
○ ResNet50
○ EfficientNetB0
○ MobileNetV3
```

### Optional Model Information

Display:

```text
Model
Accuracy
F1-score
Parameter Count
Inference Time
```

---

# 📁 Recommended Project Structure

```text
plant-disease-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_custom_cnn.ipynb
│   ├── 04_resnet50.ipynb
│   ├── 05_efficientnet.ipynb
│   ├── 06_mobilenet.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/
│   │
│   ├── data/
│   │   ├── download_data.py
│   │   ├── validate_data.py
│   │   ├── preprocessing.py
│   │   └── dataset_loader.py
│   │
│   ├── models/
│   │   ├── custom_cnn.py
│   │   ├── resnet50.py
│   │   ├── efficientnet_b0.py
│   │   └── mobilenet_v3.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── experiment_tracker.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   ├── error_analysis.py
│   │   └── model_comparison.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logger.py
│       └── helpers.py
│
├── models/
│   ├── custom_cnn/
│   ├── resnet50/
│   ├── efficientnet/
│   └── mobilenet/
│
├── results/
│   ├── metrics/
│   ├── predictions/
│   ├── experiments/
│   └── model_comparison/
│
├── figures/
│   ├── dataset/
│   ├── training/
│   ├── evaluation/
│   └── comparison/
│
├── app/
│   └── streamlit_app.py
│
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_preprocessing.py
```

---

# 👥 Recommended Group Work Distribution

## Member 1 — Dataset and Preprocessing

Responsible for:

```text
Dataset acquisition
Data validation
EDA
Class distribution
Data splitting
Preprocessing
Augmentation
```

GitHub folders:

```text
data/
src/data/
notebooks/01_dataset_exploration.ipynb
notebooks/02_preprocessing.ipynb
```

---

## Member 2 — Custom CNN and ResNet50

Responsible for:

```text
Custom CNN
ResNet50
Training experiments
Hyperparameter configuration
Learning curves
```

GitHub folders:

```text
src/models/custom_cnn.py
src/models/resnet50.py
```

---

## Member 3 — EfficientNetB0 and MobileNetV3

Responsible for:

```text
EfficientNetB0
MobileNetV3
Fine-tuning experiments
Efficiency measurements
```

GitHub folders:

```text
src/models/efficientnet_b0.py
src/models/mobilenet_v3.py
```

---

## Member 4 — Evaluation and Integration

Responsible for:

```text
Metrics
Model comparison
Confusion matrices
Error analysis
Experiment tracking
Streamlit demo
GitHub integration
```

GitHub folders:

```text
src/evaluation/
app/
results/
```

> All members should contribute to report writing, testing, experiments, discussion, and viva preparation.

---

# 🔀 GitHub Workflow

Use branches.

Recommended branches:

```text
main
develop

feature/data-preprocessing
feature/custom-cnn
feature/resnet50
feature/efficientnet
feature/mobilenet
feature/evaluation
feature/streamlit-demo
```

Workflow:

```text
Create Issue
      ↓
Create Feature Branch
      ↓
Develop Feature
      ↓
Commit Changes
      ↓
Push Branch
      ↓
Create Pull Request
      ↓
Review
      ↓
Merge into develop
```

Use meaningful commits.

Good:

```text
feat: implement stratified dataset splitting
feat: add ResNet50 transfer learning model
fix: prevent data leakage in preprocessing
docs: update experiment configuration
test: add image validation tests
```

Avoid:

```text
update
changes
final
test
done
```

---

# 🔁 Reproducibility

Set seeds for:

```text
Python
NumPy
TensorFlow
```

Save:

```text
Random seed
Dataset split
Hyperparameters
Model configuration
Library versions
```

Use:

```text
requirements.txt
```

Example dependencies:

```text
tensorflow
keras
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
Pillow
PyYAML
streamlit
tqdm
```

Optional:

```text
kaggle
mlflow
```

---

# 🚀 Development Plan

## Phase 1 — Project Setup

```text
[ ] Create GitHub repository
[ ] Create project structure
[ ] Create Python virtual environment
[ ] Create requirements.txt
[ ] Configure random seed
[ ] Assign GitHub issues
```

---

## Phase 2 — Dataset

```text
[ ] Download PlantVillage dataset
[ ] Verify dataset source
[ ] Validate images
[ ] Generate class list
[ ] Perform EDA
[ ] Create visualizations
```

---

## Phase 3 — Preprocessing

```text
[ ] Create stratified dataset split
[ ] Prevent data leakage
[ ] Resize images
[ ] Normalize images
[ ] Implement augmentation
[ ] Save preprocessing configuration
```

---

## Phase 4 — Model Development

### Custom CNN

```text
[ ] Design architecture
[ ] Train baseline
[ ] Record metrics
[ ] Save model
```

### ResNet50

```text
[ ] Implement transfer learning
[ ] Train frozen model
[ ] Fine-tune selected layers
[ ] Record metrics
```

### EfficientNetB0

```text
[ ] Implement transfer learning
[ ] Train frozen model
[ ] Fine-tune selected layers
[ ] Record metrics
```

### MobileNetV3

```text
[ ] Implement lightweight architecture
[ ] Train model
[ ] Fine-tune selected layers
[ ] Record metrics
```

---

## Phase 5 — Evaluation

```text
[ ] Final unseen test evaluation
[ ] Accuracy
[ ] Precision
[ ] Recall
[ ] F1-score
[ ] ROC-AUC where appropriate
[ ] Confusion matrices
[ ] Per-class metrics
[ ] Learning curves
[ ] Training time
[ ] Inference time
[ ] Parameter count
[ ] Model size
```

---

## Phase 6 — Critical Analysis

```text
[ ] Analyze overfitting
[ ] Analyze underfitting
[ ] Compare convergence
[ ] Compare generalization
[ ] Analyze misclassified images
[ ] Identify confused classes
[ ] Compare computational efficiency
[ ] Identify best accuracy-efficiency trade-off
[ ] Discuss limitations
[ ] Propose future improvements
```

---

## Phase 7 — Demo

```text
[ ] Select best saved model
[ ] Build Streamlit interface
[ ] Add image upload
[ ] Add prediction
[ ] Add confidence score
[ ] Add top predictions
[ ] Test application
```

---

# 📊 Final Comparison Table

The project must automatically generate a final comparison table.

Example:

| Model | Accuracy | Precision | Recall | F1 | Parameters | Training Time | Inference Time | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | - | - | - | - | - | - | - | - |
| ResNet50 | - | - | - | - | - | - | - | - |
| EfficientNetB0 | - | - | - | - | - | - | - | - |
| MobileNetV3 | - | - | - | - | - | - | - | - |

---

# 📝 Report Mapping

The implementation should directly support the assignment report.

## 1. Introduction and Problem Definition

Include:

```text
Agricultural importance
Plant disease problem
Deep Learning motivation
Research question
Objectives
```

---

## 2. Background and Related Work

Research:

```text
Plant disease detection
CNN architectures
Transfer learning
ResNet
EfficientNet
MobileNet
```

---

## 3. Dataset Description and EDA

Include:

```text
Dataset source
Dataset characteristics
Number of classes
Number of images
Class distribution
Sample images
Data quality issues
```

---

## 4. Data Preprocessing and Feature Engineering

Explain:

```text
Image validation
Data split
Data leakage prevention
Resize
Normalization
Augmentation
```

---

## 5. Experimental Design

Document:

```text
Hardware
Software
Random seed
Dataset split
Hyperparameters
Training configuration
Fair comparison strategy
```

---

## 6. Model Architectures

For every model explain:

```text
Architecture
Layers
Activation functions
Loss function
Optimizer
Learning rate
Transfer learning strategy
Fine-tuning strategy
Hyperparameters
```

---

## 7. Results and Model Comparison

Include:

```text
Learning curves
Metrics table
Confusion matrices
Efficiency comparison
Complexity comparison
```

---

## 8. Critical Analysis and Discussion

Analyze:

```text
Why models perform differently
Overfitting
Underfitting
Generalization
Convergence
Computational cost
Failure cases
Practical limitations
```

---

## 9. Conclusion

Answer:

```text
Which model performed best?
Which model generalized best?
Which model was most efficient?
Which model provides the best practical trade-off?
```

---

# 🏆 Definition of Success

The project should not define success only as achieving the highest accuracy.

A successful project will:

```text
✓ Implement four genuinely distinct Deep Learning architectures

✓ Use a sufficiently complex real-world dataset

✓ Prevent data leakage

✓ Maintain an unseen test dataset

✓ Conduct fair experiments

✓ Use multiple evaluation metrics

✓ Compare accuracy and efficiency

✓ Analyze errors and failure cases

✓ Investigate overfitting and generalization

✓ Provide reproducible experiments

✓ Maintain meaningful GitHub contributions

✓ Provide a working demonstration

✓ Support a strong technical viva
```

---

# ⚠️ Important Development Rules

1. Do not use the test dataset during hyperparameter tuning.
2. Do not change the dataset split between models.
3. Do not compare models trained under unfair conditions.
4. Record every experiment configuration.
5. Save all learning curves.
6. Save model checkpoints.
7. Use fixed random seeds.
8. Measure computational efficiency.
9. Keep meaningful GitHub commits throughout development.
10. Perform analysis beyond simply reporting accuracy.
11. Clearly document all datasets, libraries, pretrained models, and external resources used.
12. Ensure every group member can explain their contribution during the viva.

---

# 🚀 Recommended Execution Order

```text
STEP 1
Project Setup
    ↓
STEP 2
Download + Validate Dataset
    ↓
STEP 3
EDA
    ↓
STEP 4
Train / Validation / Test Split
    ↓
STEP 5
Preprocessing + Augmentation
    ↓
STEP 6
Custom CNN
    ↓
STEP 7
ResNet50
    ↓
STEP 8
EfficientNetB0
    ↓
STEP 9
MobileNetV3
    ↓
STEP 10
Fine-Tuning Experiments
    ↓
STEP 11
Final Unseen Test Evaluation
    ↓
STEP 12
Model Comparison
    ↓
STEP 13
Error Analysis
    ↓
STEP 14
Critical Discussion
    ↓
STEP 15
Streamlit Demo
    ↓
STEP 16
Report + Video + Viva Preparation
```

---

## Final Project Goal

> **Develop a reproducible Deep Learning system that compares four distinct architectures for multi-class plant disease classification and provides an evidence-based analysis of predictive performance, generalization, computational efficiency, model complexity, and practical deployment suitability.**