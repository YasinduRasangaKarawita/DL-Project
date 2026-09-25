import argparse
import json
import os
import time

import torch
import torch.nn as nn

from src.data.dataset_loader import get_dataloaders
from src.data.download_data import prepare_dataset
from src.data.validate_data import validate_dataset_integrity
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.dataset_analysis import generate_dataset_figures
from src.evaluation.error_analysis import run_error_analysis
from src.evaluation.metrics import evaluate_model_metrics
from src.evaluation.model_comparison import generate_model_comparison
from src.evaluation.training_curves import plot_learning_curves
from src.models.custom_cnn import CustomCNN
from src.models.efficientnet_b0 import get_efficientnet_b0, unfreeze_efficientnet_layers
from src.models.mobilenet_v3 import get_mobilenet_v3, unfreeze_mobilenet_layers
from src.models.resnet50 import get_resnet50, unfreeze_resnet50_layers
from src.training.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.training.experiment_tracker import ExperimentTracker
from src.training.trainer import ModelTrainer
from src.utils.helpers import get_device, load_yaml_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("pipeline_runner")

def run_pipeline(config_path: str = "configs/config.yaml", quick_mode: bool = False):
    logger.info("================================================================")
    logger.info("  🌿 STARTING PLANT DISEASE CLASSIFICATION BENCHMARK PIPELINE  ")
    logger.info("================================================================")

    config = load_yaml_config(config_path)
    seed = config["project"].get("random_seed", 42)
    set_seed(seed)

    device = get_device(config["project"].get("device", "auto"))
    logger.info(f"Using Compute Device: {device}")

    # 1. Dataset Preparation & Validation
    raw_dir = config["dataset"]["raw_dir"]
    samples_per_class = 20 if quick_mode else config["dataset"].get("benchmark_samples_per_class", 60)

    logger.info("Step 1: Preparing PlantVillage Dataset...")
    classes = prepare_dataset(raw_dir=raw_dir, samples_per_class=samples_per_class)
    num_classes = len(classes)
    logger.info(f"Dataset ready with {num_classes} classes.")

    logger.info("Step 2: Validating Dataset Integrity...")
    validate_dataset_integrity(raw_dir=raw_dir)

    logger.info("Step 3: Generating Exploratory Data Analysis Figures...")
    generate_dataset_figures(raw_dir=raw_dir, figures_dir="figures/dataset")

    # 2. Stratified Data Split and Loaders
    batch_size = 16 if quick_mode else config["training"].get("batch_size", 32)
    img_size = tuple(config["dataset"].get("image_size", [224, 224]))

    logger.info("Step 4: Loading frozen grouped data manifests...")
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(
        raw_dir=raw_dir,
        manifest_dir=config["dataset"].get("manifest_dir", "data/splits"),
        batch_size=batch_size,
        image_size=img_size,
        random_seed=seed,
        num_workers=0,
        augmentation_config=config.get("augmentation", {})
    )

    test_paths = list(test_loader.dataset.image_paths)

    epochs_init = 2 if quick_mode else config["training"].get("initial_epochs", 10)
    epochs_fine = 1 if quick_mode else config["training"].get("fine_tune_epochs", 5)

    # Models definition map
    model_configs = [
        {
            "name": "Custom_CNN",
            "builder": lambda: CustomCNN(num_classes=num_classes, dropout_rate=config["models"]["custom_cnn"]["dropout_rate"]),
            "save_path": config["models"]["custom_cnn"]["save_path"],
            "fine_tune": False
        },
        {
            "name": "ResNet50",
            "builder": lambda: get_resnet50(num_classes=num_classes, pretrained=True, freeze_base=True, dropout_rate=config["models"]["resnet50"]["dropout_rate"]),
            "save_path": config["models"]["resnet50"]["save_path"],
            "fine_tune": True,
            "unfreeze_fn": lambda m: unfreeze_resnet50_layers(m, ["layer4"])
        },
        {
            "name": "EfficientNet_B0",
            "builder": lambda: get_efficientnet_b0(num_classes=num_classes, pretrained=True, freeze_base=True, dropout_rate=config["models"]["efficientnet_b0"]["dropout_rate"]),
            "save_path": config["models"]["efficientnet_b0"]["save_path"],
            "fine_tune": True,
            "unfreeze_fn": lambda m: unfreeze_efficientnet_layers(m, ["7", "8"])
        },
        {
            "name": "MobileNet_V3",
            "builder": lambda: get_mobilenet_v3(num_classes=num_classes, pretrained=True, freeze_base=True, dropout_rate=config["models"]["mobilenet_v3"]["dropout_rate"]),
            "save_path": config["models"]["mobilenet_v3"]["save_path"],
            "fine_tune": True,
            "unfreeze_fn": lambda m: unfreeze_mobilenet_layers(m, ["14", "15", "16"])
        }
    ]

    # Training and Evaluation Loop
    for m_cfg in model_configs:
        m_name = m_cfg["name"]
        save_path = m_cfg["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        logger.info("\n==================================================")
        logger.info(f"  Training & Evaluating Architecture: {m_name}")
        logger.info("==================================================")

        model = m_cfg["builder"]().to(device)
        criterion = nn.CrossEntropyLoss()

        lr_init = config["training"].get("learning_rate", 0.001)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_init,
            weight_decay=config["training"].get("weight_decay", 1e-4)
        )

        csv_log_path = f"results/experiments/{m_name.lower()}_history.csv"
        csv_logger = CSVLogger(csv_log_path)
        checkpoint = ModelCheckpoint(save_path, monitor="val_loss", mode="min")
        early_stop = EarlyStopping(patience=config["training"].get("early_stopping_patience", 4), mode="min")
        reduce_lr = ReduceLROnPlateau(optimizer, patience=config["training"].get("reduce_lr_patience", 2), factor=0.5)

        trainer = ModelTrainer(model, optimizer, criterion, device=device)

        start_time = time.time()
        # Phase 1: Feature Extraction / Initial Training
        logger.info(f"[{m_name}] Phase 1: Training base network for {epochs_init} epochs...")
        trainer.fit(
            train_loader, val_loader,
            epochs=epochs_init,
            checkpoint_cb=checkpoint,
            early_stopping_cb=early_stop,
            reduce_lr_cb=reduce_lr,
            csv_logger_cb=csv_logger,
            start_epoch=1
        )

        # Phase 2: Fine-Tuning (if applicable)
        if m_cfg.get("fine_tune", False) and epochs_fine > 0:
            logger.info(f"[{m_name}] Phase 2: Fine-tuning upper layers with lower learning rate...")
            m_cfg["unfreeze_fn"](model)
            lr_fine = config["training"].get("fine_tune_learning_rate", 0.0001)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_fine,
                weight_decay=config["training"].get("weight_decay", 1e-4)
            )
            trainer.optimizer = optimizer
            trainer.fit(
                train_loader, val_loader,
                epochs=epochs_fine,
                checkpoint_cb=checkpoint,
                early_stopping_cb=early_stop,
                reduce_lr_cb=reduce_lr,
                csv_logger_cb=csv_logger,
                start_epoch=epochs_init + 1
            )

        total_training_sec = time.time() - start_time

        # Plot Learning Curves
        curves_path = f"figures/training/{m_name.lower()}_learning_curves.png"
        plot_learning_curves(csv_log_path, m_name, curves_path)

        # Load best checkpoint weights for final unseen test evaluation
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded best checkpoint from {save_path} for final evaluation.")

        # Unseen Test Set Evaluation
        logger.info(f"Evaluating {m_name} on UNSEEN test set...")
        metrics, targets, preds, probs, latency_ms = evaluate_model_metrics(
            model, test_loader, class_names, device
        )

        logger.info(
            f"[{m_name} Test Results] Acc: {metrics['accuracy']*100:.2f}% | "
            f"Top-3 Acc: {metrics['top3_accuracy']*100:.2f}% | "
            f"F1-Macro: {metrics['macro_f1']:.4f} | Latency: {latency_ms:.2f}ms/img"
        )

        # Save Metrics JSON
        os.makedirs("results/metrics", exist_ok=True)
        with open(f"results/metrics/{m_name.lower()}_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Generate Confusion Matrix
        cm_path = f"figures/evaluation/{m_name.lower()}_confusion_matrix.png"
        plot_confusion_matrix(targets, preds, class_names, model_name=m_name, save_path=cm_path)

        # Qualitative & Quantitative Error Analysis
        run_error_analysis(
            targets, preds, probs, test_paths, class_names,
            model_name=m_name, output_dir="results/predictions"
        )

        # Record Experiment Metadata
        tracker = ExperimentTracker(experiment_name=m_name, base_dir="results/experiments")
        tracker.record_run(
            model=model,
            model_save_path=save_path,
            hyperparameters={
                "batch_size": batch_size,
                "initial_epochs": epochs_init,
                "fine_tune_epochs": epochs_fine,
                "learning_rate": lr_init,
                "image_size": list(img_size)
            },
            training_time_seconds=total_training_sec,
            inference_time_ms=latency_ms,
            test_metrics=metrics
        )

    # Step 5: Final Cross-Model Comparison
    logger.info("\nStep 5: Generating Comparative Analysis & Comparison Table...")
    comparison_df = generate_model_comparison()
    logger.info("\n" + comparison_df.to_string(index=False))

    logger.info("================================================================")
    logger.info("  🎉 PLANT DISEASE BENCHMARK PIPELINE COMPLETED SUCCESSFULLY!   ")
    logger.info("================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--quick-run", action="store_true", help="Execute fast benchmark validation pass")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, quick_mode=args.quick_run)
