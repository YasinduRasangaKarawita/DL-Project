import os
import pytest
from src.data.download_data import prepare_dataset
from src.data.validate_data import validate_dataset_integrity
from src.data.dataset_loader import get_dataloaders

def test_data_preparation_and_validation(tmp_path):
    test_raw_dir = os.path.join(tmp_path, "raw")
    test_proc_dir = os.path.join(tmp_path, "processed")
    classes = ["Tomato___healthy", "Tomato___Early_blight"]

    # Generate benchmark samples
    prepared_classes = prepare_dataset(raw_dir=test_raw_dir, samples_per_class=10, classes=classes)
    assert len(prepared_classes) == 2

    # Validate integrity
    report = validate_dataset_integrity(raw_dir=test_raw_dir, report_path=os.path.join(test_proc_dir, "report.json"))
    assert report["valid_images"] == 20
    assert report["corrupted_images_count"] == 0

    # Test stratified dataloaders
    train_loader, val_loader, test_loader, cls_names, class_to_idx = get_dataloaders(
        raw_dir=test_raw_dir,
        processed_dir=test_proc_dir,
        batch_size=4,
        image_size=(224, 224),
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )

    assert len(cls_names) == 2
    assert len(train_loader.dataset) == 14
    assert len(val_loader.dataset) == 3
    assert len(test_loader.dataset) == 3
