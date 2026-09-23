import os
import json
from PIL import Image
from typing import Dict, Any, List
from ..utils.logger import setup_logger

logger = setup_logger("data_validation")

def validate_dataset_integrity(raw_dir: str = "data/raw", report_path: str = "results/experiments/data_validation_report.json") -> Dict[str, Any]:
    """
    Validate dataset images:
    - Check file readability
    - Verify image dimensions and channels (RGB)
    - Detect corrupted or empty files
    - Record class distribution
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    classes.sort()

    total_files = 0
    corrupted_files: List[str] = []
    class_counts: Dict[str, int] = {}
    dimensions: List[Dict[str, int]] = []
    channels_set = set()

    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        files = [f for f in os.listdir(cls_dir) if not f.startswith(".")]
        valid_class_count = 0

        for f in files:
            img_path = os.path.join(cls_dir, f)
            total_files += 1

            if os.path.getsize(img_path) == 0:
                corrupted_files.append(img_path)
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                # Reopen to check dimensions and mode
                with Image.open(img_path) as img:
                    w, h = img.size
                    channels_set.add(img.mode)
                    if len(dimensions) < 500:  # Sample up to 500 images for dimension stats
                        dimensions.append({"width": w, "height": h})
                    valid_class_count += 1
            except Exception as e:
                logger.warning(f"Corrupted image detected: {img_path} ({e})")
                corrupted_files.append(img_path)

        class_counts[cls] = valid_class_count

    report = {
        "total_classes": len(classes),
        "total_files_scanned": total_files,
        "valid_images": sum(class_counts.values()),
        "corrupted_images_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "image_modes": list(channels_set),
        "class_distribution": class_counts,
        "dimension_sample_count": len(dimensions)
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Dataset validation completed: {report['valid_images']} valid images across {report['total_classes']} classes. Corrupted: {len(corrupted_files)}")
    return report

if __name__ == "__main__":
    validate_dataset_integrity()
