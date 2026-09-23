import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Dict, Tuple
from ..utils.logger import setup_logger

logger = setup_logger("data_download")

DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy"
]

def generate_leaf_texture(width: int = 256, height: int = 256, plant_type: str = "tomato", disease: str = "healthy", seed: int = 42) -> Image.Image:
    """
    Generate synthetic plant leaf image with characteristic disease symptoms
    for reproducible benchmarking and testing.
    """
    rng = np.random.RandomState(seed)
    
    # Base background: neutral lab bench / dark background
    bg_color = (25 + rng.randint(0, 15), 30 + rng.randint(0, 15), 30 + rng.randint(0, 15))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Leaf base colors per plant type
    if "apple" in plant_type:
        leaf_color = (45 + rng.randint(0, 20), 125 + rng.randint(0, 25), 45 + rng.randint(0, 15))
    elif "corn" in plant_type:
        leaf_color = (75 + rng.randint(0, 20), 145 + rng.randint(0, 20), 40 + rng.randint(0, 15))
    elif "grape" in plant_type:
        leaf_color = (35 + rng.randint(0, 20), 115 + rng.randint(0, 25), 40 + rng.randint(0, 15))
    elif "potato" in plant_type:
        leaf_color = (40 + rng.randint(0, 20), 120 + rng.randint(0, 20), 40 + rng.randint(0, 15))
    else:  # tomato
        leaf_color = (40 + rng.randint(0, 20), 135 + rng.randint(0, 25), 35 + rng.randint(0, 15))

    # Draw leaf boundary (oval / polygon approximation)
    cx, cy = width // 2, height // 2
    rx, ry = int(width * 0.38 + rng.randint(-10, 10)), int(height * 0.44 + rng.randint(-10, 10))
    
    # Leaf contour
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=leaf_color)

    # Draw central vein
    vein_color = (min(255, leaf_color[0] + 30), min(255, leaf_color[1] + 35), min(255, leaf_color[2] + 20))
    draw.line([cx, cy - ry + 15, cx, cy + ry - 15], fill=vein_color, width=3)
    
    # Secondary lateral veins
    for dy in range(-ry + 35, ry - 35, 25):
        side = rng.choice([-1, 1])
        draw.line([cx, cy + dy, cx + side * (rx - 25), cy + dy - 15], fill=vein_color, width=2)

    # Add disease symptoms
    if "healthy" not in disease:
        num_lesions = rng.randint(6, 18)
        for _ in range(num_lesions):
            lx = cx + rng.randint(int(-rx * 0.7), int(rx * 0.7))
            ly = cy + rng.randint(int(-ry * 0.7), int(ry * 0.7))
            radius = rng.randint(6, 20)

            if "scab" in disease:
                # Olive-green to dark brown crusty lesions
                spot_color = (60 + rng.randint(0, 20), 55 + rng.randint(0, 15), 25 + rng.randint(0, 10))
                draw.ellipse([lx - radius, ly - radius, lx + radius, ly + radius], fill=spot_color)
            elif "black_rot" in disease:
                # Black circular necrotic lesions with concentric rings
                spot_color = (25 + rng.randint(0, 15), 20 + rng.randint(0, 10), 15 + rng.randint(0, 10))
                halo_color = (130 + rng.randint(0, 30), 110 + rng.randint(0, 20), 25)
                draw.ellipse([lx - radius - 3, ly - radius - 3, lx + radius + 3, ly + radius + 3], fill=halo_color)
                draw.ellipse([lx - radius, ly - radius, lx + radius, ly + radius], fill=spot_color)
            elif "early_blight" in disease:
                # Concentric target-like rings, brown with yellow chlorotic halos
                halo_color = (175 + rng.randint(0, 30), 160 + rng.randint(0, 25), 30 + rng.randint(0, 15))
                brown_center = (80 + rng.randint(0, 20), 45 + rng.randint(0, 15), 20 + rng.randint(0, 10))
                draw.ellipse([lx - radius - 4, ly - radius - 4, lx + radius + 4, ly + radius + 4], fill=halo_color)
                draw.ellipse([lx - radius, ly - radius, lx + radius, ly + radius], fill=brown_center)
            elif "late_blight" in disease:
                # Dark water-soaked lesions with pale borders
                water_soaked = (50 + rng.randint(0, 20), 60 + rng.randint(0, 20), 30 + rng.randint(0, 15))
                draw.ellipse([lx - radius, ly - radius, lx + radius, ly + radius], fill=water_soaked)
            elif "rust" in disease:
                # Reddish-orange powdery pustules
                rust_color = (195 + rng.randint(0, 40), 75 + rng.randint(0, 30), 20 + rng.randint(0, 15))
                draw.ellipse([lx - radius // 2, ly - radius // 2, lx + radius // 2, ly + radius // 2], fill=rust_color)
            elif "leaf_mold" in disease:
                # Pale yellow spots on upper surface, velvety olive brown
                mold_color = (165 + rng.randint(0, 30), 150 + rng.randint(0, 25), 45 + rng.randint(0, 15))
                draw.ellipse([lx - radius, ly - radius, lx + radius, ly + radius], fill=mold_color)
            elif "blight" in disease:
                # Elongated grayish/tan lesions
                tan_color = (120 + rng.randint(0, 25), 105 + rng.randint(0, 20), 55 + rng.randint(0, 15))
                draw.ellipse([lx - radius * 2, ly - radius, lx + radius * 2, ly + radius], fill=tan_color)

    # Slight blur to simulate natural photographic depth of field
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img

def prepare_dataset(raw_dir: str = "data/raw", samples_per_class: int = 80, classes: List[str] = None) -> List[str]:
    """
    Ensure the dataset exists in data/raw. If existing images are found, uses them.
    Otherwise generates synthetic benchmark images for each class.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    os.makedirs(raw_dir, exist_ok=True)
    existing_classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    # Check if we already have sufficient images
    total_existing = 0
    for cls in existing_classes:
        cls_path = os.path.join(raw_dir, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_existing += len(files)

    if total_existing >= len(classes) * 10:
        logger.info(f"Found existing dataset with {len(existing_classes)} classes and {total_existing} total images.")
        return sorted(existing_classes)

    logger.info(f"Generating benchmark PlantVillage dataset: {len(classes)} classes, {samples_per_class} images/class...")
    
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Parse plant and disease
        parts = cls.lower().split("___")
        plant = parts[0] if len(parts) > 0 else "plant"
        disease = parts[1] if len(parts) > 1 else "healthy"

        for idx in range(samples_per_class):
            img_path = os.path.join(cls_dir, f"{cls}_{idx:04d}.jpg")
            if not os.path.exists(img_path):
                img = generate_leaf_texture(
                    width=256,
                    height=256,
                    plant_type=plant,
                    disease=disease,
                    seed=idx * 31 + len(cls)
                )
                img.save(img_path, quality=92)

    logger.info(f"Successfully prepared benchmark dataset in {raw_dir}")
    return sorted(classes)

if __name__ == "__main__":
    prepare_dataset()
