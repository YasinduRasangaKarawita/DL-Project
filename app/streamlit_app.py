import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import json
import time
import pandas as pd
import numpy as np

# Ensure root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.custom_cnn import CustomCNN
from src.models.resnet50 import get_resnet50
from src.models.efficientnet_b0 import get_efficientnet_b0
from src.models.mobilenet_v3 import get_mobilenet_v3

# Page Configuration
st.set_page_config(
    page_title="PlantGuard AI — Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #10b981 0%, #047857 50%, #064e3b 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);
    }
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.92;
        max-width: 800px;
    }
    
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 1.2rem;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .badge-healthy {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .badge-disease {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
</style>
""", unsafe_allow_html=True)

DISEASE_GUIDE = {
    "healthy": {
        "description": "Leaf appears physiologically sound with vibrant chlorophyll distribution and no signs of bacterial, fungal, or viral infestation.",
        "management": "Maintain standard irrigation schedules, balanced N-P-K fertilization, and inspect weekly for early signs of pests."
    },
    "early_blight": {
        "description": "Caused by the fungus Alternaria solani. Symptoms present as brown-to-black necrotic lesions with distinct concentric 'target-board' rings.",
        "management": "Prune lower infected foliage, avoid overhead watering, mulch soil to reduce spore splashing, and apply copper-based organic fungicides."
    },
    "late_blight": {
        "description": "Devastating disease caused by Phytophthora infestans. Exhibits dark, water-soaked, greasy lesions that spread rapidly in cool, humid weather.",
        "management": "Immediately rogue heavily infected leaves. Ensure adequate plant spacing for airflow. Apply preventative chlorothalonil or mancozeb sprays."
    },
    "scab": {
        "description": "Fungal infection (Venturia inaequalis) causing olive-green to black velvety spots that turn corky and distort leaf margins.",
        "management": "Rake and destroy fallen leaf debris before winter. Apply captan or sulfur-based organic fungicidal sprays during early bud break."
    },
    "black_rot": {
        "description": "Destructive fungal pathogen causing circular dark brown lesions with black pycnidia and yellow halos.",
        "management": "Sterilize pruning shears between cuts. Remove mummified fruit and apply preventative copper or myclobutanil fungicides."
    },
    "rust": {
        "description": "Fungal pathogen causing raised powdery reddish-orange pustules predominantly on leaf undersides.",
        "management": "Plant resistant cultivars, eliminate weed hosts, rotate crops, and apply triadimefon or sulfur dust."
    },
    "leaf_mold": {
        "description": "Caused by Passalora fulva. Symptoms include pale greenish-yellow patches on upper leaf surfaces and olive-brown mold underneath.",
        "management": "Enhance greenhouse ventilation, lower relative humidity below 85%, and apply copper hydroxide sprays."
    }
}

@st.cache_resource
def load_class_mapping():
    class_map_path = os.path.join(PROJECT_ROOT, "data", "processed", "class_to_idx.json")
    if os.path.exists(class_map_path):
        with open(class_map_path, "r", encoding="utf-8") as f:
            c2i = json.load(f)
        idx_to_class = {v: k for k, v in c2i.items()}
        return [idx_to_class[i] for i in range(len(idx_to_class))]
    return [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___healthy",
        "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
        "Grape___Black_rot", "Grape___healthy", "Potato___Early_blight",
        "Potato___Late_blight", "Potato___healthy", "Tomato___Early_blight",
        "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___healthy"
    ]

@st.cache_resource
def load_model_instance(model_name: str, num_classes: int):
    device = torch.device("cpu")
    model_paths = {
        "Custom CNN": ("models/custom_cnn/best_model.pt", lambda: CustomCNN(num_classes=num_classes)),
        "ResNet50": ("models/resnet50/best_model.pt", lambda: get_resnet50(num_classes=num_classes, pretrained=False, freeze_base=False)),
        "EfficientNetB0": ("models/efficientnet/best_model.pt", lambda: get_efficientnet_b0(num_classes=num_classes, pretrained=False, freeze_base=False)),
        "MobileNetV3": ("models/mobilenet/best_model.pt", lambda: get_mobilenet_v3(num_classes=num_classes, pretrained=False, freeze_base=False))
    }

    path_rel, builder = model_paths[model_name]
    full_path = os.path.join(PROJECT_ROOT, path_rel)

    model = builder()
    if os.path.exists(full_path):
        try:
            ckpt = torch.load(full_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
        except Exception as e:
            st.warning(f"Could not load checkpoint weights for {model_name}: {e}. Running initialized model.")
    model.eval()
    return model

@st.cache_data
def load_comparison_data():
    csv_path = os.path.join(PROJECT_ROOT, "results", "model_comparison", "comparison_table.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def preprocess_input_image(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🌿 PlantGuard AI</div>
    <div class="sub-title">Comparative Deep Learning Framework for Multi-Class Plant Disease Recognition & Crop Health Analytics</div>
</div>
""", unsafe_allow_html=True)

classes = load_class_mapping()
num_classes = len(classes)

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Inference Settings")
selected_model_name = st.sidebar.radio(
    "Select Deep Learning Architecture:",
    ("Custom CNN", "ResNet50", "EfficientNetB0", "MobileNetV3"),
    index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Architecture Overview")
comparison_df = load_comparison_data()
if comparison_df is not None:
    matched = comparison_df[comparison_df["Model"].str.contains(selected_model_name.replace(" ", "_"), case=False, na=False)]
    if not matched.empty:
        row = matched.iloc[0]
        st.sidebar.metric("Test Accuracy", f"{row['Accuracy']*100:.1f}%")
        st.sidebar.metric("Macro F1-Score", f"{row['F1-score (Macro)']:.3f}")
        st.sidebar.metric("Parameters", f"{row['Parameters (M)']} M")
        st.sidebar.metric("Inference Latency", f"{row['Inference Latency (ms)']} ms")

# Main Content Layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📸 Leaf Image Input")
    upload_tab, sample_tab = st.tabs(["Upload Your Image", "Use Sample Benchmark Leaf"])
    
    selected_image = None
    with upload_tab:
        uploaded_file = st.file_uploader("Upload Plant Leaf (JPG, PNG, JPEG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")

    with sample_tab:
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        if os.path.exists(raw_dir):
            all_classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
            if all_classes:
                chosen_sample_cls = st.selectbox("Choose Disease Category Sample:", all_classes)
                cls_folder = os.path.join(raw_dir, chosen_sample_cls)
                sample_imgs = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.png'))]
                if sample_imgs:
                    sample_path = os.path.join(cls_folder, sample_imgs[0])
                    if st.button("Load Sample Leaf"):
                        selected_image = Image.open(sample_path).convert("RGB")

    if selected_image is not None:
        st.image(selected_image, caption="Current Input Leaf", use_container_width=True)

with col_right:
    st.markdown("### 🔬 Diagnosis & Classification")
    if selected_image is not None:
        # Load Model and Run Inference
        model = load_model_instance(selected_model_name, num_classes)
        input_tensor = preprocess_input_image(selected_image)

        t_start = time.time()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        inference_time_ms = (time.time() - t_start) * 1000

        top3_indices = np.argsort(probs)[-3:][::-1]
        top1_idx = top3_indices[0]
        top1_class = classes[top1_idx]
        top1_conf = probs[top1_idx] * 100

        # Formatted names
        plant, disease = top1_class.split("___") if "___" in top1_class else ("Plant", top1_class)
        is_healthy = "healthy" in disease.lower()

        # Diagnosis Banner
        badge_class = "badge-healthy" if is_healthy else "badge-disease"
        st.markdown(f"""
        <div style="background: {'#f0fdf4' if is_healthy else '#fef2f2'}; border: 1px solid {'#bbf7d0' if is_healthy else '#fecaca'}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;">
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: {'#15803d' if is_healthy else '#b91c1c'}; font-weight: 700;">Diagnostic Result</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: {'#166534' if is_healthy else '#991b1b'}; margin: 0.2rem 0;">
                {plant.title()} — {disease.replace('_', ' ').title()}
            </div>
            <div style="font-size: 1rem; color: #475569;">
                Confidence: <strong>{top1_conf:.1f}%</strong> | Model Latency: <strong>{inference_time_ms:.1f} ms</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top-3 Predictions Bar Chart
        st.markdown("#### 🏆 Top 3 Candidate Predictions")
        top3_data = []
        for rank, idx in enumerate(top3_indices, start=1):
            cls_name = classes[idx].replace("___", " - ").replace("_", " ")
            conf = probs[idx] * 100
            top3_data.append({"Class": cls_name, "Confidence (%)": round(conf, 2)})

        top3_df = pd.DataFrame(top3_data)
        st.bar_chart(top3_df.set_index("Class"), color="#10b981")

        # Disease Details & Agricultural Guide
        st.markdown("#### 🌾 Agricultural Management & Treatment")
        matched_guide_key = "healthy" if is_healthy else "early_blight"
        for key in DISEASE_GUIDE:
            if key in disease.lower():
                matched_guide_key = key
                break
        guide = DISEASE_GUIDE[matched_guide_key]

        st.info(f"**Symptoms & Biology**: {guide['description']}")
        st.success(f"**Recommended Action**: {guide['management']}")

    else:
        st.info("👆 Please upload a plant leaf image or choose a sample leaf from the left panel to trigger diagnosis.")

# Bottom Section: Cross-Model Comparison Table & Research Telemetry
st.markdown("---")
st.markdown("### 📈 Comprehensive Cross-Model Benchmark Comparison")
if comparison_df is not None:
    st.dataframe(
        comparison_df.style.highlight_max(subset=["Accuracy", "Top-3 Accuracy", "F1-score (Macro)"], color="#dcfce7")
                           .highlight_min(subset=["Training Time (s)", "Inference Latency (ms)", "Parameters (M)"], color="#dbeafe"),
        use_container_width=True
    )
else:
    st.info("Benchmark comparison table will be populated after executing the pipeline runner (`run_pipeline.py`).")
