import os
import glob
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

def generate_model_comparison(
    experiments_dir: str = "results/experiments",
    output_dir: str = "results/model_comparison",
    figures_dir: str = "figures/comparison"
) -> pd.DataFrame:
    """
    Collate experiment run metadata across all models and generate:
    - comparison_table.csv
    - comparison_table.md
    - cross-model comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    meta_files = glob.glob(os.path.join(experiments_dir, "*_meta.json"))
    records = []

    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            exp_name = data.get("experiment_name", os.path.basename(mf).replace("_meta.json", ""))
            params = data.get("parameters", {})
            efficiency = data.get("computational_efficiency", {})
            metrics = data.get("test_metrics", {})

            records.append({
                "Model": exp_name,
                "Accuracy": metrics.get("accuracy", 0.0),
                "Top-3 Accuracy": metrics.get("top3_accuracy", 0.0),
                "Precision (Macro)": metrics.get("macro_precision", 0.0),
                "Recall (Macro)": metrics.get("macro_recall", 0.0),
                "F1-score (Macro)": metrics.get("macro_f1", 0.0),
                "Parameters (M)": round(params.get("total_params", 0) / 1e6, 2),
                "Trainable Params (M)": round(params.get("trainable_params", 0) / 1e6, 2),
                "Training Time (s)": efficiency.get("total_training_time_sec", 0.0),
                "Inference Latency (ms)": efficiency.get("inference_latency_ms_per_image", 0.0),
                "Model Size (MB)": params.get("model_file_size_mb", 0.0)
            })
        except Exception as e:
            print(f"Error reading {mf}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="Accuracy", ascending=False)
        csv_path = os.path.join(output_dir, "comparison_table.csv")
        md_path = os.path.join(output_dir, "comparison_table.md")
        
        try:
            md_table = df.to_markdown(index=False)
        except Exception:
            header = "| " + " | ".join(df.columns) + " |\n"
            sep = "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
            rows = "\n".join(["| " + " | ".join(str(val) for val in row) + " |" for row in df.values])
            md_table = header + sep + rows
            
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 📊 Deep Learning Model Comparative Evaluation\n\n")
            f.write(md_table)
            f.write("\n")

        # Plot 1: Accuracy & F1 Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        df_melt = pd.melt(df, id_vars=["Model"], value_vars=["Accuracy", "F1-score (Macro)"], var_name="Metric", value_name="Score")
        sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="crest", ax=ax)
        ax.set_title("Classification Accuracy and Macro F1-Score Comparison", fontsize=12, weight="bold")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "accuracy_f1_comparison.png"), dpi=200)
        plt.close(fig)

        # Plot 2: Parameter Count & Model Size
        fig, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x="Model", y="Parameters (M)", color="#3b82f6", ax=ax1, alpha=0.85)
        ax1.set_ylabel("Total Parameters (Millions)", color="#1d4ed8", fontsize=11)
        ax1.set_title("Model Complexity: Parameter Count vs Architecture", fontsize=12, weight="bold")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "parameter_count_comparison.png"), dpi=200)
        plt.close(fig)

        # Plot 3: Training Time & Inference Latency
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x="Model", y="Training Time (s)", palette="viridis", ax=ax)
        ax.set_title("Computational Efficiency: Total Training Wall-Clock Time", fontsize=12, weight="bold")
        ax.set_ylabel("Seconds", fontsize=11)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "training_time_comparison.png"), dpi=200)
        plt.close(fig)

        # Plot 4: Pareto trade-off: Accuracy vs Model Complexity
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(df["Parameters (M)"], df["Accuracy"] * 100, s=df["Inference Latency (ms)"] * 15 + 80, c=df["Training Time (s)"], cmap="plasma", alpha=0.9, edgecolors="black")
        for i, row in df.iterrows():
            ax.annotate(row["Model"], (row["Parameters (M)"] + 0.3, row["Accuracy"] * 100 + 0.2), fontsize=10, weight="semibold")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Training Time (s)")
        ax.set_title("Accuracy vs Model Complexity (Bubble Size = Inference Latency)", fontsize=12, weight="bold")
        ax.set_xlabel("Model Parameters (Millions)", fontsize=11)
        ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "accuracy_vs_complexity_pareto.png"), dpi=200)
        plt.close(fig)

    return df
