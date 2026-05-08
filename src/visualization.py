from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.runtime import configure_runtime

configure_runtime()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.settings import CLASS_NAMES


def save_dataset_mosaic(records: pd.DataFrame, output_path: Path, samples_per_class: int = 6) -> None:
    figure, axes = plt.subplots(len(CLASS_NAMES), samples_per_class, figsize=(14, 5))
    for row, label_name in enumerate(CLASS_NAMES):
        sample_df = records[records["label"] == label_name].head(samples_per_class)
        for col, (_, item) in enumerate(sample_df.iterrows()):
            with Image.open(item["filepath"]) as image:
                axes[row, col].imshow(image.convert("RGB"))
            axes[row, col].set_title(label_name.capitalize())
            axes[row, col].axis("off")
    figure.suptitle("Mosaico representativo del dataset", fontsize=16, weight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_class_distribution(records: pd.DataFrame, output_path: Path) -> None:
    counts = records["label"].value_counts().reindex(CLASS_NAMES)
    figure, ax = plt.subplots(figsize=(6, 4))
    colors = ["#ff8fab", "#4c78a8"]
    bars = ax.bar(counts.index, counts.values, color=colors)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_title("Distribución por clase")
    ax.set_ylabel("Número de imágenes")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history_df: pd.DataFrame, output_path: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["loss"], label="Entrenamiento")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validación")
    axes[0].set_title("Pérdida")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Binary crossentropy")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="Entrenamiento")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validación")
    axes[1].set_title("Exactitud")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.suptitle(title, fontsize=15, weight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    figure, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión en prueba")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_hyperparameter_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results_df["name"], results_df["best_val_accuracy"], color=["#4c78a8", "#f58518"])
    for _, row in results_df.iterrows():
        ax.text(row["name"], row["best_val_accuracy"] + 0.002, f"{row['best_val_accuracy']:.3f}", ha="center")
    ax.set_ylim(0.0, min(1.0, results_df["best_val_accuracy"].max() + 0.08))
    ax.set_title("Comparación de configuraciones")
    ax.set_ylabel("Mejor accuracy de validación")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_xai_figure(
    original_rgb: np.ndarray,
    saliency_overlay: np.ndarray,
    gradcam_overlay: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Imagen preprocesada")
    axes[1].imshow(saliency_overlay)
    axes[1].set_title("Saliency Map")
    axes[2].imshow(gradcam_overlay)
    axes[2].set_title("Grad-CAM")
    for ax in axes:
        ax.axis("off")
    figure.suptitle(title, fontsize=15, weight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_markdown_table(results_df: pd.DataFrame, output_path: Path) -> None:
    headers = list(results_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in results_df.iterrows():
        formatted = [str(value) for value in row.tolist()]
        lines.append("| " + " | ".join(formatted) + " |")
    output_path.write_text("\n".join(lines), encoding="utf-8")
