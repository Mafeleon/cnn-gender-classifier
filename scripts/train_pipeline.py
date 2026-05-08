from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from src.dataset import (
    collect_image_records,
    inspect_dataset,
    load_or_create_array_cache,
    make_stratified_splits,
)
from src.inference import prepare_image_from_pil
from src.modeling import (
    build_callbacks,
    build_cnn,
    capture_model_summary,
    experiment_configs,
    history_to_frame,
)
from src.runtime import configure_runtime, seed_everything
from src.settings import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    EXPERIMENT_MODELS_DIR,
    FIGURES_DIR,
    FINAL_MODEL_PATH,
    METRICS_DIR,
    MODELING_SAMPLES_PER_CLASS,
    RESULTS_JSON_PATH,
    SEED,
    ensure_project_dirs,
)
from src.visualization import (
    save_class_distribution,
    save_confusion_matrix,
    save_dataset_mosaic,
    save_hyperparameter_comparison,
    save_training_curves,
    save_xai_figure,
)
from src.xai import make_gradcam_heatmap, make_saliency_map, overlay_heatmap


def serialize_inspection(inspection: dict) -> dict:
    serializable = {}
    for label_name, values in inspection.items():
        serializable[label_name] = {
            key: (value if not isinstance(value, np.generic) else value.item())
            for key, value in values.items()
        }
    return serializable


def train_experiment(
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    tf.keras.backend.clear_session()
    model = build_cnn(
        filters=config["filters"],
        kernel_size=config["kernel_size"],
        dense_units=config["dense_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
    )
    checkpoint_path = EXPERIMENT_MODELS_DIR / f"{config['name']}.keras"
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=config["epochs"],
        callbacks=build_callbacks(checkpoint_path),
        verbose=2,
    )
    best_model = tf.keras.models.load_model(checkpoint_path)
    history_df = history_to_frame(history)
    best_row = history_df.loc[history_df["val_accuracy"].idxmax()]
    summary = {
        **config,
        "best_epoch": int(best_row["epoch"]),
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "best_val_loss": float(best_row["val_loss"]),
        "final_train_accuracy": float(history_df.iloc[-1]["accuracy"]),
        "final_train_loss": float(history_df.iloc[-1]["loss"]),
    }
    return best_model, history_df, summary


def choose_correct_sample(model: tf.keras.Model, test_df: pd.DataFrame):
    candidates = []
    for _, row in test_df.iterrows():
        with Path(row["filepath"]).open("rb") as stream:
            from PIL import Image

            image = Image.open(stream)
            original_rgb, resized_rgb, input_tensor = prepare_image_from_pil(image, DEFAULT_IMAGE_SIZE)
        prob_male = float(model.predict(input_tensor, verbose=0)[0][0])
        predicted_id = int(prob_male >= 0.5)
        confidence = float(max(prob_male, 1 - prob_male))
        if predicted_id == int(row["label_id"]):
            candidates.append(
                {
                    "filepath": row["filepath"],
                    "label": row["label"],
                    "label_id": int(row["label_id"]),
                    "predicted_id": predicted_id,
                    "prob_male": prob_male,
                    "confidence": confidence,
                    "original_rgb": original_rgb,
                    "resized_rgb": resized_rgb,
                    "input_tensor": input_tensor,
                }
            )
    if not candidates:
        raise RuntimeError("No se encontró una imagen correctamente clasificada para generar XAI.")
    return max(candidates, key=lambda item: item["confidence"])


def run_pipeline() -> dict:
    configure_runtime()
    ensure_project_dirs()
    seed_everything(SEED)

    records = collect_image_records()
    inspection = inspect_dataset(records)
    modeling_records = (
        pd.concat(
            [
                group.sample(n=min(len(group), MODELING_SAMPLES_PER_CLASS), random_state=SEED)
                for _, group in records.groupby("label_id")
            ]
        )
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )
    train_df, val_df, test_df = make_stratified_splits(modeling_records)
    print(f"Total de imágenes: {len(records)}", flush=True)
    print(
        f"Subset estratificado para modelado: {len(modeling_records)} "
        f"({MODELING_SAMPLES_PER_CLASS} por clase como máximo)",
        flush=True,
    )
    print(
        f"Splits -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}",
        flush=True,
    )

    save_dataset_mosaic(records, FIGURES_DIR / "dataset_mosaic.png")
    save_class_distribution(records, FIGURES_DIR / "class_distribution.png")

    print("Preprocesando/cargando caché del split de entrenamiento...", flush=True)
    train_images, train_labels = load_or_create_array_cache(train_df, "train", DEFAULT_IMAGE_SIZE)
    print("Preprocesando/cargando caché del split de validación...", flush=True)
    val_images, val_labels = load_or_create_array_cache(val_df, "validation", DEFAULT_IMAGE_SIZE)
    print("Preprocesando/cargando caché del split de prueba...", flush=True)
    test_images, test_labels = load_or_create_array_cache(test_df, "test", DEFAULT_IMAGE_SIZE)
    print("Convirtiendo arreglos a float32 normalizado...", flush=True)
    train_images = train_images.astype("float32") / 255.0
    val_images = val_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    experiment_indices = (
        train_df.groupby("label_id", group_keys=False)
        .sample(frac=0.50, random_state=SEED)
        .index.to_numpy()
    )
    experiment_train_images = train_images[experiment_indices]
    experiment_train_labels = train_labels[experiment_indices]

    experiment_rows = []
    for config in experiment_configs():
        print(f"Iniciando experimento: {config['name']}", flush=True)
        model, history_df, summary = train_experiment(
            config,
            experiment_train_images,
            experiment_train_labels,
            val_images,
            val_labels,
        )
        save_training_curves(
            history_df,
            FIGURES_DIR / f"training_{config['name']}.png",
            title=f"Curvas de entrenamiento - {config['name']}",
        )
        experiment_rows.append(summary)

    experiments_df = pd.DataFrame(experiment_rows).sort_values(
        by=["best_val_accuracy", "best_val_loss"],
        ascending=[False, True],
    )
    save_hyperparameter_comparison(experiments_df, FIGURES_DIR / "hyperparameter_comparison.png")
    experiments_df.to_csv(METRICS_DIR / "hyperparameter_results.csv", index=False)

    best_name = experiments_df.iloc[0]["name"]
    best_config = next(config for config in experiment_configs() if config["name"] == best_name)
    print(f"Mejor configuración preliminar: {best_name}", flush=True)
    tf.keras.backend.clear_session()
    final_model = build_cnn(
        filters=best_config["filters"],
        kernel_size=best_config["kernel_size"],
        dense_units=best_config["dense_units"],
        dropout_rate=best_config["dropout_rate"],
        learning_rate=best_config["learning_rate"],
    )
    final_history = final_model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        batch_size=BATCH_SIZE,
        epochs=max(8, best_config["epochs"]),
        callbacks=build_callbacks(FINAL_MODEL_PATH),
        verbose=2,
    )
    print("Evaluando modelo final y generando artefactos XAI...", flush=True)
    final_model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    final_history_df = history_to_frame(final_history)
    save_training_curves(
        final_history_df,
        FIGURES_DIR / "training_final.png",
        title="Curvas de entrenamiento - modelo final",
    )
    final_model.save(FINAL_MODEL_PATH)
    (METRICS_DIR / "model_summary.txt").write_text(capture_model_summary(final_model), encoding="utf-8")

    test_metrics = final_model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE, return_dict=True, verbose=0)
    probabilities = final_model.predict(test_images, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    y_true = test_labels.astype(int)
    save_confusion_matrix(y_true, predictions, FIGURES_DIR / "confusion_matrix.png")

    report = classification_report(
        y_true,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    (METRICS_DIR / "classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    chosen = choose_correct_sample(final_model, test_df)
    saliency = make_saliency_map(final_model, chosen["input_tensor"])
    gradcam = make_gradcam_heatmap(final_model, chosen["input_tensor"], last_conv_layer_name="last_conv")
    saliency_overlay = overlay_heatmap(chosen["resized_rgb"], saliency)
    gradcam_overlay = overlay_heatmap(chosen["resized_rgb"], gradcam)
    save_xai_figure(
        chosen["resized_rgb"],
        saliency_overlay,
        gradcam_overlay,
        FIGURES_DIR / "xai_example.png",
        title="Interpretabilidad visual sobre una imagen correctamente clasificada",
    )

    results = {
        "dataset_summary": serialize_inspection(inspection),
        "split_counts": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "modeling_dataset_count": int(len(modeling_records)),
        "modeling_samples_per_class": int(MODELING_SAMPLES_PER_CLASS),
        "image_size": list(DEFAULT_IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "experiments": experiments_df.to_dict(orient="records"),
        "best_experiment": experiments_df.iloc[0].to_dict(),
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "classification_report": report,
        "xai_sample": {
            "filepath": chosen["filepath"],
            "true_label": chosen["label"],
            "predicted_label": CLASS_NAMES[chosen["predicted_id"]],
            "prob_male": float(chosen["prob_male"]),
            "confidence": float(chosen["confidence"]),
        },
        "artifacts": {
            "mosaic": str(FIGURES_DIR / "dataset_mosaic.png"),
            "class_distribution": str(FIGURES_DIR / "class_distribution.png"),
            "hyperparameter_comparison": str(FIGURES_DIR / "hyperparameter_comparison.png"),
            "confusion_matrix": str(FIGURES_DIR / "confusion_matrix.png"),
            "xai_example": str(FIGURES_DIR / "xai_example.png"),
        },
    }

    RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    summary = run_pipeline()
    print(json.dumps(summary["test_metrics"], indent=2, ensure_ascii=False))
