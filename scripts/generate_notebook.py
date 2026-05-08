from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.settings import METRICS_DIR, NOTEBOOKS_DIR, RESULTS_JSON_PATH, ensure_project_dirs


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def build_notebook(results: dict) -> dict:
    dataset_summary = results["dataset_summary"]
    split_counts = results["split_counts"]
    modeling_dataset_count = results["modeling_dataset_count"]
    modeling_samples_per_class = results["modeling_samples_per_class"]
    experiments = results["experiments"]
    best = results["best_experiment"]
    test_metrics = results["test_metrics"]
    xai_sample = results["xai_sample"]
    underfit_note = (
        "Las curvas muestran un comportamiento más cercano al subajuste que al sobreajuste, "
        "porque la accuracy se mantiene moderada y la pérdida sigue relativamente alta tanto en entrenamiento como en validación."
        if test_metrics["accuracy"] < 0.7
        else "Las curvas muestran una convergencia razonable y no evidencian un sobreajuste severo."
    )

    hyper_table_header = (
        "| Configuración | Filtros | Dropout | Learning rate | Mejor época | Mejor val accuracy | Mejor val loss |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    hyper_table_rows = "".join(
        [
            f"| {row['name']} | {list(row['filters'])} | {row['dropout_rate']} | {row['learning_rate']} | "
            f"{row['best_epoch']} | {row['best_val_accuracy']:.4f} | {row['best_val_loss']:.4f} |\n"
            for row in experiments
        ]
    )

    cells = [
        markdown_cell(
            "# Laboratorio CNNs-XAI\n\n"
            "Cuadernillo organizado para los siete ejercicios del laboratorio. "
            "Las respuestas analíticas aparecen en celdas `markdown` para que puedan copiarse luego en la parte manuscrita."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display, Markdown\n\n"
            "ROOT = Path('..').resolve() if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "RESULTS_PATH = ROOT / 'reports' / 'metrics' / 'lab_results.json'\n"
            "results = json.loads(RESULTS_PATH.read_text(encoding='utf-8'))\n"
            "results"
        ),
        markdown_cell(
            "## Ejercicio 1. Descarga y exploración del dataset\n\n"
            "### Respuesta para la parte manuscrita\n\n"
            f"El dataset contiene **{dataset_summary['male']['count']} imágenes de la clase male** y "
            f"**{dataset_summary['female']['count']} imágenes de la clase female**, para un total de "
            f"**{dataset_summary['male']['count'] + dataset_summary['female']['count']} imágenes**. "
            "La colección presenta una variabilidad alta en tamaño, formato y modo de color. "
            f"En `male`, los anchos van de **{dataset_summary['male']['width_min']}** a **{dataset_summary['male']['width_max']}** píxeles, "
            f"mientras que en `female` van de **{dataset_summary['female']['width_min']}** a **{dataset_summary['female']['width_max']}**. "
            "También se encontraron imágenes en modos `RGB`, `RGBA`, `L` y `P`, lo que justificó convertir todo explícitamente a `RGB` antes del entrenamiento.\n\n"
            "### Estructura de carpetas solicitada\n\n"
            "```text\n"
            "data/\n"
            "├── male/\n"
            "└── female/\n"
            "```\n\n"
            "### Observaciones clave\n\n"
            "- Hay mezcla de extensiones `.jpg`, `.jpeg` y `.png`.\n"
            "- Existen tamaños muy dispares, por lo que era obligatorio unificar la entrada.\n"
            "- No se detectaron archivos corruptos en la inspección inicial."
        ),
        code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'class_distribution.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'dataset_mosaic.png')))"
        ),
        markdown_cell(
            "## Ejercicio 2. Preprocesamiento y partición\n\n"
            "### Respuesta para la parte manuscrita\n\n"
            "El preprocesamiento consistió en: conversión a `RGB`, ajuste de tamaño uniforme con preservación del contenido facial mediante `resize_with_pad`, "
            "normalización al rango `[0, 1]` y partición estratificada en entrenamiento, validación y prueba. "
            f"Para la fase de modelado se tomó una **submuestra estratificada de {modeling_dataset_count} imágenes** "
            f"(**{modeling_samples_per_class} por clase**) para mantener un entrenamiento ligero y desplegable en Streamlit Cloud. "
            f"La partición final quedó en **{split_counts['train']} imágenes para entrenamiento**, "
            f"**{split_counts['validation']} para validación** y **{split_counts['test']} para prueba**.\n\n"
            "### Esquema del flujo\n\n"
            "```text\n"
            "lectura de rutas -> conversión a RGB -> resize con padding -> normalización -> partición estratificada\n"
            "```\n\n"
            "### Justificación conceptual\n\n"
            "- Mantener color es importante porque la red aprende patrones cromáticos y de textura.\n"
            "- Mantener tamaño uniforme es importante porque las capas convolucionales y densas requieren tensores consistentes.\n"
            "- La partición estratificada evita desbalance entre clases en los subconjuntos."
        ),
        code_cell(
            "pd.DataFrame([\n"
            "    {\n"
            "        'train': results['split_counts']['train'],\n"
            "        'validation': results['split_counts']['validation'],\n"
            "        'test': results['split_counts']['test'],\n"
            "        'modeling_dataset_count': results['modeling_dataset_count'],\n"
            "    }\n"
            "])"
        ),
        markdown_cell(
            "## Ejercicio 3. Construcción y entrenamiento de la CNN\n\n"
            "### Respuesta para la parte manuscrita\n\n"
            "La arquitectura final corresponde a una CNN secuencial construida desde cero con bloques `Conv2D + BatchNormalization + MaxPooling2D`, "
            "seguida por una capa `GlobalAveragePooling2D`, una capa densa oculta y una salida sigmoide para clasificación binaria.\n\n"
            "### Descripción resumida de la arquitectura\n\n"
            f"- Entrada: imagen RGB de `{results['image_size'][0]}x{results['image_size'][1]}`.\n"
            f"- Bloques convolucionales principales: filtros `{list(best['filters'])}`.\n"
            f"- Dropout seleccionado: `{best['dropout_rate']}`.\n"
            "- Salida: `Dense(1, activation='sigmoid')`.\n\n"
            "### Análisis de entrenamiento\n\n"
            f"En prueba, el modelo final obtuvo una **accuracy de {test_metrics['accuracy']:.2%}** y una "
            f"**loss de {test_metrics['loss']:.4f}**. "
            "Las curvas de entrenamiento y validación permiten verificar si la red converge de manera estable y si aparece separación excesiva entre ambas curvas, "
            "lo que sería señal de sobreajuste. "
            + underfit_note
        ),
        code_cell(
            "print((ROOT / 'reports' / 'metrics' / 'model_summary.txt').read_text(encoding='utf-8'))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'training_final.png')))"
        ),
        markdown_cell(
            "## Ejercicio 4. Ajuste de hiperparámetros\n\n"
            "### Respuesta para la parte manuscrita\n\n"
            "Se compararon al menos dos configuraciones distintas de hiperparámetros. "
            "La decisión final se tomó a partir de la métrica de validación, priorizando la mejor `accuracy` y, en caso de cercanía, la menor `loss`.\n\n"
            "### Tabla comparativa\n\n"
            + hyper_table_header
            + hyper_table_rows
            + "\n"
            f"La mejor configuración fue **{best['name']}**, con una mejor `val_accuracy` de **{best['best_val_accuracy']:.4f}** "
            f"y una `val_loss` de **{best['best_val_loss']:.4f}**. "
            "Se seleccionó porque mostró el mejor equilibrio entre rendimiento y estabilidad durante la validación. "
            "Aun así, el margen entre configuraciones fue pequeño, lo que sugiere que el principal límite estuvo más asociado a la capacidad total del modelo ligero y al tamaño de la submuestra que a una sola decisión de hiperparámetros."
        ),
        code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'hyperparameter_comparison.png')))\n"
            "pd.read_csv(ROOT / 'reports' / 'metrics' / 'hyperparameter_results.csv')"
        ),
        markdown_cell(
            "## Ejercicio 5. Interpretabilidad visual\n\n"
            "### Respuesta para la parte manuscrita\n\n"
            "El **Saliency Map** resalta los píxeles individuales cuya perturbación cambia más la salida del modelo. "
            "Por eso suele verse más granular y sensible al detalle fino. "
            "Por su parte, **Grad-CAM** utiliza las activaciones de una capa convolucional profunda para identificar regiones espaciales completas, "
            "por lo que tiende a producir mapas más estructurados y fáciles de interpretar a nivel de zonas del rostro.\n\n"
            f"Para el ejemplo seleccionado, la imagen pertenecía a la clase real **{xai_sample['true_label']}** y la predicción fue "
            f"**{xai_sample['predicted_label']}**, con probabilidad de `male` igual a **{xai_sample['prob_male']:.4f}**. "
            "La interpretación debe centrarse en si ambos mapas privilegian ojos, cejas, contorno facial, nariz o cabello, "
            "en lugar de activar el fondo o los bordes externos."
        ),
        code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'xai_example.png')))\n"
            "results['xai_sample']"
        ),
        markdown_cell(
            "## Ejercicio 6. Despliegue con Streamlit\n\n"
            "### Respuesta para la parte manuscrita\n\n"
            "La interfaz de Streamlit se diseñó con tres bloques principales:\n\n"
            "1. Área de carga de imagen.\n"
            "2. Panel de predicción con probabilidades por clase.\n"
            "3. Área de visualización de interpretabilidad con `Saliency Map` y `Grad-CAM`.\n\n"
            "### Flujo entre interfaz y modelo\n\n"
            "```text\n"
            "imagen cargada -> preprocesamiento -> predicción sigmoide -> cálculo de Saliency Map y Grad-CAM -> visualización\n"
            "```\n\n"
            "### Dos mejoras posibles\n\n"
            "- Incorporar una galería de ejemplos de prueba para demostraciones rápidas en clase.\n"
            "- Añadir un módulo de comparación entre varias imágenes o una sección de limitaciones/sesgos del modelo."
        ),
        code_cell(
            "from pathlib import Path\n"
            "app_path = ROOT / 'app' / 'streamlit_app.py'\n"
            "print(app_path)\n"
            "print(app_path.read_text(encoding='utf-8')[:2000])"
        ),
        markdown_cell(
            "## Ejercicio 7. Presentación y reflexión final\n\n"
            "### Guion breve para presentar en clase\n\n"
            "1. Mostrar la app funcionando con una imagen de rostro autorizada.\n"
            "2. Explicar la probabilidad generada por la salida sigmoide.\n"
            "3. Comparar el mapa de Saliency con Grad-CAM.\n"
            "4. Señalar si el modelo se concentró en rasgos faciales plausibles.\n\n"
            "### Reflexión final\n\n"
            "Este laboratorio muestra que una CNN construida desde cero puede resolver una tarea binaria de clasificación facial, "
            "pero también deja claro que la interpretación del modelo es indispensable para verificar si la decisión se apoya en rasgos razonables. "
            "La combinación de predicción + XAI + despliegue convierte el trabajo en una solución más completa y defendible en presentación."
        ),
        code_cell(
            "# Si desea regenerar todos los resultados desde cero, ejecute estas líneas manualmente.\n"
            "# import os, subprocess\n"
            "# subprocess.run(['python', 'scripts/train_pipeline.py'], cwd=str(ROOT), check=True)\n"
            "# subprocess.run(['python', 'scripts/generate_notebook.py'], cwd=str(ROOT), check=True)\n"
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    ensure_project_dirs()
    if not RESULTS_JSON_PATH.exists():
        raise FileNotFoundError(
            "No se encontró reports/metrics/lab_results.json. Ejecute scripts/train_pipeline.py primero."
        )

    results = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
    notebook = build_notebook(results)
    output_path = NOTEBOOKS_DIR / "cnn_gender_lab.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook generado en {output_path}")


if __name__ == "__main__":
    main()
