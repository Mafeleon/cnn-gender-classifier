from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

SECTION_FILES = [
    ("EJERCICIOS 1 Y 2", "Configuracion general del proyecto", "src/settings.py"),
    ("EJERCICIOS 1 Y 2", "Control de entorno y reproducibilidad", "src/runtime.py"),
    ("EJERCICIO 1", "Descarga y organizacion del dataset", "scripts/download_dataset.py"),
    ("EJERCICIOS 1 Y 2", "Exploracion, preprocesamiento y particiones", "src/dataset.py"),
    ("EJERCICIO 3", "Arquitectura CNN y utilidades de entrenamiento", "src/modeling.py"),
    ("EJERCICIO 4", "Pipeline completo de entrenamiento, ajuste y evaluacion", "scripts/train_pipeline.py"),
    ("EJERCICIO 5", "Generacion de saliency maps y Grad-CAM", "src/xai.py"),
    ("EJERCICIO 5", "Utilidades de visualizacion para resultados y figuras", "src/visualization.py"),
    ("EJERCICIO 6", "Preprocesamiento de inferencia para el despliegue", "src/inference.py"),
    ("EJERCICIO 6", "Aplicacion Streamlit desplegada", "app/streamlit_app.py"),
]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def render_header(section: str, title: str, relative_path: str) -> str:
    return (
        "# ============================================================\n"
        f"# {section}\n"
        f"# {title}\n"
        f"# Archivo: {relative_path}\n"
        "# ============================================================\n\n"
    )


def build_index_cell() -> dict:
    lines = [
        "# ============================================================",
        "# CUADERNILLO DE CODIGO DEL PROYECTO CNN + XAI",
        "# ============================================================",
        "",
        "PROJECT_ROOT = '.'",
        "",
        "# Estructura del cuadernillo:",
    ]
    for section, title, relative_path in SECTION_FILES:
        lines.append(f"# - {section}: {title} ({relative_path})")
    lines.extend(
        [
            "",
            "# Ejecucion manual sugerida:",
            "# !python scripts/download_dataset.py",
            "# !python scripts/train_pipeline.py",
            "# !streamlit run app/streamlit_app.py",
        ]
    )
    return code_cell("\n".join(lines))


def build_source_cell(section: str, title: str, relative_path: str) -> dict:
    source_path = ROOT_DIR / relative_path
    if not source_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo requerido para el notebook: {source_path}")

    content = source_path.read_text(encoding="utf-8").rstrip()
    return code_cell(render_header(section, title, relative_path) + content + "\n")


def build_notebook() -> dict:
    cells = [build_index_cell()]
    for section, title, relative_path in SECTION_FILES:
        cells.append(build_source_cell(section, title, relative_path))

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
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    output_path = NOTEBOOKS_DIR / "cnn_gender_lab.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook generado en {output_path}")


if __name__ == "__main__":
    main()
