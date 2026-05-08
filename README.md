# CNN Gender Classifier with XAI

Proyecto del laboratorio de CNNs-XAI para clasificar rostros en las clases `female` y `male` usando una CNN construida desde cero con TensorFlow-Keras, más mapas de interpretabilidad visual con `Saliency Map` y `Grad-CAM`.

## Estructura del repositorio

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── README.md
├── models/
│   └── model.keras
├── notebooks/
│   └── cnn_gender_lab.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── scripts/
│   ├── download_dataset.py
│   ├── generate_notebook.py
│   └── train_pipeline.py
├── src/
├── .streamlit/
│   └── config.toml
└── requirements.txt
```

## 1. Preparar entorno

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nota:
Para el entrenamiento local en Apple Silicon, en esta entrega se usó `tensorflow-macos` con `tensorflow-metal` dentro del entorno local. El `requirements.txt` del repositorio queda orientado al despliegue de la app en Streamlit Cloud con el modelo ya entrenado.

## 2. Descargar y organizar dataset

```bash
PYTHONPATH=. python scripts/download_dataset.py
```

Este script descarga el dataset de Kaggle y deja dos rutas locales coherentes con el laboratorio:

- `data/male`
- `data/female`

## 3. Entrenar el modelo y generar resultados

```bash
PYTHONPATH=. python scripts/train_pipeline.py
```

La rutina de entrenamiento:

- inspecciona el dataset;
- construye una submuestra estratificada configurable para el modelado;
- crea particiones estratificadas `train/validation/test`;
- compara dos configuraciones de hiperparámetros;
- entrena el modelo final;
- guarda `models/model.keras`;
- genera figuras y métricas en `reports/`.

## 4. Generar el cuadernillo

```bash
PYTHONPATH=. python scripts/generate_notebook.py
```

El notebook deja respuestas en celdas `markdown` para la parte manuscrita, además del flujo completo de código del laboratorio.

## 5. Ejecutar la app de Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Despliegue en Streamlit Community Cloud

1. Suba el contenido del repositorio a GitHub.
2. Cree una app nueva en [Streamlit Community Cloud](https://share.streamlit.io).
3. Seleccione este repositorio.
4. En `Main file path` use: `app/streamlit_app.py`.
5. Presione `Deploy`.

## Notas

- El dataset no se versiona en GitHub porque es demasiado grande.
- El modelo final sí debe quedar dentro de `models/model.keras`.
- Las rutas del proyecto son relativas para facilitar el despliegue.
