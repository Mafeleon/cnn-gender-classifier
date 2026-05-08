from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import tensorflow as tf
from PIL import Image

from src.dataset import prepare_image_from_pil
from src.settings import CLASS_NAMES, DEFAULT_IMAGE_SIZE, FINAL_MODEL_PATH, RESULTS_JSON_PATH
from src.xai import make_gradcam_heatmap, make_saliency_map, overlay_heatmap

st.set_page_config(
    page_title="CNN Gender Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.main > div {
    padding-top: 1.4rem;
}
.hero {
    padding: 2rem 2.2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0b3954 0%, #1f7a8c 55%, #bfd7ea 100%);
    color: #f8fbff;
    margin-bottom: 1.2rem;
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.12);
}
.hero h1 {
    margin: 0;
    font-size: 2.2rem;
}
.hero p {
    margin-top: 0.7rem;
    max-width: 820px;
    font-size: 1rem;
    line-height: 1.6;
}
.card {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: #ffffff;
    border: 1px solid rgba(16, 42, 67, 0.08);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
    margin-bottom: 1rem;
}
.prob-card {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    color: #102a43;
    background: linear-gradient(180deg, #ffffff 0%, #eef6fb 100%);
    border: 1px solid rgba(31, 122, 140, 0.14);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
}
.prob-label {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.8;
}
.prob-value {
    font-size: 2rem;
    font-weight: 700;
    margin: 0.3rem 0 0.1rem;
}
.small-note {
    color: #486581;
    font-size: 0.92rem;
}
</style>
"""


@st.cache_resource
def load_model() -> tf.keras.Model:
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado en models/model.keras. Ejecute scripts/train_pipeline.py."
        )
    return tf.keras.models.load_model(FINAL_MODEL_PATH, compile=False)


@st.cache_data
def load_results_metadata():
    if RESULTS_JSON_PATH.exists():
        return json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
    return {}


def probability_card(title: str, value: float, color: str) -> None:
    st.markdown(
        f"""
        <div class="prob-card" style="border-left: 6px solid {color};">
            <div class="prob-label">{title}</div>
            <div class="prob-value">{value:.2%}</div>
            <div class="small-note">Probabilidad estimada por la CNN.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <section class="hero">
            <h1>Clasificación de género con CNN + interpretabilidad visual</h1>
            <p>
                Esta aplicación permite cargar una imagen de rostro, ejecutar la predicción del modelo
                entrenado desde cero en TensorFlow-Keras y visualizar dos métodos de explicabilidad:
                <strong>Saliency Map</strong> y <strong>Grad-CAM</strong>.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metadata: dict):
    st.sidebar.markdown("## Resumen del modelo")
    st.sidebar.write(f"Entrada del modelo: `{DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]} RGB`")
    if metadata.get("test_metrics"):
        accuracy = metadata["test_metrics"].get("accuracy")
        loss = metadata["test_metrics"].get("loss")
        if accuracy is not None:
            st.sidebar.metric("Accuracy en prueba", f"{accuracy:.2%}")
        if loss is not None:
            st.sidebar.metric("Loss en prueba", f"{loss:.4f}")
    st.sidebar.markdown("## Flujo")
    st.sidebar.markdown(
        "1. El usuario carga una imagen.\n"
        "2. La app la convierte a RGB y la ajusta al tamaño usado en entrenamiento.\n"
        "3. El modelo genera la predicción.\n"
        "4. Se construyen los mapas Saliency y Grad-CAM."
    )
    st.sidebar.markdown("## Nota")
    st.sidebar.caption(
        "Los mapas se superponen sobre la versión preprocesada que realmente consume la red."
    )


def main():
    render_header()
    metadata = load_results_metadata()
    render_sidebar(metadata)

    st.markdown(
        """
        <div class="card">
            Cargue una imagen de rostro para obtener la predicción y las visualizaciones de interpretabilidad.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Suba una imagen en formato JPG, JPEG o PNG",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("La visualización aparecerá aquí cuando cargue una imagen.")
        return

    model = load_model()
    pil_image = Image.open(uploaded_file)
    original_rgb, resized_rgb, input_tensor = prepare_image_from_pil(pil_image, DEFAULT_IMAGE_SIZE)
    prob_male = float(model.predict(input_tensor, verbose=0)[0][0])
    prob_female = 1.0 - prob_male
    predicted_label = CLASS_NAMES[int(prob_male >= 0.5)]

    saliency = make_saliency_map(model, input_tensor)
    gradcam = make_gradcam_heatmap(model, input_tensor, last_conv_layer_name="last_conv")
    saliency_overlay = overlay_heatmap(resized_rgb, saliency)
    gradcam_overlay = overlay_heatmap(resized_rgb, gradcam)

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.image(original_rgb, caption="Imagen original cargada", use_container_width=True)
    with right:
        st.markdown(
            f"""
            <div class="card">
                <h3 style="margin-top:0;">Predicción principal</h3>
                <p style="font-size:1.15rem;margin-bottom:0.2rem;">
                    Clase estimada: <strong>{predicted_label}</strong>
                </p>
                <p class="small-note">
                    La salida sigmoide representa la probabilidad de la clase <strong>male</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prob_left, prob_right = st.columns(2)
        with prob_left:
            probability_card("Female", prob_female, "#ff8fab")
        with prob_right:
            probability_card("Male", prob_male, "#1f7a8c")

    tab_prediction, tab_saliency, tab_gradcam, tab_details = st.tabs(
        ["Predicción", "Saliency Map", "Grad-CAM", "Metodología"]
    )

    with tab_prediction:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(resized_rgb, caption="Imagen preprocesada por la CNN", use_container_width=True)
        with col_b:
            st.markdown(
                """
                <div class="card">
                    <h4 style="margin-top:0;">Lectura rápida</h4>
                    <p class="small-note">
                        La clase con mayor probabilidad es la decisión final del modelo. Esta sección
                        muestra la imagen exactamente en la forma en que entra a la red.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_saliency:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(saliency_overlay, caption="Superposición de Saliency Map", use_container_width=True)
        with col_b:
            st.markdown(
                """
                <div class="card">
                    <h4 style="margin-top:0;">Qué muestra Saliency Map</h4>
                    <p class="small-note">
                        Resalta los píxeles cuya variación afecta más directamente la salida del modelo.
                        Tiende a ser más sensible y granular.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_gradcam:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(gradcam_overlay, caption="Superposición de Grad-CAM", use_container_width=True)
        with col_b:
            st.markdown(
                """
                <div class="card">
                    <h4 style="margin-top:0;">Qué muestra Grad-CAM</h4>
                    <p class="small-note">
                        Identifica regiones espaciales activadas por las capas profundas de la CNN.
                        Suele producir mapas más estructurados y fáciles de interpretar por zonas.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_details:
        st.markdown(
            """
            <div class="card">
                <h4 style="margin-top:0;">Preprocesamiento aplicado</h4>
                <p class="small-note">
                    Conversión a RGB, ajuste a tamaño uniforme con preservación del contenido facial
                    mediante padding y normalización de valores al rango [0, 1].
                </p>
                <h4>Interpretación recomendada</h4>
                <p class="small-note">
                    Compare los dos mapas: si ambos enfatizan ojos, cejas, contorno del rostro o
                    cabello, la decisión del modelo es más coherente. Si activan fondo o bordes,
                    conviene revisar sesgos o ruido en los datos.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
