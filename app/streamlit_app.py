from __future__ import annotations

import base64
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import tensorflow as tf
from PIL import Image

from src.inference import prepare_image_from_pil
from src.settings import CLASS_NAMES, DEFAULT_IMAGE_SIZE, FINAL_MODEL_PATH, RESULTS_JSON_PATH
from src.xai import make_gradcam_heatmap, make_saliency_map, overlay_heatmap

NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "cnn_gender_lab.ipynb"
MODEL_SUMMARY_PATH = ROOT_DIR / "reports" / "metrics" / "model_summary.txt"
FIGURE_PATHS = {
    "class_distribution": ROOT_DIR / "reports" / "figures" / "class_distribution.png",
    "confusion_matrix": ROOT_DIR / "reports" / "figures" / "confusion_matrix.png",
    "dataset_mosaic": ROOT_DIR / "reports" / "figures" / "dataset_mosaic.png",
    "hyperparameter_comparison": ROOT_DIR / "reports" / "figures" / "hyperparameter_comparison.png",
    "training_final": ROOT_DIR / "reports" / "figures" / "training_final.png",
    "training_regularized": ROOT_DIR / "reports" / "figures" / "training_regularized.png",
    "xai_example": ROOT_DIR / "reports" / "figures" / "xai_example.png",
}

st.set_page_config(
    page_title="Prisma XAI Lab | Clasificacion por sexo con CNN",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Hanken+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-0: #040a11;
    --bg-1: #08111c;
    --bg-2: #0c1723;
    --surface: rgba(10, 18, 27, 0.92);
    --surface-soft: rgba(14, 23, 34, 0.84);
    --stroke: rgba(136, 221, 233, 0.16);
    --stroke-strong: rgba(109, 247, 255, 0.42);
    --cyan: #4defff;
    --cyan-bright: #7ef7ff;
    --text: #e7fbff;
    --muted: #88aab6;
    --teal: #11343b;
    --shadow: 0 30px 80px rgba(0, 0, 0, 0.42);
}

html, body, [class*="css"] {
    font-family: "Hanken Grotesk", sans-serif;
}

body {
    color: var(--text);
}

#MainMenu,
footer,
header[data-testid="stHeader"],
[data-testid="collapsedControl"],
[data-testid="stToolbar"],
[data-testid="stSidebar"] {
    display: none !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(58, 163, 184, 0.18), transparent 28%),
        radial-gradient(circle at 85% 12%, rgba(65, 196, 219, 0.12), transparent 24%),
        linear-gradient(180deg, #03070d 0%, #07111a 34%, #050b12 100%);
    color: var(--text);
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(126, 247, 255, 0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(126, 247, 255, 0.045) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: linear-gradient(180deg, transparent 0%, black 20%, black 88%, transparent 100%);
    opacity: 0.18;
}

main .block-container {
    max-width: 1220px;
    padding-top: 0.7rem;
    padding-bottom: 4rem;
    position: relative;
    z-index: 1;
}

a {
    color: inherit;
}

.section-anchor {
    position: relative;
    top: -96px;
    visibility: hidden;
}

.top-nav {
    position: sticky;
    top: 0.9rem;
    z-index: 12;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(10, 18, 28, 0.92), rgba(7, 12, 20, 0.92));
    border: 1px solid rgba(136, 221, 233, 0.12);
    box-shadow: 0 16px 50px rgba(0, 0, 0, 0.26);
    backdrop-filter: blur(18px);
}

.brand-lockup {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #dffaff;
    text-decoration: none;
    white-space: nowrap;
}

.nav-links {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.25rem;
    flex: 1;
    flex-wrap: wrap;
}

.nav-links a {
    text-decoration: none;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    color: var(--muted);
    transition: color 0.2s ease, transform 0.2s ease;
}

.nav-links a:hover {
    color: var(--text);
    transform: translateY(-1px);
}

.nav-cta {
    text-decoration: none;
    padding: 0.8rem 1rem;
    border-radius: 14px;
    border: 1px solid rgba(77, 239, 255, 0.55);
    background: linear-gradient(135deg, #63f3ff 0%, #15d8f6 100%);
    color: #041017;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    box-shadow: 0 0 0 1px rgba(109, 247, 255, 0.08), 0 10px 28px rgba(24, 214, 243, 0.28);
}

.hero-shell {
    position: relative;
    overflow: hidden;
    padding: clamp(2.6rem, 5vw, 4.2rem);
    min-height: 31rem;
    border-radius: 34px;
    background:
        linear-gradient(130deg, rgba(5, 17, 24, 0.95) 0%, rgba(6, 20, 29, 0.88) 38%, rgba(8, 24, 31, 0.82) 100%),
        radial-gradient(circle at 50% 0%, rgba(75, 223, 255, 0.12), transparent 34%);
    border: 1px solid rgba(136, 221, 233, 0.14);
    box-shadow: var(--shadow);
    margin-bottom: 3rem;
}

.hero-shell::before {
    content: "";
    position: absolute;
    inset: -10% -10% 14% -10%;
    background:
        linear-gradient(115deg, transparent 0%, rgba(79, 237, 255, 0.12) 18%, transparent 32%),
        linear-gradient(125deg, transparent 10%, rgba(79, 237, 255, 0.06) 24%, transparent 42%),
        repeating-linear-gradient(110deg, rgba(86, 241, 255, 0.05) 0 3px, transparent 3px 18px);
    animation: heroDrift 18s linear infinite;
    opacity: 0.65;
}

.hero-shell::after {
    content: "";
    position: absolute;
    inset: auto -15% -30% 25%;
    height: 24rem;
    background: radial-gradient(circle, rgba(77, 239, 255, 0.18), transparent 58%);
    filter: blur(18px);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 1;
    max-width: 760px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.9rem;
    margin-bottom: 1rem;
    border-radius: 999px;
    border: 1px solid rgba(109, 247, 255, 0.18);
    background: rgba(5, 20, 27, 0.72);
    color: var(--cyan-bright);
    font-size: 0.74rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.hero-title {
    margin: 0;
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(3.2rem, 8vw, 5.9rem);
    line-height: 0.94;
    letter-spacing: -0.06em;
    color: #e9fdff;
    text-shadow: 0 0 30px rgba(79, 237, 255, 0.08), 0 18px 40px rgba(0, 0, 0, 0.28);
}

.hero-title span {
    color: var(--cyan-bright);
}

.hero-copy {
    margin: 1.25rem 0 0;
    max-width: 620px;
    color: #b4d4db;
    font-size: 1.04rem;
    line-height: 1.75;
}

.hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.85rem;
    margin-top: 2rem;
}

.hero-button {
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    min-width: 220px;
    padding: 0.95rem 1.2rem;
    border-radius: 16px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 700;
    font-size: 0.76rem;
}

.hero-button.primary {
    background: linear-gradient(135deg, #63f3ff 0%, #12d7f4 100%);
    color: #031017;
    border: 1px solid rgba(77, 239, 255, 0.55);
    box-shadow: 0 0 0 1px rgba(109, 247, 255, 0.08), 0 14px 34px rgba(24, 214, 243, 0.24);
}

.hero-button.secondary {
    background: rgba(9, 18, 28, 0.72);
    color: #d6fbff;
    border: 1px solid rgba(136, 221, 233, 0.18);
}

.hero-meta {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin-top: 2.4rem;
}

.hero-meta-card,
.metric-card,
.glass-card {
    position: relative;
    overflow: hidden;
    padding: 1.15rem;
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(10, 18, 27, 0.96), rgba(6, 11, 18, 0.94));
    border: 1px solid rgba(136, 221, 233, 0.12);
    box-shadow: var(--shadow);
}

.hero-meta-card::after,
.metric-card::after,
.glass-card::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(109, 247, 255, 0.08), transparent 25%, transparent 78%, rgba(109, 247, 255, 0.03));
    pointer-events: none;
}

.hero-meta-card span,
.metric-card span {
    display: block;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #84a8b4;
}

.hero-meta-card strong,
.metric-card strong {
    display: block;
    margin-top: 0.45rem;
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.55rem;
    letter-spacing: -0.04em;
    color: #f1fdff;
}

.section-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    margin: 2.7rem 0 1.2rem;
}

.section-header-copy {
    max-width: 760px;
}

.section-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--cyan-bright);
    font-size: 0.72rem;
    letter-spacing: 0.17em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.section-title {
    margin: 0;
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(1.9rem, 4vw, 3rem);
    letter-spacing: -0.04em;
    color: #eafcff;
}

.section-copy {
    margin: 0.55rem 0 0;
    color: #8eaeb9;
    line-height: 1.7;
    font-size: 0.98rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.75rem;
    border-radius: 999px;
    border: 1px solid rgba(77, 239, 255, 0.22);
    background: rgba(8, 20, 26, 0.82);
    color: var(--cyan-bright);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-size: 0.69rem;
    white-space: nowrap;
}

.status-pill::before {
    content: "";
    width: 0.42rem;
    height: 0.42rem;
    border-radius: 999px;
    background: currentColor;
    box-shadow: 0 0 12px currentColor;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin-bottom: 1rem;
}

.metric-card p {
    margin: 0.5rem 0 0;
    color: #7e9ca7;
    line-height: 1.55;
    font-size: 0.9rem;
}

.media-card-header {
    margin-bottom: 0.85rem;
}

.card-eyebrow {
    display: inline-flex;
    color: var(--cyan-bright);
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.media-card-title {
    margin: 0.2rem 0 0;
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.35rem;
    letter-spacing: -0.03em;
    color: #e8fbff;
}

.media-card-subtitle {
    margin: 0.45rem 0 0;
    color: #81a0ab;
    line-height: 1.6;
    font-size: 0.92rem;
}

.media-frame,
.tensor-shot img {
    width: 100%;
    display: block;
    border-radius: 18px;
    border: 1px solid rgba(136, 221, 233, 0.14);
    background: #03070d;
}

.card-note {
    margin-top: 0.85rem;
    color: #9fc0c9;
    font-size: 0.9rem;
    line-height: 1.65;
}

.card-note strong {
    color: var(--cyan-bright);
}

.empty-workbench {
    position: relative;
    padding: 2rem 1.8rem 1.6rem;
    min-height: 22rem;
    background:
        linear-gradient(180deg, rgba(9, 16, 24, 0.95), rgba(7, 12, 19, 0.94)),
        linear-gradient(rgba(77, 239, 255, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(77, 239, 255, 0.04) 1px, transparent 1px);
    background-size: auto, 32px 32px, 32px 32px;
}

.empty-workbench::before,
.empty-workbench::after {
    content: "";
    position: absolute;
    width: 20px;
    height: 20px;
    border-color: var(--cyan-bright);
    border-style: solid;
    border-width: 0;
}

.empty-workbench::before {
    inset: 16px auto auto 16px;
    border-top-width: 2px;
    border-left-width: 2px;
}

.empty-workbench::after {
    inset: auto 16px 16px auto;
    border-right-width: 2px;
    border-bottom-width: 2px;
}

.empty-center {
    max-width: 540px;
    margin: 2.6rem auto 0;
    text-align: center;
}

.empty-icon {
    width: 72px;
    height: 72px;
    margin: 0 auto 1rem;
    display: grid;
    place-items: center;
    border-radius: 20px;
    border: 1px solid rgba(136, 221, 233, 0.18);
    background: rgba(20, 34, 48, 0.65);
    color: var(--cyan-bright);
    font-size: 1.8rem;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
}

.empty-title {
    margin: 0;
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.7rem);
    letter-spacing: -0.04em;
    color: #dcfbff;
}

.empty-copy {
    margin: 0.8rem 0 0;
    color: #92b4bf;
    line-height: 1.75;
    font-size: 0.98rem;
}

.meta-strip {
    display: flex;
    justify-content: space-between;
    gap: 0.9rem;
    flex-wrap: wrap;
    margin-top: 2rem;
    color: #7a98a3;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}

.meta-strip span {
    white-space: nowrap;
}

.session-shell {
    margin-top: 1.4rem;
}

.session-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.42rem 0.72rem;
    border-radius: 999px;
    border: 1px solid rgba(77, 239, 255, 0.16);
    background: rgba(7, 21, 29, 0.75);
    color: var(--cyan-bright);
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.session-title {
    margin: 0.85rem 0 0;
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(2rem, 5vw, 3.3rem);
    letter-spacing: -0.05em;
    color: #ecfdff;
}

.session-copy {
    margin: 0.55rem 0 1.25rem;
    color: #95b5bf;
    max-width: 720px;
    line-height: 1.7;
}

.control-caption {
    margin-top: 0.9rem;
    color: #7998a2;
    font-size: 0.88rem;
}

.tensor-header,
.result-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 0.85rem;
}

.panel-tag {
    color: var(--cyan-bright);
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    white-space: nowrap;
}

.panel-title {
    margin: 0;
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.45rem;
    color: #e9fcff;
    letter-spacing: -0.03em;
}

.panel-subtitle {
    margin: 0.35rem 0 0;
    color: #84a5af;
    line-height: 1.55;
    font-size: 0.9rem;
}

.tensor-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.95rem;
}

.tensor-shot span {
    display: block;
    margin-top: 0.55rem;
    color: #8ba9b3;
    font-size: 0.78rem;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.result-value {
    margin-top: 0.55rem;
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    line-height: 0.95;
    letter-spacing: -0.06em;
    color: var(--cyan);
    text-shadow: 0 0 24px rgba(77, 239, 255, 0.18);
}

.result-summary {
    margin-top: 0.8rem;
    color: #9dc0c9;
    line-height: 1.65;
    font-size: 0.93rem;
}

.signal-stack {
    margin-top: 1.15rem;
}

.signal-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.6rem;
    margin: 0.65rem 0 0.28rem;
    color: #b7d8df;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.signal-track {
    width: 100%;
    height: 10px;
    border-radius: 999px;
    background: rgba(77, 239, 255, 0.08);
    overflow: hidden;
    border: 1px solid rgba(136, 221, 233, 0.08);
}

.signal-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #0bb9d9 0%, #71f6ff 100%);
}

.signal-fill.alt {
    background: linear-gradient(90deg, #274a55 0%, #7aa8b7 100%);
}

.tiny-ledger {
    margin-top: 1rem;
    color: #7f9ca7;
    font-size: 0.82rem;
    line-height: 1.65;
}

.tiny-ledger strong {
    color: #c5f6fb;
}

.methodology-card {
    margin-top: 1rem;
}

.methodology-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.95rem;
    margin-top: 1rem;
}

.methodology-unit {
    padding: 0.9rem;
    border-radius: 18px;
    border: 1px solid rgba(136, 221, 233, 0.1);
    background: rgba(12, 20, 31, 0.74);
}

.methodology-unit span {
    display: block;
    color: var(--cyan-bright);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-size: 0.68rem;
}

.methodology-unit p {
    margin: 0.45rem 0 0;
    color: #9ebdc6;
    font-size: 0.9rem;
    line-height: 1.65;
}

.footer-shell {
    margin-top: 3rem;
    padding: 1.35rem 1.5rem;
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(9, 16, 24, 0.96), rgba(6, 10, 16, 0.94));
    border: 1px solid rgba(136, 221, 233, 0.12);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}

.footer-brand {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #dcfbff;
}

.footer-links {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}

.footer-links a {
    text-decoration: none;
    color: #8daab5;
    font-size: 0.86rem;
}

.footer-copy {
    color: #73909a;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}

button[kind],
.stDownloadButton button {
    min-height: 3.1rem;
    border-radius: 16px !important;
    border: 1px solid rgba(136, 221, 233, 0.18) !important;
    background: linear-gradient(180deg, rgba(12, 21, 32, 0.96), rgba(8, 13, 21, 0.96)) !important;
    color: #e1fbff !important;
    text-transform: uppercase !important;
    letter-spacing: 0.16em !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22) !important;
}

button[kind]:hover,
.stDownloadButton button:hover {
    border-color: rgba(109, 247, 255, 0.34) !important;
    color: #ffffff !important;
}

div[data-testid="stDownloadButton"] > button {
    width: 100%;
}

[data-testid="stFileUploaderDropzone"] {
    border-radius: 18px !important;
    border: 1px dashed rgba(109, 247, 255, 0.25) !important;
    background: rgba(7, 16, 24, 0.8) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stMarkdownContainer"] p {
    color: inherit;
}

[data-testid="stFileUploaderFileName"] {
    color: #d8fbff !important;
}

.stAlert {
    border-radius: 20px;
    background: rgba(9, 18, 28, 0.88);
    border: 1px solid rgba(136, 221, 233, 0.16);
}

@keyframes heroDrift {
    0% {
        transform: translateX(-2%) translateY(-1%) scale(1);
    }
    50% {
        transform: translateX(2%) translateY(1%) scale(1.02);
    }
    100% {
        transform: translateX(-2%) translateY(-1%) scale(1);
    }
}

@media (max-width: 980px) {
    .top-nav,
    .section-header,
    .footer-shell {
        flex-direction: column;
        align-items: flex-start;
    }

    .hero-meta,
    .metric-grid,
    .methodology-grid,
    .tensor-grid {
        grid-template-columns: 1fr;
    }

    .nav-links {
        justify-content: flex-start;
    }
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
def load_results_metadata() -> dict[str, Any]:
    if RESULTS_JSON_PATH.exists():
        return json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_model_summary_text() -> str:
    if MODEL_SUMMARY_PATH.exists():
        return MODEL_SUMMARY_PATH.read_text(encoding="utf-8")
    return ""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def bootstrap_state() -> None:
    if "active_image_bytes" not in st.session_state:
        st.session_state.active_image_bytes = None
    if "active_image_name" not in st.session_state:
        st.session_state.active_image_name = None
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0


def to_data_url(image_source: Image.Image | np.ndarray | Path) -> str:
    if isinstance(image_source, Path):
        raw_bytes = image_source.read_bytes()
        suffix = image_source.suffix.lower().replace(".", "") or "png"
        encoded = base64.b64encode(raw_bytes).decode("utf-8")
        return f"data:image/{suffix};base64,{encoded}"

    if isinstance(image_source, np.ndarray):
        image = Image.fromarray(image_source.astype(np.uint8))
    else:
        image = image_source

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def format_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2%}"


def label_display(label: str) -> str:
    return {"female": "FEMENINO", "male": "MASCULINO"}.get(label, label.upper().replace("_", " "))


def experiment_display_name(name: str) -> str:
    return {"regularized": "REGULARIZADO", "baseline": "BASE"}.get(name, name.upper())


def extract_total_params(summary_text: str) -> str:
    match = re.search(r"Total params:\s*([\d,]+)", summary_text)
    return match.group(1) if match else "--"


def describe_focus_zone(heatmap: np.ndarray) -> tuple[str, str]:
    y_coords, x_coords = np.indices(heatmap.shape)
    weights = heatmap.astype(np.float32) + 1e-6
    weighted_y = float((y_coords * weights).sum() / weights.sum())
    weighted_x = float((x_coords * weights).sum() / weights.sum())
    rows, cols = heatmap.shape

    vertical = "superior" if weighted_y < rows / 3 else "central" if weighted_y < 2 * rows / 3 else "inferior"
    horizontal = "izquierda" if weighted_x < cols / 3 else "centro" if weighted_x < 2 * cols / 3 else "derecha"

    highlighted_ratio = float((heatmap > (heatmap.mean() + heatmap.std())).mean())
    spread = "compacta" if highlighted_ratio < 0.18 else "distribuida"
    return f"{vertical}-{horizontal}", spread


def saliency_summary(heatmap: np.ndarray) -> str:
    region, spread = describe_focus_zone(heatmap)
    return (
        f"Los pixeles de mayor sensibilidad aparecen de forma {spread} y se concentran en la zona {region} del rostro, "
        "lo que muestra donde la salida sigmoide cambia con mayor fuerza frente a variaciones locales de textura."
    )


def gradcam_summary(heatmap: np.ndarray, confidence: float) -> str:
    region, spread = describe_focus_zone(heatmap)
    verdict = "refuerza" if confidence >= 0.60 else "solo respalda de forma parcial"
    return (
        f"Las activaciones profundas permanecen {spread} sobre la region {region}, lo que {verdict} la "
        "clasificacion actual al concentrar evidencia semantica y no ruido del fondo."
    )


def set_active_upload(uploaded_file) -> None:
    st.session_state.active_image_bytes = uploaded_file.getvalue()
    st.session_state.active_image_name = uploaded_file.name


def clear_active_image() -> None:
    st.session_state.active_image_bytes = None
    st.session_state.active_image_name = None
    st.session_state.uploader_nonce += 1


def get_active_image() -> tuple[Image.Image | None, str | None]:
    if not st.session_state.active_image_bytes:
        return None, None
    image = Image.open(io.BytesIO(st.session_state.active_image_bytes))
    image.load()
    return image, st.session_state.active_image_name


def analyze_image(pil_image: Image.Image) -> dict[str, Any]:
    model = load_model()
    original_rgb, resized_rgb, input_tensor = prepare_image_from_pil(pil_image, DEFAULT_IMAGE_SIZE)
    prob_male = float(model.predict(input_tensor, verbose=0)[0][0])
    prob_female = 1.0 - prob_male
    predicted_label = CLASS_NAMES[int(prob_male >= 0.5)]
    alternative_label = "female" if predicted_label == "male" else "male"
    confidence = max(prob_male, prob_female)
    saliency = make_saliency_map(model, input_tensor)
    gradcam = make_gradcam_heatmap(model, input_tensor, last_conv_layer_name="last_conv")
    return {
        "original_rgb": original_rgb,
        "resized_rgb": resized_rgb,
        "prob_male": prob_male,
        "prob_female": prob_female,
        "predicted_label": predicted_label,
        "alternative_label": alternative_label,
        "confidence": confidence,
        "saliency": saliency,
        "gradcam": gradcam,
        "saliency_overlay": overlay_heatmap(resized_rgb, saliency),
        "gradcam_overlay": overlay_heatmap(resized_rgb, gradcam),
    }


def build_export_payload(
    metadata: dict[str, Any],
    analysis: dict[str, Any],
    source_name: str,
) -> bytes:
    payload = {
        "fecha_generacion": datetime.now().isoformat(),
        "archivo_origen": source_name,
        "tamano_entrada": list(DEFAULT_IMAGE_SIZE),
        "prediccion": {
            "clase_predicha": analysis["predicted_label"],
            "clase_predicha_mostrada": label_display(analysis["predicted_label"]),
            "confianza": analysis["confidence"],
            "probabilidad_femenino": analysis["prob_female"],
            "probabilidad_masculino": analysis["prob_male"],
            "clase_positiva_interna_modelo": "male",
        },
        "notas_xai": {
            "saliency_map": saliency_summary(analysis["saliency"]),
            "grad_cam": gradcam_summary(analysis["gradcam"], analysis["confidence"]),
        },
        "referencia_modelo": metadata.get("test_metrics", {}),
        "mejor_experimento": metadata.get("best_experiment", {}),
    }
    return json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")


def render_section_header(kicker: str, title: str, copy: str, status: str | None = None) -> None:
    status_html = f'<div class="status-pill">{status}</div>' if status else ""
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-header-copy">
                <div class="section-kicker">{kicker}</div>
                <h2 class="section-title">{title}</h2>
                <p class="section-copy">{copy}</p>
            </div>
            {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_navbar() -> None:
    st.markdown(
        """
        <div id="top" class="section-anchor"></div>
        <nav class="top-nav">
            <a class="brand-lockup" href="#top">PRISMA XAI LAB</a>
            <div class="nav-links">
                <a href="#modelo">Modelo</a>
                <a href="#datos">Datos</a>
                <a href="#analisis">Analisis</a>
                <a href="#reportes">Reportes</a>
            </div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metadata: dict[str, Any]) -> None:
    accuracy = format_pct(metadata.get("test_metrics", {}).get("accuracy"))
    auc = metadata.get("test_metrics", {}).get("auc")
    auc_text = f"{auc:.3f}" if auc is not None else "--"
    best_experiment = experiment_display_name(metadata.get("best_experiment", {}).get("name", "regularized"))
    subset_count = metadata.get("modeling_dataset_count", "--")

    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-content">
                <div class="hero-badge">Vision explicable en linea</div>
                <h1 class="hero-title">ARQUITECTURA VISUAL <span>DEL SEXO</span></h1>
                <p class="hero-copy">
                    Una CNN entrenada desde cero se combina con explicabilidad visual para clasificar
                    rostros en las clases <strong>femenino</strong> y <strong>masculino</strong> con evidencia
                    interpretable. Esta interfaz convierte el laboratorio en una experiencia clara,
                    elegante y lista para presentacion.
                </p>
                <div class="hero-actions">
                    <a class="hero-button primary" href="#analisis">Ir al analisis</a>
                    <a class="hero-button secondary" href="#modelo">Ver modelo</a>
                </div>
            </div>
            <div class="hero-meta">
                <div class="hero-meta-card">
                    <span>Exactitud de prueba</span>
                    <strong>{accuracy}</strong>
                </div>
                <div class="hero-meta-card">
                    <span>AUC de prueba</span>
                    <strong>{auc_text}</strong>
                </div>
                <div class="hero-meta-card">
                    <span>Mejor experimento</span>
                    <strong>{best_experiment}</strong>
                </div>
                <div class="hero-meta-card">
                    <span>Submuestra de modelado</span>
                    <strong>{subset_count}</strong>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card_html(label: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
        <span>{label}</span>
        <strong>{value}</strong>
        <p>{note}</p>
    </div>
    """


def render_metric_row(cards: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(cards), gap="medium")
    for column, (label, value, note) in zip(columns, cards):
        with column:
            st.markdown(metric_card_html(label, value, note), unsafe_allow_html=True)


def render_media_card(
    title: str,
    subtitle: str,
    image_source: Path | np.ndarray | Image.Image,
    note: str = "",
    eyebrow: str = "",
) -> None:
    note_html = f'<div class="card-note">{note}</div>' if note else ""
    eyebrow_html = f'<div class="card-eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="media-card-header">
                {eyebrow_html}
                <h3 class="media-card-title">{title}</h3>
                <p class="media-card-subtitle">{subtitle}</p>
            </div>
            <img class="media-frame" src="{to_data_url(image_source)}" alt="{title}" />
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_models_section(metadata: dict[str, Any], total_params: str, summary_text: str) -> None:
    st.markdown('<div id="modelo" class="section-anchor"></div>', unsafe_allow_html=True)
    render_section_header(
        "Ficha del modelo",
        "Arquitectura experimental y evidencia de entrenamiento",
        "El clasificador se entreno desde cero en TensorFlow-Keras, se comparo en dos configuraciones y la variante regularizada fue seleccionada como artefacto final para el despliegue.",
    )

    best_experiment = metadata.get("best_experiment", {})
    test_metrics = metadata.get("test_metrics", {})
    render_metric_row(
        [
            ("Tensor de entrada", f"{DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]}", "Imagen RGB con resize_with_pad y normalizacion en coma flotante."),
            ("Parametros totales", total_params, "Arquitectura compacta pensada para mantenerse desplegable en Streamlit Cloud."),
            ("Mejor tasa de aprendizaje", str(best_experiment.get("learning_rate", "--")), "Valor elegido en el experimento regularizado ganador."),
            ("Perdida de prueba", f"{test_metrics.get('loss', 0):.4f}" if test_metrics.get("loss") is not None else "--", "Perdida final sobre el conjunto de prueba reservado."),
        ]
    )

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        if FIGURE_PATHS["hyperparameter_comparison"].exists():
            render_media_card(
                "Comparacion de hiperparametros",
                "Curvas de exactitud y perdida de validacion para la configuracion base y la regularizada.",
                FIGURE_PATHS["hyperparameter_comparison"],
                note=(
                    f"<strong>Conclusion:</strong> La configuracion <strong>{experiment_display_name(best_experiment.get('name', 'regularized'))}</strong> "
                    f"fue la ganadora con una mejor exactitud de validacion de {format_pct(best_experiment.get('best_val_accuracy'))}."
                ),
                eyebrow="Exploracion experimental",
            )
    with col_right:
        if FIGURE_PATHS["training_final"].exists():
            render_media_card(
                "Trayectoria final de entrenamiento",
                "Curvas de aprendizaje del experimento seleccionado para despliegue.",
                FIGURE_PATHS["training_final"],
                note=(
                    f"<strong>Punto de control:</strong> mejor epoca {best_experiment.get('best_epoch', '--')} con "
                    f"perdida de validacion de {best_experiment.get('best_val_loss', 0):.4f}."
                    if best_experiment.get("best_val_loss") is not None
                    else ""
                ),
                eyebrow="Curva de entrenamiento",
            )

    with st.expander("Resumen textual de la arquitectura"):
        st.code(summary_text or "No hay un resumen disponible del modelo.", language="text")


def render_datasets_section(metadata: dict[str, Any]) -> None:
    st.markdown('<div id="datos" class="section-anchor"></div>', unsafe_allow_html=True)
    render_section_header(
        "Inteligencia de datos",
        "Cobertura, estratificacion y logica de muestreo",
        "La etapa exploratoria reviso el conjunto completo de rostros femenino/masculino, mientras que el modelado utilizo una submuestra balanceada y estratificada para mantener el entrenamiento eficiente y reproducible.",
    )

    dataset_summary = metadata.get("dataset_summary", {})
    female_count = dataset_summary.get("female", {}).get("count", "--")
    male_count = dataset_summary.get("male", {}).get("count", "--")
    split_counts = metadata.get("split_counts", {})
    split_text = f"{split_counts.get('train', '--')} / {split_counts.get('validation', '--')} / {split_counts.get('test', '--')}"

    render_metric_row(
        [
            ("Imagenes femeninas", f"{female_count}", "Conteo observado durante la inspeccion del conjunto original."),
            ("Imagenes masculinas", f"{male_count}", "Conteo observado durante la inspeccion del conjunto original."),
            ("Submuestra de modelado", f"{metadata.get('modeling_dataset_count', '--')}", "Submuestra balanceada usada para entrenamiento, validacion y prueba."),
            ("Particiones T/V/P", split_text, "Cantidad de ejemplos en entrenamiento, validacion y prueba."),
        ]
    )

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        if FIGURE_PATHS["dataset_mosaic"].exists():
            render_media_card(
                "Mosaico del conjunto",
                "Vista cualitativa de la variabilidad observada durante la etapa exploratoria.",
                FIGURE_PATHS["dataset_mosaic"],
                note="El mosaico ayuda a revisar poses, iluminacion y textura antes de iniciar el ajuste del modelo.",
                eyebrow="Lectura cualitativa",
            )
    with col_right:
        if FIGURE_PATHS["class_distribution"].exists():
            render_media_card(
                "Distribucion de clases",
                "Conteos observados por etiqueta antes de construir la submuestra.",
                FIGURE_PATHS["class_distribution"],
                note="La submuestra de modelado se obligo a permanecer balanceada aunque el conjunto fuente solo este aproximadamente balanceado.",
                eyebrow="Auditoria de balance",
            )


def render_empty_workbench() -> None:
    st.markdown(
        """
        <div class="glass-card empty-workbench">
            <div class="empty-center">
                <div class="empty-icon">⌘</div>
                <h3 class="empty-title">Sube La Imagen Para Analizar</h3>
                <p class="empty-copy">
                    La aplicacion recibe imagenes faciales en formato JPG, JPEG o PNG, las convierte
                    al tensor 96x96 RGB del modelo y ejecuta automaticamente la prediccion junto con
                    Saliency Map y Grad-CAM.
                </p>
            </div>
            <div class="meta-strip">
                <span>Preprocesamiento: RGB + padding</span>
                <span>Visualizacion: Saliency + Grad-CAM</span>
                <span>Modelo: CNN + XAI v1.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_header(source_name: str) -> None:
    st.markdown(
        f"""
        <div class="session-shell">
            <div class="session-badge">Analisis activo</div>
            <h3 class="session-title">Resultado de la inferencia</h3>
            <p class="session-copy">
                Se esta evaluando la activacion de la red sobre la imagen <strong>{source_name}</strong>
                para producir una decision de clasificacion por sexo y su evidencia visual asociada.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_tensor_card(original_rgb: np.ndarray, resized_rgb: np.ndarray, source_name: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="tensor-header">
                <div>
                    <h3 class="panel-title">Analisis del tensor de entrada</h3>
                    <p class="panel-subtitle">Archivo fuente: {source_name}</p>
                </div>
                <div class="panel-tag">Fase I</div>
            </div>
            <div class="tensor-grid">
                <div class="tensor-shot">
                    <img src="{to_data_url(original_rgb)}" alt="Original input" />
                    <span>Imagen original</span>
                </div>
                <div class="tensor-shot">
                    <img src="{to_data_url(resized_rgb)}" alt="Preprocessed tensor" />
                    <span>Tensor 96x96</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(analysis: dict[str, Any], source_name: str) -> None:
    prob_female = analysis["prob_female"]
    prob_male = analysis["prob_male"]
    predicted = analysis["predicted_label"]
    alternative = analysis["alternative_label"]
    confidence = analysis["confidence"]
    margin = abs(prob_male - prob_female)
    confidence_note = (
        "Clasificacion de alta confianza"
        if confidence >= 0.75
        else "Clasificacion de confianza media"
        if confidence >= 0.60
        else "Clasificacion con margen estrecho"
    )
    predicted_probability = prob_male if predicted == "male" else prob_female
    alternative_probability = prob_female if predicted == "male" else prob_male

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="result-header">
                <div>
                    <div class="card-eyebrow">Resultado de clasificacion</div>
                    <div class="result-value">{label_display(predicted)}</div>
                    <p class="panel-subtitle">{confidence_note} obtenido a partir de <strong>{source_name}</strong>.</p>
                </div>
                <div class="panel-tag">Inferencia en vivo</div>
            </div>
            <div class="signal-stack">
                <div class="signal-row">
                    <span>Confianza</span>
                    <span>{predicted_probability:.1%}</span>
                </div>
                <div class="signal-track">
                    <div class="signal-fill" style="width: {predicted_probability * 100:.2f}%"></div>
                </div>
                <div class="signal-row">
                    <span>Alternativa ({label_display(alternative)})</span>
                    <span>{alternative_probability:.1%}</span>
                </div>
                <div class="signal-track">
                    <div class="signal-fill alt" style="width: {alternative_probability * 100:.2f}%"></div>
                </div>
            </div>
            <div class="result-summary">
                La salida sigmoide del modelo esta calibrada sobre la clase interna <strong>male</strong>.
                El margen de decision en este caso es de <strong>{margin:.1%}</strong>, por eso la probabilidad
                complementaria de la otra clase tambien se muestra para la interpretacion.
            </div>
            <div class="tiny-ledger">
                <strong>Femenino:</strong> {prob_female:.2%}<br/>
                <strong>Masculino:</strong> {prob_male:.2%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_methodology_card(metadata: dict[str, Any], analysis: dict[str, Any]) -> None:
    best = metadata.get("best_experiment", {})
    test_metrics = metadata.get("test_metrics", {})
    auc = test_metrics.get("auc")
    auc_text = f"{auc:.3f}" if auc is not None else "--"
    st.markdown(
        f"""
        <div class="glass-card methodology-card">
            <div class="media-card-header">
                <div class="card-eyebrow">Notas tecnicas y metodologia</div>
                <h3 class="media-card-title">Como se produjo esta inferencia</h3>
                <p class="media-card-subtitle">
                    Cada paso de abajo esta alineado con el pipeline de entrenamiento usado en el notebook y en la entrega final del laboratorio.
                </p>
            </div>
            <div class="methodology-grid">
                <div class="methodology-unit">
                    <span>Preprocesamiento</span>
                    <p>Conversion a RGB, normalizacion a float32 y <code>resize_with_pad</code> a {DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]} antes de entrar a la CNN.</p>
                </div>
                <div class="methodology-unit">
                    <span>Modelo</span>
                    <p>Mejor experimento: <strong>{experiment_display_name(best.get('name', 'regularized'))}</strong> con dropout {best.get('dropout_rate', '--')} y tasa de aprendizaje {best.get('learning_rate', '--')}.</p>
                </div>
                <div class="methodology-unit">
                    <span>Referencia</span>
                    <p>Exactitud de prueba de {format_pct(test_metrics.get('accuracy'))} y AUC de {auc_text} para el modelo desplegado.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workbench_section(metadata: dict[str, Any]) -> None:
    st.markdown('<div id="analisis" class="section-anchor"></div>', unsafe_allow_html=True)
    workbench_status = "Esperando imagen" if st.session_state.active_image_bytes is None else "Analisis cargado"
    render_section_header(
        "Laboratorio interactivo",
        "Area de analisis",
        "Aqui ocurre el flujo central de la aplicacion: subir una imagen, preprocesarla, clasificarla y visualizar la evidencia de Saliency Map y Grad-CAM sobre la misma entrada analizada.",
        status=workbench_status,
    )

    uploader_key = f"main_uploader_{st.session_state.uploader_nonce}"
    uploaded_file = st.file_uploader(
        "Selecciona o arrastra una imagen facial",
        type=["jpg", "jpeg", "png"],
        key=uploader_key,
    )
    st.caption("El analisis comienza automaticamente cuando la imagen se carga.")

    if uploaded_file is not None:
        set_active_upload(uploaded_file)

    if st.session_state.active_image_bytes is None:
        render_empty_workbench()
        return

    render_analysis_header(st.session_state.active_image_name or "imagen cargada")
    source_cols = st.columns([1.45, 0.55], gap="medium")
    with source_cols[0]:
        st.markdown(
            '<p class="control-caption">Si subes otra imagen en el cargador de arriba, el analisis actual se reemplaza automaticamente por el nuevo.</p>',
            unsafe_allow_html=True,
        )
    with source_cols[1]:
        if st.button("Limpiar analisis", use_container_width=True):
            clear_active_image()
            st.rerun()

    active_image, source_name = get_active_image()
    if active_image is None or source_name is None:
        st.error("No fue posible recuperar la imagen activa de la sesion.")
        return

    with st.spinner("Ejecutando la inferencia y generando los mapas de atribucion..."):
        analysis = analyze_image(active_image)

    top_left, top_right = st.columns([1.35, 1.0], gap="large")
    with top_left:
        render_input_tensor_card(analysis["original_rgb"], analysis["resized_rgb"], source_name)
    with top_right:
        render_prediction_card(analysis, source_name)
        st.download_button(
            "Descargar reporte JSON",
            data=build_export_payload(metadata, analysis, source_name),
            file_name="reporte_analisis_xai.json",
            mime="application/json",
            use_container_width=True,
        )

    bottom_left, bottom_right = st.columns(2, gap="large")
    with bottom_left:
        render_media_card(
            "Saliency Map",
            "Gradiente de atribucion a nivel de pixel sobre el tensor exacto que consumio la CNN.",
            analysis["saliency_overlay"],
            note=f"<strong>Lectura:</strong> {saliency_summary(analysis['saliency'])}",
            eyebrow="Campo de atribucion",
        )
    with bottom_right:
        render_media_card(
            "Grad-CAM",
            "Mapa de activacion por clase construido desde las representaciones convolucionales profundas.",
            analysis["gradcam_overlay"],
            note=f"<strong>Lectura:</strong> {gradcam_summary(analysis['gradcam'], analysis['confidence'])}",
            eyebrow="Traza de activacion",
        )

    render_methodology_card(metadata, analysis)


def render_archives_section() -> None:
    st.markdown('<div id="reportes" class="section-anchor"></div>', unsafe_allow_html=True)
    render_section_header(
        "Artefactos y reporte",
        "Archivos del despliegue y evidencia visual",
        "Esta seccion expone figuras, notebook y metricas que respaldan la entrega final del laboratorio y sirven como soporte tecnico del despliegue.",
    )

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        if FIGURE_PATHS["confusion_matrix"].exists():
            render_media_card(
                "Matriz de confusion",
                "Desempeno sobre el conjunto de prueba por etiquetas reales y predichas.",
                FIGURE_PATHS["confusion_matrix"],
                note="Sirve para revisar si el clasificador tiende a favorecer una clase por encima de la otra.",
                eyebrow="Artefacto de evaluacion",
            )
    with col_right:
        if FIGURE_PATHS["xai_example"].exists():
            render_media_card(
                "Ejemplo de XAI de referencia",
                "Figura compuesta generada durante el pipeline del laboratorio para una imagen correctamente clasificada.",
                FIGURE_PATHS["xai_example"],
                note="Este artefacto muestra como luce una salida completa del pipeline de interpretabilidad fuera de la interfaz interactiva.",
                eyebrow="Artefacto de interpretabilidad",
            )

    download_cols = st.columns(3, gap="medium")
    with download_cols[0]:
        if RESULTS_JSON_PATH.exists():
            st.download_button(
                "Descargar resultados JSON",
                data=RESULTS_JSON_PATH.read_bytes(),
                file_name="lab_results.json",
                mime="application/json",
                use_container_width=True,
            )
    with download_cols[1]:
        if NOTEBOOK_PATH.exists():
            st.download_button(
                "Descargar notebook",
                data=NOTEBOOK_PATH.read_bytes(),
                file_name="cnn_gender_lab.ipynb",
                mime="application/json",
                use_container_width=True,
            )
    with download_cols[2]:
        if MODEL_SUMMARY_PATH.exists():
            st.download_button(
                "Descargar resumen del modelo",
                data=MODEL_SUMMARY_PATH.read_text(encoding="utf-8"),
                file_name="model_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )


def render_protocol_section() -> None:
    render_section_header(
        "Documentacion de soporte",
        "Etica, seguridad, alcance legal y privacidad",
        "Estas tarjetas le dan contenido real a la capa documental de la interfaz para que el footer apunte a secciones concretas y utiles.",
    )

    cards = [
        (
            "etica",
            "Protocolo etico",
            "Este clasificador es una demostracion academica de CNN supervisada con explicabilidad. Sus predicciones no deben interpretarse como identidad, valor o verdad absoluta, y el alcance demografico esta limitado por el conjunto original.",
        ),
        (
            "seguridad",
            "Documentacion de seguridad",
            "La interfaz esta pensada para experimentacion controlada. Cada prediccion debe leerse junto con Saliency Map y Grad-CAM para detectar evidencia debil, sesgo de fondo o activaciones inestables antes de sacar conclusiones.",
        ),
        (
            "legal",
            "Aspectos legales",
            "El repositorio contiene el modelo entrenado, el notebook y los artefactos generados, pero no el conjunto fuente completo. El uso, redistribucion y atribucion del conjunto siguen sujetos a los terminos del proveedor original.",
        ),
        (
            "privacidad",
            "Privacidad",
            "Las imagenes cargadas se procesan dentro de la sesion de la app para inferencia y visualizacion. El despliegue no esta pensado como sistema de almacenamiento, y el repositorio solo conserva artefactos del modelo y reportes generados.",
        ),
    ]

    first_row = st.columns(2, gap="large")
    second_row = st.columns(2, gap="large")
    for column, (anchor, title, body) in zip(first_row + second_row, cards):
        with column:
            st.markdown(f'<div id="{anchor}" class="section-anchor"></div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="media-card-header">
                        <div class="card-eyebrow">Nota documental</div>
                        <h3 class="media-card-title">{title}</h3>
                    </div>
                    <div class="card-note">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_footer() -> None:
    st.markdown(
        """
        <section class="footer-shell">
            <div class="footer-brand">PRISMA XAI LAB</div>
            <div class="footer-links">
                <a href="#etica">Etica</a>
                <a href="#seguridad">Seguridad</a>
                <a href="#legal">Legal</a>
                <a href="#privacidad">Privacidad</a>
            </div>
            <div class="footer-copy">2026 Prisma XAI Lab · despliegue CNN + XAI</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    bootstrap_state()
    metadata = load_results_metadata()
    summary_text = load_model_summary_text()
    total_params = extract_total_params(summary_text)

    render_navbar()
    render_hero(metadata)
    render_models_section(metadata, total_params, summary_text)
    render_datasets_section(metadata)

    try:
        render_workbench_section(metadata)
    except FileNotFoundError as error:
        st.error(str(error))

    render_archives_section()
    render_protocol_section()
    render_footer()


if __name__ == "__main__":
    main()
