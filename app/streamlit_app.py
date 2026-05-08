from __future__ import annotations

import base64
import io
import json
import re
import secrets
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
    page_title="Aether Research | CNN Gender Classifier + XAI",
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
    if "session_code" not in st.session_state:
        st.session_state.session_code = f"AX-{secrets.randbelow(900) + 100}"
    if "active_image_bytes" not in st.session_state:
        st.session_state.active_image_bytes = None
    if "active_image_name" not in st.session_state:
        st.session_state.active_image_name = None
    if "active_source" not in st.session_state:
        st.session_state.active_source = "idle"


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
    return label.upper().replace("_", " ")


def extract_total_params(summary_text: str) -> str:
    match = re.search(r"Total params:\s*([\d,]+)", summary_text)
    return match.group(1) if match else "--"


def describe_focus_zone(heatmap: np.ndarray) -> tuple[str, str]:
    y_coords, x_coords = np.indices(heatmap.shape)
    weights = heatmap.astype(np.float32) + 1e-6
    weighted_y = float((y_coords * weights).sum() / weights.sum())
    weighted_x = float((x_coords * weights).sum() / weights.sum())
    rows, cols = heatmap.shape

    vertical = "upper" if weighted_y < rows / 3 else "central" if weighted_y < 2 * rows / 3 else "lower"
    horizontal = "left" if weighted_x < cols / 3 else "center" if weighted_x < 2 * cols / 3 else "right"

    highlighted_ratio = float((heatmap > (heatmap.mean() + heatmap.std())).mean())
    spread = "compact" if highlighted_ratio < 0.18 else "distributed"
    return f"{vertical}-{horizontal}", spread


def saliency_summary(heatmap: np.ndarray) -> str:
    region, spread = describe_focus_zone(heatmap)
    return (
        f"High-sensitivity pixels remain {spread} and cluster around the {region} facial zone, "
        "which indicates where the sigmoid output reacts most strongly to local texture changes."
    )


def gradcam_summary(heatmap: np.ndarray, confidence: float) -> str:
    region, spread = describe_focus_zone(heatmap)
    verdict = "reinforces" if confidence >= 0.60 else "only partially supports"
    return (
        f"Deeper activations stay {spread} over the {region} region, which {verdict} the current "
        "classification by concentrating semantic evidence instead of background noise."
    )


def load_demo_image() -> Image.Image:
    composite = Image.open(FIGURE_PATHS["xai_example"]).convert("RGB")
    width, height = composite.size
    crop = composite.crop(
        (
            int(width * 0.012),
            int(height * 0.11),
            int(width * 0.295),
            int(height * 0.975),
        )
    )
    return crop


def set_active_upload(uploaded_file) -> None:
    st.session_state.active_image_bytes = uploaded_file.getvalue()
    st.session_state.active_image_name = uploaded_file.name
    st.session_state.active_source = "upload"


def set_demo_source() -> None:
    image = load_demo_image()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    st.session_state.active_image_bytes = buffer.getvalue()
    st.session_state.active_image_name = "aether_demo_stream.png"
    st.session_state.active_source = "demo"


def clear_active_source() -> None:
    st.session_state.active_image_bytes = None
    st.session_state.active_image_name = None
    st.session_state.active_source = "idle"


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
        "session_code": st.session_state.session_code,
        "generated_at": datetime.now().isoformat(),
        "source_name": source_name,
        "source_mode": st.session_state.active_source,
        "image_size": list(DEFAULT_IMAGE_SIZE),
        "prediction": {
            "predicted_label": analysis["predicted_label"],
            "confidence": analysis["confidence"],
            "prob_female": analysis["prob_female"],
            "prob_male": analysis["prob_male"],
        },
        "xai_notes": {
            "saliency": saliency_summary(analysis["saliency"]),
            "gradcam": gradcam_summary(analysis["gradcam"], analysis["confidence"]),
        },
        "model_performance_reference": metadata.get("test_metrics", {}),
        "best_experiment": metadata.get("best_experiment", {}),
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
            <a class="brand-lockup" href="#top">AETHER RESEARCH</a>
            <div class="nav-links">
                <a href="#models">Models</a>
                <a href="#datasets">Datasets</a>
                <a href="#workbench">Workbench</a>
                <a href="#archives">Archives</a>
            </div>
            <a class="nav-cta" href="#workbench">Initialize Session</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metadata: dict[str, Any]) -> None:
    accuracy = format_pct(metadata.get("test_metrics", {}).get("accuracy"))
    auc = metadata.get("test_metrics", {}).get("auc")
    auc_text = f"{auc:.3f}" if auc is not None else "--"
    best_experiment = metadata.get("best_experiment", {}).get("name", "regularized").upper()
    subset_count = metadata.get("modeling_dataset_count", "--")

    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-content">
                <div class="hero-badge">Explainable vision systems online</div>
                <h1 class="hero-title">THE ARCHITECTURE OF <span>IDENTITY</span></h1>
                <p class="hero-copy">
                    Deep neural analysis meets explainable AI for high-precision gender classification.
                    This interface turns the laboratory CNN into a polished research workstation where
                    every prediction is paired with visual evidence and methodological context.
                </p>
                <div class="hero-actions">
                    <a class="hero-button primary" href="#workbench">Initialize Analysis</a>
                    <a class="hero-button secondary" href="#models">Review Model Dossier</a>
                </div>
            </div>
            <div class="hero-meta">
                <div class="hero-meta-card">
                    <span>Test accuracy</span>
                    <strong>{accuracy}</strong>
                </div>
                <div class="hero-meta-card">
                    <span>Test AUC</span>
                    <strong>{auc_text}</strong>
                </div>
                <div class="hero-meta-card">
                    <span>Best experiment</span>
                    <strong>{best_experiment}</strong>
                </div>
                <div class="hero-meta-card">
                    <span>Model subset</span>
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
    st.markdown('<div id="models" class="section-anchor"></div>', unsafe_allow_html=True)
    render_section_header(
        "Model dossier",
        "Experimental CNN architecture and training evidence",
        "The classifier was trained from scratch in TensorFlow-Keras, compared across two configurations, and retained the regularized variant as the best-performing configuration for the final deployment artifact.",
    )

    best_experiment = metadata.get("best_experiment", {})
    test_metrics = metadata.get("test_metrics", {})
    render_metric_row(
        [
            ("Input tensor", f"{DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]}", "RGB tensor with resize_with_pad and float normalization."),
            ("Total params", total_params, "Compact architecture designed to remain deployable in Streamlit Cloud."),
            ("Best learning rate", str(best_experiment.get("learning_rate", "--")), "Selected from the winning regularized experiment."),
            ("Test loss", f"{test_metrics.get('loss', 0):.4f}" if test_metrics.get("loss") is not None else "--", "Final evaluation loss on the held-out test split."),
        ]
    )

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        if FIGURE_PATHS["hyperparameter_comparison"].exists():
            render_media_card(
                "Hyperparameter comparison",
                "Validation accuracy and loss traces for the baseline and regularized configurations.",
                FIGURE_PATHS["hyperparameter_comparison"],
                note=(
                    f"<strong>Decision:</strong> The <strong>{best_experiment.get('name', 'regularized')}</strong> setup "
                    f"won with a best validation accuracy of {format_pct(best_experiment.get('best_val_accuracy'))}."
                ),
                eyebrow="Experiment scan",
            )
    with col_right:
        if FIGURE_PATHS["training_final"].exists():
            render_media_card(
                "Final training trajectory",
                "Learning curves from the experiment selected for deployment.",
                FIGURE_PATHS["training_final"],
                note=(
                    f"<strong>Checkpoint:</strong> Best epoch {best_experiment.get('best_epoch', '--')} with "
                    f"validation loss {best_experiment.get('best_val_loss', 0):.4f}."
                    if best_experiment.get("best_val_loss") is not None
                    else ""
                ),
                eyebrow="Training curve",
            )

    with st.expander("Model architecture digest"):
        st.code(summary_text or "Model summary not available.", language="text")


def render_datasets_section(metadata: dict[str, Any]) -> None:
    st.markdown('<div id="datasets" class="section-anchor"></div>', unsafe_allow_html=True)
    render_section_header(
        "Dataset intelligence",
        "Coverage, stratification and sampling logic",
        "The exploration stage scanned the full male/female face dataset, while the modeling stage used a balanced stratified subset to keep CNN training efficient and reproducible for deployment.",
    )

    dataset_summary = metadata.get("dataset_summary", {})
    female_count = dataset_summary.get("female", {}).get("count", "--")
    male_count = dataset_summary.get("male", {}).get("count", "--")
    split_counts = metadata.get("split_counts", {})
    split_text = f"{split_counts.get('train', '--')} / {split_counts.get('validation', '--')} / {split_counts.get('test', '--')}"

    render_metric_row(
        [
            ("Female images", f"{female_count}", "Exploration count observed in the original dataset scan."),
            ("Male images", f"{male_count}", "Exploration count observed in the original dataset scan."),
            ("Modeling subset", f"{metadata.get('modeling_dataset_count', '--')}", "Balanced subset used for the train/validation/test workflow."),
            ("Splits T/V/T", split_text, "Counts for train, validation and test partitions."),
        ]
    )

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        if FIGURE_PATHS["dataset_mosaic"].exists():
            render_media_card(
                "Dataset mosaic",
                "Qualitative preview of the dataset variability observed during the exploration stage.",
                FIGURE_PATHS["dataset_mosaic"],
                note="The mosaic helps inspect pose, lighting and texture diversity before any model fitting begins.",
                eyebrow="Qualitative scan",
            )
    with col_right:
        if FIGURE_PATHS["class_distribution"].exists():
            render_media_card(
                "Class distribution",
                "Observed label counts across the original dataset before subsampling.",
                FIGURE_PATHS["class_distribution"],
                note="The modeling subset was forced to remain balanced even though the source dataset is only approximately balanced.",
                eyebrow="Balance audit",
            )


def render_empty_workbench() -> None:
    st.markdown(
        """
        <div class="glass-card empty-workbench">
            <div class="empty-center">
                <div class="empty-icon">⌘</div>
                <h3 class="empty-title">Drop Entity Data Here</h3>
                <p class="empty-copy">
                    Upload raw facial imagery or activate the internal demo stream. The inference engine
                    is configured for JPG, JPEG and PNG archives and will convert every input into the
                    exact 96x96 RGB tensor consumed by the deployed CNN.
                </p>
            </div>
            <div class="meta-strip">
                <span>Encryption: AES-256 active</span>
                <span>Latency target: &lt;120 ms</span>
                <span>Model: CNN + XAI v1.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_session_header() -> None:
    st.markdown(
        f"""
        <div class="session-shell">
            <div class="session-badge">Analysis workspace active</div>
            <h3 class="session-title">Session {st.session_state.session_code}</h3>
            <p class="session-copy">
                Evaluating neural pathway activation for high-resolution demographic classification protocols.
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
                    <h3 class="panel-title">Input Tensor Analysis</h3>
                    <p class="panel-subtitle">Source archive: {source_name}</p>
                </div>
                <div class="panel-tag">Phase I</div>
            </div>
            <div class="tensor-grid">
                <div class="tensor-shot">
                    <img src="{to_data_url(original_rgb)}" alt="Original input" />
                    <span>Raw archive</span>
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
        "High-confidence classification" if confidence >= 0.75 else "Moderate-confidence classification" if confidence >= 0.60 else "Low-margin classification"
    )
    predicted_probability = prob_male if predicted == "male" else prob_female
    alternative_probability = prob_female if predicted == "male" else prob_male

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="result-header">
                <div>
                    <div class="card-eyebrow">Classification result</div>
                    <div class="result-value">{label_display(predicted)}</div>
                    <p class="panel-subtitle">{confidence_note} sourced from <strong>{source_name}</strong>.</p>
                </div>
                <div class="panel-tag">Live inference</div>
            </div>
            <div class="signal-stack">
                <div class="signal-row">
                    <span>Confidence</span>
                    <span>{predicted_probability:.1%}</span>
                </div>
                <div class="signal-track">
                    <div class="signal-fill" style="width: {predicted_probability * 100:.2f}%"></div>
                </div>
                <div class="signal-row">
                    <span>Alternative ({alternative})</span>
                    <span>{alternative_probability:.1%}</span>
                </div>
                <div class="signal-track">
                    <div class="signal-fill alt" style="width: {alternative_probability * 100:.2f}%"></div>
                </div>
            </div>
            <div class="result-summary">
                The sigmoid output is calibrated on the <strong>male</strong> class. The decision margin for this
                session is <strong>{margin:.1%}</strong>, so the complementary probability assigned to
                <strong>female</strong> remains visible for interpretation.
            </div>
            <div class="tiny-ledger">
                <strong>Female:</strong> {prob_female:.2%}<br/>
                <strong>Male:</strong> {prob_male:.2%}
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
                <div class="card-eyebrow">Technical notes & methodology</div>
                <h3 class="media-card-title">How this inference was produced</h3>
                <p class="media-card-subtitle">
                    Every step below is aligned with the training pipeline used in the notebook and the final lab submission.
                </p>
            </div>
            <div class="methodology-grid">
                <div class="methodology-unit">
                    <span>Preprocess</span>
                    <p>RGB conversion, float32 normalization and <code>resize_with_pad</code> to {DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]} before entering the CNN.</p>
                </div>
                <div class="methodology-unit">
                    <span>Model</span>
                    <p>Best experiment: <strong>{best.get('name', 'regularized')}</strong> with dropout {best.get('dropout_rate', '--')} and learning rate {best.get('learning_rate', '--')}.</p>
                </div>
                <div class="methodology-unit">
                    <span>Reference</span>
                    <p>Held-out test accuracy {format_pct(test_metrics.get('accuracy'))} and AUC {auc_text} for the deployed checkpoint.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workbench_section(metadata: dict[str, Any]) -> None:
    st.markdown('<div id="workbench" class="section-anchor"></div>', unsafe_allow_html=True)
    workbench_status = "System idle" if st.session_state.active_source == "idle" else "Session armed"
    render_section_header(
        "Interactive workstation",
        "Analysis Workbench",
        "The workbench transitions from a cinematic landing state into a live inference surface once an archive or demo stream is loaded. All actions below are wired to the deployed model and XAI pipeline.",
        status=workbench_status,
    )

    if st.session_state.active_image_bytes is None:
        render_empty_workbench()
        control_a, control_b = st.columns([1.1, 1.1], gap="large")
        with control_a:
            with st.popover("Browse Archive", use_container_width=True):
                uploaded_file = st.file_uploader(
                    "Sube una imagen facial",
                    type=["jpg", "jpeg", "png"],
                    key="archive_uploader",
                )
                st.caption("También puedes arrastrar la imagen directamente dentro del área de carga.")
                if uploaded_file is not None:
                    set_active_upload(uploaded_file)
                    st.rerun()
        with control_b:
            if st.button("Connect Stream", use_container_width=True):
                set_demo_source()
                st.rerun()

        st.markdown(
            '<p class="control-caption">Browse Archive abre una carga real de archivos. Connect Stream inyecta una muestra interna derivada del artefacto XAI del laboratorio para que la demo siga funcionando incluso en Streamlit Cloud.</p>',
            unsafe_allow_html=True,
        )
        return

    render_session_header()
    source_cols = st.columns([1.2, 1.2, 0.9], gap="medium")
    with source_cols[0]:
        with st.popover("Load New Archive", use_container_width=True):
            uploaded_file = st.file_uploader(
                "Reemplaza la imagen actual",
                type=["jpg", "jpeg", "png"],
                key="replacement_uploader",
            )
            if uploaded_file is not None:
                set_active_upload(uploaded_file)
                st.rerun()
    with source_cols[1]:
        if st.button("Replay Demo Stream", use_container_width=True):
            set_demo_source()
            st.rerun()
    with source_cols[2]:
        if st.button("Reset Session", use_container_width=True):
            clear_active_source()
            st.rerun()

    active_image, source_name = get_active_image()
    if active_image is None or source_name is None:
        st.error("No fue posible recuperar la imagen activa de la sesión.")
        return

    with st.spinner("Executing live inference and generating attribution maps..."):
        analysis = analyze_image(active_image)

    top_left, top_right = st.columns([1.35, 1.0], gap="large")
    with top_left:
        render_input_tensor_card(analysis["original_rgb"], analysis["resized_rgb"], source_name)
    with top_right:
        render_prediction_card(analysis, source_name)
        st.download_button(
            "Export Metadata",
            data=build_export_payload(metadata, analysis, source_name),
            file_name=f"{st.session_state.session_code.lower()}_metadata.json",
            mime="application/json",
            use_container_width=True,
        )

    bottom_left, bottom_right = st.columns(2, gap="large")
    with bottom_left:
        render_media_card(
            "Saliency Map",
            "Pixel-level attribution gradient over the exact tensor processed by the CNN.",
            analysis["saliency_overlay"],
            note=f"<strong>Insight:</strong> {saliency_summary(analysis['saliency'])}",
            eyebrow="Attribution field",
        )
    with bottom_right:
        render_media_card(
            "Grad-CAM",
            "Class activation mapping based on the final convolutional representations.",
            analysis["gradcam_overlay"],
            note=f"<strong>Metadata:</strong> {gradcam_summary(analysis['gradcam'], analysis['confidence'])}",
            eyebrow="Activation trace",
        )

    render_methodology_card(metadata, analysis)


def render_archives_section() -> None:
    st.markdown('<div id="archives" class="section-anchor"></div>', unsafe_allow_html=True)
    render_section_header(
        "Artifacts & reporting",
        "Deployment archives and visual evidence",
        "This section exposes the figures, notebook and metric files that support the final submission. It doubles as the destination for the top navigation archive tab and the footer documentation links.",
    )

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        if FIGURE_PATHS["confusion_matrix"].exists():
            render_media_card(
                "Confusion matrix",
                "Held-out test performance by predicted and true labels.",
                FIGURE_PATHS["confusion_matrix"],
                note="Useful for checking whether the classifier tends to favor one class over the other.",
                eyebrow="Evaluation artifact",
            )
    with col_right:
        if FIGURE_PATHS["xai_example"].exists():
            render_media_card(
                "Reference XAI example",
                "Composite figure generated during the lab pipeline for a correctly classified image.",
                FIGURE_PATHS["xai_example"],
                note="This same artifact is also used to derive the internal demo stream embedded in the deployed interface.",
                eyebrow="Interpretability artifact",
            )

    download_cols = st.columns(3, gap="medium")
    with download_cols[0]:
        if RESULTS_JSON_PATH.exists():
            st.download_button(
                "Download Results JSON",
                data=RESULTS_JSON_PATH.read_bytes(),
                file_name="lab_results.json",
                mime="application/json",
                use_container_width=True,
            )
    with download_cols[1]:
        if NOTEBOOK_PATH.exists():
            st.download_button(
                "Download Notebook",
                data=NOTEBOOK_PATH.read_bytes(),
                file_name="cnn_gender_lab.ipynb",
                mime="application/json",
                use_container_width=True,
            )
    with download_cols[2]:
        if MODEL_SUMMARY_PATH.exists():
            st.download_button(
                "Download Model Summary",
                data=MODEL_SUMMARY_PATH.read_text(encoding="utf-8"),
                file_name="model_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )


def render_protocol_section() -> None:
    render_section_header(
        "Documentation deck",
        "Ethics, safety, legal scope and privacy handling",
        "These cards provide the short-form protocol layer behind the interface so the footer links lead somewhere concrete and presentation-ready instead of acting like empty placeholders.",
    )

    cards = [
        (
            "ethics",
            "Ethics Protocol",
            "This classifier is an academic demonstration of supervised CNN classification and explainability. Its predictions should never be interpreted as identity, value or worth, and the demographic framing remains limited by the original dataset.",
        ),
        (
            "safety",
            "Safety Documentation",
            "The interface is intended for controlled experimentation. Every prediction should be read together with Saliency Map and Grad-CAM outputs to detect weak evidence, background bias or unstable activations before drawing conclusions.",
        ),
        (
            "legal",
            "Legal",
            "The repository contains the trained model, notebook and generated artifacts, but not the full source dataset. Dataset use, redistribution and attribution remain subject to the original provider terms referenced in the project documentation.",
        ),
        (
            "privacy",
            "Privacy",
            "Uploaded images are processed in-session for inference and visualization. The deployment is not designed as a storage system, and the repo only persists model artifacts and generated reports, not user-submitted images.",
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
                        <div class="card-eyebrow">Protocol note</div>
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
            <div class="footer-brand">AETHER RESEARCH</div>
            <div class="footer-links">
                <a href="#ethics">Ethics Protocol</a>
                <a href="#safety">Safety Documentation</a>
                <a href="#legal">Legal</a>
                <a href="#privacy">Privacy</a>
            </div>
            <div class="footer-copy">2026 Aether Research Interface · CNN + XAI deployment</div>
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
