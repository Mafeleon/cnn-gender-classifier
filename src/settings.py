import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
SCRIPTS_DIR = ROOT_DIR / "scripts"
SRC_DIR = ROOT_DIR / "src"
LOCAL_CACHE_DIR = ROOT_DIR / ".cache"
EXPERIMENT_MODELS_DIR = MODELS_DIR / "experiments"

DEFAULT_IMAGE_SIZE = (96, 96)
BATCH_SIZE = 128
SEED = 42
CLASS_NAMES = ["female", "male"]
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {index: name for name, index in LABEL_TO_INDEX.items()}

DATASET_HANDLE = "ashwingupta3012/male-and-female-faces-dataset"
MODELING_SAMPLES_PER_CLASS = int(os.environ.get("MODELING_SAMPLES_PER_CLASS", "900"))
KAGGLE_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "ashwingupta3012"
    / "male-and-female-faces-dataset"
    / "versions"
    / "1"
    / "Male and Female face dataset"
)

FINAL_MODEL_PATH = MODELS_DIR / "model.keras"
RESULTS_JSON_PATH = METRICS_DIR / "lab_results.json"


def ensure_project_dirs() -> None:
    for directory in [
        DATA_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        LOCAL_CACHE_DIR,
        EXPERIMENT_MODELS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
