import os
import random

import numpy as np
import tensorflow as tf

from src.settings import LOCAL_CACHE_DIR, ensure_project_dirs


def configure_runtime() -> None:
    ensure_project_dirs()
    os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    (LOCAL_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
    (LOCAL_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
