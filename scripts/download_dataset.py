from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import kagglehub

from src.settings import DATASET_HANDLE, KAGGLE_CACHE_ROOT, DATA_DIR, ensure_project_dirs


def recreate_symlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            raise RuntimeError(f"{target} ya existe y no es un enlace simbólico.")
    os.symlink(source, target)


def main() -> None:
    ensure_project_dirs()
    if not KAGGLE_CACHE_ROOT.exists():
        kagglehub.dataset_download(DATASET_HANDLE)

    source_male = KAGGLE_CACHE_ROOT / "Male Faces"
    source_female = KAGGLE_CACHE_ROOT / "Female Faces"

    if not source_male.exists() or not source_female.exists():
        raise FileNotFoundError(
            "La descarga se completó, pero no se encontraron las carpetas Male Faces/Female Faces."
        )

    recreate_symlink(source_male, DATA_DIR / "male")
    recreate_symlink(source_female, DATA_DIR / "female")

    print(f"Dataset listo en:\n- {DATA_DIR / 'male'}\n- {DATA_DIR / 'female'}")


if __name__ == "__main__":
    main()
