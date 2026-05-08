from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from src.settings import (
    CLASS_NAMES,
    DATA_DIR,
    DEFAULT_IMAGE_SIZE,
    LABEL_TO_INDEX,
    LOCAL_CACHE_DIR,
    SEED,
)

AUTOTUNE = tf.data.AUTOTUNE


def collect_image_records(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    records = []
    for label_name in CLASS_NAMES:
        class_dir = data_dir / label_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta {class_dir}. Ejecute scripts/download_dataset.py primero."
            )
        for path in sorted(class_dir.iterdir()):
            if path.is_file():
                records.append(
                    {
                        "filepath": str(path.resolve()),
                        "label": label_name,
                        "label_id": LABEL_TO_INDEX[label_name],
                        "extension": path.suffix.lower(),
                    }
                )
    if not records:
        raise RuntimeError("No se encontraron imágenes en data/male y data/female.")
    return pd.DataFrame(records)


def inspect_dataset(records: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for label_name in CLASS_NAMES:
        class_df = records[records["label"] == label_name]
        widths = []
        heights = []
        modes: Counter[str] = Counter()
        extensions = Counter(class_df["extension"].tolist())
        bad_files = []
        for filepath in class_df["filepath"]:
            try:
                with Image.open(filepath) as image:
                    widths.append(image.width)
                    heights.append(image.height)
                    modes[image.mode] += 1
            except Exception:
                bad_files.append(Path(filepath).name)

        summary[label_name] = {
            "count": int(len(class_df)),
            "extensions": dict(extensions),
            "modes": dict(modes),
            "bad_files": bad_files,
            "width_min": int(min(widths)),
            "width_max": int(max(widths)),
            "width_mean": float(np.mean(widths)),
            "width_median": float(np.median(widths)),
            "height_min": int(min(heights)),
            "height_max": int(max(heights)),
            "height_mean": float(np.mean(heights)),
            "height_median": float(np.median(heights)),
        }
    return summary


def make_stratified_splits(
    records: pd.DataFrame,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        records,
        test_size=0.30,
        random_state=seed,
        stratify=records["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label_id"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: Tuple[int, int]):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
    return image, tf.cast(label, tf.float32)


def build_tf_dataset(
    records: pd.DataFrame,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    training: bool = False,
) -> tf.data.Dataset:
    paths = records["filepath"].astype(str).to_numpy()
    labels = records["label_id"].astype("float32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(records), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: _decode_and_resize(path, label, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def _preprocess_rgb_array(rgb_array: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    tensor = tf.convert_to_tensor(rgb_array, dtype=tf.uint8)
    tensor = tf.cast(tensor, tf.float32)
    tensor = tf.image.resize_with_pad(tensor, image_size[0], image_size[1])
    tensor = tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8)
    return tensor.numpy()


def load_or_create_array_cache(
    records: pd.DataFrame,
    split_name: str,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
):
    cache_dir = LOCAL_CACHE_DIR / "preprocessed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{split_name}_{image_size[0]}x{image_size[1]}.npz"

    if cache_file.exists():
        with np.load(cache_file) as cached:
            images = cached["images"]
            labels = cached["labels"]
            if len(labels) == len(records):
                return images, labels

    images = np.empty((len(records), image_size[0], image_size[1], 3), dtype=np.uint8)
    labels = records["label_id"].astype("float32").to_numpy()
    for index, filepath in enumerate(records["filepath"]):
        if index % 500 == 0:
            print(f"[{split_name}] procesadas {index}/{len(records)} imágenes", flush=True)
        with Image.open(filepath) as image:
            rgb_array = np.array(image.convert("RGB"))
        images[index] = _preprocess_rgb_array(rgb_array, image_size)

    np.savez(cache_file, images=images, labels=labels)
    return images, labels


def build_array_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool = False,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(len(labels), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, tf.cast(label, tf.float32)),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def prepare_image_from_pil(
    pil_image: Image.Image,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
):
    rgb_image = pil_image.convert("RGB")
    rgb_array = np.array(rgb_image)
    tensor = tf.convert_to_tensor(rgb_array, dtype=tf.uint8)
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    tensor = tf.image.resize_with_pad(tensor, image_size[0], image_size[1])
    tensor = tf.expand_dims(tensor, axis=0)
    resized_rgb = tf.image.convert_image_dtype(tensor[0], tf.uint8).numpy()
    return rgb_array, resized_rgb, tensor
