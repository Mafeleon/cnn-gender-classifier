from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from src.settings import DEFAULT_IMAGE_SIZE


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
