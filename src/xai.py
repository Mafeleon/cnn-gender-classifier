from __future__ import annotations

import numpy as np
import tensorflow as tf
from matplotlib import cm


def make_saliency_map(model: tf.keras.Model, image_tensor: tf.Tensor) -> np.ndarray:
    image_tensor = tf.cast(image_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor, training=False)
        target = predictions[:, 0]
    gradients = tape.gradient(target, image_tensor)[0]
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    saliency = saliency.numpy()
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    last_conv_layer_name: str = "last_conv",
) -> np.ndarray:
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        target = predictions[:, 0]

    gradients = tape.gradient(target, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = np.nan_to_num(heatmap)
    return heatmap


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (image_rgb.shape[0], image_rgb.shape[1]),
        method="bilinear",
    ).numpy().squeeze()
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colored = (cm.get_cmap("jet")(heatmap)[..., :3] * 255).astype(np.uint8)
    overlay = ((1 - alpha) * image_rgb.astype(np.float32) + alpha * colored.astype(np.float32)).astype(
        np.uint8
    )
    return overlay
