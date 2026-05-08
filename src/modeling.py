from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import tensorflow as tf

from src.settings import DEFAULT_IMAGE_SIZE


def build_cnn(
    input_shape=(DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3),
    filters=(8, 16, 32),
    kernel_size=3,
    dense_units=64,
    dropout_rate=0.30,
    learning_rate=1e-3,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = inputs

    for index, n_filters in enumerate(filters, start=1):
        x = tf.keras.layers.Conv2D(
            n_filters,
            kernel_size,
            padding="same",
            activation="relu",
            name=f"conv_block_{index}_conv",
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"conv_block_{index}_pool")(x)

    x = tf.keras.layers.Conv2D(
        filters[-1],
        3,
        padding="same",
        activation="relu",
        name="last_conv",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_gender_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
        run_eagerly=True,
    )
    return model


def experiment_configs() -> Iterable[Dict[str, object]]:
    return [
        {
            "name": "baseline",
            "filters": (8, 16, 32),
            "kernel_size": 3,
            "dense_units": 64,
            "dropout_rate": 0.20,
            "learning_rate": 1e-3,
            "epochs": 6,
        },
        {
            "name": "regularized",
            "filters": (8, 16, 32),
            "kernel_size": 3,
            "dense_units": 64,
            "dropout_rate": 0.35,
            "learning_rate": 5e-4,
            "epochs": 8,
        },
    ]


def build_callbacks(checkpoint_path: Path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def capture_model_summary(model: tf.keras.Model) -> str:
    buffer = StringIO()
    model.summary(print_fn=lambda line: buffer.write(f"{line}\n"))
    return buffer.getvalue()


def history_to_frame(history: tf.keras.callbacks.History) -> pd.DataFrame:
    frame = pd.DataFrame(history.history)
    frame.index = frame.index + 1
    frame.index.name = "epoch"
    return frame.reset_index()
