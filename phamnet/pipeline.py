"""tf.data input pipeline.

Images are decoded to uint8 in the range 0-255. The Keras EfficientNet-B4 rescales
and ImageNet-normalises internally, so normalisation is applied exactly once.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L

AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation() -> K.Sequential:
    """Training-time augmentation: horizontal flip, rotation +-15 deg, zoom +-10 %, brightness +-0.2 x 255."""
    return K.Sequential(
        [
            L.RandomFlip("horizontal"),
            L.RandomRotation(15 / 360),
            L.RandomZoom((-0.1, 0.1)),
            L.RandomBrightness(0.2, value_range=(0, 255)),
        ],
        name="augmentation",
    )


def load_image(path, label, size: int, num_classes: int):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image = tf.image.resize(image, (size, size), method="bilinear")
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image, tf.one_hot(label, num_classes)


def make_dataset(files: np.ndarray, labels: np.ndarray, size: int, num_classes: int, batch_size: int,
                 train: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(lambda p, y: load_image(p, y, size, num_classes), num_parallel_calls=AUTOTUNE).cache()
    if train:
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if train:
        augment = build_augmentation()
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
