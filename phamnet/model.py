"""Model factory."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L

from .config import VARIANTS, TrainConfig
from .layers import ActivityL1, Attention, CaffePreprocess, mffs


def build_model(run: str, cfg: TrainConfig, num_classes: int = 4, pretrained: bool = True) -> K.Model:
    """Build and compile one model variant. The input is a raw 0-255 RGB image."""
    variant = VARIANTS[run]
    size = cfg.image_size
    inputs = L.Input((size, size, 3), name="image_0_255")
    x = mffs(inputs, variant.linear_projection) if variant.stem else inputs
    weights = "imagenet" if pretrained else None
    if variant.backbone == "vgg16":
        x = CaffePreprocess(name="caffe_preprocess")(x)
        backbone = K.applications.VGG16(include_top=False, weights=weights, input_shape=(size, size, 3))
    else:
        backbone = K.applications.EfficientNetB4(include_top=False, weights=weights, input_shape=(size, size, 3))
    backbone.trainable = True
    features = backbone(x)
    if variant.attention:
        features = Attention(features.shape[-1], variant.attention, sab_kernel=variant.sab_kernel, name="attention")(features)
    x = L.GlobalMaxPooling2D(name="gmp")(features)
    x = L.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x = L.Dense(cfg.dense_units, activation="relu", kernel_regularizer=K.regularizers.l2(cfg.l2))(x)
    x = ActivityL1(cfg.activity_l1, name="activity_l1")(x)
    x = L.Dropout(cfg.dropout, seed=42)(x)
    outputs = L.Dense(num_classes, activation="softmax", name="probs")(x)
    model = K.Model(inputs, outputs, name=run)
    lr = variant.learning_rate or cfg.learning_rate
    model.compile(optimizer=K.optimizers.Adamax(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
