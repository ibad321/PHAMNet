"""Custom Keras layers: spatial / channel attention, HAM, CBAM, BAM and the MFFS stem."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow import keras as K


class SAB(L.Layer):
    """Spatial attention block (CBAM spatial branch): sigmoid(conv_kxk([avg_c(F); max_c(F)]))."""

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.conv = L.Conv2D(1, kernel_size, padding="same", activation="sigmoid")

    def call(self, feature):
        pooled = tf.concat([tf.reduce_mean(feature, -1, keepdims=True), tf.reduce_max(feature, -1, keepdims=True)], -1)
        return self.conv(pooled)


class GCAB(L.Layer):
    """Global channel attention block (CBAM channel branch): sigmoid(MLP(GAP) + MLP(GMP)), shared MLP."""

    def __init__(self, channels: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = L.Dense(channels // reduction, activation="relu")
        self.fc2 = L.Dense(channels)

    def call(self, feature):
        avg = self.fc2(self.fc1(tf.reduce_mean(feature, [1, 2])))
        mx = self.fc2(self.fc1(tf.reduce_max(feature, [1, 2])))
        return tf.sigmoid(avg + mx)[:, None, None, :]


class Attention(L.Layer):
    """Attention head on the backbone feature map.

    Modes:
      sab, gcab      single branch
      ham            F_out = alpha * (F * A_s) + beta * (F * A_c), alpha and beta learnable (init 0.5)
      ham_fixed      same with alpha = beta = 0.5 frozen
      cbam           sequential channel -> spatial attention
      bam            parallel branches summed into one map, F + F * sigmoid(M_c + M_s)
    """

    def __init__(self, channels: int, mode: str, sab_kernel: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.sab = SAB(sab_kernel)
        self.gcab = GCAB(channels)
        if mode == "bam":
            self.mc1 = L.Dense(channels // 16, activation="relu")
            self.mc2 = L.Dense(channels)
            self.ms = [
                L.Conv2D(channels // 16, 1, activation="relu"),
                L.Conv2D(channels // 16, 3, padding="same", dilation_rate=4, activation="relu"),
                L.Conv2D(channels // 16, 3, padding="same", dilation_rate=4, activation="relu"),
                L.Conv2D(1, 1),
            ]

    def build(self, input_shape):
        trainable = self.mode == "ham"
        self.alpha = self.add_weight(name="alpha", shape=(), initializer=K.initializers.Constant(0.5), trainable=trainable)
        self.beta = self.add_weight(name="beta", shape=(), initializer=K.initializers.Constant(0.5), trainable=trainable)
        super().build(input_shape)

    def call(self, feature):
        mode = self.mode
        if mode == "sab":
            return feature * self.sab(feature)
        if mode == "gcab":
            return feature * self.gcab(feature)
        if mode in ("ham", "ham_fixed"):
            return self.alpha * (feature * self.sab(feature)) + self.beta * (feature * self.gcab(feature))
        if mode == "cbam":
            refined = feature * self.gcab(feature)
            return refined * self.sab(refined)
        if mode == "bam":
            channel = self.mc2(self.mc1(tf.reduce_mean(feature, [1, 2])))[:, None, None, :]
            spatial = feature
            for layer in self.ms:
                spatial = layer(spatial)
            return feature + feature * tf.sigmoid(channel + spatial)
        raise ValueError(f"unknown attention mode: {mode}")


class ActivityL1(L.Layer):
    """L1 activity penalty averaged over the batch (the tf.keras 2.x convention).

    Some Keras 3 versions sum the penalty over the batch instead, which makes it
    batch-size times stronger and can collapse training to a single class.
    """

    def __init__(self, l1: float, **kwargs):
        super().__init__(**kwargs)
        self.l1 = l1

    def call(self, x):
        self.add_loss(self.l1 * tf.reduce_sum(tf.abs(x)) / tf.cast(tf.shape(x)[0], x.dtype))
        return x


class CaffePreprocess(L.Layer):
    """VGG-16 ImageNet preprocessing (RGB -> BGR, subtract the ImageNet mean)."""

    def call(self, x):
        return x[..., ::-1] - tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)


def mffs(x, linear_projection: bool = False):
    """Multiscale feature fusion stem on a raw 0-255 image tensor.

    Three parallel 3x3 / 5x5 / 7x7 convolutions (32 channels, BatchNorm, ReLU) are
    concatenated (96 channels) and projected to three channels. The output is
    rescaled to 0-255 because the Keras EfficientNet normalises internally.
    """
    scaled = L.Rescaling(1 / 255.0, name="mffs_scale_in")(x)
    branches = [
        L.ReLU()(L.BatchNormalization()(L.Conv2D(32, k, padding="same", name=f"mffs_conv{k}")(scaled))) for k in (3, 5, 7)
    ]
    y = L.Concatenate(name="mffs_concat")(branches)
    y = L.Conv2D(3, 1, name="mffs_proj")(y)
    if not linear_projection:
        y = L.ReLU()(L.BatchNormalization()(y))
    return L.Rescaling(255.0, name="mffs_scale_out")(y)
