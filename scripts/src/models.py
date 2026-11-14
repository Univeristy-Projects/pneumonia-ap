import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def build_simple_cnn(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def residual_block(x, filters):
    """Custom Residual Block with skip connections and Leaky ReLU."""
    shortcut = x

    fx = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    fx = tf.keras.layers.LeakyReLU(alpha=0.01)(fx)

    fx = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(fx)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")(shortcut)

    out = tf.keras.layers.add([fx, shortcut])
    return tf.keras.layers.LeakyReLU(alpha=0.01)(out)


def build_residual_cnn(input_shape, num_blocks=2):
    """Deeper custom Residual CNN with configurable block depth."""
    inputs = tf.keras.layers.Input(shape=input_shape)

   
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    filters = 64
    for _ in range(num_blocks):
        x = residual_block(x, filters)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        filters = min(filters * 2, 256)  # Scale filters up, capped at 256

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)



@tf.keras.utils.register_keras_serializable(package="pneumonia")
class LayerScale(layers.Layer):
    """Learnable per-channel scale, initialised near zero (ConvNeXt)."""

    def __init__(self, init_value=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


def convnext_block(x, dim, drop_rate=0.0):
    shortcut = x
    x = layers.DepthwiseConv2D(7, padding="same")(x)       # 7x7 depthwise
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = layers.Conv2D(4 * dim, kernel_size=1, use_bias=False)(x)
    x = layers.Activation("gelu")(x)
    x = layers.Conv2D(dim, kernel_size=1, use_bias=False)(x)
    x = LayerScale()(x)
    if drop_rate > 0:                                      # stochastic depth
        x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1))(x)
    return layers.add([shortcut, x])


def build_convnext(input_shape, num_blocks=3, dims=(32, 64, 128, 256), drop_path=0.0):
    """Small ConvNeXt. num_blocks = depth of stage 3 (the deepest stage)."""
    depths = (1, 1, num_blocks, 1)
    rates = np.linspace(0.0, drop_path, sum(depths))
    inputs = layers.Input(shape=input_shape)

    # Stem: 4x4 stride-4 "patchify" conv + LayerNorm
    x = layers.Conv2D(dims[0], 4, strides=4)(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    k = 0
    for stage, (dim, depth) in enumerate(zip(dims, depths)):
        if stage > 0:                                      # downsample between stages
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Conv2D(dim, 2, strides=2)(x)
        for _ in range(depth):
            x = convnext_block(x, dim, float(rates[k]))
            k += 1

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="convnext")



MODEL_REGISTRY = {
    "simple_cnn": build_simple_cnn,
    "residual_cnn": build_residual_cnn,
    "convnext": build_convnext,

}
