import tensorflow as tf

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

    # First conv layer
    fx = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    fx = tf.keras.layers.LeakyReLU(alpha=0.01)(fx)

    # Second conv layer
    fx = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(fx)

    # Match dimension if filter size changed
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")(shortcut)

    # Add skip connection and apply activation
    out = tf.keras.layers.add([fx, shortcut])
    return tf.keras.layers.LeakyReLU(alpha=0.01)(out)


def build_residual_cnn(input_shape):
    """Deeper custom Residual CNN built from scratch."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Initial feature extraction
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Scale 1
    x = residual_block(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Scale 2
    x = residual_block(x, 128)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Global pooling instead of dense flattening for parameter efficiency
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


MODEL_REGISTRY = {
    "simple_cnn": build_simple_cnn,
    "residual_cnn": build_residual_cnn,
}