"""Two-stage ResNet50 fine-tuning for the cached pneumonia dataset.

Run from scripts/src:
  python resnet50_finetune.py --data_dir ../../../chest_xray_cache
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

from loader import load_split
from preprocess import build_dataset, class_weights


def build_model(input_shape, fine_tune_at=None):
    inputs = tf.keras.Input(shape=input_shape)

    # ResNet50 expects RGB images in the ImageNet preprocessing format.
    rgb = tf.keras.layers.Concatenate(name="grayscale_to_rgb")(
        [inputs, inputs, inputs]
    )
    rgb = tf.keras.layers.Rescaling(255.0, name="to_255")(rgb)
    rgb = tf.keras.applications.resnet50.preprocess_input(rgb)

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3),
        pooling="avg",
        name="resnet50_backbone",
    )
    if fine_tune_at is None:
        fine_tune_at = next(
            index
            for index, layer in enumerate(backbone.layers)
            if layer.name == "conv5_block1_1_conv"
        )
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    features = backbone(rgb, training=False)
    features = tf.keras.layers.Dropout(0.3)(features)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(features)
    return tf.keras.Model(inputs, outputs, name="pneumonia_resnet50")


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../../chest_xray_cache")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument(
        "--fine_tune_at",
        type=int,
        default=None,
        help="Optional layer index; defaults to the start of ResNet50 conv5",
    )
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Output path; defaults to models/pneumonia_residual_cnn_b50_seed{seed}.keras",
    )
    args = parser.parse_args()
    output_path = args.output or f"models/pneumonia_residual_cnn_b50_seed{args.seed}.keras"

    tf.keras.utils.set_random_seed(args.seed)

    x_train, y_train = load_split(args.data_dir, "train")
    x_val, y_val = load_split(args.data_dir, "val")
    train_ds = build_dataset(
        x_train, y_train, args.batch_size, training=True, seed=args.seed
    )
    val_ds = build_dataset(x_val, y_val, args.batch_size)
    weights = class_weights(y_train)

    model = build_model(x_train.shape[1:] + (1,), args.fine_tune_at)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="max", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=5, min_lr=1e-7
        ),
    ]

    # Warm up the new classification head before updating ResNet50.
    backbone = model.get_layer("resnet50_backbone")
    fine_tune_at = next(
        index
        for index, layer in enumerate(backbone.layers)
        if layer.name == "conv5_block1_1_conv"
    ) if args.fine_tune_at is None else args.fine_tune_at
    backbone.trainable = False
    compile_model(model, args.lr)
    model.fit(
        train_ds,
        validation_data=val_ds,
        class_weight=weights,
        epochs=args.warmup_epochs,
        callbacks=callbacks,
    )

    # Fine-tune only the later convolutional blocks with a smaller LR.
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    print(
        f"Fine-tuning from layer {fine_tune_at}: "
        f"{backbone.layers[fine_tune_at].name}"
    )
    compile_model(model, args.fine_tune_lr)
    model.fit(
        train_ds,
        validation_data=val_ds,
        class_weight=weights,
        initial_epoch=args.warmup_epochs,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
