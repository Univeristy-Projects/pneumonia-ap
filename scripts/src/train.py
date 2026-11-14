import argparse
import tensorflow as tf
from loader import load_split
from preprocess import build_dataset, class_weights
from models import MODEL_REGISTRY
import numpy as np


def get_optimizer(optimizer_name: str, learning_rate: float):
    """Optimizer Registry: Maps string names to configured optimizer instances."""
    optimizers = {
        "nadam": tf.keras.optimizers.Nadam(
            learning_rate=learning_rate
        ),
        "adam": tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        "adamw": tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0
        ),
    }

    if optimizer_name not in optimizers:
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. "
            f"Available options are: {list(optimizers.keys())}"
        )

    return optimizers[optimizer_name]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", default="../../../chest_xray_cache")
    parser.add_argument("--model_name", type=str, default="simple_cnn", help="Model Architecture to train")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of residual blocks for residual_cnn")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer choice")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    xtr, ytr = load_split(args.data_dir, "train")
    xva, yva = load_split(args.data_dir, "val")
    print(xtr.dtype, xtr.shape, xtr.min(), xtr.max())   # expect uint8, (N, H, W), 0, 255

    train_ds = build_dataset(xtr, ytr, args.batch_size, training=True, seed=args.seed)
    val_ds = build_dataset(xva, yva, args.batch_size)

    weights = class_weights(ytr)
    print(f"Class Weights: {weights}")

    input_shape = xtr.shape[1:] + (1,)
    if args.model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture '{args.model_name}'. "
            f"Available options are: {list(MODEL_REGISTRY.keys())}"
        )

    model_builder = MODEL_REGISTRY[args.model_name]

    block_based_models = ["residual_cnn", "convnext"]

    if args.model_name in block_based_models:
        model = model_builder(input_shape, num_blocks=args.num_blocks)
    else:
        model = model_builder(input_shape)

    
    optimizer = get_optimizer(args.optimizer, learning_rate=args.lr)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2)
    ]
    model.fit(train_ds, validation_data=val_ds, class_weight=weights,
              epochs=args.epochs, callbacks=callbacks)

    if args.model_name in block_based_models:
        save_filename = f"pneumonia_{args.model_name}_b{args.num_blocks}_seed{args.seed}.keras"
    else:
        save_filename = f"pneumonia_{args.model_name}_seed{args.seed}.keras"
    model.save(save_filename)
    print(f"[INFO] Model successfully saved to: {save_filename}")

if __name__ == "__main__":
    main()
