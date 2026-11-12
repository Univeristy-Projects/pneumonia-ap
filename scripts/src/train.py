import argparse
import tensorflow as tf
from loader import load_split
from preprocess import build_dataset, class_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", default="../../../chest_xray_cache")
    return parser.parse_args()


def build_model(input_shape):
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

    model = build_model(xtr.shape[1:] + (1,))           # (H, W, 1)
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True)]
    model.fit(train_ds, validation_data=val_ds, class_weight=weights,
              epochs=args.epochs, callbacks=callbacks)

    model.save(f"pneumonia_model_seed{args.seed}.keras")


if __name__ == "__main__":
    main()