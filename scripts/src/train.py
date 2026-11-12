import argparse
import tensorflow as tf
from loader import load_split
from preprocess import build_dataset, class_weights
from models import MODEL_REGISTRY

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", default="../../../chest_xray_cache")
    parser.add_argument("--model_name", type=str, default="build_simple_cnn", help="Model Architecture to train")
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
    model = model_builder(input_shape)

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