"""Two-stage ResNet50 fine-tuning (fixed). Also writes runs/pneumonia_resnet50_seed{seed}.json.

Run from scripts/src, once per seed:
  for s in 42 43 44 45 46; do python resnet50_finetune.py --seed $s; done
"""
import argparse
import json
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from loader import load_split
from preprocess import build_dataset, class_weights


def conv5_index(backbone):
    return next(i for i, l in enumerate(backbone.layers) if l.name == "conv5_block1_1_conv")


def set_trainable(backbone, fine_tune_at):
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers:                      # keep BatchNorm frozen
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    rgb = tf.keras.layers.Concatenate(name="grayscale_to_rgb")([inputs, inputs, inputs])
    rgb = tf.keras.layers.Rescaling(255.0, name="to_255")(rgb)
    rgb = tf.keras.applications.resnet50.preprocess_input(rgb)
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3),
        pooling="avg", name="resnet50_backbone")
    features = backbone(rgb, training=False)
    features = tf.keras.layers.Dropout(0.3)(features)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(features)
    return tf.keras.Model(inputs, outputs, name="pneumonia_resnet50")


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")])


def latency_ms(model, runs=100):
    x = np.zeros((1, *model.input_shape[1:]), dtype="float32")
    for _ in range(10):
        model(x, training=False)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x, training=False).numpy()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../../../chest_xray_cache")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fine_tune_lr", type=float, default=1e-5)
    p.add_argument("--fine_tune_at", type=int, default=None, help="layer index; default = start of conv5")
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--tau", type=float, default=0.4, help="decision cutoff for the pilot record")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    run_name = f"pneumonia_resnet50_seed{args.seed}"
    output_path = args.output or f"models/{run_name}.keras"

    tf.keras.utils.set_random_seed(args.seed)
    xtr, ytr = load_split(args.data_dir, "train")
    xva, yva = load_split(args.data_dir, "val")
    train_ds = build_dataset(xtr, ytr, args.batch_size, training=True, seed=args.seed)
    val_ds = build_dataset(xva, yva, args.batch_size)
    weights = class_weights(ytr)

    model = build_model(xtr.shape[1:] + (1,))
    backbone = model.get_layer("resnet50_backbone")
    fine_tune_at = conv5_index(backbone) if args.fine_tune_at is None else args.fine_tune_at

    t0 = time.perf_counter()
    # stage 1: train only the new head (no early stopping / LR schedule for 3 epochs)
    backbone.trainable = False
    compile_model(model, args.lr)
    h1 = model.fit(train_ds, validation_data=val_ds, class_weight=weights,
                   epochs=args.warmup_epochs)

    # stage 2: fine-tune conv5 with a small LR. FIX: mode="min" for val_loss, mode="max" for val_auc
    set_trainable(backbone, fine_tune_at)
    compile_model(model, args.fine_tune_lr)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-7),
    ]
    h2 = model.fit(train_ds, validation_data=val_ds, class_weight=weights,
                   initial_epoch=args.warmup_epochs, epochs=args.epochs, callbacks=callbacks)
    train_minutes = (time.perf_counter() - t0) / 60

    Path("models").mkdir(exist_ok=True)
    model.save(output_path)
    print(f"Saved model to {output_path}")

    probs = model.predict(val_ds, verbose=0).ravel()
    pred = (probs >= args.tau).astype(int)
    keys = [k for k in h1.history if k in h2.history]
    history = {k: [float(v) for v in h1.history[k] + h2.history[k]] for k in keys}
    record = {
        "run": run_name,
        "params": int(model.count_params()),
        "flops": None, "gflops": None, "flops_method": None,
        "val_recall": float(recall_score(yva, pred)),
        "val_precision": float(precision_score(yva, pred)),
        "val_roc_auc": float(roc_auc_score(yva, probs)),
        "best_epoch": args.warmup_epochs + int(np.argmin(h2.history["val_loss"])) + 1,
        "epochs_run": len(history["loss"]),
        "train_minutes": round(train_minutes, 2),
        "latency_ms_per_img": round(latency_ms(model), 3),
        "history": history,
    }
    out = Path(args.runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_name}.json").write_text(json.dumps(record, indent=2))
    print(f"Pilot record written to {out / (run_name + '.json')}")


if __name__ == "__main__":
    main()