import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write PNGs, no window needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from loader import load_split
from preprocess import build_dataset, class_weights
from models import MODEL_REGISTRY


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


def flops_from_layers(model):
    """Analytic FLOPs for one image: 2 x multiply-adds of Conv2D, DepthwiseConv2D and Dense
    layers (activations, normalisation, pooling and bias adds are ignored)."""
    total = 0
    for layer in model.layers:
        kind = type(layer).__name__
        out = layer.output.shape
        if kind == "Conv2D":
            kh, kw = layer.kernel_size
            total += 2 * out[1] * out[2] * kh * kw * layer.input.shape[-1] * out[-1]
        elif kind == "DepthwiseConv2D":
            kh, kw = layer.kernel_size
            total += 2 * out[1] * out[2] * kh * kw * out[-1]
        elif kind == "Dense":
            positions = int(np.prod(layer.input.shape[1:-1]))      # 1 for flat inputs
            total += 2 * positions * layer.input.shape[-1] * layer.units
    return int(total)


def get_flops(model):
    """Returns (flops, method). Tries the TensorFlow profiler on the inference graph
    (batch size 1); if that fails, falls back to the layer-by-layer count above."""
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph)
        spec = tf.TensorSpec([1, *model.input_shape[1:]], model.inputs[0].dtype)
        concrete = tf.function(lambda x: model(x, training=False)).get_concrete_function(spec)
        _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            info = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=opts)
        return int(info.total_float_ops), "tf_profiler"
    except Exception as error:
        print(f"[WARN] TF profiler failed ({error}); using layer-wise FLOPs count")
        try:
            return flops_from_layers(model), "layer_count"
        except Exception as error2:
            print(f"[WARN] FLOPs could not be computed: {error2}")
            return None, None


def latency_ms(model, runs=100):
    """Median single-image inference time (ms) on the current device."""
    x = np.zeros((1, *model.input_shape[1:]), dtype="float32")
    for _ in range(10):                          # warm-up
        model(x, training=False)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x, training=False).numpy()         # .numpy() waits for the GPU
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def save_curves(history, path, title):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(epochs, h["loss"], label="train")
    a.plot(epochs, h["val_loss"], label="validation")
    a.set_xlabel("Epoch"); a.set_ylabel("Loss"); a.set_title("Loss"); a.legend()
    b.plot(epochs, h["auc"], label="train")
    b.plot(epochs, h["val_auc"], label="validation")
    b.set_xlabel("Epoch"); b.set_ylabel("ROC-AUC"); b.set_title("ROC-AUC"); b.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", default="../../../chest_xray_cache")
    parser.add_argument("--model_name", type=str, default="simple_cnn", help="Model Architecture to train")
    parser.add_argument("--num_blocks", type=int, default=2, help="Blocks for residual_cnn / depth of convnext stage 3")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer choice")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--runs_dir", default="runs", help="Where curves and pilot metrics are written")
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
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5)
    ]

    t0 = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, class_weight=weights,
                        epochs=args.epochs, callbacks=callbacks)
    train_minutes = (time.perf_counter() - t0) / 60

    if args.model_name in block_based_models:
        run_name = f"pneumonia_{args.model_name}_b{args.num_blocks}_seed{args.seed}"
    else:
        run_name = f"pneumonia_{args.model_name}_seed{args.seed}"
    Path("models").mkdir(exist_ok=True)               # model.save fails if the folder is missing
    save_filename = f"models/{run_name}.keras"
    model.save(save_filename)
    print(f"[INFO] Model successfully saved to: {save_filename}")

    # ---- pilot record: validation metrics at the restored best epoch (threshold 0.5) ----
    val_probs = model.predict(val_ds, verbose=0).ravel()
    val_pred = (val_probs >= 0.4).astype(int)
    h = history.history
    flops, flops_method = get_flops(model)
    record = {
        "run": run_name,
        "params": int(model.count_params()),
        "flops": flops,
        "gflops": None if flops is None else round(flops / 1e9, 3),
        "flops_method": flops_method,
        "val_recall": float(recall_score(yva, val_pred)),
        "val_precision": float(precision_score(yva, val_pred)),
        "val_roc_auc": float(roc_auc_score(yva, val_probs)),
        "best_epoch": int(np.argmin(h["val_loss"])) + 1,
        "epochs_run": len(h["loss"]),
        "train_minutes": round(train_minutes, 2),
        "latency_ms_per_img": round(latency_ms(model), 3),
        "history": {k: [float(v) for v in vals] for k, vals in h.items()},
    }

    out = Path(args.runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_name}.json").write_text(json.dumps(record, indent=2))
    save_curves(history, out / f"{run_name}_curves.png", run_name)
    print(f"[INFO] Pilot record and learning curves written to {out}/")

if __name__ == "__main__":
    main()
