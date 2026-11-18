"""Writes runs/pneumonia_resnet50_seed{N}.json for each saved model,
matching the format train.py produces so pilot_table.py aggregates them."""
import glob, json, math, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from loader import load_split
from preprocess import build_dataset
from models import LayerScale          # registers the custom object


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


def flops_from_layers(model):
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
            positions = int(np.prod(layer.input.shape[1:-1]))
            total += 2 * positions * layer.input.shape[-1] * layer.units
    return int(total)


import argparse
p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="../../../chest_xray_cache")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--runs_dir", default="runs")
p.add_argument("--model_glob", default="models/pneumonia_resnet50_seed*.keras",
               help="glob pattern for your saved ResNet-50 .keras files")
args = p.parse_args()

xva, yva = load_split(args.data_dir, "val")
val_ds = build_dataset(xva, yva, args.batch_size)
out = Path(args.runs_dir)
out.mkdir(exist_ok=True)

paths = sorted(glob.glob(args.model_glob))
if not paths:
    raise FileNotFoundError(f"No files matched: {args.model_glob}")
print(f"Found {len(paths)} ResNet-50 models")

for path in paths:
    model = tf.keras.models.load_model(path)
    probs = model.predict(val_ds, verbose=0).ravel()
    pred  = (probs >= 0.4).astype(int)
    seed  = path.split("seed")[-1].replace(".keras", "")
    run_name = f"pneumonia_resnet50_seed{seed}"
    flops = flops_from_layers(model)
    record = {
        "run": run_name,
        "params": int(model.count_params()),
        "flops": flops,
        "gflops": round(flops / 1e9, 3),
        "flops_method": "layer_count",
        "val_recall":    float(recall_score(yva, pred, zero_division=0)),
        "val_precision": float(precision_score(yva, pred, zero_division=0)),
        "val_roc_auc":   float(roc_auc_score(yva, probs)),
        "best_epoch":    None,
        "epochs_run":    None,
        "train_minutes": None,
        "latency_ms_per_img": round(latency_ms(model), 3),
        "history":       {},
    }
    dest = out / f"{run_name}.json"
    dest.write_text(json.dumps(record, indent=2))
    print(f"  wrote {dest}  recall={record['val_recall']:.4f}  auc={record['val_roc_auc']:.4f}")

print("Done — re-run pilot_table.py to update the table.")