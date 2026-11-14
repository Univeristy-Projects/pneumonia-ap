import argparse
import os
import time
import numpy as np
import tensorflow as tf


def latency_ms(model, runs=100):
    x = np.zeros((1, *model.input_shape[1:]), dtype="float32")
    for _ in range(10):                      # warm-up (graph build, XLA, caches)
        model(x, training=False)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x, training=False).numpy()     # .numpy() forces the GPU to finish
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="paths to .keras files")
    p.add_argument("--out", default="cost_table.txt")
    args = p.parse_args()

    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    header = f"{'model':<45} {'params':>12} {'size (MB)':>10} {'latency (ms/img)':>17}"
    lines = [f"Device: {device} | batch size 1 | median of 100 runs", "", header, "-" * len(header)]

    for path in args.models:
        model = tf.keras.models.load_model(path)
        name = os.path.splitext(os.path.basename(path))[0]
        size_mb = os.path.getsize(path) / 1e6
        lines.append(
            f"{name:<45} {model.count_params():>12,} {size_mb:>10.2f} {latency_ms(model):>17.2f}"
        )

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()