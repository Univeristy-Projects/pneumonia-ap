"""Aggregate the per-seed pilot records written by train.py (runs/*.json).

For every candidate it reports mean ± 95% Student-t margin over the seeds
recall, precision and ROC-AUC, plus mean best epoch, training time and latency.
Writes runs/pilot_table.txt and one combined learning-curve figure.
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import t as student_t

    def t_crit(n):
        return float(student_t.ppf(0.975, n - 1))
except ImportError:                                   # two-sided 95% critical values
    _T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}

    def t_crit(n):
        return _T[n]


def mean_margin(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), t_crit(n) * float(v.std(ddof=1)) / math.sqrt(n)


def fmt(values, digits=4):
    m, h = mean_margin(values)
    return f"{m:.{digits}f} ± {h:.{digits}f}" if not math.isnan(h) else f"{m:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    args = p.parse_args()
    runs = Path(args.runs_dir)

    groups = defaultdict(list)
    for f in sorted(runs.glob("pneumonia_*.json")):
        rec = json.loads(f.read_text())
        key = re.sub(r"^pneumonia_", "", re.sub(r"_seed\d+$", "", rec["run"]))
        groups[key].append(rec)
    if not groups:
        raise SystemExit(f"No pneumonia_*.json files found in {runs}/")

    header = ["Candidate", "Seeds", "Params", "GFLOPs", "Val recall", "Val precision",
              "Val ROC-AUC", "Best epoch", "Train min", "Latency ms/img"]
    rows = []
    for name, recs in groups.items():
        g = recs[0].get("gflops")
        rows.append([
            name, str(len(recs)), f"{recs[0]['params']:,}", "—" if g is None else f"{g:.2f}",
            fmt([r["val_recall"] for r in recs]),
            fmt([r["val_precision"] for r in recs]),
            fmt([r["val_roc_auc"] for r in recs]),
            f"{np.mean([r['best_epoch'] for r in recs]):.1f}",
            f"{np.mean([r['train_minutes'] for r in recs]):.1f}",
            f"{np.mean([r['latency_ms_per_img'] for r in recs]):.2f}",
        ])
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))
    text = ["Validation metrics: mean ± 95% Student-t margin over seeds (threshold 0.4, restored best epoch)",
            "", line(header), line(["-" * w for w in widths])] + [line(r) for r in rows]
    (runs / "pilot_table.txt").write_text("\n".join(text) + "\n")

    # one combined learning-curve figure: train vs validation loss, mean ± std over seeds
    n = len(groups)
    cols = 2
    fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(10, 3.6 * math.ceil(n / cols)), squeeze=False)
    for ax, (name, recs) in zip(axes.ravel(), groups.items()):
        length = max(len(r["history"]["loss"]) for r in recs)
        for key, label in [("loss", "train"), ("val_loss", "validation")]:
            arr = np.full((len(recs), length), np.nan)
            for i, r in enumerate(recs):
                arr[i, :len(r["history"][key])] = r["history"][key]
            m, s = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
            x = np.arange(1, length + 1)
            ax.plot(x, m, label=label)
            ax.fill_between(x, m - s, m + s, alpha=0.2)
        ax.set_title(f"{name} (n = {len(recs)})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(runs / "learning_curves_all.png", dpi=150)
    plt.close(fig)
    print(f"[INFO] wrote {runs / 'pilot_table.txt'} and {runs / 'learning_curves_all.png'}")


if __name__ == "__main__":
    main()
