from pathlib import Path
import numpy as np


def load_split(data_dir, split):
    d = Path(data_dir) / split
    x = np.load(d / "x.npy")
    y = np.load(d / "y.npy")
    if x.ndim == 4:          # (N, H, W, 1) -> (N, H, W)
        x = x[..., 0]
    return x, y