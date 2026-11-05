import hashlib
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["normal", "pneumonia"]
SPLITS = ("train", "val", "test")


def file_sha256(path):
  digest = hashlib.sha256()
  with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def load_manifest(path):
  manifest = json.loads(Path(path).read_text())
  by_split = {split: [] for split in SPLITS}
  for record in manifest["records"]:
    by_split[record["split"]].append(record)
  for records in by_split.values():
    records.sort(key=lambda r: r["path"])
  return manifest["meta"], by_split


def resize_image(path, size):
  raw = tf.io.read_file(str(path))
  image = tf.io.decode_image(raw, channels=1, expand_animations=False)
  resized = tf.image.resize(image, (size, size), method="bicubic", antialias=True)
  resized = tf.clip_by_value(tf.round(resized), 0.0, 255.0)
  return tf.cast(resized, tf.uint8).numpy()[..., 0]


def cache_file(cache_dir, size, split):
  return Path(cache_dir) / f"size_{size}" / f"{split}.npz"


def build_cache(manifest_path, data_dir, cache_dir, size, overwrite=False):
  manifest_hash = file_sha256(manifest_path)
  _, by_split = load_manifest(manifest_path)
  data_dir = Path(data_dir)
  for split in SPLITS:
    target = cache_file(cache_dir, size, split)
    if target.exists() and not overwrite:
      print(f"{split}: cache exists, skipping")
      continue
    records = by_split[split]
    images = np.zeros((len(records), size, size), dtype=np.uint8)
    failed = []
    for index, record in enumerate(records):
      try:
        images[index] = resize_image(data_dir / record["path"], size)
      except Exception as error:
        failed.append((record["path"], type(error).__name__))
      if (index + 1) % 500 == 0:
        print(f"{split}: {index + 1}/{len(records)}")
    if failed:
      listing = "\n".join(f"  {path} ({kind})" for path, kind in failed[:20])
      raise RuntimeError(
          f"{len(failed)} images in '{split}' could not be decoded:\n{listing}\n"
          "Remove them and run make_splits.py again."
      )
    labels = np.array([r["label"] for r in records], dtype=np.uint8)
    paths = np.array([r["path"] for r in records])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        x=images,
        y=labels,
        paths=paths,
        manifest_sha=manifest_hash,
        size=size,
    )
    temporary.replace(target)
    print(f"{split}: cached {len(records)} images -> {target}")


def load_split(cache_dir, size, split, manifest_path=None):
  with np.load(cache_file(cache_dir, size, split)) as data:
    images, labels = data["x"], data["y"]
    paths = [str(p) for p in data["paths"]]
    stored_hash = str(data["manifest_sha"])
  if manifest_path is not None and stored_hash != file_sha256(manifest_path):
    raise RuntimeError(
        "cache was built from a different manifest; rebuild with --overwrite"
    )
  return images, labels, paths


def class_weights(labels):
  counts = np.bincount(labels, minlength=len(CLASS_NAMES))
  total = counts.sum()
  return {i: float(total / (len(CLASS_NAMES) * c)) for i, c in enumerate(counts)}


def augment(image, seed, hflip):
  seeds = tf.random.experimental.stateless_split(seed, num=3)
  image = tf.image.stateless_random_brightness(image, 25.5, seeds[0])
  image = tf.image.stateless_random_contrast(image, 0.9, 1.1, seeds[1])
  if hflip:
    image = tf.image.stateless_random_flip_left_right(image, seeds[2])
  return tf.clip_by_value(image, 0.0, 255.0)


def build_dataset(images, labels, batch_size, training=False, seed=0, hflip=False):
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  if training:
    dataset = dataset.shuffle(
        len(images), seed=seed, reshuffle_each_iteration=True
    )
  dataset = dataset.enumerate()

  def prepare(index, item):
    image, label = item
    image = tf.cast(image[..., tf.newaxis], tf.float32)
    if training:
      step_seed = tf.stack([tf.constant(seed, tf.int64), index])
      image = augment(image, step_seed, hflip)
    image = tf.image.grayscale_to_rgb(image)
    image = keras.applications.resnet50.preprocess_input(image)
    return image, tf.cast(label, tf.float32)

  dataset = dataset.map(
      prepare, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
  )
  return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_distribution(by_split):
  rows = {}
  header = f"{'split':<6} {'normal':>7} {'pneumonia':>10} {'total':>7} {'pneu%':>7} {'patients':>9}"
  print(header)
  for split in SPLITS:
    records = by_split[split]
    normal = sum(r["label"] == 0 for r in records)
    pneumonia = sum(r["label"] == 1 for r in records)
    total = normal + pneumonia
    patients = len({r["group"] for r in records})
    share = 100 * pneumonia / total if total else 0.0
    print(
        f"{split:<6} {normal:>7} {pneumonia:>10} {total:>7} {share:>6.2f}% {patients:>9}"
    )
    rows[split] = {"normal": normal, "pneumonia": pneumonia}
  return rows


def intensity_outliers(cache_dir, size, manifest_path=None, threshold=3.0):
  means, paths = [], []
  for split in SPLITS:
    images, _, split_paths = load_split(cache_dir, size, split, manifest_path)
    if len(images):
      means.append(images.reshape(len(images), -1).mean(axis=1, dtype=np.float64))
    paths.extend(split_paths)
  means = np.concatenate(means)
  mu = float(means.mean())
  sigma = float(means.std()) or 1e-5
  z_scores = (means - mu) / sigma
  mask = np.abs(z_scores) > threshold
  print(f"images analyzed: {len(means)} (cached {size}px, grayscale)")
  print(f"mean intensity: {mu:.2f} | std: {sigma:.2f}")
  print(f"outliers (|z| > {threshold:g}): {int(mask.sum())}")
  for path, z in zip(np.array(paths)[mask], z_scores[mask]):
    print(f"  {path} (z = {z:.2f})")
  return means, mask, mu, sigma


def plot_eda(rows, means, mask, mu, sigma, out_path=None, show=True):
  import matplotlib.pyplot as plt

  figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 10))
  x = np.arange(len(SPLITS))
  width = 0.35
  normal = [rows[s]["normal"] for s in SPLITS]
  pneumonia = [rows[s]["pneumonia"] for s in SPLITS]
  bars = [
      top.bar(x - width / 2, normal, width, label="Normal", color="#0E7B81"),
      top.bar(x + width / 2, pneumonia, width, label="Pneumonia", color="#FF3C38"),
  ]
  for group in bars:
    top.bar_label(group, fontsize=8, padding=3)
  top.set_xticks(x)
  top.set_xticklabels([s.upper() for s in SPLITS])
  top.set_ylabel("Number of images")
  top.set_title("Class distribution across splits", fontweight="bold")
  top.legend()

  edges = np.linspace(means.min(), means.max(), 45)
  bottom.hist(means[~mask], bins=edges, color="#0E7B81", edgecolor="white", label="|z| <= 3")
  if mask.any():
    bottom.hist(means[mask], bins=edges, color="#FF3C38", edgecolor="white", label="|z| > 3")
  bottom.axvline(mu, color="#FF680A", linewidth=2, label=f"Mean: {mu:.1f}")
  bottom.axvline(mu + 3 * sigma, color="#FF3C38", linestyle="--", label="+3 sigma")
  bottom.axvline(mu - 3 * sigma, color="#FF3C38", linestyle="--", label="-3 sigma")
  bottom.set_xlabel("Mean pixel intensity (0-255)")
  bottom.set_ylabel("Image count")
  bottom.set_title("Mean pixel intensity and outliers", fontweight="bold")
  bottom.legend()

  for axis in (top, bottom):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
  figure.tight_layout()
  if out_path:
    figure.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")
  if show:
    plt.show()
  plt.close(figure)