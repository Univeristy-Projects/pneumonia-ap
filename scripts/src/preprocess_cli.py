# Usage (run from scripts/src; make_splits.py must have written splits.json):
#   python preprocess_cli.py cache --manifest splits.json --data-dir ../../../chest_xray --cache-dir ../../../chest_xray_cache --size 224
#   python preprocess_cli.py eda --manifest splits.json --cache-dir ../../../chest_xray_cache --size 224 --plot-out eda.png
#   python preprocess_cli.py verify-batch --manifest splits.json --cache-dir ../../../chest_xray_cache --size 224 --split train
# cache: resize every image once and store uint8 arrays | eda: class counts and intensity outliers | verify-batch: inspect a batch
import argparse

import numpy as np

from preprocess import (
    SPLITS,
    build_cache,
    build_dataset,
    class_distribution,
    class_weights,
    intensity_outliers,
    load_manifest,
    load_split,
    plot_eda,
)


def parse_args():
  parser = argparse.ArgumentParser()
  commands = parser.add_subparsers(dest="command", required=True)

  def shared(command):
    command.add_argument("--manifest", default="splits.json")
    command.add_argument("--cache-dir", default="../../../chest_xray_cache")
    command.add_argument("--size", type=int, default=224)

  cache = commands.add_parser("cache")
  shared(cache)
  cache.add_argument("--data-dir", default="../../../chest_xray")
  cache.add_argument("--overwrite", action="store_true")

  eda = commands.add_parser("eda")
  shared(eda)
  eda.add_argument("--plot-out")
  eda.add_argument("--skip-plots", action="store_true")

  verify = commands.add_parser("verify-batch")
  shared(verify)
  verify.add_argument("--split", choices=SPLITS, default="train")
  verify.add_argument("--batch-size", type=int, default=32)
  verify.add_argument("--seed", type=int, default=0)
  verify.add_argument("--hflip", action="store_true")
  return parser.parse_args()


def run_cache(args):
  build_cache(
      args.manifest, args.data_dir, args.cache_dir, args.size, args.overwrite
  )


def run_eda(args):
  _, by_split = load_manifest(args.manifest)
  rows = class_distribution(by_split)
  means, mask, mu, sigma = intensity_outliers(
      args.cache_dir, args.size, args.manifest
  )
  if args.plot_out or not args.skip_plots:
    plot_eda(
        rows, means, mask, mu, sigma,
        out_path=args.plot_out,
        show=not args.skip_plots,
    )


def first_batch(images, labels, args, training):
  dataset = build_dataset(
      images, labels, args.batch_size,
      training=training, seed=args.seed, hflip=args.hflip,
  )
  return next(iter(dataset))


def run_verify(args):
  images, labels, _ = load_split(
      args.cache_dir, args.size, args.split, args.manifest
  )
  training = args.split == "train"
  batch_images, batch_labels = first_batch(images, labels, args, training)
  batch_images = batch_images.numpy()
  batch_labels = batch_labels.numpy()
  print(f"split: {args.split} | samples: {len(images)} | training mode: {training}")
  print(f"batch images: {batch_images.shape} {batch_images.dtype}")
  print(f"pixel range after preprocess_input: [{batch_images.min():.1f}, {batch_images.max():.1f}]")
  print(f"batch labels: normal {int((batch_labels == 0).sum())} | pneumonia {int((batch_labels == 1).sum())}")
  print(f"class weights: {class_weights(labels)}")
  again_images, again_labels = first_batch(images, labels, args, training)
  same = np.array_equal(batch_images, again_images.numpy()) and np.array_equal(
      batch_labels, again_labels.numpy()
  )
  print(f"same seed gives identical first batch: {same}")


def main():
  args = parse_args()
  if args.command == "cache":
    run_cache(args)
  elif args.command == "eda":
    run_eda(args)
  else:
    run_verify(args)


if __name__ == "__main__":
  main()