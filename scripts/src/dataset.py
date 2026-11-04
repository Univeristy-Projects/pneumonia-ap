# Usage (run from scripts/src, --data-dir must contain train/, val/, test/):
#   python make_splits.py --data-dir ../../../chest_xray --mode audit
#   python make_splits.py --data-dir ../../../chest_xray --mode group
#   python make_splits.py --data-dir ../../../chest_xray --mode group --max-train-per-patient 3
# Modes: audit = print counts only | group = keep all images, split by patient
# Options: --seed 42  --ratios 0.70 0.15 0.15  --out path/to/splits.json
# Output: splits.json (path, label, group, split per image); normal = 0, pneumonia = 1
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = {"normal": 0, "pneumonia": 1}
SPLIT_DIRS = {"train", "val", "test"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PATIENT_PATTERN = re.compile(r"person\d+|IM-\d+", re.IGNORECASE)


def collect_records(data_dir):
  records = []
  for split_dir in sorted(data_dir.iterdir()):
    if not split_dir.is_dir() or split_dir.name.lower() not in SPLIT_DIRS:
      continue
    for class_dir in sorted(split_dir.iterdir()):
      label = CLASSES.get(class_dir.name.lower())
      if label is None or not class_dir.is_dir():
        continue
      for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
          records.append({
              "path": path.relative_to(data_dir).as_posix(),
              "name": path.name,
              "label": label,
          })
  return sorted(records, key=lambda r: r["path"])


def sha256_of(path):
  digest = hashlib.sha256()
  with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


class UnionFind:

  def __init__(self):
    self.parent = {}

  def find(self, item):
    self.parent.setdefault(item, item)
    while self.parent[item] != item:
      self.parent[item] = self.parent[self.parent[item]]
      item = self.parent[item]
    return item

  def union(self, a, b):
    root_a, root_b = self.find(a), self.find(b)
    if root_a != root_b:
      self.parent[max(root_a, root_b)] = min(root_a, root_b)


def patient_key(record):
  match = PATIENT_PATTERN.search(record["name"])
  token = match.group(0).lower() if match else Path(record["name"]).stem.lower()
  return f"{record['label']}:{token}", match is not None


def assign_groups(records, data_dir):
  union = UnionFind()
  by_hash = defaultdict(list)
  unmatched = 0
  for record in records:
    key, matched = patient_key(record)
    record["group"] = key
    union.find(key)
    unmatched += not matched
    by_hash[sha256_of(data_dir / record["path"])].append(key)
  duplicate_files = 0
  for keys in by_hash.values():
    if len(keys) > 1:
      duplicate_files += len(keys) - 1
      for key in keys[1:]:
        union.union(keys[0], key)
  for record in records:
    record["group"] = union.find(record["group"])
  return {"unmatched_names": unmatched, "exact_duplicate_files": duplicate_files}


def drop_mixed_label_groups(records):
  labels = defaultdict(set)
  for record in records:
    labels[record["group"]].add(record["label"])
  mixed = {group for group, found in labels.items() if len(found) > 1}
  kept = [r for r in records if r["group"] not in mixed]
  return kept, len(mixed)


def split_groups(records, ratios, seed):
  rng = random.Random(seed)
  for label in sorted(CLASSES.values()):
    groups = defaultdict(list)
    for record in records:
      if record["label"] == label:
        groups[record["group"]].append(record)
    ordered = sorted(groups)
    rng.shuffle(ordered)
    total = sum(len(items) for items in groups.values())
    train_end = ratios[0] * total
    val_end = (ratios[0] + ratios[1]) * total
    seen = 0
    for group in ordered:
      if seen < train_end:
        split = "train"
      elif seen < val_end:
        split = "val"
      else:
        split = "test"
      for record in groups[group]:
        record["split"] = split
      seen += len(groups[group])


def cap_train(records, limit, seed):
  rng = random.Random(seed)
  per_group = defaultdict(list)
  for record in records:
    if record["split"] == "train":
      per_group[record["group"]].append(record)
  dropped = set()
  for group in sorted(per_group):
    items = per_group[group]
    if len(items) > limit:
      rng.shuffle(items)
      dropped.update(r["path"] for r in items[limit:])
  return [r for r in records if r["path"] not in dropped], len(dropped)


def assert_no_group_leakage(records):
  homes = defaultdict(set)
  for record in records:
    homes[record["group"]].add(record["split"])
  leaked = [group for group, splits in homes.items() if len(splits) > 1]
  if leaked:
    raise RuntimeError(f"{len(leaked)} groups appear in more than one split")


def summarize(records):
  summary = {}
  print(f"{'split':<6} {'normal':>7} {'pneumonia':>10} {'total':>7} {'pneu%':>7} {'groups':>7}")
  for split in ("train", "val", "test"):
    subset = [r for r in records if r["split"] == split]
    normal = sum(r["label"] == 0 for r in subset)
    pneumonia = sum(r["label"] == 1 for r in subset)
    total = normal + pneumonia
    groups = len({r["group"] for r in subset})
    share = 100 * pneumonia / total if total else 0.0
    print(f"{split:<6} {normal:>7} {pneumonia:>10} {total:>7} {share:>6.2f}% {groups:>7}")
    summary[split] = {"normal": normal, "pneumonia": pneumonia, "groups": groups}
  positives = summary["test"]["pneumonia"]
  if positives:
    step = 100 / positives
    half_width = 196 * math.sqrt(0.97 * 0.03 / positives)
    print(f"test pneumonia images: {positives} | one image = {step:.2f}% recall")
    print(f"recall 95% half-width at 0.97 (test sampling only): ~{half_width:.2f} points")
  return summary


def audit(records, info):
  print(f"images: {len(records)}")
  print(f"filenames without a patient pattern: {info['unmatched_names']}")
  print(f"exact duplicate files (same bytes): {info['exact_duplicate_files']}")
  for name, label in CLASSES.items():
    counts = Counter(r["group"] for r in records if r["label"] == label)
    sizes = Counter(counts.values())
    multi = sum(1 for size in counts.values() if size > 1)
    print(f"\n[{name}] images: {sum(counts.values())} | patients: {len(counts)}")
    print(f"  patients with more than one image: {multi}")
    print(f"  max images per patient: {max(counts.values(), default=0)}")
    print(f"  scans-per-patient histogram: {dict(sorted(sizes.items())[:10])}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--mode", choices=["audit", "group"], required=True)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
  parser.add_argument("--max-train-per-patient", type=int)
  args = parser.parse_args()

  if abs(sum(args.ratios) - 1.0) > 1e-6:
    parser.error("ratios must sum to 1")
  if args.max_train_per_patient is not None and args.max_train_per_patient < 1:
    parser.error("--max-train-per-patient must be at least 1")

  data_dir = args.data_dir.resolve()
  records = collect_records(data_dir)
  if not records:
    raise SystemExit(f"no images found under {data_dir}")

  info = assign_groups(records, data_dir)
  records, mixed = drop_mixed_label_groups(records)
  info["mixed_label_groups_dropped"] = mixed

  if args.mode == "audit":
    audit(records, info)
    return

  split_groups(records, args.ratios, args.seed)

  capped = 0
  if args.max_train_per_patient is not None:
    records, capped = cap_train(records, args.max_train_per_patient, args.seed)

  assert_no_group_leakage(records)
  print(f"mode: {args.mode} | seed: {args.seed} | images kept: {len(records)}")
  print(f"mixed-label groups dropped: {mixed} | train images capped: {capped}")
  summary = summarize(records)

  out = args.out or Path("splits.json")
  manifest = {
      "meta": {
          "mode": args.mode,
          "seed": args.seed,
          "ratios": args.ratios,
          "max_train_per_patient": args.max_train_per_patient,
          "label_map": CLASSES,
          "counts": summary,
          "info": info,
      },
      "records": [
          {
              "path": r["path"],
              "label": r["label"],
              "group": r["group"],
              "split": r["split"],
          }
          for r in sorted(records, key=lambda r: r["path"])
      ],
  }
  out.write_text(json.dumps(manifest, indent=1))
  print(f"wrote {out.resolve()}")


if __name__ == "__main__":
  main()