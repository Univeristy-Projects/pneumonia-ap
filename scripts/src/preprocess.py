from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

class DatasetExploration:
    """Performs exploratory data analysis by computing class distributions, pixel intensity statistics, and rendering analytical charts."""
    def __init__(self, root_path="../../../chest_xray_final", classes=None, splits=None):
        self.root = Path(root_path)
        self.classes = classes or ["normal", "pneumonia"]
        self.splits = splits or ["train", "val", "test"]

    def compute_class_distribution(self):
        split_counts = {cls: [] for cls in self.classes}
        for split_name in self.splits:
            split_path = self.root / split_name
            print(f"\n[{split_name.upper()} SET]")
            if not split_path.exists():
                print(f"  Warning: Directory not found at {split_path.resolve()}")
                for cls in self.classes:
                    split_counts[cls].append(0)
                continue
            counts = {}
            total = 0
            for cls in self.classes:
                cls_dir = split_path / cls
                count = len([f for f in cls_dir.iterdir() if f.is_file()]) if cls_dir.exists() else 0
                counts[cls] = count
                total += count
            for cls in self.classes:
                split_counts[cls].append(counts[cls])
            if total > 0:
                for cls, count in counts.items():
                    print(f"  - {cls}: {count} ({(count / total) * 100:.2f}%)")
                majority_class = max(counts, key=counts.get)
                print(f"Majority Baseline: {(counts[majority_class] / total) * 100:.2f}% (guessing '{majority_class}')")
            else:
                print("Warning: No images found in this split.")
        return split_counts



    def analyze_pixel_intensities(self):
        file_paths, mean_intensities = [], []
        for img_path in self.root.glob("**/*.*"):
            if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(img_path).convert("L") as img:
                        mean_intensities.append(np.mean(np.array(img, dtype=np.float32)))
                        file_paths.append(img_path)
                except Exception:
                    pass
        mean_intensities = np.array(mean_intensities)
        if len(mean_intensities) == 0:
            print("Error: No images found to analyze.")
            return None, None, None, None, None
        mu = np.mean(mean_intensities)
        sigma = np.std(mean_intensities) or 1e-5
        z_scores = (mean_intensities - mu) / sigma
        outlier_mask = np.abs(z_scores) > 3
        print(f"Total Images Analyzed: {len(mean_intensities)}")
        print(f"Mean Intensity (μ): {mu:.2f} | Std Dev (σ): {sigma:.2f}")
        print(f"Outliers Detected (|z| > 3): {np.sum(outlier_mask)} images found.")
        if np.any(outlier_mask):
            print("Flagged Outlier Samples:")
            for path, z in zip(np.array(file_paths)[outlier_mask], z_scores[outlier_mask]):
                print(f"  - {path.relative_to(self.root)} (Z-score: {z:.2f})")
        return mean_intensities, outlier_mask, mu, sigma, file_paths

    def plot_eda_dashboard(self, split_counts, mean_intensities, outlier_mask, mu, sigma):
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        ax1 = axes[0]
        x = np.arange(len(self.splits))
        width = 0.35
        rects1 = ax1.bar(x - width / 2, split_counts["normal"], width, label="Normal", color="#0E7B81")
        rects2 = ax1.bar(x + width / 2, split_counts["pneumonia"], width, label="Pneumonia", color="#FF3C38")
        ax1.set_ylabel("Number of Images", fontsize=11)
        ax1.set_title("Class Distribution Across Splits (Stratified Check)", fontsize=13, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.upper() for s in self.splits], fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                ax1.annotate(f"{height}", xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        if mean_intensities is not None:
            ax2 = axes[1]
            bins = np.linspace(mean_intensities.min(), mean_intensities.max(), 45)
            ax2.hist(mean_intensities[~outlier_mask], bins=bins, color="#0E7B81", edgecolor="white", alpha=0.85,
                     label="Normal Range (|z| <= 3)")
            if np.any(outlier_mask):
                ax2.hist(mean_intensities[outlier_mask], bins=bins, color="#FF3C38", edgecolor="white", alpha=0.9,
                         label="Outliers (|z| > 3)")
            ax2.set_xlim(mean_intensities.min() - 5, mean_intensities.max() + 5)
            ax2.axvline(mu, color="#FF680A", linestyle="-", linewidth=2, label=f"Mean: {mu:.1f}")
            ax2.axvline(mu + 3 * sigma, color="#FF3C38", linestyle="--", linewidth=1.5, label="+3σ Threshold")
            ax2.axvline(mu - 3 * sigma, color="#FF3C38", linestyle="--", linewidth=1.5, label="-3σ Threshold")
            ax2.set_title("Mean Pixel Intensity Distribution & Outlier Detection", fontsize=13, fontweight="bold")
            ax2.set_xlabel("Mean Pixel Intensity (0–255 Grayscale)", fontsize=11)
            ax2.set_ylabel("Frequency (Image Count)", fontsize=11)
            ax2.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="upper right")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.show()

class SplitPathLoader:
    """Extracts image file paths and numeric category labels from structured dataset split directories."""
    def __init__(self, root_path="../../../chest_xray_final", classes=None):
        self.root = Path(root_path)
        self.classes = classes or ["normal", "pneumonia"]

    def get_paths_and_labels(self, split_name):
        split_path = self.root / split_name
        image_paths, labels = [], []
        for label_idx, cls_name in enumerate(self.classes):
            cls_dir = split_path / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob("*.*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        image_paths.append(str(img_path))
                        labels.append(label_idx)
        return image_paths, labels

class PneumoniaSequenceDataset(keras.utils.Sequence):
    """Keras Sequence dataset loader handling batching, grayscale decoding, aspect-ratio padding resizing, and training augmentations."""
    def __init__(self, image_paths, labels, img_size=448, batch_size=32, is_training=False):
        self.image_paths = image_paths
        self.labels = np.array(labels)
        self.img_size = img_size
        self.batch_size = batch_size
        self.is_training = is_training
        self.resizer = keras.layers.Resizing(height=self.img_size, width=self.img_size, pad_to_aspect_ratio=True)
        self.indexes = np.arange(len(self.image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size: (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size: (index + 1) * self.batch_size]

        batch_images = []
        for path in batch_paths:
            # 1. Load and convert to grayscale
            img = Image.open(path).convert("L")
            img_arr = np.array(img, dtype=np.float32) / 255.0

            # Ensure channel dimension exists (H, W, 1)
            if len(img_arr.shape) == 2:
                img_arr = np.expand_dims(img_arr, axis=-1)

            # 2. Apply aspect-ratio-preserving resize/pad (via self.resizer)
            img_tensor = tf.convert_to_tensor(img_arr)
            resized_img = self.resizer(img_tensor)

            # 3. Apply conditional training augmentations
            if self.is_training:
                resized_img = self._augment(resized_img)

            batch_images.append(resized_img)

        # Stack into a unified batch tensor (B, H, W, C)
        images = tf.stack(batch_images)
        labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)

        return images, labels

    def _augment(self, img_tensor):
        # Random horizontal flip
        img_tensor = tf.image.random_flip_left_right(img_tensor)
        # Random brightness adjustment
        img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.1)
        # Random contrast adjustment
        img_tensor = tf.image.random_contrast(img_tensor, lower=0.9, upper=1.1)
        # Clip values back to valid [0.0, 1.0] range
        return tf.clip_by_value(img_tensor, 0.0, 1.0)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)