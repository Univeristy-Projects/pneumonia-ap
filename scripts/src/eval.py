import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
import tensorflow as tf

from loader import load_split
from preprocess import build_dataset


def compute_student_t_ci(values, confidence=0.95):
    """Computes mean and Student-t confidence interval for n runs:
    mean ± t_{n-1} * (sigma / sqrt(n))
    """
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, mean, mean, 0.0

    std_dev = np.std(values, ddof=1)
    std_err = std_dev / np.sqrt(n)


    t_crit = stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    margin = t_crit * std_err

    return mean, mean - margin, mean + margin, margin


def evaluate_multi_seed(data_dir: str, model_pattern: str = "pneumonia_model_seed*.keras", batch_size: int = 32):
    print(f"Loading test data from: {Path(data_dir) / 'test'}")
    xts, yts = load_split(data_dir, "test")
    test_ds = build_dataset(xts, yts, batch_size=batch_size)

    model_files = sorted(list(Path(".").glob(model_pattern)))
    if not model_files:
        raise FileNotFoundError(f"No model files found matching pattern: {model_pattern}")

    print(f"\nFound {len(model_files)} seed models to evaluate: {[f.name for f in model_files]}")

    recalls = []
    precisions = []
    aucs = []

    last_y_true = None
    last_y_pred = None
    last_cm = None

    for model_path in model_files:
        print(f"\nEvaluating {model_path.name}...")
        model = tf.keras.models.load_model(model_path)

        y_true = []
        y_pred_probs = []

        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred_probs.extend(preds.flatten())

        y_true = np.array(y_true, dtype=int)
        y_pred_probs = np.array(y_pred_probs)
        y_pred = (y_pred_probs >= 0.5).astype(int)


        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_probs)

        recalls.append(rec)
        precisions.append(prec)
        aucs.append(auc)


        last_y_true = y_true
        last_y_pred = y_pred
        last_cm = confusion_matrix(y_true, y_pred)

    # --- Student-t Confidence Intervals across n = 5 runs ---
    print("\n" + "=" * 50)
    print(f"--- Multi-Seed Evaluation Summary (n = {len(model_files)}) ---")
    print("=" * 50)

    metrics = {
        "Recall (Pneumonia)": recalls,
        "Precision (Pneumonia)": precisions,
        "ROC-AUC": aucs
    }

    ci_results = {}
    for name, vals in metrics.items():
        mean, lower, upper, margin = compute_student_t_ci(vals)
        ci_results[name] = {"mean": mean, "lower": lower, "upper": upper, "margin": margin}
        print(f"{name:<22}: {mean:.4f}  (95% CI: [{lower:.4f}, {upper:.4f}] | ±{margin:.4f})")

    print("=" * 50)


    target_names = ["Normal (0)", "Pneumonia (1)"]
    print("\nClassification Report (Last Seed Run):")
    print(classification_report(last_y_true, last_y_pred, target_names=target_names))

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        last_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Multi-Seed Test Sample")
    plt.tight_layout()

    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"\nConfusion matrix plot saved successfully to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-seed evaluation script with Student-t confidence intervals."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../chest_xray_cache",
        help="Path to directory containing test split.",
    )
    parser.add_argument(
        "--model_pattern",
        type=str,
        default="pneumonia_model_seed*.keras",
        help="Glob pattern for saved seed models.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation."
    )

    args = parser.parse_args()
    evaluate_multi_seed(
        data_dir=args.data_dir,
        model_pattern=args.model_pattern,
        batch_size=args.batch_size,
    )