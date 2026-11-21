import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import tensorflow as tf
import glob

from loader import load_split
from preprocess import build_dataset
from models import LayerScale  # Ensures custom layer registration is loaded


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


def evaluate_multi_seed_val(data_dir: str, model_name: str, num_blocks: int = 2, batch_size: int = 32):
    print(f"Loading validation data from: {Path(data_dir) / 'val'}")
    xva, yva = load_split(data_dir, "val")
    val_ds = build_dataset(xva, yva, batch_size=batch_size)

    block_based_models = ["residual_cnn", "convnext"]
    if model_name in block_based_models:
        file_pattern = f"models/pneumonia_{model_name}_b{num_blocks}_seed*.keras"
        model_tag = f"{model_name}_b{num_blocks}"
    else:
        file_pattern = f"models/pneumonia_{model_name}_seed*.keras"
        model_tag = model_name

    model_paths = sorted(glob.glob(file_pattern))

    if not model_paths:
        raise FileNotFoundError(f"No model files found matching pattern: {file_pattern}")

    print(f"Found {len(model_paths)} models to evaluate on validation set for: {model_tag}")

    recalls = []
    precisions = []
    aucs = []
    thresholds_used = []
    all_cms = []

    last_y_true = None
    last_y_pred = None

    for model_path in model_paths:
        print(f"\nEvaluating {model_path} on validation split...")
        model = tf.keras.models.load_model(model_path)

        y_true = []
        y_pred_probs = []

        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred_probs.extend(preds.flatten())

        y_true = np.array(y_true, dtype=int)
        y_pred_probs = np.array(y_pred_probs)

        # Dynamic Youden's J optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        thresholds_used.append(optimal_threshold)

        y_pred = (y_pred_probs >= 0.4).astype(int)

        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_probs)

        recalls.append(rec)
        precisions.append(prec)
        aucs.append(auc)
        all_cms.append(confusion_matrix(y_true, y_pred))

        last_y_true = y_true
        last_y_pred = y_pred

    # --- Student-t Confidence Intervals across n runs ---
    print("\n" + "=" * 55)
    print(f"--- Multi-Seed VALIDATION Summary (n = {len(model_paths)}) ---")
    print("=" * 55)

    metrics = {
        "Val Recall (Pneumonia)": recalls,
        "Val Precision (Pneumonia)": precisions,
        "Val ROC-AUC": aucs,
    }

    report_lines = [f"=== Multi-Seed Validation Report: {model_tag} (n={len(model_paths)}) ==="]
    for name, vals in metrics.items():
        mean, lower, upper, margin = compute_student_t_ci(vals)
        line = f"{name:<26}: {mean:.4f}  (95% CI: [{lower:.4f}, {upper:.4f}] | ±{margin:.4f})"
        print(line)
        report_lines.append(line)

    print("=" * 55)

    target_names = ["Normal (0)", "Pneumonia (1)"]
    print("\nValidation Classification Report (Last Seed Run):")
    report_lines.append("\nValidation Classification Report (Last Seed Run):")
    cls_rep = classification_report(last_y_true, last_y_pred, target_names=target_names)
    print(cls_rep)
    report_lines.append(cls_rep)

    txt_path = f"{model_tag}_val_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Validation report saved successfully to: {txt_path}")

    # Sum confusion matrices across all n seeds
    agg_cm = np.mean(all_cms, axis=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        agg_cm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Aggregated Validation Confusion Matrix (n={len(model_paths)}) - {model_tag}")
    plt.tight_layout()

    plot_path = f"{model_tag}_val_confusion_matrix.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Validation confusion matrix saved successfully to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-seed VALIDATION evaluation script with Student-t confidence intervals."
    )
    parser.add_argument("--model_name", default="simple_cnn", help="Architecture name to evaluate")
    parser.add_argument("--num_blocks", type=int, default=3, help="Number of blocks for residual_cnn / convnext")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../chest_xray_cache",
        help="Path to directory containing val split.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation."
    )

    args = parser.parse_args()
    evaluate_multi_seed_val(
        data_dir=args.data_dir,
        model_name=args.model_name,
        num_blocks=args.num_blocks,
        batch_size=args.batch_size,
    )