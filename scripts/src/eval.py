import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
import tensorflow as tf
import glob
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


def evaluate_multi_seed(data_dir: str, model_name: str, num_blocks: int = 2, batch_size: int = 32):
    print(f"Loading test data from: {Path(data_dir) / 'test'}")
    xts, yts = load_split(data_dir, "test")
    test_ds = build_dataset(xts, yts, batch_size=batch_size)

    if args.model_name == "residual_cnn":
        file_pattern = f"pneumonia_{args.model_name}_b{args.num_blocks}_seed*.keras"
    else:
        file_pattern = f"pneumonia_{args.model_name}_seed*.keras"

    model_paths = sorted(glob.glob(file_pattern))

    if not model_paths:
        raise FileNotFoundError(f"No model files found matching pattern: {file_pattern}")

    print(f"Found {len(model_paths)} models to evaluate for architecture: {args.model_name}")

    recalls = []
    precisions = []
    aucs = []

    last_y_true = None
    last_y_pred = None
    last_cm = None

    for model_path in model_paths:
        print(f"\nEvaluating {model_path}...")
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
    print(f"--- Multi-Seed Evaluation Summary (n = {len(model_paths)}) ---")
    print("=" * 50)

    metrics = {
        "Recall (Pneumonia)": recalls,
        "Precision (Pneumonia)": precisions,
        "ROC-AUC": aucs
    }
    report_lines=[]
    ci_results = {}
    for name, vals in metrics.items():
        mean, lower, upper, margin = compute_student_t_ci(vals)
        ci_results[name] = {"mean": mean, "lower": lower, "upper": upper, "margin": margin}
        print(f"{name:<22}: {mean:.4f}  (95% CI: [{lower:.4f}, {upper:.4f}] | ±{margin:.4f})")
        report_lines.append(f"{name:<22}: {mean:.4f}  (95% CI: [{lower:.4f}, {upper:.4f}] | ±{margin:.4f})")

    print("=" * 50)
    if model_name == "residual_cnn":
        model_tag = f"{model_name}_b{num_blocks}"
    else:
        model_tag = model_name

    target_names = ["Normal (0)", "Pneumonia (1)"]
    print("\nClassification Report (Last Seed Run):")
    report_lines.append("\nClassification Report (Last Seed Run):")
    print(classification_report(last_y_true, last_y_pred, target_names=target_names))
    report_lines.append(classification_report(last_y_true, last_y_pred, target_names=target_names))
    full_report = "\n".join(report_lines)
    txt_path = f"{model_tag}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_report + "\n")
    print(f"Evaluation report saved successfully to {txt_path}")

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

    plot_path = f"{model_tag}_confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"\nConfusion matrix plot saved successfully to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-seed evaluation script with Student-t confidence intervals."
    )
    parser.add_argument("--model_name", default="simple_cnn", help="Architecture name to evaluate")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of blocks if evaluating residual_cnn")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../chest_xray_cache",
        help="Path to directory containing test split.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation."
    )

    args = parser.parse_args()
    evaluate_multi_seed(
        data_dir=args.data_dir,
        model_name=args.model_name,
        num_blocks=args.num_blocks,
        batch_size=args.batch_size,
    )

