import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import argparse


def plot_averaged_history(model_name: str, num_blocks: int = 3, runs_dir: str = "runs"):
    if model_name in ["residual_cnn", "convnext"]:
        tag = f"{model_name}_b{num_blocks}"
    else:
        tag = model_name

    # Find all json files for this architecture across the 5 seeds
    file_pattern = f"{runs_dir}/pneumonia_{tag}_seed*_*.json"  # matches seed 42 to 46
    json_files = sorted(glob.glob(f"{runs_dir}/pneumonia_{tag}_seed*.json"))

    if not json_files:
        print(f"No history JSON files found in '{runs_dir}/' matching pattern for {tag}.")
        return

    print(f"Found {len(json_files)} seed records to average.")

    all_train_losses = []
    all_val_losses = []
    all_val_aucs = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
            hist = data["history"]
            all_train_losses.append(hist['loss'])
            all_val_losses.append(hist['val_loss'])
            all_val_aucs.append(hist['val_auc'])

    # Truncate to shortest run for clean numpy averaging (due to early stopping)
    min_epochs = min(len(l) for l in all_train_losses)

    train_losses = np.array([l[:min_epochs] for l in all_train_losses])
    val_losses = np.array([l[:min_epochs] for l in all_val_losses])
    val_aucs = np.array([l[:min_epochs] for l in all_val_aucs])

    epochs = np.arange(1, min_epochs + 1)

    # Compute Mean and Standard Deviation across seeds
    mean_train_loss = np.mean(train_losses, axis=0)
    std_train_loss = np.std(train_losses, axis=0)

    mean_val_loss = np.mean(val_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)

    mean_val_auc = np.mean(val_aucs, axis=0)
    std_val_auc = np.std(val_aucs, axis=0)

    # --- Plotting 1 Single Averaged Graph ---
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mean_train_loss, label="Train Loss (Mean)", color="blue")
    plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color="blue",
                     alpha=0.15)
    plt.plot(epochs, mean_val_loss, label="Val Loss (Mean)", color="orange", linestyle="--")
    plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color="orange", alpha=0.15)
    plt.title(f"Averaged Loss Curve (n=5) - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Validation AUC Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mean_val_auc, label="Val AUC (Mean)", color="green")
    plt.fill_between(epochs, mean_val_auc - std_val_auc, mean_val_auc + std_val_auc, color="green", alpha=0.15)
    plt.title(f"Averaged Validation AUC (n=5) - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"{tag}_averaged_training_plot.png"
    plt.savefig(output_path, dpi=150)
    print(f"[INFO] Single averaged summary plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="simple_cnn")
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--runs_dir", default="runs")
    args = parser.parse_args()

    plot_averaged_history(args.model_name, args.num_blocks, args.runs_dir)