import argparse
from pathlib import Path
from preprocess import (
    DatasetExploration,
    SplitPathLoader,
    PneumoniaSequenceDataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to run dataset analysis, split loading, and dataset verification for Pediatric Pneumonia Chest X-Rays."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda_parser = subparsers.add_parser(
        "eda",
        help="Run exploratory data analysis (class distribution & pixel intensity outliers).",
    )
    eda_parser.add_argument(
        "--data-dir",
        type=str,
        default="../../../chest_xray_final",
        help="Path to the root dataset directory containing train/val/test splits.",
    )
    eda_parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Run statistical computations and prints without displaying matplotlib windows.",
    )

    load_parser = subparsers.add_parser(
        "load", help="Extract image paths and binary labels for a specific split."
    )
    load_parser.add_argument(
        "--data-dir",
        type=str,
        default="../../../chest_xray_final",
        help="Path to the root dataset directory.",
    )
    load_parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Target dataset split name.",
    )

    verify_parser = subparsers.add_parser(
        "verify-batch",
        help="Instantiate Keras Sequence and inspect a generated batch.",
    )
    verify_parser.add_argument(
        "--data-dir",
        type=str,
        default="../../../chest_xray_final",
        help="Path to the root dataset directory.",
    )
    verify_parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to instantiate.",
    )
    verify_parser.add_argument(
        "--img-size",
        type=int,
        default=448,
        help="Target height/width for aspect-ratio padded resizing.",
    )
    verify_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for dataset sequence.",
    )

    args = parser.parse_args()

    match args.command:
        case "eda":

            explorer = DatasetExploration(root_path=args.data_dir)
            split_counts = explorer.compute_class_distribution()
            mean_intensities, outlier_mask, mu, sigma, _ = explorer.analyze_pixel_intensities()
            if not args.skip_plots:
                explorer.plot_eda_dashboard(split_counts, mean_intensities, outlier_mask, mu, sigma)

        case "load":
            loader = SplitPathLoader(root_path=args.data_dir)
            paths, labels = loader.get_paths_and_labels(args.split)
            print(f"[{args.split.upper()} SET]")
            print(f"Total samples loaded: {len(paths)}")
            print(
                f"Normal (0): {labels.count(0)} | Pneumonia (1): {labels.count(1)}"
            )

        case "verify-batch":
            loader = SplitPathLoader(root_path=args.data_dir)
            paths, labels = loader.get_paths_and_labels(args.split)
            is_train = args.split == "train"
            dataset = PneumoniaSequenceDataset(
                paths,
                labels,
                img_size=args.img_size,
                batch_size=args.batch_size,
                is_training=is_train,
            )
            print(f"Total Batches: {len(dataset)}")
            if len(dataset) > 0:
                images, batch_labels = dataset[0]
                print(f"Batch 0 Images Shape : {images.shape}")
                print(f"Batch 0 Labels Shape : {batch_labels.shape}")
                print(
                    f"Pixel Range          : [{images.numpy().min():.2f}, {images.numpy().max():.2f}]"
                )


if __name__ == "__main__":
    main()