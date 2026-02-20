"""
Split the processed HR-LR dataset into train/val/test sets.
Verifies pair completeness and saves split files with lr_path|hr_path format.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.image_utils import is_valid_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def verify_pair(lr_path: str, hr_path: str) -> bool:
    """Check that both files exist and are valid images."""
    return Path(lr_path).exists() and Path(hr_path).exists() and \
           is_valid_image(lr_path) and is_valid_image(hr_path)


def estimate_disk_size_mb(paths: list) -> float:
    """Estimate total disk usage of a list of file paths in MB."""
    total_bytes = sum(Path(p).stat().st_size for p in paths if Path(p).exists())
    return total_bytes / (1024 * 1024)


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    random.seed(config["project"]["seed"])
    np.random.seed(config["project"]["seed"])

    dataset_cfg = config["dataset"]
    hr_dir = Path(dataset_cfg["hr_dir"])
    lr_dir = Path(dataset_cfg["lr_dir"])
    splits_dir = Path(dataset_cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_ratio = dataset_cfg["split_ratio"]

    logger.info(f"Scanning HR directory: {hr_dir}")
    hr_files = sorted(hr_dir.glob("*.png"))
    logger.info(f"Found {len(hr_files)} HR files")

    valid_pairs = []
    skipped = 0

    for hr_path in hr_files:
        lr_path = lr_dir / hr_path.name
        hr_str = str(hr_path)
        lr_str = str(lr_path)

        if verify_pair(lr_str, hr_str):
            valid_pairs.append((lr_str, hr_str))
        else:
            logger.warning(f"Incomplete/invalid pair for: {hr_path.name} — skipping")
            skipped += 1

    logger.info(f"Valid pairs: {len(valid_pairs)}, Skipped: {skipped}")

    if not valid_pairs:
        logger.error("No valid pairs found. Run 2_build_dataset.py first.")
        return

    random.shuffle(valid_pairs)
    total = len(valid_pairs)

    n_train = int(total * split_ratio["train"])
    n_val = int(total * split_ratio["val"])
    n_test = total - n_train - n_val

    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:n_train + n_val]
    test_pairs = valid_pairs[n_train + n_val:]

    def write_split(pairs: list, filename: str) -> None:
        path = splits_dir / filename
        with open(str(path), "w") as f:
            for lr, hr in pairs:
                f.write(f"{lr}|{hr}\n")
        logger.info(f"Saved {len(pairs)} pairs to {path}")

    write_split(train_pairs, "train.txt")
    write_split(val_pairs, "val.txt")
    write_split(test_pairs, "test.txt")

    hr_paths = [p[1] for p in valid_pairs]
    lr_paths = [p[0] for p in valid_pairs]
    hr_mb = estimate_disk_size_mb(hr_paths)
    lr_mb = estimate_disk_size_mb(lr_paths)

    print("\n============================================")
    print("DATASET SPLIT SELESAI")
    print("============================================")
    print(f"Total pasangan valid  : {total}")
    print(f"Training set          : {len(train_pairs)} ({split_ratio['train']*100:.0f}%)")
    print(f"Validation set        : {len(val_pairs)} ({split_ratio['val']*100:.0f}%)")
    print(f"Test set              : {len(test_pairs)} ({split_ratio['test']*100:.0f}%)")
    print(f"\nEstimasi ukuran disk:")
    print(f"- HR images           : ~{hr_mb:.1f} MB")
    print(f"- LR images           : ~{lr_mb:.1f} MB")
    print("============================================\n")


if __name__ == "__main__":
    main()