"""
PyTorch Dataset and DataLoader factory for the Babad Banyumas manuscript dataset.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from src.data.augmentation import PairedAugmentation
from src.utils.image_utils import load_image, numpy_to_tensor, random_crop


class BabadBanyumasDataset(Dataset):
    """
    Dataset for LR-HR manuscript image pairs.
    Reads pairs from a split file with format: lr_path|hr_path (one per line).
    """

    def __init__(self, split_file: str, config: dict, mode: str = "train"):
        """
        Args:
            split_file: Path to splits/train.txt, val.txt, or test.txt.
            config: Full project config dict.
            mode: 'train' applies augmentation; 'val'/'test' does not.
        """
        self.config = config
        self.mode = mode
        self.patch_size = config["training"]["patch_size"]
        self.lr_scale = config["dataset"]["lr_scale"]

        self.pairs = []
        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("|")
                    if len(parts) == 2:
                        self.pairs.append((parts[0], parts[1]))

        self.augmentation = PairedAugmentation(config) if mode == "train" else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return an LR-HR pair as tensors.

        Returns:
            dict with keys:
                'lr': float32 tensor [3, lr_h, lr_w] in [0, 1]
                'hr': float32 tensor [3, hr_h, hr_w] in [0, 1]
                'lr_path': str
                'hr_path': str
        """
        lr_path, hr_path = self.pairs[idx]

        try:
            if self.mode == "train":
                lr, hr = self._load_and_crop(lr_path, hr_path)
            else:
                lr = load_image(lr_path)
                hr = load_image(hr_path)

            if self.mode == "train" and self.augmentation is not None:
                lr, hr = self.augmentation(lr, hr)

            lr_tensor = numpy_to_tensor(lr)
            hr_tensor = numpy_to_tensor(hr)

        except Exception as e:
            lr_tensor = torch.zeros(3, self.patch_size, self.patch_size)
            hr_tensor = torch.zeros(3, self.patch_size * self.lr_scale, self.patch_size * self.lr_scale)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "lr_path": lr_path,
            "hr_path": hr_path,
        }

    def _load_and_crop(
        self, lr_path: str, hr_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load LR and HR images and extract aligned patch crops for training.

        Args:
            lr_path: Path to LR image.
            hr_path: Path to HR image.

        Returns:
            Tuple (lr_crop, hr_crop) as numpy arrays.
        """
        hr = load_image(hr_path)
        lr = load_image(lr_path)

        hr_h, hr_w = hr.shape[:2]
        lr_h, lr_w = lr.shape[:2]

        hr_patch_size = self.patch_size * self.lr_scale
        lr_patch_size = self.patch_size

        if hr_h < hr_patch_size or hr_w < hr_patch_size:
            import cv2
            hr = cv2.resize(hr, (max(hr_w, hr_patch_size), max(hr_h, hr_patch_size)))
            lr = cv2.resize(lr, (max(lr_w, lr_patch_size), max(lr_h, lr_patch_size)))
            hr_h, hr_w = hr.shape[:2]
            lr_h, lr_w = lr.shape[:2]

        top_hr = np.random.randint(0, hr_h - hr_patch_size + 1)
        left_hr = np.random.randint(0, hr_w - hr_patch_size + 1)

        top_lr = top_hr // self.lr_scale
        left_lr = left_hr // self.lr_scale

        top_lr = min(top_lr, lr_h - lr_patch_size)
        left_lr = min(left_lr, lr_w - lr_patch_size)
        top_lr = max(0, top_lr)
        left_lr = max(0, left_lr)

        hr_crop = hr[top_hr:top_hr + hr_patch_size, left_hr:left_hr + hr_patch_size]
        lr_crop = lr[top_lr:top_lr + lr_patch_size, left_lr:left_lr + lr_patch_size]

        return lr_crop, hr_crop


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from config split files.

    Args:
        config: Full project config dict.

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    splits_dir = Path(config["dataset"]["splits_dir"])
    num_workers = config["training"]["num_workers"]
    batch_size = config["training"]["batch_size"]

    train_dataset = BabadBanyumasDataset(
        split_file=str(splits_dir / "train.txt"),
        config=config,
        mode="train"
    )
    val_dataset = BabadBanyumasDataset(
        split_file=str(splits_dir / "val.txt"),
        config=config,
        mode="val"
    )
    test_dataset = BabadBanyumasDataset(
        split_file=str(splits_dir / "test.txt"),
        config=config,
        mode="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader