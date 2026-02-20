"""
Paired augmentation for LR-HR image pairs.
Geometric transforms applied identically to both; color only to LR.
"""

import cv2
import numpy as np
import random
from typing import Tuple


class PairedAugmentation:
    """
    Augmentation pipeline for LR-HR pairs.
    Geometric augmentation is applied identically to both LR and HR.
    Color augmentation is applied only to LR to simulate scanner variance.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full project config dict (reads config['augmentation']).
        """
        self.cfg = config["augmentation"]

    def _apply_geometric(
        self, lr: np.ndarray, hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the same geometric transformations to both LR and HR images.

        Transformations: horizontal flip, vertical flip, 90/180/270 rotation.

        Args:
            lr: LR image [H, W, C] uint8.
            hr: HR image [H, W, C] uint8.

        Returns:
            Tuple (augmented_lr, augmented_hr) with identical transforms applied.
        """
        # Horizontal flip
        if random.random() < self.cfg["horizontal_flip_prob"]:
            lr = cv2.flip(lr, 1)
            hr = cv2.flip(hr, 1)

        # Vertical flip
        if random.random() < self.cfg["vertical_flip_prob"]:
            lr = cv2.flip(lr, 0)
            hr = cv2.flip(hr, 0)

        # Rotation (90, 180, 270 degrees)
        if random.random() < self.cfg["rotation_prob"]:
            k = random.choice([1, 2, 3])
            lr = np.rot90(lr, k)
            hr = np.rot90(hr, k)

        return lr, hr

    def _apply_color_to_lr(self, lr: np.ndarray) -> np.ndarray:
        """
        Apply brightness and contrast jitter only to the LR image.

        Args:
            lr: LR image [H, W, C] uint8.

        Returns:
            Color-augmented LR image.
        """
        brightness_delta = random.uniform(*self.cfg["brightness_range"])
        contrast_factor = random.uniform(*self.cfg["contrast_range"])

        img = lr.astype(np.float32)
        img = img * contrast_factor + brightness_delta * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __call__(
        self, lr: np.ndarray, hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the full augmentation pipeline to a LR-HR pair.

        Args:
            lr: LR image [H, W, C] uint8.
            hr: HR image [H, W, C] uint8.

        Returns:
            Tuple (augmented_lr, augmented_hr).
        """
        lr, hr = self._apply_geometric(lr, hr)
        lr = self._apply_color_to_lr(lr)
        return lr, hr