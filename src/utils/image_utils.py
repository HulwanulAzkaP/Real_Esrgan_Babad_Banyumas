"""
Image utility functions for loading, saving, converting, and transforming images.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple


def load_image(path: str) -> np.ndarray:
    """
    Load an image as a numpy array in BGR format (OpenCV default).

    Args:
        path: File path to the image.

    Returns:
        numpy array of shape [H, W, C] in BGR uint8 format.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save a numpy array as an image file. Creates parent directories automatically.

    Args:
        image: numpy array [H, W, C] in BGR uint8 format.
        path: Destination file path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a float tensor [C, H, W] in range [0, 1] to numpy uint8 [H, W, C] BGR.

    Args:
        tensor: torch.Tensor of shape [C, H, W] with values in [0, 1].

    Returns:
        numpy array of shape [H, W, C] in BGR uint8 format.
    """
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    img = tensor.numpy()
    img = (img * 255.0).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy uint8 [H, W, C] BGR image to a float tensor [C, H, W] in [0, 1].

    Args:
        image: numpy array of shape [H, W, C] in BGR uint8 format.

    Returns:
        torch.Tensor of shape [C, H, W] with values in [0, 1].
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    return tensor


def resize_image(image: np.ndarray, size: Tuple[int, int], method: str = "bicubic") -> np.ndarray:
    """
    Resize an image to the given (width, height) using the specified interpolation method.

    Args:
        image: numpy array [H, W, C] in BGR uint8 format.
        size: Target (width, height) tuple.
        method: Interpolation method: 'bicubic', 'bilinear', 'area', or 'nearest'.

    Returns:
        Resized numpy array.
    """
    method_map = {
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
    }
    interp = method_map.get(method, cv2.INTER_CUBIC)
    return cv2.resize(image, size, interpolation=interp)


def random_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Extract a random square crop from an image.

    Args:
        image: numpy array [H, W, C].
        crop_size: Side length of the square crop in pixels.

    Returns:
        Cropped numpy array of shape [crop_size, crop_size, C].

    Raises:
        ValueError: If the image is smaller than crop_size in either dimension.
    """
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Image ({h}x{w}) is smaller than crop size ({crop_size}x{crop_size})"
        )
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    return image[top:top + crop_size, left:left + crop_size]


def is_valid_image(path: str, min_size: Tuple[int, int] = (256, 256)) -> bool:
    """
    Check whether a file is a valid, readable image meeting minimum size requirements.

    Args:
        path: Path to the image file.
        min_size: Minimum (height, width) in pixels.

    Returns:
        True if valid and large enough, False otherwise.
    """
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        return h >= min_size[0] and w >= min_size[1]
    except Exception:
        return False


def create_comparison_grid(
    lr: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray
) -> np.ndarray:
    """
    Create a side-by-side comparison image: LR | SR | HR with text labels.

    All images are resized to match HR dimensions before concatenation.

    Args:
        lr: Low-resolution image [H, W, C] BGR uint8.
        sr: Super-resolved image [H, W, C] BGR uint8.
        hr: High-resolution ground truth [H, W, C] BGR uint8.

    Returns:
        Concatenated comparison image [H, W*3, C] BGR uint8.
    """
    target_h, target_w = hr.shape[:2]

    lr_resized = cv2.resize(lr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    sr_resized = cv2.resize(sr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)
    shadow = (0, 0, 0)

    for img, label in [(lr_resized, "LR"), (sr_resized, "SR"), (hr, "HR")]:
        pos = (10, 35)
        cv2.putText(img, label, pos, font, font_scale, shadow, thickness + 2)
        cv2.putText(img, label, pos, font, font_scale, color, thickness)

    grid = np.concatenate([lr_resized, sr_resized, hr], axis=1)
    return grid