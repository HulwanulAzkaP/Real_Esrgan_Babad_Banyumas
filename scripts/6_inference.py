"""
Inference script for restoring manuscript images using a trained Real-ESRGAN model.
Supports single image, batch directory, and tile-based processing for large images.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.generator import RRDBNet
from src.utils.image_utils import load_image, save_image, tensor_to_numpy, numpy_to_tensor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-ESRGAN inference")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="results/restored/")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--compare", action="store_true", help="Also save LR|SR comparison")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def tile_inference(
    generator: torch.nn.Module,
    image: np.ndarray,
    tile_size: int,
    scale: int,
    device: torch.device,
    overlap: int = 10,
) -> np.ndarray:
    """
    Process a large image in overlapping tiles and blend the results.

    Args:
        generator: Trained generator model in eval mode.
        image: Input LR image [H, W, C] BGR uint8.
        tile_size: Size of each tile in pixels.
        scale: Upscale factor.
        device: Compute device.
        overlap: Overlap in pixels between adjacent tiles.

    Returns:
        SR image [H*scale, W*scale, C] BGR uint8.
    """
    h, w = image.shape[:2]
    sr_h = h * scale
    sr_w = w * scale

    output = np.zeros((sr_h, sr_w, 3), dtype=np.float32)
    weight_map = np.zeros((sr_h, sr_w, 1), dtype=np.float32)

    step = tile_size - overlap

    y_starts = list(range(0, h - tile_size + 1, step))
    if not y_starts or y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))

    x_starts = list(range(0, w - tile_size + 1, step))
    if not x_starts or x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))

    for y in y_starts:
        for x in x_starts:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]

            tile_t = numpy_to_tensor(tile).unsqueeze(0).to(device)
            with torch.no_grad():
                sr_tile_t = generator(tile_t)
            sr_tile = tensor_to_numpy(sr_tile_t[0])

            sy = y * scale
            sx = x * scale
            ey = y_end * scale
            ex = x_end * scale

            output[sy:ey, sx:ex] += sr_tile[:ey - sy, :ex - sx].astype(np.float32)
            weight_map[sy:ey, sx:ex] += 1.0

    weight_map = np.maximum(weight_map, 1.0)
    output = output / weight_map
    return np.clip(output, 0, 255).astype(np.uint8)


def process_image(
    generator: torch.nn.Module,
    image: np.ndarray,
    tile_size: int,
    scale: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run super-resolution on a single image, using tiling for large inputs.

    Args:
        generator: Trained generator model in eval mode.
        image: Input LR image [H, W, C] BGR uint8.
        tile_size: Tile size threshold.
        scale: Upscale factor.
        device: Compute device.

    Returns:
        SR image [H*scale, W*scale, C] BGR uint8.
    """
    h, w = image.shape[:2]
    use_tiles = (h > 1000 or w > 1000)

    if use_tiles:
        return tile_inference(generator, image, tile_size, scale, device)
    else:
        img_t = numpy_to_tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_t = generator(img_t)
        return tensor_to_numpy(sr_t[0])


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device_str = config["project"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    gen_cfg = config["model"]["generator"]
    generator = RRDBNet(**gen_cfg).to(device)

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()
    logger.info(f"Model loaded from: {ckpt_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    allowed = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = [p for p in input_path.rglob("*") if p.suffix.lower() in allowed]
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return

    logger.info(f"Processing {len(image_paths)} image(s)...")

    for i, img_path in enumerate(image_paths):
        try:
            image = load_image(str(img_path))
            h_in, w_in = image.shape[:2]
            use_tiles = (h_in > 1000 or w_in > 1000)
            mode_str = "Tiled" if use_tiles else "Direct (no tiling)"

            t0 = time.time()
            sr_image = process_image(
                generator, image, args.tile_size, args.scale, device
            )
            elapsed = time.time() - t0

            h_out, w_out = sr_image.shape[:2]
            out_name = f"{img_path.stem}_restored.png"
            out_path = output_dir / out_name
            save_image(sr_image, str(out_path))

            if args.compare:
                lr_resized = cv2.resize(image, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
                comparison = np.concatenate([lr_resized, sr_image], axis=1)
                cmp_name = f"{img_path.stem}_comparison.png"
                save_image(comparison, str(output_dir / cmp_name))

            print(f"\n[{i+1}/{len(image_paths)}] {img_path.name}")
            print(f"    Input  : {w_in} × {h_in} px")
            print(f"    Output : {w_out} × {h_out} px")
            print(f"    Mode   : {mode_str}")
            print(f"    Waktu  : {elapsed:.2f} detik")
            print(f"    Saved  : {out_path}")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue

    print(f"\nInference selesai. Output disimpan di: {output_dir}\n")


if __name__ == "__main__":
    main()