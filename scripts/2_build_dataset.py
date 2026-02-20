"""
Build paired LR-HR dataset from scraped raw images.
Applies the full degradation pipeline to generate LR counterparts.
"""

import argparse
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.degradation import DegradationPipeline
from src.utils.image_utils import is_valid_image, load_image, save_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LR-HR dataset pairs")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pairs that already exist in output dirs",
    )
    return parser.parse_args()


def collect_source_images(source_dir: str, min_size: tuple) -> list:
    """Collect all valid images recursively from source directory."""
    source_path = Path(source_dir)
    allowed = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    paths = []
    for p in source_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed:
            if is_valid_image(str(p), min_size):
                paths.append(str(p))
    return paths


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    random.seed(config["project"]["seed"])
    np.random.seed(config["project"]["seed"])

    dataset_cfg = config["dataset"]
    hr_size = dataset_cfg["hr_size"]
    total_target = dataset_cfg["total_target"]
    hr_dir = Path(dataset_cfg["hr_dir"])
    lr_dir = Path(dataset_cfg["lr_dir"])
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    source_dir = args.source_dir or config["scraping"]["output_dir"]
    min_size = tuple(config["scraping"]["min_image_size"])

    logger.info(f"Collecting source images from: {source_dir}")
    source_images = collect_source_images(source_dir, min_size)
    logger.info(f"Found {len(source_images)} valid source images")

    if not source_images:
        logger.error("No source images found. Run 1_scrape_images.py first.")
        return

    degradation = DegradationPipeline(config)
    pairs_created = 0
    source_images_used = set()

    existing_count = len(list(hr_dir.glob("*.png")))
    if args.resume:
        pairs_created = existing_count
        logger.info(f"Resuming from {existing_count} existing pairs")

    random.shuffle(source_images)

    progress = tqdm(total=total_target, initial=pairs_created, desc="Building pairs")

    for img_path in source_images:
        if pairs_created >= total_target:
            break

        try:
            image = load_image(img_path)
            h, w = image.shape[:2]

            if h < hr_size or w < hr_size:
                continue

            max_crops = max(1, min(5, (h * w) // (hr_size * hr_size)))

            for _ in range(max_crops):
                if pairs_created >= total_target:
                    break

                top = random.randint(0, h - hr_size)
                left = random.randint(0, w - hr_size)
                hr_crop = image[top:top + hr_size, left:left + hr_size]

                lr_image, hr_image = degradation.apply_full_pipeline(hr_crop)

                pair_id = str(uuid.uuid4())
                hr_save_path = hr_dir / f"{pair_id}.png"
                lr_save_path = lr_dir / f"{pair_id}.png"

                if args.resume and hr_save_path.exists():
                    continue

                save_image(hr_image, str(hr_save_path))
                save_image(lr_image, str(lr_save_path))

                source_images_used.add(img_path)
                pairs_created += 1
                progress.update(1)

        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            continue

    progress.close()

    metadata = {
        "total_pairs": pairs_created,
        "hr_size": hr_size,
        "lr_size": dataset_cfg["lr_size"],
        "scale_factor": dataset_cfg["lr_scale"],
        "created_at": datetime.now().isoformat(),
        "source_images_used": len(source_images_used),
    }

    meta_path = Path("data/processed/dataset_info.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n============================================")
    print("DATASET BUILD SELESAI")
    print("============================================")
    print(f"Total pasangan dibuat  : {pairs_created}")
    print(f"Source images dipakai  : {len(source_images_used)}")
    print(f"HR directory           : {hr_dir}")
    print(f"LR directory           : {lr_dir}")
    print(f"Metadata tersimpan     : {meta_path}")
    print("============================================\n")


if __name__ == "__main__":
    main()