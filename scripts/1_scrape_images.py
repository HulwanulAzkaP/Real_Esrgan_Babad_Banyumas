"""
Entry point for scraping manuscript images from Google and Bing.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.scraper import ManuscriptScraper
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape manuscript images")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be scraped without downloading",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])

    if args.dry_run:
        print("\n[DRY RUN] Would scrape the following:")
        for source in config["scraping"]["sources"]:
            print(f"\n  Source: {source['name']}")
            for kw in source["keywords"]:
                print(f"    - '{kw}' (max {source['max_per_keyword']} images)")
        print("\nNo files downloaded (--dry-run mode).\n")
        return

    scraper = ManuscriptScraper(config)
    manifest = scraper.run()

    source_counts = {}
    for source in config["scraping"]["sources"]:
        name = source["name"]
        count = len(manifest[manifest["source"].str.contains(name, na=False)]) if len(manifest) > 0 else 0
        source_counts[name] = count

    print("\n============================================")
    print("SCRAPING SELESAI")
    print("============================================")
    print(f"Total gambar valid  : {len(manifest)}")
    for source_name, count in source_counts.items():
        print(f"- {source_name:<20}: {count}")
    manifest_path = Path(config["scraping"]["output_dir"]) / "manifest.csv"
    print(f"Manifest tersimpan  : {manifest_path}")
    print("============================================\n")


if __name__ == "__main__":
    main()