import os
import time
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

from src.utils.image_utils import is_valid_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ManuscriptScraper:
    """
    Scraper gambar naskah kuno dari berbagai sumber internet.
    Mendukung Google Images dan Bing Images via icrawler.
    """

    def __init__(self, config: dict) -> None:
        """Inisialisasi dari konfigurasi scraping."""
        self.config = config
        self.scraping_cfg = config["scraping"]
        self.output_dir = Path(self.scraping_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_size = tuple(self.scraping_cfg["min_image_size"])
        self.delay = self.scraping_cfg["delay_between_requests"]

    def scrape_google_images(self, keyword: str, max_num: int, save_dir: str) -> List[str]:
        """
        Scrape Google Images menggunakan icrawler.
        Return: list path file yang berhasil didownload.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        try:
            crawler = GoogleImageCrawler(
                storage={"root_dir": save_dir},
                log_level=40,
                num_t=4,
            )
            crawler.crawl(keyword=keyword, max_num=max_num)
            time.sleep(self.delay)
        except Exception as e:
            logger.error(f"Google scrape error for '{keyword}': {e}")

        files = [str(p) for p in Path(save_dir).glob("*") if p.is_file()]
        return files

    def scrape_bing_images(self, keyword: str, max_num: int, save_dir: str) -> List[str]:
        """
        Scrape Bing Images menggunakan icrawler.BingImageCrawler.
        Return: list path file yang berhasil didownload.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        try:
            crawler = BingImageCrawler(
                storage={"root_dir": save_dir},
                log_level=40,
                num_t=4,
            )
            crawler.crawl(keyword=keyword, max_num=max_num)
            time.sleep(self.delay)
        except Exception as e:
            logger.error(f"Bing scrape error for '{keyword}': {e}")

        files = [str(p) for p in Path(save_dir).glob("*") if p.is_file()]
        return files

    def validate_and_filter(self, directory: str) -> List[str]:
        """
        Scan direktori, validasi semua gambar menggunakan is_valid_image().
        Hapus file yang tidak valid.
        Return: list path gambar yang valid.
        """
        valid_paths: List[str] = []
        for p in Path(directory).glob("*"):
            if not p.is_file():
                continue
            if is_valid_image(str(p), self.min_size):
                valid_paths.append(str(p))
            else:
                try:
                    p.unlink()
                    logger.debug(f"Removed invalid image: {p}")
                except Exception as e:
                    logger.warning(f"Could not remove {p}: {e}")
        return valid_paths

    def build_manifest(self, all_paths: List[str]) -> pd.DataFrame:
        """
        Buat manifest DataFrame dan simpan ke data/raw/manifest.csv.
        """
        import cv2
        records = []
        for path in all_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                size_kb = os.path.getsize(path) / 1024
                fmt = Path(path).suffix.lower().strip(".")
                source = Path(path).parent.name
                records.append({
                    "filepath": path,
                    "source": source,
                    "filename": Path(path).name,
                    "width": w,
                    "height": h,
                    "file_size_kb": round(size_kb, 2),
                    "format": fmt,
                })
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")

        df = pd.DataFrame(records)
        manifest_path = self.output_dir / "manifest.csv"
        df.to_csv(manifest_path, index=False)
        logger.info(f"Manifest saved to {manifest_path}")
        return df

    def run(self) -> pd.DataFrame:
        """
        Jalankan seluruh proses scraping.
        """
        all_valid_paths: List[str] = []

        for source in self.scraping_cfg["sources"]:
            source_name = source["name"]
            keywords = source["keywords"]
            max_per_kw = source["max_per_keyword"]

            logger.info(f"Scraping source: {source_name}")
            for keyword in tqdm(keywords, desc=f"{source_name}"):
                keyword_slug = keyword.replace(" ", "_")[:40]
                save_dir = str(self.output_dir / source_name / keyword_slug)

                try:
                    if source_name == "google_images":
                        self.scrape_google_images(keyword, max_per_kw, save_dir)
                    elif source_name == "bing_images":
                        self.scrape_bing_images(keyword, max_per_kw, save_dir)
                    else:
                        logger.warning(f"Unknown source: {source_name}")
                        continue

                    valid = self.validate_and_filter(save_dir)
                    all_valid_paths.extend(valid)
                    logger.info(f"  '{keyword}': {len(valid)} valid images")

                except Exception as e:
                    logger.error(f"Failed scraping '{keyword}' from {source_name}: {e}")

        df = self.build_manifest(all_valid_paths)
        return df