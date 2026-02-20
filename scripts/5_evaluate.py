"""
Comprehensive evaluation of the trained Real-ESRGAN model on the test set.
Computes PSNR, SSIM, LPIPS vs bicubic baseline and saves visual comparisons.
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import yaml
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import BabadBanyumasDataset
from src.models.generator import RRDBNet
from src.utils.image_utils import create_comparison_grid, save_image, tensor_to_numpy
from src.utils.logger import get_logger
from src.utils.metrics import LPIPSMetric, calculate_psnr, calculate_ssim

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Real-ESRGAN on test set")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val"],
        help="Dataset split to evaluate on"
    )
    return parser.parse_args()


def upscale_bicubic(lr: np.ndarray, scale: int) -> np.ndarray:
    """Upscale LR image using bicubic interpolation as baseline."""
    h, w = lr.shape[:2]
    return cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device_str = config["project"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    eval_cfg = config["evaluation"]
    results_dir = Path(eval_cfg["results_dir"])
    visuals_dir = results_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = config["model"]["generator"]
    generator = RRDBNet(**gen_cfg).to(device)

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    splits_dir = Path(config["dataset"]["splits_dir"])
    split_file = splits_dir / f"{args.split}.txt"
    dataset = BabadBanyumasDataset(str(split_file), config, mode=args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    lr_scale = config["dataset"]["lr_scale"]
    lpips_metric = LPIPSMetric(network=eval_cfg["lpips_network"])

    records = []
    num_visual = eval_cfg["num_visual_samples"]
    visual_indices = set(
        random.sample(range(len(dataset)), min(num_visual, len(dataset)))
    )

    psnr_sr_list: List[float] = []
    psnr_base_list: List[float] = []
    ssim_sr_list: List[float] = []
    ssim_base_list: List[float] = []
    lpips_sr_list: List[float] = []
    lpips_base_list: List[float] = []

    logger.info(f"Evaluating {len(dataset)} samples from {args.split} split...")

    for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
        lr_tensor = batch["lr"].to(device)
        hr_tensor = batch["hr"].to(device)
        lr_path = batch["lr_path"][0]
        hr_path = batch["hr_path"][0]

        with torch.no_grad():
            sr_tensor = generator(lr_tensor)

        lr_np = tensor_to_numpy(lr_tensor[0])
        sr_np = tensor_to_numpy(sr_tensor[0])
        hr_np = tensor_to_numpy(hr_tensor[0])
        baseline_np = upscale_bicubic(lr_np, lr_scale)

        # Resize baseline to match HR if needed
        if baseline_np.shape != hr_np.shape:
            baseline_np = cv2.resize(baseline_np, (hr_np.shape[1], hr_np.shape[0]))
        if sr_np.shape != hr_np.shape:
            sr_np = cv2.resize(sr_np, (hr_np.shape[1], hr_np.shape[0]))

        psnr_sr = calculate_psnr(sr_np, hr_np)
        psnr_base = calculate_psnr(baseline_np, hr_np)
        ssim_sr = calculate_ssim(sr_np, hr_np)
        ssim_base = calculate_ssim(baseline_np, hr_np)
        lpips_sr = lpips_metric.calculate(sr_np, hr_np)
        lpips_base = lpips_metric.calculate(baseline_np, hr_np)

        psnr_sr_list.append(psnr_sr)
        psnr_base_list.append(psnr_base)
        ssim_sr_list.append(ssim_sr)
        ssim_base_list.append(ssim_base)
        lpips_sr_list.append(lpips_sr)
        lpips_base_list.append(lpips_base)

        img_name = Path(hr_path).stem
        records.append({
            "image_name": img_name,
            "psnr_baseline": round(psnr_base, 4),
            "psnr_sr": round(psnr_sr, 4),
            "ssim_baseline": round(ssim_base, 4),
            "ssim_sr": round(ssim_sr, 4),
            "lpips_baseline": round(lpips_base, 4),
            "lpips_sr": round(lpips_sr, 4),
        })

        if i in visual_indices:
            grid = create_comparison_grid(lr_np, sr_np, hr_np)
            label = f"PSNR: {psnr_sr:.2f}dB  SSIM: {ssim_sr:.4f}"
            cv2.putText(grid, label, (10, grid.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(grid, label, (10, grid.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            save_image(grid, str(visuals_dir / f"compare_{img_name}.png"))

    csv_path = results_dir / "evaluation_results.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    def run_ttest(a: List[float], b: List[float], name: str) -> None:
        t_stat, p_val = stats.ttest_rel(a, b)
        sig = "Significant improvement (p<0.05)" if p_val < 0.05 else "Not significant"
        print(f"  {name}: t={t_stat:.3f}, p={p_val:.4f} — {sig}")

    n = len(records)
    mean_psnr_base = np.mean(psnr_base_list)
    mean_psnr_sr = np.mean(psnr_sr_list)
    mean_ssim_base = np.mean(ssim_base_list)
    mean_ssim_sr = np.mean(ssim_sr_list)
    mean_lpips_base = np.mean(lpips_base_list)
    mean_lpips_sr = np.mean(lpips_sr_list)

    print(f"\n=====================================================")
    print(f"HASIL EVALUASI — {args.split.upper()} SET (N={n} gambar)")
    print(f"=====================================================")
    print(f"{'Metrik':<14} | {'Bicubic Baseline':^16} | {'Real-ESRGAN Ours':^16}")
    print(f"{'PSNR (dB) ↑':<14} | {mean_psnr_base:^16.2f} | {mean_psnr_sr:^16.2f}")
    print(f"{'SSIM ↑':<14} | {mean_ssim_base:^16.4f} | {mean_ssim_sr:^16.4f}")
    print(f"{'LPIPS ↓':<14} | {mean_lpips_base:^16.4f} | {mean_lpips_sr:^16.4f}")
    print(f"=====================================================")
    print(f"Improvement:")
    print(f"PSNR   : {mean_psnr_sr - mean_psnr_base:+.2f} dB")
    print(f"SSIM   : {mean_ssim_sr - mean_ssim_base:+.4f}")
    print(f"LPIPS  : {mean_lpips_sr - mean_lpips_base:+.4f} (lower is better)")
    print(f"=====================================================")
    print(f"\nPaired t-test (SR vs Bicubic):")
    run_ttest(psnr_sr_list, psnr_base_list, "PSNR")
    run_ttest(ssim_sr_list, ssim_base_list, "SSIM")
    run_ttest(
        [-x for x in lpips_sr_list],
        [-x for x in lpips_base_list],
        "LPIPS"
    )
    print(f"\nResults saved to: {csv_path}")
    print(f"Visuals saved to: {visuals_dir}\n")


if __name__ == "__main__":
    main()