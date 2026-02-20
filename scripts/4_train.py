"""
Main training script for Real-ESRGAN on the Babad Banyumas manuscript dataset.
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Tuple

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_dataloaders
from src.models.discriminator import UNetDiscriminator
from src.models.generator import RRDBNet
from src.models.losses import RelativisticGANLoss, TotalGeneratorLoss
from src.utils.image_utils import tensor_to_numpy
from src.utils.logger import get_logger
from src.utils.metrics import calculate_psnr, calculate_ssim

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Real-ESRGAN")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    tag: str,
    checkpoint_dir: str,
    best_psnr: float = 0.0,
) -> None:
    """
    Save model and optimizer states to a checkpoint file.

    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        optimizer_g: Generator optimizer.
        optimizer_d: Discriminator optimizer.
        epoch: Current epoch number.
        tag: Checkpoint tag string (e.g. 'best', 'epoch_0100').
        checkpoint_dir: Directory to save checkpoints.
        best_psnr: Best validation PSNR so far.
    """
    ckpt_path = Path(checkpoint_dir) / f"{tag}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "best_psnr": best_psnr,
        },
        str(ckpt_path),
    )
    logger.info(f"Checkpoint saved: {ckpt_path}")


def load_checkpoint(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Load model and optimizer states from a checkpoint.

    Returns:
        Tuple (start_epoch, best_psnr).
    """
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_psnr = ckpt.get("best_psnr", 0.0)
    logger.info(f"Resumed from checkpoint: {path} (epoch {start_epoch})")
    return start_epoch, best_psnr


def run_validation(
    generator: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run validation pass and compute mean PSNR and SSIM.

    Args:
        generator: Generator model in eval mode.
        val_loader: Validation DataLoader.
        device: Compute device.

    Returns:
        Tuple (mean_psnr, mean_ssim).
    """
    generator.eval()
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            sr = generator(lr)

            sr_np = tensor_to_numpy(sr[0])
            hr_np = tensor_to_numpy(hr[0])

            psnr_values.append(calculate_psnr(sr_np, hr_np))
            ssim_values.append(calculate_ssim(sr_np, hr_np))

    generator.train()
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seeds(config["project"]["seed"])

    device_str = config["project"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    train_cfg = config["training"]
    checkpoint_dir = train_cfg["checkpoint_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    gen_cfg = config["model"]["generator"]
    disc_cfg = config["model"]["discriminator"]

    generator = RRDBNet(**gen_cfg).to(device)
    discriminator = UNetDiscriminator(**disc_cfg).to(device)

    logger.info(f"Generator     : {count_params(generator):,} parameters")
    logger.info(f"Discriminator : {count_params(discriminator):,} parameters")

    total_loss_fn = TotalGeneratorLoss(
        weights=train_cfg["loss_weights"], device=device_str
    )
    gan_loss_fn = RelativisticGANLoss()

    opt_cfg = train_cfg["optimizer"]
    betas = tuple(opt_cfg["betas"])

    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt_cfg["lr_g"], betas=betas
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt_cfg["lr_d"], betas=betas
    )

    sched_cfg = train_cfg["scheduler"]
    scheduler_G = MultiStepLR(optimizer_G, milestones=sched_cfg["milestones"], gamma=sched_cfg["gamma"])
    scheduler_D = MultiStepLR(optimizer_D, milestones=sched_cfg["milestones"], gamma=sched_cfg["gamma"])

    start_epoch = 0
    best_psnr = 0.0

    resume_path = args.resume or train_cfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        start_epoch, best_psnr = load_checkpoint(
            generator, discriminator, optimizer_G, optimizer_D, resume_path, device
        )

    train_loader, val_loader, _ = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    total_epochs = train_cfg["epochs"]
    save_every = train_cfg["save_every"]
    validate_every = train_cfg["validate_every"]

    training_log = []

    for epoch in range(start_epoch, total_epochs):
        generator.train()
        discriminator.train()

        epoch_losses = {"G_total": [], "G_l1": [], "G_perceptual": [], "G_gan": [], "D": []}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")

        for batch in progress_bar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            # === UPDATE DISCRIMINATOR ===
            optimizer_D.zero_grad()
            with autocast(enabled=use_amp):
                sr = generator(lr).detach()
                real_pred = discriminator(hr)
                fake_pred = discriminator(sr)
                loss_D = gan_loss_fn.discriminator_loss(real_pred, fake_pred)

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            # === UPDATE GENERATOR ===
            optimizer_G.zero_grad()
            with autocast(enabled=use_amp):
                sr = generator(lr)
                real_pred = discriminator(hr).detach()
                fake_pred = discriminator(sr)
                losses = total_loss_fn(sr, hr, real_pred, fake_pred)

            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer_G)
            scaler.update()

            epoch_losses["G_total"].append(losses["total"].item())
            epoch_losses["G_l1"].append(losses["l1"].item())
            epoch_losses["G_perceptual"].append(losses["perceptual"].item())
            epoch_losses["G_gan"].append(losses["gan"].item())
            epoch_losses["D"].append(loss_D.item())

            progress_bar.set_postfix({
                "G": f"{losses['total'].item():.4f}",
                "D": f"{loss_D.item():.4f}",
            })

        avg_g = mean(epoch_losses["G_total"])
        avg_d = mean(epoch_losses["D"])
        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f}"
        )

        log_entry = {
            "epoch": epoch + 1,
            "G_total": avg_g,
            "G_l1": mean(epoch_losses["G_l1"]),
            "G_perceptual": mean(epoch_losses["G_perceptual"]),
            "G_gan": mean(epoch_losses["G_gan"]),
            "D": avg_d,
        }

        if (epoch + 1) % validate_every == 0:
            val_psnr, val_ssim = run_validation(generator, val_loader, device)
            logger.info(f"  Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

            log_entry["val_psnr"] = val_psnr
            log_entry["val_ssim"] = val_ssim

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    generator, discriminator, optimizer_G, optimizer_D,
                    epoch + 1, "best", checkpoint_dir, best_psnr
                )

        training_log.append(log_entry)

        if (epoch + 1) % save_every == 0:
            tag = f"epoch_{(epoch + 1):04d}"
            save_checkpoint(
                generator, discriminator, optimizer_G, optimizer_D,
                epoch + 1, tag, checkpoint_dir, best_psnr
            )

        scheduler_G.step()
        scheduler_D.step()

    log_csv_path = Path("results") / "training_log.csv"
    if training_log:
        keys = training_log[0].keys()
        with open(str(log_csv_path), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(training_log)
    logger.info(f"Training log saved to {log_csv_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()