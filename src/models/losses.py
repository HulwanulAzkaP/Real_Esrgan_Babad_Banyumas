import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List


class L1Loss(nn.Module):
    """Pixel-wise L1 loss antara SR dan HR."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(sr, hr)


class VGGFeatureExtractor(nn.Module):
    """Helper module to extract intermediate VGG19 features."""

    def __init__(self, layer_indices: List[int]) -> None:
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg_features: nn.Sequential = vgg19.features  # type: ignore[assignment]
        self.blocks = nn.ModuleList()
        prev_idx = 0
        for idx in layer_indices:
            block = nn.Sequential(*list(vgg_features.children())[prev_idx:idx + 1])
            self.blocks.append(block)
            prev_idx = idx + 1
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss menggunakan VGG19 pre-trained features.
    Ekstrak fitur dari relu3_4, relu4_4, relu5_4.
    """

    # VGG19 layer indices for relu3_4=18, relu4_4=27, relu5_4=36
    LAYER_MAP: Dict[str, int] = {
        "relu3_4": 18,
        "relu4_4": 27,
        "relu5_4": 36,
    }

    def __init__(
        self,
        layers: List[str] = ["relu3_4", "relu4_4", "relu5_4"],
        device: str = "cpu",
    ) -> None:
        super().__init__()
        layer_indices = [self.LAYER_MAP[l] for l in layers]
        self.extractor = VGGFeatureExtractor(layer_indices).to(device)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = self.mean  # type: ignore[assignment]
        std: torch.Tensor = self.std    # type: ignore[assignment]
        return (x - mean) / std

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_norm = self._normalize(sr)
        hr_norm = self._normalize(hr)

        sr_features = self.extractor(sr_norm)
        hr_features = self.extractor(hr_norm)

        loss = torch.tensor(0.0, device=sr.device)
        for sf, hf in zip(sr_features, hr_features):
            loss = loss + F.mse_loss(sf, hf.detach())
        loss = loss / len(sr_features)
        return loss


class RelativisticGANLoss(nn.Module):
    """
    Relativistic Average GAN Loss (RaGAN).
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def generator_loss(
        self, real_pred: torch.Tensor, fake_pred: torch.Tensor
    ) -> torch.Tensor:
        real_mean = real_pred.mean(dim=0, keepdim=True).detach()
        fake_mean = fake_pred.mean(dim=0, keepdim=True).detach()
        ones = torch.ones_like(fake_pred)
        zeros = torch.zeros_like(real_pred)
        loss = self.bce(fake_pred - real_mean, ones) + self.bce(real_pred - fake_mean, zeros)
        return loss

    def discriminator_loss(
        self, real_pred: torch.Tensor, fake_pred: torch.Tensor
    ) -> torch.Tensor:
        real_mean = real_pred.mean(dim=0, keepdim=True).detach()
        fake_mean = fake_pred.mean(dim=0, keepdim=True).detach()
        ones = torch.ones_like(real_pred)
        zeros = torch.zeros_like(fake_pred)
        loss = self.bce(real_pred - fake_mean, ones) + self.bce(fake_pred - real_mean, zeros)
        return loss


class TotalGeneratorLoss(nn.Module):
    """
    Kombinasi loss untuk generator:
    L_total = w_l1 * L1 + w_perceptual * L_perceptual + w_gan * L_GAN
    """

    def __init__(self, weights: dict, device: str = "cpu") -> None:
        super().__init__()
        self.w_l1 = weights.get("l1", 1.0)
        self.w_perceptual = weights.get("perceptual", 1.0)
        self.w_gan = weights.get("gan", 0.1)

        self.l1_loss = L1Loss()
        self.perceptual_loss = PerceptualLoss(device=device)
        self.gan_loss = RelativisticGANLoss()

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l1 = self.l1_loss(sr, hr)
        perceptual = self.perceptual_loss(sr, hr)
        gan = self.gan_loss.generator_loss(real_pred, fake_pred)

        total = self.w_l1 * l1 + self.w_perceptual * perceptual + self.w_gan * gan

        return {
            "total": total,
            "l1": l1,
            "perceptual": perceptual,
            "gan": gan,
        }