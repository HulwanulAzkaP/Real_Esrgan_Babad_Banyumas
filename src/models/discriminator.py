
"""
U-Net Discriminator with Spectral Normalization for Real-ESRGAN.
Produces per-pixel probability maps instead of a single scalar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import List


def make_conv(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
              use_sn: bool = True) -> nn.Conv2d:
    """Create a Conv2d layer optionally wrapped with spectral norm."""
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2)
    if use_sn:
        return spectral_norm(conv)
    return conv


class UNetDiscriminator(nn.Module):
    """
    U-Net Discriminator with Spectral Normalization for Real-ESRGAN.
    Generates a per-pixel probability map [B, 1, H, W].

    Encoder: 3 -> 64 -> 128 -> 256 -> 512 -> 512 (stride=2 each)
    Middle:  512 -> 512
    Decoder: upsample + skip concat -> conv back up
    Output:  Conv2d(64 -> 1)
    """

    def __init__(self, in_channels: int = 3, num_features: int = 64,
                 num_blocks: int = 5, spectral_norm_enabled: bool = True):
        super().__init__()
        sn = spectral_norm_enabled

        nf = num_features  # 64

        # ---- ENCODER ----
        # Each block: SN-Conv(stride=2) + LeakyReLU
        self.enc1 = nn.Sequential(
            make_conv(in_channels, nf, stride=2, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf
        self.enc2 = nn.Sequential(
            make_conv(nf, nf * 2, stride=2, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf*2
        self.enc3 = nn.Sequential(
            make_conv(nf * 2, nf * 4, stride=2, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf*4
        self.enc4 = nn.Sequential(
            make_conv(nf * 4, nf * 8, stride=2, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf*8
        self.enc5 = nn.Sequential(
            make_conv(nf * 8, nf * 8, stride=2, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf*8

        # ---- MIDDLE ----
        self.middle = nn.Sequential(
            make_conv(nf * 8, nf * 8, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf*8  (this is d5 / bottleneck)

        # ---- DECODER ----
        # After upsample, we concat with encoder skip, so in_ch = dec_ch + enc_ch
        # dec5 input: middle(nf*8) upsample -> concat with enc4(nf*8) = nf*8 + nf*8 = nf*16
        self.dec5 = nn.Sequential(
            make_conv(nf * 8 + nf * 8, nf * 8, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf*8

        # dec4 input: dec5(nf*8) upsample -> concat with enc3(nf*4) = nf*8 + nf*4 = nf*12
        self.dec4 = nn.Sequential(
            make_conv(nf * 8 + nf * 4, nf * 4, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf*4

        # dec3 input: dec4(nf*4) upsample -> concat with enc2(nf*2) = nf*4 + nf*2 = nf*6
        self.dec3 = nn.Sequential(
            make_conv(nf * 4 + nf * 2, nf * 2, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf*2

        # dec2 input: dec3(nf*2) upsample -> concat with enc1(nf) = nf*2 + nf = nf*3
        self.dec2 = nn.Sequential(
            make_conv(nf * 2 + nf, nf, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf

        # dec1 input: dec2(nf) upsample (no skip here, enc0 is the raw input)
        self.dec1 = nn.Sequential(
            make_conv(nf, nf, use_sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> nf

        # ---- OUTPUT ----
        self.out_conv = nn.Conv2d(nf, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, 1, H, W] — per-pixel probability map (no sigmoid, handled in loss)
        """
        # Encoder
        e1 = self.enc1(x)    # H/2
        e2 = self.enc2(e1)   # H/4
        e3 = self.enc3(e2)   # H/8
        e4 = self.enc4(e3)   # H/16
        e5 = self.enc5(e4)   # H/32

        # Middle (bottleneck)
        m = self.middle(e5)  # H/32

        # Decoder with skip connections
        d5 = self.dec5(torch.cat([
            F.interpolate(m, size=e4.shape[2:], mode="bilinear", align_corners=False),
            e4
        ], dim=1))  # H/16

        d4 = self.dec4(torch.cat([
            F.interpolate(d5, size=e3.shape[2:], mode="bilinear", align_corners=False),
            e3
        ], dim=1))  # H/8

        d3 = self.dec3(torch.cat([
            F.interpolate(d4, size=e2.shape[2:], mode="bilinear", align_corners=False),
            e2
        ], dim=1))  # H/4

        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e1.shape[2:], mode="bilinear", align_corners=False),
            e1
        ], dim=1))  # H/2

        d1 = self.dec1(
            F.interpolate(d2, size=x.shape[2:], mode="bilinear", align_corners=False)
        )  # H (original size, no skip)

        return self.out_conv(d1)