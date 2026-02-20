"""
RRDB-based generator network for Real-ESRGAN.
No BatchNorm or GroupNorm is used anywhere in this model.
"""

import torch
import torch.nn as nn
from typing import List


class DenseLayer(nn.Module):
    """
    A single convolutional layer inside a Residual Dense Block.
    Uses LeakyReLU(0.2) activation. No normalization.
    """

    def __init__(self, in_channels: int, growth_channels: int = 32):
        """
        Args:
            in_channels: Number of input feature channels.
            growth_channels: Number of output channels per dense layer.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_channels, 3, 1, 1, bias=True)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class DenseBlock(nn.Module):
    """
    Residual Dense Block with 5 densely connected layers.
    Each layer receives the concatenation of all previous outputs.
    Output is residual-scaled before adding to the block input.
    """

    def __init__(
        self,
        num_features: int = 64,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
    ):
        """
        Args:
            num_features: Number of input/output channels.
            growth_channels: Channels added per dense layer.
            residual_scaling: Scaling factor for the residual connection.
        """
        super().__init__()
        self.residual_scaling = residual_scaling

        self.layer1 = DenseLayer(num_features, growth_channels)
        self.layer2 = DenseLayer(num_features + growth_channels, growth_channels)
        self.layer3 = DenseLayer(num_features + 2 * growth_channels, growth_channels)
        self.layer4 = DenseLayer(num_features + 3 * growth_channels, growth_channels)
        self.layer5 = DenseLayer(num_features + 4 * growth_channels, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(torch.cat([x, x1], dim=1))
        x3 = self.layer3(torch.cat([x, x1, x2], dim=1))
        x4 = self.layer4(torch.cat([x, x1, x2, x3], dim=1))
        x5 = self.layer5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + self.residual_scaling * x5


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block: three DenseBlocks with a residual skip.
    """

    def __init__(
        self,
        num_features: int = 64,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
    ):
        """
        Args:
            num_features: Number of feature channels throughout.
            growth_channels: Growth channels per dense layer.
            residual_scaling: Residual scaling for both dense block and RRDB level.
        """
        super().__init__()
        self.residual_scaling = residual_scaling
        self.block1 = DenseBlock(num_features, growth_channels, residual_scaling)
        self.block2 = DenseBlock(num_features, growth_channels, residual_scaling)
        self.block3 = DenseBlock(num_features, growth_channels, residual_scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block3(self.block2(self.block1(x)))
        return x + self.residual_scaling * out


class RRDBNet(nn.Module):
    """
    Full RRDB Network used as the Real-ESRGAN generator.

    Architecture:
        Input -> Conv3x3 -> N x RRDB -> Conv3x3 -> [Upsample x2 -> Upsample x2]
               -> Conv3x3 -> LeakyReLU -> Conv3x3 -> Output

    Upsampling uses PixelShuffle (sub-pixel convolution) for 4x total upscale.
    No BatchNorm anywhere. LeakyReLU(0.2) throughout.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_rrdb: int = 23,
        upscale_factor: int = 4,
        residual_scaling: float = 0.2,
    ):
        """
        Args:
            in_channels: Number of input image channels (3 for RGB).
            out_channels: Number of output image channels.
            num_features: Base number of feature channels.
            num_rrdb: Number of RRDB blocks to stack.
            upscale_factor: Total spatial upscale factor (must be 4).
            residual_scaling: Residual scaling applied inside RRDB blocks.
        """
        super().__init__()
        self.upscale_factor = upscale_factor

        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1, bias=True)

        # RRDB trunk
        rrdb_blocks: List[nn.Module] = [
            RRDB(num_features, growth_channels=32, residual_scaling=residual_scaling)
            for _ in range(num_rrdb)
        ]
        self.rrdb_trunk = nn.Sequential(*rrdb_blocks)
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)

        # Upsampling: two x2 stages using PixelShuffle
        self.upsample1 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1, bias=True),
            nn.PixelShuffle(2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1, bias=True),
            nn.PixelShuffle(2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Output convolutions
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all Conv2d weights with kaiming_normal_."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.

        Args:
            x: Input LR tensor [B, 3, H, W].

        Returns:
            SR tensor [B, 3, H*4, W*4].
        """
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.rrdb_trunk(feat))
        feat = feat + trunk

        feat = self.upsample1(feat)
        feat = self.upsample2(feat)

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out