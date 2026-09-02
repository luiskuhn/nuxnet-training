from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-convolution residual block with spatial feature-map dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout_1 = nn.Dropout3d(dropout)
        self.dropout_2 = nn.Dropout3d(dropout)

        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm3d(out_channels)

        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm3d(out_channels)

        self.non_linearity = nn.ReLU(inplace=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.non_linearity(x)
        x = self.dropout_1(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.dropout_2(x)
        return self.non_linearity(x + identity)


class InputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.conv_block_1 = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block_1(x)


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Dropout3d(dropout),
            ConvBlock(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Dropout3d(dropout),
        )
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class OutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_1(x)


class UNet3D(nn.Module):
    """3D U-Net architecture ported from the NuMorph nucleus-marker training implementation."""

    def __init__(self, in_channels: int = 1, classes: int = 2, dropout: float = 0.10) -> None:
        super().__init__()
        self.inc = InputBlock(in_channels, 32, dropout=dropout)
        self.down1 = DownSamplingBlock(32, 64, dropout=dropout)
        self.down2 = DownSamplingBlock(64, 128, dropout=dropout)
        self.mid = ConvBlock(128, 128, dropout=dropout)
        self.up1 = UpSamplingBlock(128, 64, 64, dropout=dropout)
        self.up2 = UpSamplingBlock(64, 32, 32, dropout=dropout)
        self.outc = OutputBlock(32, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.mid(x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)
