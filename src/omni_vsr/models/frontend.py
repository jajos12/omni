"""3D frontend + 2D residual backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class BasicBlock2D(nn.Module):
    """Standard residual block used in the spatial backbone."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return self.relu(x)


class ResNet3DFrontend(nn.Module):
    """Temporal 3D stem followed by a per-frame ResNet-18 style backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
            bias=False,
        )
        self.bn3d = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool3d = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.layer1 = self._make_layer(64, 64, block_count=2, stride=1)
        self.layer2 = self._make_layer(64, 128, block_count=2, stride=2)
        self.layer3 = self._make_layer(128, 256, block_count=2, stride=2)
        self.layer4 = self._make_layer(256, 512, block_count=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._reset_parameters()

    def _make_layer(self, in_channels: int, out_channels: int, block_count: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock2D(in_channels, out_channels, stride=stride)]
        for _ in range(1, block_count):
            layers.append(BasicBlock2D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count = videos.shape[:2]
        x = videos.permute(0, 2, 1, 3, 4)
        x = self.pool3d(self.relu(self.bn3d(self.conv3d(x))))
        x = x.permute(0, 2, 1, 3, 4)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return rearrange(x, "(b t) d -> b t d", b=batch_size, t=frame_count)
