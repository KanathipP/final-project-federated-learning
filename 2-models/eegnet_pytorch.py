# ===== eegnet_baseline.py =====
"""
eegnet_baseline.py
------------------
Baseline single-task EEGNet (motor imagery friendly).

- Input: (B, C, T) or (B, 1, C, T)
- Output: logits of shape (B, num_classes)
- Canonical ordering: temporal conv -> depthwise spatial -> separable temporal -> classifier
- Uses BatchNorm2d + SpatialDropout2d (like common EEGNet implementations)
- Safe Kaiming/Xavier init
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------- Utils ----------
def _kaiming_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------- Building blocks ----------
class Block1(nn.Module):
    """
    Temporal conv -> (BN) -> Depthwise spatial conv across C -> (BN) -> ELU -> AvgPool(T) -> SpatialDropout
    Input : (B, 1, C, T)
    Output: (B, F1*d, 1, T/pool1)
    """
    def __init__(
        self,
        in_channel: int,
        f1: int = 8,
        d: int = 2,
        kernel_length: int = 64,
        pool1: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv_temporal = nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        # depthwise across electrodes (height=C)
        self.depthwise = nn.Conv2d(f1, f1 * d, (in_channel, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d)

        self.act = nn.ELU(inplace=True)
        self.pool = nn.AvgPool2d((1, pool1), stride=(1, pool1))
        self.drop = nn.Dropout2d(dropout)

        self.out_channels = f1 * d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class Block2(nn.Module):
    """
    Separable temporal: depthwise along T -> pointwise -> BN -> ELU -> AvgPool(T) -> SpatialDropout
    Input : (B, Fin, 1, T)
    Output: (B, Fin, 1, T/pool2)  # channel count unchanged here
    """
    def __init__(
        self,
        in_channels: int,
        sep_kernel: int = 16,
        pool2: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.sep_depth = nn.Conv2d(in_channels, in_channels, (1, sep_kernel),
                                   padding=(0, sep_kernel // 2), groups=in_channels, bias=False)
        self.sep_point = nn.Conv2d(in_channels, in_channels, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ELU(inplace=True)
        self.pool = nn.AvgPool2d((1, pool2), stride=(1, pool2))
        self.drop = nn.Dropout2d(dropout)
        self.out_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class Classifier(nn.Module):
    """Global average pooling + Linear classifier head (logits)."""
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)          # (B, F, 1, 1)
        x = torch.flatten(x, 1)  # (B, F)
        return self.fc(x)


# ---------- EEGNet (single-task) ----------
class EEGNet(nn.Module):
    """
    Baseline EEGNet for single-task classification (e.g., motor imagery).

    Args:
        in_channel: number of EEG electrodes (C)
        num_classes: number of classes (e.g., 4 for MI)
        f1, d: EEGNet hyperparameters
        kernel_length, sep_kernel: temporal kernel sizes
        pool1, pool2: temporal pooling factors
        dropout: SpatialDropout2d probability
    """
    def __init__(
        self,
        in_channel: int,
        num_classes: int = 4,
        f1: int = 8,
        d: int = 2,
        kernel_length: int = 64,
        sep_kernel: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.block1 = Block1(in_channel, f1=f1, d=d, kernel_length=kernel_length, pool1=pool1, dropout=dropout)
        self.block2 = Block2(self.block1.out_channels, sep_kernel=sep_kernel, pool2=pool2, dropout=dropout)
        self.classifier = Classifier(self.block2.out_channels, num_classes=num_classes)

        _kaiming_init(self)

    def _ensure_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:          # (B, C, T)
            return x.unsqueeze(1)   # -> (B, 1, C, T)
        if x.ndim == 4 and x.shape[1] == 1:
            return x
        raise ValueError(f"Expected (B,C,T) or (B,1,C,T), got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_input(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


# ---------- quick smoke test ----------
if __name__ == "__main__":
    B, C, T = 2, 64, 256
    x = torch.randn(B, C, T)
    model = EEGNet(in_channel=C, num_classes=4)
    y = model(x)
    print("EEGNet output shape:", tuple(y.shape))  # (B, 4)
