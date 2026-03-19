"""Common tensor helpers for sequence models."""

from __future__ import annotations

import math

import torch


def build_sinusoidal_encoding(max_length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(max_length).unsqueeze(1).float()
    divisor = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(max_length, d_model)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding.unsqueeze(0)


def make_padding_mask(lengths: torch.Tensor | None, max_length: int | None = None) -> torch.Tensor | None:
    if lengths is None:
        return None
    if max_length is None:
        max_length = int(lengths.max().item())
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)
