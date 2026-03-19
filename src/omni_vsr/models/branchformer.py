"""E-Branchformer encoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from omni_vsr.models.common import build_sinusoidal_encoding, make_padding_mask


class ConvolutionalGatedMLP(nn.Module):
    """Local modeling branch with depthwise temporal convolution."""

    def __init__(self, d_model: int, ff_expand: int = 4, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = d_model * ff_expand
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim * 2)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x_main, x_gate = self.fc1(x).chunk(2, dim=-1)
        x_main = F.gelu(x_main)
        x_main = self.depthwise(x_main.transpose(1, 2)).transpose(1, 2)
        x = x_main * torch.sigmoid(x_gate)
        return self.dropout(self.fc2(x))


class EBranchformerBlock(nn.Module):
    """Parallel global-attention and local-convolution block."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        ff_expand: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_merge = nn.LayerNorm(2 * d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.cgmlp = ConvolutionalGatedMLP(
            d_model=d_model,
            ff_expand=ff_expand,
            kernel_size=conv_kernel,
            dropout=dropout,
        )
        self.merge_dw = nn.Conv1d(
            2 * d_model,
            2 * d_model,
            kernel_size=3,
            padding=1,
            groups=2 * d_model,
        )
        self.merge_fc = nn.Linear(2 * d_model, d_model)
        self.merge_drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expand),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        attn_out = self.attn_drop(attn_out)
        local_out = self.cgmlp(x_norm)

        merged = torch.cat([attn_out, local_out], dim=-1)
        merged = self.norm_merge(merged)
        merged = self.merge_dw(merged.transpose(1, 2)).transpose(1, 2)
        x = x + self.merge_drop(self.merge_fc(merged))
        x = x + 0.5 * self.ff(self.norm_ff(x))
        return self.norm_final(x)


class EBranchformerEncoder(nn.Module):
    """Stacked E-Branchformer encoder."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 18,
        n_heads: int = 4,
        ff_expand: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        inter_ctc_every: int = 4,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_enc", build_sinusoidal_encoding(8192, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                EBranchformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_expand=ff_expand,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.inter_ctc_indices = (
            set(range(inter_ctc_every - 1, n_layers, inter_ctc_every))
            if inter_ctc_every > 0
            else set()
        )
        self.gradient_checkpointing = gradient_checkpointing

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        max_length = x.shape[1]
        x = self.input_drop(self.input_norm(x) + self.pos_enc[:, :max_length, :])
        padding_mask = make_padding_mask(lengths, max_length=max_length)
        intermediate_outputs: list[torch.Tensor] = []

        for index, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda hidden: layer(hidden, padding_mask),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, padding_mask)
            if index in self.inter_ctc_indices:
                intermediate_outputs.append(x)
        return x, intermediate_outputs
