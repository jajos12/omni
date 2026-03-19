"""Autoregressive decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from omni_vsr.models.common import build_sinusoidal_encoding


class TransformerDecoder(nn.Module):
    """Standard Transformer decoder for character generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_enc", build_sinusoidal_encoding(4096, d_model))
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(
        self,
        decoder_inputs: torch.Tensor,
        memory: torch.Tensor,
        decoder_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_length = decoder_inputs.shape[1]
        hidden = self.embed(decoder_inputs) * math.sqrt(self.d_model)
        hidden = self.dropout(hidden + self.pos_enc[:, :sequence_length, :])
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=decoder_inputs.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output(self.norm(hidden))
