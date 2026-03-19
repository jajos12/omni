"""Model factory helpers."""

from __future__ import annotations

from omni_vsr.config import ExperimentConfig
from omni_vsr.models.vsr import VSRModel


def build_model_from_config(config: ExperimentConfig) -> VSRModel:
    return VSRModel(
        d_model=config.model.d_model,
        encoder_layers=config.model.encoder_layers,
        decoder_layers=config.model.decoder_layers,
        encoder_heads=config.model.encoder_heads,
        decoder_heads=config.model.decoder_heads,
        ff_expand=config.model.ff_expand,
        decoder_ff_dim=config.model.decoder_ff_dim,
        conv_kernel=config.model.conv_kernel,
        dropout=config.model.dropout,
        inter_ctc_every=config.model.inter_ctc_every,
        gradient_checkpointing=config.model.gradient_checkpointing,
    )
