"""Top-level visual speech recognition model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from omni_vsr.models.branchformer import EBranchformerEncoder
from omni_vsr.models.common import make_padding_mask
from omni_vsr.models.decoder import TransformerDecoder
from omni_vsr.models.frontend import ResNet3DFrontend
from omni_vsr.tokenizer import TOKENIZER, VOCAB_SIZE


@dataclass
class VSRForwardOutput:
    encoder_output: torch.Tensor
    ctc_log_probs: torch.Tensor
    inter_ctc_log_probs: list[torch.Tensor]
    decoder_logits: torch.Tensor


class VSRModel(nn.Module):
    """3D frontend + E-Branchformer encoder + CTC + autoregressive decoder."""

    def __init__(
        self,
        d_model: int = 256,
        encoder_layers: int = 18,
        decoder_layers: int = 6,
        encoder_heads: int = 4,
        decoder_heads: int = 4,
        ff_expand: int = 4,
        decoder_ff_dim: int = 2048,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        inter_ctc_every: int = 4,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.frontend = ResNet3DFrontend()
        self.proj = nn.Linear(512, d_model)
        self.encoder = EBranchformerEncoder(
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            ff_expand=ff_expand,
            conv_kernel=conv_kernel,
            dropout=dropout,
            inter_ctc_every=inter_ctc_every,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.ctc_head = nn.Linear(d_model, VOCAB_SIZE)
        self.inter_ctc_head = nn.Linear(d_model, VOCAB_SIZE)
        self.decoder = TransformerDecoder(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            ff_dim=decoder_ff_dim,
            dropout=dropout,
        )

    def encode(self, videos: torch.Tensor, video_lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        features = self.frontend(videos)
        features = self.proj(features)
        encoded, intermediate = self.encoder(features, lengths=video_lengths)
        ctc_logits = self.ctc_head(encoded)
        return encoded, intermediate, ctc_logits

    def forward(
        self,
        videos: torch.Tensor,
        decoder_inputs: torch.Tensor,
        video_lengths: torch.Tensor | None = None,
        decoder_lengths: torch.Tensor | None = None,
    ) -> VSRForwardOutput:
        encoder_output, intermediate, ctc_logits = self.encode(videos, video_lengths=video_lengths)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
        inter_ctc_log_probs = [
            F.log_softmax(self.inter_ctc_head(hidden), dim=-1).permute(1, 0, 2)
            for hidden in intermediate
        ]
        memory_padding_mask = make_padding_mask(video_lengths, max_length=encoder_output.shape[1])
        decoder_padding_mask = make_padding_mask(decoder_lengths, max_length=decoder_inputs.shape[1])
        decoder_logits = self.decoder(
            decoder_inputs=decoder_inputs,
            memory=encoder_output,
            decoder_padding_mask=decoder_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )
        return VSRForwardOutput(
            encoder_output=encoder_output,
            ctc_log_probs=ctc_log_probs,
            inter_ctc_log_probs=inter_ctc_log_probs,
            decoder_logits=decoder_logits,
        )

    @torch.no_grad()
    def greedy_decode(self, videos: torch.Tensor, video_lengths: torch.Tensor | None = None) -> list[str]:
        self.eval()
        _, _, ctc_logits = self.encode(videos, video_lengths=video_lengths)
        predictions = ctc_logits.argmax(dim=-1).cpu().tolist()
        results: list[str] = []
        for batch_index, token_ids in enumerate(predictions):
            if video_lengths is not None:
                token_ids = token_ids[: int(video_lengths[batch_index].item())]
            results.append(TOKENIZER.collapse_ctc(token_ids))
        return results

    def count_parameters(self) -> dict[str, float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total": float(total),
            "trainable": float(trainable),
            "total_M": total / 1e6,
            "trainable_M": trainable / 1e6,
        }
