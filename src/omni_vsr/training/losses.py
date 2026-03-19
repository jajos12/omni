"""Training losses."""

from __future__ import annotations

import torch
import torch.nn as nn

from omni_vsr.models.vsr import VSRForwardOutput
from omni_vsr.tokenizer import BLANK_IDX, IGNORE_INDEX


class JointCTCAttentionLoss(nn.Module):
    """Joint CTC and decoder cross-entropy loss with intermediate CTC regularization."""

    def __init__(
        self,
        blank_idx: int = BLANK_IDX,
        ignore_index: int = IGNORE_INDEX,
        ctc_weight: float = 0.3,
        inter_ctc_weight: float = 0.1,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.ctc_weight = ctc_weight
        self.inter_ctc_weight = inter_ctc_weight

    @staticmethod
    def _flatten_ctc_targets(ctc_targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                ctc_targets[index, : int(lengths[index].item())]
                for index in range(ctc_targets.shape[0])
            ],
            dim=0,
        )

    def forward(
        self,
        outputs: VSRForwardOutput,
        ctc_targets: torch.Tensor,
        ctc_target_lengths: torch.Tensor,
        decoder_targets: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        flat_targets = self._flatten_ctc_targets(ctc_targets, ctc_target_lengths)
        ctc_loss = self.ctc_loss(
            outputs.ctc_log_probs,
            flat_targets.cpu(),
            input_lengths.cpu(),
            ctc_target_lengths.cpu(),
        )
        attn_loss = self.ce_loss(
            outputs.decoder_logits.reshape(-1, outputs.decoder_logits.shape[-1]),
            decoder_targets.reshape(-1),
        )

        inter_loss = torch.tensor(0.0, device=outputs.decoder_logits.device)
        if outputs.inter_ctc_log_probs:
            for inter_logits in outputs.inter_ctc_log_probs:
                inter_loss = inter_loss + self.ctc_loss(
                    inter_logits,
                    flat_targets.cpu(),
                    input_lengths.cpu(),
                    ctc_target_lengths.cpu(),
                )
            inter_loss = inter_loss / len(outputs.inter_ctc_log_probs)

        total = self.ctc_weight * ctc_loss + (1.0 - self.ctc_weight) * attn_loss + self.inter_ctc_weight * inter_loss
        return {
            "total": total,
            "ctc": ctc_loss,
            "attn": attn_loss,
            "inter": inter_loss,
        }
