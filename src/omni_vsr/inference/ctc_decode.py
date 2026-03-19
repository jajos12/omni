"""CTC decoding utilities."""

from __future__ import annotations

from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

from omni_vsr.tokenizer import BLANK_IDX, EOS_IDX, SOS_IDX, TOKENIZER, VOCAB

try:
    from pyctcdecode import build_ctcdecoder
except ImportError:  # pragma: no cover
    build_ctcdecoder = None


def _sanitize_log_probs(log_probs: np.ndarray) -> np.ndarray:
    sanitized = np.asarray(log_probs, dtype=np.float32).copy()
    sanitized[:, SOS_IDX] = -1.0e9
    sanitized[:, EOS_IDX] = -1.0e9
    return sanitized


def build_beam_decoder(lm_path: str | Path, alpha: float, beta: float):
    if build_ctcdecoder is None:
        return None
    lm_path = Path(lm_path)
    if not lm_path.exists():
        return None
    labels = ["" if index == BLANK_IDX else token for index, token in enumerate(VOCAB)]
    return build_ctcdecoder(
        labels=labels,
        kenlm_model=str(lm_path),
        alpha=alpha,
        beta=beta,
        ctc_token_idx=BLANK_IDX,
    )


def greedy_decode_log_probs(log_probs_batch: list[np.ndarray]) -> list[str]:
    outputs: list[str] = []
    for log_probs in log_probs_batch:
        token_ids = _sanitize_log_probs(log_probs).argmax(axis=-1).tolist()
        outputs.append(TOKENIZER.collapse_ctc(token_ids))
    return outputs


def decode_batch_ctc(
    log_probs_batch: list[np.ndarray],
    decoder,
    beam_width: int = 80,
    decode_workers: int = 4,
) -> list[str]:
    sanitized_batch = [_sanitize_log_probs(log_probs) for log_probs in log_probs_batch]
    if decoder is None or beam_width <= 1:
        return greedy_decode_log_probs(sanitized_batch)

    with Pool(processes=max(1, decode_workers)) as pool:
        decoded = decoder.decode_batch(pool, sanitized_batch, beam_width=beam_width)
    return [TOKENIZER.normalize(text.replace("<sos>", " ").replace("<eos>", " ")) for text in decoded]
