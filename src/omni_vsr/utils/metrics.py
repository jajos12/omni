"""Metrics and transcript normalization."""

from __future__ import annotations

from typing import Iterable

from omni_vsr.tokenizer import TOKENIZER

try:
    import editdistance  # type: ignore
except ImportError:  # pragma: no cover
    editdistance = None


def levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    if editdistance is not None:
        return int(editdistance.eval(reference, hypothesis))

    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = TOKENIZER.normalize(reference).split()
    hyp_words = TOKENIZER.normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def corpus_word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    total_edits = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words = TOKENIZER.normalize(reference).split()
        hyp_words = TOKENIZER.normalize(hypothesis).split()
        total_edits += levenshtein_distance(ref_words, hyp_words)
        total_words += max(1, len(ref_words))
    return total_edits / total_words if total_words else 0.0
