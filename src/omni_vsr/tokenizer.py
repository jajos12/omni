"""Character vocabulary and transcript normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass


BLANK_TOKEN = "<blank>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

VOCAB = [BLANK_TOKEN, SOS_TOKEN, EOS_TOKEN] + list("abcdefghijklmnopqrstuvwxyz") + [" "]
VOCAB_SIZE = len(VOCAB)
CHAR_TO_INDEX = {token: index for index, token in enumerate(VOCAB)}
INDEX_TO_CHAR = {index: token for token, index in CHAR_TO_INDEX.items()}

BLANK_IDX = CHAR_TO_INDEX[BLANK_TOKEN]
SOS_IDX = CHAR_TO_INDEX[SOS_TOKEN]
EOS_IDX = CHAR_TO_INDEX[EOS_TOKEN]
IGNORE_INDEX = -100

_NON_TEXT_PATTERN = re.compile(r"[^a-z\s]+")
_MULTISPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class CharacterTokenizer:
    """Tokenizer for CTC and autoregressive decoding."""

    blank_index: int = BLANK_IDX
    sos_index: int = SOS_IDX
    eos_index: int = EOS_IDX
    ignore_index: int = IGNORE_INDEX

    def normalize(self, text: str) -> str:
        normalized = _NON_TEXT_PATTERN.sub(" ", text.lower())
        normalized = _MULTISPACE_PATTERN.sub(" ", normalized).strip()
        return normalized

    def encode_ctc(self, text: str) -> list[int]:
        normalized = self.normalize(text)
        return [CHAR_TO_INDEX[char] for char in normalized if char in CHAR_TO_INDEX]

    def encode_sequence(self, text: str) -> list[int]:
        return [self.sos_index, *self.encode_ctc(text), self.eos_index]

    def decode_sequence(self, token_ids: list[int] | tuple[int, ...]) -> str:
        chars = [INDEX_TO_CHAR[token_id] for token_id in token_ids if token_id in INDEX_TO_CHAR]
        text = "".join(
            char
            for char in chars
            if char not in {BLANK_TOKEN, SOS_TOKEN, EOS_TOKEN}
        )
        return self.normalize(text)

    def collapse_ctc(self, token_ids: list[int] | tuple[int, ...]) -> str:
        collapsed: list[int] = []
        previous = None
        for token_id in token_ids:
            if token_id == self.blank_index:
                previous = token_id
                continue
            if token_id != previous:
                collapsed.append(token_id)
            previous = token_id
        return self.decode_sequence(collapsed)

    def pyctcdecode_labels(self) -> list[str]:
        labels = []
        for token in VOCAB:
            if token == BLANK_TOKEN:
                labels.append("")
            elif token in {SOS_TOKEN, EOS_TOKEN}:
                labels.append("")
            else:
                labels.append(token)
        return labels


TOKENIZER = CharacterTokenizer()
