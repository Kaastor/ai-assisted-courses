"""Vocabulary utilities for the SMS spam classification lab."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence


@dataclass
class Vocab:
    """Simple token-to-index vocabulary."""

    stoi: Dict[str, int]
    itos: Sequence[str]
    unk_token: str = "<unk>"

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def unk_index(self) -> int:
        return self.stoi.get(self.unk_token, 0)

    def encode(self, tokens: Iterable[str]) -> List[int]:
        encoded = [self.stoi.get(token, self.unk_index) for token in tokens]
        return encoded or [self.unk_index]


def _build_vocab(
    texts: Iterable[str],
    tokenizer: Callable[[str], Iterable[str]],
    min_freq: int,
    max_size: int,
) -> Vocab:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    sorted_tokens = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    itos = ["<unk>"]
    for token, freq in sorted_tokens:
        if freq < min_freq:
            continue
        if len(itos) >= max_size:
            break
        itos.append(token)
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def build_word_vocab(
    texts: Iterable[str], tokenizer: Callable[[str], Iterable[str]], min_freq: int = 2, max_size: int = 30000
) -> Vocab:
    """Construct a vocabulary of word tokens."""

    return _build_vocab(texts, tokenizer, min_freq=min_freq, max_size=max_size)


def build_char_vocab(texts: Iterable[str], min_freq: int = 1, max_size: int = 256) -> Vocab:
    """Construct a vocabulary of characters."""

    return _build_vocab(texts, tokenizer=lambda text: list(text), min_freq=min_freq, max_size=max_size)
