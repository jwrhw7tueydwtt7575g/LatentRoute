from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .pair_stats import Pair, Token, get_pair_counts


class BPETokenizer:
    """Standard BPE tokenizer that merges the highest-frequency adjacent pair."""

    def __init__(self, vocab_size: int = 50_000, end_of_word: str = "</w>"):
        self.vocab_size = vocab_size
        self.end_of_word = end_of_word
        self.merges: Dict[Pair, Token] = {}
        self.vocab: Dict[Token, int] = {}

    def _prepare_word_freqs(self, text: str) -> Counter[str]:
        return Counter(text.split())

    def _prepare_sequences(self, word_freqs: Mapping[str, int]) -> Dict[Tuple[Token, ...], int]:
        sequences: Dict[Tuple[Token, ...], int] = {}
        for word, freq in word_freqs.items():
            seq = tuple(list(word) + [self.end_of_word])
            sequences[seq] = freq
        return sequences

    def _init_vocab(self, sequences: Mapping[Tuple[Token, ...], int]) -> None:
        symbols = sorted({token for seq in sequences for token in seq})
        self.vocab = {tok: i for i, tok in enumerate(symbols)}

    def _apply_merge(
        self,
        sequences: Mapping[Tuple[Token, ...], int],
        pair: Pair,
        merged: Token,
    ) -> Dict[Tuple[Token, ...], int]:
        new_sequences: Dict[Tuple[Token, ...], int] = {}
        for seq, freq in sequences.items():
            out: List[Token] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            out_seq = tuple(out)
            new_sequences[out_seq] = new_sequences.get(out_seq, 0) + freq
        return new_sequences

    def _select_best_pair(self, pair_counts: Counter[Pair]) -> Pair:
        return max(pair_counts, key=pair_counts.get)

    def train(self, text: str) -> None:
        word_freqs = self._prepare_word_freqs(text)
        sequences = self._prepare_sequences(word_freqs)
        self._init_vocab(sequences)

        while len(self.vocab) < self.vocab_size:
            pair_counts = get_pair_counts(sequences)
            if not pair_counts:
                break

            best_pair = self._select_best_pair(pair_counts)
            merged = "".join(best_pair)

            sequences = self._apply_merge(sequences, best_pair, merged)
            self.merges[best_pair] = merged
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

    def encode(self, word: str) -> List[Token]:
        tokens: List[Token] = list(word) + [self.end_of_word]
        if not self.merges:
            return tokens

        changed = True
        while changed:
            changed = False
            i = 0
            out: List[Token] = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in self.merges:
                    out.append(self.merges[(tokens[i], tokens[i + 1])])
                    i += 2
                    changed = True
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    def decode(self, tokens: Iterable[Token]) -> str:
        text = "".join(tokens)
        return text.replace(self.end_of_word, "")
