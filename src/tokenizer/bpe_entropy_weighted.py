from __future__ import annotations

from collections import Counter
from math import exp
from typing import Dict, Mapping, Tuple

from .bpe_standard import BPETokenizer
from .pair_stats import Pair, Token, distribution_entropy, get_next_token_distributions


class EntropyWeightedBPETokenizer(BPETokenizer):
    """BPE variant that scores merges by frequency * exp(-entropy(context))."""

    def score_pair(
        self,
        pair: Pair,
        freq: int,
        next_distributions: Mapping[Pair, Counter[Token]],
    ) -> float:
        context = next_distributions.get(pair, Counter())
        entropy = distribution_entropy(context)
        return float(freq) * exp(-entropy)

    def _select_best_pair(self, pair_counts: Counter[Pair], sequences=None) -> Pair:  # type: ignore[override]
        if sequences is None:
            return super()._select_best_pair(pair_counts)

        next_distributions = get_next_token_distributions(sequences)
        best_pair = None
        best_score = -1.0
        for pair, freq in pair_counts.items():
            score = self.score_pair(pair, freq, next_distributions)
            if score > best_score:
                best_score = score
                best_pair = pair
        assert best_pair is not None
        return best_pair

    def train(self, text: str) -> None:  # type: ignore[override]
        word_freqs = self._prepare_word_freqs(text)
        sequences = self._prepare_sequences(word_freqs)
        self._init_vocab(sequences)

        while len(self.vocab) < self.vocab_size:
            pair_counts = Counter()
            for seq, freq in sequences.items():
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += freq

            if not pair_counts:
                break

            best_pair = self._select_best_pair(pair_counts, sequences=sequences)
            merged = "".join(best_pair)

            sequences = self._apply_merge(sequences, best_pair, merged)
            self.merges[best_pair] = merged
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
