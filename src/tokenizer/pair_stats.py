from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

Token = str
Pair = Tuple[Token, Token]


def get_pair_counts(sequences: Mapping[Tuple[Token, ...], int]) -> Counter[Pair]:
    """Count adjacent token pairs across weighted token sequences."""
    counts: Counter[Pair] = Counter()
    for tokens, freq in sequences.items():
        if len(tokens) < 2:
            continue
        for i in range(len(tokens) - 1):
            counts[(tokens[i], tokens[i + 1])] += freq
    return counts


def get_next_token_distributions(
    sequences: Mapping[Tuple[Token, ...], int],
) -> Dict[Pair, Counter[Token]]:
    """For each pair (a,b), count which token c follows it in observed sequences."""
    follow: Dict[Pair, Counter[Token]] = defaultdict(Counter)
    for tokens, freq in sequences.items():
        for i in range(len(tokens) - 2):
            pair = (tokens[i], tokens[i + 1])
            follow[pair][tokens[i + 2]] += freq
    return follow


def normalize(counter: Mapping[Token, int]) -> Dict[Token, float]:
    total = float(sum(counter.values()))
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def shannon_entropy_from_probs(probabilities: Iterable[float]) -> float:
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * log(p)
    return entropy


def distribution_entropy(counter: Mapping[Token, int]) -> float:
    return shannon_entropy_from_probs(normalize(counter).values())
