from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .pair_stats import Pair, Token


def save_tokenizer(path: str, merges: Dict[Pair, Token], vocab: Dict[Token, int]) -> None:
    payload = {
        "merges": [[a, b, merged] for (a, b), merged in merges.items()],
        "vocab": vocab,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tokenizer(path: str) -> Tuple[Dict[Pair, Token], Dict[Token, int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    merges = {(a, b): merged for a, b, merged in payload.get("merges", [])}
    vocab = payload.get("vocab", {})
    return merges, vocab
