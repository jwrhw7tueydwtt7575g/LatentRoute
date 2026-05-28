from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from src.tokenizer import BPETokenizer
from src.tokenizer.io_utils import save_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--num_articles", type=int, default=2000)
    parser.add_argument("--output", type=str, default="tokenizer_vocab.json")
    args = parser.parse_args()

    print(f"Loading Wikipedia dataset to train tokenizer (vocab_size={args.vocab_size})...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
    
    iterator = iter(ds["train"])
    corpus = []
    for i in tqdm(range(args.num_articles), desc="Fetching articles"):
        try:
            sample = next(iterator)
            corpus.append(sample["text"])
        except StopIteration:
            break
            
    full_text = "\n".join(corpus)
    print(f"Loaded {len(corpus)} articles, total length {len(full_text)} chars.")
    print("Training BPE Tokenizer (this may take a while)...")
    
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(full_text)
    
    save_tokenizer(args.output, tokenizer.merges, tokenizer.vocab)
    print(f"Tokenizer saved to {args.output} with vocab size {len(tokenizer.vocab)}")

if __name__ == "__main__":
    main()
