from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import LLM, ModelConfig
from src.tokenizer import BPETokenizer
from src.tokenizer.io_utils import load_tokenizer

def generate(model: LLM, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    model.eval()
    device = input_ids.device
    
    current_ids = input_ids
    # Setup initial cache
    b, seq_len = current_ids.shape
    _, caches = model(current_ids, use_cache=True)
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Pass only the last token, providing the cache
            next_token_id = current_ids[:, -1:]
            logits, caches = model(next_token_id, use_cache=True, caches=caches)
            
            # Get last pos
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
    return torch.cat(generated, dim=1)


def main():
    parser = argparse.ArgumentParser(description="Test model inference with text payload.")
    parser.add_argument("--text", type=str, required=True, help="Input text payload")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--model_path", type=str, default="final_model.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer_vocab.json", help="Path to tokenizer")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--d_c", type=int, default=64)
    parser.add_argument("--d_rope", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    if not Path(args.tokenizer_path).exists():
        print("Tokenizer file not found!")
        return
        
    merges, vocab = load_tokenizer(args.tokenizer_path)
    tokenizer = BPETokenizer()
    tokenizer.merges = merges
    tokenizer.vocab = vocab
    vocab_size = len(tokenizer.vocab)
    print(f"Tokenizer vocab size: {vocab_size}")

    # 2. Initialize Model
    print("Initializing model architecture...")
    model = LLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_c=args.d_c,
        n_experts=args.n_experts,
        max_seq_len=args.max_seq_len,
        d_rope=args.d_rope,
    ).to(device)
    
    if Path(args.model_path).exists():
        print(f"Loading trained weights from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Warning: {args.model_path} not found. Using untrained random weights.")
    
    # 3. Process Input
    tokens = tokenizer.encode(args.text)
    token_ids = [tokenizer.vocab[t] for t in tokens if t in tokenizer.vocab]
    
    if not token_ids:
        print("Empty input after tokenization.")
        return
        
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    print("\n--- Input payload ---")
    print(args.text)
    print(f"Tokens: {input_tensor.tolist()[0]}")
    
    # 4. Generate
    print("\n--- Generating ---")
    out_tensor = generate(model, input_tensor, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    out_ids = out_tensor[0].tolist()
    
    # Decode back to text
    out_tokens = [list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(idx)] for idx in out_ids]
    decoded_text = tokenizer.decode(out_tokens)
    
    print(decoded_text)
    
    if not Path(args.model_path).exists():
        print("\n(Note: The model is not trained yet, so the output is random text)")

if __name__ == "__main__":
    main()
