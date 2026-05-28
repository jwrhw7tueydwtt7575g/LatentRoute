import os
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import LLM, ModelConfig
from src.training.optim import OptimizerConfig, create_adamw, create_scheduler
from src.training.objectives import TrainingLossConfig, compute_language_model_loss
from src.tokenizer import BPETokenizer
from src.tokenizer.io_utils import load_tokenizer

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer_path = args.tokenizer_path
    print(f"Loading tokenizer from {tokenizer_path}...")
    if not Path(tokenizer_path).exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}! Please run `python scripts/train_tokenizer_offline.py` first.")
        return
        
    merges, vocab = load_tokenizer(tokenizer_path)
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.merges = merges
    tokenizer.vocab = vocab
    
    # 2. Dataset (Streaming)
    print("Loading Wikipedia dataset (streaming=True)...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
    train_data = dataset["train"]
    it = iter(train_data)

    # 3. Model
    model = LLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_c=args.d_c,
        n_experts=args.n_experts,
        max_seq_len=args.max_seq_len,
        d_rope=args.d_rope,
    ).to(device)

    # 3. Training Utils
    optim_cfg = OptimizerConfig(
        lr_max=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps
    )
    loss_cfg = TrainingLossConfig(lambda_route=0.01, lambda_aux=0.001)
    
    optimizer = create_adamw(model, optim_cfg)
    scheduler = create_scheduler(optimizer, optim_cfg)

    # 4. Training Loop
    print("Starting training...")
    model.train()
    step = 0
    
    # Text processing queue
    token_buffer = []
    
    def get_batch():
        nonlocal token_buffer, it
        target_len = args.batch_size * (args.max_seq_len + 1) # +1 for labels
        
        while len(token_buffer) < target_len:
            try:
                sample = next(it)
                text = sample["text"]
                tokens = tokenizer.encode(text)
                ids = [tokenizer.vocab[t] for t in tokens if t in tokenizer.vocab]
                token_buffer.extend(ids)
            except StopIteration:
                it = iter(train_data)
                
        # Slice out the batch
        batch_tokens = token_buffer[:target_len]
        token_buffer = token_buffer[target_len:]
        
        # Shape: (batch_size, seq_len + 1)
        batch_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device).view(args.batch_size, args.max_seq_len + 1)
        
        input_ids = batch_tensor[:, :-1].contiguous()
        labels = batch_tensor[:, 1:].contiguous()
        return input_ids, labels
        
    pbar = tqdm(total=args.total_steps, desc="Training Steps")
    while step < args.total_steps:
        input_ids, labels = get_batch()

        loss, parts = compute_language_model_loss(model, input_ids, labels, loss_cfg)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "ce": f"{parts['loss_ce'].item():.4f}"})
        
        step += 1
        pbar.update(1)

        if step % args.save_interval == 0:
            torch.save(model.state_dict(), f"checkpoint_step_{step}.pt")

    pbar.close()
    print("Training complete.")
    torch.save(model.state_dict(), "final_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--d_c", type=int, default=64)
    parser.add_argument("--d_rope", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer_vocab.json")
    
    args = parser.parse_args()
    train(args)
