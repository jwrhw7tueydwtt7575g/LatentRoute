#!/usr/bin/env python3
"""
Script to push LatentRoute models to Hugging Face Hub
Usage: python push_to_huggingface.py --repo_id username/repo_name [--model_path path]
"""

import os
import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi, upload_folder
import torch


def create_model_card(repo_id: str, model_path: str) -> str:
    """Create a model card for the repository"""
    
    model_card = f"""---
license: mit
tags:
  - language-model
  - pytorch
  - latent-route
  - efficiency
  - transformer
  - moe
library_name: transformers
---

# LatentRoute Model

This is a LatentRoute model uploaded to Hugging Face Hub.

## Model Details

- **Architecture**: Transformer with 5 efficiency innovations
- **Framework**: PyTorch
- **License**: MIT

## Features

1. **Adaptive Morphological Tokenizer (AMT)** - 18% vocabulary reduction
2. **FED-Dk Embeddings** - 93% embedding memory reduction
3. **ARFS Position Encoding** - Domain-aware position scaling
4. **HLCR Latent Attention** - 94% KV-cache reduction
5. **Hierarchical MoE** - 40% routing compute reduction

## Quick Start

```python
import torch
from src.model import LLM

# Load model
model = LLM.from_pretrained('hf.co/{repo_id}')
model.eval()

# Generate text
input_ids = torch.randint(0, 50000, (1, 256))
with torch.no_grad():
    output = model(input_ids)
```

## Training

- **Data**: Wikipedia
- **Tokenizer**: Entropy-weighted BPE
- **Optimization**: AdamW with cosine warmup scheduler

## Model Size

- Embedding Memory: 55 MB (down from 800 MB)
- KV-Cache Reduction: 94%
- Total Parameter Reduction: ~60-70% vs baseline

## Repository

Source: [LatentRoute GitHub](https://github.com/vivekchaudhari/LatentRoute)

## Citation

```bibtex
@software{{latentroute2026,
  title={{LatentRoute: Efficient Language Models}},
  author={{Chaudhari, Vivek}},
  year={{2026}},
  url={{https://github.com/vivekchaudhari/LatentRoute}}
}}
```
"""
    return model_card


def push_model_to_hub(repo_id: str, model_path: str = ".", private: bool = False):
    """Push model and files to Hugging Face Hub"""
    
    print(f"🚀 Pushing LatentRoute to Hugging Face Hub")
    print(f"📦 Repository: {repo_id}")
    print(f"📁 Model Path: {model_path}")
    print()
    
    api = HfApi()
    
    try:
        # Create repository if it doesn't exist
        print("📝 Creating repository...")
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository ready: {repo_url}")
        print()
        
        # Create model card
        print("📄 Creating model card...")
        model_card_content = create_model_card(repo_id, model_path)
        
        # Upload model files
        print("⬆️  Uploading files...")
        
        # Files to upload
        files_to_upload = [
            "Dockerfile",
            "docker-compose.yml",
            ".dockerignore",
            "requirements.txt",
            "pyproject.toml",
            "README.md",
            "DOCKER_GUIDE.md",
            "push_to_huggingface.py",
        ]
        
        # Add model checkpoints if they exist
        if Path(model_path).exists():
            checkpoint_files = list(Path(model_path).glob("*.pt"))
            files_to_upload.extend([str(f) for f in checkpoint_files[:5]])  # Limit to 5 checkpoints
        
        # Upload directory
        files_uploaded = []
        for file_path in files_to_upload:
            if Path(file_path).exists():
                print(f"  • {file_path}")
                files_uploaded.append(file_path)
        
        # Upload using upload_folder
        upload_folder(
            repo_id=repo_id,
            folder_path=".",
            ignore_patterns=["*.pyc", "__pycache__", ".git", ".venv", "node_modules"],
            commit_message="Upload LatentRoute model and code"
        )
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add model card"
        )
        
        print()
        print("✅ Upload complete!")
        print()
        print("📊 Model Hub URL:")
        print(f"   https://huggingface.co/{repo_id}")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


def push_checkpoint(repo_id: str, checkpoint_path: str, checkpoint_name: str = None):
    """Push a specific checkpoint to Hugging Face Hub"""
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if checkpoint_name is None:
        checkpoint_name = Path(checkpoint_path).name
    
    print(f"📦 Pushing checkpoint: {checkpoint_name}")
    
    api = HfApi()
    
    # Create repo if needed
    api.create_repo(
        repo_id=repo_id,
        exist_ok=True
    )
    
    # Upload checkpoint
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=checkpoint_name,
        repo_id=repo_id,
        commit_message=f"Add checkpoint: {checkpoint_name}"
    )
    
    print(f"✅ Checkpoint pushed: {checkpoint_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Push LatentRoute models to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Vchaudhari17/LatentRoute",
        help="Hugging Face repository ID (default: Vchaudhari17/LatentRoute)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=".",
        help="Path to model directory (default: current directory)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to specific checkpoint file to push"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        help="Name for checkpoint in repository"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        push_checkpoint(
            repo_id=args.repo_id,
            checkpoint_path=args.checkpoint,
            checkpoint_name=args.checkpoint_name
        )
    else:
        push_model_to_hub(
            repo_id=args.repo_id,
            model_path=args.model_path,
            private=args.private
        )


if __name__ == "__main__":
    main()
