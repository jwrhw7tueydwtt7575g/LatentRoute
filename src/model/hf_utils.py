import os

def upload_to_hf(model_dir, repo_id, token=None, private=True):
    """Upload a saved model directory to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("Error: 'huggingface_hub' package is not installed.")
        print("Please install it with: pip install huggingface_hub")
        return

    api = HfApi(token=token)
    
    print(f"Creating/Checking repo: {repo_id}")
    create_repo(repo_id, token=token, private=private, exist_ok=True)
    
    print(f"Uploading files from {model_dir} to {repo_id}...")
    upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        commit_message="Initial LatentRoute model upload",
        token=token
    )
    print("Upload complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing config.json and pytorch_model.bin")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo ID (e.g., 'username/my-latentroute-model')")
    parser.add_argument("--token", type=str, help="HF API token (optional if already logged in)")
    parser.add_argument("--public", action="store_true", help="Make repo public")
    
    args = parser.parse_args()
    upload_to_hf(args.model_dir, args.repo_id, token=args.token, private=not args.public)
