#!/bin/bash
# Simple HF Upload Script - uploads only project files

cd /home/vivek/Desktop/LatentRoute

echo "📦 Uploading LatentRoute to Hugging Face..."
echo ""

# Activate venv
source .venv/bin/activate

# Files to upload - one by one
FILES=(
  "README.md"
  "requirements.txt"
  "pyproject.toml"
  "Dockerfile"
  ".dockerignore"
  "DOCKER_GUIDE.md"
  "docker-build.sh"
  "docker-run-train.sh"
  "docker-compose.sh"
  "docker-push-registry.sh"
  "docker-compose-runner.sh"
  "docker-compose.yml"
  ".env.docker"
  "push_to_huggingface.py"
)

# Upload scripts directory
echo "⬆️  Uploading scripts/"
hf upload Vchaudhari17/LatentRoute scripts/ --repo-type model

# Upload src directory  
echo "⬆️  Uploading src/"
hf upload Vchaudhari17/LatentRoute src/ --repo-type model

# Upload tests directory
echo "⬆️  Uploading tests/"
hf upload Vchaudhari17/LatentRoute tests/ --repo-type model

# Upload individual files
for file in "${FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "⬆️  Uploading $file..."
    hf upload Vchaudhari17/LatentRoute "$file" --repo-type model
  fi
done

echo ""
echo "✅ Upload complete!"
echo "📍 View at: https://huggingface.co/Vchaudhari17/LatentRoute"
