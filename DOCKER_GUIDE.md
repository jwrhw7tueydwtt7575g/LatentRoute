# LatentRoute Docker Setup Guide

Complete guide for dockerizing and deploying LatentRoute on distributed servers for training on 50k+ data.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Building the Image](#building-the-image)
- [Running Training](#running-training)
- [Docker Compose](#docker-compose)
- [Distributed Training](#distributed-training)
- [Registry Deployment](#registry-deployment)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Development Machine

1. **Docker** (≥ 20.10)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install docker.io
   sudo usermod -aG docker $USER
   newgrp docker

   # macOS
   brew install docker
   # Then start Docker Desktop
   ```

2. **NVIDIA Docker Runtime** (for GPU support)
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Docker Compose** (≥ 1.29)
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
     -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

4. **Disk Space**: At least **100 GB** for:
   - Docker image: ~8 GB
   - Training data cache: ~50 GB
   - Model checkpoints: ~10-20 GB

### Target Server (for distributed training)

Same as above, plus:
- NVIDIA GPU (ideally V100, A100, or H100)
- Docker, nvidia-docker, docker-compose
- Network access to pull from registry

---

## Quick Start

### 1. Build Docker Image

```bash
cd /path/to/LatentRoute
chmod +x docker-build.sh
./docker-build.sh
```

Expected output:
```
✅ Docker image built successfully!
Image: latentroute:latest
```

### 2. Run Training

```bash
chmod +x docker-run-train.sh
./docker-run-train.sh -s 1000 -b 4
```

This starts training with:
- 1000 total steps
- Batch size 4
- 1 GPU
- Outputs saved to `./models/`

### 3. View Logs

```bash
docker logs -f latentroute-train
```

### 4. Stop Training

```bash
docker stop latentroute-train
docker rm latentroute-train
```

---

## Building the Image

### Simple Build

```bash
./docker-build.sh
```

### Advanced Build Options

```bash
# Build without cache (rebuild all layers)
./docker-build.sh --no-cache

# Build and push to registry
./docker-build.sh --push --registry docker.io --image-name myusername/latentroute

# Specific tag
./docker-build.sh --image-tag v1.0
```

### Dockerfile Structure

The provided `Dockerfile` uses a **multi-stage build** approach:

- **Stage 1 (Builder)**: Installs all dependencies in nvidia/cuda:13.0-cudnn9-devel
- **Stage 2 (Runtime)**: Uses lighter nvidia/cuda:13.0-cudnn9-runtime

Benefits:
- Final image: ~8 GB (instead of ~15 GB)
- Faster downloads to servers
- Reduced storage on target machines

### Image Size Breakdown

```
Layer                               Size
─────────────────────────────────────────
CUDA base (runtime)                 4.5 GB
Python 3.12 + system libs           1.2 GB
PyTorch + dependencies              2.0 GB
LatentRoute source code              <50 MB
─────────────────────────────────────────
Total                               ~8 GB
```

---

## Running Training

### Using Shell Script (Recommended)

```bash
./docker-run-train.sh [COMMAND] [OPTIONS]
```

#### Commands

| Command | Purpose | GPU Required |
|---------|---------|------------|
| `train` | Full model training (default) | ✅ Yes |
| `tokenizer` | Train BPE tokenizer | ❌ No |
| `prepare-data` | Download Wikipedia data | ❌ No |
| `test` | Run smoke tests | ❌ No |
| `inference` | Test model inference | ✅ Optional |

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-g, --gpus NUM` | Number of GPUs | 1 |
| `-b, --batch-size NUM` | Batch size | 4 |
| `-s, --steps NUM` | Total training steps | 1000 |
| `-l, --learning-rate LR` | Learning rate | 0.0001 |
| `-i, --interactive` | Interactive bash | false |
| `-d, --detach` | Run in background | false |
| `-n, --name NAME` | Container name | latentroute-train |

#### Examples

```bash
# Train with defaults (1 GPU, batch_size=4, 1000 steps)
./docker-run-train.sh

# Train with 2 GPUs and 5000 steps
./docker-run-train.sh train -g 2 -s 5000

# Larger batch size, more steps
./docker-run-train.sh -b 8 -s 50000 -l 5e-5

# Interactive shell (for debugging)
./docker-run-train.sh --interactive

# Run in background
./docker-run-train.sh --detach
docker logs -f latentroute-train

# Train tokenizer first
./docker-run-train.sh tokenizer --num_articles 5000 --vocab_size 50000

# Run smoke tests
./docker-run-train.sh test

# Test inference
./docker-run-train.sh inference --prompt "Hello world"
```

### Direct Docker Run

```bash
# Single GPU training
docker run --gpus 1 -it \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  -e PYTHONUNBUFFERED=1 \
  latentroute:latest train --total_steps 5000 --batch_size 4

# Multi-GPU training
docker run --gpus all -it \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  -e PYTHONUNBUFFERED=1 \
  latentroute:latest train --total_steps 50000 --batch_size 8 --num_workers 4
```

### Environment Variables

Set in `.env.docker` or pass via `-e`:

```bash
# Training
TOTAL_STEPS=1000           # Training steps
BATCH_SIZE=4               # Batch size
LEARNING_RATE=0.0001       # Learning rate
WARMUP_STEPS=100           # LR warmup steps

# Model config
D_MODEL=512                # Model dimension
N_LAYERS=6                 # Number of layers
N_HEADS=8                  # Attention heads
N_EXPERTS=8                # MoE experts
VOCAB_SIZE=50000           # Vocabulary size
MAX_SEQ_LEN=256            # Max sequence length

# Caching
TORCH_HOME=/cache/torch
HF_HOME=/cache/huggingface

# GPU
CUDA_VISIBLE_DEVICES=0     # GPU IDs (comma-separated)
```

---

## Docker Compose

### Quick Start with Compose

```bash
chmod +x docker-compose-runner.sh

# Build
./docker-compose-runner.sh build

# Start training
./docker-compose-runner.sh up

# View logs
./docker-compose-runner.sh logs

# Stop
./docker-compose-runner.sh down
```

### Direct docker-compose Commands

```bash
# Build image
docker-compose build

# Start training service
docker-compose up -d latentroute-train

# View logs
docker-compose logs -f latentroute-train

# Enter container shell
docker-compose exec latentroute-train /bin/bash

# Stop all services
docker-compose down

# Clean up volumes
docker-compose down -v
```

### Compose File Structure

```yaml
services:
  latentroute-train:
    image: latentroute:latest
    runtime: nvidia              # Enable GPU
    gpus:
      count: 1                   # Number of GPUs
      capabilities: [gpu]
    volumes:
      - ./models:/models         # Checkpoint output
      - ./data:/data             # Input data
      - cache-huggingface:/cache/huggingface
    ports:
      - "8265:8265"              # Ray Tune dashboard
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
```

---

## Distributed Training

### Multi-GPU on Single Server

For training on 50k+ samples efficiently:

```bash
# 4 GPUs, larger batches
./docker-run-train.sh \
  -g 4 \
  -b 16 \
  -s 100000

# Or with environment variables
docker run --gpus all -it \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  latentroute:latest train \
  --total_steps 100000 \
  --batch_size 16 \
  --num_workers 4
```

### Multi-Server with Ray Tune

For truly distributed training across multiple servers:

#### 1. Start Ray Head Node (on server 1)

```bash
docker run --gpus all -it \
  -p 8265:8265 \
  -p 6379:6379 \
  -e PYTHONUNBUFFERED=1 \
  latentroute:latest \
  python -c "import ray; ray.init(address='auto', ignore_reinit_error=True)" && \
  python scripts/train_on_wiki.py \
    --total_steps 100000 \
    --batch_size 8 \
    --num_workers 4
```

#### 2. Join Worker Nodes (on servers 2, 3, ...)

```bash
export RAY_HEAD_IP=<server1_ip>

docker run --gpus all -it \
  -e PYTHONUNBUFFERED=1 \
  -e RAY_HEAD_IP=$RAY_HEAD_IP \
  latentroute:latest \
  python -c "import ray; ray.init(address=f'ray://{os.getenv(\"RAY_HEAD_IP\")}:6379')"
```

#### 3. Monitor Training

```bash
# Access Ray Tune dashboard on server 1
http://<server1_ip>:8265
```

---

## Registry Deployment

### Push to Docker Hub

```bash
chmod +x docker-push-registry.sh

# Login (one time)
./docker-push-registry.sh --login-only --docker-hub -u your_username

# Push image
./docker-push-registry.sh --docker-hub -u your_username

# Output: your_username/latentroute:latest
```

### Push to AWS ECR

```bash
# Configure AWS credentials first
aws configure

# Create ECR repository
aws ecr create-repository --repository-name latentroute

# Get registry URI
REGISTRY_URI=$(aws ecr describe-repositories \
  --repository-names latentroute \
  --query 'repositories[0].repositoryUri' \
  --output text)

# Push
./docker-push-registry.sh --aws-ecr $REGISTRY_URI
```

### Push to GCP GCR

```bash
# Configure GCP credentials
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Push
./docker-push-registry.sh --gcp-gcr YOUR_PROJECT_ID

# Output: gcr.io/YOUR_PROJECT_ID/latentroute:latest
```

### Push to Azure ACR

```bash
# Create ACR (if needed)
az acr create --resource-group myRG --name myregistry --sku Basic

# Push
./docker-push-registry.sh --azure myregistry

# Output: myregistry.azurecr.io/latentroute:latest
```

### Pull on Target Server

Once pushed to registry:

```bash
# Pull from Docker Hub
docker pull your_username/latentroute:latest

# Pull from AWS ECR
docker pull $REGISTRY_URI

# Pull from GCP GCR
docker pull gcr.io/YOUR_PROJECT_ID/latentroute:latest

# Run training
./docker-run-train.sh --image gcr.io/YOUR_PROJECT_ID/latentroute:latest \
  -s 50000 -b 8
```

---

## Troubleshooting

### GPU Not Detected

**Symptom**: `CUDA Available: False`

```bash
# Check nvidia-docker installation
nvidia-docker --version

# Test GPU access
nvidia-docker run --rm nvidia/cuda:13.0-runtime nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker

# Run with explicit GPU
docker run --gpus all ...
```

### Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

```bash
# Reduce batch size
./docker-run-train.sh -b 2  # Down from 4

# Reduce model size in .env.docker
D_MODEL=256                # Down from 512
N_LAYERS=4                 # Down from 6
N_EXPERTS=4                # Down from 8

# Enable gradient checkpointing
-e GRADIENT_CHECKPOINTING=true
```

### Slow Data Loading

**Symptom**: GPU utilization low, CPU high

```bash
# Increase number of workers
docker run ... train --num_workers 8

# Pre-download Wikipedia data
docker run ... prepare-data --num_articles 100000

# Use SSD instead of HDD for /cache/huggingface
```

### Network Issues (HF Dataset Download)

**Symptom**: `ConnectionError` or timeouts

```bash
# Pre-download data on machine with good internet
./docker-run-train.sh prepare-data --num_articles 50000

# Copy ./data/wiki_corpus.jsonl to target server

# Use local data
docker run -v /path/to/wiki_corpus.jsonl:/data/wiki_corpus.jsonl ...
```

### Image Build Failures

**Symptom**: Build fails at PyTorch install

```bash
# Build without cache
./docker-build.sh --no-cache

# Check disk space
df -h

# Increase Docker resources
# Docker Desktop → Preferences → Resources → Disk Image Size
```

### Container Exits Immediately

**Symptom**: Container starts then stops

```bash
# Check logs
docker logs <container_id>

# Run interactively to see errors
./docker-run-train.sh --interactive

# Verify Python/dependencies
docker run -it latentroute:latest python -c "import torch; print(torch.__version__)"
```

---

## Performance Tips

### For 50k+ Training Data

1. **Use Multiple GPUs**
   ```bash
   ./docker-run-train.sh -g 4 -b 32 -s 100000
   ```

2. **Pre-download Data**
   ```bash
   ./docker-run-train.sh prepare-data --num_articles 50000
   ```

3. **Use Fast Storage**
   - SSD for `/cache/huggingface` and `/models`
   - NVMe for best performance

4. **Enable Gradient Checkpointing** (saves memory, slight slowdown)
   ```bash
   -e GRADIENT_CHECKPOINTING=true
   ```

5. **Monitor with Ray Dashboard**
   ```bash
   http://localhost:8265
   ```

6. **Use Larger Model for Training**
   ```bash
   -e D_MODEL=1024 -e N_LAYERS=12 -e N_EXPERTS=16
   ```

---

## Example: Complete Workflow

### Local Training (Quick Test)

```bash
# 1. Build
./docker-build.sh

# 2. Prepare data (if not streaming)
./docker-run-train.sh prepare-data --num_articles 1000

# 3. Train tokenizer
./docker-run-train.sh tokenizer --num_articles 1000

# 4. Quick test with 100 steps
./docker-run-train.sh -b 4 -s 100

# 5. Check results
ls -lh models/
```

### Server Deployment (50k+ Data)

```bash
# 1. On build server:
./docker-build.sh
./docker-push-registry.sh --docker-hub -u myusername
# Output: myusername/latentroute:latest

# 2. On target server:
docker pull myusername/latentroute:latest

# 3. Run 50k training
docker run --gpus all -it \
  -v /mnt/ssd/models:/models \
  -v /mnt/ssd/cache:/cache \
  -e PYTHONUNBUFFERED=1 \
  myusername/latentroute:latest train \
  --total_steps 50000 \
  --batch_size 16 \
  --lr 1e-4

# 4. Monitor
docker logs -f <container_id>

# 5. Collect results
scp -r /mnt/ssd/models/* local_machine:/results/
```

---

## Additional Resources

- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Ray Documentation](https://docs.ray.io/)

---

**Last Updated**: May 2026
**LatentRoute Version**: 0.1.0
