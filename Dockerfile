# Dockerfile for LatentRoute - GPU Training with CUDA 12.2 and cuDNN 8
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_HOME=/cache/torch \
    HF_HOME=/cache/huggingface \
    HF_DATASETS_OFFLINE=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch and all dependencies
RUN python3.11 -m pip install -r /tmp/requirements.txt --no-cache-dir

# Create working directory and cache directories
RUN mkdir -p /workspace /cache/torch /cache/huggingface /models /data && \
    chmod -R 777 /workspace /cache /models /data

# Copy project code
COPY . /workspace/

# Set working directory
WORKDIR /workspace

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 LatentRoute Docker Container Starting..."\n\
echo "Python: $(python --version)"\n\
echo "PyTorch: $(python -c \"import torch; print(torch.__version__)\")"  \n\
echo "CUDA Available: $(python -c \"import torch; print(torch.cuda.is_available())\")"  \n\
\n\
if [ "$1" = "train" ]; then\n\
    echo "📚 Starting training..."\n\
    python scripts/train_on_wiki.py "${@:2}"\n\
elif [ "$1" = "tokenizer" ]; then\n\
    echo "🔤 Training tokenizer..."\n\
    python scripts/train_tokenizer_offline.py "${@:2}"\n\
elif [ "$1" = "prepare-data" ]; then\n\
    echo "📥 Preparing Wikipedia data..."\n\
    python scripts/prepare_wiki.py "${@:2}"\n\
elif [ "$1" = "inference" ]; then\n\
    echo "🤖 Running inference..."\n\
    python scripts/test_inference.py "${@:2}"\n\
elif [ "$1" = "test" ]; then\n\
    echo "✅ Running smoke tests..."\n\
    python scripts/run_smoke_test.py\n\
else\n\
    exec "$@"\n\
fi\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["train", "--help"]
