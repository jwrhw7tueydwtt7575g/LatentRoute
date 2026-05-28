#!/bin/bash

# ========================================
# Docker Run Script for LatentRoute Training
# ========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
IMAGE_NAME="${IMAGE_NAME:-latentroute}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-latentroute-train}"
COMMAND="${COMMAND:-train}"

# Print helper functions
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Parse arguments
GPUS="1"
BATCH_SIZE=4
TOTAL_STEPS=1000
LR=0.0001
INTERACTIVE=false
DETACH=false
VOLUME_MODELS="./models:/models"
VOLUME_DATA="./data:/data"
EXTRA_ARGS=""

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -g, --gpus NUM          Number of GPUs (default: 1)
    -b, --batch-size NUM    Batch size (default: 4)
    -s, --steps NUM         Total training steps (default: 1000)
    -l, --learning-rate LR  Learning rate (default: 0.0001)
    -i, --interactive       Run interactive bash session
    -d, --detach            Run container in background
    -n, --name NAME         Container name (default: latentroute-train)
    --models-dir PATH       Mount point for models (default: ./models:/models)
    --data-dir PATH         Mount point for data (default: ./data:/data)
    --image IMAGE           Docker image (default: latentroute:latest)

COMMANDS:
    train          Train model (default)
    tokenizer      Train BPE tokenizer
    prepare-data   Download and prepare Wikipedia data
    test           Run smoke tests
    inference      Test model inference

EXAMPLES:
    # Train with defaults
    ./docker-run-train.sh

    # Train with 2 GPUs and 5000 steps
    ./docker-run-train.sh -g 2 -s 5000

    # Interactive shell
    ./docker-run-train.sh --interactive

    # Train tokenizer
    ./docker-run-train.sh tokenizer

    # Run background training (detached)
    ./docker-run-train.sh --detach
EOF
}

# Parse command
COMMAND_ARG=""
if [[ $1 == "train" || $1 == "tokenizer" || $1 == "prepare-data" || $1 == "test" || $1 == "inference" ]]; then
    COMMAND="$1"
    shift
fi

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -s|--steps)
            TOTAL_STEPS="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LR="$2"
            shift 2
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --models-dir)
            VOLUME_MODELS="$2"
            shift 2
            ;;
        --data-dir)
            VOLUME_DATA="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME=$(echo "$2" | cut -d: -f1)
            IMAGE_TAG=$(echo "$2" | cut -d: -f2)
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Verify image exists
if ! docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} &> /dev/null; then
    print_error "Docker image ${IMAGE_NAME}:${IMAGE_TAG} not found"
    print_info "Build the image first: ./docker-build.sh"
    exit 1
fi

# Create necessary directories
mkdir -p models data logs

print_info "=========================================="
print_info "LatentRoute Training Container"
print_info "=========================================="
print_info "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
print_info "Container: $CONTAINER_NAME"
print_info "Command: $COMMAND"
print_info "GPUs: $GPUS"
print_info "Batch Size: $BATCH_SIZE"
print_info "Total Steps: $TOTAL_STEPS"
print_info "Learning Rate: $LR"

# Build docker run command
RUN_CMD="docker run"

# Add flags
if [ "$DETACH" = true ]; then
    RUN_CMD="$RUN_CMD -d"
    print_info "Running in background (detached mode)"
fi

if [ "$INTERACTIVE" = true ]; then
    RUN_CMD="$RUN_CMD -it"
    COMMAND="bash"
    print_info "Starting interactive bash session"
else
    RUN_CMD="$RUN_CMD -it"
fi

# Add GPU support
RUN_CMD="$RUN_CMD --gpus ${GPUS}"

# Add container name
RUN_CMD="$RUN_CMD --name $CONTAINER_NAME"

# Add volumes
RUN_CMD="$RUN_CMD -v $VOLUME_MODELS"
RUN_CMD="$RUN_CMD -v $VOLUME_DATA"
RUN_CMD="$RUN_CMD -v $(pwd)/logs:/workspace/logs"

# Add environment variables
RUN_CMD="$RUN_CMD -e CUDA_VISIBLE_DEVICES=0"
RUN_CMD="$RUN_CMD -e PYTHONUNBUFFERED=1"
RUN_CMD="$RUN_CMD -e TOTAL_STEPS=$TOTAL_STEPS"
RUN_CMD="$RUN_CMD -e BATCH_SIZE=$BATCH_SIZE"
RUN_CMD="$RUN_CMD -e LEARNING_RATE=$LR"

# Network
RUN_CMD="$RUN_CMD --network latentroute-network"

# Resource limits
RUN_CMD="$RUN_CMD --memory=120g"

# Image and command
RUN_CMD="$RUN_CMD ${IMAGE_NAME}:${IMAGE_TAG}"
RUN_CMD="$RUN_CMD $COMMAND"

# Add extra arguments (training hyperparameters)
if [ ! -z "$EXTRA_ARGS" ]; then
    RUN_CMD="$RUN_CMD $EXTRA_ARGS"
else
    # Add default training arguments
    if [ "$COMMAND" = "train" ]; then
        RUN_CMD="$RUN_CMD --total_steps $TOTAL_STEPS"
        RUN_CMD="$RUN_CMD --batch_size $BATCH_SIZE"
        RUN_CMD="$RUN_CMD --lr $LR"
    fi
fi

print_info "Executing: $RUN_CMD"
echo ""

# Remove existing container with same name if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    print_warning "Removing existing container: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME > /dev/null 2>&1 || true
fi

# Run the container
eval $RUN_CMD

CONTAINER_ID=$(docker ps -aqf "name=${CONTAINER_NAME}" | head -1)

if [ ! -z "$CONTAINER_ID" ]; then
    print_success "Container started successfully!"
    print_info "Container ID: $CONTAINER_ID"
    print_info "View logs: docker logs -f $CONTAINER_ID"
    
    if [ "$DETACH" = false ] && [ "$INTERACTIVE" = false ]; then
        print_info "Waiting for container to finish..."
        docker wait $CONTAINER_ID
        print_success "Training completed!"
        print_info "Checkpoints saved to: ./models/"
    fi
else
    print_error "Failed to start container"
    exit 1
fi

print_success "=========================================="
