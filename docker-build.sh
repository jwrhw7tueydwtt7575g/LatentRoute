#!/bin/bash

# ========================================
# Docker Build Script for LatentRoute
# ========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="${IMAGE_NAME:-latentroute}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"

# Print helper function
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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if nvidia-docker is available (for GPU support)
if ! command -v nvidia-docker &> /dev/null; then
    print_warning "nvidia-docker not found. GPU support may not work. Install nvidia-docker for GPU support."
fi

print_info "=========================================="
print_info "LatentRoute Docker Build Script"
print_info "=========================================="
print_info "Image Name: $IMAGE_NAME"
print_info "Image Tag: $IMAGE_TAG"
print_info "Registry: ${REGISTRY:-none}"

# Parse arguments
NO_CACHE=false
PUSH_REGISTRY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE=true
            print_info "Building without cache"
            shift
            ;;
        --push)
            PUSH_REGISTRY=true
            print_info "Will push to registry after build"
            shift
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    print_error "Dockerfile not found in current directory"
    exit 1
fi

print_info "Starting Docker build..."

# Build Docker image
BUILD_CMD="docker build"

if [ "$NO_CACHE" = true ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

BUILD_CMD="$BUILD_CMD -t ${IMAGE_NAME}:${IMAGE_TAG}"
BUILD_CMD="$BUILD_CMD -f Dockerfile"
BUILD_CMD="$BUILD_CMD ."

print_info "Executing: $BUILD_CMD"
eval $BUILD_CMD

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully!"
    print_info "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
else
    print_error "Docker build failed"
    exit 1
fi

# Tag for registry if specified
if [ ! -z "$REGISTRY" ]; then
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    print_info "Tagging image for registry: $FULL_IMAGE"
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} $FULL_IMAGE
    
    if [ "$PUSH_REGISTRY" = true ]; then
        print_info "Pushing to registry..."
        docker push $FULL_IMAGE
        if [ $? -eq 0 ]; then
            print_success "Image pushed to registry successfully!"
        else
            print_error "Failed to push image to registry"
            exit 1
        fi
    fi
fi

print_success "=========================================="
print_success "Build Complete!"
print_success "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run training: ./docker-run-train.sh"
echo "  2. Or use docker-compose: docker-compose up -d latentroute-train"
echo ""
