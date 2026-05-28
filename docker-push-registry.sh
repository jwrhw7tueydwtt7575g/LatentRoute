#!/bin/bash

# ========================================
# Docker Registry Push Script for LatentRoute
# ========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -r, --registry URL      Docker registry URL (required)
    -u, --username USER     Docker registry username
    -p, --password PASS     Docker registry password
    -i, --image IMAGE       Docker image name (default: latentroute)
    -t, --tag TAG           Docker image tag (default: latest)
    --docker-hub            Push to Docker Hub
    --aws-ecr REPO_URI      Push to AWS ECR
    --gcp-gcr PROJECT       Push to GCP GCR
    --azure ACR_NAME        Push to Azure ACR
    --login-only            Only login to registry (don't push)
    -h, --help              Show this help message

EXAMPLES:
    # Push to Docker Hub
    ./docker-push-registry.sh --docker-hub -u myusername

    # Push to AWS ECR
    ./docker-push-registry.sh --aws-ecr 123456789.dkr.ecr.us-east-1.amazonaws.com/latentroute

    # Push to custom registry
    ./docker-push-registry.sh --registry registry.example.com -u user -p password

    # GCP GCR
    ./docker-push-registry.sh --gcp-gcr my-project

    # Azure ACR
    ./docker-push-registry.sh --azure myregistryname
EOF
}

REGISTRY=""
USERNAME=""
PASSWORD=""
IMAGE_NAME="latentroute"
IMAGE_TAG="latest"
LOGIN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -u|--username)
            USERNAME="$2"
            shift 2
            ;;
        -p|--password)
            PASSWORD="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --docker-hub)
            REGISTRY="docker.io"
            print_info "Using Docker Hub registry"
            shift
            ;;
        --aws-ecr)
            REGISTRY="$2"
            print_info "Using AWS ECR: $REGISTRY"
            shift 2
            ;;
        --gcp-gcr)
            REGISTRY="gcr.io/$2"
            print_info "Using GCP GCR: $REGISTRY"
            shift 2
            ;;
        --azure)
            REGISTRY="${2}.azurecr.io"
            print_info "Using Azure ACR: $REGISTRY"
            shift 2
            ;;
        --login-only)
            LOGIN_ONLY=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate registry
if [ -z "$REGISTRY" ]; then
    print_error "Registry URL not specified"
    show_usage
    exit 1
fi

print_info "=========================================="
print_info "Docker Registry Push Script"
print_info "=========================================="
print_info "Registry: $REGISTRY"
print_info "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Docker Hub login
if [ "$REGISTRY" = "docker.io" ]; then
    if [ -z "$USERNAME" ]; then
        print_info "Enter Docker Hub username:"
        read -p "> " USERNAME
    fi
    
    print_info "Logging in to Docker Hub..."
    docker login --username "$USERNAME"
    
    if [ "$LOGIN_ONLY" = true ]; then
        print_success "Logged in successfully!"
        exit 0
    fi
    
    FULL_IMAGE="${USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# AWS ECR
elif [[ "$REGISTRY" == *.dkr.ecr.*.amazonaws.com ]]; then
    print_info "Logging in to AWS ECR..."
    REGION=$(echo "$REGISTRY" | awk -F. '{print $4}')
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY
    
    if [ "$LOGIN_ONLY" = true ]; then
        print_success "Logged in successfully!"
        exit 0
    fi
    
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# GCP GCR
elif [[ "$REGISTRY" == gcr.io/* ]]; then
    print_info "Logging in to GCP GCR..."
    gcloud auth configure-docker
    
    if [ "$LOGIN_ONLY" = true ]; then
        print_success "Logged in successfully!"
        exit 0
    fi
    
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Azure ACR
elif [[ "$REGISTRY" == *.azurecr.io ]]; then
    print_info "Logging in to Azure ACR..."
    az acr login --name $(echo "$REGISTRY" | cut -d. -f1)
    
    if [ "$LOGIN_ONLY" = true ]; then
        print_success "Logged in successfully!"
        exit 0
    fi
    
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Generic registry
else
    if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
        print_error "Username and password required for custom registry"
        show_usage
        exit 1
    fi
    
    print_info "Logging in to $REGISTRY..."
    echo "$PASSWORD" | docker login -u "$USERNAME" --password-stdin "$REGISTRY"
    
    if [ "$LOGIN_ONLY" = true ]; then
        print_success "Logged in successfully!"
        exit 0
    fi
    
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

# Check if image exists locally
if ! docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} &> /dev/null; then
    print_error "Docker image ${IMAGE_NAME}:${IMAGE_TAG} not found locally"
    print_info "Build it first: ./docker-build.sh"
    exit 1
fi

# Tag image for registry
print_info "Tagging image: ${IMAGE_NAME}:${IMAGE_TAG} -> $FULL_IMAGE"
docker tag ${IMAGE_NAME}:${IMAGE_TAG} $FULL_IMAGE

# Push image
print_info "Pushing image to registry..."
print_info "This may take a few minutes..."
docker push $FULL_IMAGE

if [ $? -eq 0 ]; then
    print_success "=========================================="
    print_success "Image pushed successfully!"
    print_success "=========================================="
    print_info "Image: $FULL_IMAGE"
    print_info ""
    echo "To pull and run on another server:"
    echo "  docker pull $FULL_IMAGE"
    echo "  ./docker-run-train.sh --image $FULL_IMAGE"
else
    print_error "Failed to push image"
    exit 1
fi
