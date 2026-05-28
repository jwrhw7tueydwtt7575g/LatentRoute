#!/bin/bash

# ========================================
# Docker Compose Script for LatentRoute
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
Usage: $0 [COMMAND] [OPTIONS]

COMMANDS:
    build              Build Docker image
    up                 Start training container
    down               Stop and remove containers
    logs               Show container logs
    shell              Open interactive bash in running container
    push               Push image to registry
    clean              Remove image and volumes

OPTIONS (for up):
    --gpus NUM         Number of GPUs (default: 1)
    --compose-file     Path to docker-compose.yml (default: docker-compose.yml)

EXAMPLES:
    ./docker-compose-runner.sh build
    ./docker-compose-runner.sh up --gpus 4
    ./docker-compose-runner.sh logs
    ./docker-compose-runner.sh shell
    ./docker-compose-runner.sh down
EOF
}

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose not installed. Install it with: sudo apt install docker-compose"
    exit 1
fi

# Check if .env.docker exists
if [ ! -f ".env.docker" ]; then
    print_warning ".env.docker not found, creating from template..."
    # Create basic .env.docker if it doesn't exist
    touch .env.docker
fi

COMMAND="${1:-help}"
GPUS="${3:-1}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

case $COMMAND in
    build)
        print_info "Building Docker image..."
        docker-compose -f $COMPOSE_FILE build
        print_success "Build complete!"
        ;;
    
    up)
        print_info "Starting training container..."
        # Create necessary directories
        mkdir -p models data logs
        
        # Override GPU count if specified
        if [ ! -z "$GPUS" ] && [ "$GPUS" != "1" ]; then
            print_info "Setting GPU count to: $GPUS"
            export NVIDIA_VISIBLE_DEVICES=$GPUS
        fi
        
        docker-compose -f $COMPOSE_FILE up -d latentroute-train
        print_success "Container started!"
        print_info "View logs: docker-compose -f $COMPOSE_FILE logs -f"
        ;;
    
    down)
        print_info "Stopping and removing containers..."
        docker-compose -f $COMPOSE_FILE down
        print_success "Containers stopped!"
        ;;
    
    logs)
        print_info "Showing logs..."
        docker-compose -f $COMPOSE_FILE logs -f latentroute-train
        ;;
    
    shell)
        CONTAINER_ID=$(docker-compose -f $COMPOSE_FILE ps -q latentroute-train)
        if [ -z "$CONTAINER_ID" ]; then
            print_error "Container not running. Start it with: $0 up"
            exit 1
        fi
        print_info "Opening shell in container..."
        docker exec -it $CONTAINER_ID /bin/bash
        ;;
    
    push)
        print_info "Pushing image to registry..."
        docker-compose -f $COMPOSE_FILE push
        print_success "Image pushed!"
        ;;
    
    clean)
        print_warning "Removing Docker image and volumes..."
        docker-compose -f $COMPOSE_FILE down -v
        docker rmi latentroute:latest
        print_success "Cleaned up!"
        ;;
    
    ps|status)
        print_info "Container status:"
        docker-compose -f $COMPOSE_FILE ps
        ;;
    
    help|*)
        show_usage
        ;;
esac
