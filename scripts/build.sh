#!/bin/bash
# =============================================================================
# NeuralMem V1.6 — Docker Build Script
# =============================================================================
# Usage:
#   ./scripts/build.sh              — Build runtime image only
#   ./scripts/build.sh --all        — Build all stages + run tests
#   ./scripts/build.sh --push       — Build and push to registry
#   ./scripts/build.sh --tag v1.6.0 — Override version tag
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Defaults
TAG="v1.6.0"
REGISTRY=""
PUSH=false
BUILD_ALL=false
PLATFORMS="linux/amd64,linux/arm64"
NO_CACHE=""
VERBOSE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[build]${NC} $*"; }
warn() { echo -e "${YELLOW}[build]${NC} $*"; }
err() { echo -e "${RED}[build]${NC} $*" >&2; }

usage() {
    cat <<EOF
NeuralMem Docker Build Script

Usage: $0 [OPTIONS]

Options:
  --tag TAG         Image tag (default: v1.6.0)
  --registry URL    Docker registry prefix, e.g. ghcr.io/neuralmem
  --push            Push image after build
  --all             Build all stages and run tests
  --platforms       Target platforms (default: linux/amd64,linux/arm64)
  --no-cache        Disable Docker build cache
  --verbose         Verbose Docker build output
  -h, --help        Show this help message

Examples:
  $0                          # Build neuralmem:v1.6.0 locally
  $0 --tag dev --no-cache     # Fresh build with 'dev' tag
  $0 --push --registry ghcr.io/neuralmem  # Build and push to GHCR
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --all)
            BUILD_ALL=true
            shift
            ;;
        --platforms)
            PLATFORMS="$2"
            shift 2
            ;;
            --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

IMAGE_NAME="${REGISTRY:+${REGISTRY}/}neuralmem"
FULL_TAG="${IMAGE_NAME}:${TAG}"
LATEST_TAG="${IMAGE_NAME}:latest"

log "Project root: $PROJECT_ROOT"
log "Docker context: $DOCKER_DIR"
log "Target image: $FULL_TAG"

# Check Docker is available
if ! command -v docker >/dev/null 2>&1; then
    err "Docker is not installed or not in PATH"
    exit 1
fi

# Check docker buildx for multi-platform (optional)
if docker buildx version >/dev/null 2>&1; then
    BUILDER="docker buildx build"
    BUILDX_OPTS="--platform $PLATFORMS"
else
    warn "docker buildx not available — falling back to 'docker build' (single platform only)"
    BUILDER="docker build"
    BUILDX_OPTS=""
fi

# ---------------------------------------------------------------------------
# Build stages
# ---------------------------------------------------------------------------

build_stage() {
    local stage_name="$1"
    local target_tag="$2"
    local extra_args="${3:-}"

    log "Building stage: $stage_name → $target_tag"

    local cache_args=""
    if [[ -n "$NO_CACHE" ]]; then
        cache_args="--no-cache"
    fi

    local progress_arg=""
    if [[ "$VERBOSE" == true ]]; then
        progress_arg="--progress=plain"
    fi

    if [[ "$BUILDER" == "docker buildx build" && "$PUSH" == true ]]; then
        # Buildx with push
        $BUILDER \
            $BUILDX_OPTS \
            --file "$DOCKER_DIR/Dockerfile" \
            --target "$stage_name" \
            --tag "$target_tag" \
            $cache_args \
            $progress_arg \
            $extra_args \
            --push \
            "$PROJECT_ROOT"
    else
        # Local build
        $BUILDER \
            $BUILDX_OPTS \
            --file "$DOCKER_DIR/Dockerfile" \
            --target "$stage_name" \
            --tag "$target_tag" \
            $cache_args \
            $progress_arg \
            $extra_args \
            "$PROJECT_ROOT"
    fi

    log "Stage $stage_name built successfully"
}

# Build builder stage (intermediate, not tagged)
if [[ "$BUILD_ALL" == true ]]; then
    log "=== Building all stages ==="
    build_stage "builder" "${IMAGE_NAME}:builder-${TAG}" ""
    build_stage "dashboard-builder" "${IMAGE_NAME}:dashboard-builder-${TAG}" ""
fi

# Build runtime stage (the production image)
log "=== Building runtime image ==="
build_stage "runtime" "$FULL_TAG" ""

# Tag as latest
if [[ "$TAG" != "latest" ]]; then
    log "Tagging $FULL_TAG as $LATEST_TAG"
    docker tag "$FULL_TAG" "$LATEST_TAG"
fi

# ---------------------------------------------------------------------------
# Run smoke tests on the built image
# ---------------------------------------------------------------------------

run_tests() {
    log "=== Running Docker smoke tests ==="

    local test_container="neuralmem-smoke-test"

    # Clean up any previous test container
    docker rm -f "$test_container" >/dev/null 2>&1 || true

    # Start container in background
    docker run -d \
        --name "$test_container" \
        -p 18080:8080 \
        -p 18081:8081 \
        -e MODE=both \
        "$FULL_TAG" \
        >/dev/null

    log "Waiting for services to start ..."
    sleep 8

    # Test MCP health endpoint
    local mcp_ok=false
    for i in {1..10}; do
        if curl -fsS http://localhost:18080/health >/dev/null 2>&1; then
            mcp_ok=true
            break
        fi
        sleep 1
    done

    # Test dashboard health endpoint
    local dash_ok=false
    for i in {1..10}; do
        if curl -fsS http://localhost:18081/api/health >/dev/null 2>&1; then
            dash_ok=true
            break
        fi
        sleep 1
    done

    # Cleanup
    docker rm -f "$test_container" >/dev/null 2>&1 || true

    if [[ "$mcp_ok" == true && "$dash_ok" == true ]]; then
        log "Smoke tests PASSED — MCP and Dashboard both healthy"
    else
        err "Smoke tests FAILED"
        [[ "$mcp_ok" != true ]] && err "  - MCP health check failed"
        [[ "$dash_ok" != true ]] && err "  - Dashboard health check failed"
        exit 1
    fi
}

if [[ "$BUILD_ALL" == true ]]; then
    run_tests
fi

# ---------------------------------------------------------------------------
# Push to registry
# ---------------------------------------------------------------------------

if [[ "$PUSH" == true ]]; then
    if [[ -z "$REGISTRY" ]]; then
        err "--push requires --registry to be set"
        exit 1
    fi

    log "=== Pushing images to registry ==="
    docker push "$FULL_TAG"
    docker push "$LATEST_TAG"
    log "Pushed: $FULL_TAG and $LATEST_TAG"
fi

log "=== Build complete ==="
log "Image: $FULL_TAG"
[[ "$TAG" != "latest" ]] && log "Also tagged: $LATEST_TAG"

# Print quick-start commands
cat <<EOF

Quick start:
  docker run -p 8080:8080 -p 8081:8081 -v neuralmem-data:/data $FULL_TAG

Or with docker-compose:
  docker compose -f docker/docker-compose.yml up -d

With optional Weaviate vector DB:
  docker compose -f docker/docker-compose.yml --profile weaviate up -d

With optional ChromaDB vector DB:
  docker compose -f docker/docker-compose.yml --profile chroma up -d

Full stack (app + dashboard + all vector DBs):
  docker compose -f docker/docker-compose.yml --profile full up -d
EOF
