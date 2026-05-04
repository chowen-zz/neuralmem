#!/bin/bash
# =============================================================================
# NeuralMem V1.6 — Container Entrypoint Script
# =============================================================================
# Modes:
#   mcp       — Start MCP server on PORT (default 8080)
#   dashboard — Start Dashboard server on DASHBOARD_PORT (default 8081)
#   both      — Start both services (dashboard in background, MCP in foreground)
# =============================================================================

set -euo pipefail

MODE="${MODE:-both}"
PORT="${PORT:-8080}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8081}"
DB_PATH="${NEURALMEM_DB_PATH:-/data/memory.db}"
LOG_LEVEL="${NEURALMEM_LOG_LEVEL:-INFO}"

# Ensure data directory exists
mkdir -p "$(dirname "$DB_PATH")"
mkdir -p /app/logs

log() {
    echo "[neuralmem-entrypoint] $(date -Iseconds) — $*"
}

start_mcp() {
    log "Starting NeuralMem MCP server on port $PORT ..."
    exec neuralmem mcp --http --port "$PORT"
}

start_dashboard() {
    log "Starting NeuralMem Dashboard server on port $DASHBOARD_PORT ..."
    # The dashboard server is started via Python module
    exec python -m neuralmem.dashboard.server \
        --host 0.0.0.0 \
        --port "$DASHBOARD_PORT" \
        --db-path "$DB_PATH"
}

start_both() {
    log "Starting NeuralMem in dual-mode (MCP + Dashboard) ..."

    # Start dashboard in background
    python -m neuralmem.dashboard.server \
        --host 0.0.0.0 \
        --port "$DASHBOARD_PORT" \
        --db-path "$DB_PATH" \
        > /app/logs/dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    log "Dashboard PID: $DASHBOARD_PID"

    # Wait briefly for dashboard to bind
    sleep 2

    # Start MCP server in foreground (this keeps the container alive)
    log "Handing over to MCP server on port $PORT ..."
    exec neuralmem mcp --http --port "$PORT"
}

# Parse any CLI arguments passed to the entrypoint
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --dashboard-port)
            DASHBOARD_PORT="$2"
            shift 2
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
NeuralMem Entrypoint — Usage:
  --mode {mcp|dashboard|both}   Service mode to start (default: both)
  --port PORT                   MCP server port (default: 8080)
  --dashboard-port PORT         Dashboard port (default: 8081)
  --db-path PATH                SQLite database path (default: /data/memory.db)
  --log-level LEVEL             Logging level (default: INFO)
EOF
            exit 0
            ;;
        *)
            log "Unknown argument: $1"
            exit 1
            ;;
    esac
done

log "MODE=$MODE | PORT=$PORT | DASHBOARD_PORT=$DASHBOARD_PORT | DB=$DB_PATH"

case "$MODE" in
    mcp)
        start_mcp
        ;;
    dashboard)
        start_dashboard
        ;;
    both)
        start_both
        ;;
    *)
        log "Unknown mode: $MODE. Use --mode {mcp|dashboard|both}"
        exit 1
        ;;
esac
