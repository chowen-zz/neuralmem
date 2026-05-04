# =============================================================================
# NeuralMem V1.6 — Multi-stage Dockerfile
# =============================================================================
# Build stages:
#   1. builder      — Compile Python deps, install package with [server] extras
#   2. dashboard    — (optional) Build frontend assets if needed
#   3. runtime      — Slim final image with app + dashboard + health checks
# =============================================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libsqlite3-dev \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

# Install package with server extras into a dedicated target directory
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install ".[server]"

# ---- Stage 2: Dashboard asset builder (optional) ----
FROM node:20-slim AS dashboard-builder

WORKDIR /dashboard

# Copy dashboard static files if they exist; otherwise create minimal placeholder
COPY src/neuralmem/dashboard/static/ ./static/ 2>/dev/null || true

# If a package.json exists in dashboard, build it
RUN if [ -f package.json ]; then npm ci && npm run build; fi

# ---- Stage 3: Runtime (slim final image) ----
FROM python:3.11-slim AS runtime

LABEL maintainer="NeuralMem Team <team@neuralmem.dev>"
LABEL description="NeuralMem V1.6 — Local-first, MCP-native agent memory with Dashboard"
LABEL version="1.6.0"
LABEL org.opencontainers.image.source="https://github.com/neuralmem/neuralmem"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Install runtime dependencies: curl for healthcheck, sqlite3 for CLI ops
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl sqlite3 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -r neuralmem \
 && useradd -r -g neuralmem -d /app -s /sbin/nologin neuralmem \
 && mkdir -p /app /data /app/dashboard/static \
 && chown -R neuralmem:neuralmem /app /data

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source (needed for dashboard static files and any runtime imports)
COPY --chown=neuralmem:neuralmem src/neuralmem/dashboard/static/ /app/dashboard/static/
COPY --chown=neuralmem:neuralmem docker/entrypoint.sh /app/entrypoint.sh

# Ensure scripts are executable and packages are importable
RUN chmod +x /app/entrypoint.sh \
 && python -c "import neuralmem; print('NeuralMem import OK')"

# Environment configuration
ENV NEURALMEM_DB_PATH=/data/memory.db
ENV NEURALMEM_DASHBOARD_STATIC_DIR=/app/dashboard/static
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080
ENV DASHBOARD_PORT=8081

# Expose both MCP HTTP port and Dashboard port
EXPOSE 8080 8081

USER neuralmem
WORKDIR /app

# Health check — hits the dashboard health endpoint (also covers MCP via shared mem)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:${DASHBOARD_PORT:-8081}/api/health >/dev/null \
         || curl -fsS http://localhost:${PORT:-8080}/health >/dev/null \
         || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--mode", "both"]
