# ---- Builder stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[server]"

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

LABEL maintainer="NeuralMem Team"
LABEL description="NeuralMem — Local-first, MCP-native agent memory"
LABEL version="0.2.0"

# Install curl for healthcheck and create non-root user
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -r neuralmem \
 && useradd -r -g neuralmem -d /home/neuralmem -s /sbin/nologin neuralmem \
 && mkdir -p /home/neuralmem /data \
 && chown -R neuralmem:neuralmem /home/neuralmem /data

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV NEURALMEM_DB_PATH=/data/memory.db
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

USER neuralmem
WORKDIR /home/neuralmem

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["neuralmem", "mcp", "--http"]
CMD ["--port", "8080"]
