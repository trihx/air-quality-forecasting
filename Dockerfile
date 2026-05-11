# ═══════════════════════════════════════════
# PM2.5 Forecasting — Dockerfile (UV-based)
# Multi-stage build: Python 3.11 + PyTorch CPU
# Shared base image for both API and Dashboard
# ═══════════════════════════════════════════

# --- Stage 1: Builder (UV for fast installs) ---
FROM python:3.11-slim AS builder
# IMPORTANT: Use /app as WORKDIR so UV writes shebangs as
# #!/app/.venv/bin/python — matching the runtime stage path.
# Using a different path (e.g. /build) causes "no such file" errors.
WORKDIR /app

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies with UV (much faster than pip)
# --frozen: use exact lockfile versions
# --no-dev: skip dev dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Remove massive CUDA/NVIDIA binaries that pip installs by default on Linux
# This alone saves ~4-5GB of space
RUN rm -rf /app/.venv/lib/python*/site-packages/nvidia* && \
    rm -rf /app/.venv/lib/python*/site-packages/triton*

# PyTorch CPU-only overlay (saves ~1.4GB vs full PyTorch)
RUN uv pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Strip .so files to save more space
RUN find /app/.venv -name "*.so" -exec strip {} \; || true

# --- Stage 2: Runtime ---
FROM python:3.11-slim
WORKDIR /app

# Non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

# Copy venv from builder (paths match: /app/.venv → /app/.venv)
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Copy application files (filtered by .dockerignore)
COPY . .

# Ensure writable dirs for runtime
RUN mkdir -p /app/models/user_trained \
             /app/research/experiments/dashboard_runs \
             /app/.chroma_db && \
    chown -R appuser:appgroup /app

USER appuser

# Ports: 8501 (Streamlit) and 8000 (FastAPI)
EXPOSE 8501 8000

# Health check — overridden per service in docker-compose.yml
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Default: run Streamlit dashboard
# Override in docker-compose.yml for API service
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
