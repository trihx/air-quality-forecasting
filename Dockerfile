# ═══════════════════════════════════════════
# PM2.5 Forecasting — Dockerfile for Production
# Runs Streamlit (port 7860) + FastAPI (port 8000) via supervisord
# Compatible with Render.com & HuggingFace Spaces
# ═══════════════════════════════════════════

# --- Stage 1: Builder (UV for fast installs) ---
FROM python:3.11-slim AS builder
WORKDIR /app

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies with UV
RUN uv sync --frozen --no-dev --no-install-project

# Remove massive CUDA/NVIDIA binaries (not needed on CPU)
RUN rm -rf /app/.venv/lib/python*/site-packages/nvidia* && \
    rm -rf /app/.venv/lib/python*/site-packages/triton*

# PyTorch CPU-only overlay (saves ~1.4GB vs full PyTorch)
RUN uv pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Strip .so files to save space
RUN find /app/.venv -name "*.so" -exec strip {} \; || true

# --- Stage 2: Runtime ---
FROM python:3.11-slim
WORKDIR /app

# Install supervisord, libgomp1 (OpenMP for LightGBM), libpq5 (PostgreSQL) and curl (Healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor libgomp1 libpq5 curl && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Copy application files
COPY . .

# Ensure writable dirs for runtime
RUN mkdir -p /app/models/user_trained \
             /app/research/experiments/dashboard_runs \
             /app/research/logs \
             /app/.chroma_db

# Supervisord config: run both Streamlit + FastAPI
RUN cat > /etc/supervisor/conf.d/app.conf << 'EOF'
[supervisord]
nodaemon=true
logfile=/dev/null
logfile_maxbytes=0

[program:fastapi]
command=uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level warning
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0

[program:streamlit]
command=sh -c 'streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --server.maxUploadSize=50'
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0
EOF

# Expose default port (Render will override via $PORT)
EXPOSE 7860 8000

CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
