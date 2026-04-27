# ═══════════════════════════════════════════
# PM2.5 Forecasting Dashboard — Dockerfile
# Multi-stage build: Python 3.11 + PyTorch CPU
# Image: trihx/pm25-forecasting
# ═══════════════════════════════════════════

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder
WORKDIR /build

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch CPU-only (saves ~1.4GB vs full PyTorch)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir . 2>/dev/null || \
    pip install --no-cache-dir \
    streamlit openai chromadb sentence-transformers \
    lightgbm xgboost scikit-learn \
    pandas numpy scipy statsmodels \
    plotly matplotlib seaborn \
    pmdarima optuna loguru pyyaml watchdog shap pymupdf

# --- Stage 2: Runtime ---
FROM python:3.11-slim
WORKDIR /app

# Non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Copy application files (filtered by .dockerignore)
COPY . .

# Ensure writable dirs for runtime
RUN mkdir -p /app/models/user_trained /app/research/experiments/dashboard_runs && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
