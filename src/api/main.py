"""FastAPI application — PM2.5 Forecasting API.

Endpoints:
    - /health              — Health check (DB + model status)
    - /api/v1/experiments  — Experiment CRUD
    - /api/v1/predict      — Model inference
    - /api/v1/audit        — Data/model hash audit

Usage:
    uv run uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.database import Base, engine
from src.api.routers import audit, content, experiments, inference
from src.api.schemas import HealthResponse

# ── Production Logging Config ──
# Log format: timestamp | level | request_id | message
logger.add(
    "research/logs/api_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    enqueue=True,  # Thread-safe for uvicorn workers
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Startup: create tables if they don't exist
    logger.info("API starting — initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    # Auto-seed info_cards if empty (first-run in Docker)
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM info_cards"))
            count = result.scalar()
        if count == 0:
            logger.info("info_cards table empty — running auto-seed...")
            from scripts.seed_info_cards import seed as seed_info_cards
            seed_info_cards()
            logger.info("Auto-seed complete.")
        else:
            logger.info(f"info_cards table has {count} rows — skipping seed.")
    except Exception as e:
        logger.warning(f"Auto-seed skipped (table may not exist yet): {e}")

    # Log model availability
    models_dir = Path("models/exported")
    if models_dir.exists():
        model_files = list(models_dir.glob("*"))
        logger.info(f"Model weights found: {len(model_files)} files in {models_dir}")
    else:
        logger.warning(f"Model directory not found: {models_dir}")

    yield
    # Shutdown
    logger.info("API shutting down gracefully.")


app = FastAPI(
    title="PM2.5 Forecasting API",
    description="Scientific forecasting pipeline for PM2.5 air quality prediction.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",    # Streamlit dev
        "http://dashboard:8501",   # Docker internal
        "http://localhost:3000",    # React dev (future)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Logging Middleware ──
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    t0 = time.time()
    response = await call_next(request)
    elapsed = (time.time() - t0) * 1000  # ms

    # Skip noisy health check logs in production
    if request.url.path != "/health":
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)"
        )

    return response


# ── Routers ──
app.include_router(experiments.router, prefix="/api/v1", tags=["experiments"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(audit.router, prefix="/api/v1", tags=["audit"])
app.include_router(content.router, prefix="/api/v1", tags=["content"])


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """API health check — verifies DB connectivity and model availability."""
    # Quick DB connectivity test
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    # Check model weights availability
    models_dir = Path("models/exported")
    model_count = len(list(models_dir.glob("*"))) if models_dir.exists() else 0

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        version="1.0.0",
        database=db_status,
        models_loaded=model_count,
    )
