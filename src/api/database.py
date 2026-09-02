"""Database engine, session, and Base for SQLAlchemy 2.0.

Usage:
    from src.api.database import get_db, engine, Base

    # In FastAPI route:
    @app.get("/items")
    def list_items(db: Session = Depends(get_db)):
        ...

    # Init tables:
    Base.metadata.create_all(bind=engine)
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── Connection URL ──
# Dev default: SQLite fallback for local dev without Docker.
# Production: PostgreSQL via DATABASE_URL env var.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./research/experiments/forecasting.db",
)

# Fix postgres:// -> postgresql:// for SQLAlchemy 2.0 compatibility
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLAlchemy 2.0 engine
# - SQLite needs check_same_thread=False for FastAPI async
# - PostgreSQL uses a lightweight connection pool optimized for Render Free (512MB RAM) & Supabase Supavisor
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=2,          # Compact connection pool to conserve RAM & avoid Supabase pool exhaustion
        max_overflow=3,       # Allow burst of up to 3 extra connections during traffic spikes
        pool_recycle=300,     # Recycle connection every 5 minutes (prevents Supabase idle TCP timeout)
        pool_pre_ping=True,   # Verify connection liveness before checkout (avoids dropped socket errors)
        connect_args={
            "connect_timeout": 10,
        },
        echo=False,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session, auto-closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
