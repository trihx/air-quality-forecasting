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

# SQLAlchemy 2.0 engine
# - SQLite needs check_same_thread=False for FastAPI async
# - PostgreSQL uses pool with sensible defaults
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
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
