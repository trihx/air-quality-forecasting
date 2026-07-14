"""Initialize database — create all tables.

Usage:
    # From project root:
    uv run python -m src.api.init_db

    # Or import and call:
    from src.api.init_db import init_database
    init_database()
"""

from __future__ import annotations

from loguru import logger

from src.api.database import Base, engine

# Import all models so Base.metadata knows about them
from src.api.models import (  # noqa: F401
    Experiment,
    FeatureImportance,
    Metric,
    Run,
    RunModel,
)


def init_database() -> None:
    """Create all tables if they don't exist."""
    logger.info(f"Initializing database: {engine.url}")
    Base.metadata.create_all(bind=engine)
    table_names = list(Base.metadata.tables.keys())
    logger.info(f"Tables ready: {table_names}")


if __name__ == "__main__":
    init_database()
    print("✅ Database initialized successfully!", flush=True)
