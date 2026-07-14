"""FastAPI dependencies — DB session, model cache."""

from __future__ import annotations

from src.api.database import get_db

# Re-export for convenient import in routers
__all__ = ["get_db"]
