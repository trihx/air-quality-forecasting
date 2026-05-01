"""Content management router — serves info cards from PostgreSQL.

Endpoints:
    GET /content/info-cards         — List all cards (filter by ?page=)
    GET /content/info-cards/{key}   — Get single card by key
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models import InfoCard

router = APIRouter(prefix="/content", tags=["content"])


# ── Response schemas ──


class InfoCardResponse(BaseModel):
    """Response schema for a single info card."""

    id: int
    card_key: str
    title: str
    content: str
    page: str
    display_order: int

    model_config = {"from_attributes": True}


class InfoCardUpdateRequest(BaseModel):
    """Schema for updating an info card."""

    title: str | None = None
    content: str | None = None


# ── Endpoints ──


@router.get("/info-cards", response_model=list[InfoCardResponse])
def list_info_cards(
    page: str | None = Query(None, description="Filter by page name (e.g., 'overview')"),
    db: Session = Depends(get_db),
) -> list[InfoCardResponse]:
    """List all info cards, optionally filtered by page."""
    stmt = select(InfoCard).order_by(InfoCard.page, InfoCard.display_order)
    if page:
        stmt = stmt.where(InfoCard.page == page)
    cards = db.scalars(stmt).all()
    return [InfoCardResponse.model_validate(c) for c in cards]


@router.get("/info-cards/{card_key}", response_model=InfoCardResponse)
def get_info_card(
    card_key: str,
    db: Session = Depends(get_db),
) -> InfoCardResponse:
    """Get a single info card by its unique key."""
    stmt = select(InfoCard).where(InfoCard.card_key == card_key)
    card = db.scalars(stmt).first()
    if not card:
        raise HTTPException(status_code=404, detail=f"Info card '{card_key}' not found")
    return InfoCardResponse.model_validate(card)


@router.put("/info-cards/{card_key}", response_model=InfoCardResponse)
def update_info_card(
    card_key: str,
    update_data: InfoCardUpdateRequest,
    db: Session = Depends(get_db),
) -> InfoCardResponse:
    """Update an info card's title or content."""
    stmt = select(InfoCard).where(InfoCard.card_key == card_key)
    card = db.scalars(stmt).first()
    if not card:
        raise HTTPException(status_code=404, detail=f"Info card '{card_key}' not found")
    
    if update_data.title is not None:
        card.title = update_data.title
    if update_data.content is not None:
        card.content = update_data.content
        
    db.commit()
    db.refresh(card)
    return InfoCardResponse.model_validate(card)
