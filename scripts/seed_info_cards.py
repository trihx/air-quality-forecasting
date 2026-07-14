import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from sqlalchemy import select
from src.api.database import SessionLocal, engine, Base
from src.api.models import InfoCard
from scripts.seed_data import INFO_CARDS

def seed():
    print("Initializing database connection...")
    Base.metadata.create_all(bind=engine)
    
    with SessionLocal() as db:
        for card_data in INFO_CARDS:
            stmt = select(InfoCard).where(InfoCard.card_key == card_data["card_key"])
            existing = db.scalars(stmt).first()
            if existing:
                existing.title = card_data["title"]
                existing.content = card_data["content"]
                existing.page = card_data["page"]
                existing.display_order = card_data.get("display_order", 0)
                print(f"Updated: {card_data['card_key']}")
            else:
                new_card = InfoCard(**card_data)
                db.add(new_card)
                print(f"Inserted: {card_data['card_key']}")
        db.commit()
        print(f"\\nSeeded {len(INFO_CARDS)} info cards successfully.")

if __name__ == "__main__":
    seed()
