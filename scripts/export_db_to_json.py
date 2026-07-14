"""Export all PostgreSQL tables to JSON files for local dev fallback.

Usage:
    # On Docker (with PostgreSQL):
    docker compose exec api python scripts/export_db_to_json.py

    # On local (with SQLite fallback):
    uv run python scripts/export_db_to_json.py

Output directory: research/experiments/db_export/
Files created:
    - info_cards.json        (key-value by card_key)
    - experiments.json       (list of experiments)
    - runs.json              (list of runs)
    - run_models.json        (list of run models)
    - metrics.json           (list of metrics)
    - feature_importances.json (list of feature importances)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from sqlalchemy import select
from src.api.database import SessionLocal, engine, Base
from src.api.models import (
    Experiment,
    FeatureImportance,
    InfoCard,
    Metric,
    Run,
    RunModel,
)

OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "db_export"


def _serialize_datetime(obj):
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _write_json(data, filename: str) -> None:
    """Write data to JSON file with proper encoding."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_serialize_datetime)
    print(f"  [OK] {filename}: {_count_label(data)}")


def _count_label(data) -> str:
    """Human-readable count label."""
    if isinstance(data, dict):
        return f"{len(data)} entries"
    if isinstance(data, list):
        return f"{len(data)} records"
    return "exported"


def export_info_cards(db) -> None:
    """Export info_cards table as key-value dict (by card_key)."""
    result = db.execute(
        select(InfoCard).order_by(InfoCard.page, InfoCard.display_order)
    )
    cards = result.scalars().all()

    data = {}
    for c in cards:
        data[c.card_key] = {
            "title": c.title,
            "content": c.content,
            "page": c.page,
            "display_order": c.display_order,
        }

    _write_json(data, "info_cards.json")


def export_experiments(db) -> None:
    """Export experiments table."""
    result = db.execute(select(Experiment).order_by(Experiment.id))
    rows = result.scalars().all()

    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "pipeline_version": r.pipeline_version,
            "data_hash_md5": r.data_hash_md5,
            "config": r.config,
            "created_at": r.created_at,
        })

    _write_json(data, "experiments.json")


def export_runs(db) -> None:
    """Export runs table."""
    result = db.execute(select(Run).order_by(Run.id))
    rows = result.scalars().all()

    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "experiment_id": r.experiment_id,
            "horizon": r.horizon,
            "status": r.status,
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "created_at": r.created_at,
        })

    _write_json(data, "runs.json")


def export_run_models(db) -> None:
    """Export run_models table."""
    result = db.execute(select(RunModel).order_by(RunModel.id))
    rows = result.scalars().all()

    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "run_id": r.run_id,
            "model_name": r.model_name,
            "training_time_s": r.training_time_s,
            "hyperparameters": r.hyperparameters,
            "feature_set": r.feature_set,
            "weight_hash_md5": r.weight_hash_md5,
            "weight_path": r.weight_path,
            "created_at": r.created_at,
        })

    _write_json(data, "run_models.json")


def export_metrics(db) -> None:
    """Export metrics table."""
    result = db.execute(select(Metric).order_by(Metric.id))
    rows = result.scalars().all()

    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "run_model_id": r.run_model_id,
            "mae": r.mae,
            "rmse": r.rmse,
            "mase": r.mase,
            "r2": r.r2,
            "mape": r.mape,
            "smape": r.smape,
            "forecast_bias": r.forecast_bias,
            "medae": r.medae,
            "extra_metrics": r.extra_metrics,
            "created_at": r.created_at,
        })

    _write_json(data, "metrics.json")


def export_feature_importances(db) -> None:
    """Export feature_importances table."""
    result = db.execute(select(FeatureImportance).order_by(FeatureImportance.id))
    rows = result.scalars().all()

    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "run_model_id": r.run_model_id,
            "feature_name": r.feature_name,
            "importance_score": r.importance_score,
            "rank": r.rank,
            "created_at": r.created_at,
        })

    _write_json(data, "feature_importances.json")


def main():
    print("=" * 60)
    print("Export PostgreSQL -> JSON (local dev fallback)")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        print(f"\nDatabase: {engine.url}")
        print(f"Output:   {OUTPUT_DIR}\n")

        export_info_cards(db)
        export_experiments(db)
        export_runs(db)
        export_run_models(db)
        export_metrics(db)
        export_feature_importances(db)

    # Write metadata
    meta = {
        "exported_at": datetime.now().isoformat(),
        "database_url": str(engine.url),
        "tables": [
            "info_cards", "experiments", "runs",
            "run_models", "metrics", "feature_importances",
        ],
    }
    _write_json(meta, "_export_meta.json")

    print(f"\n[DONE] Export complete -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
