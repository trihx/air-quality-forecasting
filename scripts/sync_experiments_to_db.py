"""Sync existing JSON experiment results → Database.

One-shot script that reads all JSON files in research/experiments/v8_final/
and inserts them into the DB via ExperimentLogger (direct SQLAlchemy).

Usage:
    uv run python scripts/sync_experiments_to_db.py 2>&1 | tee research/logs/sync_db.log

    # Force re-sync (delete existing and re-insert):
    uv run python scripts/sync_experiments_to_db.py --force
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = PROJECT_ROOT / "research" / "experiments" / "v8_final"

# Files to sync — mapping: (filename_pattern, experiment_name, description)
SYNC_TARGETS = [
    ("lightgbm_20260429_163822.json", "v8_lightgbm_20260429", "LightGBM tuned — v8 pipeline"),
    ("sklearn_20260429_163822.json", "v8_sklearn_20260429", "Sklearn models (RF, GB, Stacking, Ensemble) — v8 pipeline"),
    ("arima_20260429_163822.json", "v8_arima_20260429", "ARIMA/SARIMA — v8 pipeline"),
    ("dl_20260501_071715.json", "v8_dl_20260501", "Deep Learning (GRU, LSTM v2 log) — v8 pipeline"),
    ("tft_multi_horizon_20260429_163633.json", "v8_tft_20260429", "Temporal Fusion Transformer — v8 pipeline"),
]


def _load_json_safe(path: Path) -> dict | None:
    """Load JSON with NaN/Infinity handling."""
    text = path.read_text()
    # Replace JavaScript-style NaN/Infinity with null for valid JSON
    text = text.replace(": NaN", ": null")
    text = text.replace(": Infinity", ": null")
    text = text.replace(": -Infinity", ": null")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error: {e}", flush=True)
        return None


def _delete_experiment(name: str):
    """Delete experiment by name (for --force mode)."""
    from src.api.database import SessionLocal
    from src.api.models import Experiment

    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(Experiment.name == name).first()
        if exp:
            db.delete(exp)
            db.commit()
            print(f"  🗑️  Deleted existing experiment: {name}", flush=True)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Sync JSON experiments → Database")
    parser.add_argument("--force", action="store_true", help="Force re-sync (delete existing first)")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("📦 Sync Experiments JSON → Database", flush=True)
    print(f"Timestamp: {datetime.now().isoformat()}", flush=True)
    print(f"Source: {EXPERIMENTS_DIR}", flush=True)
    print("=" * 60, flush=True)

    from src.api.experiment_logger import ExperimentLogger

    logger = ExperimentLogger(mode="db")
    t0 = time.time()

    synced = 0
    skipped = 0
    errors = 0

    for filename, exp_name, description in SYNC_TARGETS:
        filepath = EXPERIMENTS_DIR / filename
        print(f"\n📄 {filename}", flush=True)

        if not filepath.exists():
            print(f"  ⚠️  File not found — skipping", flush=True)
            skipped += 1
            continue

        # Force mode: delete existing first
        if args.force:
            _delete_experiment(exp_name)

        # Load JSON
        data = _load_json_safe(filepath)
        if data is None:
            errors += 1
            continue

        # Log to DB
        exp_id = logger.log_full_result(
            experiment_name=exp_name,
            result_dict=data,
            description=description,
            version="v8",
            source_file=filename,
        )

        if exp_id is not None:
            print(f"  ✅ Synced → experiment_id={exp_id}", flush=True)
            synced += 1
        else:
            print(f"  ⏭️  Already exists — skipped", flush=True)
            skipped += 1

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print(f"📊 SYNC SUMMARY", flush=True)
    print(f"  Synced:  {synced}", flush=True)
    print(f"  Skipped: {skipped}", flush=True)
    print(f"  Errors:  {errors}", flush=True)
    print(f"  Time:    {elapsed:.1f}s", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Verify counts
    from src.api.database import SessionLocal
    from src.api.models import Experiment, Metric, Run, RunModel

    db = SessionLocal()
    try:
        print(f"\n📋 DB Counts:", flush=True)
        print(f"  Experiments: {db.query(Experiment).count()}", flush=True)
        print(f"  Runs:        {db.query(Run).count()}", flush=True)
        print(f"  RunModels:   {db.query(RunModel).count()}", flush=True)
        print(f"  Metrics:     {db.query(Metric).count()}", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
