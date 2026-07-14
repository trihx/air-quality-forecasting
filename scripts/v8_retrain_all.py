"""v8 Phase 4.2-4.5 — Retrain all models on rebuilt data.

Runs models sequentially: LightGBM → sklearn → ARIMA → DL
Each family saves results independently to v8_final/.

Usage:
    uv run python scripts/v8_retrain_all.py --family lightgbm 2>&1 | tee research/logs/v8_retrain_lgbm.log
    uv run python scripts/v8_retrain_all.py --family sklearn 2>&1 | tee research/logs/v8_retrain_sklearn.log
    uv run python scripts/v8_retrain_all.py --family arima 2>&1 | tee research/logs/v8_retrain_arima.log
    uv run python scripts/v8_retrain_all.py --family dl 2>&1 | tee research/logs/v8_retrain_dl.log
    uv run python scripts/v8_retrain_all.py --family all 2>&1 | tee research/logs/v8_retrain_all.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the existing retrain functions
from scripts.retrain_v7_full import (
    TIMESTAMP,
    prepare_data,
    run_lightgbm,
    run_sklearn,
    run_arima,
    run_dl,
    _save,
)


def main():
    parser = argparse.ArgumentParser(description="v8 Retrain — per family")
    parser.add_argument(
        "--family",
        choices=["lightgbm", "sklearn", "arima", "dl", "all"],
        required=True,
        help="Model family to retrain",
    )
    args = parser.parse_args()

    print(f"[v8] Preparing data (KNN past-only)...", flush=True)
    _, df_feat = prepare_data()

    families = [args.family] if args.family != "all" else ["lightgbm", "sklearn", "arima", "dl"]

    for fam in families:
        print(f"\n{'=' * 70}", flush=True)
        print(f"[v8] Retraining: {fam.upper()}", flush=True)
        print(f"{'=' * 70}", flush=True)

        if fam == "lightgbm":
            out = run_lightgbm(df_feat)
        elif fam == "sklearn":
            out = run_sklearn(df_feat)
        elif fam == "arima":
            out = run_arima(df_feat)
        elif fam == "dl":
            out = run_dl(df_feat)
        else:
            continue

        _save(out["metrics"], fam)
        _save(out["preds"], f"{fam}_preds")
        print(f"[v8] ✅ {fam.upper()} saved.", flush=True)


if __name__ == "__main__":
    main()
