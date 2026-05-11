"""v8 Phase 4.1 — Re-impute data with KNN past-only fix.

Runs: Raw → Clean → Impute (hybrid, KNN past-only) → Features
Saves new marts_features.csv and reports comparison with old data.

Usage:
    uv run python scripts/v8_rebuild_data.py 2>&1 | tee research/logs/v8_rebuild_data.log
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    print("=" * 70, flush=True)
    print(f"v8 DATA REBUILD — KNN Past-Only Fix", flush=True)
    print(f"Timestamp: {TIMESTAMP}", flush=True)
    print("=" * 70, flush=True)

    t_total = time.time()

    # ── Step 1: Load old data for comparison ──
    old_path = PROJECT_ROOT / "dataset" / "processed" / "marts_features.csv"
    if old_path.exists():
        df_old = pd.read_csv(old_path, index_col=0, parse_dates=True)
        old_stats = {
            "shape": list(df_old.shape),
            "pm25_mean": round(float(df_old["pm25"].mean()), 4),
            "pm25_std": round(float(df_old["pm25"].std()), 4),
            "pm25_median": round(float(df_old["pm25"].median()), 4),
        }
        print(f"[OLD] Shape: {df_old.shape}", flush=True)
        print(f"[OLD] PM2.5: mean={old_stats['pm25_mean']}, std={old_stats['pm25_std']}", flush=True)
    else:
        df_old = None
        old_stats = None
        print("[OLD] No existing data found — fresh build.", flush=True)

    # ── Step 2: Run full pipeline (clean → impute → features) ──
    from src.data.cleaner import (
        _clip_physical_bounds,
        _handle_outliers,
        _remove_duplicates,
        _resample,
        _set_datetime_index,
    )
    from src.data.imputer import impute_missing_data, get_imputation_stats
    from src.data.loader import load_raw_data
    from src.features.builder import build_features

    print("\n[STEP 1/4] Loading raw data...", flush=True)
    df_raw = load_raw_data()
    print(f"  Raw: {df_raw.shape}", flush=True)

    print("[STEP 2/4] Cleaning...", flush=True)
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, clip_report = _clip_physical_bounds(df)
    df, outlier_report = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")
    print(f"  Cleaned: {df.shape}", flush=True)
    print(f"  NaN before impute: {df.isna().sum().sum()}", flush=True)

    print("[STEP 3/4] Imputing (hybrid, KNN past-only)...", flush=True)
    df_hybrid = impute_missing_data(
        df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24,
        knn_neighbors=5, verbose=True,
    )
    imp_stats = get_imputation_stats(df_hybrid)
    print(f"  Imputed: {imp_stats}", flush=True)

    print("[STEP 4/4] Building features...", flush=True)
    is_imputed = df_hybrid["is_imputed"].copy()
    df_for_feat = df_hybrid.drop(columns=["is_imputed"])
    df_feat = build_features(df_for_feat, include_fourier=True, fourier_order=3)
    df_feat["is_imputed"] = is_imputed.reindex(df_feat.index).fillna(False)
    print(f"  Features: {df_feat.shape}", flush=True)

    # ── Step 3: Save ──
    # Backup old
    if old_path.exists():
        backup_path = old_path.with_name(f"marts_features_pre_v8_{TIMESTAMP}.csv")
        df_old.to_csv(backup_path)
        print(f"\n[BACKUP] Old data saved to: {backup_path.name}", flush=True)

    # Save new
    df_feat.to_csv(old_path)
    print(f"[SAVE] New data saved to: {old_path}", flush=True)

    # ── Step 4: Comparison report ──
    new_stats = {
        "shape": list(df_feat.shape),
        "pm25_mean": round(float(df_feat["pm25"].mean()), 4),
        "pm25_std": round(float(df_feat["pm25"].std()), 4),
        "pm25_median": round(float(df_feat["pm25"].median()), 4),
        "imputation": imp_stats,
    }

    report = {
        "timestamp": TIMESTAMP,
        "description": "v8 data rebuild with KNN past-only fix",
        "old_data": old_stats,
        "new_data": new_stats,
        "changes": {},
    }

    if old_stats:
        report["changes"] = {
            "rows_delta": new_stats["shape"][0] - old_stats["shape"][0],
            "cols_delta": new_stats["shape"][1] - old_stats["shape"][1],
            "pm25_mean_delta": round(new_stats["pm25_mean"] - old_stats["pm25_mean"], 4),
            "pm25_std_delta": round(new_stats["pm25_std"] - old_stats["pm25_std"], 4),
        }

    report_path = PROJECT_ROOT / "research" / "experiments" / "v8_final" / f"data_rebuild_{TIMESTAMP}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}", flush=True)
    print(f"✅ DATA REBUILD COMPLETE — {elapsed:.0f}s", flush=True)
    print(f"   Rows: {new_stats['shape'][0]:,} | Cols: {new_stats['shape'][1]}", flush=True)
    print(f"   PM2.5 mean: {new_stats['pm25_mean']} (old: {old_stats['pm25_mean'] if old_stats else 'N/A'})", flush=True)
    if old_stats:
        print(f"   Δ mean: {report['changes']['pm25_mean_delta']}", flush=True)
    print(f"   Report: {report_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
