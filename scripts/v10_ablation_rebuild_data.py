"""v10 Ablation Study — Phase 1: Rebuild data with IQR-truncated PM2.5.

Mô phỏng lỗi cũ: Dùng IQR×3 cho PM2.5 thay vì Domain Bounds [0, 500].
Chỉ build 30m resolution.

Usage:
    uv run python scripts/v10_ablation_rebuild_data.py 2>&1 | tee research/logs/v10_rebuild.log
"""
from __future__ import annotations

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "v10_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _handle_outliers_iqr_all(df: pd.DataFrame, threshold: float = 3.0) -> tuple[pd.DataFrame, int]:
    """Handle outliers using IQR for ALL columns including PM2.5.

    This is the OLD (incorrect) method that clips PM2.5 extreme events.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_replaced = 0

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = mask.sum()
        if n_outliers > 0:
            df.loc[mask, col] = np.nan
            total_replaced += n_outliers
            print(f"  {col}: {n_outliers} outliers -> NaN (range: [{lower:.1f}, {upper:.1f}])", flush=True)

    return df, total_replaced


def main():
    print("=" * 70, flush=True)
    print(f"v10 ABLATION — IQR-Truncated Data Rebuild (30m only)", flush=True)
    print(f"Timestamp: {TIMESTAMP}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()

    from src.data.cleaner import (
        _clip_physical_bounds,
        _remove_duplicates,
        _resample,
        _set_datetime_index,
    )
    from src.data.imputer import get_imputation_stats, impute_missing_data
    from src.data.loader import load_raw_data
    from src.features.builder import build_features

    freq = "30min"
    min_segment_steps = 48  # 24h

    # Step 1: Load + Clean (IQR for ALL including PM2.5)
    print("[1/5] Loading + Cleaning (IQR for PM2.5 — mô phỏng lỗi)...", flush=True)
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    # KEY DIFFERENCE: Use IQR for ALL columns (including PM2.5)
    df, n_outliers = _handle_outliers_iqr_all(df, threshold=3.0)
    print(f"  Total outliers replaced: {n_outliers}", flush=True)
    df = _resample(df, freq=freq)
    print(f"  After resample ({freq}): {df.shape}", flush=True)

    # Step 2: Impute
    print("[2/5] Imputing (hybrid, KNN past-only)...", flush=True)
    steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(freq))
    df_imputed = impute_missing_data(
        df, strategy="hybrid",
        max_gap_interp=6 * steps_per_hour,
        max_gap_ml=24 * steps_per_hour,
        knn_neighbors=5, verbose=True,
    )
    imp_stats = get_imputation_stats(df_imputed)
    print(f"  Imputed: {imp_stats}", flush=True)

    # Step 3: Segment
    print("[3/5] Segmenting by time gaps...", flush=True)
    is_imputed = df_imputed["is_imputed"].copy()
    df_seg = df_imputed.drop(columns=["is_imputed"])

    expected_step = pd.Timedelta(freq)
    time_diffs = df_seg.index.to_series().diff()
    is_new_segment = time_diffs > expected_step * 1.5
    segment_ids = is_new_segment.cumsum() + 1
    df_seg["segment_id"] = segment_ids.values

    seg_sizes = df_seg.groupby("segment_id").size()
    keep_segs = seg_sizes[seg_sizes >= min_segment_steps].index
    n_dropped = len(seg_sizes) - len(keep_segs)
    df_seg = df_seg[df_seg["segment_id"].isin(keep_segs)].copy()

    unique_segs = sorted(df_seg["segment_id"].unique())
    seg_map = {old: new for new, old in enumerate(unique_segs, start=1)}
    df_seg["segment_id"] = df_seg["segment_id"].map(seg_map).astype(int)
    print(f"  Dropped {n_dropped} short segments", flush=True)

    # Step 4: Features
    print("[4/5] Building features (segment-aware)...", flush=True)
    df_feat = build_features(
        df_seg, include_fourier=True, fourier_order=3,
        drop_na=True, segment_col="segment_id",
    )
    df_feat["is_imputed"] = is_imputed.reindex(df_feat.index).fillna(False)
    print(f"  Features: {df_feat.shape}", flush=True)

    # Step 5: Save base (for DL) and full features
    print("[5/5] Saving...", flush=True)
    from src.features.calendar import create_calendar_features
    from src.features.fourier import create_fourier_features
    df_base = df_seg.copy()
    df_base = create_calendar_features(df_base)
    df_base = create_fourier_features(df_base, order=3)
    df_base["is_imputed"] = is_imputed.reindex(df_base.index).fillna(False)
    df_base.to_csv(OUTPUT_DIR / "marts_features_30m_base.csv")
    df_feat.to_csv(OUTPUT_DIR / "marts_features_30m.csv")

    elapsed = time.time() - t0

    # Compare with v9
    v9_path = PROJECT_ROOT / "dataset" / "processed" / "marts_features_30m.csv"
    df_v9 = pd.read_csv(v9_path, index_col=0, parse_dates=True)

    print(f"\n{'='*70}", flush=True)
    print(f"COMPARISON: v9 (Domain Bounds) vs v10 (IQR)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Metric':<25} {'v9 (Domain)':<15} {'v10 (IQR)':<15}", flush=True)
    print(f"{'Rows':<25} {len(df_v9):<15} {len(df_feat):<15}", flush=True)
    print(f"{'PM2.5 max':<25} {df_v9['pm25'].max():<15.2f} {df_feat['pm25'].max():<15.2f}", flush=True)
    print(f"{'PM2.5 mean':<25} {df_v9['pm25'].mean():<15.2f} {df_feat['pm25'].mean():<15.2f}", flush=True)
    print(f"{'PM2.5 > 54':<25} {(df_v9['pm25'] > 54).sum():<15} {(df_feat['pm25'] > 54).sum():<15}", flush=True)

    # Save metadata
    meta = {
        "timestamp": TIMESTAMP,
        "description": "v10 Ablation: IQR-truncated PM2.5 (mô phỏng lỗi cũ)",
        "outlier_method": "IQR×3 for ALL columns including PM2.5",
        "freq": "30m",
        "shape": list(df_feat.shape),
        "pm25_max": round(float(df_feat["pm25"].max()), 4),
        "pm25_mean": round(float(df_feat["pm25"].mean()), 4),
        "pm25_gt_54": int((df_feat["pm25"] > 54).sum()),
        "v9_pm25_max": round(float(df_v9["pm25"].max()), 4),
        "v9_pm25_gt_54": int((df_v9["pm25"] > 54).sum()),
        "elapsed_s": round(elapsed, 1),
    }
    report_dir = PROJECT_ROOT / "research" / "experiments" / "v10_ablation"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / f"data_rebuild_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ v10 data rebuild complete in {elapsed:.0f}s", flush=True)
    print(f"   Output: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
