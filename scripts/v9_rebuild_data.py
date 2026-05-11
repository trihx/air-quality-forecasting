"""v9 Phase 5A — Rebuild data at high resolution with segment-aware features.

Builds data at BOTH 15min and 30min resolutions:
  Raw → Clean (freq=15min|30min) → Impute (hybrid, KNN past-only)
  → Segment → Features (segment-aware) → Save

Usage:
    uv run python scripts/v9_rebuild_data.py 2>&1 | tee research/logs/v9_rebuild_data.log
"""

from __future__ import annotations

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


def build_pipeline(freq: str, min_segment_steps: int) -> dict:
    """Run the full data pipeline at a given frequency.

    Args:
        freq: Resampling frequency ("15min" or "30min").
        min_segment_steps: Minimum segment length in steps to keep.

    Returns:
        Summary dict with stats.
    """
    from src.data.cleaner import (
        _clip_physical_bounds,
        _handle_outliers,
        _remove_duplicates,
        _resample,
        _set_datetime_index,
    )
    from src.data.imputer import get_imputation_stats, impute_missing_data
    from src.data.loader import load_raw_data
    from src.features.builder import build_features

    print(f"\n{'='*70}", flush=True)
    print(f"  BUILDING {freq} PIPELINE (min_segment={min_segment_steps} steps)", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()

    # Step 1: Load + Clean
    print("[1/5] Loading + Cleaning...", flush=True)
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq=freq)
    print(f"  After resample ({freq}): {df.shape}", flush=True)

    # Step 2: Impute
    print("[2/5] Imputing (hybrid, KNN past-only)...", flush=True)
    # Adjust max_gap for resolution: max_gap_interp and max_gap_ml are in STEPS
    steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(freq))
    max_gap_interp_steps = 6 * steps_per_hour  # 6 hours in steps
    max_gap_ml_steps = 24 * steps_per_hour  # 24 hours in steps

    df_imputed = impute_missing_data(
        df,
        strategy="hybrid",
        max_gap_interp=max_gap_interp_steps,
        max_gap_ml=max_gap_ml_steps,
        knn_neighbors=5,
        verbose=True,
    )
    imp_stats = get_imputation_stats(df_imputed)
    print(f"  Imputed: {imp_stats}", flush=True)

    # Step 3: Segment by TIME GAPS (not NaN)
    # After imputation, gaps are DROPPED (not NaN). So we detect segments
    # by checking time differences between consecutive rows.
    print("[3/5] Segmenting by time gaps...", flush=True)
    is_imputed = df_imputed["is_imputed"].copy()
    df_seg = df_imputed.drop(columns=["is_imputed"])

    # Detect gaps: where time diff > 1.5x the expected step size
    expected_step = pd.Timedelta(freq)
    max_gap = expected_step * 1.5
    time_diffs = df_seg.index.to_series().diff()
    is_new_segment = time_diffs > max_gap
    segment_ids = is_new_segment.cumsum() + 1  # Segments start from 1
    df_seg["segment_id"] = segment_ids.values

    # Drop short segments
    seg_sizes = df_seg.groupby("segment_id").size()
    keep_segs = seg_sizes[seg_sizes >= min_segment_steps].index
    n_dropped = len(seg_sizes) - len(keep_segs)
    n_dropped_rows = int(seg_sizes[seg_sizes < min_segment_steps].sum())
    df_seg = df_seg[df_seg["segment_id"].isin(keep_segs)].copy()

    # Re-number sequentially
    unique_segs = sorted(df_seg["segment_id"].unique())
    seg_map = {old: new for new, old in enumerate(unique_segs, start=1)}
    df_seg["segment_id"] = df_seg["segment_id"].map(seg_map).astype(int)

    from src.data.segmenter import get_segment_stats, validate_segment_boundaries
    seg_stats = get_segment_stats(df_seg)
    print(f"  Dropped {n_dropped} short segments ({n_dropped_rows} rows)", flush=True)
    print(f"  Segments: {seg_stats}", flush=True)

    # Validate no false continuity
    max_allowed_gap = pd.Timedelta(freq).total_seconds() / 3600 * 1.5
    validate_segment_boundaries(df_seg, max_allowed_gap_hours=max_allowed_gap)

    # Step 4: Features (segment-aware)
    print("[4/5] Building features (segment-aware)...", flush=True)
    df_feat = build_features(
        df_seg,
        include_fourier=True,
        fourier_order=3,
        drop_na=True,
        segment_col="segment_id",
    )

    # Re-attach is_imputed
    df_feat["is_imputed"] = is_imputed.reindex(df_feat.index).fillna(False)
    print(f"  Features: {df_feat.shape}", flush=True)

    # Step 5: Save
    print("[5/5] Saving...", flush=True)
    output_dir = PROJECT_ROOT / "dataset" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    freq_label = freq.replace("min", "m")

    # Save base segment data for DL Expert Pipeline (before tabular warmup rows are dropped)
    # This file has MORE rows because it doesn't lose warmup from lag/rolling features
    from src.features.calendar import create_calendar_features
    from src.features.fourier import create_fourier_features
    df_base = df_seg.copy()
    df_base = create_calendar_features(df_base)
    df_base = create_fourier_features(df_base, order=3)
    df_base["is_imputed"] = is_imputed.reindex(df_base.index).fillna(False)
    base_output_path = output_dir / f"marts_features_{freq_label}_base.csv"
    df_base.to_csv(base_output_path)
    print(f"  Base (DL Expert): {base_output_path} ({len(df_base):,} rows)", flush=True)
    output_path = output_dir / f"marts_features_{freq_label}.csv"
    df_feat.to_csv(output_path)
    print(f"  Saved to: {output_path}", flush=True)

    elapsed = time.time() - t0

    summary = {
        "freq": freq,
        "min_segment_steps": min_segment_steps,
        "shape": list(df_feat.shape),
        "n_segments": seg_stats["n_segments"],
        "pm25_mean": round(float(df_feat["pm25"].mean()), 4),
        "pm25_std": round(float(df_feat["pm25"].std()), 4),
        "imputation": imp_stats,
        "segments": seg_stats,
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n  ✅ {freq} pipeline complete in {elapsed:.0f}s", flush=True)
    print(f"     Shape: {summary['shape']}, Segments: {summary['n_segments']}", flush=True)

    return summary


def main():
    print("=" * 70, flush=True)
    print(f"v9 DATA REBUILD — High-Resolution Segmented Pipeline", flush=True)
    print(f"Timestamp: {TIMESTAMP}", flush=True)
    print("=" * 70, flush=True)

    t_total = time.time()

    results = {}

    # Build 30min pipeline (min segment = 48 steps = 24h)
    results["30min"] = build_pipeline("30min", min_segment_steps=48)

    # Build 15min pipeline (min segment = 96 steps = 24h)
    results["15min"] = build_pipeline("15min", min_segment_steps=96)

    # Comparison with v8 (1h)
    old_path = PROJECT_ROOT / "dataset" / "processed" / "marts_features.csv"
    if old_path.exists():
        df_old = pd.read_csv(old_path, index_col=0, parse_dates=True)
        results["1h_v8"] = {
            "shape": list(df_old.shape),
            "pm25_mean": round(float(df_old["pm25"].mean()), 4),
        }
    else:
        results["1h_v8"] = {"shape": [0, 0], "pm25_mean": 0}

    # Save report
    report = {
        "timestamp": TIMESTAMP,
        "description": "v9 high-resolution segmented data rebuild",
        "results": results,
    }

    report_dir = PROJECT_ROOT / "research" / "experiments" / "v9_final"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"data_rebuild_{TIMESTAMP}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - t_total

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print(f"FINAL COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Pipeline':<12} {'Rows':>8} {'Cols':>6} {'PM2.5 Mean':>12} {'Segments':>10}", flush=True)
    for key in ["1h_v8", "30min", "15min"]:
        r = results[key]
        segs = r.get("n_segments", "N/A")
        print(f"{key:<12} {r['shape'][0]:>8,} {r['shape'][1]:>6} {r['pm25_mean']:>12} {segs:>10}", flush=True)

    print(f"\n✅ Total elapsed: {total_elapsed:.0f}s", flush=True)
    print(f"   Report: {report_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
