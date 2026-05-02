"""EDA: Analyze contiguous segments at multiple resolutions.

Usage:
    uv run python scripts/eda_segment_analysis.py 2>&1 | tee research/logs/eda_segments.log
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_data
from src.data.cleaner import (
    _remove_duplicates,
    _set_datetime_index,
    _clip_physical_bounds,
    _handle_outliers,
    _resample,
)


def analyze_segments(df: pd.DataFrame, freq: str, label: str) -> dict:
    """Analyze contiguous non-NaN segments in PM2.5 column."""
    df_rs = _resample(df.copy(), freq=freq)
    pm = df_rs["pm25"]

    is_valid = pm.notna()
    segment_id = (is_valid != is_valid.shift()).cumsum()
    valid_segments = segment_id[is_valid]
    seg_lengths = valid_segments.value_counts().sort_index()

    # Steps per day at this frequency
    steps_per_day = int(pd.Timedelta("24h") / pd.Timedelta(freq))

    print(f"\n{'='*60}")
    print(f"  {label} (freq={freq}, {steps_per_day} steps/day)")
    print(f"{'='*60}")
    print(f"  Total rows:     {len(df_rs):,}")
    print(f"  Valid rows:     {is_valid.sum():,}")
    print(f"  NaN rows:       {(~is_valid).sum():,}")
    print(f"  Segments:       {len(seg_lengths)}")
    print(f"  Seg length (steps): min={seg_lengths.min()}, median={seg_lengths.median():.0f}, "
          f"mean={seg_lengths.mean():.0f}, max={seg_lengths.max()}")

    for days in [1, 4, 7]:
        min_steps = steps_per_day * days
        good = seg_lengths[seg_lengths >= min_steps]
        print(f"  Segments >= {days}d ({min_steps} steps): "
              f"{len(good)} segs, {good.sum():,} rows")

    return {
        "freq": freq,
        "total_rows": len(df_rs),
        "valid_rows": int(is_valid.sum()),
        "n_segments": len(seg_lengths),
        "max_seg_len": int(seg_lengths.max()),
    }


def main():
    print("=" * 60, flush=True)
    print("EDA: Contiguous Segment Analysis (Multi-Resolution)", flush=True)
    print("=" * 60, flush=True)

    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)

    results = []
    for freq, label in [
        ("15min", "HIGH-RES 15min"),
        ("30min", "MEDIUM-RES 30min"),
        ("1h", "CURRENT 1h"),
    ]:
        r = analyze_segments(df, freq, label)
        results.append(r)

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Freq':<8} {'Valid Rows':>12} {'Segments':>10} {'Max Seg':>10} {'Gain vs 1h':>12}")
    base = results[-1]["valid_rows"]
    for r in results:
        gain = f"{r['valid_rows'] / base:.1f}x"
        print(f"{r['freq']:<8} {r['valid_rows']:>12,} {r['n_segments']:>10} {r['max_seg_len']:>10} {gain:>12}")

    print("\n✅ Analysis complete.", flush=True)


if __name__ == "__main__":
    main()
