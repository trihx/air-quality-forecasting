"""Gap Analysis Script — Analyze missing data patterns in raw IoT dataset.

Outputs:
- Gap distribution statistics
- Segment analysis
- Recommendations for imputation strategy
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "dataset" / "raw" / "final_dataset.csv"
CLEAN_CSV = PROJECT_ROOT / "dataset" / "interim" / "cleaned_hourly.csv"
OUTPUT_DIR = PROJECT_ROOT / "research" / "eda"


def main() -> None:
    print("=" * 60, flush=True)
    print("GAP ANALYSIS — PM2.5 IoT Sensor Data", flush=True)
    print("=" * 60, flush=True)

    # ── Step 1: Load raw data ──
    print("\n[1/5] Loading raw data...", flush=True)
    df_raw = pd.read_csv(
        RAW_CSV,
        parse_dates=["ngay_tao"],
        usecols=["ngay_tao", "pm25", "nhiet_do", "do_am", "diem_suong", "co2"],
    )
    df_raw = df_raw.set_index("ngay_tao").sort_index()
    print(f"  Raw records: {len(df_raw):,}", flush=True)
    print(f"  Date range: {df_raw.index.min()} → {df_raw.index.max()}", flush=True)
    span_hours = int((df_raw.index.max() - df_raw.index.min()).total_seconds() / 3600)
    print(f"  Total span: {span_hours:,} hours ({span_hours // 24} days)", flush=True)

    # ── Step 2: Resample to hourly ──
    print("\n[2/5] Resampling to hourly (mean)...", flush=True)
    hourly = df_raw.resample("1h").mean()
    n_total = len(hourly)
    n_nan = int(hourly["pm25"].isna().sum())
    n_ok = n_total - n_nan
    print(f"  Total hourly slots: {n_total:,}", flush=True)
    print(f"  Slots WITH data:    {n_ok:,} ({n_ok / n_total * 100:.1f}%)", flush=True)
    print(f"  Slots WITHOUT data: {n_nan:,} ({n_nan / n_total * 100:.1f}%)", flush=True)

    # ── Step 3: Gap analysis ──
    print("\n[3/5] Analyzing gaps...", flush=True)
    mask = hourly["pm25"].isna()
    groups = (mask != mask.shift()).cumsum()

    # Gap groups (consecutive NaN blocks)
    gap_groups = groups[mask]
    if len(gap_groups) == 0:
        print("  No gaps found!", flush=True)
        return

    gap_sizes = gap_groups.value_counts()
    gs = gap_sizes.values  # array of gap lengths in hours

    print(f"  Total gaps: {len(gs)}", flush=True)
    print("  Gap stats (hours):", flush=True)
    print(f"    Min:    {gs.min()}", flush=True)
    print(f"    Median: {int(np.median(gs))}", flush=True)
    print(f"    Mean:   {int(gs.mean())}", flush=True)
    print(f"    Max:    {gs.max()}", flush=True)
    print(f"    Total missing hours: {gs.sum():,}", flush=True)

    # Distribution by category
    print("\n  Gap duration distribution:", flush=True)
    thresholds = [
        (2, "≤2h (current fill)"),
        (6, "≤6h (short)"),
        (12, "≤12h (medium)"),
        (24, "≤24h (daily)"),
        (48, "≤48h (2 days)"),
        (168, "≤168h (1 week)"),
    ]
    cumulative = 0
    gap_distribution = {}
    for thresh, label in thresholds:
        count = int((gs <= thresh).sum())
        hours_recoverable = int(gs[gs <= thresh].sum())
        cumulative = hours_recoverable
        pct_recovery = cumulative / n_nan * 100 if n_nan > 0 else 0
        print(
            f"    {label:25s}: {count:4d} gaps, {hours_recoverable:6,}h recoverable ({pct_recovery:.1f}% of missing)",
            flush=True,
        )
        gap_distribution[label] = {
            "count": count,
            "hours_recoverable": hours_recoverable,
            "pct_of_missing": round(pct_recovery, 1),
        }

    long_gaps = int((gs > 168).sum())
    long_hours = int(gs[gs > 168].sum())
    print(
        f"    {'>168h (>1 week)':25s}: {long_gaps:4d} gaps, {long_hours:6,}h UNRECOVERABLE",
        flush=True,
    )

    # ── Step 4: Data with each strategy ──
    print("\n[4/5] Estimating dataset size per strategy...", flush=True)
    for max_gap_h in [2, 6, 12, 24]:
        # Simulate: fill gaps ≤ max_gap_h, drop the rest
        filled = hourly.copy()
        for col in filled.columns:
            filled[col] = filled[col].interpolate(method="linear", limit=max_gap_h)
        remaining = filled.dropna()
        pct = len(remaining) / n_total * 100
        print(
            f"  max_gap={max_gap_h:3d}h: {len(remaining):,} rows "
            f"({pct:.1f}% of hourly, +{len(remaining) - n_ok:,} imputed)",
            flush=True,
        )

    # ── Step 5: Load current cleaned data for segment analysis ──
    print("\n[5/5] Segment analysis (current cleaned data)...", flush=True)
    if CLEAN_CSV.exists():
        df_c = pd.read_csv(CLEAN_CSV, index_col=0, parse_dates=True)
        print(f"  Cleaned rows: {len(df_c):,}", flush=True)

        # Segments
        diffs = df_c.index.to_series().diff()
        breaks = diffs > pd.Timedelta("1h")
        segs = breaks.cumsum()
        seg_sizes = segs.value_counts().sort_values(ascending=False)
        print(f"  Continuous segments: {len(seg_sizes)}", flush=True)
        print("  Segment sizes (hours):", flush=True)
        print(f"    Min:    {seg_sizes.min()}", flush=True)
        print(f"    Median: {int(seg_sizes.median())}", flush=True)
        print(f"    Max:    {seg_sizes.max()}", flush=True)
        print("  Top 5 largest segments:", flush=True)
        for i, (seg_id, n) in enumerate(seg_sizes.head(5).items()):
            s = df_c[segs == seg_id]
            print(
                f"    #{i + 1}: {n} hours ({s.index.min().date()} → {s.index.max().date()})",
                flush=True,
            )
    else:
        print(f"  Cleaned file not found: {CLEAN_CSV}", flush=True)

    # ── Save report ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_records": len(df_raw),
        "span_hours": span_hours,
        "hourly_total": n_total,
        "hourly_with_data": n_ok,
        "hourly_missing": n_nan,
        "pct_missing": round(n_nan / n_total * 100, 1),
        "gap_count": len(gs),
        "gap_min_h": int(gs.min()),
        "gap_median_h": int(np.median(gs)),
        "gap_mean_h": int(gs.mean()),
        "gap_max_h": int(gs.max()),
        "gap_distribution": gap_distribution,
    }
    report_path = OUTPUT_DIR / "gap_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Report saved: {report_path}", flush=True)
    print("=" * 60, flush=True)
    print("GAP ANALYSIS COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
