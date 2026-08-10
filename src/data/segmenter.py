"""Data segmenter — Identify and manage contiguous data segments.

For IoT time series with large gaps (sensor offline), data after cleaning
consists of discontinuous blocks. This module identifies those blocks and
assigns segment IDs so that lag/rolling features are computed ONLY within
each contiguous segment — eliminating False Continuity.

Usage:
    from src.data.segmenter import identify_contiguous_segments, validate_segment_boundaries

    df = identify_contiguous_segments(df, target_col="pm25", min_length=24)
    validate_segment_boundaries(df, segment_col="segment_id")
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.data.loader import TARGET_COL


def identify_contiguous_segments(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    min_length: int = 1,
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """Assign segment IDs to contiguous non-NaN blocks in the target column.

    Each contiguous block of valid (non-NaN) data gets a unique integer ID.
    Blocks shorter than `min_length` are dropped.

    Args:
        df: DataFrame with DatetimeIndex.
        target_col: Column to check for NaN gaps.
        min_length: Minimum segment length (in time steps) to keep.
        segment_col: Name of the output segment ID column.

    Returns:
        DataFrame with `segment_col` added, rows from short segments dropped.
    """
    df = df.copy()

    is_valid = df[target_col].notna()
    n_valid_before = int(is_valid.sum())

    # Assign group IDs: each transition (valid→NaN or NaN→valid) starts a new group
    groups = (is_valid != is_valid.shift()).cumsum()

    # Keep only valid (non-NaN) groups
    df[segment_col] = pd.NA
    df.loc[is_valid, segment_col] = groups[is_valid]

    # Drop NaN rows (gap rows)
    df = df[is_valid].copy()

    # Re-number segments sequentially (1, 2, 3, ...)
    unique_segs = df[segment_col].unique()
    seg_map = {old: new for new, old in enumerate(sorted(unique_segs), start=1)}
    df[segment_col] = df[segment_col].map(seg_map).astype(int)

    # Compute segment lengths
    seg_lengths = df.groupby(segment_col).size()
    n_total_segs = len(seg_lengths)

    # Filter out short segments
    if min_length > 1:
        keep_segs = seg_lengths[seg_lengths >= min_length].index
        n_dropped_segs = n_total_segs - len(keep_segs)
        n_dropped_rows = int(seg_lengths[seg_lengths < min_length].sum())
        df = df[df[segment_col].isin(keep_segs)].copy()

        # Re-number again after filtering
        unique_segs = sorted(df[segment_col].unique())
        seg_map = {old: new for new, old in enumerate(unique_segs, start=1)}
        df[segment_col] = df[segment_col].map(seg_map).astype(int)

        logger.info(
            f"[Segmenter] Dropped {n_dropped_segs} short segments ({n_dropped_rows} rows, min_length={min_length})"
        )

    # Final stats
    final_seg_lengths = df.groupby(segment_col).size()
    n_segs = len(final_seg_lengths)
    n_rows = len(df)

    logger.info(
        f"[Segmenter] {n_segs} segments, {n_rows:,} rows "
        f"(min={final_seg_lengths.min()}, median={final_seg_lengths.median():.0f}, "
        f"max={final_seg_lengths.max()})"
    )

    return df


def get_segment_stats(df: pd.DataFrame, segment_col: str = "segment_id") -> dict:
    """Get summary statistics about segments in a DataFrame.

    Args:
        df: DataFrame with segment_col.
        segment_col: Name of the segment ID column.

    Returns:
        Dictionary with segment statistics.
    """
    if segment_col not in df.columns:
        return {"error": f"Column '{segment_col}' not found"}

    seg_lengths = df.groupby(segment_col).size()

    return {
        "n_segments": len(seg_lengths),
        "total_rows": len(df),
        "min_length": int(seg_lengths.min()),
        "max_length": int(seg_lengths.max()),
        "mean_length": round(float(seg_lengths.mean()), 1),
        "median_length": round(float(seg_lengths.median()), 1),
        "p25_length": round(float(seg_lengths.quantile(0.25)), 1),
        "p75_length": round(float(seg_lengths.quantile(0.75)), 1),
    }


def validate_segment_boundaries(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    max_allowed_gap_hours: float = 1.5,
) -> bool:
    """Validate that no segment contains a time gap larger than allowed.

    This catches False Continuity: rows from different time periods
    accidentally placed in the same segment.

    Args:
        df: DataFrame with DatetimeIndex and segment_col.
        segment_col: Name of the segment ID column.
        max_allowed_gap_hours: Maximum allowed gap (hours) within a segment.

    Returns:
        True if all segments are contiguous, False otherwise.

    Raises:
        ValueError: If a segment contains an internal gap > max_allowed_gap_hours.
    """
    if segment_col not in df.columns:
        raise ValueError(f"Column '{segment_col}' not found in DataFrame")

    max_gap_td = pd.Timedelta(hours=max_allowed_gap_hours)
    violations = []

    for seg_id, group in df.groupby(segment_col):
        if len(group) < 2:
            continue

        time_diffs = group.index.to_series().diff().dropna()
        max_diff = time_diffs.max()

        if max_diff > max_gap_td:
            violations.append(
                {
                    "segment_id": seg_id,
                    "max_gap": str(max_diff),
                    "location": str(time_diffs.idxmax()),
                }
            )

    if violations:
        for v in violations:
            logger.error(
                f"[Segmenter] FALSE CONTINUITY in segment {v['segment_id']}: gap={v['max_gap']} at {v['location']}"
            )
        raise ValueError(
            f"Found {len(violations)} segments with internal gaps > {max_allowed_gap_hours}h. "
            f"This indicates False Continuity — lag/rolling features would be incorrect."
        )

    logger.info(
        f"[Segmenter] ✅ All {df[segment_col].nunique()} segments validated (no gap > {max_allowed_gap_hours}h)"
    )
    return True
