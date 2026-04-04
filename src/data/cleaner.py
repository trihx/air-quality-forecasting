"""Data cleaner — clean, handle missing values, outliers, and resample.

Pipeline: raw → interim (Intermediate layer per SKILL.md §3.5).
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.data.loader import DATETIME_COL, FEATURE_COLS, TARGET_COL

# Physical bounds for sensor readings
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "nhiet_do": (-10.0, 60.0),  # Temperature °C
    "do_am": (0.0, 100.0),  # Humidity %
    "diem_suong": (-20.0, 40.0),  # Dew point °C
    "co2": (0.0, 5000.0),  # CO2 ppm
    "pm25": (0.0, 500.0),  # PM2.5 µg/m³ (AQI scale)
}


def clean_data(
    df: pd.DataFrame,
    resample_freq: str = "1h",
    interpolation_method: str = "linear",
    max_gap_interpolate: str = "2h",
    outlier_method: str = "iqr",
    outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    """Full cleaning pipeline: duplicates → bounds → outliers → resample → interpolate.

    Args:
        df: Raw DataFrame from loader (must have DATETIME_COL parsed).
        resample_freq: Resampling frequency (default: 1h).
        interpolation_method: Interpolation method for missing values.
        max_gap_interpolate: Max gap size to interpolate across.
        outlier_method: Outlier detection method ('iqr' or 'zscore').
        outlier_threshold: Threshold for outlier detection.

    Returns:
        Cleaned DataFrame with regular time index.
    """
    logger.info("=" * 60)
    logger.info("Data Cleaning Pipeline Started")
    logger.info("=" * 60)

    n_before = len(df)
    report: dict[str, Any] = {"n_input": n_before}

    # Step 1: Remove exact duplicates
    df = _remove_duplicates(df)
    report["n_after_dedup"] = len(df)

    # Step 2: Set datetime index
    df = _set_datetime_index(df)

    # Step 3: Clip to physical bounds
    df, n_clipped = _clip_physical_bounds(df)
    report["n_clipped"] = n_clipped

    # Step 4: Handle outliers
    df, n_outliers = _handle_outliers(df, method=outlier_method, threshold=outlier_threshold)
    report["n_outliers_replaced"] = n_outliers

    # Step 5: Resample to regular frequency
    df = _resample(df, freq=resample_freq)
    report["n_after_resample"] = len(df)

    # Step 6: Interpolate missing values
    df, n_interpolated = _interpolate_gaps(
        df,
        method=interpolation_method,
        max_gap=max_gap_interpolate,
    )
    report["n_interpolated"] = n_interpolated

    # Step 7: Drop remaining NaN rows
    n_nan_before = df.isna().any(axis=1).sum()
    df = df.dropna()
    report["n_dropped_nan"] = n_nan_before - (len(df) - len(df.dropna()))

    n_after = len(df)
    report["n_output"] = n_after
    report["pct_retained"] = float(round(n_after / n_before * 100, 1)) if n_before > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"Cleaning Complete: {n_before:,} → {n_after:,} rows ({report['pct_retained']}% retained)")
    logger.info(f"Report: {report}")
    logger.info("=" * 60)

    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows, keeping first occurrence."""
    # Exact duplicate rows
    n_exact = df.duplicated().sum()
    if n_exact > 0:
        df = df.drop_duplicates()
        logger.info(f"[1/7] Removed {n_exact} exact duplicate rows")
    else:
        logger.info("[1/7] No exact duplicates found")

    # Duplicate timestamps (keep first)
    n_ts_dup = df[DATETIME_COL].duplicated().sum()
    if n_ts_dup > 0:
        df = df.drop_duplicates(subset=[DATETIME_COL], keep="first")
        logger.info(f"[1/7] Removed {n_ts_dup} duplicate timestamps (kept first)")

    return df


def _set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set datetime column as index."""
    df = df.set_index(DATETIME_COL)
    df = df.sort_index()
    logger.info(f"[2/7] Datetime index set: {df.index.min()} → {df.index.max()}")
    return df


def _clip_physical_bounds(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Clip values to physically valid ranges."""
    total_clipped: int = 0
    for col, (lower, upper) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        n_below = int((df[col] < lower).sum())
        n_above = int((df[col] > upper).sum())
        n_col_clipped = n_below + n_above
        if n_col_clipped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.debug(f"  {col}: clipped {n_below} below {lower}, {n_above} above {upper}")
            total_clipped += n_col_clipped

    if total_clipped > 0:
        logger.info(f"[3/7] Clipped {total_clipped} values to physical bounds")
    else:
        logger.info("[3/7] All values within physical bounds")

    return df, total_clipped


def _handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
) -> tuple[pd.DataFrame, int]:
    """Detect and replace outliers with NaN (will be interpolated later).

    Args:
        df: Input DataFrame.
        method: 'iqr' (default) or 'zscore'.
        threshold: IQR multiplier or z-score threshold.

    Returns:
        Tuple of (cleaned DataFrame, total outliers replaced).
    """
    total_outliers: int = 0
    numeric_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]

    for col in numeric_cols:
        if method == "iqr":
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        elif method == "zscore":
            mean = float(df[col].mean())
            std = float(df[col].std())
            lower = mean - threshold * std
            upper = mean + threshold * std
        else:
            raise ValueError(f"Unknown outlier method: {method}. Use 'iqr' or 'zscore'.")

        mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(mask.sum())
        if n_outliers > 0:
            df.loc[mask, col] = np.nan  # Replace with NaN → interpolate later
            logger.debug(f"  {col}: {n_outliers} outliers → NaN (range: [{lower:.1f}, {upper:.1f}])")
            total_outliers += n_outliers

    if total_outliers > 0:
        logger.info(f"[4/7] Replaced {total_outliers} outliers with NaN (method={method}, threshold={threshold})")
    else:
        logger.info("[4/7] No outliers detected")

    return df, total_outliers


def _resample(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Resample to regular frequency using mean aggregation."""
    n_before = len(df)
    df = df.resample(freq).mean()
    n_after = len(df)
    logger.info(f"[5/7] Resampled: {n_before:,} → {n_after:,} rows (freq={freq})")
    return df


def _interpolate_gaps(
    df: pd.DataFrame,
    method: str = "linear",
    max_gap: str = "2h",
) -> tuple[pd.DataFrame, int]:
    """Interpolate missing values, respecting max gap size.

    Args:
        df: DataFrame with potential NaN values.
        method: Interpolation method (linear, time, cubic).
        max_gap: Maximum gap to interpolate across.

    Returns:
        Tuple of (interpolated DataFrame, number of values filled).
    """
    n_nan_before = df.isna().sum().sum()

    if n_nan_before == 0:
        logger.info("[6/7] No missing values to interpolate")
        return df, 0

    # Parse max_gap to number of periods
    max_gap_td = pd.Timedelta(max_gap)
    freq = pd.tseries.frequencies.to_offset(df.index.freq or pd.infer_freq(df.index))
    max_periods = int(max_gap_td / freq) if freq is not None else 2

    # Interpolate
    df = df.interpolate(method=method, limit=max_periods, limit_direction="forward")
    n_nan_after = df.isna().sum().sum()
    n_filled = n_nan_before - n_nan_after

    logger.info(
        f"[6/7] Interpolated {n_filled:,} values (method={method}, max_gap={max_gap}, remaining NaN: {n_nan_after:,})"
    )

    return df, n_filled


def get_cleaning_stats(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict[str, Any]:
    """Compare raw vs cleaned data for reporting."""
    stats: dict[str, Any] = {
        "raw_rows": len(df_raw),
        "clean_rows": len(df_clean),
        "pct_retained": float(round(len(df_clean) / len(df_raw) * 100, 1)),
        "columns": {},
    }

    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df_raw.columns and col in df_clean.columns:
            raw_col = df_raw[col].dropna()
            clean_col = df_clean[col].dropna() if col in df_clean.columns else pd.Series(dtype=float)
            stats["columns"][col] = {
                "raw_mean": float(round(float(raw_col.mean()), 2)),
                "clean_mean": float(round(float(clean_col.mean()), 2)),
                "raw_std": float(round(float(raw_col.std()), 2)),
                "clean_std": float(round(float(clean_col.std()), 2)),
            }

    return stats
