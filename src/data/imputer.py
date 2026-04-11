"""Data imputer — Multi-strategy missing data recovery for IoT time series.

Strategies:
    A. segment_only: Drop all gaps (current baseline)
    B. extended_interp: Linear/Cubic interpolation with extended max_gap
    C. ml_impute: KNN-based multivariate imputation
    D. hybrid: Tiered approach (Spline ≤6h, KNN 6-24h, Drop >24h)

CRITICAL RULE: Test data MUST be real (non-imputed) only.
Imputed data is labeled via `is_imputed` column for tracking.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import CubicSpline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from src.data.loader import FEATURE_COLS, TARGET_COL

# ── Strategy Type ──
Strategy = Literal["segment_only", "extended_interp", "ml_impute", "hybrid"]


def impute_missing_data(
    df: pd.DataFrame,
    strategy: Strategy = "hybrid",
    max_gap_interp: int = 6,
    max_gap_ml: int = 24,
    knn_neighbors: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply missing data imputation strategy to hourly-resampled data.

    Args:
        df: DataFrame with DatetimeIndex (hourly), containing NaN gaps.
            Expected columns: pm25, nhiet_do, do_am, diem_suong, co2.
        strategy: Imputation strategy to use.
        max_gap_interp: Max gap (hours) for interpolation (strategy B, D).
        max_gap_ml: Max gap (hours) for ML imputation (strategy C, D).
        knn_neighbors: Number of neighbors for KNN imputation.
        verbose: Print detailed progress.

    Returns:
        DataFrame with imputed values and `is_imputed` column.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"  [Imputer] {msg}", flush=True)

    _log(f"Strategy: {strategy}")
    _log(f"Input: {len(df):,} rows, {df[TARGET_COL].isna().sum():,} missing PM2.5")

    # Add tracking column
    df = df.copy()
    df["is_imputed"] = False

    # Mark which PM2.5 values are originally NaN (candidates for imputation)
    pm25_was_nan = df[TARGET_COL].isna()

    if strategy == "segment_only":
        df = _strategy_segment_only(df, verbose=verbose)
    elif strategy == "extended_interp":
        df = _strategy_extended_interp(df, max_gap=max_gap_interp, verbose=verbose)
    elif strategy == "ml_impute":
        df = _strategy_ml_impute(df, max_gap=max_gap_ml, knn_neighbors=knn_neighbors, verbose=verbose)
    elif strategy == "hybrid":
        df = _strategy_hybrid(
            df,
            max_gap_interp=max_gap_interp,
            max_gap_ml=max_gap_ml,
            knn_neighbors=knn_neighbors,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Mark imputed rows (was NaN but now has value)
    filled_mask = pm25_was_nan & df[TARGET_COL].notna()
    if "is_imputed" in df.columns:
        df.loc[filled_mask, "is_imputed"] = True

    # Drop remaining NaN (gaps too large for any strategy)
    n_remaining_nan = df[TARGET_COL].isna().sum()
    df = df.dropna(subset=[TARGET_COL])

    # Also drop rows where auxiliary features are NaN
    aux_cols = [c for c in FEATURE_COLS if c in df.columns]
    n_aux_nan = df[aux_cols].isna().any(axis=1).sum()
    if n_aux_nan > 0:
        # Interpolate auxiliary features (they're less critical)
        for col in aux_cols:
            df[col] = df[col].interpolate(method="linear", limit=24)
        df = df.dropna(subset=aux_cols)

    n_imputed = int(df["is_imputed"].sum())
    n_real = len(df) - n_imputed

    _log(f"Output: {len(df):,} rows ({n_real:,} real + {n_imputed:,} imputed)")
    _log(f"Dropped {n_remaining_nan:,} rows with unfillable gaps")
    _log(f"Imputed ratio: {n_imputed / len(df) * 100:.1f}%")

    logger.info(
        f"Imputation [{strategy}]: {len(df):,} rows "
        f"({n_real:,} real + {n_imputed:,} imputed, "
        f"{n_remaining_nan:,} dropped)"
    )

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy A: Segment Only (current baseline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _strategy_segment_only(
    df: pd.DataFrame,
    max_gap: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop all gaps > max_gap. Interpolate only very short gaps (≤2h)."""

    def _log(msg: str) -> None:
        if verbose:
            print(f"    [A-SegmentOnly] {msg}", flush=True)

    _log(f"Interpolating gaps ≤{max_gap}h, dropping rest")
    df[TARGET_COL] = df[TARGET_COL].interpolate(method="linear", limit=max_gap, limit_direction="forward")
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=max_gap, limit_direction="forward")

    n_still_nan = df[TARGET_COL].isna().sum()
    _log(f"Remaining NaN after interp: {n_still_nan:,}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy B: Extended Interpolation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _strategy_extended_interp(
    df: pd.DataFrame,
    max_gap: int = 12,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extend interpolation window to fill longer gaps."""

    def _log(msg: str) -> None:
        if verbose:
            print(f"    [B-ExtInterp] {msg}", flush=True)

    _log(f"Cubic spline interpolation, max_gap={max_gap}h")

    for col in [TARGET_COL] + FEATURE_COLS:
        if col not in df.columns:
            continue
        _log(f"  Interpolating {col}...")
        df[col] = _cubic_spline_fill(df[col], max_gap=max_gap)

    n_still_nan = df[TARGET_COL].isna().sum()
    _log(f"Remaining NaN: {n_still_nan:,}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy C: ML-based Imputation (KNN)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _strategy_ml_impute(
    df: pd.DataFrame,
    max_gap: int = 24,
    knn_neighbors: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """KNN imputation using auxiliary features (temperature, humidity, CO2).

    ANTI-LEAKAGE: Does NOT use pm25 lag features to impute pm25.
    Uses only: nhiet_do, do_am, diem_suong, co2, hour_of_day, day_of_week.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"    [C-MLImpute] {msg}", flush=True)

    # Identify gaps that are within max_gap limit
    gap_info = _identify_gaps(df[TARGET_COL])
    fillable = gap_info[gap_info["length"] <= max_gap]

    if len(fillable) == 0:
        _log("No fillable gaps within max_gap limit")
        return df

    _log(f"Found {len(fillable)} fillable gap segments (≤{max_gap}h)")
    _log(f"Total hours to fill: {fillable['length'].sum()}")

    # Build feature matrix for KNN
    _log("Building KNN feature matrix...")
    features_for_knn = _build_knn_features(df)

    # Fit KNN on complete rows, predict missing
    _log(f"Fitting KNN (k={knn_neighbors})...")
    df = _apply_knn_imputation(
        df,
        features_for_knn,
        gap_info=fillable,
        n_neighbors=knn_neighbors,
        verbose=verbose,
    )

    n_still_nan = df[TARGET_COL].isna().sum()
    _log(f"Remaining NaN after KNN: {n_still_nan:,}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy D: Hybrid Tiered (RECOMMENDED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _strategy_hybrid(
    df: pd.DataFrame,
    max_gap_interp: int = 6,
    max_gap_ml: int = 24,
    knn_neighbors: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Tiered approach:
    - Gap ≤ max_gap_interp: Cubic Spline interpolation
    - Gap max_gap_interp < x ≤ max_gap_ml: KNN imputation
    - Gap > max_gap_ml: Drop
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"    [D-Hybrid] {msg}", flush=True)

    # Phase 1: Cubic Spline for short gaps
    _log(f"Phase 1: Cubic Spline for gaps ≤{max_gap_interp}h")
    for col in [TARGET_COL] + FEATURE_COLS:
        if col in df.columns:
            df[col] = _cubic_spline_fill(df[col], max_gap=max_gap_interp)

    n_after_spline = df[TARGET_COL].isna().sum()
    _log(f"  After Spline: {n_after_spline:,} NaN remaining")

    # Phase 2: KNN for medium gaps
    gap_info = _identify_gaps(df[TARGET_COL])
    medium_gaps = gap_info[(gap_info["length"] > 0) & (gap_info["length"] <= max_gap_ml)]

    if len(medium_gaps) > 0:
        _log(f"Phase 2: KNN for {len(medium_gaps)} medium gaps ({medium_gaps['length'].sum()}h)")
        features_for_knn = _build_knn_features(df)
        df = _apply_knn_imputation(
            df,
            features_for_knn,
            gap_info=medium_gaps,
            n_neighbors=knn_neighbors,
            verbose=verbose,
        )
    else:
        _log("Phase 2: No medium gaps to fill")

    n_after_knn = df[TARGET_COL].isna().sum()
    _log(f"Phase 3: Dropping {n_after_knn:,} rows with gaps >{max_gap_ml}h")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _cubic_spline_fill(series: pd.Series, max_gap: int = 6) -> pd.Series:
    """Fill NaN gaps using Cubic Spline, respecting max_gap limit.

    Only fills gaps of length ≤ max_gap.
    """
    result = series.copy()
    is_nan = series.isna()

    if not is_nan.any():
        return result

    # Identify gap boundaries
    groups = (is_nan != is_nan.shift()).cumsum()
    gap_groups = groups[is_nan]

    if len(gap_groups) == 0:
        return result

    gap_lengths = gap_groups.value_counts()

    # Only fill short gaps
    fillable_groups = gap_lengths[gap_lengths <= max_gap].index

    # Get known (non-NaN) data points
    known_mask = ~is_nan
    if known_mask.sum() < 4:
        # Not enough points for cubic spline
        return result.interpolate(method="linear", limit=max_gap)

    # Build spline from known points
    known_idx = np.where(known_mask)[0]
    known_vals = series.iloc[known_idx].values

    try:
        cs = CubicSpline(known_idx, known_vals, extrapolate=False)
    except ValueError:
        # Fallback to linear if spline fails
        return result.interpolate(method="linear", limit=max_gap)

    # Fill only fillable gap groups
    for g in fillable_groups:
        gap_mask = groups == g
        gap_positions = np.where(gap_mask)[0]
        interpolated = cs(gap_positions)

        # Clip to non-negative (physical constraint for PM2.5)
        interpolated = np.clip(interpolated, 0, None)
        result.iloc[gap_positions] = interpolated

    return result


def _identify_gaps(series: pd.Series) -> pd.DataFrame:
    """Identify contiguous NaN gap segments.

    Returns:
        DataFrame with columns: group_id, start_idx, end_idx, length
    """
    is_nan = series.isna()
    if not is_nan.any():
        return pd.DataFrame(columns=["group_id", "start_idx", "end_idx", "length"])

    groups = (is_nan != is_nan.shift()).cumsum()
    gap_groups = groups[is_nan]

    records = []
    for gid, indices in gap_groups.groupby(gap_groups).groups.items():
        idx_positions = [series.index.get_loc(i) for i in indices]
        records.append(
            {
                "group_id": gid,
                "start_idx": min(idx_positions),
                "end_idx": max(idx_positions),
                "length": len(idx_positions),
            }
        )

    return pd.DataFrame(records)


def _build_knn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for KNN imputation.

    ANTI-LEAKAGE: Uses only auxiliary sensors + temporal features.
    Does NOT use pm25, pm25_lag, or any target-derived features.
    """
    features = pd.DataFrame(index=df.index)

    # Auxiliary sensor features (multivariate context)
    for col in FEATURE_COLS:
        if col in df.columns:
            features[col] = df[col]

    # Temporal features (cyclic encoding)
    if hasattr(df.index, "hour"):
        features["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        features["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        features["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    return features


def _apply_knn_imputation(
    df: pd.DataFrame,
    knn_features: pd.DataFrame,
    gap_info: pd.DataFrame,
    n_neighbors: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply KNN imputation for specified gap segments.

    Uses auxiliary features + temporal features to predict missing PM2.5.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"      [KNN] {msg}", flush=True)

    df = df.copy()

    # Combine features with target for KNN
    knn_data = knn_features.copy()
    knn_data[TARGET_COL] = df[TARGET_COL]

    # Get indices to impute
    positions_to_fill = set()
    for _, gap in gap_info.iterrows():
        positions_to_fill.update(range(gap["start_idx"], gap["end_idx"] + 1))

    _log(f"Positions to fill: {len(positions_to_fill)}")

    # Standardize features (important for KNN distance)
    feature_cols = [c for c in knn_data.columns if c != TARGET_COL]

    # Fill feature NaNs with column mean for KNN to work
    for col in feature_cols:
        if knn_data[col].isna().any():
            knn_data[col] = knn_data[col].fillna(knn_data[col].mean())

    scaler = StandardScaler()
    knn_data[feature_cols] = scaler.fit_transform(knn_data[feature_cols])

    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputed_array = imputer.fit_transform(knn_data.values)

    # Extract imputed PM2.5 values
    target_col_idx = list(knn_data.columns).index(TARGET_COL)
    n_filled = 0

    for pos in positions_to_fill:
        if pos < len(df):
            imputed_val = imputed_array[pos, target_col_idx]
            # Clip to physical range
            imputed_val = max(0.0, min(500.0, imputed_val))
            df.iloc[pos, df.columns.get_loc(TARGET_COL)] = imputed_val
            n_filled += 1

    _log(f"Filled {n_filled} values via KNN")
    return df


def get_imputation_stats(df: pd.DataFrame) -> dict:
    """Get statistics about real vs imputed data in a DataFrame."""
    if "is_imputed" not in df.columns:
        return {"total": len(df), "real": len(df), "imputed": 0, "imputed_pct": 0.0}

    n_imputed = int(df["is_imputed"].sum())
    n_real = len(df) - n_imputed
    return {
        "total": len(df),
        "real": n_real,
        "imputed": n_imputed,
        "imputed_pct": round(n_imputed / len(df) * 100, 1) if len(df) > 0 else 0.0,
    }


def split_real_imputed(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into real and imputed portions.

    Use this to ensure test set only contains real data.
    """
    if "is_imputed" not in df.columns:
        return df, pd.DataFrame()

    real = df[~df["is_imputed"]].copy()
    imputed = df[df["is_imputed"]].copy()
    return real, imputed
