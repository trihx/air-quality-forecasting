"""Temporal feature engineering — lag, rolling, EWM features.

Implements SKILL.md §5.1 specifications.
Anti-leakage: All features use ONLY past data (shift ≥ 1).
"""

import pandas as pd
from loguru import logger

from src.data.loader import FEATURE_COLS, TARGET_COL

# Default feature specs from SKILL.md §5.1
LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 168]  # hours
ROLLING_WINDOWS = [3, 6, 12, 24, 48, 168]  # hours
ROLLING_FUNCS = ["mean", "std", "min", "max"]
EWM_SPANS = [12, 24, 48]  # hours


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: list[int] | None = None,
    include_features: bool = True,
) -> pd.DataFrame:
    """Create lag features for target and optionally for feature columns.

    Args:
        df: DataFrame with DatetimeIndex.
        target_col: Target column to create lags for.
        lags: List of lag periods (hours). Defaults to LAG_FEATURES.
        include_features: Also create lags for FEATURE_COLS.

    Returns:
        DataFrame with lag features appended.
    """
    if lags is None:
        lags = LAG_FEATURES

    df = df.copy()
    cols_to_lag = [target_col]
    if include_features:
        cols_to_lag += [c for c in FEATURE_COLS if c in df.columns]

    n_created = 0
    for col in cols_to_lag:
        for lag in lags:
            col_name = f"{col}_lag_{lag}h"
            df[col_name] = df[col].shift(lag)
            n_created += 1

    logger.info(f"Created {n_created} lag features (lags={lags})")
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    windows: list[int] | None = None,
    funcs: list[str] | None = None,
    include_features: bool = False,
) -> pd.DataFrame:
    """Create rolling window statistics.

    Shift(1) ensures no data leakage — window ends at t-1.

    Args:
        df: DataFrame with DatetimeIndex.
        target_col: Target column.
        windows: Window sizes in hours.
        funcs: Aggregation functions.
        include_features: Also create rolling for FEATURE_COLS.

    Returns:
        DataFrame with rolling features appended.
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    if funcs is None:
        funcs = ROLLING_FUNCS

    df = df.copy()
    cols = [target_col]
    if include_features:
        cols += [c for c in FEATURE_COLS if c in df.columns]

    n_created = 0
    for col in cols:
        # Shift(1) to prevent leakage: window uses [t-w, t-1], NOT including t
        shifted = df[col].shift(1)
        for window in windows:
            for func in funcs:
                col_name = f"{col}_roll_{window}h_{func}"
                if func == "mean":
                    df[col_name] = shifted.rolling(window=window, min_periods=1).mean()
                elif func == "std":
                    df[col_name] = shifted.rolling(window=window, min_periods=2).std()
                elif func == "min":
                    df[col_name] = shifted.rolling(window=window, min_periods=1).min()
                elif func == "max":
                    df[col_name] = shifted.rolling(window=window, min_periods=1).max()
                n_created += 1

    logger.info(f"Created {n_created} rolling features (windows={windows}, funcs={funcs})")
    return df


def create_ewm_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    spans: list[int] | None = None,
) -> pd.DataFrame:
    """Create Exponentially Weighted Moving features.

    Shift(1) to prevent leakage.

    Args:
        df: DataFrame with DatetimeIndex.
        target_col: Target column.
        spans: EWM span periods in hours.

    Returns:
        DataFrame with EWM features appended.
    """
    if spans is None:
        spans = EWM_SPANS

    df = df.copy()
    shifted = df[target_col].shift(1)

    n_created = 0
    for span in spans:
        df[f"{target_col}_ewm_{span}h_mean"] = shifted.ewm(span=span, min_periods=1).mean()
        df[f"{target_col}_ewm_{span}h_std"] = shifted.ewm(span=span, min_periods=2).std()
        n_created += 2

    logger.info(f"Created {n_created} EWM features (spans={spans})")
    return df


def create_diff_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Create rate-of-change / difference features.

    Domain-specific: PM2.5 rate of change is important for air quality forecasting.

    ANTI-LEAKAGE: Uses shift(1) BEFORE diff/pct_change.
    - Without shift: diff(1) = y[t] - y[t-1] → contains y[t] = LEAKAGE
    - With shift:    shift(1).diff(1) = y[t-1] - y[t-2] → only past values ✅

    Args:
        df: DataFrame with DatetimeIndex.
        target_col: Target column.

    Returns:
        DataFrame with diff features appended.
    """
    df = df.copy()

    # Shift first to prevent leakage: compute diff on PAST values only
    shifted = df[target_col].shift(1)

    # First-order differences (rate of change) — uses y[t-1] - y[t-2]
    df[f"{target_col}_diff_1h"] = shifted.diff(1)
    df[f"{target_col}_diff_24h"] = shifted.diff(24)

    # Percentage change — uses (y[t-1] - y[t-2]) / y[t-2]
    df[f"{target_col}_pct_change_1h"] = shifted.pct_change(1)
    df[f"{target_col}_pct_change_24h"] = shifted.pct_change(24)

    logger.info("Created 4 diff/pct_change features (shift(1) anti-leakage)")
    return df
