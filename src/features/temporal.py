"""Temporal feature engineering — lag, rolling, EWM features.

Implements SKILL.md §5.1 specifications.
Anti-leakage: All features use ONLY past data (shift ≥ 1).

v9 Enhancement: Segment-aware mode.
When segment_col is provided, all temporal operations (shift, rolling, ewm, diff)
are computed WITHIN each segment using groupby().transform(). This prevents
False Continuity — where lag features span across large data gaps.
"""

import pandas as pd
from loguru import logger

from src.data.loader import FEATURE_COLS, TARGET_COL

# Default feature specs from SKILL.md §5.1
# NOTE: These are in TIME STEPS, not hours. At 1h freq → 1 step = 1h.
# At 30min freq → 1 step = 30min. Callers should adjust accordingly.
LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 168]  # steps
ROLLING_WINDOWS = [3, 6, 12, 24, 48, 168]  # steps
ROLLING_FUNCS = ["mean", "std", "min", "max", "range"]
EWM_SPANS = [12, 24, 48]  # steps


def _shift_col(
    series: pd.Series,
    periods: int,
    segment_col_series: pd.Series | None = None,
) -> pd.Series:
    """Shift a series, optionally within segments.

    Args:
        series: The series to shift.
        periods: Number of periods to shift.
        segment_col_series: If provided, shift within each segment group.

    Returns:
        Shifted series.
    """
    if segment_col_series is not None:
        return series.groupby(segment_col_series).shift(periods)
    return series.shift(periods)


def _rolling_col(
    series: pd.Series,
    window: int,
    func: str,
    min_periods: int,
    segment_col_series: pd.Series | None = None,
) -> pd.Series:
    """Apply rolling aggregation, optionally within segments.

    Args:
        series: The (already shifted) series to roll over.
        window: Window size in steps.
        func: Aggregation function name.
        min_periods: Minimum number of observations required.
        segment_col_series: If provided, roll within each segment group.

    Returns:
        Rolling aggregation result.
    """
    def _apply_roll(s: pd.Series) -> pd.Series:
        roller = s.rolling(window=window, min_periods=min_periods)
        if func == "mean":
            return roller.mean()
        elif func == "std":
            return roller.std()
        elif func == "min":
            return roller.min()
        elif func == "max":
            return roller.max()
        elif func == "range":
            return roller.max() - roller.min()
        else:
            raise ValueError(f"Unknown rolling func: {func}")

    if segment_col_series is not None:
        return series.groupby(segment_col_series).transform(
            lambda s: _apply_roll(s)
        )
    return _apply_roll(series)


def _ewm_col(
    series: pd.Series,
    span: int,
    func: str,
    min_periods: int,
    segment_col_series: pd.Series | None = None,
) -> pd.Series:
    """Apply EWM aggregation, optionally within segments."""
    def _apply_ewm(s: pd.Series) -> pd.Series:
        ewmer = s.ewm(span=span, min_periods=min_periods)
        if func == "mean":
            return ewmer.mean()
        elif func == "std":
            return ewmer.std()
        else:
            raise ValueError(f"Unknown ewm func: {func}")

    if segment_col_series is not None:
        return series.groupby(segment_col_series).transform(
            lambda s: _apply_ewm(s)
        )
    return _apply_ewm(series)


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: list[int] | None = None,
    include_features: bool = True,
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Create lag features for target and optionally for feature columns.

    Args:
        df: DataFrame with DatetimeIndex.
        target_col: Target column to create lags for.
        lags: List of lag periods (in time steps). Defaults to LAG_FEATURES.
        include_features: Also create lags for FEATURE_COLS.
        segment_col: If provided, compute lags within each segment (prevents
            cross-segment contamination).

    Returns:
        DataFrame with lag features appended.
    """
    if lags is None:
        lags = LAG_FEATURES

    df = df.copy()
    cols_to_lag = [target_col]
    if include_features:
        cols_to_lag += [c for c in FEATURE_COLS if c in df.columns]

    seg_series = df[segment_col] if segment_col and segment_col in df.columns else None

    n_created = 0
    for col in cols_to_lag:
        for lag in lags:
            col_name = f"{col}_lag_{lag}s"
            df[col_name] = _shift_col(df[col], lag, seg_series)
            n_created += 1

    mode = f"segment-aware ({segment_col})" if seg_series is not None else "global"
    logger.info(f"Created {n_created} lag features (lags={lags}, mode={mode})")
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    windows: list[int] | None = None,
    funcs: list[str] | None = None,
    include_features: bool = False,
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Create rolling window statistics.

    Shift(1) ensures no data leakage — window ends at t-1.

    Args:
        df: DataFrame with DatetimeIndex.
        target_col: Target column.
        windows: Window sizes in steps.
        funcs: Aggregation functions.
        include_features: Also create rolling for FEATURE_COLS.
        segment_col: If provided, compute rolling within each segment.

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

    seg_series = df[segment_col] if segment_col and segment_col in df.columns else None

    n_created = 0
    for col in cols:
        # Shift(1) to prevent leakage: window uses [t-w, t-1], NOT including t
        shifted = _shift_col(df[col], 1, seg_series)
        for window in windows:
            for func in funcs:
                col_name = f"{col}_roll_{window}s_{func}"
                min_p = 2 if func == "std" else 1
                df[col_name] = _rolling_col(shifted, window, func, min_p, seg_series)
                n_created += 1

    mode = f"segment-aware ({segment_col})" if seg_series is not None else "global"
    logger.info(f"Created {n_created} rolling features (windows={windows}, funcs={funcs}, mode={mode})")
    return df


def create_ewm_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    spans: list[int] | None = None,
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Create Exponentially Weighted Moving features.

    Shift(1) to prevent leakage.

    Args:
        df: DataFrame with DatetimeIndex.
        target_col: Target column.
        spans: EWM span periods in steps.
        segment_col: If provided, compute EWM within each segment.

    Returns:
        DataFrame with EWM features appended.
    """
    if spans is None:
        spans = EWM_SPANS

    df = df.copy()
    seg_series = df[segment_col] if segment_col and segment_col in df.columns else None
    shifted = _shift_col(df[target_col], 1, seg_series)

    n_created = 0
    for span in spans:
        df[f"{target_col}_ewm_{span}s_mean"] = _ewm_col(shifted, span, "mean", 1, seg_series)
        df[f"{target_col}_ewm_{span}s_std"] = _ewm_col(shifted, span, "std", 2, seg_series)
        n_created += 2

    mode = f"segment-aware ({segment_col})" if seg_series is not None else "global"
    logger.info(f"Created {n_created} EWM features (spans={spans}, mode={mode})")
    return df


def create_diff_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Create rate-of-change / difference features.

    Domain-specific: PM2.5 rate of change is important for air quality forecasting.

    ANTI-LEAKAGE: Uses shift(1) BEFORE diff/pct_change.
    - Without shift: diff(1) = y[t] - y[t-1] → contains y[t] = LEAKAGE
    - With shift:    shift(1).diff(1) = y[t-1] - y[t-2] → only past values ✅

    Args:
        df: DataFrame with DatetimeIndex.
        target_col: Target column.
        segment_col: If provided, compute diffs within each segment.

    Returns:
        DataFrame with diff features appended.
    """
    df = df.copy()
    seg_series = df[segment_col] if segment_col and segment_col in df.columns else None

    # Shift first to prevent leakage: compute diff on PAST values only
    shifted = _shift_col(df[target_col], 1, seg_series)

    if seg_series is not None:
        # Segment-aware diff: groupby then diff within each segment
        df[f"{target_col}_diff_1s"] = shifted.groupby(seg_series).diff(1)
        df[f"{target_col}_diff_24s"] = shifted.groupby(seg_series).diff(24)
        df[f"{target_col}_pct_change_1s"] = shifted.groupby(seg_series).pct_change(1)
        df[f"{target_col}_pct_change_24s"] = shifted.groupby(seg_series).pct_change(24)
    else:
        # First-order differences (rate of change) — uses y[t-1] - y[t-2]
        df[f"{target_col}_diff_1s"] = shifted.diff(1)
        df[f"{target_col}_diff_24s"] = shifted.diff(24)
        # Percentage change — uses (y[t-1] - y[t-2]) / y[t-2]
        df[f"{target_col}_pct_change_1s"] = shifted.pct_change(1)
        df[f"{target_col}_pct_change_24s"] = shifted.pct_change(24)

    mode = f"segment-aware ({segment_col})" if seg_series is not None else "global"
    logger.info(f"Created 4 diff/pct_change features (shift(1) anti-leakage, mode={mode})")
    return df
