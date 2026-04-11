"""Feature builder — orchestrates all feature creation and produces Marts-ready data.

Pipeline: Intermediate (cleaned) → Marts (feature-rich, train-ready)
Per SKILL.md §3.5 and §5.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.data.loader import FEATURE_COLS, TARGET_COL
from src.data.validator import DataValidator
from src.features.calendar import create_calendar_features
from src.features.fourier import create_fourier_features
from src.features.temporal import (
    create_diff_features,
    create_ewm_features,
    create_lag_features,
    create_rolling_features,
)


def build_features(
    df: pd.DataFrame,
    lag_hours: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    rolling_funcs: list[str] | None = None,
    ewm_spans: list[int] | None = None,
    include_feature_lags: bool = True,
    include_feature_rolling: bool = False,
    include_fourier: bool = True,
    fourier_order: int = 3,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Build all features and produce Marts-ready DataFrame.

    Pipeline:
    1. Calendar features (hour, month, cyclical)
    1.5. Fourier features (daily + weekly seasonality)
    2. Lag features (target + optionally features)
    3. Rolling window statistics
    4. EWM features
    5. Diff / rate-of-change features
    6. Domain-specific features
    7. Drop NaN rows (from lag/rolling warmup)

    Args:
        df: Cleaned DataFrame with DatetimeIndex (from Intermediate layer).
        lag_hours: Custom lag periods. None = use defaults from SKILL.md.
        rolling_windows: Custom rolling windows. None = use defaults.
        rolling_funcs: Custom aggregation functions. None = use defaults.
        include_fourier: Include Fourier features for seasonality.
        fourier_order: Number of Fourier harmonics (default 3 → 12 features).
        ewm_spans: Custom EWM spans. None = use defaults.
        include_feature_lags: Create lags for FEATURE_COLS too.
        include_feature_rolling: Create rolling for FEATURE_COLS too.
        drop_na: Drop rows with NaN from warmup period.

    Returns:
        Marts-ready DataFrame with all features.
    """
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline Started")
    logger.info("=" * 60)

    n_before = len(df)
    n_cols_before = len(df.columns)

    # 1. Calendar features
    df = create_calendar_features(df)

    # 1.5. Fourier features (daily + weekly seasonality)
    if include_fourier:
        df = create_fourier_features(df, order=fourier_order)

    # 2. Lag features
    df = create_lag_features(
        df,
        target_col=TARGET_COL,
        lags=lag_hours,
        include_features=include_feature_lags,
    )

    # 3. Rolling features
    df = create_rolling_features(
        df,
        target_col=TARGET_COL,
        windows=rolling_windows,
        funcs=rolling_funcs,
        include_features=include_feature_rolling,
    )

    # 4. EWM features
    df = create_ewm_features(df, target_col=TARGET_COL, spans=ewm_spans)

    # 5. Diff features
    df = create_diff_features(df, target_col=TARGET_COL)

    # 6. Domain-specific features
    df = _create_domain_features(df)

    # 7. Drop warmup NaN rows
    n_nan_rows = int(df.isna().any(axis=1).sum())
    if drop_na and n_nan_rows > 0:
        df = df.dropna()
        logger.info(f"Dropped {n_nan_rows} warmup rows with NaN")

    n_after = len(df)
    n_cols_after = len(df.columns)

    logger.info("=" * 60)
    logger.info(
        f"Feature Engineering Complete: {n_cols_before} → {n_cols_after} columns, {n_before:,} → {n_after:,} rows"
    )
    logger.info("=" * 60)

    return df


def _create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-specific features for air quality per SKILL.md §5.3.

    ANTI-LEAKAGE: Uses pm25_lag_1h instead of pm25[t] for ratio and AQI features.
    pm25[t] is the target — using it as a feature causes direct leakage.
    """
    df = df.copy()

    # Determine PM2.5 source: use lag_1h (past value) to prevent leakage
    pm25_past_col = "pm25_lag_1h" if "pm25_lag_1h" in df.columns else None

    # CO2/PM2.5 ratio (interaction feature) — uses PAST pm25 value
    if "co2" in df.columns and pm25_past_col is not None:
        df["co2_pm25_ratio"] = df["co2"] / df[pm25_past_col].replace(0, float("nan"))
    elif "co2" in df.columns and TARGET_COL in df.columns:
        # Fallback: if lag not available, skip ratio to prevent leakage
        logger.warning("Skipping co2_pm25_ratio: pm25_lag_1h not available (would cause leakage)")

    # Temperature-Humidity interaction (Heat Index proxy) — OK, no target involved
    if "nhiet_do" in df.columns and "do_am" in df.columns:
        df["temp_humidity_interaction"] = df["nhiet_do"] * df["do_am"] / 100

    # PM2.5 AQI category (WHO breakpoints) — uses PAST pm25 value
    if pm25_past_col is not None:
        df["pm25_aqi_cat"] = pd.cut(
            df[pm25_past_col],
            bins=[0, 12, 35.4, 55.4, 150.4, 250.4, 500],
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True,
        ).astype(float)
    elif TARGET_COL in df.columns:
        logger.warning("Skipping pm25_aqi_cat: pm25_lag_1h not available (would cause leakage)")

    # ── Interaction features (inspired by RC) ──
    # PM2.5 × weather interactions — uses PAST pm25 value
    if pm25_past_col is not None:
        if "nhiet_do" in df.columns:
            df["pm25_x_temp"] = df[pm25_past_col] * df["nhiet_do"]
        if "do_am" in df.columns:
            df["pm25_x_humidity"] = df[pm25_past_col] * df["do_am"]
        # PM2.5 × time interactions
        if "hour" in df.columns:
            df["pm25_x_hour"] = df[pm25_past_col] * df["hour"]
        if "is_night" in df.columns:
            df["pm25_x_is_night"] = df[pm25_past_col] * df["is_night"]
        elif "hour" in df.columns:
            is_night = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
            df["pm25_x_is_night"] = df[pm25_past_col] * is_night

    # Weather × weather: Temperature - Dew Point (humidity stress proxy)
    if "nhiet_do" in df.columns and "diem_suong" in df.columns:
        df["temp_dew_diff"] = df["nhiet_do"] - df["diem_suong"]

    # PM2.5 relative to 24h rolling mean (deviation indicator)
    roll_24_col = "pm25_roll_24h_mean"
    if pm25_past_col is not None and roll_24_col in df.columns:
        df["pm25_relative_24h"] = df[pm25_past_col] / (df[roll_24_col] + 0.1)

    logger.info("Created domain + interaction features (anti-leakage: using pm25_lag_1h)")
    return df


def get_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Categorize feature columns by type for documentation.

    Returns:
        Dictionary with feature groups: original, lag, rolling, ewm, calendar, domain.
    """
    all_cols = list(df.columns)

    groups: dict[str, list[str]] = {
        "original": [c for c in FEATURE_COLS if c in all_cols],
        "target": [TARGET_COL] if TARGET_COL in all_cols else [],
        "lag": [c for c in all_cols if "_lag_" in c],
        "rolling": [c for c in all_cols if "_roll_" in c],
        "ewm": [c for c in all_cols if "_ewm_" in c],
        "diff": [c for c in all_cols if "_diff_" in c or "_pct_change_" in c],
        "fourier": [c for c in all_cols if c.startswith("fourier_")],
        "calendar": [
            c
            for c in all_cols
            if c
            in [
                "hour",
                "day_of_week",
                "day_of_month",
                "month",
                "is_weekend",
                "is_rush_hour",
                "season",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
                "dow_sin",
                "dow_cos",
            ]
        ],
        "interaction": [c for c in all_cols if c.startswith("pm25_x_") or c in ["temp_dew_diff", "pm25_relative_24h"]],
        "domain": [c for c in all_cols if c in ["co2_pm25_ratio", "temp_humidity_interaction", "pm25_aqi_cat"]],
    }

    for group_name, cols in groups.items():
        logger.debug(f"  {group_name}: {len(cols)} features")

    return groups


def save_marts_data(
    df: pd.DataFrame,
    output_path: str | Path = "dataset/processed/marts_features.csv",
    validate: bool = True,
) -> Path:
    """Save Marts-ready data with optional validation.

    Args:
        df: Feature-rich DataFrame.
        output_path: Path to save CSV.
        validate: Run Marts validation before saving.

    Returns:
        Path to saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if validate:
        validator = DataValidator()
        validator.validate_marts(df, target_col=TARGET_COL)
        report = validator.get_report()
        n_critical = sum(1 for r in report["checks"] if r["severity"].endswith("CRITICAL") and not r["passed"])
        if n_critical > 0:
            logger.error(f"❌ Marts validation failed with {n_critical} critical issues")
        else:
            logger.info("✅ Marts validation passed")

    df.to_csv(path)
    logger.info(f"💾 Marts data saved to {path} ({len(df):,} rows × {len(df.columns)} cols)")

    # Feature summary
    groups = get_feature_columns(df)
    summary: dict[str, Any] = {g: len(cols) for g, cols in groups.items()}
    logger.info(f"Feature summary: {summary}")

    return path
