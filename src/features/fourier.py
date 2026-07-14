"""Fourier feature engineering — capture daily and weekly seasonality.

Fourier features linearize periodic patterns, allowing linear models
(ElasticNet, Lasso) to capture seasonality that would otherwise require
non-linear models.

Reference: Hyndman & Athanasopoulos (2018) "Forecasting: Principles and Practice"
           — Dynamic Harmonic Regression (Fourier terms).
Inspired by: research_code/03_feature_engineering.py (Phase 8)

ANTI-LEAKAGE: Fourier features are computed purely from timestamps,
              so they contain NO target information. Safe by design.
"""

import numpy as np
import pandas as pd
from loguru import logger


def create_fourier_features(
    df: pd.DataFrame,
    order: int = 3,
    include_daily: bool = True,
    include_weekly: bool = True,
) -> pd.DataFrame:
    """Create Fourier sin/cos features for daily and weekly seasonality.

    For each cycle (daily=24h, weekly=168h), creates `order` pairs of
    sin/cos features = 2 × order features per cycle.

    Args:
        df: DataFrame with DatetimeIndex.
        order: Number of Fourier harmonics (1 = fundamental, 2+ = overtones).
        include_daily: Include daily cycle (period = 24h).
        include_weekly: Include weekly cycle (period = 168h).

    Returns:
        DataFrame with Fourier features appended.
    """
    df = df.copy()
    idx = df.index

    n_created = 0

    # Time in hours: 0.0 to 23.99..
    t_hour = idx.hour + idx.minute / 60.0

    # 1. Daily Seasonality (Period = 24h)
    if include_daily:
        for k in range(1, order + 1):
            df[f"fourier_daily_sin_{k}"] = np.sin(2 * np.pi * k * t_hour / 24)
            df[f"fourier_daily_cos_{k}"] = np.cos(2 * np.pi * k * t_hour / 24)
            n_created += 2

    # 2. Weekly Seasonality (Period = 168h = 7 × 24)
    if include_weekly:
        t_week = idx.dayofweek * 24 + t_hour
        for k in range(1, order + 1):
            df[f"fourier_weekly_sin_{k}"] = np.sin(2 * np.pi * k * t_week / 168)
            df[f"fourier_weekly_cos_{k}"] = np.cos(2 * np.pi * k * t_week / 168)
            n_created += 2

    logger.info(
        f"Created {n_created} Fourier features "
        f"(order={order}, daily={include_daily}, weekly={include_weekly})"
    )
    return df
