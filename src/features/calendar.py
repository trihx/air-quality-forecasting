"""Calendar feature engineering — time-based categorical features.

Implements SKILL.md §5.2 specifications.
"""

import numpy as np
import pandas as pd
from loguru import logger


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar/temporal categorical features from DatetimeIndex.

    Features per SKILL.md §5.2:
    - hour, day_of_week, day_of_month, month
    - is_weekend, is_rush_hour, season
    - Cyclical encoding (sin/cos) for hour and month

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with calendar features appended.
    """
    df = df.copy()
    idx = df.index

    # Basic calendar features
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["day_of_month"] = idx.day
    df["month"] = idx.month

    # Binary features
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_rush_hour"] = idx.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Season mapping (Vietnam: tropical climate)
    # Dry season: Nov-Apr, Rainy season: May-Oct
    df["season"] = df["month"].map(_get_season)

    # Cyclical encoding (prevents discontinuity: 23h → 0h, Dec → Jan)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    n_features = 13  # 4 basic + 2 binary + 1 season + 6 cyclical
    logger.info(f"Created {n_features} calendar features")

    return df


def _get_season(month: int) -> int:
    """Map month to season (Vietnam tropical climate).

    0: Spring transition (Feb-Mar)
    1: Dry hot (Apr-May)
    2: Rainy early (Jun-Aug)
    3: Rainy late (Sep-Oct)
    4: Dry cool (Nov-Jan)
    """
    season_map = {
        1: 4,
        2: 0,
        3: 0,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
        10: 3,
        11: 4,
        12: 4,
    }
    return season_map.get(month, 0)
