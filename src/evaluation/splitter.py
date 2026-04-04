"""Data splitter for time series — temporal split, NO shuffle.

Per SKILL.md §6.3: Walk-Forward Validation with Expanding Window.
Per configs/base_config.yaml: train=80%, val=10%, test=10%.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.data.loader import TARGET_COL


def temporal_train_val_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data temporally: oldest → train → val → test → newest.

    NEVER shuffles data — per SKILL.md §6.3.

    Args:
        df: Feature-rich DataFrame with DatetimeIndex.
        target_col: Target column name.
        train_ratio: Fraction for training (default 0.8).
        val_ratio: Fraction for validation (default 0.1).
        test_ratio: Fraction for test (default 0.1).

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    # Separate X and y
    feature_cols = [c for c in df.columns if c != target_col]
    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]
    X_test = df_test[feature_cols]
    y_train = df_train[target_col]
    y_val = df_val[target_col]
    y_test = df_test[target_col]

    logger.info("Temporal Split (NO shuffle):")
    logger.info(f"  Train: {len(X_train):,} rows [{df_train.index[0]} → {df_train.index[-1]}]")
    logger.info(f"  Val:   {len(X_val):,} rows [{df_val.index[0]} → {df_val.index[-1]}]")
    logger.info(f"  Test:  {len(X_test):,} rows [{df_test.index[0]} → {df_test.index[-1]}]")

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_naive_predictions(y_series: pd.Series, horizon: int = 1) -> dict[str, np.ndarray]:
    """Create naive baseline predictions for MASE calculation.

    Per SKILL.md Level 0:
    - Persistence: y_pred(t) = y(t - horizon)
    - Seasonal Naive: y_pred(t) = y(t - 24)  (daily cycle)
    - Mean: y_pred(t) = mean(y_train)

    Args:
        y_series: Target series.
        horizon: Forecast horizon in hours.

    Returns:
        Dictionary of naive method name → predictions array.
    """
    y = y_series.values

    naive_preds = {
        "persistence": np.roll(y, horizon),  # Shift forward by horizon
        "seasonal_24h": np.roll(y, 24),  # Same hour yesterday
        "mean": np.full_like(y, np.mean(y), dtype=float),
    }

    # Fix rolled values at the start (those are invalid)
    for key in ["persistence", "seasonal_24h"]:
        shift = 24 if key == "seasonal_24h" else horizon
        naive_preds[key][:shift] = np.nan

    return naive_preds
