"""Level 0: Naive Baseline Models.

Per SKILL.md §6.1: Model phức tạp PHẢI chứng minh tốt hơn baseline.
Implements: Persistence, Seasonal Naive, Historical Mean.
"""

import numpy as np
import pandas as pd
from loguru import logger


class NaiveBaseline:
    """Base class for naive forecasting models."""

    name: str = "BaseNaive"

    def fit(self, y_train: pd.Series) -> "NaiveBaseline":
        """Store training data for baseline predictions."""
        self.y_train_ = y_train.copy()
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions (subclass must implement)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}()"


class PersistenceModel(NaiveBaseline):
    """Persistence (Naive) Forecast: y_pred(t) = y(t-1).

    Simplest possible baseline — predicts the last known value.
    """

    name = "Persistence"

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon

    def fit(self, y_train: pd.Series) -> "PersistenceModel":
        super().fit(y_train)
        self.last_value_ = float(y_train.iloc[-1])
        logger.info(f"  {self.name}: fitted, last_value={self.last_value_:.2f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using lag feature if available, otherwise last value."""
        lag_col = f"pm25_lag_{self.horizon}h"
        if lag_col in X.columns:
            return X[lag_col].values.astype(float)
        # Fallback: just repeat last known value
        return np.full(len(X), self.last_value_)


class SeasonalNaiveModel(NaiveBaseline):
    """Seasonal Naive: y_pred(t) = y(t - seasonal_period).

    For hourly data: seasonal_period=24 means same hour yesterday.
    """

    name = "SeasonalNaive"

    def __init__(self, seasonal_period: int = 24) -> None:
        self.seasonal_period = seasonal_period

    def fit(self, y_train: pd.Series) -> "SeasonalNaiveModel":
        super().fit(y_train)
        # Store last `seasonal_period` values for first predictions
        self.seasonal_values_ = y_train.iloc[-self.seasonal_period :].values.astype(float)
        logger.info(
            f"  {self.name}: fitted, period={self.seasonal_period}h, "
            f"seasonal_mean={float(np.mean(self.seasonal_values_)):.2f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using 24h lag feature if available."""
        lag_col = f"pm25_lag_{self.seasonal_period}h"
        if lag_col in X.columns:
            return X[lag_col].values.astype(float)
        # Fallback: cycle last seasonal_period values
        n = len(X)
        return np.tile(self.seasonal_values_, (n // self.seasonal_period + 1))[:n]


class MeanModel(NaiveBaseline):
    """Historical Mean: y_pred(t) = mean(y_train).

    The simplest statistical baseline.
    """

    name = "HistoricalMean"

    def fit(self, y_train: pd.Series) -> "MeanModel":
        super().fit(y_train)
        self.mean_ = float(y_train.mean())
        self.std_ = float(y_train.std())
        logger.info(f"  {self.name}: fitted, mean={self.mean_:.2f} ± {self.std_:.2f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class HourlyMeanModel(NaiveBaseline):
    """Hourly Mean: y_pred(t) = mean(y_train for hour=h).

    Better than global mean — captures diurnal pattern.
    """

    name = "HourlyMean"

    def fit(self, y_train: pd.Series) -> "HourlyMeanModel":
        super().fit(y_train)
        self.hourly_means_ = y_train.groupby(y_train.index.hour).mean().to_dict()
        self.global_mean_ = float(y_train.mean())
        logger.info(
            f"  {self.name}: fitted, "
            f"peak_hour={max(self.hourly_means_, key=self.hourly_means_.get)}, "  # type: ignore[arg-type]
            f"trough_hour={min(self.hourly_means_, key=self.hourly_means_.get)}"  # type: ignore[arg-type]
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "hour" in X.columns:
            return X["hour"].map(self.hourly_means_).fillna(self.global_mean_).values.astype(float)
        if isinstance(X.index, pd.DatetimeIndex):
            hours = X.index.hour
            return np.array([self.hourly_means_.get(h, self.global_mean_) for h in hours])
        return np.full(len(X), self.global_mean_)


def get_all_baselines() -> list[NaiveBaseline]:
    """Get all Level 0 baseline models."""
    return [
        PersistenceModel(horizon=1),
        SeasonalNaiveModel(seasonal_period=24),
        MeanModel(),
        HourlyMeanModel(),
    ]
