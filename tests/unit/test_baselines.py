"""Tests for baseline models."""

import numpy as np
import pandas as pd
import pytest
from src.models.baselines import (
    HourlyMeanModel,
    MeanModel,
    PersistenceModel,
    SeasonalNaiveModel,
    get_all_baselines,
)


@pytest.fixture()
def sample_data():
    """Create sample train/test data with features."""
    np.random.seed(42)
    dates_train = pd.date_range("2024-01-01", periods=200, freq="1h")
    dates_test = pd.date_range("2024-01-09 08:00", periods=48, freq="1h")

    y_train = pd.Series(
        np.random.normal(15, 5, 200).clip(1, 50),
        index=dates_train,
        name="pm25",
    )

    X_test = pd.DataFrame(
        {
            "pm25_lag_1h": np.random.normal(15, 5, 48),
            "pm25_lag_24h": np.random.normal(15, 5, 48),
            "hour": dates_test.hour,
        },
        index=dates_test,
    )

    return y_train, X_test


class TestPersistenceModel:
    def test_uses_lag_feature(self, sample_data):
        y_train, X_test = sample_data
        model = PersistenceModel(horizon=1)
        model.fit(y_train)
        preds = model.predict(X_test)
        np.testing.assert_array_equal(preds, X_test["pm25_lag_1h"].values)

    def test_fallback_to_last_value(self, sample_data):
        y_train, X_test = sample_data
        X_no_lag = X_test.drop(columns=["pm25_lag_1h"])
        model = PersistenceModel(horizon=1)
        model.fit(y_train)
        preds = model.predict(X_no_lag)
        assert np.all(preds == y_train.iloc[-1])

    def test_output_shape(self, sample_data):
        y_train, X_test = sample_data
        model = PersistenceModel(horizon=1)
        model.fit(y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)


class TestSeasonalNaiveModel:
    def test_uses_24h_lag(self, sample_data):
        y_train, X_test = sample_data
        model = SeasonalNaiveModel(seasonal_period=24)
        model.fit(y_train)
        preds = model.predict(X_test)
        np.testing.assert_array_equal(preds, X_test["pm25_lag_24h"].values)


class TestMeanModel:
    def test_predicts_mean(self, sample_data):
        y_train, X_test = sample_data
        model = MeanModel()
        model.fit(y_train)
        preds = model.predict(X_test)
        expected = y_train.mean()
        np.testing.assert_almost_equal(preds[0], expected)

    def test_constant_predictions(self, sample_data):
        y_train, X_test = sample_data
        model = MeanModel()
        model.fit(y_train)
        preds = model.predict(X_test)
        assert np.all(preds == preds[0])


class TestHourlyMeanModel:
    def test_uses_hour_feature(self, sample_data):
        y_train, X_test = sample_data
        model = HourlyMeanModel()
        model.fit(y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        # Predictions should vary by hour
        assert not np.all(preds == preds[0])


class TestGetAllBaselines:
    def test_returns_4_models(self):
        models = get_all_baselines()
        assert len(models) == 4

    def test_all_have_name(self):
        for model in get_all_baselines():
            assert hasattr(model, "name")
            assert len(model.name) > 0
