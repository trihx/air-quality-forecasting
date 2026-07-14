"""Tests for evaluation metrics and data splitter."""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.metrics import evaluate_forecast, mae, mape, mase, r2_score, rmse, smape
from src.evaluation.splitter import create_naive_predictions, temporal_train_val_test_split

# ============================================================
# Metrics Tests
# ============================================================


class TestMAE:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == 1.0


class TestRMSE:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_penalizes_large_errors(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 6.0])  # One big error
        assert rmse(y_true, y_pred) > mae(y_true, y_pred)


class TestMAPE:
    def test_perfect_predictions(self):
        y = np.array([10.0, 20.0, 30.0])
        assert mape(y, y) == 0.0

    def test_skips_near_zero(self):
        y_true = np.array([0.5, 10.0, 20.0])  # 0.5 < threshold=1.0
        y_pred = np.array([5.0, 10.0, 20.0])
        result = mape(y_true, y_pred, threshold=1.0)
        assert result == 0.0  # Only non-zero values considered, which are perfect


class TestSMAPE:
    def test_symmetric(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 18.0])
        # sMAPE should give same result when swapping true/pred
        assert abs(smape(y_true, y_pred) - smape(y_pred, y_true)) < 1e-10


class TestMASE:
    def test_better_than_naive(self):
        y_true = np.array([10.0, 12.0, 14.0, 16.0])
        y_pred = np.array([10.5, 12.2, 13.8, 15.9])  # Good predictions
        y_naive = np.array([8.0, 10.0, 12.0, 14.0])  # Persistence
        result = mase(y_true, y_pred, y_naive)
        assert result < 1.0  # Better than naive

    def test_same_as_naive(self):
        y_true = np.array([10.0, 12.0, 14.0])
        y_naive = np.array([8.0, 10.0, 12.0])
        result = mase(y_true, y_naive, y_naive)
        assert result == 1.0


class TestR2:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, y) == 1.0

    def test_mean_model(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_mean = np.full(3, 2.0)
        assert abs(r2_score(y_true, y_mean)) < 1e-10  # R² = 0 for mean model


class TestEvaluateForecast:
    def test_returns_all_metrics(self):
        y_true = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        y_pred = np.array([10.5, 12.2, 13.8, 15.9, 18.1])
        y_naive = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
        result = evaluate_forecast(y_true, y_pred, y_naive, model_name="Test")
        assert "mae" in result
        assert "rmse" in result
        assert "mase" in result
        assert "r2" in result
        assert "pass_naive" in result


# ============================================================
# Splitter Tests
# ============================================================


class TestTemporalSplit:
    @pytest.fixture()
    def sample_df(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "pm25": np.random.normal(15, 5, 100).clip(1, 50),
            },
            index=dates,
        )

    def test_split_sizes(self, sample_df):
        X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(sample_df)
        assert len(X_train) == 80
        assert len(X_val) == 10
        assert len(X_test) == 10

    def test_no_shuffle(self, sample_df):
        """Train must come before val, val before test."""
        X_train, X_val, X_test, _, _, _ = temporal_train_val_test_split(sample_df)
        assert X_train.index[-1] < X_val.index[0]
        assert X_val.index[-1] < X_test.index[0]

    def test_target_not_in_features(self, sample_df):
        X_train, _, _, _, _, _ = temporal_train_val_test_split(sample_df)
        assert "pm25" not in X_train.columns


class TestNaivePredictions:
    def test_persistence(self):
        y = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
        preds = create_naive_predictions(y, horizon=1)
        assert "persistence" in preds
        # After roll, persistence[1:] should be [10, 12, 14, 16]
        np.testing.assert_array_equal(preds["persistence"][1:], [10, 12, 14, 16])

    def test_seasonal_24h(self):
        y = pd.Series(range(48), dtype=float)
        preds = create_naive_predictions(y, horizon=1)
        assert "seasonal_24h" in preds
        # First 24 values are NaN (no history)
        assert np.all(np.isnan(preds["seasonal_24h"][:24]))

    def test_mean(self):
        y = pd.Series([10.0, 20.0, 30.0])
        preds = create_naive_predictions(y, horizon=1)
        assert preds["mean"][0] == 20.0
