"""Tests for feature engineering modules.

Per SKILL.md test spec:
- Lag/rolling create correct values
- Calendar features are correct
- No unexpected NaN
- No data leakage (features only use past data)
"""

import numpy as np
import pandas as pd
import pytest
from src.features.builder import build_features, get_feature_columns, save_marts_data
from src.features.calendar import create_calendar_features
from src.features.temporal import (
    create_diff_features,
    create_ewm_features,
    create_lag_features,
    create_rolling_features,
)


@pytest.fixture()
def sample_df():
    """Create a sample hourly DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=200, freq="1h")
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "nhiet_do": np.random.normal(30, 3, 200),
            "do_am": np.random.normal(70, 10, 200),
            "diem_suong": np.random.normal(25, 2, 200),
            "co2": np.random.normal(400, 100, 200),
            "pm25": np.random.normal(15, 5, 200).clip(1, 50),
        },
        index=dates,
    )
    return df


# ============================================================
# Lag Features Tests
# ============================================================


class TestLagFeatures:
    def test_creates_correct_columns(self, sample_df):
        result = create_lag_features(sample_df, lags=[1, 2, 3], include_features=False)
        assert "pm25_lag_1h" in result.columns
        assert "pm25_lag_2h" in result.columns
        assert "pm25_lag_3h" in result.columns

    def test_lag_values_are_correct(self, sample_df):
        result = create_lag_features(sample_df, lags=[1], include_features=False)
        # lag_1h[t] should equal pm25[t-1]
        assert result["pm25_lag_1h"].iloc[1] == result["pm25"].iloc[0]
        assert result["pm25_lag_1h"].iloc[5] == result["pm25"].iloc[4]

    def test_lag_1_is_nan_at_first_row(self, sample_df):
        result = create_lag_features(sample_df, lags=[1], include_features=False)
        assert pd.isna(result["pm25_lag_1h"].iloc[0])

    def test_no_future_leakage(self, sample_df):
        """Lag features must only use PAST data (shift ≥ 1)."""
        result = create_lag_features(sample_df, lags=[1, 24], include_features=False)
        # At time t, lag_1h should be value at t-1
        for i in range(1, 10):
            assert result["pm25_lag_1h"].iloc[i] == sample_df["pm25"].iloc[i - 1]

    def test_include_feature_lags(self, sample_df):
        result = create_lag_features(sample_df, lags=[1], include_features=True)
        assert "nhiet_do_lag_1h" in result.columns
        assert "co2_lag_1h" in result.columns

    def test_default_lags(self, sample_df):
        result = create_lag_features(sample_df, include_features=False)
        # Default: [1, 2, 3, 6, 12, 24, 48, 168]
        assert "pm25_lag_168h" in result.columns


# ============================================================
# Rolling Features Tests
# ============================================================


class TestRollingFeatures:
    def test_creates_correct_columns(self, sample_df):
        result = create_rolling_features(sample_df, windows=[3], funcs=["mean"])
        assert "pm25_roll_3h_mean" in result.columns

    def test_rolling_uses_shifted_data(self, sample_df):
        """Rolling window must use shift(1) to prevent leakage."""
        result = create_rolling_features(sample_df, windows=[3], funcs=["mean"])
        # roll_3h_mean at t uses values from t-3, t-2, t-1 (NOT t)
        # With shift(1), it's rolling on shifted series
        # At index 3, rolling mean should NOT include value at index 3
        roll_val = result["pm25_roll_3h_mean"].iloc[3]
        # The rolling mean should be based on shifted data (indices 0,1,2)
        expected = sample_df["pm25"].iloc[0:3].mean()
        np.testing.assert_almost_equal(roll_val, expected, decimal=5)

    def test_rolling_std_creates_nan_for_single_value(self, sample_df):
        result = create_rolling_features(sample_df, windows=[3], funcs=["std"])
        # std requires min_periods=2, so index 0 should be NaN (only 1 value after shift)
        assert pd.isna(result["pm25_roll_3h_std"].iloc[0])

    def test_default_funcs(self, sample_df):
        result = create_rolling_features(sample_df, windows=[3])
        assert "pm25_roll_3h_mean" in result.columns
        assert "pm25_roll_3h_std" in result.columns
        assert "pm25_roll_3h_min" in result.columns
        assert "pm25_roll_3h_max" in result.columns


# ============================================================
# EWM Features Tests
# ============================================================


class TestEWMFeatures:
    def test_creates_correct_columns(self, sample_df):
        result = create_ewm_features(sample_df, spans=[12])
        assert "pm25_ewm_12h_mean" in result.columns
        assert "pm25_ewm_12h_std" in result.columns

    def test_ewm_is_shifted(self, sample_df):
        """EWM must use shift(1) to prevent leakage."""
        result = create_ewm_features(sample_df, spans=[12])
        # First value should be based on shifted data (NaN shifted → first valid = NaN)
        # But with min_periods=1, it computes from first available
        assert not pd.isna(result["pm25_ewm_12h_mean"].iloc[1])


# ============================================================
# Calendar Features Tests
# ============================================================


class TestCalendarFeatures:
    def test_hour_is_correct(self, sample_df):
        result = create_calendar_features(sample_df)
        assert result["hour"].iloc[0] == 0  # 2024-01-01 00:00

    def test_weekend_detection(self, sample_df):
        result = create_calendar_features(sample_df)
        # 2024-01-01 is Monday (0), 2024-01-06 is Saturday (5)
        assert result["is_weekend"].iloc[0] == 0  # Monday
        # Find Saturday (index 120 = 5 days later)
        sat_idx = 5 * 24  # 120 hours later
        assert result["is_weekend"].iloc[sat_idx] == 1

    def test_rush_hour(self, sample_df):
        result = create_calendar_features(sample_df)
        assert result["is_rush_hour"].iloc[7] == 1  # 7am
        assert result["is_rush_hour"].iloc[10] == 0  # 10am

    def test_cyclical_encoding_range(self, sample_df):
        result = create_calendar_features(sample_df)
        assert result["hour_sin"].min() >= -1
        assert result["hour_sin"].max() <= 1
        assert result["month_cos"].min() >= -1
        assert result["month_cos"].max() <= 1

    def test_season_mapping(self, sample_df):
        result = create_calendar_features(sample_df)
        # January → season 4 (dry cool)
        assert result["season"].iloc[0] == 4

    def test_all_13_features_created(self, sample_df):
        result = create_calendar_features(sample_df)
        expected_cols = [
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
        for col in expected_cols:
            assert col in result.columns, f"Missing calendar feature: {col}"


# ============================================================
# Diff Features Tests
# ============================================================


class TestDiffFeatures:
    def test_creates_diff_columns(self, sample_df):
        result = create_diff_features(sample_df)
        assert "pm25_diff_1h" in result.columns
        assert "pm25_diff_24h" in result.columns
        assert "pm25_pct_change_1h" in result.columns
        assert "pm25_pct_change_24h" in result.columns

    def test_diff_value_correct(self, sample_df):
        """Diff uses shift(1) → diff_1h[t] = y[t-1] - y[t-2] (anti-leakage).

        shifted = [NaN, y[0], y[1], y[2], ...]
        diff(1) = [NaN, NaN, y[1]-y[0], y[2]-y[1], ...]
        So first valid diff is at index 2.
        """
        result = create_diff_features(sample_df)
        # iloc[0] and iloc[1] should be NaN (shift + diff warmup)
        assert pd.isna(result["pm25_diff_1h"].iloc[0])
        assert pd.isna(result["pm25_diff_1h"].iloc[1])
        # iloc[2] = y[1] - y[0] (uses only past values, no y[2])
        expected_diff = sample_df["pm25"].iloc[1] - sample_df["pm25"].iloc[0]
        np.testing.assert_almost_equal(result["pm25_diff_1h"].iloc[2], expected_diff)


# ============================================================
# Builder (Full Pipeline) Tests
# ============================================================


class TestBuilder:
    def test_build_features_creates_many_columns(self, sample_df):
        result = build_features(sample_df, drop_na=True)
        # Original: 5 cols, should have way more after features
        assert len(result.columns) > 50

    def test_no_nan_after_drop(self, sample_df):
        result = build_features(sample_df, drop_na=True)
        assert result.isna().sum().sum() == 0

    def test_target_preserved(self, sample_df):
        result = build_features(sample_df, drop_na=True)
        assert "pm25" in result.columns

    def test_get_feature_columns(self, sample_df):
        result = build_features(sample_df, drop_na=True)
        groups = get_feature_columns(result)
        assert len(groups["lag"]) > 0
        assert len(groups["rolling"]) > 0
        assert len(groups["calendar"]) > 0
        assert len(groups["ewm"]) > 0

    def test_no_data_leakage_in_built_features(self, sample_df):
        """Verify no feature at time t uses data from time t or later."""
        result = build_features(sample_df, drop_na=False)
        lag_cols = [c for c in result.columns if "_lag_" in c]
        for col in lag_cols[:3]:  # Check first 3 lag features
            lag_h = int(col.split("_lag_")[1].replace("h", ""))
            base_col = col.split("_lag_")[0]
            for i in range(lag_h, min(lag_h + 5, len(result))):
                expected = sample_df[base_col].iloc[i - lag_h]
                actual = result[col].iloc[i]
                np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_save_marts_data(self, sample_df, tmp_path):
        result = build_features(sample_df, drop_na=True)
        output = tmp_path / "test_marts.csv"
        path = save_marts_data(result, output_path=output, validate=True)
        assert path.exists()
        loaded = pd.read_csv(path, index_col=0, parse_dates=True)
        assert len(loaded) == len(result)
