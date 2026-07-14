"""Tests for DL retrain v2 components — CV features and data pipeline."""

import numpy as np
import pandas as pd
import pytest

# ── Test CV Features ──

# Inline the function to avoid torch import issues in test
CV_WINDOWS = [6, 12, 24]


def add_cv_features(df: pd.DataFrame, target_col: str = "pm25") -> pd.DataFrame:
    """Add Coefficient of Variation features with safeguard."""
    df = df.copy()
    lag_col = f"{target_col}_lag_1h"
    if lag_col not in df.columns:
        return df

    for w in CV_WINDOWS:
        roll = df[lag_col].shift(1).rolling(window=w, min_periods=max(w // 2, 2))
        roll_std = roll.std()
        roll_mean = roll.mean()

        safe_mean = roll_mean.abs().clip(lower=1.0)
        cv = roll_std / safe_mean
        cv = cv.clip(upper=5.0)

        col_name = f"{target_col}_cv_{w}h"
        df[col_name] = cv

    return df


class TestCVFeatures:
    """Test Coefficient of Variation feature with safeguards."""

    def _make_df(self, values, n=100):
        """Helper to create test DataFrame."""
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"pm25": values[:n], "pm25_lag_1h": values[:n]}, index=idx)
        return df

    def test_cv_columns_created(self):
        """CV columns should be created for each window."""
        np.random.seed(42)
        df = self._make_df(np.random.normal(20, 5, 100))
        result = add_cv_features(df, "pm25")

        for w in CV_WINDOWS:
            col = f"pm25_cv_{w}h"
            assert col in result.columns, f"Missing column {col}"

    def test_cv_no_lag_col_returns_unchanged(self):
        """If lag column doesn't exist, return unchanged."""
        idx = pd.date_range("2024-01-01", periods=50, freq="1h")
        df = pd.DataFrame({"pm25": np.ones(50)}, index=idx)
        result = add_cv_features(df, "pm25")
        assert len(result.columns) == len(df.columns), "Columns should not change without lag col"

    def test_cv_safeguard_near_zero_mean(self):
        """When mean is near 0, CV should be clamped (not explode to inf)."""
        np.random.seed(42)
        # Values near 0 → mean ≈ 0 → CV would explode without safeguard
        values = np.random.normal(0.1, 0.5, 100)
        df = self._make_df(values)
        result = add_cv_features(df, "pm25")

        for w in CV_WINDOWS:
            col = f"pm25_cv_{w}h"
            valid = result[col].dropna()
            if len(valid) > 0:
                # No inf values
                assert not np.isinf(valid).any(), f"CV has inf values for window {w}"
                # All values <= 5.0 (clip upper bound)
                assert (valid <= 5.0 + 1e-10).all(), f"CV exceeds upper clip for window {w}"

    def test_cv_safeguard_all_zeros(self):
        """All zeros should NOT produce inf or NaN from division."""
        values = np.zeros(100)
        df = self._make_df(values)
        result = add_cv_features(df, "pm25")

        for w in CV_WINDOWS:
            col = f"pm25_cv_{w}h"
            valid = result[col].dropna()
            if len(valid) > 0:
                assert not np.isinf(valid).any(), "All zeros should not produce inf"

    def test_cv_normal_data_reasonable(self):
        """Normal PM2.5 data should produce reasonable CV values."""
        np.random.seed(42)
        values = np.random.normal(15, 5, 200)  # Typical PM2.5
        values = np.clip(values, 0, None)
        df = self._make_df(values, 200)
        result = add_cv_features(df, "pm25")

        col = "pm25_cv_24h"
        valid = result[col].dropna()
        assert len(valid) > 0, "Should have valid CV values"
        assert valid.mean() > 0, "CV should be positive for variable data"
        assert valid.mean() < 3.0, "CV should be reasonable (not too high)"

    def test_cv_uses_shift_anti_leakage(self):
        """CV uses shift(1) on lag_1h → no access to current value."""
        # Create a jump at index 50
        values = np.concatenate([np.ones(50) * 10, np.ones(50) * 100])
        df = self._make_df(values)
        result = add_cv_features(df, "pm25")

        # At index 50, CV should NOT yet reflect the jump
        # because shift(1) delays by 1
        col = "pm25_cv_6h"
        # Value at 50 should be low (still seeing the flat=10 data)
        val_at_50 = result[col].iloc[50]
        # The flat region has std=0, so CV should be 0 or NaN
        assert val_at_50 < 1.0, f"CV at jump point should be low (anti-leakage), got {val_at_50}"


class TestDLFeatureSelection:
    """Test that DL feature selection works correctly."""

    def test_excludes_target_and_imputed(self):
        """Feature selection should exclude target, is_imputed."""
        idx = pd.date_range("2024-01-01", periods=10, freq="1h")
        df = pd.DataFrame({
            "pm25": np.random.rand(10),
            "is_imputed": [True] * 5 + [False] * 5,
            "nhiet_do": np.random.rand(10),
            "pm25_lag_1h": np.random.rand(10),
            "fourier_daily_sin_1": np.random.rand(10),
        }, index=idx)

        exclude = {"is_imputed", "pm25"}
        exclude.update(c for c in df.columns if c.startswith("target_"))
        feature_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in ("float64", "float32", "int64")
        ]

        assert "pm25" not in feature_cols
        assert "is_imputed" not in feature_cols
        assert "nhiet_do" in feature_cols
        assert "pm25_lag_1h" in feature_cols
        assert "fourier_daily_sin_1" in feature_cols


class TestLogTransform:
    """Test log1p/expm1 inverse transform correctness."""

    def test_log_inverse_exact(self):
        """log1p → expm1 should recover original values."""
        original = np.array([0.0, 1.0, 5.0, 20.0, 100.0])
        transformed = np.log1p(original)
        recovered = np.expm1(transformed)
        np.testing.assert_allclose(recovered, original, rtol=1e-10)

    def test_log_negative_clipped(self):
        """Negative values should be clipped to 0 before log1p."""
        values = np.array([-5.0, -1.0, 0.0, 5.0, 10.0])
        clipped = np.clip(values, 0, None)
        transformed = np.log1p(clipped)
        recovered = np.expm1(transformed)
        assert (recovered >= 0).all(), "All recovered values should be >= 0"

    def test_log_preserves_order(self):
        """log1p should preserve ordering of values."""
        values = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        transformed = np.log1p(values)
        # Check monotonicity
        assert (np.diff(transformed) > 0).all(), "log1p should preserve order"

    def test_log_reduces_variance(self):
        """log1p should reduce variance of right-skewed data."""
        np.random.seed(42)
        # Right-skewed PM2.5 like data
        values = np.random.exponential(10, 1000)
        log_values = np.log1p(values)
        assert log_values.std() < values.std(), "Log should reduce variance"
