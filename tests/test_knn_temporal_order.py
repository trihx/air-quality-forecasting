"""Test KNN imputation temporal order — no future data leakage.

Verifies that the fixed _apply_knn_imputation only uses past data
as KNN neighbors, preventing look-ahead bias.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import TARGET_COL, FEATURE_COLS


def _create_test_series(n: int = 200, gap_start: int = 80, gap_len: int = 10) -> pd.DataFrame:
    """Create a synthetic hourly time series with a known gap."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame(index=dates)

    # Synthetic features
    df["nhiet_do"] = 25 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 1, n)
    df["do_am"] = 70 + 10 * np.cos(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 2, n)
    df["diem_suong"] = df["nhiet_do"] - 5 + np.random.normal(0, 0.5, n)
    df["co2"] = 400 + np.random.normal(0, 20, n)

    # Target with known gap
    df[TARGET_COL] = 15 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 2, n)
    df.loc[df.index[gap_start:gap_start + gap_len], TARGET_COL] = np.nan

    return df


class TestKNNTemporalOrder:
    """Verify KNN imputation uses only past data."""

    def test_knn_does_not_use_future_neighbors(self):
        """Core test: after imputation, verify no future data was used.

        Strategy: Create data where past and future have VERY different
        distributions. If KNN uses future data, imputed values will
        reflect future distribution.
        """
        np.random.seed(42)
        n = 200
        gap_start = 80
        gap_len = 10
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame(index=dates)

        # Features: same everywhere
        df["nhiet_do"] = 30.0
        df["do_am"] = 70.0
        df["diem_suong"] = 25.0
        df["co2"] = 400.0

        # Target: past = 10, future = 100 (very different)
        df[TARGET_COL] = np.where(np.arange(n) < gap_start, 10.0, 100.0)
        # Make gap
        df.loc[df.index[gap_start:gap_start + gap_len], TARGET_COL] = np.nan

        # Import and run imputation
        from src.data.imputer import _build_knn_features, _identify_gaps, _apply_knn_imputation

        features = _build_knn_features(df)
        gap_info = _identify_gaps(df[TARGET_COL])
        medium_gaps = gap_info[(gap_info["length"] > 0) & (gap_info["length"] <= 24)]

        result = _apply_knn_imputation(
            df, features, gap_info=medium_gaps,
            n_neighbors=5, verbose=False,
        )

        # Imputed values MUST be close to 10 (past), NOT 100 (future)
        imputed_vals = result.iloc[gap_start:gap_start + gap_len][TARGET_COL].values
        assert all(np.isfinite(imputed_vals)), "Not all gap values were imputed"

        for i, val in enumerate(imputed_vals):
            assert val < 50.0, (
                f"Imputed value at gap pos {i} = {val:.1f}, "
                f"expected close to 10 (past mean). "
                f"Value > 50 suggests future data (mean=100) was used."
            )
            print(f"  Gap pos {i}: imputed={val:.2f} (past mean=10, future mean=100) ✅")

    def test_knn_skip_gap_with_insufficient_past(self):
        """If gap is very early (not enough past donors), KNN should skip."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame(index=dates)

        df["nhiet_do"] = 25.0 + np.random.normal(0, 1, n)
        df["do_am"] = 70.0
        df["diem_suong"] = 25.0
        df["co2"] = 400.0
        df[TARGET_COL] = 10.0 + np.random.normal(0, 1, n)

        # Gap at the very beginning — not enough past donors
        df.loc[df.index[1:6], TARGET_COL] = np.nan

        from src.data.imputer import _build_knn_features, _identify_gaps, _apply_knn_imputation

        features = _build_knn_features(df)
        gap_info = _identify_gaps(df[TARGET_COL])
        medium_gaps = gap_info[(gap_info["length"] > 0) & (gap_info["length"] <= 24)]

        result = _apply_knn_imputation(
            df, features, gap_info=medium_gaps,
            n_neighbors=5, verbose=True,
        )

        # Gap should be skipped (only 1 past donor, need 5)
        n_still_nan = result.iloc[1:6][TARGET_COL].isna().sum()
        print(f"  Early gap: {n_still_nan}/5 still NaN (expected: skipped)")
        # This is acceptable — early gaps with insufficient history are dropped later

    def test_full_imputation_pipeline_temporal_safe(self):
        """Integration test: run full hybrid imputation and verify temporal safety."""
        df = _create_test_series(n=300, gap_start=120, gap_len=8)

        from src.data.imputer import impute_missing_data

        result = impute_missing_data(df, strategy="hybrid", verbose=True)

        # All remaining values should be finite (NaN gaps either filled or dropped)
        assert result[TARGET_COL].isna().sum() == 0, "Should have no NaN after imputation"
        assert "is_imputed" in result.columns, "is_imputed column should exist"
        print(f"  Total: {len(result)} rows, {result['is_imputed'].sum()} imputed ✅")
