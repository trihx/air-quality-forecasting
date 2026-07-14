"""Tests for data cleaner module."""

import numpy as np
import pandas as pd
import pytest
from src.data.cleaner import (
    PHYSICAL_BOUNDS,
    _clip_physical_bounds,
    _handle_outliers,
    _remove_duplicates,
    _resample,
    clean_data,
)
from src.data.loader import DATETIME_COL


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Create sample raw DataFrame mimicking real sensor data."""
    n = 500
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="2min")
    return pd.DataFrame(
        {
            "nhiet_do": rng.uniform(25, 35, n),
            "do_am": rng.uniform(50, 90, n),
            "diem_suong": rng.uniform(20, 30, n),
            "co2": rng.uniform(300, 800, n),
            "pm25": rng.uniform(5, 50, n),
            "ngay_tao": dates,
        }
    )


class TestRemoveDuplicates:
    def test_no_duplicates(self, raw_df: pd.DataFrame) -> None:
        result = _remove_duplicates(raw_df)
        assert len(result) == len(raw_df)

    def test_exact_duplicates_removed(self, raw_df: pd.DataFrame) -> None:
        df = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
        result = _remove_duplicates(df)
        assert len(result) == len(raw_df)

    def test_timestamp_duplicates_keeps_first(self, raw_df: pd.DataFrame) -> None:
        df = raw_df.copy()
        df.loc[1, DATETIME_COL] = df.loc[0, DATETIME_COL]
        result = _remove_duplicates(df)
        assert not result[DATETIME_COL].duplicated().any()


class TestClipPhysicalBounds:
    def test_no_clipping_needed(self, raw_df: pd.DataFrame) -> None:
        df = raw_df.set_index(DATETIME_COL)
        result, n_clipped = _clip_physical_bounds(df)
        assert n_clipped == 0

    def test_clips_negative_pm25(self, raw_df: pd.DataFrame) -> None:
        df = raw_df.copy()
        df.loc[0, "pm25"] = -10.0
        df = df.set_index(DATETIME_COL)
        result, n_clipped = _clip_physical_bounds(df)
        assert result["pm25"].min() >= PHYSICAL_BOUNDS["pm25"][0]
        assert n_clipped >= 1

    def test_clips_extreme_temperature(self, raw_df: pd.DataFrame) -> None:
        df = raw_df.copy()
        df.loc[0, "nhiet_do"] = 100.0  # Physically impossible
        df = df.set_index(DATETIME_COL)
        result, n_clipped = _clip_physical_bounds(df)
        assert result["nhiet_do"].max() <= PHYSICAL_BOUNDS["nhiet_do"][1]


class TestHandleOutliers:
    def test_iqr_method(self, raw_df: pd.DataFrame) -> None:
        df = raw_df.copy()
        df.loc[0, "pm25"] = 9999.0  # Extreme outlier
        df = df.set_index(DATETIME_COL)
        result, n_outliers = _handle_outliers(df, method="iqr", threshold=1.5)
        assert n_outliers > 0
        assert pd.isna(result.loc[result.index[0], "pm25"])

    def test_zscore_method(self, raw_df: pd.DataFrame) -> None:
        df = raw_df.copy()
        df.loc[0, "pm25"] = 9999.0
        df = df.set_index(DATETIME_COL)
        result, n_outliers = _handle_outliers(df, method="zscore", threshold=3.0)
        assert n_outliers > 0

    def test_invalid_method_raises(self, raw_df: pd.DataFrame) -> None:
        df = raw_df.set_index(DATETIME_COL)
        with pytest.raises(ValueError, match="Unknown outlier method"):
            _handle_outliers(df, method="invalid")


class TestResample:
    def test_resample_reduces_rows(self, raw_df: pd.DataFrame) -> None:
        df = raw_df.set_index(DATETIME_COL)
        result = _resample(df, freq="1h")
        # 500 rows at 2min → ~16h → should be ~17 hourly rows
        assert len(result) < len(df)

    def test_resample_regular_index(self, raw_df: pd.DataFrame) -> None:
        df = raw_df.set_index(DATETIME_COL)
        result = _resample(df, freq="1h")
        freq = pd.infer_freq(result.index)
        assert freq is not None


class TestCleanData:
    def test_full_pipeline(self, raw_df: pd.DataFrame) -> None:
        """Full cleaning pipeline should run without errors."""
        result = clean_data(raw_df)
        assert len(result) > 0
        assert isinstance(result.index, pd.DatetimeIndex)
        # Should have no NaN after full pipeline
        assert result.isna().sum().sum() == 0

    def test_full_pipeline_with_outliers(self, raw_df: pd.DataFrame) -> None:
        """Pipeline should handle data with outliers."""
        df = raw_df.copy()
        df.loc[0, "pm25"] = 9999.0
        df.loc[1, "nhiet_do"] = -50.0
        result = clean_data(df)
        assert len(result) > 0

    def test_pipeline_preserves_datetime_index(self, raw_df: pd.DataFrame) -> None:
        result = clean_data(raw_df)
        assert result.index.is_monotonic_increasing


class TestRealDataCleaning:
    """Integration test with real dataset."""

    def test_clean_real_data(self) -> None:
        from src.data.loader import load_raw_data

        df = load_raw_data()
        result = clean_data(df)
        assert len(result) > 0
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.is_monotonic_increasing
        # Should retain >50% of original rows after resampling
        assert len(result) > 5_000
