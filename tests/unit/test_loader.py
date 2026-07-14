"""Tests for data loader module."""

import numpy as np
import pandas as pd
import pytest
from src.data.loader import (
    DATETIME_COL,
    EXPECTED_COLUMNS,
    FEATURE_COLS,
    TARGET_COL,
    get_data_summary,
    load_raw_data,
    validate_dataset,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small sample DataFrame for testing."""
    n = 100
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "nhiet_do": rng.uniform(25, 35, n),
            "do_am": rng.uniform(50, 90, n),
            "diem_suong": rng.uniform(20, 30, n),
            "co2": rng.uniform(300, 800, n),
            "pm25": rng.uniform(5, 50, n),
            "ngay_tao": pd.date_range("2024-01-01", periods=n, freq="2min"),
        }
    )


class TestLoadRawData:
    """Tests for load_raw_data function."""

    def test_load_default_path(self) -> None:
        """Load data from default path succeeds."""
        df = load_raw_data()
        assert len(df) > 200_000
        assert set(EXPECTED_COLUMNS).issubset(set(df.columns))
        assert str(df[DATETIME_COL].dtype).startswith("datetime64")

    def test_load_sorted_by_datetime(self) -> None:
        """Data should be sorted by datetime after loading."""
        df = load_raw_data()
        assert df[DATETIME_COL].is_monotonic_increasing

    def test_load_correct_dtypes(self) -> None:
        """Numeric columns should be float64."""
        df = load_raw_data()
        for col in FEATURE_COLS + [TARGET_COL]:
            assert df[col].dtype == np.float64, f"{col} should be float64"

    def test_load_nonexistent_path(self) -> None:
        """Loading from nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("dataset/raw/nonexistent.csv")


class TestGetDataSummary:
    """Tests for get_data_summary function."""

    def test_summary_keys(self, sample_df: pd.DataFrame) -> None:
        """Summary should contain expected keys."""
        summary = get_data_summary(sample_df)
        assert "n_rows" in summary
        assert "n_cols" in summary
        assert "missing" in summary
        assert "stats" in summary
        assert "date_range" in summary

    def test_summary_stats(self, sample_df: pd.DataFrame) -> None:
        """Stats should have min, max, mean, median, std, skewness."""
        summary = get_data_summary(sample_df)
        for col in FEATURE_COLS + [TARGET_COL]:
            assert col in summary["stats"]
            stat = summary["stats"][col]
            assert "min" in stat
            assert "max" in stat
            assert "mean" in stat
            assert stat["min"] <= stat["median"] <= stat["max"]

    def test_summary_row_count(self, sample_df: pd.DataFrame) -> None:
        """Summary should report correct row count."""
        summary = get_data_summary(sample_df)
        assert summary["n_rows"] == 100


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_valid_data_no_issues(self, sample_df: pd.DataFrame) -> None:
        """Clean data should produce no issues."""
        issues = validate_dataset(sample_df)
        assert len(issues) == 0

    def test_detect_negative_values(self, sample_df: pd.DataFrame) -> None:
        """Should detect negative values."""
        df = sample_df.copy()
        df.loc[0, "pm25"] = -5.0
        issues = validate_dataset(df)
        assert any("Negative" in issue for issue in issues)

    def test_detect_duplicated_timestamps(self, sample_df: pd.DataFrame) -> None:
        """Should detect duplicated timestamps."""
        df = sample_df.copy()
        df.loc[1, "ngay_tao"] = df.loc[0, "ngay_tao"]
        issues = validate_dataset(df)
        assert any("Duplicated" in issue for issue in issues)

    def test_detect_large_gaps(self) -> None:
        """Should detect gaps > 1 hour."""
        dates = list(pd.date_range("2024-01-01", periods=50, freq="2min"))
        # Insert a 3-hour gap
        dates.append(dates[-1] + pd.Timedelta(hours=3))
        rng = np.random.default_rng(42)
        n = len(dates)
        df = pd.DataFrame(
            {
                "nhiet_do": rng.uniform(25, 35, n),
                "do_am": rng.uniform(50, 90, n),
                "diem_suong": rng.uniform(20, 30, n),
                "co2": rng.uniform(300, 800, n),
                "pm25": rng.uniform(5, 50, n),
                "ngay_tao": dates,
            }
        )
        issues = validate_dataset(df)
        assert any("gap" in issue.lower() for issue in issues)


class TestRealDataValidation:
    """Integration tests against the real dataset."""

    def test_real_data_loads(self) -> None:
        """Real dataset should load successfully."""
        df = load_raw_data()
        assert len(df) > 0

    def test_real_data_summary(self) -> None:
        """Real dataset summary should have reasonable values."""
        df = load_raw_data()
        summary = get_data_summary(df)

        # PM2.5 should be >= 0
        assert summary["stats"]["pm25"]["min"] >= 0
        # Temperature should be in tropical range
        assert summary["stats"]["nhiet_do"]["median"] > 20
        # Should have > 3 years of data
        assert summary["date_range"]["days"] > 1000

    def test_real_data_validate(self) -> None:
        """Real dataset validation should complete without crash."""
        df = load_raw_data()
        issues = validate_dataset(df)
        # Log issues but don't fail — we expect some gaps in real data
        for issue in issues:
            print(f"  ⚠️ {issue}")
