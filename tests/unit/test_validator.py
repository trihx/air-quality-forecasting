"""Tests for data validator module."""

import numpy as np
import pandas as pd
from src.data.validator import DataValidator, Severity


def _make_raw_df(n: int = 100) -> pd.DataFrame:
    """Create raw DataFrame for staging validation."""
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


def _make_clean_df(n: int = 5000) -> pd.DataFrame:
    """Create DatetimeIndex DataFrame for intermediate validation."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "nhiet_do": rng.uniform(25, 35, n),
            "do_am": rng.uniform(50, 90, n),
            "diem_suong": rng.uniform(20, 30, n),
            "co2": rng.uniform(300, 800, n),
            "pm25": rng.uniform(5, 50, n),
        },
        index=idx,
    )


class TestStagingValidation:
    def test_valid_data_passes_all(self) -> None:
        validator = DataValidator()
        df = _make_raw_df(n=1000)
        validator.validate_staging(df)
        assert not validator.has_critical_failures()

    def test_missing_columns_critical(self) -> None:
        validator = DataValidator()
        df = _make_raw_df().drop(columns=["pm25"])
        results = validator.validate_staging(df)
        col_check = next(r for r in results if r.check_name == "columns_exist")
        assert not col_check.passed
        assert col_check.severity == Severity.CRITICAL

    def test_empty_df_critical(self) -> None:
        validator = DataValidator()
        df = _make_raw_df(n=0)
        results = validator.validate_staging(df)
        empty_check = next(r for r in results if r.check_name == "non_empty")
        assert not empty_check.passed
        assert empty_check.severity == Severity.CRITICAL

    def test_pm25_out_of_range_warning(self) -> None:
        validator = DataValidator()
        df = _make_raw_df()
        df.loc[0, "pm25"] = 600.0  # Out of [0, 500]
        results = validator.validate_staging(df)
        range_check = next(r for r in results if r.check_name == "pm25_range")
        assert not range_check.passed
        assert range_check.severity == Severity.WARNING

    def test_high_missing_rate_warning(self) -> None:
        validator = DataValidator()
        df = _make_raw_df(n=100)
        df.loc[:14, "pm25"] = np.nan  # 15% missing
        results = validator.validate_staging(df)
        missing_check = next(r for r in results if r.check_name == "missing_rates")
        assert not missing_check.passed


class TestIntermediateValidation:
    def test_clean_data_passes(self) -> None:
        validator = DataValidator()
        df = _make_clean_df()
        validator.validate_intermediate(df)
        assert not validator.has_critical_failures()

    def test_nan_after_cleaning_critical(self) -> None:
        validator = DataValidator()
        df = _make_clean_df()
        df.iloc[0, 0] = np.nan  # Introduce NaN
        results = validator.validate_intermediate(df)
        nan_check = next(r for r in results if r.check_name == "no_nan")
        assert not nan_check.passed
        assert nan_check.severity == Severity.CRITICAL

    def test_non_monotonic_index_critical(self) -> None:
        validator = DataValidator()
        df = _make_clean_df()
        # Shuffle index to break monotonicity
        df = df.sample(frac=1)
        results = validator.validate_intermediate(df)
        mono_check = next(r for r in results if r.check_name == "monotonic_index")
        assert not mono_check.passed

    def test_sufficient_data_warning(self) -> None:
        validator = DataValidator()
        df = _make_clean_df(n=50)  # Too few
        results = validator.validate_intermediate(df)
        data_check = next(r for r in results if r.check_name == "sufficient_data")
        assert not data_check.passed


class TestMartsValidation:
    def test_no_leakage(self) -> None:
        validator = DataValidator()
        df = _make_clean_df()
        results = validator.validate_marts(df)
        leakage_check = next(r for r in results if r.check_name == "no_leakage")
        assert leakage_check.passed

    def test_detect_leakage(self) -> None:
        validator = DataValidator()
        df = _make_clean_df()
        df["leaked_pm25_copy"] = df["pm25"] * 1.0001  # Near-perfect copy
        results = validator.validate_marts(df)
        leakage_check = next(r for r in results if r.check_name == "no_leakage")
        assert not leakage_check.passed

    def test_zero_variance_feature_warning(self) -> None:
        validator = DataValidator()
        df = _make_clean_df()
        df["constant_feature"] = 42.0
        results = validator.validate_marts(df)
        var_check = next(r for r in results if r.check_name == "feature_variance")
        assert not var_check.passed


class TestValidatorReport:
    def test_report_structure(self) -> None:
        validator = DataValidator()
        df = _make_raw_df(n=1000)
        validator.validate_staging(df)
        report = validator.get_report()
        assert "total_checks" in report
        assert "passed" in report
        assert "failed" in report
        assert "critical_failures" in report
        assert "checks" in report
        assert len(report["checks"]) == report["total_checks"]

    def test_has_critical_failures(self) -> None:
        validator = DataValidator()
        df = _make_raw_df().drop(columns=["pm25"])
        validator.validate_staging(df)
        assert validator.has_critical_failures()


class TestRealDataValidation:
    def test_real_staging_validation(self) -> None:
        from src.data.loader import load_raw_data

        df = load_raw_data()
        validator = DataValidator()
        results = validator.validate_staging(df)
        # Schema should pass on real data
        schema_check = next(r for r in results if r.check_name == "schema_types")
        assert schema_check.passed
        # Columns should exist
        col_check = next(r for r in results if r.check_name == "columns_exist")
        assert col_check.passed
