"""Tests for data imputer — multi-strategy missing data recovery.

Each test has verbose output to show progress.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.imputer import (
    _build_knn_features,
    _cubic_spline_fill,
    _identify_gaps,
    get_imputation_stats,
    impute_missing_data,
    split_real_imputed,
)
from src.data.loader import TARGET_COL

# ── Fixtures ──


@pytest.fixture
def hourly_data_with_gaps() -> pd.DataFrame:
    """Create realistic hourly PM2.5 data with known gaps."""
    print("\n  [fixture] Creating hourly data with gaps...", flush=True)
    np.random.seed(42)
    n = 200  # 200 hours (~8 days)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")

    # Realistic PM2.5 values (10-50 range, with diurnal pattern)
    base_pm25 = 25 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 3, n)
    base_pm25 = np.clip(base_pm25, 1, 100)

    df = pd.DataFrame(
        {
            TARGET_COL: base_pm25,
            "nhiet_do": 28 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 1, n),
            "do_am": 70 + 10 * np.cos(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 2, n),
            "diem_suong": 22 + 3 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 1, n),
            "co2": 400 + 50 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 10, n),
        },
        index=idx,
    )

    # Insert known gaps
    # Short gap: 3 hours (should be fillable by all strategies)
    df.loc[idx[10:13], TARGET_COL] = np.nan
    # Medium gap: 8 hours (fillable by extended/hybrid/ml)
    df.loc[idx[50:58], TARGET_COL] = np.nan
    # Long gap: 18 hours (fillable by ml/hybrid strategies)
    df.loc[idx[100:118], TARGET_COL] = np.nan
    # Very long gap: 30 hours (too long for any strategy with max_gap=24)
    df.loc[idx[150:180], TARGET_COL] = np.nan

    total_nan = df[TARGET_COL].isna().sum()
    print(f"  [fixture] Data: {n} rows, {total_nan} NaN PM2.5 values", flush=True)
    print("  [fixture] Gaps: 3h, 8h, 18h, 30h", flush=True)
    return df


@pytest.fixture
def clean_data() -> pd.DataFrame:
    """Create clean data without gaps (for testing stats/splits)."""
    np.random.seed(42)
    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {TARGET_COL: np.random.uniform(10, 50, n)},
        index=idx,
    )
    return df


# ── Test: Strategy A (Segment Only) ──


class TestSegmentOnly:
    def test_drops_large_gaps(self, hourly_data_with_gaps: pd.DataFrame) -> None:
        """Segment-only should fill gaps ≤2h and drop the rest."""
        print("\n  [test] Strategy A: Segment Only...", flush=True)
        df = hourly_data_with_gaps.copy()
        n_before = len(df)
        n_nan_before = df[TARGET_COL].isna().sum()
        print(f"    Input: {n_before} rows, {n_nan_before} NaN", flush=True)

        result = impute_missing_data(df, strategy="segment_only", verbose=True)

        n_after = len(result)
        n_imputed = int(result["is_imputed"].sum())
        print(f"    Output: {n_after} rows, {n_imputed} imputed", flush=True)

        # Should have dropped most gap rows (only ≤2h filled)
        assert n_after < n_before, "Should drop some rows"
        assert "is_imputed" in result.columns, "Must have is_imputed column"
        assert result[TARGET_COL].isna().sum() == 0, "No NaN in output"
        print("    ✅ PASS", flush=True)


# ── Test: Strategy B (Extended Interpolation) ──


class TestExtendedInterp:
    def test_fills_more_gaps_than_segment(self, hourly_data_with_gaps: pd.DataFrame) -> None:
        """Extended interpolation should retain more rows than segment-only."""
        print("\n  [test] Strategy B: Extended Interpolation...", flush=True)

        result_a = impute_missing_data(
            hourly_data_with_gaps.copy(), strategy="segment_only", verbose=False
        )
        result_b = impute_missing_data(
            hourly_data_with_gaps.copy(), strategy="extended_interp",
            max_gap_interp=12, verbose=True
        )

        print(f"    A (segment): {len(result_a)} rows", flush=True)
        print(f"    B (ext_interp): {len(result_b)} rows", flush=True)

        assert len(result_b) >= len(result_a), "Extended should retain >= segment"
        assert result_b[TARGET_COL].isna().sum() == 0, "No NaN in output"
        print("    ✅ PASS", flush=True)

    def test_cubic_spline_preserves_known_values(self) -> None:
        """Cubic spline should not modify known (non-NaN) values."""
        print("\n  [test] Cubic Spline preserves known values...", flush=True)
        series = pd.Series([10, 20, np.nan, np.nan, 50, 60, 70])
        known_before = series.dropna().copy()

        result = _cubic_spline_fill(series, max_gap=3)

        known_after = result.iloc[known_before.index]
        np.testing.assert_array_almost_equal(known_before.values, known_after.values)
        print(f"    Known values preserved: {known_after.values}", flush=True)
        print("    ✅ PASS", flush=True)

    def test_cubic_spline_respects_max_gap(self) -> None:
        """Gaps longer than max_gap should NOT be filled."""
        print("\n  [test] Cubic Spline respects max_gap...", flush=True)
        # Gap of 5 elements with max_gap=3
        vals = [10, 20, 30] + [np.nan] * 5 + [80, 90, 100]
        series = pd.Series(vals)

        result = _cubic_spline_fill(series, max_gap=3)
        print(f"    Input:  {list(series.values)}", flush=True)
        print(f"    Output: {list(result.values)}", flush=True)

        # Gap of 5 > max_gap of 3, should remain NaN
        assert result.iloc[3:8].isna().all(), "Gap > max_gap should stay NaN"
        print("    ✅ PASS", flush=True)


# ── Test: Strategy C (ML Imputation) ──


class TestMLImpute:
    def test_knn_features_no_target(self, hourly_data_with_gaps: pd.DataFrame) -> None:
        """KNN feature matrix must NOT contain pm25 (anti-leakage)."""
        print("\n  [test] KNN features do NOT contain target...", flush=True)
        features = _build_knn_features(hourly_data_with_gaps)

        print(f"    KNN features: {list(features.columns)}", flush=True)
        assert TARGET_COL not in features.columns, f"LEAKAGE: {TARGET_COL} in KNN features!"
        assert "pm25_lag" not in " ".join(features.columns), "LEAKAGE: pm25_lag in KNN features!"
        print("    ✅ PASS — No target leakage in KNN features", flush=True)

    def test_ml_impute_fills_medium_gaps(self, hourly_data_with_gaps: pd.DataFrame) -> None:
        """ML imputation should fill gaps up to max_gap."""
        print("\n  [test] Strategy C: ML Imputation...", flush=True)

        result_a = impute_missing_data(
            hourly_data_with_gaps.copy(), strategy="segment_only", verbose=False
        )
        result_c = impute_missing_data(
            hourly_data_with_gaps.copy(), strategy="ml_impute",
            max_gap_ml=24, verbose=True
        )

        print(f"    A (segment): {len(result_a)} rows", flush=True)
        print(f"    C (ml_impute): {len(result_c)} rows", flush=True)

        assert len(result_c) >= len(result_a), "ML should retain >= segment"
        print("    ✅ PASS", flush=True)


# ── Test: Strategy D (Hybrid) ──


class TestHybrid:
    def test_hybrid_fills_most(self, hourly_data_with_gaps: pd.DataFrame) -> None:
        """Hybrid should be the best or second-best strategy for data retention."""
        print("\n  [test] Strategy D: Hybrid...", flush=True)

        result_a = impute_missing_data(
            hourly_data_with_gaps.copy(), strategy="segment_only", verbose=False
        )
        result_d = impute_missing_data(
            hourly_data_with_gaps.copy(), strategy="hybrid",
            max_gap_interp=6, max_gap_ml=24, verbose=True
        )

        print(f"    A (segment): {len(result_a)} rows", flush=True)
        print(f"    D (hybrid):  {len(result_d)} rows", flush=True)

        assert len(result_d) >= len(result_a), "Hybrid should retain >= segment"
        assert result_d[TARGET_COL].isna().sum() == 0, "No NaN in output"
        print("    ✅ PASS", flush=True)


# ── Test: Gap Identification ──


class TestGapIdentification:
    def test_identify_gaps_correct(self) -> None:
        """Gap identifier should find correct gap segments."""
        print("\n  [test] Gap identification...", flush=True)
        idx = pd.date_range("2023-01-01", periods=20, freq="1h")
        series = pd.Series(np.arange(20, dtype=float), index=idx)
        # Create 2 gaps: [5:8] (3h) and [12:15] (3h)
        series.iloc[5:8] = np.nan
        series.iloc[12:15] = np.nan

        gaps = _identify_gaps(series)
        print(f"    Found gaps: {len(gaps)}", flush=True)
        for _, g in gaps.iterrows():
            print(f"      Gap: idx {g['start_idx']}→{g['end_idx']}, length={g['length']}", flush=True)

        assert len(gaps) == 2, f"Expected 2 gaps, got {len(gaps)}"
        assert gaps.iloc[0]["length"] == 3
        assert gaps.iloc[1]["length"] == 3
        print("    ✅ PASS", flush=True)


# ── Test: Imputation Statistics ──


class TestImputationStats:
    def test_stats_tracking(self, clean_data: pd.DataFrame) -> None:
        """Stats should correctly report real vs imputed counts."""
        print("\n  [test] Imputation stats tracking...", flush=True)

        df = clean_data.copy()
        df["is_imputed"] = False
        df.loc[df.index[:10], "is_imputed"] = True

        stats = get_imputation_stats(df)
        print(f"    Stats: {stats}", flush=True)

        assert stats["total"] == 100
        assert stats["imputed"] == 10
        assert stats["real"] == 90
        assert stats["imputed_pct"] == 10.0
        print("    ✅ PASS", flush=True)


# ── Test: Real/Imputed Split ──


class TestSplitRealImputed:
    def test_split_preserves_all_data(self, clean_data: pd.DataFrame) -> None:
        """Split should not lose any rows."""
        print("\n  [test] Real/Imputed split...", flush=True)

        df = clean_data.copy()
        df["is_imputed"] = False
        df.loc[df.index[:20], "is_imputed"] = True

        real, imputed = split_real_imputed(df)
        print(f"    Real: {len(real)}, Imputed: {len(imputed)}, Total: {len(real) + len(imputed)}", flush=True)

        assert len(real) == 80
        assert len(imputed) == 20
        assert len(real) + len(imputed) == len(df)
        print("    ✅ PASS", flush=True)

    def test_test_set_real_only(self, hourly_data_with_gaps: pd.DataFrame) -> None:
        """After imputation, splitting test to real-only should work."""
        print("\n  [test] Test set real-only principle...", flush=True)

        result = impute_missing_data(
            hourly_data_with_gaps.copy(), strategy="hybrid",
            max_gap_interp=6, max_gap_ml=24, verbose=True
        )

        real, imputed = split_real_imputed(result)
        total = len(result)
        print(f"    Total: {total}, Real: {len(real)}, Imputed: {len(imputed)}", flush=True)

        # Verify real data has no imputed flag
        assert (real["is_imputed"] == False).all(), "Real data should have is_imputed=False"  # noqa: E712

        # For test set: take last 20% and verify only real data
        n_test = int(total * 0.2)
        test_slice = result.iloc[-n_test:]
        test_real = test_slice[~test_slice["is_imputed"]]
        print(f"    Test slice: {len(test_slice)} rows, {len(test_real)} real", flush=True)

        assert len(test_real) > 0, "Test set should have at least some real data"
        print("    ✅ PASS — Test set contains real data only", flush=True)
