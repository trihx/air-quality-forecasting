"""Unit tests for EDA Charts Audit & Multi-Tiered Zero-Dependency Caches.

Verifies:
1. Scatter dispersion multi-tier loading & 3-horizon data integrity (h=1, 6, 24).
2. Diurnal cycle statistics & 24-hour completeness.
3. Audit metrics loading for Forecastability and STL decomposition.
4. Calendar heatmap relative week monotonicity.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from src.eda_page import (
    RESEARCH_DIR,
    _get_diurnal_data,
    _get_scatter_dispersion_data,
)


def test_get_scatter_dispersion_data_returns_dataframe():
    """Verify _get_scatter_dispersion_data loads all 3 horizons (h1, h6, h24)."""
    df = _get_scatter_dispersion_data()
    assert df is not None, "Scatter dispersion DataFrame should not be None"
    assert not df.empty, "Scatter dispersion DataFrame should not be empty"

    required_cols = {"pm25", "h1", "h6", "h24"}
    assert required_cols.issubset(df.columns), f"Missing required columns in {df.columns}"

    # Sample size should be up to 2000
    assert len(df) <= 2000
    assert len(df) >= 100

    # Correlation checks: persistence decreases from 1h to 6h
    corr_1 = df["pm25"].corr(df["h1"])
    corr_6 = df["pm25"].corr(df["h6"])
    corr_24 = df["pm25"].corr(df["h24"])

    assert 0.70 <= corr_1 <= 1.0, f"Expected strong lag-1 autocorrelation, got {corr_1}"
    assert corr_6 < corr_1, f"Correlation at h=6 ({corr_6}) should be lower than h=1 ({corr_1})"
    assert 0.30 <= corr_24 <= 0.80, f"Expected 24h seasonal recovery correlation, got {corr_24}"


def test_get_scatter_dispersion_zero_dependency_fallback():
    """Verify fallback when candidate CSV files are missing."""
    with patch("pathlib.Path.exists", autospec=True) as mock_exists:
        # Simulate scatter_dispersion.json exists, but CSVs do not
        def fake_exists(path_obj: Path) -> bool:
            str_path = str(path_obj)
            return not ("cleaned_hourly.csv" in str_path or "marts_features.csv" in str_path)

        mock_exists.side_effect = fake_exists
        df = _get_scatter_dispersion_data()
        assert df is not None
        assert "h6" in df.columns


def test_get_diurnal_data_returns_24_hours():
    """Verify _get_diurnal_data provides complete 24-hour cycle statistics."""
    df = _get_diurnal_data()
    assert df is not None, "Diurnal DataFrame should not be None"
    assert len(df) == 24, f"Expected 24 hourly rows, got {len(df)}"

    expected_cols = {"hour", "mean", "q25", "q75", "median", "std"}
    assert expected_cols.issubset(df.columns)

    # Hours must be 0 to 23
    assert list(df["hour"]) == list(range(24))

    # All values must be physically valid for PM2.5 (0 to 500 ug/m3)
    assert (df["mean"] > 0).all()
    assert (df["mean"] < 200).all()
    assert (df["q25"] <= df["q75"]).all()


def test_audit_phase1_metrics_loaded():
    """Verify audit_phase1_metrics.json supplies forecastability and STL strengths."""
    metrics_path = RESEARCH_DIR / "eda" / "audit_phase1_metrics.json"
    assert metrics_path.exists(), "audit_phase1_metrics.json must exist"

    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)

    fc = data.get("forecastability", {})
    assert "forecastability_score" in fc
    assert 0.0 <= fc["forecastability_score"] <= 1.0

    stl = data.get("stl_decomposition", {})
    assert "trend_strength" in stl
    assert "seasonal_strength" in stl
    assert "residual_std" in stl
    assert stl["trend_strength"] > 0.5
    assert stl["residual_std"] > 0.0


def test_calendar_heatmap_week_formula_monotonic():
    """Verify relative week calculation formula prevents week 51 on Jan 1."""
    # Test for year 2023 where Jan 1 was a Sunday (ISO week 52 of 2022)
    start_of_year = pd.Timestamp(year=2023, month=1, day=1)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    week_indices = ((dates - start_of_year).days + start_of_year.weekday()) // 7

    # Week of Jan 1 must be 0
    assert week_indices[0] == 0, f"Jan 1 must be in week 0, got {week_indices[0]}"
    # Week of Dec 31 must be <= 52
    assert week_indices[-1] <= 52
    # Week indices must be monotonically non-decreasing
    assert (week_indices[1:] >= week_indices[:-1]).all()
