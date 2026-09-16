"""Unit tests for Phase 1 data pipeline audit and thesis alignment.

Verifies:
1. Expected raw columns, data types, and physical bounds.
2. Domain-based outlier strategy preserving extreme PM2.5 pollution events.
3. Multi-strategy Tiered Imputation (is_imputed tracking, past-only KNN donors).
4. Segment-aware contiguous block identification (False Continuity protection).
5. Feature engineering anti-leakage invariants (shift(1) rolling, lagged domain interactions).
6. Table 4.1 (ADF & KPSS Stationarity) and Table 3.4 (Data Distribution & Anchor Test Sizing) thesis fidelity.
7. Zero occurrences of stale 1.200h placeholders and strict "đề án" terminology compliance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.data.cleaner import PHYSICAL_BOUNDS
from src.data.loader import EXPECTED_COLUMNS, FEATURE_COLS, TARGET_COL
from src.data.segmenter import identify_contiguous_segments, validate_segment_boundaries
from src.features.temporal import create_rolling_features


class TestPhase1DataParameters:
    """Test raw data parameters, schemas, and physical bounds."""

    def test_expected_columns(self) -> None:
        """Expected columns must contain the 5 sensor variables plus datetime."""
        assert "pm25" in EXPECTED_COLUMNS
        assert "nhiet_do" in EXPECTED_COLUMNS
        assert "do_am" in EXPECTED_COLUMNS
        assert "diem_suong" in EXPECTED_COLUMNS
        assert "co2" in EXPECTED_COLUMNS
        assert "ngay_tao" in EXPECTED_COLUMNS
        assert TARGET_COL == "pm25"
        assert len(FEATURE_COLS) == 4

    def test_physical_bounds_values(self) -> None:
        """Physical bounds must reflect valid atmospheric sensor limits."""
        assert PHYSICAL_BOUNDS["pm25"] == (0.0, 500.0)
        assert PHYSICAL_BOUNDS["nhiet_do"] == (-10.0, 60.0)
        assert PHYSICAL_BOUNDS["do_am"] == (0.0, 100.0)
        assert PHYSICAL_BOUNDS["diem_suong"] == (-20.0, 40.0)
        assert PHYSICAL_BOUNDS["co2"] == (0.0, 5000.0)

    def test_outlier_strategy_preserves_pm25_fat_tail(self) -> None:
        """Cleaner code must enforce domain bounds for PM2.5, rejecting IQR clipping of real pollution."""
        cleaner_code = Path("src/data/cleaner.py").read_text(encoding="utf-8")
        assert (
            'DOMAIN_BOUNDS: dict[str, tuple[float, float]] = {\n        "pm25": (0.0, 500.0),' in cleaner_code
            or '"pm25": (0.0, 500.0)' in cleaner_code
        )
        assert "fat-tailed" in cleaner_code.lower()


class TestPhase1ImputerAndSegmenter:
    """Test imputation tracking, anti-leakage invariants, and segmentation."""

    def test_knn_imputation_past_only_donors(self) -> None:
        """_apply_knn_imputation must fit KNN on past donors only (:gap_start)."""
        imputer_code = Path("src/data/imputer.py").read_text(encoding="utf-8")
        assert "past_data = knn_data.iloc[:gap_start]" in imputer_code
        assert "past_complete_mask = past_data[TARGET_COL].notna()" in imputer_code
        assert "ANTI-LEAKAGE (v8 fix)" in imputer_code

    def test_segmenter_identifies_contiguous_blocks(self) -> None:
        """Segmenter must detect contiguous non-NaN blocks and assign segment IDs."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        df = pd.DataFrame({TARGET_COL: [10.0, 12.0, 15.0, None, None, 20.0, 22.0, 25.0, 28.0, 30.0]}, index=dates)
        segmented = identify_contiguous_segments(df, target_col=TARGET_COL, min_length=2)
        assert "segment_id" in segmented.columns
        assert segmented["segment_id"].nunique() == 2
        assert len(segmented) == 8  # 3 in segment 1, 5 in segment 2

    def test_validate_segment_boundaries_catches_gap(self) -> None:
        """validate_segment_boundaries must raise ValueError if internal gap > threshold."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1h")
        df = pd.DataFrame(
            {"segment_id": [1, 1, 1, 1, 1]},
            index=[dates[0], dates[1], dates[2], dates[2] + pd.Timedelta(hours=4), dates[4]],
        )
        with pytest.raises(ValueError, match="FALSE CONTINUITY|Found 1 segments"):
            validate_segment_boundaries(df, segment_col="segment_id", max_allowed_gap_hours=1.5)


class TestPhase1FeatureEngineeringAntiLeakage:
    """Test feature engineering causality and anti-leakage rules."""

    def test_rolling_features_strictly_shift_one(self) -> None:
        """Rolling features must shift(1) to avoid leaking target y[t]."""
        dates = pd.date_range("2024-01-01", periods=20, freq="1h")
        df = pd.DataFrame({TARGET_COL: range(20)}, index=dates)
        res = create_rolling_features(df, target_col=TARGET_COL, windows=[3], funcs=["mean"])
        # At t=3 (row 3, index 3), rolling mean of window 3 must use rows 0, 1, 2 (values 0, 1, 2 -> mean = 1.0)
        assert res.loc[dates[3], f"{TARGET_COL}_roll_3s_mean"] == 1.0

    def test_domain_features_use_lagged_target(self) -> None:
        """Domain features builder must use pm25_past_col (lagged) rather than pm25[t]."""
        builder_code = Path("src/features/builder.py").read_text(encoding="utf-8")
        assert 'df["co2"] / df[pm25_past_col]' in builder_code
        assert (
            'df["pm25_aqi_cat"] = pd.cut(\n            df[pm25_past_col]' in builder_code
            or "df[pm25_past_col]" in builder_code
        )


class TestPhase1ThesisFidelity:
    """Test numerical and structural alignment with Thesis Draft CTU 1799."""

    def test_table_4_1_stationarity_values_in_eda_page(self) -> None:
        """EDA page must display exact Table 4.1 ADF and KPSS statistics."""
        eda_code = Path("src/eda_page.py").read_text(encoding="utf-8")
        assert "-8,42" in eda_code  # Raw ADF
        assert "1,28" in eda_code  # Raw KPSS
        assert "-24,15" in eda_code  # 1st diff ADF
        assert "0,08" in eda_code  # 1st diff KPSS
        assert "-31,22" in eda_code  # Seasonal diff ADF
        assert "0,04" in eda_code  # Seasonal diff KPSS

    def test_table_3_4_sample_sizes_and_anchor_test_in_eda_and_overview(self) -> None:
        """EDA and Overview pages must display clean sample sizes and 10% Anchor Test sizing."""
        eda_code = Path("src/eda_page.py").read_text(encoding="utf-8")
        overview_code = Path("src/frontend/pages/overview.py").read_text(encoding="utf-8")

        assert "18.355" in eda_code  # 15m clean samples
        assert "8.625" in eda_code  # 30m clean samples
        assert "6.689" in eda_code  # 1h clean samples
        assert "669h ở 1h" in eda_code
        assert "863 mẫu ở 30m" in eda_code
        assert "1.836 mẫu ở 15m" in eda_code

        assert "18.355" in overview_code
        assert "8.625" in overview_code
        assert "6.689" in overview_code
        assert "669h ở 1h" in overview_code

    def test_no_stale_1200_hours_in_phase1_files(self) -> None:
        """Zero occurrences of stale '1.200 giờ' or '1.200h' in Phase 1 and Dashboard views."""
        files = [
            Path("src/eda_page.py"),
            Path("src/frontend/pages/overview.py"),
            Path("src/frontend/pages/actual_vs_predicted.py"),
            Path("src/frontend/pages/multi_horizon.py"),
        ]
        for f in files:
            content = f.read_text(encoding="utf-8")
            assert "1.200 giờ" not in content, f"Found stale '1.200 giờ' in {f}"
            assert "1.200h" not in content, f"Found stale '1.200h' in {f}"

    def test_no_luan_van_in_phase1_source_files(self) -> None:
        """Zero occurrences of 'luận văn' in Phase 1 data, feature, and frontend files."""
        files = [
            Path("src/data/loader.py"),
            Path("src/data/cleaner.py"),
            Path("src/data/imputer.py"),
            Path("src/data/segmenter.py"),
            Path("src/features/builder.py"),
            Path("src/features/temporal.py"),
            Path("src/eda_page.py"),
            Path("src/pipeline_walkthrough.py"),
            Path("src/frontend/pages/overview.py"),
        ]
        for f in files:
            content = f.read_text(encoding="utf-8")
            assert "luận văn" not in content.lower(), f"Found 'luận văn' in {f}"
