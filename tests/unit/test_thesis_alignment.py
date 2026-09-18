"""Unit tests to verify alignment of dashboard data and pages with master's thesis report (04-M2522016_HOANG XUAN TRI.pdf)."""

from __future__ import annotations

import re
from pathlib import Path

from src.reporting.content import ContentManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestThesisDataAlignment:
    """Test that all thesis report tables (4.3, 4.4, 4.6, 4.7, 4.8) match official values."""

    def setup_method(self):
        self.content = ContentManager()

    def test_table_4_3_anchor_test_summary(self):
        """Verify Table 4.3: Anchor Test Set results across horizons and resolutions."""
        data = self.content.get_anchor_test_summary()
        assert len(data) >= 11, f"Expected at least 11 model configurations, got {len(data)}"

        # Find key models
        models = {f"{d['Horizon']}_{d['Mô hình']}": d for d in data}

        # 1h Persistence
        p1 = models.get("1 giờ_Persistence_1h (Baseline)")
        assert p1 is not None
        assert p1["MAE"] == 2.596
        assert p1["RMSE"] == 4.247
        assert p1["MASE"] == 1.000

        # 1h GRU 15m (breaks autocorrelation trap)
        gru15 = models.get("1 giờ_GRU_v9_15m")
        assert gru15 is not None
        assert gru15["MASE"] == 0.667

        # 6h Ensemble 30m (champion)
        ens6 = models.get("6 giờ_Ensemble_Weighted_v9_30m")
        assert ens6 is not None
        assert ens6["MAE"] == 3.493
        assert ens6["RMSE"] == 5.079
        assert ens6["MASE"] == 0.382

        # 24h Ensemble 30m (champion)
        ens24 = models.get("24 giờ_Ensemble_Weighted_v9_30m")
        assert ens24 is not None
        assert ens24["MAE"] == 3.417
        assert ens24["RMSE"] == 4.872
        assert ens24["MASE"] == 0.469

    def test_table_4_4_residual_diagnostics(self):
        """Verify Table 4.4: Ljung-Box residual diagnostics at lag=6."""
        data = self.content.get_residual_diagnostics()
        assert len(data) == 7, f"Expected 7 entries in Table 4.4, got {len(data)}"

        # Check Persistence 6h vs LightGBM 6h for variance reduction
        persist_6h = next(d for d in data if d["Mô hình"] == "Persistence" and d["Horizon"] == "6 giờ")
        lgbm_6h = next(d for d in data if d["Mô hình"] == "LightGBM" and d["Horizon"] == "6 giờ")

        assert persist_6h["Mean"] == 0.029
        assert persist_6h["Std"] == 8.272
        assert lgbm_6h["Std"] == 5.296
        # Variance reduction > 35%
        assert (persist_6h["Std"] - lgbm_6h["Std"]) / persist_6h["Std"] > 0.35

        # All ML/DL models have positive skewness > 1.1
        ml_dl = [d for d in data if d["Mô hình"] in ("GRU", "LightGBM")]
        for m in ml_dl:
            assert m["Skewness"] > 1.1

    def test_table_4_6_conformal_prediction(self):
        """Verify Table 4.6: Conformal Prediction, CQR, and ACI metrics."""
        data = self.content.get_conformal_prediction()
        assert len(data) == 9, f"Expected 9 entries (3 methods x 3 horizons), got {len(data)}"

        aci_entries = [d for d in data if "Adaptive" in d["Phương pháp"]]
        assert len(aci_entries) == 3
        for entry in aci_entries:
            # Empirical coverage should be ~89.5%
            cov = float(entry["Coverage"].replace("%", ""))
            assert 89.0 <= cov <= 90.0, f"Unexpected ACI coverage: {cov}"
            # NMPIW should be <= 0.35
            assert entry["NMPIW"] <= 0.35
            # Winkler score should be < 10.0
            assert entry["Winkler Score"] < 10.0

    def test_table_4_7_diebold_mariano(self):
        """Verify Table 4.7: Diebold-Mariano test results."""
        data = self.content.get_dm_test_data()
        assert len(data) == 4, f"Expected 4 comparison pairs in Table 4.7, got {len(data)}"

        ens_6h = next(
            d for d in data if "Ensemble_30m so với Persistence" in d["Cặp so sánh"] and d["Horizon"] == "6 giờ"
        )
        assert ens_6h["Thống kê DM"] == "-8.452"
        assert ens_6h["p-value"] == "< 0.001"

        ens_24h = next(
            d for d in data if "Ensemble_30m so với Persistence" in d["Cặp so sánh"] and d["Horizon"] == "24 giờ"
        )
        assert ens_24h["Thống kê DM"] == "-5.891"
        assert ens_24h["p-value"] == "< 0.001"

    def test_table_4_8_who_threshold_alert(self):
        """Verify Table 4.8: WHO 45 µg/m³ early warning evaluation."""
        data = self.content.get_who_threshold_alert()
        assert len(data) == 4, f"Expected 4 entries in Table 4.8, got {len(data)}"

        ens_6h = next(d for d in data if d["Mô hình"] == "Ensemble_30m" and d["Horizon"] == "6 giờ")
        assert ens_6h["Precision"] == 0.812
        assert ens_6h["Recall"] == 0.754
        assert ens_6h["F1-Score"] == 0.782
        assert ens_6h["Số đợt ô nhiễm"] == 57


class TestTerminologyDeAn:
    """Verify that all user-facing text uses 'đề án' instead of 'luận văn'."""

    PAGES_TO_CHECK = [
        PROJECT_ROOT / "src" / "conclusion_page.py",
        PROJECT_ROOT / "src" / "frontend" / "pages" / "multi_horizon.py",
        PROJECT_ROOT / "src" / "frontend" / "pages" / "actual_vs_predicted.py",
        PROJECT_ROOT / "src" / "frontend" / "pages" / "prediction_intervals.py",
        PROJECT_ROOT / "src" / "frontend" / "pages" / "benchmarks.py",
        PROJECT_ROOT / "src" / "thesis_figures.py",
        PROJECT_ROOT / "src" / "explainability_hub.py",
        PROJECT_ROOT / "app.py",
    ]

    def test_no_luan_van_in_dashboard_display_strings(self):
        """Check that no user-facing string literals in key pages contain 'luận văn'."""
        pattern = re.compile(r"luận\s*văn", re.IGNORECASE)

        for page_file in self.PAGES_TO_CHECK:
            assert page_file.exists(), f"File {page_file} does not exist"
            content = page_file.read_text(encoding="utf-8")

            # Filter out comments or citations references
            lines = content.splitlines()
            for line_no, line in enumerate(lines, 1):
                # Ignore pure comment lines or bibliography citation titles
                stripped = line.strip()
                if stripped.startswith("#") or "doi.org" in stripped or "IEEE_REFS" in stripped:
                    continue
                match = pattern.search(line)
                assert match is None, f"Found 'luận văn' at {page_file.name}:{line_no}: {line.strip()}"


class TestConclusionPageStructure:
    """Verify that conclusion_page.py implements all 4 sections of Chapter 5."""

    def test_has_all_chapter_5_sections(self):
        from src.conclusion_page import (
            _render_future_work,
            _render_limitations,
            _render_practical_applications,
            _render_summary,
            page_conclusion,
        )

        assert callable(page_conclusion)
        assert callable(_render_summary)
        assert callable(_render_practical_applications)
        assert callable(_render_limitations)
        assert callable(_render_future_work)


class TestFigure11Alignment:
    """Verify Figure 1.1 caption and structure fidelity with official thesis report."""

    def test_hinh_1_1_exists_and_matches_master_hash(self):
        import hashlib

        p1 = PROJECT_ROOT / "research" / "figures" / "thesis" / "Hinh_1.1_Overview_Pipeline.png"
        assert p1.exists(), f"Missing master figure: {p1}"

        # Master SHA256 for cleanly rendered Figure 1.1 (zero text collision, 9-column invariant)
        master_sha256 = "2ef841234340227aa2137cbac9fe4a8d410c6d751cd2240f13881d93c3a046a8"
        current_sha256 = hashlib.sha256(p1.read_bytes()).hexdigest()
        assert current_sha256 == master_sha256, (
            f"Hinh_1.1 SHA256 mismatch! Expected {master_sha256}, got {current_sha256}. "
            "Do not alter Figure 1.1 structure away from official thesis report."
        )

    def test_hinh_1_1_official_caption_in_walkthrough_and_figures(self):
        expected_caption = "Hình 1.1: Sơ đồ dòng chảy dữ liệu quy trình nghiên cứu 7 bước (7-Step Workflow)"

        # Check pipeline_walkthrough.py
        walkthrough_py = PROJECT_ROOT / "src" / "pipeline_walkthrough.py"
        content_wt = walkthrough_py.read_text(encoding="utf-8")
        assert expected_caption in content_wt, f"Caption mismatch in {walkthrough_py}"

        # Check thesis_figures.py
        thesis_figures_py = PROJECT_ROOT / "src" / "thesis_figures.py"
        content_tf = thesis_figures_py.read_text(encoding="utf-8")
        assert expected_caption in content_tf, f"Caption mismatch in {thesis_figures_py}"
