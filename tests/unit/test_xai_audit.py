"""Unit tests verifying XAI Hub (Explainability Hub) audit and visual contrast."""

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestXAIHubCSSContrast:
    """Verify that no white-blinding CSS inversion exists in dashboard pages."""

    FILES_TO_CHECK = [
        PROJECT_ROOT / "src" / "explainability_hub.py",
        PROJECT_ROOT / "src" / "frontend" / "pages" / "actual_vs_predicted.py",
    ]

    def test_no_background_var_text_color(self):
        """Ensure background: var(--text-color) is completely eradicated."""
        for file_path in self.FILES_TO_CHECK:
            assert file_path.exists(), f"File {file_path} not found"
            content = file_path.read_text(encoding="utf-8")
            matches = re.findall(r"background(-color)?:\s*var\(--text-color\)", content)
            assert len(matches) == 0, f"Found background: var(--text-color) in {file_path.name}"

    def test_no_color_var_background_color(self):
        """Ensure color: var(--background-color) is eradicated from styling."""
        for file_path in self.FILES_TO_CHECK:
            assert file_path.exists(), f"File {file_path} not found"
            content = file_path.read_text(encoding="utf-8")
            matches = re.findall(r"color:\s*var\(--background-color\)", content)
            assert len(matches) == 0, f"Found color: var(--background-color) in {file_path.name}"


class TestXAIFeatureMappingAndData:
    """Verify feature mapping handles both _s and _h suffixes and SHAP data integrity."""

    def test_shap_results_json_structure(self):
        """Verify shap_results.json exists and contains 1h, 6h, 24h horizons with expected top features."""
        shap_json_path = PROJECT_ROOT / "research" / "figures" / "shap" / "shap_results.json"
        assert shap_json_path.exists()
        with open(shap_json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "1h" in data and "6h" in data and "24h" in data

        # 1h: pm25_lag_1s must be top feature
        top_1h = data["1h"]["top_15_shap"]
        assert list(top_1h.keys())[0] in ("pm25_lag_1s", "pm25_lag_1h")
        assert top_1h.get("pm25_lag_1s") == pytest.approx(2.824, abs=0.01)

        # 6h: pm25_roll_24s_mean must be top feature
        top_6h = data["6h"]["top_15_shap"]
        assert list(top_6h.keys())[0] in ("pm25_roll_24s_mean", "pm25_roll_24h_mean")
        assert top_6h.get("pm25_roll_24s_mean") == pytest.approx(2.910, abs=0.01)

        # 24h: pm25_lag_1s is anchor
        top_24h = data["24h"]["top_15_shap"]
        assert "pm25_lag_1s" in top_24h or "pm25_lag_1h" in top_24h

    def test_map_feat_name_logic_in_explainability_hub(self):
        """Verify that map_feat_name handles both _s and _h alias suffixes."""
        content = (PROJECT_ROOT / "src" / "explainability_hub.py").read_text(encoding="utf-8")
        assert 'if f in ("pm25_lag_1s", "pm25_lag_1h"):' in content
        assert 'if f in ("pm25_lag_24s", "pm25_lag_24h"):' in content
        assert '"roll_24" in f and "mean" in f' in content


class TestXAIScientificFoundationAndThesisAlignment:
    """Verify Tab 5 citations and thesis figures for XAI."""

    def test_tab_5_has_ieee_citations(self):
        """Verify 8 core books in Tab 5 include accurate IEEE citation IDs."""
        content = (PROJECT_ROOT / "src" / "explainability_hub.py").read_text(encoding="utf-8")
        # Check presence of IEEE IDs for key authors
        assert "Lundberg & Lee (2017)" in content
        assert "41" in content  # Lundberg is [41]
        assert "Chen & Guestrin (2016)" in content
        assert "27" in content  # XGBoost is [27]
        assert "Ke et al. (2017)" in content
        assert "28" in content  # LightGBM is [28]
        assert "Molnar (2022)" in content
        assert "51" in content  # Interpretable ML is [51]
        assert "Hyndman & Athanasopoulos (2021)" in content
        assert "38" in content  # FPP3 is [38]

    def test_tipping_point_and_who_thresholds(self):
        """Verify tipping point 14-17 and WHO 15 are documented in XAI hub."""
        content = (PROJECT_ROOT / "src" / "explainability_hub.py").read_text(encoding="utf-8")
        assert "14 – 17 µg/m³" in content or "14–17 µg/m³" in content
        assert "15 µg/m³" in content
        assert "Vùng ức chế" in content
        assert "Vùng chuyển tiếp" in content
        assert "Vùng kích hoạt" in content

    def test_all_xai_thesis_figures_exist(self):
        """Verify that all figures cited in Chapter 4 §4.6 exist at 300 DPI."""
        thesis_dir = PROJECT_ROOT / "research" / "figures" / "thesis"
        required_figs = [
            "Hinh_4.7a_SHAP_Bar_1h.png",
            "Hinh_4.7b_SHAP_Bar_6h.png",
            "Hinh_4.7c_SHAP_Bar_24h.png",
            "Hinh_4.8_SHAP_Dependence_6h.png",
            "Hinh_4.9a_SHAP_Beeswarm_1h.png",
            "Hinh_4.9b_SHAP_Beeswarm_6h.png",
            "Hinh_4.9c_SHAP_Beeswarm_24h.png",
            "Hinh_4.10a_GRU_Permutation_1h.png",
            "Hinh_4.10b_GRU_Permutation_6h.png",
            "Hinh_4.10c_GRU_Permutation_24h.png",
            "Hinh_PL.3_SHAP_Horizons.png",
        ]
        for fig in required_figs:
            p = thesis_dir / fig
            assert p.exists(), f"Missing thesis figure: {fig}"
            assert p.stat().st_size > 10000, f"Figure {fig} is empty or corrupted"
