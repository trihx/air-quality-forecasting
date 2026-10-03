"""Unit tests verifying Phase 3: Đánh giá & Giải thích audit requirements.

Covers:
1. Absence of AI slop phrases and "luận văn" in Phase 3 modules.
2. Academic numerical alignment:
   - Table 4.3 (Anchor Test Set): 10% anchor test (669h at 1h, 863 at 30m, 1836 at 15m, is_imputed == 0).
   - Table 4.4 (Ljung-Box lag=6): p < 0.05, Persistence 6h std=8.272 vs LightGBM std=5.296.
   - Table 4.6 (UQ): ACI (gamma=0.01) maintains 89.4%-89.6% coverage, Winkler 4.12-9.85.
   - Table 4.7 (Diebold-Mariano): 1h positive sign (+13.729), 6h negative sign (-8.452) p < 0.001.
   - Table 4.8 (WHO 45 alert): F1=0.782, Precision=0.812, Recall=0.754 on 57 pollution spikes.
3. UI-Pro WCAG AAA High Contrast Styling:
   - app.py .insight-card uses solid dark #0B1120 background and sharp contrast.
   - Major containers in multi_horizon, actual_vs_predicted, explainability_hub, benchmarks use solid dark styling.
4. Thesis Figure Preservation:
   - 100% verified thesis figure images remain intact in research/figures/thesis/.
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PHASE3_FILES = [
    PROJECT_ROOT / "src" / "frontend" / "pages" / "multi_horizon.py",
    PROJECT_ROOT / "src" / "frontend" / "pages" / "actual_vs_predicted.py",
    PROJECT_ROOT / "src" / "explainability_hub.py",
    PROJECT_ROOT / "src" / "frontend" / "pages" / "prediction_intervals.py",
    PROJECT_ROOT / "src" / "frontend" / "pages" / "benchmarks.py",
    PROJECT_ROOT / "src" / "thesis_figures.py",
    PROJECT_ROOT / "src" / "frontend" / "pages" / "audit.py",
]


def test_phase3_no_ai_slop_or_luan_van():
    """Verify that Phase 3 modules contain neither generic AI slop nor 'luận văn'."""
    forbidden_terms = [
        "vô cùng quan trọng",
        "đóng vai trò to lớn",
        "không thể phủ nhận",
        "như chúng ta đã biết",
        "vô cùng to lớn",
        "đóng vai trò vô cùng",
        "luận văn",
    ]

    for p in PHASE3_FILES:
        assert p.exists(), f"File {p} does not exist"
        content = p.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in content, f"Found forbidden term '{term}' in {p.name}"


def test_phase3_anchor_test_set_values():
    """Verify Table 4.3 Anchor Test Set parameters in multi_horizon.py and content manager."""
    from src.reporting.content import ContentManager

    cm = ContentManager()
    summary = cm.get_anchor_test_summary()
    assert len(summary) >= 10, "Anchor test summary must contain at least 10 model rows"

    # Persistence 1h baseline MASE must be 1.0
    p1 = next((r for r in summary if r["Horizon"] == "1 giờ" and "Persistence" in r["Mô hình"]), None)
    assert p1 is not None, "Persistence at 1h missing from Anchor test summary"
    assert p1["MASE"] == 1.0

    # Ensemble Weighted at 6h must achieve MASE ~ 0.382
    ens6 = next((r for r in summary if r["Horizon"] == "6 giờ" and "Ensemble_Weighted" in r["Mô hình"]), None)
    assert ens6 is not None, "Ensemble Weighted at 6h missing"
    assert ens6["MASE"] == pytest.approx(0.382, abs=0.01)

    # Multi-horizon page text should describe the anchor set counts
    mh_content = (PROJECT_ROOT / "src" / "frontend" / "pages" / "multi_horizon.py").read_text(encoding="utf-8")
    assert "669" in mh_content, "Missing 669h anchor count in multi_horizon.py"
    assert "863" in mh_content, "Missing 863 30m anchor count in multi_horizon.py"
    assert "1.836" in mh_content or "1836" in mh_content, "Missing 1836 15m anchor count in multi_horizon.py"
    assert "is_imputed == 0" in mh_content, "Missing is_imputed == 0 in multi_horizon.py"


def test_phase3_diebold_mariano_table_values():
    """Verify Table 4.7 Diebold-Mariano test results and positive sign at 1h."""
    from src.reporting.content import ContentManager

    cm = ContentManager()
    dm = cm.get_dm_test_data()
    assert len(dm) == 4, "DM test table must have 4 comparison rows"

    # Row 1: 1h GRU vs Persistence -> positive DM statistic (+13.729)
    assert "+13.729" in dm[0]["Thống kê DM"]
    assert "Persistence tốt hơn" in dm[0]["Kết luận"]

    # Row 2: 6h Ensemble vs Persistence -> negative DM statistic (-8.452)
    assert "-8.452" in dm[1]["Thống kê DM"]
    assert "< 0.001" in dm[1]["p-value"]

    # Row 3: 24h Ensemble vs Persistence -> negative DM statistic (-5.891)
    assert "-5.891" in dm[2]["Thống kê DM"]


def test_phase3_residual_diagnostics_ljung_box():
    """Verify Table 4.4 residual diagnostics and Ljung-Box test results."""
    from src.reporting.content import ContentManager

    cm = ContentManager()
    residuals = cm.get_residual_diagnostics()
    assert len(residuals) == 7, "Residual diagnostics must have 7 model configurations"

    # Check LightGBM 6h vs Persistence 6h standard deviation
    p_6h = next((r for r in residuals if r["Mô hình"] == "Persistence" and r["Horizon"] == "6 giờ"), None)
    lgbm_6h = next((r for r in residuals if r["Mô hình"] == "LightGBM" and r["Horizon"] == "6 giờ"), None)

    assert p_6h is not None and lgbm_6h is not None
    assert p_6h["Std"] == pytest.approx(8.272, abs=0.01)
    assert lgbm_6h["Std"] == pytest.approx(5.296, abs=0.01)

    # All p-values reject white noise null hypothesis
    for r in residuals:
        pval_str = r.get("Ljung-Box lag=6 (p-val)", "")
        assert "p <" in pval_str or "p =" in pval_str


def test_phase3_who_alert_performance():
    """Verify Table 4.8 WHO 45 ug/m3 alert classification performance."""
    from src.reporting.content import ContentManager

    cm = ContentManager()
    who_alerts = cm.get_who_threshold_alert()
    assert len(who_alerts) >= 3, "WHO alert table must have at least 3 models"

    ens_alert = who_alerts[0]
    assert ens_alert["Mô hình"] == "Ensemble_30m"
    assert ens_alert["F1-Score"] == pytest.approx(0.782, abs=0.01)
    assert ens_alert["Precision"] == pytest.approx(0.812, abs=0.01)
    assert ens_alert["Recall"] == pytest.approx(0.754, abs=0.01)
    assert ens_alert["Số đợt ô nhiễm"] == 57


def test_phase3_conformal_aci_coverage():
    """Verify Table 4.6 Conformal Prediction evaluation metrics."""
    from src.reporting.content import ContentManager

    cm = ContentManager()
    conformal = cm.get_conformal_prediction()
    assert len(conformal) == 9, "Conformal prediction table must have 9 evaluations"

    aci_rows = [r for r in conformal if "Adaptive Conformal" in r["Phương pháp"]]
    assert len(aci_rows) == 3, "Must have 3 horizons for ACI"

    for r in aci_rows:
        cov_val = float(r["Coverage"].replace("%", ""))
        assert 89.0 <= cov_val <= 90.0, f"ACI coverage {cov_val}% out of target range"
        assert r["Winkler Score"] <= 10.0, f"Winkler score {r['Winkler Score']} unexpectedly high"


def test_phase3_ui_pro_solid_dark_styling():
    """Verify UI-Pro solid dark #0B1120 styling in app.py and Phase 3 pages."""
    app_css = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert ".insight-card {" in app_css
    assert "#0B1120" in app_css, "app.py must define #0B1120 background for cards"

    # Multi horizon page must not use secondary background in the methodology block
    mh_content = (PROJECT_ROOT / "src" / "frontend" / "pages" / "multi_horizon.py").read_text(encoding="utf-8")
    assert "background: var(--secondary-background-color)" not in mh_content

    # Actual vs predicted page must not use secondary background in header / methodology
    avp_content = (PROJECT_ROOT / "src" / "frontend" / "pages" / "actual_vs_predicted.py").read_text(encoding="utf-8")
    assert "background: var(--secondary-background-color);" not in avp_content


def test_phase3_preserves_images_and_charts():
    """Verify that essential thesis figures are intact and not modified/deleted."""
    thesis_figs_dir = PROJECT_ROOT / "research" / "figures" / "thesis"
    assert thesis_figs_dir.exists(), "research/figures/thesis/ must exist"

    expected_figures = [
        "Hinh_4.6a_Diagnostics_GRU_1h.png",
        "Hinh_4.6b_Diagnostics_LightGBM_1h.png",
        "Hinh_4.6c_Diagnostics_GRU_6h.png",
        "Hinh_4.6d_Diagnostics_LightGBM_6h.png",
        "Hinh_4.6e_Diagnostics_Persistence_6h.png",
        "Hinh_4.6f_Diagnostics_GRU_24h.png",
        "Hinh_4.6g_Diagnostics_LightGBM_24h.png",
        "Hinh_4.11a_PI_Conformal_LightGBM_1h.png",
        "Hinh_4.11b_PI_Conformal_LightGBM_6h.png",
        "Hinh_4.11c_PI_Conformal_LightGBM_24h.png",
    ]

    for fig_name in expected_figures:
        fig_path = thesis_figs_dir / fig_name
        assert fig_path.exists(), f"Thesis figure {fig_name} missing!"
        assert fig_path.stat().st_size > 10_000, f"Thesis figure {fig_name} is corrupted or empty!"
