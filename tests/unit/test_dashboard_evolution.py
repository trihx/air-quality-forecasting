"""
Unit tests for Dashboard v1 to v10 Evolution Matrix & Council Audit.
"""

from __future__ import annotations

from src.reporting.content import ContentManager


def test_evolution_matrix_contains_ten_versions():
    """Verify that ContentManager loads all 10 versions in the evolution matrix."""
    mgr = ContentManager()
    matrix = mgr.get_evolution_matrix()

    expected_versions = [
        "v1_baseline",
        "v2_enhanced",
        "v3_sklearn_ensemble",
        "v4_roc_auc",
        "v5_dl_retrain",
        "v6_pca_tft",
        "v7_cqr",
        "v8_scientific_audit",
        "v9_multi_resolution",
        "v10_ablation",
    ]

    for v in expected_versions:
        assert v in matrix, f"Missing version {v} in evolution matrix"


def test_each_version_has_complete_five_pillars():
    """Verify each version has all 5 academic pillars and council verdict."""
    mgr = ContentManager()
    matrix = mgr.get_evolution_matrix()

    required_keys = [
        "title",
        "badge",
        "date",
        "category",
        "hypothesis",
        "technical_upgrades",
        "results_summary",
        "breakthrough_traps",
        "takeaway_next",
        "council_verdict",
        "metrics_summary",
    ]

    for v_key, info in matrix.items():
        for k in required_keys:
            assert k in info, f"Version {v_key} missing required key: {k}"
            assert len(str(info[k])) > 0, f"Version {v_key} key {k} is empty"

        metrics = info["metrics_summary"]
        for h in ("1h", "6h", "24h"):
            assert h in metrics, f"Version {v_key} missing metrics for {h}"
            assert "best_model" in metrics[h], f"Version {v_key} missing best_model in {h}"
            assert "mae" in metrics[h], f"Version {v_key} missing mae in {h}"
            assert "mase" in metrics[h], f"Version {v_key} missing mase in {h}"


def test_production_standard_v9_highlights():
    """Verify v9_multi_resolution has expected production standard highlights."""
    mgr = ContentManager()
    v9 = mgr.get_version_evolution("v9_multi_resolution")

    assert "Standard" in v9["badge"]
    assert "0.382" in v9["results_summary"] or "0,382" in v9["results_summary"]
    assert "0.667" in v9["results_summary"] or "0,667" in v9["results_summary"]
    assert v9["metrics_summary"]["6h"]["mase"] == 0.382
    assert v9["metrics_summary"]["24h"]["mase"] == 0.469


def test_counterfactual_ablation_v10_highlights():
    """Verify v10_ablation is documented as a counterfactual study."""
    mgr = ContentManager()
    v10 = mgr.get_version_evolution("v10_ablation")

    assert "Ablation" in v10["badge"]
    assert "Counterfactual Ablation Study" in v10["title"]
    assert "1.908" in v10["results_summary"] or "1908" in v10["results_summary"]


def test_no_defense_or_council_keywords_in_evolution_matrix():
    """Verify that the evolution matrix contains no academic defense/council exam jargon or AI slop."""
    import re

    mgr = ContentManager()
    matrix = mgr.get_evolution_matrix()

    forbidden_patterns = [
        r"hội đồng",
        r"phản biện",
        r"chất vấn",
        r"bảo vệ đề án",
        r"bảo vệ luận văn",
        r"luận văn",
        r"đỉnh cao",
        r"production standard",
        r"vô địch",
        r"viên đạn bạc",
    ]

    for v_key, info in matrix.items():
        text_content = " ".join(str(v) for k, v in info.items() if isinstance(v, str))
        for pattern in forbidden_patterns:
            matches = re.findall(pattern, text_content, flags=re.IGNORECASE)
            assert not matches, f"Version {v_key} contains forbidden term '{pattern}': {matches}"


def test_sticky_version_cell_rendering_and_links():
    """Verify sticky cell renders clean magnifying glass without 'soi' and has correct deep link."""
    from src.frontend.pages.overview import _render_sticky_version_cell

    mgr = ContentManager()
    matrix = mgr.get_evolution_matrix()

    for v_key, info in matrix.items():
        cell_html = _render_sticky_version_cell(v_key, info)
        assert f"?selected_ver={v_key}#five-pillar-inspector" in cell_html
        assert "🔍" in cell_html
        assert "🔍 soi" not in cell_html
        assert "soi</span>" not in cell_html
        assert "\n" not in cell_html, f"Cell HTML for {v_key} must be compact single-line"


def test_clean_text_removes_raw_latex_and_markdown():
    """Verify _clean_text converts LaTeX math syntax and markdown to clean Unicode & safe HTML."""
    from src.frontend.pages.overview import _clean_text

    raw_sample = "Bẫy tự tương quan ($r \\approx 0.86$ ở chuỗi giờ, $r \\approx 0.97$ ở 15m), $y_{t+h} = y_t$, $z = \\log(1 + y)$, $p \\le 0.05$ và **in đậm** cùng `code`."
    cleaned = _clean_text(raw_sample)

    assert "\\approx" not in cleaned
    assert "\\log" not in cleaned
    assert "\\le" not in cleaned
    assert "$" not in cleaned
    assert "r ≈ 0.86" in cleaned
    assert "r ≈ 0.97" in cleaned
    assert "y(t+h) = y_t" in cleaned
    assert "z = log(1 + y)" in cleaned
    assert "p ≤ 0.05" in cleaned
    assert "<b>in đậm</b>" in cleaned
    assert "<code>code</code>" in cleaned


if __name__ == "__main__":
    test_evolution_matrix_contains_ten_versions()
    test_each_version_has_complete_five_pillars()
    test_production_standard_v9_highlights()
    test_counterfactual_ablation_v10_highlights()
    test_no_defense_or_council_keywords_in_evolution_matrix()
    test_sticky_version_cell_rendering_and_links()
    test_clean_text_removes_raw_latex_and_markdown()
    print("All 7 evolution matrix tests passed successfully!")


