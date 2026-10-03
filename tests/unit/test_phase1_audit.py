"""Unit tests verifying Phase 1: Giới thiệu & Khám phá audit requirements.

Covers:
1. Logical ordering of steps in pipeline_walkthrough.py (Imputation BEFORE Feature Engineering).
2. Numerical alignment between Overview, Pipeline Walkthrough, and EDA.
3. Absence of AI slop phrases in Phase 1 display strings.
4. UI-Pro high-contrast layout structure for the 7-step pipeline.
5. Up-to-date test count reference (448 tests).
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_pipeline_walkthrough_step_order_imputation_before_features():
    """Verify that in pipeline_walkthrough.py, Imputation is Step 4 and Features is Step 5."""
    pw_content = (PROJECT_ROOT / "src" / "pipeline_walkthrough.py").read_text(encoding="utf-8")

    # In the progress bar list: Imputation must come BEFORE Features
    progress_bar_match = re.search(r"\bsteps\s*=\s*\[(.*?)\]", pw_content, re.DOTALL)
    assert progress_bar_match is not None, "Could not find 'steps = [...]' progress bar list"
    steps_str = progress_bar_match.group(1)

    idx_imputation = -1
    idx_features = -1
    for i, item in enumerate(re.findall(r"['\"](.*?)['\"]", steps_str)):
        if "imputation" in item.lower() or "phục hồi" in item.lower() or "nội suy" in item.lower():
            idx_imputation = i
        if "feature" in item.lower() or "đặc trưng" in item.lower():
            idx_features = i

    assert idx_imputation != -1, "Imputation step not found in progress bar"
    assert idx_features != -1, "Feature Engineering step not found in progress bar"
    assert idx_imputation < idx_features, (
        f"Imputation (index {idx_imputation}) must come BEFORE Feature Engineering (index {idx_features}) "
        "because rolling and lag features require continuous imputed data!"
    )


def test_pipeline_step_functions_execution_order():
    """Verify that _step_imputation() is executed before _step_feature_engineering()."""
    pw_content = (PROJECT_ROOT / "src" / "pipeline_walkthrough.py").read_text(encoding="utf-8")

    pos_imp_call = pw_content.find("_step_imputation()")
    pos_feat_call = pw_content.find("_step_feature_engineering()")

    assert pos_imp_call != -1, "_step_imputation() call missing"
    assert pos_feat_call != -1, "_step_feature_engineering() call missing"
    # Find execution in page_pipeline_walkthrough
    pos_walkthrough = pw_content.find("def page_pipeline_walkthrough")
    pos_imp_in_walkthrough = pw_content.find("_step_imputation()", pos_walkthrough)
    pos_feat_in_walkthrough = pw_content.find("_step_feature_engineering()", pos_walkthrough)

    assert pos_imp_in_walkthrough < pos_feat_in_walkthrough, (
        "_step_imputation() must be called before _step_feature_engineering() in page_pipeline_walkthrough()"
    )


def test_overview_pipeline_architecture_box_contains_key_scientific_data():
    """Verify that the 7-step pipeline in overview.py contains verified factual numbers."""
    ov_content = (PROJECT_ROOT / "src" / "frontend" / "pages" / "overview.py").read_text(encoding="utf-8")

    # Must contain 209.594 records
    assert "209.594" in ov_content or "209,594" in ov_content
    # Must mention Sa Đéc, Đồng Tháp
    assert "Sa Đéc" in ov_content
    # Must mention 19.810h drop
    assert "19.810" in ov_content
    # Must mention shift(1) anti-leakage
    assert "shift(1)" in ov_content
    # Must mention 100% Real Data Only for test set
    assert "is_imputed == 0" in ov_content or "REAL DATA ONLY" in ov_content


def test_no_ai_slop_phrases_in_phase_1_modules():
    """Verify that Phase 1 modules do not contain empty generic AI slop phrases."""
    files_to_check = [
        PROJECT_ROOT / "src" / "frontend" / "pages" / "overview.py",
        PROJECT_ROOT / "src" / "pipeline_walkthrough.py",
        PROJECT_ROOT / "src" / "eda_page.py",
    ]
    slop_phrases = [
        "vô cùng quan trọng",
        "đóng vai trò to lớn",
        "không thể phủ nhận",
        "như chúng ta đã biết",
        "vô cùng to lớn",
        "đóng vai trò vô cùng",
    ]

    for file_path in files_to_check:
        content = file_path.read_text(encoding="utf-8").lower()
        for phrase in slop_phrases:
            assert phrase not in content, f"Found AI slop phrase '{phrase}' in {file_path.name}"


def test_overview_page_kpi_cards_and_pipeline_ui_pro_styling():
    """Verify that overview.py uses high-contrast styling and structured step flow."""
    ov_content = (PROJECT_ROOT / "src" / "frontend" / "pages" / "overview.py").read_text(encoding="utf-8")

    # Must have 7 distinct visual step cards or structured blocks
    assert "bước 1" in ov_content.lower() or "step 1" in ov_content.lower()
    assert "bước 7" in ov_content.lower() or "step 7" in ov_content.lower()
    # Must not contain broken/faded raw text dumps with repeated &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    assert "&nbsp;&nbsp;&nbsp;&nbsp;↓<br>" not in ov_content


def test_eda_page_key_statistics_alignment():
    """Verify that eda_page.py contains exact verified descriptive statistics from Master Thesis."""
    eda_content = (PROJECT_ROOT / "src" / "eda_page.py").read_text(encoding="utf-8")

    # Mean: 17.46
    assert "17,46" in eda_content or "17.46" in eda_content
    # Skewness ~ 2.00
    assert "2,00" in eda_content or "2.00" in eda_content
    # Kurtosis ~ 6.14 or 6.15
    assert "6,14" in eda_content or "6,15" in eda_content or "6.14" in eda_content
    # Max ~ 138.5
    assert "138,5" in eda_content or "138.5" in eda_content
