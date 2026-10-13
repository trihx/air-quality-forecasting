"""Unit test kiểm định tái hiện Diebold-Mariano (Bảng 4.7 & v9)."""

import json
from pathlib import Path

import numpy as np
from scripts.reproduce_table_4_7 import calculate_dm_hln

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_calculate_dm_hln_mathematical_properties():
    """Kiểm tra các thuộc tính toán học cơ bản của kiểm định DM-HLN."""
    # Khi model 1 tốt hơn model 2 (sai số e1 nhỏ hơn e2) -> DM phải mang dấu âm
    np.random.seed(42)
    e_good = np.random.normal(0, 1.0, 500)
    e_bad = np.random.normal(0, 3.0, 500)

    res_sq = calculate_dm_hln(e_good, e_bad, horizon=1, loss="squared")
    assert res_sq["HLN_statistic"] < 0, "DM phải mang dấu âm khi mô hình 1 tốt hơn mô hình 2"
    assert res_sq["HLN_p_value"] < 0.001, "Sự vượt trội rõ rệt phải có p < 0.001"

    # Khi model 1 tệ hơn model 2 (sai số e1 lớn hơn e2) -> DM phải mang dấu dương
    res_pos = calculate_dm_hln(e_bad, e_good, horizon=1, loss="squared")
    assert res_pos["HLN_statistic"] > 0, "DM phải mang dấu dương khi mô hình 1 kém hơn mô hình 2"

    # Kiểm tra loss=absolute
    res_abs = calculate_dm_hln(e_good, e_bad, horizon=4, loss="absolute")
    assert res_abs["HLN_statistic"] < 0
    assert 0.0 <= res_abs["HLN_p_value"] <= 1.0


def test_reproduced_table_4_7_artifacts_exist():
    """Kiểm tra các artifact tái hiện Bảng 4.7 và full matrix v9 đã được tạo đầy đủ."""
    reproduce_json = PROJECT_ROOT / "research" / "experiments" / "v9_final" / "table_4_7_reproduction.json"
    full_matrix_json = PROJECT_ROOT / "research" / "experiments" / "v9_final" / "dm_test_hln_all.json"

    assert reproduce_json.exists(), f"Thiếu file {reproduce_json}"
    assert full_matrix_json.exists(), f"Thiếu file {full_matrix_json}"

    with open(reproduce_json, encoding="utf-8") as f:
        rep_data = json.load(f)
    assert "table_4_7_reproduction" in rep_data
    assert len(rep_data["table_4_7_reproduction"]) >= 4

    with open(full_matrix_json, encoding="utf-8") as f:
        matrix_data = json.load(f)
    for h in ["1h", "6h", "24h"]:
        assert h in matrix_data
        assert len(matrix_data[h]) > 0


def test_table_4_7_statistical_consistency():
    """Kiểm tra tính nhất quán thống kê của các cặp so sánh trong Bảng 4.7 tái hiện."""
    reproduce_json = PROJECT_ROOT / "research" / "experiments" / "v9_final" / "table_4_7_reproduction.json"
    with open(reproduce_json, encoding="utf-8") as f:
        rep_data = json.load(f)

    records = {r["case_id"]: r for r in rep_data["table_4_7_reproduction"]}

    # Row 2 (6h Ensemble vs Persistence): phải có ý nghĩa thống kê cao (p < 0.05)
    r2 = records["Row 2"]["reproduced_on_codebase"]
    assert r2["squared_loss_h_steps"]["p_value"] < 0.05
    assert r2["squared_loss_h_steps"]["HLN_statistic"] < 0

    # Row 3 (24h Ensemble vs Persistence): phải có ý nghĩa thống kê (p < 0.05)
    r3 = records["Row 3"]["reproduced_on_codebase"]
    assert r3["squared_loss_h_steps"]["p_value"] < 0.05
    assert r3["squared_loss_h_steps"]["HLN_statistic"] < 0

    # Row 4 (6h Ensemble vs LSTM): p-value quanh ngưỡng 0.03-0.05
    r4 = records["Row 4"]["reproduced_on_codebase"]
    assert r4["squared_loss_h_steps"]["HLN_statistic"] < 0
    assert r4["squared_loss_h_steps"]["p_value"] < 0.06
