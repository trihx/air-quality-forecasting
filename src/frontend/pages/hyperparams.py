"""Hyperparameters and model architecture configurations page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.frontend.citations import cite, render_references_section
from src.frontend.components import (
    RESEARCH_DIR,
    insight_card,
    kpi_card,
    load_json,
    section_header,
)
from src.info_cards import (
    cards_hyperparams,
    get_current_version,
    render_version_badge,
)


def page_hyperparams(results: dict[str, Any]) -> None:
    """Render hyperparameter configurations and multi-resolution matrix page."""
    st.markdown(
        """
    <h1 style="font-size: 2rem;">⚙️ Cấu Hình Siêu Tham Số (Hyperparameters)</h1>
    <p style="opacity: 0.7;">Chi tiết cấu hình tối ưu siêu tham số cho từng mô hình và khung thời gian dự báo (horizon)</p>
    """,
        unsafe_allow_html=True,
    )

    # ── Version-aware info cards ──
    ver = get_current_version()
    render_version_badge(ver)
    cards_hyperparams(ver)

    configs = results.get("configs")

    # ── Load hyperparameter reference file (single source of truth) ──
    hp_path = RESEARCH_DIR / "experiments" / "hyperparameter_configs.json"
    hp_configs = load_json(hp_path) or {}
    if not isinstance(hp_configs, dict):
        hp_configs = {}

    # ── KPI Header ──
    matrix_info = hp_configs.get("matrix_summary", {})
    st.markdown(
        f"""
<div class="kpi-row">
    {kpi_card("Tổng Số Cấu Hình", f"{matrix_info.get('total_configurations', 41)}", "15 ở 15m · 15 ở 30m · 11 ở 1h")}
    {kpi_card("Mô Hình Có Tham Số", f"{matrix_info.get('parametric_models', 36)}", "Phân tích Bias & Thặng dư chuyên sâu")}
    {kpi_card("Baseline Không Tham Số", "5", "Persistence & Đóng băng lịch sử")}
    {kpi_card("Thời Gian Huấn Luyện", "< 45 phút", "Tối ưu hóa toàn diện trên GPU/CPU")}
</div>
""",
        unsafe_allow_html=True,
    )

    tab_mat, tab_lgb, tab_tft, tab_dl, tab_skl, tab_ens, tab_ari, tab_raw = st.tabs(
        [
            "📊 Ma Trận 41 Cấu Hình",
            "🌲 LightGBM",
            "⚡ Simplified TFT (25K)",
            "🧠 GRU / LSTM",
            "🌳 RF & ElasticNet",
            "🔀 Stacking Ensemble",
            "📈 ARIMA / SARIMA",
            "📄 Raw JSON",
        ]
    )

    with tab_mat:
        section_header("📊", "Ma Trận Phân Bổ 41 Cấu Hình Thực Nghiệm (3 Độ Phân Giải)")
        mat_rows = [
            {
                "Độ phân giải": "15 phút (15m)",
                "Số cấu hình": 15,
                "Nhóm mô hình": "Persistence, ARIMA, SARIMA, LightGBM, RF, ElasticNet, GB, GRU, LSTM, TFT, GRU_Expert, LSTM_Expert, TFT_Expert, Stacking, Weighted Ensemble",
                "Ghi chú": "Khai thác tối đa biến động vi mô",
            },
            {
                "Độ phân giải": "30 phút (30m)",
                "Số cấu hình": 15,
                "Nhóm mô hình": "Persistence, ARIMA, SARIMA, LightGBM, RF, ElasticNet, GB, GRU, LSTM, TFT, GRU_Expert, LSTM_Expert, TFT_Expert, Stacking, Weighted Ensemble",
                "Ghi chú": "Cân bằng vàng: tiết kiệm 50% tính toán, MASE=0,382",
            },
            {
                "Độ phân giải": "1 giờ (1h)",
                "Số cấu hình": 11,
                "Nhóm mô hình": "Persistence, ARIMA, SARIMA, LightGBM, RF, ElasticNet, GRU, LSTM, TFT, Stacking, Weighted Ensemble",
                "Ghi chú": "Chu kỳ vĩ mô, bẫy tự tương quan r ≈ 0,86",
            },
        ]
        st.dataframe(pd.DataFrame(mat_rows), use_container_width=True, hide_index=True)

        section_header("📋", "Bảng 3.4: Quy Mô Phân Bổ Dữ Liệu Thực Tế Theo Cấu Trúc 80:10:10")
        table_3_4_rows = [
            {
                "Độ phân giải": "15 phút (15m)",
                "Lưới lý thuyết (38 tháng)": "110.592",
                "Tổng mẫu sạch thực tế": "18.355",
                "Tập Train (80%)": "14.684",
                "Tập Val (10%)": "1.835",
                "Tập Test mỏ neo (10%)": "1.836",
                "Cửa sổ trượt (DL)": "288 bước (72h)",
                "Đặc điểm tín hiệu & Đánh giá": "Tín hiệu vi mô, GRU bắt nhịp xuất sắc (MASE = 0,667)",
            },
            {
                "Độ phân giải": "30 phút (30m)",
                "Lưới lý thuyết (38 tháng)": "55.296",
                "Tổng mẫu sạch thực tế": "8.625",
                "Tập Train (80%)": "6.900",
                "Tập Val (10%)": "862",
                "Tập Test mỏ neo (10%)": "863",
                "Cửa sổ trượt (DL)": "144 bước (72h)",
                "Đặc điểm tín hiệu & Đánh giá": "Điểm cân bằng vàng: Tiết kiệm 50% chi phí, MASE = 0,382",
            },
            {
                "Độ phân giải": "1 giờ (1h)",
                "Lưới lý thuyết (38 tháng)": "27.648",
                "Tổng mẫu sạch thực tế": "6.689",
                "Tập Train (80%)": "5.351",
                "Tập Val (10%)": "669",
                "Tập Test mỏ neo (10%)": "669",
                "Cửa sổ trượt (DL)": "72 bước (72h)",
                "Đặc điểm tín hiệu & Đánh giá": "Tự tương quan cao (r ≈ 0,86), mất mát xung ô nhiễm ngắn",
            },
        ]
        st.dataframe(pd.DataFrame(table_3_4_rows), use_container_width=True, hide_index=True)
        st.caption(
            "*Ghi chú Bảng 3.4: Khoảng chênh lệch giữa lưới lý thuyết và mẫu sạch thực tế là kết quả của chiến lược Tiered Imputation: "
            "loại bỏ hoàn toàn các khoảng mất tín hiệu kéo dài trên 24 giờ (tổng cộng 19.810 giờ khuyết do mất nguồn/bảo trì trạm Sa Đéc) "
            "nhằm triệt tiêu rủi ro sinh dữ liệu giả mạo (imputation hallucination), bảo toàn tính liêm chính khoa học.*"
        )

        section_header("📐", "Bảng 3.5: Khung Ma Trận Tiêu Chí Kiểm Định Tính Dừng & Thiết Kế Pipeline")
        table_3_5_rows = [
            {
                "Trạng thái kiểm định": "Dừng hoàn toàn (I(0))",
                "Tiêu chí ADF & KPSS": "ADF bác bỏ H₀ (p < 0,05), KPSS không bác bỏ (p > 0,05)",
                "Dạng biến đổi dữ liệu": "Giữ nguyên chuỗi gốc",
                "Họ mô hình áp dụng": "Toàn bộ các mô hình",
                "Mục đích kỹ thuật": "Khai thác tối đa biên độ nồng độ gốc",
            },
            {
                "Trạng thái kiểm định": "Không dừng có xu thế (I(1))",
                "Tiêu chí ADF & KPSS": "ADF không bác bỏ (p > 0,05), KPSS bác bỏ H₀ (p < 0,05)",
                "Dạng biến đổi dữ liệu": "Sai phân bậc 1 (d=1) y'_t = y_t - y_{t-1}",
                "Họ mô hình áp dụng": "ARIMA(p,1,q), Tabular ML",
                "Mục đích kỹ thuật": "Triệt tiêu xu thế dài hạn, tạo đặc trưng vi phân Δy_{t-1}",
            },
            {
                "Trạng thái kiểm định": "Tuần hoàn mùa vụ ngày đêm",
                "Tiêu chí ADF & KPSS": "ACF lặp lại đỉnh sóng tại bội số 24h",
                "Dạng biến đổi dữ liệu": "Sai phân mùa (D=1, S=24) y''_t = y_t - y_{t-24}",
                "Họ mô hình áp dụng": "SARIMA(p,d,q)×(P,1,Q)₂₄",
                "Mục đích kỹ thuật": "Triệt tiêu biến động chu kỳ nghịch nhiệt ngày đêm",
            },
            {
                "Trạng thái kiểm định": "Phương sai không thuần nhất",
                "Tiêu chí ADF & KPSS": "Phân phối lệch phải đuôi dài (Skewness = 2,0046 > 1,5)",
                "Dạng biến đổi dữ liệu": "Biến đổi logarit tự nhiên ln(PM2.5)",
                "Họ mô hình áp dụng": "ElasticNet, Linear Ridge",
                "Mục đích kỹ thuật": "Ổn định phương sai, nén biên độ cực trị dị thường",
            },
            {
                "Trạng thái kiểm định": "Học sâu (Deep Learning)",
                "Tiêu chí ADF & KPSS": "Cơ chế cổng hồi quy và Multi-Head Attention",
                "Dạng biến đổi dữ liệu": "Chuỗi gốc chuẩn hóa thang đo Robust / MinMax",
                "Họ mô hình áp dụng": "GRU, LSTM, TFT",
                "Mục đích kỹ thuật": "Mạng nơ-ron tự học phi tuyến mức nền mà không cần sai phân",
            },
        ]
        st.dataframe(pd.DataFrame(table_3_5_rows), use_container_width=True, hide_index=True)

        insight_card(
            "💡 Phương Pháp Luận Thiết Kế Ma Trận (Chương 3 §3.4, §3.5 & Chương 4 §4.2)",
            "Nghiên cứu áp dụng quy trình đánh giá công bằng (Fair Benchmark) trên cả 3 độ phân giải không gian thời gian. "
            "Trong 41 cấu hình thực nghiệm, <b>36 mô hình tham số</b> được huấn luyện và đánh giá chẩn đoán độ lệch (Forecast Bias) độc lập tại các chân trời dự báo 1h, 6h và 24h. "
            f"Toàn bộ quy trình tối ưu siêu tham số dựa trên Bayesian Optimization (Optuna) {cite('akiba2019')}, "
            "hoàn tất trong <b>dưới 45 phút</b>, chứng minh tính khả thi cao khi triển khai tại các trạm quan trắc tài nguyên môi trường.",
        )

    with tab_lgb:
        lgbm_cfg = hp_configs.get("lightgbm", {})
        section_header("🌲", "LightGBM (Optuna Bayesian Optimization)")
        if lgbm_cfg:
            lgbm_table = pd.DataFrame(
                {
                    "Tham số": lgbm_cfg["params"],
                    "h=1": lgbm_cfg["horizons"]["h1"],
                    "h=6": lgbm_cfg["horizons"]["h6"],
                    "h=24": lgbm_cfg["horizons"]["h24"],
                }
            )
            st.dataframe(lgbm_table, use_container_width=True, hide_index=True)
            st.caption(f"*{lgbm_cfg.get('description', '')}*")
            insight_card(
                "💡 Phân tích LightGBM",
                f"LightGBM {cite('ke2017')} khai thác triệt để 119 đặc trưng trễ, thống kê cửa sổ và tương tác khí tượng. "
                f"Nhờ cơ chế TPE của Optuna {cite('akiba2019')}, mô hình tự động chọn các siêu tham số tối ưu hóa trực tiếp hàm mục tiêu MAE (L1 loss).",
            )

    with tab_tft:
        tft_cfg = hp_configs.get("simplified_tft", {})
        section_header("⚡", "Simplified Temporal Fusion Transformer (PyTorch Native)")
        if tft_cfg:
            tft_params = tft_cfg.get("params", {})
            st.markdown(f"**Kiến trúc:** `{tft_cfg.get('architecture', '')}`")
            tft_table = pd.DataFrame(
                {
                    "Tham số": list(tft_params.keys()),
                    "Giá trị": [str(v) for v in tft_params.values()],
                }
            )
            st.dataframe(tft_table, use_container_width=True, hide_index=True)
            insight_card(
                "💡 Ưu điểm của Simplified TFT (25.089 Tham số)",
                f"Được thiết kế tinh gọn bằng PyTorch thuần dựa trên kiến trúc Temporal Fusion Transformer {cite('lim2021')} "
                "để tối ưu cho chuỗi dữ liệu môi trường vừa và nhỏ. "
                "TFT tích hợp Variable Selection Network (VSN) giúp lọc đặc trưng nhiễu và Interpretable Multi-Head Attention giúp bắt trọn tương tác ngắn hạn (đạt MASE tiệm cận 1,029 ở 1h) với kích thước tham số nhỏ hơn GRU 39%.",
            )

    with tab_dl:
        dl_cfg = hp_configs.get("deep_learning", {})
        section_header("🧠", "Học Sâu Chuỗi Thời Gian (GRU / LSTM)")
        if dl_cfg:
            dl_params = dl_cfg["params"]
            dl_table = pd.DataFrame(
                {
                    "Tham số": list(dl_params.keys()),
                    "Giá trị": [str(v) for v in dl_params.values()],
                }
            )
            st.dataframe(dl_table, use_container_width=True, hide_index=True)
            st.caption(f"*{dl_cfg.get('description', '')}*")
            insight_card(
                "💡 GRU & LSTM phá vỡ giới hạn tự tương quan",
                f"Tại độ phân giải 15m, mô hình GRU {cite('cho2014')} và LSTM {cite('hochreiter1997')} tận dụng cơ chế cổng hồi quy để học các mẫu hình vi mô không tuyến tính, "
                "đạt MASE = 0,667 (vượt trội hơn hẳn Persistence MASE = 1,000).",
            )

    with tab_skl:
        skl_cfg = hp_configs.get("sklearn_tabular", {})
        section_header("🌳", "Mô Hình Bảng Bổ Sung (Random Forest & ElasticNet)")
        if skl_cfg:
            models_list = skl_cfg.get("models", [])
            for m in models_list:
                st.markdown(f"### 🔹 {m['name']} ({m.get('family', '')})")
                details = {k: str(v) for k, v in m.items() if k not in ["name", "family"]}
                st.dataframe(
                    pd.DataFrame(list(details.items()), columns=["Thuộc tính", "Giá trị"]),
                    use_container_width=True,
                    hide_index=True,
                )
            insight_card(
                "💡 Vai trò của ElasticNet & Random Forest",
                f"<b>ElasticNet:</b> Kế thừa phương pháp hồi quy chính quy hóa {cite('joseph2022')} kết hợp điều hòa L1/L2 trên dữ liệu biến đổi logarit ln(PM2.5) giúp ổn định phương sai cho chuỗi phân phối lệch phải đuôi dài (Skewness = 2,00).<br>"
                f"<b>Random Forest:</b> Đóng vai trò kiểm chuẩn tính vững giảm phương sai của họ mô hình Bagging với 100 cây quyết định {cite('breiman2001')}.",
            )

    with tab_ens:
        ens_cfg = hp_configs.get("stacking_ensemble", {})
        section_header("🔀", "Mô Hình Kết Hợp (Stacking & Weighted Blending)")
        st.markdown(f"**Meta-Learner:** `{ens_cfg.get('meta_learner', 'Ridge Regression / Blending')}`")
        st.markdown(f"**Mô hình thành phần (Base Models):** {', '.join(ens_cfg.get('base_models', []))}")
        st.markdown(f"**Chiến lược kết hợp:** {ens_cfg.get('combination_strategy', 'Inverse Variance Weighting')}")
        insight_card(
            "💡 Đỉnh Cao Hiệu Năng ở 30m (MASE = 0,382)",
            f"Mô hình Weighted Ensemble tại độ phân giải 30m vận dụng lý thuyết Stacking Generalization của Wolpert {cite('wolpert1992')} "
            f"và nguyên tắc đa dạng mô hình của Dietterich {cite('dietterich2000')}, kết hợp sức mạnh phi tuyến của LightGBM và khả năng học biểu diễn chuỗi của GRU, "
            "đạt kỷ lục MASE = 0,382 (giảm 61,8% sai số so với baseline).",
        )

    with tab_ari:
        arima_cfg = hp_configs.get("arima", {})
        section_header("📈", "ARIMA / SARIMA (Chuẩn Thống Kê Cổ Điển)")
        if arima_cfg:
            arima_models = arima_cfg["models"]
            arima_table = pd.DataFrame(
                {
                    "Mô hình": [m["name"] for m in arima_models],
                    "Bậc (p,d,q)": [m["order"] for m in arima_models],
                    "Seasonal (P,D,Q,s)": [m.get("seasonal_order") or "—" for m in arima_models],
                    "Phương pháp chọn": [m["method"] for m in arima_models],
                    "Rolling window": [m["rolling_window"] for m in arima_models],
                }
            )
            st.dataframe(arima_table, use_container_width=True, hide_index=True)
            insight_card(
                "💡 Hạn chế của Thống kê Tuyến tính Box-Jenkins",
                f"Phương pháp Box-Jenkins kinh điển {cite('box2015')} chỉ dựa trên quá khứ đơn biến, không khai thác được 119 đặc trưng ngoại sinh và tương tác khí tượng phức tạp, dẫn đến sai số MASE tăng cao khi horizon mở rộng.",
            )

    with tab_raw:
        if configs or hp_configs:
            section_header("📄", "Raw Configurations (JSON)")
            st.json(hp_configs if hp_configs else configs)

    render_references_section()
