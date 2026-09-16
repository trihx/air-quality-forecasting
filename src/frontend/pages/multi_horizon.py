"""Multi-Horizon Forecasting Evaluation page for PM2.5 forecasting dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.frontend.citations import cite, render_references_section
from src.frontend.components import (
    insight_card,
    section_header,
)
from src.info_cards import (
    cards_multi_horizon,
    get_current_version,
    get_version_data,
    render_version_badge,
)
from src.reporting import ReportingEngine
from src.reporting import charts as rpt_charts
from src.reporting.content import ContentManager
from src.reporting.engine import get_model_type
from src.viz.chart_factory import (
    render_chart as _render_chart,
)


def page_multi_horizon(results: dict[str, Any]) -> None:
    """Render Multi-Horizon performance comparison page."""
    st.markdown(
        """
    <h1 style="font-size: 2rem;">📊 Kết Quả Multi-Horizon</h1>
    <p style="opacity: 0.7;">So sánh hiệu suất dự báo PM2.5 tại 3 horizons: 1h, 6h, 24h</p>
    """,
        unsafe_allow_html=True,
    )

    # ── Version-aware info cards ──
    ver = get_current_version()
    render_version_badge(ver)
    cards_multi_horizon(ver)

    # ── Methodology note ──
    st.markdown(
        f"""
    <div style="background: var(--secondary-background-color); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.2); color: var(--text-color) !important;">
        <div style="font-size: 0.85rem; opacity: 0.65;">
            📌 Metrics chính: <b>MASE</b> {cite("hyndman2006")} (scale-independent, unified baseline),
            MAE {cite("willmott2005")}, và Forecast Bias {cite("hyndman2021")}.
            Đánh giá trên temporal test set (80/10/10) {cite("tashman2000")}, chỉ dùng real data.
            So sánh thống kê giữa models bằng Diebold-Mariano test {cite("diebold1995")}.
            Họ mô hình: LightGBM {cite("ke2017")}, GRU {cite("cho2014")}, Ensemble {cite("peixeiro2022")}.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Initialize ReportingEngine & ContentManager ──
    v_data = get_version_data(ver) if ver else {}
    rpt = ReportingEngine(v_data)
    content = ContentManager()
    insights = rpt.generate_insights()

    # ── MASE: Top-5 Representatives (clean bar chart) ──
    section_header("📊", f"MASE — Top Models theo Family ({rpt.version} — unified baseline)")
    fig_mase_top5 = rpt_charts.plot_mase_comparison_top5(rpt)
    _render_chart(fig_mase_top5, filename="mase_top5")
    st.caption(
        "*Mỗi cột đại diện cho mô hình tốt nhất (MASE thấp nhất trung bình) của từng họ. Đường đỏ MASE=1.0 = Persistence Baseline.*"
    )

    # ── MASE Ranking Table (interactive) ──
    section_header("🏅", "Bảng Xếp Hạng MASE — Chọn Horizon & Models")
    col_hz_m, col_n_m = st.columns([1, 1])
    with col_hz_m:
        hz_mase = st.selectbox(
            "Horizon:",
            ["1h", "6h", "24h"],
            index=1,
            key="mase_ranking_horizon",
        )
    with col_n_m:
        topn_mase = st.selectbox(
            "Số lượng hiển thị:",
            [5, 10, 15, 20, "Tất cả"],
            index=1,
            key="mase_ranking_topn",
        )

    ranked_mase_models = rpt.get_models_ranked_by_mase(hz_mase)
    actual_topn_m = int(topn_mase) if isinstance(topn_mase, int) else len(ranked_mase_models)

    selected_mase_models = st.multiselect(
        f"🔍 Lọc mô hình (xếp theo ranking, mặc định Top {topn_mase}):",
        options=ranked_mase_models,
        default=(ranked_mase_models[:actual_topn_m] if len(ranked_mase_models) > actual_topn_m else ranked_mase_models),
        key=f"mase_ranking_filter_{hz_mase}_{topn_mase}",
    )

    mase_ranking_df = rpt.get_mase_ranking_table(hz_mase, top_n=len(ranked_mase_models))
    if selected_mase_models:
        mase_ranking_df = mase_ranking_df[mase_ranking_df["Model"].isin(selected_mase_models)]
    mase_ranking_df = mase_ranking_df.head(actual_topn_m)

    st.dataframe(
        mase_ranking_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "MASE": st.column_config.NumberColumn(format="%.3f"),
            "MAE (µg/m³)": st.column_config.NumberColumn(format="%.3f"),
            "RMSE (µg/m³)": st.column_config.NumberColumn(format="%.3f"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
            "DA (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "vs Persistence (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption(
        f"*Tất cả MASE sử dụng Unified Persistence MAE. Source: {rpt.version} snapshot ({len(rpt.models)} models).*"
    )

    # ── Expander: Full 41-model bar chart ──
    with st.expander(
        f"📋 Xem toàn bộ {len(rpt.models)} models (MASE Bar Chart)",
        expanded=False,
    ):
        fig_mase_full = rpt_charts.plot_mase_comparison(rpt)
        _render_chart(fig_mase_full, filename="mase_full")

    # ── Data Storytelling: 3 insights by horizon (dynamic) ──
    col1, col2, col3 = st.columns(3)
    with col1:
        insight_card(
            "🟢 h=1: Autocorrelation Trap",
            insights["h1"],
        )
    with col2:
        insight_card(
            f"🟢 h=6: {rpt.get_best_model('6h')['model']} Leads",
            insights["h6"],
        )
    with col3:
        insight_card(
            f"🔵 h=24: {rpt.get_best_model('24h')['model']} = Champion",
            insights["h24"],
        )

    # ── MAE Trend: Top-5 Representatives (clean chart) ──
    section_header("📈", "MAE Theo Horizon — Top Models theo Family")
    fig_mae_top5 = rpt_charts.plot_mae_trend_top5(rpt)
    _render_chart(fig_mae_top5, filename="mae_top5")
    st.caption(
        "*Mỗi cột đại diện cho mô hình tốt nhất (MASE thấp nhất trung bình) của từng họ: Baseline, Statistical, ML, Deep Learning, Transformer, Ensemble. Sắp xếp từ trái → phải theo MAE trung bình tăng dần.*"
    )

    # ── MAE Ranking Table (interactive) ──
    section_header("🏆", "Bảng Xếp Hạng MAE — Chọn Horizon & Models")
    col_hz, col_n = st.columns([1, 1])
    with col_hz:
        hz_select = st.selectbox(
            "Horizon:",
            ["1h", "6h", "24h"],
            index=1,
            key="mae_ranking_horizon",
        )
    with col_n:
        top_n_select = st.selectbox(
            "Số lượng hiển thị:",
            [5, 10, 15, 20, "Tất cả"],
            index=1,
            key="mae_ranking_topn",
        )

    ranked_models = rpt.get_models_ranked_by_mae(hz_select)
    actual_top_n = int(top_n_select) if isinstance(top_n_select, int) else len(ranked_models)

    selected_models = st.multiselect(
        f"🔍 Lọc mô hình (xếp theo ranking, mặc định Top {top_n_select}):",
        options=ranked_models,
        default=(ranked_models[:actual_top_n] if len(ranked_models) > actual_top_n else ranked_models),
        key=f"mae_ranking_filter_{hz_select}_{top_n_select}",
    )

    ranking_df = rpt.get_mae_ranking_table(hz_select, top_n=len(ranked_models))
    if selected_models:
        ranking_df = ranking_df[ranking_df["Model"].isin(selected_models)]
    ranking_df = ranking_df.head(actual_top_n)

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAE (µg/m³)": st.column_config.NumberColumn(format="%.3f"),
            "RMSE (µg/m³)": st.column_config.NumberColumn(format="%.3f"),
            "MASE": st.column_config.NumberColumn(format="%.3f"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
            "DA (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "vs Persistence (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    # ── Expander: Full 41-model spaghetti chart ──
    with st.expander(
        f"📋 Xem toàn bộ {len(rpt.models)} models (Spaghetti Chart)",
        expanded=False,
    ):
        fig_mae_full = rpt_charts.plot_mae_trend(rpt)
        _render_chart(fig_mae_full, filename="mae_full")

    # ── Summary insight (dynamic) ──
    b1 = rpt.get_best_model("1h")
    b6 = rpt.get_best_model("6h")
    b24 = rpt.get_best_model("24h")
    insight_content = content.get_multi_horizon_insight()
    insight_card(
        insight_content.get("title", "💡 Insight: No Single Best Model"),
        f"{insight_content.get('conclusion', '')}<br>"
        f"• <b>1h</b>: {b1['model']} (MASE={b1['mase']:.3f})"
        f"{' — vượt Persistence!' if b1['mase'] < 1.0 else ' — Persistence vẫn mạnh nhất'}<br>"
        f"• <b>6h</b>: {b6['model']} (MASE={b6['mase']:.3f}) — ↓{abs(b6['improvement_pct']):.1f}% vs Persistence<br>"
        f"• <b>24h</b>: {b24['model']} (MASE={b24['mase']:.3f}) — ↓{abs(b24['improvement_pct']):.1f}% vs Persistence<br><br>"
        f"{insight_content.get('why', '')}",
    )

    # ── Bảng 4.3 Đề án: Tổng Hợp Kết Quả Thực Nghiệm Trên Tập Kiểm Thử Mỏ Neo ──
    section_header("📋", "Bảng 4.3: Tổng Hợp Kết Quả Dự Báo Trên Tập Kiểm Thử Mỏ Neo (Anchor Test Set)")
    st.markdown(
        f"""
    <div style="background: var(--secondary-background-color); border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;
                border-left: 4px solid #00D4AA; font-size: 0.88rem; line-height: 1.65;">
        Bảng đối chuẩn tổng hợp hiệu năng dự báo của các mô hình đại diện thuộc 5 họ thuật toán trên tập kiểm thử mỏ neo
        (<b>Anchor Test Set</b> — 1.200 giờ cố định cuối chuỗi, 100% dữ liệu thực <code>is_imputed == 0</code>) {cite("tashman2000")}.
    </div>
    """,
        unsafe_allow_html=True,
    )

    anchor_summary_data = content.get_anchor_test_summary()
    if anchor_summary_data:
        df_anchor = pd.DataFrame(anchor_summary_data)
        st.dataframe(
            df_anchor,
            use_container_width=True,
            hide_index=True,
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.3f µg/m³"),
                "RMSE": st.column_config.NumberColumn(format="%.3f µg/m³"),
                "MASE": st.column_config.NumberColumn(format="%.3f"),
                "R2": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    insight_card(
        "💡 Giải Mã Hiện Tượng R² Âm Ngoài Mẫu & Hiệu Ứng Phạt Đúp (Double Penalty Effect)",
        f"<b>1. Tại sao một số mô hình ghi nhận R² âm nhẹ tại chân trời xa?</b><br>"
        f"Dưới góc độ phương pháp luận kiểm thử ngoài mẫu (Out-of-sample Evaluation) {cite('tashman2000')}, "
        f"đây là hiện tượng toán học hoàn toàn bình thường trong dự báo chuỗi thời gian ô nhiễm không khí {cite('hyndman2006')}. "
        f"Chuỗi PM2.5 tại Sa Đéc thường xuất hiện các đỉnh xung ô nhiễm cục bộ (Spikes) với biên độ cao trong thời gian ngắn (2–3 giờ). "
        f"Khi mô hình dự báo đỉnh bị lệch pha thời gian dù chỉ 1–2 bước trễ, hàm mất mát bậc hai sẽ chịu <b>hiệu ứng phạt đúp (Double Penalty Effect)</b>: "
        f"sai số bị cộng dồn nghiêm trọng ở cả pha tăng thực tế và pha trễ dự báo, dẫn đến tổng bình phương phần dư $SS_{{res}}$ vượt qua tổng phương sai mẫu $SS_{{tot}}$.<br><br>"
        f"<b>2. Cơ sở lý thuyết chọn MASE thay vì R²:</b><br>"
        f"Dù R² âm nhẹ, mô hình <code>Ensemble_Weighted_v9_30m</code> vẫn đạt mức cải thiện sai số tuyệt đối vượt bậc (giảm 49,6% MAE) "
        f"và chỉ số chuẩn hóa <b>MASE = 0,382</b> tại 6 giờ {cite('hyndman2006')}. Thực tế này củng cố quan điểm lý thuyết kinh điển của Hyndman & Koehler (2006) "
        f"rằng hệ số R² không phản ánh trung thực năng lực dự báo trên chuỗi thời gian ngoài mẫu do quá nhạy cảm với ngoại lệ bình phương, "
        f"và khẳng định việc nghiên cứu chọn <b>MASE</b> làm thước đo cốt lõi là hoàn toàn khoa học và chuẩn xác.",
    )

    # ── Bảng 4.7 Đề án: Kiểm định Diebold-Mariano ──
    section_header("📐", "Bảng 4.7: Kiểm Định Ý Nghĩa Thống Kê Diebold-Mariano Giữa Các Mô Hình")
    dm_data_list = content.get_dm_test_data()
    dm_data = pd.DataFrame(dm_data_list) if dm_data_list else pd.DataFrame()
    st.dataframe(dm_data, use_container_width=True, hide_index=True)
    st.markdown(
        f'<div style="font-size: 0.85rem; color: gray; margin-top: 0.3rem; margin-bottom: 1.5rem; font-style: italic;">'
        f"Kiểm định Diebold-Mariano {cite('diebold1995')}: Thống kê DM âm với p < 0,05 chứng minh mô hình đề xuất vượt trội "
        f"mô hình đối chuẩn một cách có ý nghĩa thống kê, loại trừ hoàn toàn yếu tố biến động ngẫu nhiên của mẫu thử."
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Bảng 4.8 Đề án: Đánh giá cảnh báo vượt ngưỡng WHO ──
    section_header("🚨", "Bảng 4.8: Hiệu Năng Cảnh Báo Sớm Các Đợt Ô Nhiễm Vượt Ngưỡng WHO (45 µg/m³)")
    who_alert_list = content.get_who_threshold_alert()
    if who_alert_list:
        df_who = pd.DataFrame(who_alert_list)
        st.dataframe(
            df_who,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "F1-Score": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    insight_card(
        "🎯 Giá Trị Thực Tiễn Của Hệ Thống Cảnh Báo Sớm WHO (F1-Score = 0,782)",
        f"Bên cạnh sai số hồi quy liên tục, mô hình <b>Ensemble_Weighted_v9_30m</b> đạt <b>F1-Score = 0,782</b> "
        f"tại mốc 6 giờ với <b>Precision = 0,812</b> và <b>Recall = 0,754</b> trên tổng số 57 đợt bùng phát ô nhiễm vượt ngưỡng "
        f"khuyến cáo của WHO (45 µg/m³ — tiến sát QCVN) {cite('who2021')}. "
        f"Mô hình cho phép phát hiện trước 6 giờ khoảng <b>75% các đợt ô nhiễm nguy hại</b> để kịp thời phát đi khuyến cáo bảo vệ sức khỏe cộng đồng. "
        f"Việc kết hợp cả hai hệ thống tiêu chí — sai số định lượng (MAE, RMSE, MASE) và phân loại nhị phân (Precision, Recall, F1) — "
        f"tuân thủ nghiêm ngặt nguyên tắc phương pháp luận dự báo chuẩn tắc của J. S. Armstrong (2001) {cite('armstrong2001')}.",
    )

    # ── Residual Diagnostics: Forecast Bias ──
    section_header("📊", "Residual Diagnostics — Forecast Bias")
    st.caption(
        "*Forecast Bias = Mean Error trung bình. "
        "Bias dương → model **overestimate** (dự báo cao hơn thực tế). "
        "Bias âm → model **underestimate** (dự báo thấp hơn thực tế). "
        "Bias ≈ 0 → model không thiên lệch hệ thống.*"
    )

    # Build bias table with independent horizon selector
    bias_horizon = st.radio(
        "Chọn Horizon (Forecast Bias):",
        ["1h", "6h", "24h"],
        index=1,
        horizontal=True,
        key="bias_horizon_selector",
    )

    bias_rows = []
    h_data_bias = rpt.results.get(bias_horizon, {})
    for model_name, metrics in h_data_bias.items():
        bias = metrics.get("forecast_bias")
        if bias is None:
            continue
        bias_direction = "Overestimate ↑" if bias > 0.05 else ("Underestimate ↓" if bias < -0.05 else "Neutral ≈ 0")
        bias_rows.append(
            {
                "Model": model_name,
                "Type": get_model_type(model_name),
                "Forecast Bias": round(bias, 4),
                "Direction": bias_direction,
                "Severity": ("✅ Low" if abs(bias) < 0.5 else ("⚠️ Moderate" if abs(bias) < 2.0 else "❌ High")),
            }
        )

    if bias_rows:
        bias_df = pd.DataFrame(bias_rows).sort_values("Forecast Bias", key=abs)

        fig_bias = go.Figure()
        bias_df_sorted = bias_df.sort_values("Forecast Bias")
        colors = ["#EF4444" if b > 0 else "#3B82F6" for b in bias_df_sorted["Forecast Bias"]]

        fig_bias.add_trace(
            go.Bar(
                x=bias_df_sorted["Model"],
                y=bias_df_sorted["Forecast Bias"],
                marker_color=colors,
            )
        )
        fig_bias.update_layout(
            yaxis_title="Forecast Bias (µg/m³)",
            xaxis_tickangle=-45,
            height=400,
            showlegend=False,
            margin={"b": 120},
        )
        fig_bias.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        st.plotly_chart(fig_bias, use_container_width=True)

        with st.expander(
            f"📊 Bảng Chi Tiết Forecast Bias — {bias_horizon} ({len(bias_rows)} models)",
            expanded=False,
        ):
            st.dataframe(bias_df, use_container_width=True, hide_index=True)
            biases = [r["Forecast Bias"] for r in bias_rows]
            avg_bias = sum(biases) / len(biases)
            min_bias_model = min(bias_rows, key=lambda x: abs(x["Forecast Bias"]))
            st.info(
                f"**Mean Bias**: {avg_bias:.4f} | "
                f"**Least Biased**: {min_bias_model['Model']} (bias={min_bias_model['Forecast Bias']:.4f})"
            )

    # ── References ──
    render_references_section()
