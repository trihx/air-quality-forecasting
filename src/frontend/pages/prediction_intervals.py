"""Prediction Intervals page for PM2.5 forecasting dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.frontend.citations import cite, render_references_section
from src.frontend.components import (
    CHART_COLORS,
    RESEARCH_DIR,
    insight_card,
    kpi_card,
    load_json,
    section_header,
)
from src.info_cards import (
    cards_prediction_intervals,
    get_current_version,
    render_version_badge,
)
from src.viz.chart_factory import (
    add_baseline,
    add_simple_bar_labels,
)
from src.viz.chart_factory import (
    chart as _chart,
)
from src.viz.chart_factory import (
    render_chart as _render_chart,
)


def page_prediction_intervals(results: dict[str, Any]) -> None:
    """Render prediction intervals and uncertainty quantification (UQ) page."""
    st.markdown(
        """
    <h1 style="font-size: 2rem;">📈 Khoảng Dự Báo Bất Định (Prediction Intervals)</h1>
    <p style="opacity: 0.7;">Khoảng dự báo 90% — Adaptive Conformal Inference (ACI) vs Conformalized Quantile Regression (CQR)</p>
    """,
        unsafe_allow_html=True,
    )

    # ── Version-aware info cards ──
    ver = get_current_version()
    render_version_badge(ver)
    cards_prediction_intervals(ver)

    pi_data = results.get("prediction_intervals", [])
    aci_data = results.get("aci_intervals", [])

    if not aci_data and not pi_data:
        st.warning("Chưa có kết quả. Chạy `uv run python scripts/v8_aci_prediction_intervals.py`")
        return

    # Ensure pi_data is loaded (fallback to academic artifact if needed)
    if not pi_data:
        pi_fallback_file = (
            RESEARCH_DIR / "experiments" / "prediction_intervals" / "prediction_intervals_20260904_134804.json"
        )
        if pi_fallback_file.exists():
            pi_data = load_json(pi_fallback_file) or []

    # Filter ACI for gamma=0.01
    aci_best = [d for d in aci_data if d.get("gamma") == 0.01]

    tab_uq, tab_aci = st.tabs(
        [
            "📊 Đối chuẩn 3 Phương Pháp UQ (Bảng 4.6 Đề Án)",
            "🔄 Hiệu chuẩn Thích ứng (ACI vs CQR Tĩnh — Concept Drift)",
        ]
    )

    # ── TAB 1: 3 UQ Methods Benchmark (Table 4.6) ──
    with tab_uq:
        if pi_data:
            pi_df = pd.DataFrame(pi_data)
            best_pi = pi_df.loc[pi_df["coverage"].idxmax()]
            st.markdown(
                f"""
<div class="kpi-row">
    {kpi_card("Best Coverage (UQ)", f"{best_pi['coverage']:.1%}", f"{best_pi['method']} — {best_pi['model']} {best_pi['horizon']}h")}
    {kpi_card("Phương Pháp Kiểm Thử", "3", "Conformal (CQR) · Quantile Reg · MC Dropout")}
    {kpi_card("Mục Tiêu Độ Phủ", "90%", f"α = 0.10 {cite('romano2019')} {cite('gibbs2021')}")}
</div>
""",
                unsafe_allow_html=True,
            )

            col_uq1, col_uq2 = st.columns(2)
            with col_uq1:
                section_header("📊", "Coverage vs Target (90%)")
                fig_uq1 = _chart(
                    yaxis_title="Coverage (%)",
                    height=380,
                    barmode="group",
                    layout_overrides={"yaxis_range": [0, 110]},
                )
                methods = pi_df["method"].unique()
                for i, method in enumerate(methods):
                    subset = pi_df[pi_df["method"] == method]
                    fig_uq1.add_trace(
                        go.Bar(
                            name=method.replace("_", " ").title(),
                            x=[f"{r['horizon']}h" for _, r in subset.iterrows()],
                            y=[v * 100 for v in subset["coverage"].values],
                            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                            text=[f"{v:.1%}" for v in subset["coverage"]],
                        )
                    )
                add_baseline(fig_uq1, y=90, label="Target 90%")
                add_simple_bar_labels(fig_uq1, orientation="v")
                _render_chart(fig_uq1, filename="coverage_pi")

            with col_uq2:
                section_header("📏", "Độ Rộng Khoảng Dự Báo (µg/m³)")
                fig_uq2 = _chart(
                    yaxis_title="Avg Width (µg/m³)",
                    height=380,
                    barmode="group",
                )
                for i, method in enumerate(methods):
                    subset = pi_df[pi_df["method"] == method]
                    fig_uq2.add_trace(
                        go.Bar(
                            name=method.replace("_", " ").title(),
                            x=[f"{r['horizon']}h" for _, r in subset.iterrows()],
                            y=subset["avg_width"].values,
                            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                            text=[f"{v:.1f}" for v in subset["avg_width"]],
                        )
                    )
                add_simple_bar_labels(fig_uq2, orientation="v")
                _render_chart(fig_uq2, filename="width_pi")

            # Official Table 4.6 from Thesis
            from src.reporting.content import ContentManager

            content_mgr = ContentManager()
            conformal_table = content_mgr.get_conformal_prediction()

            section_header(
                "📋",
                "Bảng 4.6: Kết Quả Đánh Giá Khoảng Dự Báo 90% (Quantile Regression, CQR và ACI)",
            )
            if conformal_table:
                st.dataframe(
                    pd.DataFrame(conformal_table),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "NMPIW": st.column_config.NumberColumn(format="%.2f"),
                        "Winkler Score": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
            else:
                display_pi_df = pi_df[
                    [
                        "method",
                        "model",
                        "horizon",
                        "coverage",
                        "avg_width",
                        "mae",
                    ]
                ].copy()
                display_pi_df["coverage"] = display_pi_df["coverage"].apply(lambda x: f"{x:.1%}")
                display_pi_df["avg_width"] = display_pi_df["avg_width"].apply(lambda x: f"{x:.2f}")
                display_pi_df["mae"] = display_pi_df["mae"].apply(lambda x: f"{x:.3f}")
                display_pi_df.columns = [
                    "Phương pháp",
                    "Mô hình",
                    "Horizon (h)",
                    "Độ phủ (Coverage)",
                    "Độ rộng TB (µg/m³)",
                    "MAE (µg/m³)",
                ]
                st.dataframe(display_pi_df, use_container_width=True, hide_index=True)

            insight_card(
                "💡 Kết Luận Học Thuật (Bảng 4.6 Đề Án)",
                "<b>1. Thất bại của MC Dropout (14.9% - 19.2%):</b> Mặc dù mạng nơ-ron học sâu biểu diễn tốt giá trị trung bình, MC Dropout đánh giá thấp phương sai thực tế trong dữ liệu chuỗi thời gian do hiện tượng overconfidence của mạng nơ-ron.<br>"
                "<b>2. Quantile Regression gốc (79.1% - 86.2%):</b> Đạt độ phủ khá nhưng độ rộng chuẩn hóa (NMPIW lên tới 0.62 ở 24h) và điểm phạt Winkler Score cao (12.35), làm giảm tính thực dụng khi hỗ trợ ra quyết định y tế.<br>"
                "<b>3. Conformal Prediction tĩnh (CQR, 76.0% - 80.5%):</b> Bị sụt giảm độ phủ dưới hiện tượng trôi dạt khái niệm (concept drift), không đạt mục tiêu 90% khi phân phối tập kiểm thử biến động mạnh.<br>"
                "<b>4. Adaptive Conformal Inference (ACI, γ=0.01):</b> Duy trì độ phủ lý tưởng <b>89.4% – 89.6%</b> (bám sát mục tiêu 90%) trên tất cả horizons, đồng thời tối ưu hóa độ rộng NMPIW (0.20 – 0.35) và đạt điểm <b>Winkler Score thấp nhất (4.12 – 9.85)</b>, chứng minh tính ưu việt toán học tuyệt đối của phương pháp thích ứng.",
            )
        else:
            st.info("Chưa có dữ liệu đối chuẩn 3 phương pháp UQ.")

    # ── TAB 2: ACI vs CQR Calibration (Concept Drift) ──
    with tab_aci:
        if aci_best:
            aci_df = pd.DataFrame(aci_best)
            best_aci = aci_df.loc[aci_df["coverage"].idxmax()]

            st.markdown(
                f"""
<div class="kpi-row">
    {kpi_card("Best Coverage (ACI)", f"{best_aci['coverage']:.1%}", f"ACI (γ={best_aci['gamma']}) — {best_aci['horizon']}h")}
    {kpi_card("Improvement over CQR", f"+{best_aci['improvement_over_cqr'] * 100:.1f} pts", f"tại mốc {best_aci['horizon']}h")}
    {kpi_card("Mục Tiêu Độ Phủ", "90%", f"α = 0.10 {cite('romano2019')} {cite('gibbs2021')}")}
</div>
""",
                unsafe_allow_html=True,
            )

            col_aci1, col_aci2 = st.columns(2)
            with col_aci1:
                section_header("📊", "Coverage vs Target (90%) — ACI vs CQR")
                fig_aci1 = _chart(
                    yaxis_title="Coverage (%)",
                    height=380,
                    barmode="group",
                    layout_overrides={"yaxis_range": [0, 110]},
                )
                horizons_str = [f"{h}h" for h in aci_df["horizon"]]
                fig_aci1.add_trace(
                    go.Bar(
                        name="Adaptive Conformal (ACI)",
                        x=horizons_str,
                        y=[v * 100 for v in aci_df["coverage"].values],
                        marker_color="#00D4AA",
                        text=[f"{v:.1%}" for v in aci_df["coverage"]],
                    )
                )
                fig_aci1.add_trace(
                    go.Bar(
                        name="Static Conformal (CQR)",
                        x=horizons_str,
                        y=[v * 100 for v in aci_df["cqr_coverage"].values],
                        marker_color="#F43F5E",
                        text=[f"{v:.1%}" for v in aci_df["cqr_coverage"]],
                    )
                )
                add_baseline(fig_aci1, y=90, label="Target 90%")
                add_simple_bar_labels(fig_aci1, orientation="v")
                _render_chart(fig_aci1, filename="coverage_aci_cqr")

            with col_aci2:
                section_header("📏", "Interval Width (µg/m³)")
                fig_aci2 = _chart(
                    yaxis_title="Avg Width (µg/m³)",
                    height=380,
                    barmode="group",
                )
                fig_aci2.add_trace(
                    go.Bar(
                        name="Adaptive Conformal (ACI)",
                        x=horizons_str,
                        y=aci_df["avg_width"].values,
                        marker_color="#00D4AA",
                        text=[f"{v:.1f}" for v in aci_df["avg_width"]],
                    )
                )
                fig_aci2.add_trace(
                    go.Bar(
                        name="Static Conformal (CQR)",
                        x=horizons_str,
                        y=aci_df["cqr_avg_width"].values,
                        marker_color="#F43F5E",
                        text=[f"{v:.1f}" for v in aci_df["cqr_avg_width"]],
                    )
                )
                add_simple_bar_labels(fig_aci2, orientation="v")
                _render_chart(fig_aci2, filename="width_aci_cqr")

            section_header("📋", "Tổng Hợp Chi Tiết ACI (γ=0.01) vs CQR Tĩnh")
            display_df = aci_df[
                [
                    "horizon",
                    "gamma",
                    "coverage",
                    "cqr_coverage",
                    "avg_width",
                    "cqr_avg_width",
                ]
            ].copy()
            display_df["coverage"] = display_df["coverage"].apply(lambda x: f"{x:.1%}")
            display_df["cqr_coverage"] = display_df["cqr_coverage"].apply(lambda x: f"{x:.1%}")
            display_df["avg_width"] = display_df["avg_width"].apply(lambda x: f"{x:.1f}")
            display_df["cqr_avg_width"] = display_df["cqr_avg_width"].apply(lambda x: f"{x:.1f}")
            display_df.columns = [
                "Horizon (h)",
                "Gamma",
                "ACI Coverage",
                "CQR Coverage",
                "ACI Width",
                "CQR Width",
            ]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            insight_card(
                "💡 Phân tích Trade-off (Coverage vs Interval Width)",
                f"<b>Biểu đồ 1 - Distribution Shift:</b> Phương pháp CQR (tĩnh) {cite('romano2019')} sụt giảm độ phủ mạnh (đặc biệt ở 6h chỉ còn ~74%), chứng tỏ dữ liệu Test có biến động lớn hơn Calibration. Trong khi đó, <b>Adaptive Conformal Inference (ACI)</b> {cite('gibbs2021')} tự động bù đắp sai số, giữ vững độ phủ bám sát mục tiêu 90% (đạt ~89.5% ở tất cả horizons) mà không cần train lại model.<br><br>"
                "<b>Biểu đồ 2 - Sự đánh đổi (Width vs Coverage):</b> Để đạt được độ phủ 90% ổn định trong điều kiện nhiễu (như ở 6h), phương pháp ACI buộc phải nới rộng khoảng dự báo (Avg Width tăng lên 16.1 µg/m³ so với 12.5 của CQR). Đây là sự đánh đổi bắt buộc (trade-off) và hoàn toàn hợp lý về mặt toán học để đảm bảo tính toàn vẹn của khoảng tin cậy dưới rủi ro distribution shift.",
            )
        else:
            st.info("Chưa có dữ liệu ACI intervals.")

    # ── Đề án Figures 4.11a-c (300 DPI Conformal Prediction Intervals) ──
    section_header(
        "🎯",
        "Trực Quan Dải Băng Khoảng Dự Báo Conformal Prediction (300 DPI)",
    )
    h_pi = st.selectbox("Chọn Horizon hiển thị:", ["1h", "6h", "24h"], key="pi_fig_horizon")
    pi_fig_map = {
        "1h": (
            RESEARCH_DIR / "figures" / "thesis" / "Hinh_4.11a_PI_Conformal_LightGBM_1h.png",
            "Hình 4.11a: Khoảng dự báo 90% Conformalized Quantile Regression (CQR) kết hợp LightGBM trên tập kiểm tra mốc 1 giờ",
        ),
        "6h": (
            RESEARCH_DIR / "figures" / "thesis" / "Hinh_4.11b_PI_Conformal_LightGBM_6h.png",
            "Hình 4.11b: Khoảng dự báo 90% Conformalized Quantile Regression (CQR) kết hợp LightGBM trên tập kiểm tra mốc 6 giờ",
        ),
        "24h": (
            RESEARCH_DIR / "figures" / "thesis" / "Hinh_4.11c_PI_Conformal_LightGBM_24h.png",
            "Hình 4.11c: Khoảng dự báo 90% Conformalized Quantile Regression (CQR) kết hợp LightGBM trên tập kiểm tra mốc 24 giờ",
        ),
    }
    target_pi_img, target_pi_cap = pi_fig_map.get(h_pi, (None, ""))
    if target_pi_img and target_pi_img.exists():
        st.image(str(target_pi_img), caption=target_pi_cap, use_container_width=True)
    else:
        st.info("Chưa có ảnh dải băng khoảng dự báo 300 DPI.")

    # ── References ──
    render_references_section()
