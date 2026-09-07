"""
EDA & Data Storytelling Page — PM2.5 Forecasting Dashboard.
Completely synchronized with Master's Thesis Chapter 4 (§4.1.1 - §4.1.5).

Structure:
  1. 📋 Tổng Quan Dữ Liệu (Dataset Overview & Thống kê mô tả)
  2. 📊 §4.1.1 Tính Dừng & 5 Trạng Thái Biến Đổi (Bảng 4.1 & Hình 4.1a-e 300 DPI)
  3. 🔄 §4.1.2 Bẫy Tự Tương Quan & Phân Tán Theo Horizon (Hình 4.2a-b 300 DPI)
  4. ⚡ §4.1.3 Phân Phối Đuôi Dài & Đỉnh Dị Thường (Hình 4.3a-b 300 DPI)
  5. 🔀 §4.1.4 Trôi Dạt Khái Niệm & Mật Độ Đa Biến (Hình 4.4a-b 300 DPI)
  6. 🧱 §4.1.5 Mã Vạch Dữ Liệu Khuyết & Giới Hạn Nội Suy (Hình 4.5a-b 300 DPI)
  7. 💡 §4.1.6 Cơ Sở Thiết Kế Pipeline & Deep Insights (The 'Why' & Diagnostics)
"""

from __future__ import annotations

import calendar
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.frontend.citations import cite, render_references_section
from src.info_cards import cards_eda, get_current_version, render_version_badge
from src.viz.chart_factory import (
    add_simple_bar_labels,
)
from src.viz.chart_factory import (
    chart as _chart,
)
from src.viz.chart_factory import (
    figure_caption as _caption,
)
from src.viz.chart_factory import (
    render_chart as _render_chart,
)
from src.viz.theme import PALETTE_SEMANTIC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
THESIS_FIG_DIR = RESEARCH_DIR / "figures" / "thesis"

COLORS = {
    "primary": PALETTE_SEMANTIC["primary"],
    "secondary": PALETTE_SEMANTIC["secondary"],
    "accent": PALETTE_SEMANTIC["accent"],
    "warning": PALETTE_SEMANTIC["warning"],
    "success": PALETTE_SEMANTIC["success"],
    "danger": PALETTE_SEMANTIC["danger"],
}


def _insight_card(title: str, text: str, card_type: str = "default"):
    """Render standardized insight card."""
    cls = "warning" if card_type == "warning" else ""
    parsed_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    parsed_text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", parsed_text)
    st.markdown(
        f"""
    <div class="insight-card {cls}" style="background: var(--secondary-background-color); 
                border-left: 4px solid #00D4AA; padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; 
                margin: 1rem 0; border: 1px solid rgba(0,212,170,0.2);">
        <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #00D4AA;">{title}</h4>
        <div style="font-size: 0.92rem; line-height: 1.6; opacity: 0.9;">{parsed_text}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def page_eda(results):
    """Render EDA & Data Storytelling page."""
    st.markdown(
        """
    <h1 style="font-size: 2rem;">📉 Cốt Truyện Dữ Liệu (Data Storytelling)</h1>
    <p style="opacity: 0.7; font-size: 1.05rem; margin-bottom: 0.5rem;">
        Hành trình khám phá dữ liệu IoT và cơ sở thực nghiệm nền tảng thiết kế Feature Engineering & Pipeline v9
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Version-aware info cards
    ver = get_current_version()
    render_version_badge(ver)
    cards_eda(ver)

    # Load EDA json cache if available
    eda_json_path = RESEARCH_DIR / "eda" / "eda_results.json"
    eda_data = {}
    if eda_json_path.exists():
        with open(eda_json_path, encoding="utf-8") as f:
            eda_data = json.load(f)

    phase5_json = RESEARCH_DIR / "eda" / "phase5_dashboard_data.json"
    p5_data = {}
    if phase5_json.exists():
        with open(phase5_json, encoding="utf-8") as f:
            p5_data = json.load(f)

    pm25_desc = eda_data.get("descriptive", {}).get("pm25", {})

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "1. 📋 Tổng Quan",
            "2. 📊 §4.1.1 Tính Dừng",
            "3. 🔄 §4.1.2 Bẫy Tự Tương Quan",
            "4. ⚡ §4.1.3 Phân Phối & Spikes",
            "5. 🔀 §4.1.4 Drift & Đa Biến",
            "6. 🧱 §4.1.5 Dữ Liệu Khuyết",
            "7. 💡 §4.1.6 Bài Học & Insights",
        ]
    )

    # ══════════════════════════════════════════════════════════════════
    # TAB 1: 📋 TỔNG QUAN DỮ LIỆU
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 1. 📋 Tổng Quan & Cường Độ Dữ Liệu Quan Trắc (Dataset Overview)")

        # Data Summary Card
        st.markdown(
            """
        <div style="background: var(--secondary-background-color); border-radius: 12px; padding: 1.2rem;
                    margin-bottom: 1.5rem; border: 1px solid rgba(0,212,170,0.2);">
            <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.8rem; color: var(--text-color);">
                📋 Tóm Tắt Bộ Dữ Liệu Quan Trắc Môi Trường Không Khí
            </div>
            <table style="width: 100%; font-size: 0.88rem; color: var(--text-color);">
                <tr><td style="padding: 4px 0; opacity: 0.6;">🏭 Nguồn</td><td>Trạm cảm biến IoT quan trắc ngoài trời tại Sa Đéc, Đồng Tháp (phân tích dữ liệu môi trường cấp huyện)</td></tr>
                <tr><td style="padding: 4px 0; opacity: 0.6;">📅 Giai đoạn</td><td><b>16/03/2022 — 11/05/2025</b> (38 tháng, ~3.1 năm liên tục)</td></tr>
                <tr><td style="padding: 4px 0; opacity: 0.6;">⏱️ Tần suất gốc (Raw IoT)</td><td><b>~2 phút/lần</b> (209.594 bản ghi thô từ cảm biến)</td></tr>
                <tr><td style="padding: 4px 0; opacity: 0.6;">📊 Tái lấy mẫu (Resample)</td><td><b>15 phút</b> (~110K) · <b>30 phút</b> (~55K) · <b>1 giờ</b> (~27K sạch)</td></tr>
                <tr><td style="padding: 4px 0; opacity: 0.6;">🧪 Features</td><td>119 đặc trưng thuộc 7 nhóm (anti-leakage, shift(1) strictly enforced)</td></tr>
                <tr><td style="padding: 4px 0; opacity: 0.6;">✂️ Anchor Test Set</td><td>1.200 giờ cuối cố định — 100% dữ liệu thực không nội suy</td></tr>
            </table>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "Dữ liệu PM2.5 thu thập từ trạm cảm biến IoT Sa Đéc với tần suất gốc **~2 phút/lần (209.594 bản ghi thô, "
            "giai đoạn 16/03/2022 — 11/05/2025)**. Qua quy trình làm sạch và tái lấy mẫu đa phân giải, "
            "chuỗi 1 giờ (1h) gồm 27.649 mốc thời gian hoàn chỉnh với các chỉ số thống kê mô tả nồng độ PM2.5 như sau:"
        )

        cols = st.columns(4)
        cols[0].metric("Tổng số điểm (sau clean)", f"{pm25_desc.get('count', 27649):,}")
        cols[1].metric("Trung bình (Mean)", "17.46 µg/m³")
        cols[2].metric("Trung vị (Median)", "13.67 µg/m³")
        cols[3].metric("Đỉnh điểm (Max)", "138.5 µg/m³", delta="Cực đoan", delta_color="inverse")

        _insight_card(
            "💡 Phân tích Thống Kê & Phân Phối (Skewness = 2,0046, Kurtosis = 6,1458)",
            "Phân phối nồng độ PM2.5 mang đặc tính lệch phải rõ rệt: Mean (17,46 µg/m³) cao hơn đáng kể so với Median (13,67 µg/m³), "
            "Độ lệch (Skewness) = 2,0046 và Độ nhọn (Kurtosis) = 6,1458 với giá trị cực đại đạt 138,5 µg/m³. "
            "Điều này chứng minh dữ liệu môi trường thực tế không tuân theo phân phối chuẩn mà chứa các đợt phát thải cực đoan (Fat-Tailed Spikes), "
            "đòi hỏi mô hình học máy phi tuyến và khoảng tin cậy thích ứng ACI để kiểm soát rủi ro y tế.",
        )

        _insight_card(
            "🚨 Hạn chế & Bài học Thực Nghiệm (Data Sparsity)",
            "**1. Khả năng dự báo:** Mô hình hoạt động rất tốt cho dự báo ngắn hạn (1-24h) nhờ chu kỳ ngày (diurnal) liền mạch. "
            "Tuy nhiên, khả năng bắt chu kỳ mùa (seasonality) bị giới hạn do thiếu hụt 89 ngày/năm (mù hoàn toàn tháng 2 và tháng 9).<br>"
            "**2. Nguyên tắc Nội suy (Imputation):** Tuyệt đối KHÔNG dùng Machine Learning để nội suy các khoảng trống (gaps) dài > 1 tuần. "
            "Việc này gây ra rò rỉ dữ liệu (Data Leakage) và tín hiệu giả (Hallucination). Áp dụng quy tắc cắt bỏ (Drop) cho gap dài.<br>"
            "**3. Giải pháp Cải thiện:** Dự án đã thử nghiệm kéo dữ liệu ngoại lai từ Open-Meteo API để bổ trợ tín hiệu thời tiết cho các vùng bị khuyết.",
        )

        # Calendar Heatmap
        st.markdown("---")
        st.markdown("#### 📅 Calendar Heatmap — PM2.5 Trung Bình Theo Ngày")
        st.markdown("*Nhìn tổng quan nhanh: đợt ô nhiễm kéo dài, khoảng trống dữ liệu, và xu hướng mùa vụ*")

        @st.cache_data(ttl=3600)
        def _get_calendar_data():
            _df = pd.read_csv(
                PROJECT_ROOT / "dataset" / "raw" / "final_dataset.csv",
                parse_dates=["ngay_tao"],
                index_col="ngay_tao",
                usecols=["ngay_tao", "pm25"],
            )
            return _df["pm25"].resample("D").mean().dropna()

        cal_data = _get_calendar_data()
        if cal_data is not None and len(cal_data) > 0:
            dates = cal_data.index
            values = cal_data.values
            years = sorted(dates.year.unique())
            selected_year = st.selectbox("Năm quan trắc:", years, index=len(years) - 1, key="cal_year_eda")

            year_mask = dates.year == selected_year
            year_dates = dates[year_mask]
            year_values = values[year_mask]

            if len(year_dates) > 0:
                weeks = year_dates.isocalendar().week.values
                weekdays = year_dates.weekday.values
                months = year_dates.month.values
                max_week = 53
                z_grid = [[None] * max_week for _ in range(7)]
                text_grid = [[""] * max_week for _ in range(7)]

                for d, v, w, wd in zip(year_dates, year_values, weeks, weekdays):
                    w_idx = int(w) - 1
                    if 0 <= w_idx < max_week:
                        z_grid[wd][w_idx] = round(float(v), 1)
                        text_grid[wd][w_idx] = f"{d.strftime('%Y-%m-%d')}<br>PM2.5: {v:.1f} µg/m³"

                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                month_ticks = []
                for m in range(1, 13):
                    m_dates = year_dates[months == m]
                    if len(m_dates) > 0:
                        m_week = int(m_dates[0].isocalendar().week)
                        month_ticks.append((m_week - 1, month_names[m - 1]))

                fig_cal = _chart(
                    height=250,
                    hovermode="closest",
                    margin=dict(l=40, r=20, t=30, b=30),
                    layout_overrides={
                        "yaxis": dict(
                            tickmode="array",
                            tickvals=list(range(7)),
                            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                            autorange="reversed",
                            tickfont=dict(size=10),
                        ),
                        "xaxis": dict(
                            tickmode="array",
                            tickvals=[t[0] for t in month_ticks],
                            ticktext=[t[1] for t in month_ticks],
                            tickfont=dict(size=11),
                        ),
                    },
                )
                fig_cal.add_trace(
                    go.Heatmap(
                        z=z_grid,
                        text=text_grid,
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=[
                            [0.0, "#00D4AA"],
                            [0.24, "#4CAF50"],
                            [0.50, "#FFC107"],
                            [0.75, "#FF6B35"],
                            [1.0, "#E53935"],
                        ],
                        zmin=0,
                        zmax=50,
                        colorbar=dict(
                            title=dict(text="<b>PM2.5</b><br>(µg/m³)", font=dict(size=11)),
                            thickness=12,
                            len=0.8,
                            tickfont=dict(size=10),
                        ),
                        xgap=2,
                        ygap=2,
                    )
                )
                _render_chart(fig_cal, filename=f"calendar_{selected_year}")
                _caption(f"Calendar Heatmap — PM2.5 trung bình theo ngày năm {selected_year}")

        # Forecastability Assessment
        fc = eda_data.get("forecastability", {})
        if fc:
            st.markdown("---")
            st.markdown("#### 🎯 Forecastability Assessment")
            st.markdown(f"*Đo mức độ khả thi dự báo TRƯỚC khi chọn model {cite('joseph2022')}*", unsafe_allow_html=True)
            fc_cols = st.columns(5)
            fc_cols[0].metric(
                "CoV (σ/μ)", f"{fc.get('cov', 0):.3f}", help="Coefficient of Variation — cao = khó dự báo"
            )
            fc_cols[1].metric(
                "ApEn", f"{fc.get('approximate_entropy', 0):.3f}", help="Approximate Entropy — cao = phức tạp"
            )
            fc_cols[2].metric("Seasonality", f"{fc.get('seasonality_strength', 0):.3f}", help="Sức mạnh mùa vụ từ STL")
            fc_cols[3].metric("ACF(1)", f"{fc.get('acf_lag1', 0):.3f}", help="Tự tương quan lag-1")
            fc_cols[4].metric("Score", f"{fc.get('forecastability_score', 0):.3f}", delta=fc.get("interpretation", ""))

        # Interactive Correlation Heatmap
        st.markdown("---")
        st.markdown("#### 🔥 Ma Trận Tương Quan Đa Biến (Interactive Correlation)")
        st.markdown(
            f"*Pearson đo tương quan tuyến tính, Spearman đo tương quan đơn điệu phi tuyến {cite('zhang2017')}*",
            unsafe_allow_html=True,
        )

        @st.cache_data(ttl=3600)
        def _get_corr_matrices():
            _df = pd.read_csv(
                PROJECT_ROOT / "dataset" / "raw" / "final_dataset.csv",
                usecols=["nhiet_do", "do_am", "diem_suong", "co2", "pm25"],
            ).dropna()
            vars_order = ["nhiet_do", "do_am", "diem_suong", "co2", "pm25"]
            return {
                "pearson": _df[vars_order].corr(method="pearson").round(3).values.tolist(),
                "spearman": _df[vars_order].corr(method="spearman").round(3).values.tolist(),
            }

        corr_matrices = _get_corr_matrices()
        if corr_matrices:
            labels = ["Nhiệt độ", "Độ ẩm", "Điểm sương", "CO₂", "PM2.5"]
            corr_method = st.radio(
                "Phương pháp tương quan:", ["Pearson", "Spearman"], horizontal=True, key="corr_method_eda"
            )
            z_data = corr_matrices["pearson"] if corr_method == "Pearson" else corr_matrices["spearman"]

            fig_corr = _chart(
                height=420,
                hovermode="closest",
                margin=dict(l=20, r=20, t=20, b=20),
                layout_overrides={"xaxis": dict(side="bottom"), "yaxis": dict(autorange="reversed")},
            )
            fig_corr.add_trace(
                go.Heatmap(
                    z=z_data,
                    x=labels,
                    y=labels,
                    colorscale="RdBu_r",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=[[f"{v:.3f}" for v in row] for row in z_data],
                    texttemplate="%{text}",
                    textfont={"size": 13},
                    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
                    colorbar=dict(title=dict(text="r"), thickness=15),
                )
            )
            _render_chart(fig_corr, filename=f"corr_{corr_method.lower()}")
            _caption(f"Ma trận tương quan {corr_method} giữa các biến quan trắc")

            _insight_card(
                "🔥 Nhận Định Tương Quan Đa Biến",
                "• <b>Sự vượt trội của Spearman so với Pearson:</b> Hệ số Spearman luôn cao hơn đáng kể chứng minh mối quan hệ đơn điệu phi tuyến chiếm ưu thế tuyệt đối trong dữ liệu môi trường.<br>"
                "• <b>CO₂:</b> Tương quan tuyến tính (Pearson) thấp (0,069) nhưng tương quan đơn điệu (Spearman) lại dương mạnh nhất (+0,251), xác nhận CO₂ và PM2.5 cùng nguồn phát thải đốt cháy.<br>"
                "• <b>Nhiệt độ:</b> Tương quan âm nhất quán ở cả 2 ma trận (-0,18 đến -0,48), khẳng định hiện tượng nghịch nhiệt làm tăng nồng độ PM2.5.",
            )

        # Mutual Information & Complexity Radar
        c_mi, c_rad = st.columns([1, 1])
        with c_mi:
            st.markdown(f"#### 🧠 Mutual Information {cite('zhang2017')}")
            mi_dict = {"CO₂": 0.142, "Điểm sương": 0.128, "Nhiệt độ": 0.121, "Độ ẩm": 0.108}
            mi_df = pd.DataFrame(list(mi_dict.items()), columns=["Feature", "MI Score"]).sort_values(
                "MI Score", ascending=True
            )
            fig_mi = px.bar(
                mi_df,
                x="MI Score",
                y="Feature",
                orientation="h",
                color="MI Score",
                color_continuous_scale="Viridis",
                text="MI Score",
            )
            add_simple_bar_labels(fig_mi, orientation="h", fmt=".3f")
            fig_mi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, Arial, sans-serif", size=10),
                height=300,
                margin=dict(l=20, r=20, t=10, b=30),
                coloraxis_showscale=False,
                xaxis_title="Mutual Information Score",
                yaxis_title="",
            )
            _render_chart(fig_mi, filename="mutual_info_eda")
            _caption("Mutual Information với Target PM2.5")

        with c_rad:
            st.markdown(f"#### 🕸️ Complexity Profile Radar {cite('kang2017')}")
            radar_data = p5_data.get("complexity_radar")
            if radar_data:
                fig_radar = _chart(height=300, showlegend=False, margin=dict(l=20, r=20, t=10, b=20))
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=radar_data["values"] + [radar_data["values"][0]],
                        theta=radar_data["metrics"] + [radar_data["metrics"][0]],
                        fill="toself",
                        line_color="#00D4AA",
                        fillcolor="rgba(0, 212, 170, 0.3)",
                    )
                )
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(139,149,165,0.3)"),
                        angularaxis=dict(gridcolor="rgba(139,149,165,0.3)"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                _render_chart(fig_radar, filename="complexity_radar_eda")
                _caption("Độ phức tạp đa chiều của chuỗi PM2.5")

    # ══════════════════════════════════════════════════════════════════
    # TAB 2: 📊 §4.1.1 TÍNH DỪNG & 5 TRẠNG THÁI BIẾN ĐỔI
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 2. §4.1.1 Kiểm Định Tính Dừng & Đồ Thị Chẩn Đoán Qua 5 Trạng Thái Biến Đổi")
        st.markdown(
            "Thực thi quy trình kiểm định kép ADF và KPSS trên chuỗi thời gian PM2.5 thực tế quan trắc tại Sa Đéc, "
            "kết quả thu được trên chuỗi gốc và các dạng biến đổi sai phân được lượng hóa trong Bảng 4.1 của Luận văn:"
        )

        # Bảng 4.1 Luận văn
        st.markdown("#### 📋 Bảng 4.1: Kết quả kiểm định tính dừng ADF và KPSS trên chuỗi PM2.5 gốc và sai phân")
        df_bang_4_1 = pd.DataFrame(
            [
                {
                    "Chuỗi dữ liệu": "PM2.5 gốc (Raw)",
                    "Thống kê ADF": "-8,42",
                    "p-value ADF": "< 0,001",
                    "Thống kê KPSS": "1,28",
                    "p-value KPSS": "< 0,01",
                    "Kết luận thực nghiệm": "Dừng có xu hướng (Trend-stationary)",
                },
                {
                    "Chuỗi dữ liệu": "PM2.5 sai phân d=1",
                    "Thống kê ADF": "-24,15",
                    "p-value ADF": "< 0,001",
                    "Thống kê KPSS": "0,08",
                    "p-value KPSS": "> 0,10",
                    "Kết luận thực nghiệm": "Dừng hoàn toàn (Strictly stationary)",
                },
                {
                    "Chuỗi dữ liệu": "PM2.5 sai phân mùa D=1, S=24",
                    "Thống kê ADF": "-31,22",
                    "p-value ADF": "< 0,001",
                    "Thống kê KPSS": "0,04",
                    "p-value KPSS": "> 0,10",
                    "Kết luận thực nghiệm": "Dừng hoàn toàn (Strictly stationary)",
                },
            ]
        )
        st.dataframe(df_bang_4_1, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 🖼️ Biểu đồ Chẩn Đoán Thực Nghiệm 300 DPI Chính Thức (Hình 4.1a – 4.1e)")
        st_opt = st.radio(
            "Chọn trạng thái biến đổi để khảo sát:",
            [
                "Hình 4.1a: PM2.5 Gốc (Raw)",
                "Hình 4.1b: Sai phân bậc 1 (d=1)",
                "Hình 4.1c: Sai phân mùa (Seasonal S=24h)",
                "Hình 4.1d: Biến đổi Log PM2.5",
                "Hình 4.1e: Sai phân bậc 1 của Log PM2.5",
            ],
            horizontal=True,
            key="stat_radio_tabs",
        )

        stat_dict = {
            "Hình 4.1a: PM2.5 Gốc (Raw)": (
                THESIS_FIG_DIR / "Hinh_4.1a_Stationarity_Raw_PM25.png",
                "Hình 4.1a: Kiểm định tính dừng — Chuỗi nồng độ PM2.5 gốc (Raw)",
                "<b>Biện luận thực nghiệm Hình 4.1a:</b> Chuỗi PM2.5 gốc dao động quanh mức trung bình 13,18 μg/m³ với biên độ biến động mở rộng định kỳ trong các mùa khô. "
                "Hàm ACF suy giảm chậm theo hàm mũ (slow decay), duy trì hệ số tự tương quan cao (>0,40) tại độ trễ 72 giờ (3 ngày) với các đỉnh sóng lặp lại sau mỗi chu kỳ 24 giờ phản ánh tính mùa vụ ngày đêm rõ nét. "
                "Biểu đồ PACF xuất hiện gai nhọn vượt bậc tại trễ lag=1 (r ≈ 0,86 trên chuỗi giờ và ~0,97 trên chuỗi thô) rồi ngắt đột ngột. Kết hợp ADF (p < 0,001) và KPSS (p < 0,01) cho thấy chuỗi có tính dừng có xu hướng.<br>"
                "<b>Quyết định kỹ thuật:</b> Đối với các mô hình học sâu (GRU, LSTM, TFT), chuỗi gốc được giữ nguyên dạng nồng độ tuyệt đối vì các cổng hồi quy và cơ chế tự chú ý có khả năng tự học các quy luật tuần hoàn phi tuyến mà không cần triệt tiêu mức nền dữ liệu.",
            ),
            "Hình 4.1b: Sai phân bậc 1 (d=1)": (
                THESIS_FIG_DIR / "Hinh_4.1b_Stationarity_1st_Diff.png",
                "Hình 4.1b: Kiểm định tính dừng — Sai phân bậc 1 (d=1)",
                "<b>Biện luận thực nghiệm Hình 4.1b:</b> Phép sai phân bậc 1 (y'_t = y_t - y_{t-1}) loại bỏ hoàn toàn xu thế dài hạn, đưa giá trị trung bình về xấp xỉ 0,00. "
                "Hàm ACF suy giảm nhanh chóng về 0 sau một bước trễ đầu tiên. Thống kê ADF đạt -24,15 (p < 0,001) và KPSS đạt 0,08 (p > 0,10), khẳng định chuỗi sau sai phân bậc 1 đạt tính dừng hoàn toàn (I(0)).<br>"
                "<b>Quyết định kỹ thuật:</b> Thiết lập bắt buộc bậc sai phân phi mùa vụ d=1 cho mô hình ARIMA(p,1,q) và tạo các đặc trưng sai phân vi phân ngắn hạn (Δy_{t-1}, Δy_{t-2}) cho các mô hình học máy dạng bảng.",
            ),
            "Hình 4.1c: Sai phân mùa (Seasonal S=24h)": (
                THESIS_FIG_DIR / "Hinh_4.1c_Stationarity_Seasonal_Diff_24h.png",
                "Hình 4.1c: Kiểm định tính dừng — Sai phân chu kỳ ngày (Seasonal d=24h)",
                "<b>Biện luận thực nghiệm Hình 4.1c:</b> Sai phân mùa bậc 1 theo chu kỳ 24 giờ (y''_t = y_t - y_{t-24}) triệt tiêu hoàn toàn thành phần biến động ngày đêm do đối lưu nhiệt và nghịch nhiệt bức xạ sáng sớm. "
                "Hàm ACF không còn xuất hiện các đỉnh sóng tại bội số của 24 giờ. Thống kê ADF đạt -31,22 (p < 0,001) và KPSS đạt 0,04 (p > 0,10).<br>"
                "<b>Quyết định kỹ thuật:</b> Xác định cấu hình mùa vụ tối ưu cho mô hình SARIMA là (P, 1, Q)₂₄ với chu kỳ S=24, đồng thời xây dựng đặc trưng sai phân chu kỳ ngày (Δy_{t-24}) trong bộ 119 đặc trưng kỹ nghệ.",
            ),
            "Hình 4.1d: Biến đổi Log PM2.5": (
                THESIS_FIG_DIR / "Hinh_4.1d_Stationarity_Log_PM25.png",
                "Hình 4.1d: Kiểm định tính dừng — Biến đổi Log PM2.5",
                "<b>Biện luận thực nghiệm Hình 4.1d:</b> Biến đổi logarit tự nhiên ln(PM2.5) giúp ổn định phương sai và làm giảm độ lệch phải của phân phối nồng độ đuôi dài. "
                "Phép biến đổi này nén biên độ của các đỉnh ô nhiễm dị thường (spikes).<br>"
                "<b>Quyết định kỹ thuật:</b> Biến đổi log được tích hợp vào nhánh mô hình hồi quy tuyến tính điều hòa (ElasticNet) nhằm hạn chế việc các giá trị cực đoan chi phối quá mức hàm mất mát bình phương tối thiểu.",
            ),
            "Hình 4.1e: Sai phân bậc 1 của Log PM2.5": (
                THESIS_FIG_DIR / "Hinh_4.1e_Stationarity_Log_1st_Diff.png",
                "Hình 4.1e: Kiểm định tính dừng — Sai phân bậc 1 của Log PM2.5",
                "<b>Biện luận thực nghiệm Hình 4.1e:</b> Sự kết hợp giữa biến đổi Log và sai phân bậc 1 tạo ra chuỗi tốc độ tăng trưởng tương đối (log-returns), "
                "đồng thời đạt được hai điều kiện lý tưởng: phương sai thuần nhất và tính dừng hoàn toàn.<br>"
                "<b>Quyết định kỹ thuật:</b> Cung cấp định dạng chuẩn hóa thay thế phục vụ cho việc huấn luyện các mô hình thống kê kiểm chuẩn trong hệ thống kết hợp (Ensemble).",
            ),
        }

        img_f, cap_f, desc_f = stat_dict[st_opt]
        if img_f.exists():
            st.image(str(img_f), caption=cap_f, use_container_width=True)
            st.markdown(
                f"""
            <div style="background: rgba(0,212,170,0.05); border-left: 3px solid #00D4AA; padding: 0.8rem 1rem; border-radius: 4px; margin-top: 0.5rem; margin-bottom: 1.5rem; font-size: 0.92rem; line-height: 1.6;">
                {desc_f}
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Plotly ACF/PACF Interactive Tool
        acf_cache = RESEARCH_DIR / "eda" / "plotly_cache" / "acf_pacf.json"
        if acf_cache.exists():
            with st.expander("🔍 Xem đồ thị tương tác ACF / PACF (Interactive Plotly)", expanded=False):
                with open(acf_cache) as f:
                    acf_data = json.load(f)
                fig_acf = make_subplots(
                    rows=2, cols=1, subplot_titles=("Autocorrelation (ACF)", "Partial Autocorrelation (PACF)")
                )
                lags = list(range(len(acf_data["acf"])))
                fig_acf.add_trace(
                    go.Bar(x=lags, y=acf_data["acf"], name="ACF", marker_color="#3B82F6", width=0.1, showlegend=False),
                    row=1,
                    col=1,
                )
                fig_acf.add_trace(
                    go.Scatter(
                        x=lags,
                        y=acf_data["acf"],
                        mode="markers",
                        marker=dict(color="#3B82F6", size=5),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig_acf.add_trace(
                    go.Scatter(
                        x=lags,
                        y=acf_data["acf_conf_upper"],
                        mode="lines",
                        line=dict(color="rgba(156,163,175,0.5)", width=0),
                        fill="tonexty",
                        fillcolor="rgba(156,163,175,0.2)",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig_acf.add_trace(
                    go.Scatter(
                        x=lags,
                        y=acf_data["acf_conf_lower"],
                        mode="lines",
                        line=dict(color="rgba(156,163,175,0.5)", width=0),
                        fill="tonexty",
                        fillcolor="rgba(156,163,175,0.2)",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig_acf.add_trace(
                    go.Bar(
                        x=lags, y=acf_data["pacf"], name="PACF", marker_color="#10B981", width=0.1, showlegend=False
                    ),
                    row=2,
                    col=1,
                )
                fig_acf.add_trace(
                    go.Scatter(
                        x=lags,
                        y=acf_data["pacf"],
                        mode="markers",
                        marker=dict(color="#10B981", size=5),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )
                fig_acf.add_trace(
                    go.Scatter(
                        x=lags,
                        y=acf_data["pacf_conf_upper"],
                        mode="lines",
                        line=dict(color="rgba(156,163,175,0.5)", width=0),
                        fill="tonexty",
                        fillcolor="rgba(156,163,175,0.2)",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )
                fig_acf.add_trace(
                    go.Scatter(
                        x=lags,
                        y=acf_data["pacf_conf_lower"],
                        mode="lines",
                        line=dict(color="rgba(156,163,175,0.5)", width=0),
                        fill="tonexty",
                        fillcolor="rgba(156,163,175,0.2)",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )
                fig_acf.update_layout(
                    height=400,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _render_chart(fig_acf, filename="acf_pacf_eda")

    # ══════════════════════════════════════════════════════════════════
    # TAB 3: 🔄 §4.1.2 BẪY TỰ TƯƠNG QUAN & ĐỘ PHÂN TÁN THEO HORIZON
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 3. §4.1.2 Bẫy Tự Tương Quan & Độ Phân Tán Theo Horizon")
        st.markdown(
            "Để đánh giá quán tính khí quyển và mức độ ghi nhớ thông tin lịch sử của nồng độ PM2.5, "
            "hàm tự tương quan (ACF) và tự tương quan riêng phần (PACF) từ trễ 1 giờ đến 72 giờ được trực quan hóa trên **Hình 4.2a** "
            "và mức độ phân tán mở rộng theo horizon được thể hiện trên **Hình 4.2b**:"
        )

        c_ac1, c_ac2 = st.columns([1.2, 1])
        fig_4_2a = THESIS_FIG_DIR / "Hinh_4.2a_EDA_Autocorrelation_Memory.png"
        fig_4_2b = THESIS_FIG_DIR / "Hinh_4.2b_EDA_Horizon_Scatter_Dispersion.png"
        with c_ac1:
            if fig_4_2a.exists():
                st.image(
                    str(fig_4_2a),
                    caption="Hình 4.2a: Bẫy tự tương quan — Suy giảm tự tương quan ACF/PACF theo độ trễ",
                    use_container_width=True,
                )
        with c_ac2:
            if fig_4_2b.exists():
                st.image(
                    str(fig_4_2b),
                    caption="Hình 4.2b: Độ phân tán dữ liệu thực tế theo từng mốc thời gian dự báo (h=1 vs h=24)",
                    use_container_width=True,
                )

        _insight_card(
            "🔄 Biện Luận Bẫy Tự Tương Quan (Autocorrelation Trap)",
            "Đồ thị Hình 4.2a chỉ ra rằng hệ số tự tương quan ACF tại trễ 1 giờ đạt mức rất cao (r ≈ 0,86 trên chuỗi giờ và ~0,97 trên chuỗi quan trắc gốc) "
            "và suy giảm tiệm cận hàm mũ khi độ trễ tăng lên 6 giờ và 24 giờ. "
            "Điều này lý giải tại sao mô hình quán tính Persistence đạt hiệu năng vượt trội ở mốc 1 giờ nhưng suy giảm nghiêm trọng khi mở rộng chân trời dự báo. "
            "Sự gia tăng độ bất định này được minh họa qua Hình 4.2b: các điểm phân tán mở rộng hình quạt khi tăng khoảng cách dự báo từ 1h lên 24h, "
            "phản ánh sự tích tụ sai số phi tuyến tính của động lực học khí quyển đô thị.",
        )

        # Plotly Interactive Dispersion Tool
        with st.expander("🔍 Xem biểu đồ phân tán tương tác (Interactive Plotly Scatter)", expanded=False):
            try:
                df_clean = pd.read_csv(
                    PROJECT_ROOT / "dataset" / "interim" / "cleaned_hourly.csv", usecols=["pm25"]
                ).dropna()
                df_disp = df_clean.copy()
                df_disp["h1"] = df_disp["pm25"].shift(-1)
                df_disp["h24"] = df_disp["pm25"].shift(-24)
                df_disp = df_disp.dropna().sample(min(2000, len(df_disp)))

                fig_disp = make_subplots(
                    rows=1, cols=2, subplot_titles=("h=1 (Tương quan chặt chẽ)", "h=24 (Phân tán mở rộng)")
                )
                fig_disp.add_trace(
                    go.Scatter(
                        x=df_disp["pm25"],
                        y=df_disp["h1"],
                        mode="markers",
                        marker=dict(size=3, color="#3B82F6", opacity=0.5),
                    ),
                    row=1,
                    col=1,
                )
                fig_disp.add_trace(
                    go.Scatter(
                        x=df_disp["pm25"],
                        y=df_disp["h24"],
                        mode="markers",
                        marker=dict(size=3, color="#F59E0B", opacity=0.5),
                    ),
                    row=1,
                    col=2,
                )
                max_val = df_disp["pm25"].max()
                fig_disp.add_trace(
                    go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(color="gray", dash="dash")),
                    row=1,
                    col=1,
                )
                fig_disp.add_trace(
                    go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(color="gray", dash="dash")),
                    row=1,
                    col=2,
                )
                fig_disp.update_layout(
                    height=280,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _render_chart(fig_disp, filename="scatter_dispersion_eda")
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════
    # TAB 4: ⚡ §4.1.3 PHÂN PHỐI ĐUÔI DÀI & ĐỈNH DỊ THƯỜNG
    # ══════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 4. §4.1.3 Phân Phối Nồng Độ PM2.5 Đuôi Dài & Đỉnh Ô Nhiễm Dị Thường")
        st.markdown(
            "Đặc tính phân phối xác suất và độ lệch chuẩn của chuỗi dữ liệu nồng độ bụi mịn PM2.5 được kiểm chứng qua **Hình 4.3a** (KDE & Q-Q Plot) "
            "và diễn biến các đợt bùng phát ô nhiễm cục bộ trong thực tế được minh họa qua **Hình 4.3b**:"
        )

        c_sp1, c_sp2 = st.columns([1, 1.4])
        fig_4_3a = THESIS_FIG_DIR / "Hinh_4.3a_EDA_PM25_Fat_Tailed_Distribution.png"
        fig_4_3b = THESIS_FIG_DIR / "Hinh_4.3b_EDA_PM25_Erratic_Spikes.png"
        with c_sp1:
            if fig_4_3a.exists():
                st.image(
                    str(fig_4_3a),
                    caption="Hình 4.3a: Phân phối nồng độ PM2.5 dạng đuôi dài (Fat-Tailed Distribution)",
                    use_container_width=True,
                )
        with c_sp2:
            if fig_4_3b.exists():
                st.image(
                    str(fig_4_3b),
                    caption="Hình 4.3b: Hiện tượng các đỉnh ô nhiễm dị thường đột ngột (Erratic Spikes - Tháng 12/2022)",
                    use_container_width=True,
                )

        _insight_card(
            "⚡ Biện Luận Phân Phối Đuôi Dài & Nguy Cơ Y Tế Cực Đoan",
            "Kết quả Hình 4.3a khẳng định phân phối PM2.5 tại Sa Đéc có độ lệch phải rõ rệt (hệ số bất đối xứng Skewness đạt 2,0046 > 1,5, Kurtosis = 6,1458), "
            "tập trung phần lớn quan sát ở ngưỡng 10–15 μg/m³ nhưng xuất hiện dải đuôi dài kéo dài sang phải. "
            "Đồ thị Q-Q Plot xác nhận chuỗi dữ liệu vi phạm nghiêm trọng giả định phân phối chuẩn do định kỳ chịu tác động từ các đợt bùng phát ô nhiễm cục bộ.<br>"
            "Đặc biệt, các đỉnh nồng độ vượt 60 μg/m³ nằm ở vùng đuôi dài chứa nguy cơ y tế nghiêm trọng nhưng thường bị các hàm mất mát tiêu chuẩn (MSE, MAE) coi là nhiễu ngoại lai, "
            "dẫn đến hiện tượng dự báo thấp hơn thực tế (under-estimation). "
            "Trên Hình 4.3b, dữ liệu tháng 12/2022 ghi nhận đỉnh cực đoan 103,5 μg/m³ lúc 19h00 ngày 30/12/2022 — vượt gần 7 lần ngưỡng Good theo chuẩn AQI quốc tế. "
            "Trong 4 ngày cuối tháng có tới 16 lần nồng độ vượt 50 μg/m³, liên quan đến đốt rơm rạ sau vụ mùa Đông-Xuân kết hợp nghịch nhiệt ban đêm.",
        )

        _insight_card(
            "⚠️ Bẫy Loại Bỏ Ngoại Lai (Outlier Removal Trap)",
            "<b>Cảnh báo phương pháp luận:</b> Áp dụng các bộ lọc thống kê tiêu chuẩn (như IQR 1.5 hay Z-Score) để cắt bỏ các giá trị lớn hơn ngưỡng sẽ xóa nhầm chính các đỉnh ô nhiễm cực đoan thật sự. "
            "Hành động này tạo ra hội chứng 'Độ chính xác giả mạo' (False Sense of Accuracy): mô hình có MAE rất thấp nhưng hoàn toàn bị 'mù' trước các thảm họa môi trường. "
            "<b>Giải pháp Luận văn:</b> Bắt buộc sử dụng <b>Domain Bounds (0 – 500 µg/m³)</b> theo giới hạn vật lý và quy chuẩn kỹ thuật quốc gia, bảo toàn nguyên vẹn 100% các đỉnh bùng phát thực tế.",
            card_type="warning",
        )

        # Plotly Q-Q Plot Interactive
        qq_cache = RESEARCH_DIR / "eda" / "plotly_cache" / "qq_plot.json"
        if qq_cache.exists():
            with st.expander("🔍 Xem chi tiết kiểm định phân phối chuẩn Q-Q Plot (Interactive)", expanded=False):
                with open(qq_cache) as f:
                    qq_data = json.load(f)
                fig_qq = make_subplots(
                    rows=1, cols=2, subplot_titles=("Raw PM2.5 (Lệch phải nặng)", "Log-Transformed PM2.5 (Chuẩn hóa)")
                )
                d_raw = qq_data["raw"]
                fig_qq.add_trace(
                    go.Scatter(
                        x=d_raw["theo"], y=d_raw["sample"], mode="markers", marker=dict(color="#3B82F6", size=4)
                    ),
                    row=1,
                    col=1,
                )
                min_th, max_th = min(d_raw["theo"]), max(d_raw["theo"])
                fig_qq.add_trace(
                    go.Scatter(
                        x=[min_th, max_th],
                        y=[min(d_raw["sample"]), max(d_raw["sample"])],
                        mode="lines",
                        line=dict(color="#EF4444", dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
                d_log = qq_data["log"]
                fig_qq.add_trace(
                    go.Scatter(
                        x=d_log["theo"], y=d_log["sample"], mode="markers", marker=dict(color="#10B981", size=4)
                    ),
                    row=1,
                    col=2,
                )
                fig_qq.add_trace(
                    go.Scatter(
                        x=[min_th, max_th],
                        y=[min(d_log["sample"]), max(d_log["sample"])],
                        mode="lines",
                        line=dict(color="#EF4444", dash="dash"),
                    ),
                    row=1,
                    col=2,
                )
                fig_qq.update_layout(
                    height=320,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _render_chart(fig_qq, filename="qq_plot_eda")

    # ══════════════════════════════════════════════════════════════════
    # TAB 5: 🔀 §4.1.4 TRÔI DẠT KHÁI NIỆM & MẬT ĐỘ ĐA BIẾN
    # ══════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("### 5. §4.1.4 Tương Quan Động Trượt (Concept Drift) & Phân Bố Mật Độ Đa Biến")
        st.markdown(
            "Sự biến thiên động lực học của mối tương quan giữa nồng độ PM2.5 và các yếu tố khí tượng (Nhiệt độ môi trường, Độ ẩm không khí) "
            "theo các chu kỳ mùa trong năm được định lượng qua hệ số tương quan Spearman động trượt trên **Hình 4.4a** "
            "và cấu trúc mật độ phân bố đồng thời giữa Nhiệt độ - Độ ẩm - PM2.5 được thể hiện qua **Hình 4.4b**:"
        )

        c_dr1, c_dr2 = st.columns(2)
        fig_4_4a = THESIS_FIG_DIR / "Hinh_4.4a_EDA_Rolling_Correlation.png"
        fig_4_4b = THESIS_FIG_DIR / "Hinh_4.4b_EDA_Hexbin_Multivariate.png"
        with c_dr1:
            if fig_4_4a.exists():
                st.image(
                    str(fig_4_4a),
                    caption="Hình 4.4a: Tương quan động trượt giữa PM2.5 và các biến khí tượng (Concept Drift)",
                    use_container_width=True,
                )
        with c_dr2:
            if fig_4_4b.exists():
                st.image(
                    str(fig_4_4b),
                    caption="Hình 4.4b: Bản đồ mật độ Hexbin đa biến giữa Nhiệt độ - Độ ẩm - PM2.5",
                    use_container_width=True,
                )

        _insight_card(
            "🔀 Biện Luận Trôi Dạt Khái Niệm Theo Mùa (Seasonal Concept Drift)",
            "Hình 4.4a minh họa hiện tượng trôi dạt khái niệm có tính chu kỳ theo mùa thông qua sự phân hóa đối xứng giữa hai yếu tố vi khí tượng chủ đạo:<br>"
            "• <b>Nhiệt độ (đường cam):</b> Tương quan duy trì ở miền âm rất sâu vào cao điểm mùa khô (hệ số r chạm đáy xấp xỉ -0,64 trong giai đoạn tháng 3 – 5/2023). Bức xạ nhiệt mạnh ban ngày kích hoạt đối lưu không khí, thúc đẩy phát tán và làm loãng nồng độ bụi. Khi vào mùa mưa, tương quan suy yếu rõ rệt (r từ -0,15 đến 0) do hiệu ứng rửa trôi khí quyển từ nước mưa lấn át ảnh hưởng của nhiệt độ.<br>"
            "• <b>Độ ẩm (đường xanh):</b> Thể hiện quan hệ đồng biến mạnh mẽ và đạt đỉnh từ +0,45 đến +0,65 vào các tháng chuyển mùa và mùa mưa do sol khí hút ẩm kết hợp nghịch nhiệt ban đêm giữ bụi mịn lơ lửng sát mặt đất.<br>"
            "• <b>Bản đồ Hexbin đa biến (Hình 4.4b):</b> Khẳng định sự phi tuyến tính sâu sắc: vùng mật độ tập trung cao nhất ở dải nhiệt độ 26–28,5°C và độ ẩm 80–90%. Các trường hợp nồng độ bụi tăng vọt (>50 μg/m³) phân bố tập trung tại giao điểm nhiệt độ thấp trong ngày (25–28°C) và độ ẩm rất cao (>85%), tương thích chặt chẽ với hiện tượng nghịch nhiệt bức xạ sáng sớm.",
        )

        # Diurnal and Seasonality Plots
        st.markdown("---")
        st.markdown("#### 🌅 Chu Kỳ Ngày Đêm (Diurnal) & Phân Rã Thành Phần STL")
        c_diu, c_stl = st.columns([1, 1.2])
        with c_diu:
            try:
                df_clean = pd.read_csv(
                    PROJECT_ROOT / "dataset" / "interim" / "cleaned_hourly.csv",
                    usecols=["ngay_tao", "pm25"],
                    parse_dates=["ngay_tao"],
                )
                df_clean["hour"] = df_clean["ngay_tao"].dt.hour
                hourly_mean = df_clean.groupby("hour")["pm25"].mean().reset_index()
                fig_diurnal = px.line(
                    hourly_mean, x="hour", y="pm25", markers=True, color_discrete_sequence=["#F59E0B"]
                )
                fig_diurnal.update_layout(
                    height=260,
                    margin=dict(l=30, r=20, t=20, b=30),
                    xaxis_title="Giờ trong ngày",
                    yaxis_title="PM2.5 trung bình (µg/m³)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _render_chart(fig_diurnal, filename="diurnal_cycle_eda")
                _caption("Chu kỳ nồng độ trung bình theo giờ trong ngày (Diurnal)")
            except Exception:
                pass

        with c_stl:
            stl_data = eda_data.get("stl", {})
            if stl_data:
                stl_cols = st.columns(3)
                stl_cols[0].metric("Trend Strength", f"{stl_data.get('trend_strength', 0):.3f}")
                stl_cols[1].metric("Seasonal Strength", f"{stl_data.get('seasonal_strength', 0):.3f}")
                stl_cols[2].metric(
                    "Residual σ", f"{stl_data.get('residual_std', 0):.2f} µg/m³", help="Sàn sai số hiệu năng"
                )

            stl_cache = RESEARCH_DIR / "eda" / "plotly_cache" / "stl.json"
            if stl_cache.exists():
                with open(stl_cache) as f:
                    stl_data_cache = json.load(f)
                df_stl = pd.DataFrame(stl_data_cache)
                df_stl["index"] = pd.to_datetime(df_stl["index"], format="%Y-%m-%d %H")
                df_stl.set_index("index", inplace=True)
                monthly_counts = df_stl.resample("ME").count()
                best_month = monthly_counts["original"].idxmax()
                start = best_month.replace(day=1, hour=0, minute=0, second=0)
                end = best_month.replace(
                    day=calendar.monthrange(best_month.year, best_month.month)[1], hour=23, minute=0, second=0
                )
                full_index = pd.date_range(start, end, freq="h")
                df_stl = df_stl.reindex(full_index)

                fig_stl = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
                fig_stl.add_trace(
                    go.Scatter(x=df_stl.index, y=df_stl["trend"], name="Trend", line=dict(color="#F59E0B", width=2)),
                    row=1,
                    col=1,
                )
                fig_stl.add_trace(
                    go.Scatter(
                        x=df_stl.index, y=df_stl["seasonal"], name="Seasonal", line=dict(color="#3B82F6", width=1)
                    ),
                    row=2,
                    col=1,
                )
                fig_stl.add_trace(
                    go.Scatter(
                        x=df_stl.index,
                        y=df_stl["resid"],
                        name="Residual",
                        mode="markers",
                        marker=dict(color="#EF4444", size=3),
                    ),
                    row=3,
                    col=1,
                )
                fig_stl.update_layout(
                    height=260,
                    margin=dict(l=30, r=20, t=10, b=20),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _render_chart(fig_stl, filename="stl_summary_eda")
                _caption(f"STL Decomposition mẫu ({start.strftime('%m/%Y')})")

    # ══════════════════════════════════════════════════════════════════
    # TAB 6: 🧱 §4.1.5 MÃ VẠCH DỮ LIỆU KHUYẾT & GIỚI HẠN NỘI SUY
    # ══════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown("### 6. §4.1.5 Mã Vạch Mất Mát Dữ Liệu & Giới Hạn Phục Hồi Của Các Thuật Toán Nội Suy")
        st.markdown(
            "Vị trí và độ dài của các khoảng trống mất tín hiệu đo lường trong toàn bộ chu kỳ quan trắc được trực quan hóa qua **Hình 4.5a** (Missing Barcode) "
            "và sai số phục hồi của các thuật toán nội suy được đánh giá thực nghiệm trên **Hình 4.5b**:"
        )

        c_gap1, c_gap2 = st.columns([1.6, 1])
        fig_4_5a = THESIS_FIG_DIR / "Hinh_4.5a_EDA_Missing_Data_Barcode.png"
        fig_4_5b = THESIS_FIG_DIR / "Hinh_4.5b_EDA_Recovery_Limits.png"
        with c_gap1:
            if fig_4_5a.exists():
                st.image(
                    str(fig_4_5a),
                    caption="Hình 4.5a: Mã vạch phân bố khoảng trống mất mát dữ liệu (Missing Data Barcode)",
                    use_container_width=True,
                )
        with c_gap2:
            if fig_4_5b.exists():
                st.image(
                    str(fig_4_5b),
                    caption="Hình 4.5b: Đánh giá sai số và giới hạn phục hồi dữ liệu của các thuật toán nội suy",
                    use_container_width=True,
                )

        _insight_card(
            "🧱 Biện Luận Chiến Lược Nội Suy Phân Tầng (Tiered Imputation Strategy)",
            "Đồ thị Hình 4.5b chỉ rõ: <b>Cubic Spline</b> đạt sai số thấp nhất ở khoảng trống ≤6h; <b>KNN (K=5)</b> đạt sai số tối ưu ở khoảng 6–24h; "
            "trong khi việc cố gắng nội suy khoảng trống >24h khiến sai số RMSE bùng nổ vượt 12 μg/m³.<br>"
            "Kết quả này khẳng định cơ sở thực nghiệm của quy tắc <b>cắt bỏ (Drop)</b> khoảng trống dài: "
            "các khoảng trống vượt 24 giờ phá vỡ tính liên tục của chu kỳ mùa, do đó bắt buộc phải loại bỏ thay vì nội suy "
            "nhằm tránh tạo ra dữ liệu giả (data leakage) gây sai lệch kết quả huấn luyện mô hình. "
            "Chiến lược phân tầng đã phục hồi 656 giờ (3,2% tổng missing) và cắt bỏ 19.810 giờ (96,8%), "
            "bảo toàn 100% dữ liệu thực cho tập kiểm thử mỏ neo (Anchor Test Set 1.200 giờ cuối).",
        )

        # Gap Length Distribution & Missing Barcode Interactive
        st.markdown("---")
        st.markdown("#### 📊 Phân Bổ Độ Dài Khoảng Trống & Ma Trận Mù (Synthetic Year)")
        gap_json_path = RESEARCH_DIR / "eda" / "gap_analysis_report.json"
        if gap_json_path.exists():
            with open(gap_json_path, encoding="utf-8") as _f:
                gap_data = json.load(_f)
            gap_dist = gap_data.get("gap_distribution", {})
            if gap_dist:
                c_g1, c_g2 = st.columns([1.2, 1])
                with c_g1:
                    gap_labels = list(gap_dist.keys())
                    gap_counts = [v["count"] for v in gap_dist.values()]
                    gap_hours = [v["hours_recoverable"] for v in gap_dist.values()]
                    gap_pcts = [v["pct_of_missing"] for v in gap_dist.values()]
                    bar_colors = ["#00D4AA", "#00D4AA", "#F59E0B", "#F59E0B", "#EF4444", "#EF4444"]
                    fig_gap = _chart(
                        xaxis_title="Khoảng gap",
                        yaxis_title="Số lượng gaps",
                        height=320,
                        showlegend=False,
                        margin=dict(l=40, r=20, t=20, b=20),
                    )
                    fig_gap.add_trace(
                        go.Bar(
                            x=gap_labels,
                            y=gap_counts,
                            marker_color=bar_colors[: len(gap_labels)],
                            text=[f"{c} gaps<br>{h}h ({p}%)" for c, h, p in zip(gap_counts, gap_hours, gap_pcts)],
                        )
                    )
                    add_simple_bar_labels(fig_gap, orientation="v", yshift=8)
                    _render_chart(fig_gap, filename="gap_dist_eda")
                    _caption(f"Phân bổ {gap_data.get('gap_count', 110)} gaps (tổng missing: 74,0%)")

                with c_g2:
                    st.markdown(
                        """
                    <div style="background: rgba(239,68,68,0.05); border: 1px solid rgba(239,68,68,0.2); border-radius: 8px; padding: 1rem; font-size: 0.9rem; line-height: 1.6;">
                        <b style="color: #EF4444;">🚨 Điểm Mù Dữ Liệu (Blind Spots):</b><br>
                        Khi gộp dữ liệu thành một năm tổng hợp điển hình, chuỗi quan trắc vẫn bị mù hoàn toàn <b>89 ngày</b> (~24,3%).<br>
                        Trong đó <b>Tháng 2 và Tháng 9</b> gần như mất tín hiệu hoàn toàn (tỉ lệ bao phủ ~13%).<br><br>
                        <b>Khuyến nghị:</b> Không ép mô hình dự báo chu kỳ năm dài hạn trên các tháng mù này, tập trung năng lực dự báo vào các tầm nhìn khả thi (1h - 24h).
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    # ══════════════════════════════════════════════════════════════════
    # TAB 7: 💡 §4.1.6 CƠ SỞ THIẾT KẾ PIPELINE & DEEP INSIGHTS
    # ══════════════════════════════════════════════════════════════════
    with tab7:
        st.markdown("### 7. 💡 Cơ Sở Phương Pháp Luận Thiết Kế Pipeline & Phân Tích Chuyên Sâu")
        st.markdown(
            "Những phát hiện thực nghiệm qua 5 tiểu mục EDA giải thích trực tiếp 6 nguyên lý nền tảng "
            "trong kiến trúc hệ thống ML Data Engineering & Pipeline v9:"
        )

        st.info(
            "**1. Xử lý Gaps (Chiến lược Tiered Imputation):** Missing data chiếm 74,0% (110 gaps) do sấm sét và gián đoạn nguồn. "
            "Áp dụng phân tầng: Gaps ngắn ≤ 24h (656 giờ, 3,2%) khôi phục bằng PCHIP/Spline/KNN để giữ chu kỳ ngày; "
            "Gaps dài > 24h (19.810 giờ, 96,8%) bắt buộc cắt bỏ thành các chuỗi liên tục độc lập, "
            "bảo toàn 100% dữ liệu thực cho Anchor Test Set (1.200 giờ cuối) không bị rò rỉ dữ liệu (Anti-Leakage)."
        )
        st.info(
            "**2. Xử lý Spikes (Phân phối Fat-Tailed, Skewness = 2,0046):** PM2.5 có các đỉnh đột biến (Max 138,5 µg/m³) tàn phá hàm mất mát MSE. "
            "Dự án áp dụng mô hình phi tuyến (LightGBM, XGBoost, GRU) kết hợp khoảng dự báo Conformalized Quantile Regression (CQR) "
            "và Adaptive Conformal Inference (ACI) để kiểm soát rủi ro cực đoan."
        )
        st.info(
            "**3. Nắm bắt Mùa Vụ & Nghịch Nhiệt:** Chu kỳ sớm 6h–8h sáng (đối lưu nghịch nhiệt) và tương quan âm với Nhiệt độ (r = -0,48), "
            "Độ ẩm (r = -0,26) được mã hóa qua 119 features (Fourier, Cyclic Time-of-Day, Rolling stats)."
        )
        st.info(
            "**4. Thước đo MASE & Sức mạnh Baseline:** Do tự tương quan lag-1 rất cao (r ≈ 0,86 ở chuỗi giờ 1h và r ≈ 0,97 ở chuỗi 15m), "
            "MASE là metric chuẩn mực (Hyndman & Koehler, 2006). Trên chuỗi 1h, các mô hình tham số đạt MASE = 1,158 ~ 1,236 (> 1,0) "
            "là hiện tượng kinh tế lượng bình thường do bẫy tự tương quan."
        )
        st.info(
            "**5. Sàn Sai Số STL Residual:** Phân tích STL cho thấy phương sai dư σ ≈ 5,2 µg/m³ — "
            "mô hình đạt MAE gần ngưỡng này phản ánh đã khai thác tối đa tín hiệu có thể dự đoán."
        )
        st.info(
            "**6. Đột phá Đa Phân Giải (Multi-Resolution - v9):** Bằng cách khai thác dữ liệu tần số cao 15m, "
            "mô hình học sâu GRU 15m thu nhận được vi biến động nội giờ và chính thức đạt MASE = 0,667 (< 1,0), "
            "đánh bại hoàn toàn Persistence baseline và phá vỡ bẫy tự tương quan."
        )

        st.markdown("---")
        st.markdown("#### 🔬 Phân Tích Chuyên Sâu: Granger Causality & Đa Độ Phân Giải (15m vs 30m vs 1h)")

        # Deep Insights & Granger Causality
        di_path = RESEARCH_DIR / "eda" / "deep_insights_results.json"
        di_data = {}
        if di_path.exists():
            with open(di_path, encoding="utf-8") as f:
                di_data = json.load(f)

        gc = di_data.get("granger_causality", {})
        if gc:
            st.markdown(f"**Kiểm định Nhân quả Granger (Granger Causality Test - α=0.05)** {cite('peixeiro2022')}:")
            gc_rows = []
            for col_name, gr in gc.items():
                if "error" not in gr:
                    gc_rows.append(
                        {
                            "Biến Ngoại Sinh": gr.get("label", col_name),
                            "Độ Trễ Tối Ưu": f"{gr.get('best_lag', '?')}h",
                            "p-value": f"{gr.get('best_p_value', 0):.2e}"
                            if gr.get("best_p_value", 0) > 0
                            else "< 1e-10",
                            "Có Ý Nghĩa Thống Kê (α=0.05)": "✅ Yes" if gr.get("significant_at_005") else "❌ No",
                        }
                    )
            if gc_rows:
                st.dataframe(pd.DataFrame(gc_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "Kết luận: Tất cả các biến môi trường phụ (Nhiệt độ, Độ ẩm, CO₂) đều Granger-cause PM2.5 (p < 0.05), chứng minh tính đúng đắn khi đưa vào bộ đặc trưng."
                )

    # ── References ──
    render_references_section()
