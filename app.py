"""Streamlit Dashboard — PM2.5 Forecasting Results.

Design Direction: "Scientific Observatory" — clean dark theme with
precise data presentation, inspired by environmental monitoring stations.

Key Design Decisions:
- Dark theme with teal accent (#00D4AA) for ecological feel
- Gradient KPI cards with subtle glassmorphism
- Data storytelling flow: Hook → Context → Insight → Action
- Interactive Plotly charts with consistent color palette

Usage:
    uv run streamlit run app.py
"""

from __future__ import annotations

import os

# Prevent OpenMP multi-runtime crash (PyTorch vs Homebrew LightGBM on macOS)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
from typing import Any

import streamlit as st
from src.frontend.components import (
    CHART_COLORS,
    COLORS,
    EDA_DIR,
    FIGURES_DIR,
    PI_DIR,
    PROJECT_ROOT,
    RESEARCH_DIR,
    SHAP_DIR,
    _count_tests,
    _get_pipeline_metrics,
    insight_card,
    kpi_card,
    load_experiment_results,
    load_json,
    section_header,
)
from src.info_cards import version_selector_sidebar

# Re-export shared components and helpers for backward compatibility
__all__ = [
    "CHART_COLORS",
    "COLORS",
    "EDA_DIR",
    "FIGURES_DIR",
    "PI_DIR",
    "PROJECT_ROOT",
    "RESEARCH_DIR",
    "SHAP_DIR",
    "_count_tests",
    "_get_pipeline_metrics",
    "insight_card",
    "kpi_card",
    "load_experiment_results",
    "load_json",
    "main",
    "section_header",
    "sidebar",
]

# ── Page config ──
st.set_page_config(
    page_title="PM2.5 Forecasting — Scientific Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ── */
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; }
    h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
    h2 { font-weight: 600 !important; color: #00D4AA !important; }

    /* ── KPI Cards ── */
    .kpi-row { display: flex; gap: 1rem; margin: 1.5rem 0; }
    .kpi-card {
        flex: 1;
        background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
        border: 1px solid var(--faded-text-color, rgba(128,128,128,0.2));
        border-radius: 16px;
        padding: 1.5rem;
        position: relative;
        overflow: visible;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: var(--text-color);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,212,170,0.15);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00D4AA, #4ECDC4);
    }
    .kpi-label { font-size: 0.75rem; color: var(--text-color); opacity: 0.7; font-weight: 600; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: var(--text-color); font-family: 'JetBrains Mono', monospace; letter-spacing: -0.04em; }
    .kpi-delta { font-size: 0.9rem; margin-top: 0.3rem; }
    .kpi-delta.positive { color: #00D4AA; }
    .kpi-delta.negative { color: #FF6B6B; }

    /* ── Section Headers ── */
    .section-header {
        display: flex; align-items: center; gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0,212,170,0.3);
    }
    .section-header .icon { font-size: 1.5rem; }
    .section-header .title { font-size: 1.25rem; font-weight: 600; color: #00D4AA; }

    /* ── Insight Cards ── */
    .insight-card {
        background: linear-gradient(135deg, rgba(0,212,170,0.08) 0%, rgba(78,205,196,0.04) 100%);
        border-left: 4px solid #00D4AA;
        border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    }
    .insight-card.warning {
        background: linear-gradient(135deg, rgba(255,107,107,0.08) 0%, rgba(255,230,109,0.04) 100%);
        border-left-color: #FF6B6B;
    }
    .insight-card h4 { margin: 0 0 0.5rem 0; font-weight: 600; }
    .insight-card .insight-text { margin: 0; opacity: 0.85; line-height: 1.6; }

    /* ── Data Table Styling ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Sidebar Navigation ── */
    [data-testid="stSidebar"] hr { border-color: rgba(0,212,170,0.2); }

    /* Hide native radio circles */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* Style radio items as navigation links */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        padding: 0.4rem 1rem !important;
        margin-bottom: 0.1rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        background-color: transparent !important;
    }

    /* Make text block robust */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        color: var(--text-color) !important;
        opacity: 0.85 !important;
    }

    /* Hover effect */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: rgba(0, 212, 170, 0.08) !important;
        transform: translateX(4px) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover p {
        color: #00D4AA !important;
        opacity: 1 !important;
    }

    /* Selected item style */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] {
        background: linear-gradient(90deg, rgba(0,212,170,0.15) 0%, transparent 100%) !important;
        border-left: 4px solid #00D4AA !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p {
        font-weight: 700 !important;
        color: #00D4AA !important;
    }

    /* ── Sidebar Section Headers ── */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(1),
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(4),
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(7),
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(14),
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(16) {
        margin-top: 2.8rem !important;
        position: relative !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *::before {
        position: absolute;
        top: -2.0rem;
        left: 0.2rem;
        font-size: 1.15rem;
        font-weight: 900;
        color: var(--text-color);
        letter-spacing: 0.02em;
        text-transform: uppercase;
        white-space: nowrap;
        pointer-events: none;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(1)::before { content: "PHASE 1: GIỚI THIỆU & KHÁM PHÁ"; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(4)::before { content: "PHASE 2: HUẤN LUYỆN MÔ HÌNH"; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(7)::before { content: "PHASE 3: ĐÁNH GIÁ & GIẢI THÍCH"; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(14)::before { content: "PHASE 4: ỨNG DỤNG & KẾT LUẬN"; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(16)::before { content: "CÔNG CỤ HỖ TRỢ"; }

    /* ── Pipeline diagram ── */
    .pipeline-box {
        background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
        border: 1px solid rgba(0,212,170,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.8;
        color: var(--text-color);
    }
    .pipeline-box .highlight { color: #00D4AA; font-weight: 600; }
    .pipeline-box .warn { color: #FFE66D; }
    .pipeline-box .accent { color: #FF6B6B; }

    /* ── Metric emphasis ── */
    .metric-highlight {
        display: inline-block;
        background: rgba(0,212,170,0.15);
        color: #00D4AA;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* ── Fix Fullscreen Button Overlap ── */
    [data-testid="StyledFullScreenButton"] {
        z-index: 9999 !important;
        pointer-events: auto !important;
    }
    button[title="View fullscreen"] {
        z-index: 9999 !important;
        pointer-events: auto !important;
    }

    /* ── Hide Streamlit branding ── */
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


def sidebar() -> str:
    """Render scientific dashboard sidebar navigation."""
    logo_path = PROJECT_ROOT / "assets" / "ctu_logo.svg"
    logo_html = '<div style="font-size: 2.5rem;">🌫️</div>'
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/svg+xml;base64,{b64}" width="100" style="margin-bottom: 0.5rem;" />'

    st.sidebar.markdown(
        f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="display: flex; justify-content: center; align-items: center;">
            {logo_html}
        </div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #00D4AA; margin-top: 0.5rem; line-height: 1.35;">
            Ứng Dụng Business Intelligence Để Phân Tích Dữ Liệu Môi Trường Cho Một Huyện
        </div>
        <div style="font-size: 0.8rem; color: var(--text-color); opacity: 0.85; margin-top: 0.4rem; line-height: 1.45;">
            <b>HV:</b> Hoàng Xuân Trí (M2522016)<br>
            <b>CBHD:</b> TS. Nguyễn Minh Khiêm
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    st.sidebar.markdown(
        """
    <div style="font-size: 0.85rem; color: #00D4AA; text-transform: uppercase;
                letter-spacing: 0.12em; margin: 0.5rem 0 0.5rem 0.2rem; font-weight: 800;">
        📌 Quy trình nghiên cứu
    </div>
    """,
        unsafe_allow_html=True,
    )

    page: str = st.sidebar.radio(
        "",
        [
            # ── Phase 1: Giới thiệu & Khám phá ──
            "🏠 Tổng Quan",
            "📜 Quy Trình Pipeline",
            "📊 Khám Phá & Dẫn Luận Dữ Liệu (EDA)",
            # ── Phase 2: Huấn luyện mô hình ──
            "⚙️ Cấu Hình Siêu Tham Số (Hyperparameters)",
            "🏋️ Huấn Luyện Mô Hình",
            "📋 Nhật Ký Thực Nghiệm (Experiment Runs)",
            # ── Phase 3: Đánh giá & Giải thích ──
            "📈 Kết Quả Đa Tầm Dự Báo (Multi-Horizon)",
            "📉 Thực Tế vs Dự Báo (Actual vs Predicted)",
            "🧠 Khả Năng Giải Thích Mô Hình (XAI)",
            "📊 Khoảng Dự Báo Bất Định (Prediction Intervals)",
            "📚 Đối Chiếu Khoa Học (Scientific Benchmark)",
            "📊 Hệ Thống Biểu Đồ Đề Án (Thesis Figures)",
            "🔬 Kiểm Định Khoa Học (Scientific Audit)",
            # ── Phase 4: Ứng dụng & Kết luận ──
            "🔮 Dự Báo PM2.5",
            "📝 Kết Luận & Hướng Phát Triển",
            # ── Công cụ hỗ trợ ──
            "💬 Trợ Lý AI",
            "✏️ Quản Lý Nội Dung",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    version_selector_sidebar()

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
    <div style="background: var(--secondary-background-color); color: var(--text-color) !important; border-radius: 12px; padding: 1rem; border: 1px solid rgba(0,212,170,0.15);">
        <div style="font-size: 0.75rem; color: var(--text-color); opacity: 0.6; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">Thông Số Dữ Liệu Quan Trắc</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.85rem;">
            <div style="color: var(--text-color);">📅 <span style="color: var(--text-color); opacity: 0.7;">Data</span></div><div style="color:#00D4AA">38 tháng</div>
            <div style="color: var(--text-color);">📦 <span style="color: var(--text-color); opacity: 0.7;">Records</span></div><div style="color:#00D4AA">209.594</div>
            <div style="color: var(--text-color);">🎯 <span style="color: var(--text-color); opacity: 0.7;">Target</span></div><div style="color:#00D4AA">PM2.5</div>
            <div style="color: var(--text-color);">🧪 <span style="color: var(--text-color); opacity: 0.7;">Tests</span></div><div style="color:#00D4AA">{_count_tests()} ✅</div>
            <div style="color: var(--text-color);">📐 <span style="color: var(--text-color); opacity: 0.7;">Features</span></div><div style="color:#00D4AA">119</div>
            <div style="color: var(--text-color);">🚫 <span style="color: var(--text-color); opacity: 0.7;">Leakage</span></div><div style="color:#00D4AA">0</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.checkbox(
        "🖨️ Print Mode (B&W)",
        value=st.session_state.get("print_mode", False),
        help="Chuyển toàn bộ charts sang chế độ trắng đen, tối ưu cho in đề án",
        key="print_mode",
    )

    return page


def main() -> None:
    """Main routing controller for Streamlit dashboard."""
    results: dict[str, Any] = load_experiment_results()
    page = sidebar()

    if page == "🏠 Tổng Quan":
        from src.frontend.pages.overview import page_overview

        page_overview(results)
        return

    if page == "📜 Quy Trình Pipeline":
        from src.pipeline_walkthrough import page_pipeline_walkthrough

        page_pipeline_walkthrough(results)
        return

    if page in ("📊 Khám Phá & Dẫn Luận Dữ Liệu (EDA)", "📊 EDA & Khám Phá Dữ Liệu"):
        from src.eda_page import page_eda

        page_eda(results)
        return

    if page in ("⚙️ Cấu Hình Siêu Tham Số (Hyperparameters)", "⚙️ Cấu Hình & Hyperparameters"):
        from src.frontend.pages.hyperparams import page_hyperparams

        page_hyperparams(results)
        return

    if page == "🏋️ Huấn Luyện Mô Hình":
        from src.frontend.pages.training import page_training

        page_training(results)
        return

    if page in ("📋 Nhật Ký Thực Nghiệm (Experiment Runs)", "📋 Lịch Sử Thí Nghiệm"):
        from src.frontend.pages.experiment_runs import page_experiment_runs

        page_experiment_runs(results)
        return

    if page in ("📈 Kết Quả Đa Tầm Dự Báo (Multi-Horizon)", "📈 Kết Quả Multi-Horizon"):
        from src.frontend.pages.multi_horizon import page_multi_horizon

        page_multi_horizon(results)
        return

    if page in ("📉 Thực Tế vs Dự Báo (Actual vs Predicted)", "📉 Actual vs Predicted"):
        from src.frontend.pages.actual_vs_predicted import (
            page_actual_vs_predicted,
        )

        page_actual_vs_predicted(results)
        return

    if page in ("🧠 Khả Năng Giải Thích Mô Hình (XAI)", "🧠 Giải Thích Trực Quan"):
        from src.explainability_hub import page_explainability_hub

        page_explainability_hub(results)
        return

    if page in ("📊 Khoảng Dự Báo Bất Định (Prediction Intervals)", "📊 Khoảng Tin Cậy Dự Báo"):
        from src.frontend.pages.prediction_intervals import (
            page_prediction_intervals,
        )

        page_prediction_intervals(results)
        return

    if page in ("📚 Đối Chiếu Khoa Học (Scientific Benchmark)", "📚 Đối Chiếu Khoa Học"):
        from src.frontend.pages.benchmarks import page_scientific_benchmark

        page_scientific_benchmark(results)
        return

    if page in ("📊 Hệ Thống Biểu Đồ Đề Án (Thesis Figures)", "📊 Thesis Figures"):
        from src.thesis_figures import page_thesis_figures

        page_thesis_figures(results)
        return

    if page in ("🔬 Kiểm Định Khoa Học (Scientific Audit)", "🔬 Scientific Audit"):
        from src.frontend.pages.audit import page_scientific_audit

        page_scientific_audit(results)
        return

    if page == "🔮 Dự Báo PM2.5":
        from src.frontend.pages.forecast import page_forecast

        page_forecast(results)
        return

    if page == "📝 Kết Luận & Hướng Phát Triển":
        from src.conclusion_page import page_conclusion

        page_conclusion(results)
        return

    if page == "💬 Trợ Lý AI":
        from src.chatbot.chat_page import page_ai_assistant

        page_ai_assistant(results)
        return

    if page == "✏️ Quản Lý Nội Dung":
        from src.frontend.pages.content_manager import page_content_manager

        page_content_manager(results)
        return

    # Fallback
    from src.frontend.pages.overview import page_overview

    page_overview(results)


if __name__ == "__main__":
    main()
