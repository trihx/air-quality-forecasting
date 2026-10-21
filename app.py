"""Streamlit Dashboard — PM2.5 Multi-Horizon Forecasting System.

Design Direction: "Scientific Observatory" — adaptive Light & Dark theme
with WCAG AA/AAA contrast, unified semantic design tokens, and rigorous
data science presentation.

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
    page_title="PM2.5 Forecasting — Scientific Observatory",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Unified Adaptive Light/Dark Design System CSS ──
# Note: Dark theme baseline uses #0B1120 for deep cards while color-mix()
# automatically adapts all --ds-* tokens when Streamlit switches to Light Theme.
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Adaptive Design System Tokens (Auto Light/Dark via color-mix) ── */
    :root, .stApp {
        --ds-bg-page: var(--background-color, #0E1117);
        --ds-bg-card: color-mix(in srgb, var(--background-color, #0B1120) 78%, var(--secondary-background-color, #111827) 22%);
        --ds-bg-surface: color-mix(in srgb, var(--secondary-background-color, #111827) 88%, var(--background-color, #0E1117) 12%);
        --ds-bg-subtle: color-mix(in srgb, var(--secondary-background-color, #1E293B) 55%, var(--background-color, #0E1117) 45%);
        --ds-border: color-mix(in srgb, var(--text-color, #F8FAFC) 15%, var(--background-color, #0E1117) 85%);
        --ds-border-strong: color-mix(in srgb, var(--text-color, #F8FAFC) 26%, var(--background-color, #0E1117) 74%);
        --ds-text-primary: var(--text-color, #F8FAFC);
        --ds-text-secondary: color-mix(in srgb, var(--text-color, #CBD5E1) 86%, var(--background-color, #0E1117) 14%);
        --ds-text-muted: color-mix(in srgb, var(--text-color, #94A3B8) 68%, var(--background-color, #0E1117) 32%);

        /* Adaptive semantic accents: automatically darken on Light Mode & brighten on Dark Mode */
        --ds-teal: color-mix(in srgb, #0D9488 76%, var(--text-color, #00D4AA) 24%);
        --ds-teal-bright: #00D4AA;
        --ds-teal-bg: color-mix(in srgb, #0D9488 10%, var(--ds-bg-card) 90%);
        --ds-teal-border: color-mix(in srgb, #0D9488 38%, var(--ds-border) 62%);

        --ds-amber: color-mix(in srgb, #D97706 78%, var(--text-color, #F59E0B) 22%);
        --ds-amber-bg: color-mix(in srgb, #F59E0B 10%, var(--ds-bg-card) 90%);
        --ds-amber-border: color-mix(in srgb, #F59E0B 38%, var(--ds-border) 62%);

        --ds-rose: color-mix(in srgb, #E11D48 78%, var(--text-color, #F87171) 22%);
        --ds-rose-bg: color-mix(in srgb, #F43F5E 10%, var(--ds-bg-card) 90%);
        --ds-rose-border: color-mix(in srgb, #F43F5E 38%, var(--ds-border) 62%);

        --ds-blue: color-mix(in srgb, #2563EB 78%, var(--text-color, #60A5FA) 22%);
        --ds-blue-bg: color-mix(in srgb, #3B82F6 10%, var(--ds-bg-card) 90%);
        --ds-blue-border: color-mix(in srgb, #3B82F6 38%, var(--ds-border) 62%);

        --ds-purple: color-mix(in srgb, #7C3AED 78%, var(--text-color, #C084FC) 22%);
        --ds-purple-bg: color-mix(in srgb, #8B5CF6 10%, var(--ds-bg-card) 90%);
        --ds-purple-border: color-mix(in srgb, #8B5CF6 38%, var(--ds-border) 62%);

        --ds-shadow: 0 2px 12px rgba(15, 23, 42, 0.08);
    }

    /* ── Global Typography & Container ── */
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; }
    h1 { font-weight: 700 !important; letter-spacing: -0.02em; color: var(--ds-text-primary) !important; }
    h2 { font-weight: 600 !important; color: var(--ds-teal) !important; }
    h3 { font-weight: 600 !important; color: var(--ds-text-primary) !important; }

    /* ── Equal-Height Column Flex Alignment ── */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }

    /* ── KPI Cards ── */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin: 1.25rem 0;
        align-items: stretch;
        flex-wrap: wrap;
    }
    .kpi-card {
        flex: 1 1 200px;
        background: var(--ds-bg-card);
        border: 1px solid var(--ds-border);
        border-radius: 14px;
        padding: 1.35rem 1.4rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        color: var(--ds-text-primary);
        box-shadow: var(--ds-shadow);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 128px;
        box-sizing: border-box;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: var(--ds-teal-border);
        box-shadow: 0 8px 24px rgba(13, 148, 136, 0.14);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--ds-teal), #3B82F6);
    }
    .kpi-label {
        font-size: 0.74rem;
        color: var(--ds-text-muted) !important;
        font-weight: 700;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-value {
        font-size: clamp(1.35rem, 1.8vw, 1.9rem);
        font-weight: 700;
        color: var(--ds-text-primary) !important;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        line-height: 1.25;
    }
    .kpi-delta { font-size: 0.85rem; margin-top: 0.4rem; font-weight: 500; }
    .kpi-delta.positive { color: var(--ds-teal) !important; }
    .kpi-delta.negative { color: var(--ds-rose) !important; }

    /* ── Section Headers ── */
    .section-header {
        display: flex; align-items: center; gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--ds-teal-border);
    }
    .section-header .icon { font-size: 1.45rem; }
    .section-header .title {
        font-size: 1.22rem;
        font-weight: 700;
        color: var(--ds-teal) !important;
    }

    /* ── Insight Cards (WCAG AAA Contrast in both Light & Dark Mode) ── */
    /* Dark mode base token reference: #0B1120 */
    .insight-card {
        background: var(--ds-bg-card);
        border: 1px solid var(--ds-teal-border);
        border-left: 4px solid var(--ds-teal);
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.45rem;
        margin: 1rem 0;
        box-shadow: var(--ds-shadow);
    }
    .insight-card.warning {
        background: var(--ds-bg-card);
        border: 1px solid var(--ds-rose-border);
        border-left: 4px solid var(--ds-rose);
    }
    .insight-card.info {
        background: var(--ds-bg-card);
        border: 1px solid var(--ds-blue-border);
        border-left: 4px solid var(--ds-blue);
    }
    .insight-card.success {
        background: var(--ds-bg-card);
        border: 1px solid var(--ds-teal-border);
        border-left: 4px solid var(--ds-teal);
    }
    .insight-card h4 {
        margin: 0 0 0.5rem 0 !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        color: var(--ds-text-primary) !important;
    }
    .insight-card .insight-text,
    .insight-card .insight-text p,
    .insight-card .insight-text li,
    .insight-card .insight-text ul,
    .insight-card .insight-text ol,
    .insight-card .insight-text span,
    .insight-card .insight-text div {
        margin: 0;
        color: var(--ds-text-secondary) !important;
        opacity: 1 !important;
        line-height: 1.68;
    }
    .insight-card .insight-text li {
        margin-bottom: 0.28rem;
    }
    .insight-card .insight-text b,
    .insight-card .insight-text strong {
        color: var(--ds-text-primary) !important;
        font-weight: 600;
    }
    .insight-card .insight-text code {
        background: var(--ds-bg-surface) !important;
        color: var(--ds-teal) !important;
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
        border: 1px solid var(--ds-border);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84em;
    }

    /* ── Data Table Styling ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Sidebar Navigation ── */
    [data-testid="stSidebar"] hr { border-color: var(--ds-border); }

    /* Hide native radio circles */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* Style radio items as navigation links */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        padding: 0.4rem 0.9rem !important;
        margin-bottom: 0.1rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        color: var(--ds-text-secondary) !important;
        opacity: 0.92 !important;
    }

    /* Hover effect */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: var(--ds-teal-bg) !important;
        transform: translateX(3px) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover p {
        color: var(--ds-teal) !important;
        opacity: 1 !important;
    }

    /* Selected item style */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] {
        background: linear-gradient(90deg, var(--ds-teal-bg) 0%, transparent 100%) !important;
        border-left: 4px solid var(--ds-teal) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p {
        font-weight: 700 !important;
        color: var(--ds-teal) !important;
    }

    /* ── Sidebar Section Headers ── */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(1),
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(4),
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(7),
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(14),
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(16) {
        margin-top: 2.6rem !important;
        position: relative !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *::before {
        position: absolute;
        top: -1.85rem;
        left: 0.2rem;
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--ds-text-muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        white-space: nowrap;
        pointer-events: none;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(1)::before { content: "I. DỮ LIỆU & KIẾN TRÚC PIPELINE"; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(4)::before { content: "II. MÔ HÌNH HÓA & THỰC NGHIỆM"; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(7)::before { content: "III. ĐÁNH GIÁ, XAI & KIỂM ĐỊNH"; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(14)::before { content: "IV. DỰ BÁO & TỔNG KẾT HỆ THỐNG"; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(16)::before { content: "V. TRỢ LÝ AI & QUẢN TRỊ NỘI DUNG"; }

    /* ── Pipeline diagram ── */
    .pipeline-box {
        background: var(--ds-bg-card);
        border: 1px solid var(--ds-border);
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.8;
        color: var(--ds-text-primary);
        box-shadow: var(--ds-shadow);
    }
    .pipeline-box .highlight { color: var(--ds-teal); font-weight: 600; }
    .pipeline-box .warn { color: var(--ds-amber); }
    .pipeline-box .accent { color: var(--ds-rose); }

    /* ── Metric emphasis ── */
    .metric-highlight {
        display: inline-block;
        background: var(--ds-teal-bg);
        color: var(--ds-teal);
        border: 1px solid var(--ds-teal-border);
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


def _apply_theme_override(theme_choice: str) -> None:
    """Inject explicit Light or Dark CSS token overrides when selected by user."""
    choice = theme_choice.lower()
    if "sáng" in choice or "light" in choice:
        st.markdown(
            """
        <style>
            :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                --background-color: #FFFFFF !important;
                --secondary-background-color: #F1F5F9 !important;
                --text-color: #0F172A !important;
                --ds-bg-page: #FFFFFF !important;
                --ds-bg-card: #FFFFFF !important;
                --ds-bg-surface: #F8FAFC !important;
                --ds-bg-subtle: #F1F5F9 !important;
                --ds-border: #CBD5E1 !important;
                --ds-border-strong: #94A3B8 !important;
                --ds-text-primary: #0F172A !important;
                --ds-text-secondary: #334155 !important;
                --ds-text-muted: #475569 !important;
                --ds-teal: #0F766E !important;
                --ds-teal-bright: #0D9488 !important;
                --ds-teal-bg: #F0FDFA !important;
                --ds-teal-border: #5EEAD4 !important;
                --ds-amber: #B45309 !important;
                --ds-amber-bg: #FFFBEB !important;
                --ds-amber-border: #FCD34D !important;
                --ds-rose: #BE123C !important;
                --ds-rose-bg: #FFF1F2 !important;
                --ds-rose-border: #FDA4AF !important;
                --ds-blue: #1D4ED8 !important;
                --ds-blue-bg: #EFF6FF !important;
                --ds-blue-border: #93C5FD !important;
                --ds-purple: #6D28D9 !important;
                --ds-purple-bg: #F5F3FF !important;
                --ds-purple-border: #C4B5FD !important;
                --ds-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
                background-color: #FFFFFF !important;
                color: #0F172A !important;
            }
            [data-testid="stSidebar"] {
                background-color: #F8FAFC !important;
                border-right: 1px solid #E2E8F0 !important;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
    elif "tối" in choice or "dark" in choice:
        st.markdown(
            """
        <style>
            :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                --background-color: #0E1117 !important;
                --secondary-background-color: #161B26 !important;
                --text-color: #F8FAFC !important;
                --ds-bg-page: #0E1117 !important;
                --ds-bg-card: #0B1120 !important;
                --ds-bg-surface: #111827 !important;
                --ds-bg-subtle: #1E293B !important;
                --ds-border: #1E293B !important;
                --ds-border-strong: #334155 !important;
                --ds-text-primary: #F8FAFC !important;
                --ds-text-secondary: #CBD5E1 !important;
                --ds-text-muted: #94A3B8 !important;
                --ds-teal: #2DD4BF !important;
                --ds-teal-bright: #00D4AA !important;
                --ds-teal-bg: rgba(0, 212, 170, 0.08) !important;
                --ds-teal-border: rgba(0, 212, 170, 0.35) !important;
                --ds-amber: #FBBF24 !important;
                --ds-amber-bg: rgba(245, 158, 11, 0.10) !important;
                --ds-amber-border: rgba(245, 158, 11, 0.35) !important;
                --ds-rose: #FB7185 !important;
                --ds-rose-bg: rgba(244, 63, 94, 0.10) !important;
                --ds-rose-border: rgba(244, 63, 94, 0.35) !important;
                --ds-blue: #60A5FA !important;
                --ds-blue-bg: rgba(59, 130, 246, 0.10) !important;
                --ds-blue-border: rgba(59, 130, 246, 0.35) !important;
                --ds-purple: #C084FC !important;
                --ds-purple-bg: rgba(168, 85, 247, 0.10) !important;
                --ds-purple-border: rgba(168, 85, 247, 0.35) !important;
                --ds-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
                background-color: #0E1117 !important;
                color: #F8FAFC !important;
            }
            [data-testid="stSidebar"] {
                background-color: #111827 !important;
                border-right: 1px solid #1E293B !important;
            }
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
            logo_html = f'<img src="data:image/svg+xml;base64,{b64}" width="92" style="margin-bottom: 0.4rem;" />'

    st.sidebar.markdown(
        f"""
    <div style="text-align:center; padding: 0.75rem 0 0.5rem 0;">
        <div style="display: flex; justify-content: center; align-items: center;">
            {logo_html}
        </div>
        <div style="font-size: 0.92rem; font-weight: 700; color: var(--ds-teal); margin-top: 0.4rem; line-height: 1.35;">
            Ứng dụng Business Intelligence để phân tích dữ liệu môi trường cho một huyện
        </div>
        <div style="font-size: 0.8rem; color: var(--ds-text-secondary); margin-top: 0.35rem; line-height: 1.5;">
            Trạm quan trắc IoT Sa Đéc — Đồng Tháp<br>
            <span style="color: var(--ds-text-muted); font-size: 0.76rem; display: inline-block; margin-top: 0.2rem;">
                <b>CBHD:</b> TS. Nguyễn Minh Khiêm<br>
                <b>HVTH:</b> Hoàng Xuân Trí (M2522016)
            </span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    st.sidebar.markdown(
        """
    <div style="font-size: 0.8rem; color: var(--ds-teal); text-transform: uppercase;
                letter-spacing: 0.1em; margin: 0.25rem 0 0.4rem 0.2rem; font-weight: 800;">
        📌 Danh mục phân hệ
    </div>
    """,
        unsafe_allow_html=True,
    )

    page: str = st.sidebar.radio(
        "Điều hướng phân hệ",
        [
            # ── I. Dữ liệu & Kiến trúc Pipeline ──
            "🏠 Tổng Quan",
            "📜 Quy Trình Pipeline",
            "📊 Khám Phá & Dẫn Luận Dữ Liệu (EDA)",
            # ── II. Mô hình hóa & Thực nghiệm ──
            "⚙️ Cấu Hình Siêu Tham Số (Hyperparameters)",
            "🏋️ Huấn Luyện Mô Hình",
            "📋 Nhật Ký Thực Nghiệm (Experiment Runs)",
            # ── III. Đánh giá, XAI & Kiểm định ──
            "📈 Kết Quả Đa Tầm Dự Báo (Multi-Horizon)",
            "📉 Thực Tế vs Dự Báo (Actual vs Predicted)",
            "🧠 Khả Năng Giải Thích Mô Hình (XAI)",
            "📊 Khoảng Dự Báo Bất Định (Prediction Intervals)",
            "📚 Đối Chiếu Khoa Học (Scientific Benchmark)",
            "📊 Thư Viện Biểu Đồ Kỹ Thuật (System Figures)",
            "🔬 Kiểm Định Khoa Học (Scientific Audit)",
            # ── IV. Dự báo & Tổng kết hệ thống ──
            "🔮 Dự Báo PM2.5",
            "📝 Kết Luận & Hướng Phát Triển",
            # ── V. Trợ lý AI & Quản trị nội dung ──
            "💬 Trợ Lý AI",
            "✏️ Quản Lý Nội Dung",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    # Theme Mode Selector (Auto / Light / Dark)
    st.sidebar.markdown(
        """
    <div style="font-size: 0.74rem; color: var(--ds-text-muted); text-transform: uppercase;
                letter-spacing: 0.08em; margin: 0.2rem 0 0.3rem 0; font-weight: 700;">
        🎨 Giao diện hiển thị (Theme)
    </div>
    """,
        unsafe_allow_html=True,
    )
    theme_choice = st.sidebar.selectbox(
        "Chế độ giao diện",
        [
            "🔄 Tự động (Theo cài đặt)",
            "☀️ Giao diện Sáng (Light)",
            "🌙 Giao diện Tối (Dark)",
        ],
        index=0,
        key="ui_theme_mode",
        label_visibility="collapsed",
    )
    _apply_theme_override(theme_choice)

    version_selector_sidebar()

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
    <div style="background: var(--ds-bg-card); color: var(--ds-text-primary); border-radius: 12px; padding: 1rem; border: 1px solid var(--ds-border); box-shadow: var(--ds-shadow);">
        <div style="font-size: 0.74rem; color: var(--ds-text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.65rem;">Thông Số Dữ Liệu Quan Trắc</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; font-size: 0.84rem;">
            <div style="color: var(--ds-text-secondary);">📅 Thời lượng</div><div style="color: var(--ds-teal); font-weight: 600;">38 tháng</div>
            <div style="color: var(--ds-text-secondary);">📦 Bản ghi thô</div><div style="color: var(--ds-teal); font-weight: 600;">209.594</div>
            <div style="color: var(--ds-text-secondary);">🎯 Biến mục tiêu</div><div style="color: var(--ds-teal); font-weight: 600;">PM2.5</div>
            <div style="color: var(--ds-text-secondary);">🧪 Kiểm thử tự động</div><div style="color: var(--ds-teal); font-weight: 600;">{_count_tests()} pass</div>
            <div style="color: var(--ds-text-secondary);">📐 Đặc trưng (v9)</div><div style="color: var(--ds-teal); font-weight: 600;">119</div>
            <div style="color: var(--ds-text-secondary);">🚫 Rò rỉ dữ liệu</div><div style="color: var(--ds-teal); font-weight: 600;">0 (shift=1)</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.checkbox(
        "🖨️ Chế độ In ấn (Grayscale B&W)",
        value=st.session_state.get("print_mode", False),
        help="Chuyển toàn bộ biểu đồ sang chế độ đen trắng (độ tương phản cao), tối ưu cho in ấn tài liệu khoa học",
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

    if page in (
        "📊 Thư Viện Biểu Đồ Kỹ Thuật (System Figures)",
        "📊 Hệ Thống Biểu Đồ Đề Án (Thesis Figures)",
        "📊 Thesis Figures",
        "📊 System Figures",
    ):
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
