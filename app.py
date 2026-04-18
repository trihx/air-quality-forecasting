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

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = PROJECT_ROOT / "research"
FIGURES_DIR = RESEARCH_DIR / "figures"
SHAP_DIR = FIGURES_DIR / "shap"
PI_DIR = FIGURES_DIR / "prediction_intervals"
EDA_DIR = RESEARCH_DIR / "eda" / "visualizations"

# ── Design System ──
COLORS = {
    "primary": "#00D4AA",      # Teal — ecological, scientific
    "secondary": "#4ECDC4",    # Light teal
    "accent": "#FF6B6B",       # Coral — for warnings/alerts
    "warning": "#FFE66D",      # Yellow
    "bg_dark": "#0E1117",      # Streamlit dark bg
    "card_bg": "#1A1F2E",      # Card background
    "text": "#FAFAFA",         # Primary text
    "text_muted": "#8B95A5",   # Secondary text
    "success": "#00D4AA",
    "danger": "#FF6B6B",
}

# Chart color palette — scientific, distinguishable
CHART_COLORS = [
    "#00D4AA",  # Teal (primary)
    "#FF6B6B",  # Coral
    "#4ECDC4",  # Sea green
    "#FFE66D",  # Yellow
    "#A78BFA",  # Purple
    "#FB923C",  # Orange
    "#60A5FA",  # Blue
    "#F472B6",  # Pink
]

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": COLORS["text"], "family": "Inter, sans-serif"},
        "xaxis": {"gridcolor": "rgba(139,149,165,0.15)", "zerolinecolor": "rgba(139,149,165,0.15)"},
        "yaxis": {"gridcolor": "rgba(139,149,165,0.15)", "zerolinecolor": "rgba(139,149,165,0.15)"},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
    }
}


# ── Page config ──
st.set_page_config(
    page_title="PM2.5 Forecasting — Scientific Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
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
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3D 100%);
        border: 1px solid rgba(0,212,170,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
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
    .kpi-label { font-size: 0.85rem; color: #8B95A5; font-weight: 500; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #FAFAFA; font-family: 'JetBrains Mono', monospace; }
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
    .insight-card p { margin: 0; color: #C5CDD8; line-height: 1.6; }

    /* ── Data Table Styling ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151B28 100%);
    }
    [data-testid="stSidebar"] hr { border-color: rgba(0,212,170,0.2); }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] span { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #CBD5E1 !important; }

    /* ── High-contrast text fixes ── */
    .stSelectbox label, .stNumberInput label, .stSlider label,
    .stRadio label, .stMultiSelect label { color: #E2E8F0 !important; }
    p, li, span, .stMarkdown { color: #D1D5DB; }
    .stRadio [data-testid="stMarkdownContainer"] p { color: #E2E8F0 !important; font-weight: 500; }

    /* ── Pipeline diagram ── */
    .pipeline-box {
        background: #1A1F2E;
        border: 1px solid rgba(0,212,170,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.8;
        color: #C5CDD8;
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

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


@st.cache_data
def load_json(path: Path) -> dict | list | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_experiment_results():
    results = {}
    for name, subdir in [
        ("multi_horizon", "multi_horizon"),
        ("dl", "dl"),
        ("ensemble", "ensemble"),
        ("prediction_intervals", "prediction_intervals"),
    ]:
        d = RESEARCH_DIR / "experiments" / subdir
        if d.exists():
            jsons = sorted(d.glob("*.json"))
            if jsons:
                results[name] = load_json(jsons[-1])

    cfg = RESEARCH_DIR / "best_models_configs.json"
    if cfg.exists():
        results["configs"] = load_json(cfg)

    shap_json = SHAP_DIR / "shap_results.json"
    if shap_json.exists():
        results["shap"] = load_json(shap_json)

    return results


def kpi_card(label, value, delta=None, delta_class="positive"):
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def section_header(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <span class="icon">{icon}</span>
        <span class="title">{title}</span>
    </div>
    """, unsafe_allow_html=True)


def insight_card(title, text, card_type="default"):
    cls = "warning" if card_type == "warning" else ""
    st.markdown(f"""
    <div class="insight-card {cls}">
        <h4>{title}</h4>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)


def apply_plotly_style(fig, height=450):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif", size=13),
        xaxis=dict(gridcolor="rgba(139,149,165,0.12)", zerolinecolor="rgba(139,149,165,0.12)"),
        yaxis=dict(gridcolor="rgba(139,149,165,0.12)", zerolinecolor="rgba(139,149,165,0.12)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════


def sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">🌫️</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00D4AA; margin-top: 0.5rem;">
            PM2.5 Forecasting
        </div>
        <div style="font-size: 0.8rem; color: #8B95A5; margin-top: 0.25rem;">
            Đề án Thạc sĩ — ĐH Cần Thơ
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    # -- Research workflow navigation --
    st.sidebar.markdown("""
    <div style="font-size: 0.65rem; color: #5A6577; text-transform: uppercase;
                letter-spacing: 0.12em; margin: 0.5rem 0 0.3rem 0.2rem; font-weight: 700;">
        📌 Quy trình nghiên cứu
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "📂 Navigation",
        [
            # ── Phase 1: Giới thiệu & Khám phá ──
            "🏠 Tổng Quan",
            "📊 EDA & Khám Phá Dữ Liệu",
            # ── Phase 2: Huấn luyện mô hình ──
            "⚙️ Cấu Hình & Hyperparameters",
            "🏋️ Huấn Luyện Mô Hình",
            "📋 Lịch Sử Thí Nghiệm",
            # ── Phase 3: Đánh giá & Giải thích ──
            "📈 Kết Quả Multi-Horizon",
            "📉 Actual vs Predicted",
            "🔍 Giải Thích Mô Hình (SHAP)",
            "📊 Khoảng Tin Cậy Dự Báo",
            # ── Phase 4: Ứng dụng ──
            "🔮 Dự Báo PM2.5",
            # ── Công cụ hỗ trợ ──
            "💬 Trợ Lý AI",
        ],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style="background: #1A1F2E; border-radius: 12px; padding: 1rem; border: 1px solid rgba(0,212,170,0.15);">
        <div style="font-size: 0.75rem; color: #8B95A5; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">Project Stats</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.85rem;">
            <div>📅 <span style="color:#8B95A5">Data</span></div><div style="color:#00D4AA">3.1 năm</div>
            <div>📦 <span style="color:#8B95A5">Records</span></div><div style="color:#00D4AA">209K</div>
            <div>🎯 <span style="color:#8B95A5">Target</span></div><div style="color:#00D4AA">PM2.5</div>
            <div>🧪 <span style="color:#8B95A5">Tests</span></div><div style="color:#00D4AA">133 ✅</div>
            <div>📐 <span style="color:#8B95A5">Features</span></div><div style="color:#00D4AA">95</div>
            <div>🚫 <span style="color:#8B95A5">Leakage</span></div><div style="color:#00D4AA">0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return page


# ══════════════════════════════════════════════════════════════════════
# Page: Overview
# ══════════════════════════════════════════════════════════════════════


def page_overview(results):
    st.markdown("""
    <h1 style="font-size: 2.2rem; margin-bottom: 0.25rem;">
        🌫️ Dự Báo Nồng Độ PM2.5 — Tổng Quan
    </h1>
    <p style="color: #8B95A5; font-size: 1.05rem; margin-bottom: 2rem;">
        Pipeline end-to-end từ IoT sensor → Feature Engineering (anti-leakage) → Multi-horizon Forecasting
    </p>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    st.markdown(f"""
    <div class="kpi-row">
        {kpi_card("Best Model (6h)", "GRU v2+log", "↓ 31.0% vs Persistence | MASE=0.692")}
        {kpi_card("Best MASE (1h)", "TFT v1 1.029", "Transformer ≈ Persistence")}
        {kpi_card("Anti-Leakage Tests", "167/167", "✅ All passed (v7)")}
        {kpi_card("Models × Versions", "28 · v7", "7 snapshot versions")}
    </div>
    """, unsafe_allow_html=True)

    # ── Hook: Data Storytelling ──
    section_header("📖", "Câu Chuyện Dữ Liệu")
    insight_card(
        "💡 Phát hiện quan trọng nhất",
        "<b>Feature engineering là con dao hai lưỡi cho Deep Learning.</b> "
        "Tại horizon 1h, Persistence (copy y[t-1]) bất bại do autocorrelation ≈ 0.99. "
        "Mở rộng features từ 5 → 117 (Fourier, tương tác, CV) thậm chí HẠI hiệu suất 1h. "
        "PCA (37 components) và Feature Selection (Top-40) đều không cứu được. "
        "Nhưng tại <b>6h, GRU v2+log transform đạt MASE = 0.692</b> — kết quả tốt nhất toàn pipeline, "
        "giảm <b>31%</b> lỗi so với Persistence và <b>↓14.8%</b> so với GRU v1.",
    )

    # ── Pipeline ──
    section_header("🔧", "Pipeline Architecture")
    st.markdown("""
    <div class="pipeline-box">
        <span class="highlight">IoT Sensor</span> (209K records, ~2 phút/mẫu, 3.1 năm)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span class="highlight">[1]</span> Raw Data → <span class="highlight">[2]</span> Clean (IQR 3.0, resample 1h → 27,649 rows)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span class="highlight">[3]</span> Impute (<span class="warn">Hybrid</span>: Spline ≤6h + KNN 6-24h) → 7,742 rows<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span class="highlight">[4]</span> Features (119 cols v2: lags, rolling, ewm, diff, Fourier, interactions, CV — <span class="accent">shift(1) anti-leakage</span>)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span class="highlight">[5]</span> Split 80/10/10 (temporal) → <span class="accent">TEST = REAL DATA ONLY</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span class="highlight">[6]</span> Models: Persistence → ARIMA → LightGBM → RF → GRU/LSTM/TFT → Ensemble<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span class="highlight">[7]</span> Evaluate: <span class="warn">MAE</span> (primary) + <span class="warn">MASE</span> (mandatory) + RMSE + R² + ROC-AUC + <span class="accent">Forecast Bias + MedAE + Residual Diagnostics (v7)</span>
    </div>
    """, unsafe_allow_html=True)

    # ── v7 Audit Info Card ──
    section_header("🔬", "v7 — Pipeline Audit Enhancement")
    insight_card(
        "🔬 What: Audit đối chiếu 8+ sách chuyên ngành",
        "<b>Why:</b> Đảm bảo pipeline đạt chuẩn academic (Manu Joseph, Peixeiro, Brownlee, Vishwas & Patel, Huang).<br>"
        "<b>What:</b> Bổ sung 5 biểu đồ EDA thiếu (STL, BoxPlot/Hour, Q-Q, Periodogram, Forecastability) "
        "+ 4 metrics (Forecast Bias, MedAE, RMSE/MAE Ratio) + Residual Diagnostics (Ljung-Box).<br>"
        "<b>Result:</b> STL Residual σ=5.18 µg/m³ → thiết lập 'sàn hiệu suất'. "
        "Forecastability Score=0.434 (Trung bình) → giải thích MASE>1 ở h=1 là expected. "
        "PM2.5 NOT Normal (p=1.4e-50) → justify MASE thay vì MAPE. Tests: 167/167 ✅",
    )

    # ── v7-exp Deseasonalizing Experiment ──
    insight_card(
        "🧪 v7-exp: Deseasonalizing Transform Experiment",
        "<b>Why:</b> Manu Joseph Ch.7 khuyến nghị target transformation. PM2.5 seasonal strength = 0.343.<br>"
        "<b>How:</b> 3 biến thể GRU h=6: (A) raw (0.731), (B) seasonal_diff y[t]-y[t-24] (0.903), (C) STL residual train-only (0.736).<br>"
        "<b>Result:</b> Cả seasonal_diff (0.903) lẫn STL leak-free (0.736) đều <b>KHÔNG cải thiện</b> so với raw (0.731). "
        "Fourier features ĐÃ capture seasonality → deseasonalizing thêm = redundant.<br>"
        "<b>⚠️ Leakage audit:</b> STL fitted full data (0.507) → STL train-only (0.736) = <b>+45% inflation</b> do look-ahead bias.<br>"
        "<b>🔑 Key Insight:</b> <i>Fourier features make explicit deseasonalizing REDUNDANT. GRU v2+log (0.692) remains BEST.</i>",
    )

    # ── Rankings ──
    section_header("🏆", "Final Model Rankings — v7 Updated")

    ranking_data = pd.DataFrame({
        "Mô hình": [
            "Persistence", "ARIMA(2,1,1)",
            "SARIMA×(2,1,0,24)", "LightGBM (Optuna)",
            "RandomForest", "Ensemble_Weighted",
            "LSTM v1 (5 feat)", "**GRU v1 (5 feat)**", "**TFT v1 (Transformer)**",
            "GRU v2+log (117 feat)", "LSTM v2 (117 feat)", "TFT v2 (113+4 feat)",
            "GRU_pca (37 comp)", "GRU_top40",
        ],
        "Loại": [
            "Baseline", "Statistical", "Statistical",
            "ML", "ML", "Ensemble",
            "DL v1", "DL v1", "Transformer v1",
            "DL v2", "DL v2", "Transformer v2",
            "DL+PCA", "DL+FeaSel",
        ],
        "1h MASE": [
            "1.000", "1.023", "1.283", "1.492",
            "—", "1.249",
            "1.560", "1.173", "⭐ 1.029",
            "1.531", "1.888", "1.976",
            "1.572", "1.497",
        ],
        "6h MASE": [
            "1.000", "0.856", "0.762", "0.745",
            "0.706", "0.705",
            "0.914", "0.812", "0.822",
            "⭐⭐⭐ 0.692", "0.719", "0.850",
            "—", "—",
        ],
        "24h MASE": [
            "1.000", "0.913", "0.813", "0.842",
            "0.798", "0.797",
            "0.830", "⭐⭐ 0.727", "0.812",
            "0.781", "0.734", "0.886",
            "—", "—",
        ],
    })
    st.dataframe(ranking_data, use_container_width=True, hide_index=True)
    st.caption("*v1 = 5 raw features. v2 = 117 features (Fourier, interactions, CV, lags, rolling).*")

    # ── Key Findings ──
    col1, col2 = st.columns(2)
    with col1:
        insight_card(
            "✅ Thành công chính",
            "• <b>GRU v2+log</b> giảm <b>31.0%</b> lỗi so với Persistence tại 6h (MASE=0.692) ⭐<br>"
            "• Feature engineering v2 cải thiện GRU ↓14.8% và LSTM ↓21.3% so với v1 tại 6h<br>"
            "• TFT v1 đạt MASE=1.029 tại 1h — tốt nhất trong tất cả ML/DL<br>"
            "• Anti-leakage pipeline: 154 tests passed, 4 nguồn leakage đã loại bỏ<br>"
            "• Diebold-Mariano test xác nhận ý nghĩa thống kê (p < 0.001)",
        )
    with col2:
        insight_card(
            "⚠️ Hạn chế & Bài học",
            "• Persistence bất bại ở h=1h — autocorrelation 0.99 chi phối hoàn toàn<br>"
            "• Feature engineering KHÔNG giúp 1h: PCA (37 comp) → 1.572, Top-40 → 1.497, v1 (5 feat) vẫn best<br>"
            "• TFT v2 tệ hơn v1 (+92% ở 1h) do hidden_dim=32 không đủ cho 113 features<br>"
            "• Log transform phụ thuộc kiến trúc: GRU thích log, LSTM ưa raw ở 6h",
            card_type="warning",
        )


# ══════════════════════════════════════════════════════════════════════
# Page: Multi-Horizon
# ══════════════════════════════════════════════════════════════════════


def page_multi_horizon(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">📊 Kết Quả Multi-Horizon</h1>
    <p style="color: #8B95A5;">So sánh hiệu suất dự báo PM2.5 tại 3 horizons: 1h, 6h, 24h</p>
    """, unsafe_allow_html=True)

    # ── MASE Chart ──
    section_header("📊", "MASE — So Sánh Toàn Bộ Mô Hình (v7 updated)")

    models = [
        "Persistence", "ARIMA", "SARIMA",
        "LightGBM", "RF", "GRU v1", "TFT v1",
        "GRU v2+log", "LSTM v2",
    ]
    mase_data = {
        "Persistence": [1.000, 1.000, 1.000],
        "ARIMA": [1.023, 0.856, 0.913],
        "SARIMA": [1.283, 0.762, 0.813],
        "LightGBM": [1.492, 0.745, 0.842],
        "RF": [None, 0.706, 0.798],
        "GRU v1": [1.173, 0.812, 0.727],
        "TFT v1": [1.029, 0.822, 0.812],
        "GRU v2+log": [1.531, 0.692, 0.781],
        "LSTM v2": [1.888, 0.719, 0.734],
    }
    horizons = ["1h", "6h", "24h"]

    fig = go.Figure()
    for i, model in enumerate(models):
        vals = mase_data[model]
        fig.add_trace(go.Bar(
            name=model, x=horizons, y=[v if v else 0 for v in vals],
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            text=[f"{v:.3f}" if v else "—" for v in vals],
            textposition="outside", textfont={"size": 10},
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["accent"],
                  line_width=2,
                  annotation_text="Baseline (MASE = 1.0)",
                  annotation_font_color=COLORS["accent"])

    fig.update_layout(
        barmode="group",
        yaxis_title="MASE (thấp hơn = tốt hơn)",
        xaxis_title="Forecast Horizon",
        legend={"orientation": "h", "y": 1.14, "x": 0.5, "xanchor": "center"},
        title={"text": "MASE < 1.0 → Mô hình vượt trội Persistence Baseline", "font": {"size": 15}},
    )
    fig = apply_plotly_style(fig, height=520)
    st.plotly_chart(fig, use_container_width=True)

    # ── Data Storytelling: 3 insights by horizon ──
    col1, col2, col3 = st.columns(3)
    with col1:
        insight_card(
            "🔴 h=1: Autocorrelation Trap",
            "PM2.5 autocorrelation ≈ 0.99 tại lag-1h. <br>"
            "<b>Persistence bất bại</b> — mọi feature engineering chỉ thêm nhiễu.<br><br>"
            "PCA (117→37 components) = 1.57<br>"
            "Top-40 features = 1.50<br>"
            "GRU v1 (5 raw) = 1.17<br>"
            "<b>TFT v1 = 1.029 (closest)</b>",
        )
    with col2:
        insight_card(
            "🟢 h=6: Feature Engineering Shines",
            "Autocorrelation giảm → temporal patterns quan trọng.<br>"
            "<b>GRU v2+log = 0.692 ⭐ NEW BEST</b><br><br>"
            "v2 features cải thiện:<br>"
            "• GRU: 0.81 → 0.69 (↓14.8%)<br>"
            "• LSTM: 0.91 → 0.72 (↓21.3%)<br>"
            "Log transform giúp GRU thêm 9.1%",
        )
    with col3:
        insight_card(
            "🔵 h=24: GRU v1 = Champion",
            "Multivariate patterns (nhiet_do, diem_suong) trở nên quan trọng.<br>"
            "<b>GRU v1 = 0.727 ⭐⭐ (↓27.3% lỗi)</b><br><br>"
            "DM test: p = 0.012 (significant)<br>"
            "LSTM v2 cũng đạt 0.734 (↓11.5%)<br>"
            "TFT v1 = 0.812 (dataset quá nhỏ)",
        )

    # ── MAE Trend ──
    section_header("📈", "MAE Theo Horizon — Xu Hướng Sai Số")
    mae_data = {
        "Persistence": [2.390, 6.305, 6.279],
        "LightGBM": [3.720, 5.046, 5.178],
        "GRU v1": [2.805, 5.119, 4.562],
        "TFT v1": [2.573, 5.565, 4.999],
        "GRU v2+log": [3.660, 4.360, 4.880],
        "LSTM v2": [4.510, 4.530, 4.610],
    }

    fig2 = go.Figure()
    trend_colors = ["#FF6B6B", "#FFE66D", "#4ECDC4", "#A78BFA", "#00D4AA", "#60A5FA"]
    for i, (model, values) in enumerate(mae_data.items()):
        line_width = 4 if model in ("GRU v2+log", "GRU v1") else 2
        fig2.add_trace(go.Scatter(
            name=model, x=horizons, y=values, mode="lines+markers+text",
            line={"color": trend_colors[i], "width": line_width},
            marker={"size": 12 if line_width == 4 else 8, "line": {"width": 2, "color": "#0E1117"}},
            text=[f"{v:.2f}" for v in values],
            textposition="top center", textfont={"size": 10},
        ))
    fig2.update_layout(
        yaxis_title="MAE (µg/m³) — thấp hơn = chính xác hơn",
        xaxis_title="Forecast Horizon",
        title={"text": "Xu Hướng Sai Số: model giỏi ở 1h chưa chắc giỏi ở 24h", "font": {"size": 14}},
    )
    fig2 = apply_plotly_style(fig2, height=450)
    st.plotly_chart(fig2, use_container_width=True)

    insight_card(
        "💡 Insight: No Single Best Model",
        "<b>Kết luận chính:</b> Không có 1 mô hình duy nhất tốt nhất cho mọi horizon.<br>"
        "• <b>1h</b>: TFT v1 (5 feat) — Attention khai thác short-term<br>"
        "• <b>6h</b>: GRU v2+log (117 feat) — Feature engineering + log transform<br>"
        "• <b>24h</b>: GRU v1 (5 feat) — Tổng quát hóa tốt với dataset nhỏ<br><br>"
        "<b>Tại sao?</b> Autocorrelation giảm dần: 0.99 (1h) → 0.85 (6h) → 0.45 (24h). "
        "Khi autocorr giảm, multivariate features và feature engineering bắt đầu tạo giá trị.",
    )

    # ── DM Test ──
    section_header("📐", "Diebold-Mariano — Ý Nghĩa Thống Kê")
    dm_data = pd.DataFrame({
        "So sánh": [
            "GRU v2+log vs Persistence (6h)", "GRU v1 vs Persistence (24h)",
            "LightGBM vs Persistence (6h)", "LightGBM vs Persistence (24h)",
        ],
        "DM Statistic": [-4.21, -3.89, -3.57, -2.45],
        "p-value": ["< 0.001", "< 0.001", "< 0.001", "0.014"],
        "Δ vs Persistence": ["-31.0%", "-27.3%", "-25.5%", "-15.8%"],
        "Kết luận": ["✅ Significant", "✅ Significant", "✅ Significant", "✅ Significant"],
    })
    st.dataframe(dm_data, use_container_width=True, hide_index=True)
    st.caption("*Diebold-Mariano test (1995): p < 0.05 → sự khác biệt có ý nghĩa thống kê. GRU v2+log = best significance.*")


# ══════════════════════════════════════════════════════════════════════
# Page: SHAP
# ══════════════════════════════════════════════════════════════════════


def page_shap(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">🔍 SHAP Explainability</h1>
    <p style="color: #8B95A5;">SHapley Additive exPlanations (LightGBM) + Permutation Importance (GRU)</p>
    """, unsafe_allow_html=True)

    insight_card(
        "💡 Tại sao Explainability quan trọng?",
        "SHAP giải thích <b>tại sao</b> mô hình dự đoán giá trị cụ thể, không chỉ <b>chính xác bao nhiêu</b>. "
        "Điều này giúp xác nhận rằng mô hình học đúng pattern vật lý (nhiệt độ, chu kỳ ngày đêm) "
        "thay vì exploit noise trong dữ liệu.",
    )

    tab1, tab2, tab3 = st.tabs(["📊 SHAP Bar", "🌊 SHAP Beeswarm", "🧠 GRU Permutation"])

    with tab1:
        section_header("📊", "Top Features — SHAP Mean Absolute Values")
        h = st.selectbox("Chọn horizon", ["1h", "6h", "24h"], key="shap_bar_h")
        img_path = SHAP_DIR / f"shap_bar_{h}.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning(f"File chưa tồn tại: {img_path.name}")

    with tab2:
        section_header("🌊", "Feature Impact Distribution")
        h2 = st.selectbox("Chọn horizon", ["1h", "6h", "24h"], key="shap_bee_h")
        img_path2 = SHAP_DIR / f"shap_beeswarm_{h2}.png"
        if img_path2.exists():
            st.image(str(img_path2), use_container_width=True)
        else:
            st.warning(f"File chưa tồn tại: {img_path2.name}")

    with tab3:
        section_header("🧠", "GRU — Permutation Importance")
        h3 = st.selectbox("Chọn horizon", ["1h", "6h", "24h"], key="gru_perm_h")
        img_path3 = SHAP_DIR / f"gru_permutation_{h3}.png"
        if img_path3.exists():
            st.image(str(img_path3), use_container_width=True)
        else:
            st.warning(f"File chưa tồn tại: {img_path3.name}")

    # ── SHAP Table ──
    section_header("📋", "Top 5 Features (LightGBM SHAP)")
    shap_df = pd.DataFrame({
        "Hạng": [1, 2, 3, 4, 5],
        "h=1 (SHAP)": [
            "pm25_lag_1h (3.420)", "co2 (1.200)",
            "pm25_roll_24h_mean (0.988)", "pm25_lag_24h (0.562)", "do_am (0.469)",
        ],
        "h=6 (SHAP)": [
            "pm25_roll_24h_mean (2.351)", "hour_sin (1.900)",
            "pm25_roll_24h_min (1.180)", "co2_lag_48h (0.589)", "pm25_roll_6h_min (0.511)",
        ],
        "h=24 (SHAP)": [
            "pm25_lag_1h (2.204)", "hour_cos (0.908)",
            "pm25_lag_24h (0.656)", "diem_suong_lag_1h (0.534)", "diem_suong_lag_24h (0.394)",
        ],
    })
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    # ── SHAP Dependence (bonus) ──
    section_header("🔗", "SHAP Dependence Plots")
    dep_images = sorted(SHAP_DIR.glob("shap_dep_*.png"))
    if dep_images:
        h_select = st.selectbox("Horizon", ["1h", "6h", "24h"], key="shap_dep_h")
        filtered = [img for img in dep_images if f"_{h_select}_" in img.name]
        if filtered:
            cols = st.columns(min(len(filtered), 3))
            for i, img in enumerate(filtered):
                with cols[i % 3]:
                    feature_name = img.stem.split(f"_{h_select}_")[-1]
                    st.image(str(img), caption=feature_name, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# Page: Prediction Intervals
# ══════════════════════════════════════════════════════════════════════


def page_prediction_intervals(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">📈 Prediction Intervals</h1>
    <p style="color: #8B95A5;">Khoảng dự báo 90% — 3 phương pháp: Conformal, Quantile, MC Dropout</p>
    """, unsafe_allow_html=True)

    pi_data = results.get("prediction_intervals", [])
    if not pi_data:
        st.warning("Chưa có kết quả. Chạy `uv run python scripts/prediction_intervals.py`")
        return

    pi_df = pd.DataFrame(pi_data)

    # ── KPI ──
    best = pi_df.loc[pi_df["coverage"].idxmax()]
    st.markdown(f"""
    <div class="kpi-row">
        {kpi_card("Best Coverage", f"{best['coverage']:.1%}", f"{best['method']} — {best['model']} {best['horizon']}h")}
        {kpi_card("Methods Tested", "3", "Conformal · Quantile · MC Dropout")}
        {kpi_card("Confidence Level", "90%", f"α = {COLORS.get('alpha', 0.10)}")}
    </div>
    """, unsafe_allow_html=True)

    # ── Coverage Chart ──
    section_header("📊", "Coverage vs Target (90%)")
    fig = go.Figure()
    methods = pi_df["method"].unique()
    for i, method in enumerate(methods):
        subset = pi_df[pi_df["method"] == method]
        fig.add_trace(go.Bar(
            name=method.replace("_", " ").title(),
            x=[f"{r['horizon']}h" for _, r in subset.iterrows()],
            y=[v * 100 for v in subset["coverage"].values],
            marker_color=CHART_COLORS[i],
            text=[f"{v:.1%}" for v in subset["coverage"]], textposition="outside",
        ))
    fig.add_hline(y=90, line_dash="dash", line_color=COLORS["accent"],
                  annotation_text="Target 90%", annotation_font_color=COLORS["accent"])
    fig.update_layout(barmode="group", yaxis_title="Coverage (%)", yaxis_range=[0, 110])
    fig = apply_plotly_style(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ── Width Chart ──
    section_header("📏", "Interval Width (µg/m³)")
    fig2 = go.Figure()
    for i, method in enumerate(methods):
        subset = pi_df[pi_df["method"] == method]
        fig2.add_trace(go.Bar(
            name=method.replace("_", " ").title(),
            x=[f"{r['horizon']}h" for _, r in subset.iterrows()],
            y=subset["avg_width"].values,
            marker_color=CHART_COLORS[i],
            text=[f"{v:.1f}" for v in subset["avg_width"]], textposition="outside",
        ))
    fig2.update_layout(barmode="group", yaxis_title="Avg Width (µg/m³)")
    fig2 = apply_plotly_style(fig2, height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # ── PI Plots ──
    pi_images = sorted(PI_DIR.glob("pi_*.png")) if PI_DIR.exists() else []
    if pi_images:
        section_header("🖼️", "Sample Prediction Intervals")
        cols = st.columns(min(len(pi_images), 3))
        for i, img in enumerate(pi_images):
            with cols[i % 3]:
                st.image(str(img), caption=img.stem.replace("pi_", "").replace("_", " "), use_container_width=True)

    # ── Table ──
    section_header("📋", "Tổng Hợp Chi Tiết")
    display_df = pi_df[["method", "model", "horizon", "coverage", "avg_width", "mae"]].copy()
    display_df["coverage"] = display_df["coverage"].apply(lambda x: f"{x:.1%}")
    display_df.columns = ["Phương pháp", "Mô hình", "Horizon (h)", "Coverage", "Width (µg/m³)", "MAE (µg/m³)"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    insight_card(
        "💡 Nhận xét",
        "<b>Quantile Regression</b> cho coverage cao nhất (~83-86%) nhưng interval rộng (~16-19 µg/m³).<br>"
        "<b>Conformal Prediction</b> cân bằng hơn: coverage ~77-80%, width ~11-15 µg/m³.<br>"
        "<b>MC Dropout</b> coverage thấp vì GRU dropout rate nhỏ → uncertainty estimate quá hẹp.",
    )


# ══════════════════════════════════════════════════════════════════════
# Page: EDA
# ══════════════════════════════════════════════════════════════════════


def page_eda(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">📉 Cốt Truyện Dữ Liệu (Data Storytelling)</h1>
    <p style="color: #8B95A5;">Hành trình khám phá dữ liệu IoT và cơ sở nền tảng thiết kế Feature Engineering</p>
    """, unsafe_allow_html=True)

    import json
    eda_json_path = RESEARCH_DIR / "eda" / "eda_results.json"
    eda_data = {}
    if eda_json_path.exists():
        with open(eda_json_path, "r") as f:
            eda_data = json.load(f)

    pm25_desc = eda_data.get("descriptive", {}).get("pm25", {})

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Tổng Quan",
        "2. Gaps & Spikes (Điểm Yếu)",
        "3. Tính Dừng & Mùa Vụ",
        "4. Autocorr & Drift (Điểm Mạnh)",
        "5. The 'Why' (Actions)",
        "6. Deep Insights (v7)"
    ])

    with tab1:
        st.markdown("### 1. Tổng Quan & Cường Độ (Dataset Overview)")
        st.markdown("Dữ liệu PM2.5 thu thập từ cảm biến IoT với tần suất cao (5 phút/lần). Dưới đây là mảng thông tin tổng quan trước khi đi sâu vào các câu chuyện dữ liệu.")

        cols = st.columns(4)
        cols[0].metric("Tổng số điểm (sau clean)", f"{pm25_desc.get('count', 0):,}")
        cols[1].metric("Trung bình (Mean)", f"{pm25_desc.get('mean', 0):.1f} µg/m³")
        cols[2].metric("Trung vị (Median)", f"{pm25_desc.get('median', 0):.1f} µg/m³")
        cols[3].metric("Đỉnh điểm (Max)", f"{pm25_desc.get('max', 0):.1f} µg/m³", delta="Cực đoan", delta_color="inverse")

        insight_card("💡 Phân tích Tổng Quan",
                     "Sự chênh lệch lớn giữa Mean (~13.2) và Max (~54.0) cho thấy PM2.5 không phân bố đều mà chứa các đỉnh ô nhiễm cục bộ. "
                     "Tần suất lấy mẫu cung cấp độ phân giải cao, lý tưởng để nắm bắt các biến động ngắn hạn nhưng cũng chứa nhiều nhiễu.")

        img_path = RESEARCH_DIR / "eda" / "02_pm25_timeseries.png"
        if img_path.exists():
            st.image(str(img_path), caption="Tổng quan chuỗi thời gian PM2.5", use_container_width=True)

        # ── Forecastability Assessment (P0-2) ──
        fc = eda_data.get("forecastability", {})
        if fc:
            st.markdown("---")
            st.markdown("#### 🎯 Forecastability Assessment")
            st.markdown("*Đo mức độ khả thi dự báo TRƯỚC khi chọn model (Ref: Manu Joseph Ch.4)*")

            fc_cols = st.columns(5)
            fc_cols[0].metric("CoV (σ/μ)", f"{fc.get('cov', 0):.3f}", help="Coefficient of Variation — cao = khó dự báo")
            fc_cols[1].metric("ApEn", f"{fc.get('approximate_entropy', 0):.3f}", help="Approximate Entropy — cao = phức tạp")
            fc_cols[2].metric("Seasonality", f"{fc.get('seasonality_strength', 0):.3f}", help="Sức mạnh mùa vụ từ STL")
            fc_cols[3].metric("ACF(1)", f"{fc.get('acf_lag1', 0):.3f}", help="Tự tương quan lag-1")
            fc_cols[4].metric("Score", f"{fc.get('forecastability_score', 0):.3f}", delta=fc.get("interpretation", ""), delta_color="normal")

            insight_card("📊 Đánh Giá Forecastability",
                        f"**Score = {fc.get('forecastability_score', 0):.3f}** → {fc.get('interpretation', 'N/A')}. "
                        f"CoV cao ({fc.get('cov', 0):.2f}) cho thấy biến động mạnh. "
                        f"ACF(1) = {fc.get('acf_lag1', 0):.3f} rất cao → Persistence baseline cực mạnh ở h=1. "
                        "Điều này giải thích tại sao MASE > 1.0 ở h=1 là **expected** chứ không phải model kém.")
            
            # P1-6: Complexity Profile Radar chart
            st.markdown("---")
            st.markdown("#### 🕸️ P1-6: Complexity Profile Radar")
            st.markdown("*Ref: Visualizing multiple dimensions of time series complexity*")
            
            phase5_json = EDA_DIR / "phase5_dashboard_data.json"
            if phase5_json.exists():
                with open(phase5_json, "r") as f:
                    p5_data = json.load(f)
                
                radar_data = p5_data.get("complexity_radar")
                if radar_data:
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_data["values"] + [radar_data["values"][0]],
                        theta=radar_data["metrics"] + [radar_data["metrics"][0]],
                        fill='toself',
                        name='Complexity',
                        line_color='#00D4AA',
                        fillcolor='rgba(0, 212, 170, 0.3)'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(139,149,165,0.15)'),
                            angularaxis=dict(gridcolor='rgba(139,149,165,0.15)'),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=COLORS["text"]),
                        margin=dict(l=40, r=40, t=20, b=20)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    insight_card("🕸️ Complexity Profile",
                        "Biểu đồ Radar tổng hợp 5 chiều độ phức tạp (quy về thang 0-1). "
                        "Diện tích đa giác càng phủ rộng (ra rìa ngoài) nghĩa là chuỗi thời gian càng phức tạp, khó đoán và biến động mạnh. "
                        "Mặt trái: Noise và CoV lớn. Mặt phải: Seasonality và Trend tạo cấu trúc bám víu cho model.")

    with tab2:
        st.markdown("### 2. Điểm Yếu Của IoT: Gaps & Spikes")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🕳️ Data Quality Gaps")
            st.markdown("""
            **Bản chất:** Cảm biến rớt mạng theo chùm. Dữ liệu thật chỉ chiếm một phần rất nhỏ ban đầu.
            Missing data không ngẫu nhiên (Non-MCAR) mà theo chuỗi offline.
            """)
            img1 = EDA_DIR / "04a_missing_barcode.png"
            if img1.exists():
                st.image(str(img1), caption="Mô hình khuyết thiếu (Missing Barcode) - vạch đen là missing", use_container_width=True)
            img_recovery = EDA_DIR / "04b_recovery_limits.png"
            if img_recovery.exists():
                st.image(str(img_recovery), caption="Giới hạn khôi phục: Gap ≤6h (Spline) → 6-24h (KNN) → >24h (Drop)", use_container_width=True)

        with c2:
            st.markdown("#### ⚡ Fat-Tailed Spikes")
            st.markdown("""
            **Bản chất:** Dữ liệu có đỉnh đuôi dài (Fat-Tailed), vi phạm giả định phân phối chuẩn (Non-normal).
            Đây là những khoảng rủi ro y tế cao nhất (đỉnh ô nhiễm dị thường).
            """)
            img2 = RESEARCH_DIR / "eda" / "03_distributions.png"
            if img2.exists():
                st.image(str(img2), caption="Phân phối không chuẩn của PM2.5 (đuôi dài lệch phải)", use_container_width=True)
            img_fat = EDA_DIR / "02a_pm25_fat_tailed_distribution.png"
            if img_fat.exists():
                st.image(str(img_fat), caption="Phân tích chi tiết: Fat-Tailed Distribution với các percentile", use_container_width=True)

        # P1-1: Q-Q Plot
        st.markdown("---")
        st.markdown("#### 📐 Q-Q Plot — Kiểm Tra Tính Chuẩn (Normality)")
        st.markdown("*Ref: Peixeiro Ch.6 — So sánh phân phối PM2.5 với normal distribution*")
        img_qq = RESEARCH_DIR / "eda" / "03c_qq_plot.png"
        if img_qq.exists():
            st.image(str(img_qq), caption="Q-Q Plot: Raw (trái) vs Log-Transformed (phải) — đường thẳng = normal", use_container_width=True)
        norm_data = eda_data.get("normality", {})
        if norm_data:
            insight_card("📐 Kết Quả Normality Test",
                        f"**Shapiro-Wilk (raw):** p = {norm_data.get('shapiro_p_raw', 'N/A')} → {'Normal ✅' if norm_data.get('is_normal_raw') else 'NOT Normal ❌'} | "
                        f"**Shapiro-Wilk (log):** p = {norm_data.get('shapiro_p_log', 'N/A')} → {'Normal ✅' if norm_data.get('is_normal_log') else 'NOT Normal ❌'}. "
                        "Cả raw lẫn log-transform đều KHÔNG đạt normal. Đây là lý do cần **MASE** thay vì MAPE (phụ thuộc giả định normal).")

    with tab3:
        st.markdown("### 3. Tính Dừng & Tính Mùa Vụ (Stationarity / Seasonality)")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📊 Stationarity (Tính dừng)")
            st.markdown("""
            Kết quả kiểm định ADF và KPSS mâu thuẫn (Inconclusive).
            Dữ liệu IoT thường có "Variance dừng" (nhiễu đồng nhất) nhưng "Mean không dừng" (phụ thuộc mùa/tháng).
            """)
            img1 = RESEARCH_DIR / "eda" / "06_acf_pacf.png"
            if img1.exists():
                st.image(str(img1), caption="Đồ thị ACF/PACF cho thấy hiện tượng tự tương quan kéo dài", use_container_width=True)

        with c2:
            st.markdown("#### 🌅 Seasonality (Mùa vụ)")
            st.markdown("""
            **Nhịp điệu sinh học:** PM2.5 cao nhất vào ban đêm/sáng sớm (đỉnh ~6h) do hiện tượng nghịch nhiệt,
            và chạm đáy vào giữa trưa (12h-14h) nhờ hiệu ứng đối lưu không khí.
            """)
            img2 = RESEARCH_DIR / "eda" / "07_temporal_patterns.png"
            if img2.exists():
                st.image(str(img2), caption="Chu kỳ thay đổi trong ngày (Diurnal) và tháng", use_container_width=True)

        # P0-6: Box Plot per Hour
        st.markdown("---")
        st.markdown("#### 📦 Box Plot Theo Giờ — Seasonal Pattern Trực Quan")
        st.markdown("*Ref: Vishwas & Patel Ch.4 — Phân phối PM2.5 tại mỗi giờ trong ngày*")
        img_box = RESEARCH_DIR / "eda" / "05b_boxplot_hourly.png"
        if img_box.exists():
            st.image(str(img_box), caption="Peak 6h sáng (nghịch nhiệt), Trough 12h trưa (đối lưu) — chứng cứ diurnal cycle", use_container_width=True)

        # P0-1: STL Decomposition
        st.markdown("---")
        st.markdown("#### 🔬 STL Decomposition — Tách Thành Phần Chuỗi Thời Gian")
        st.markdown("*Ref: Manu Joseph Ch.3 — Seasonal-Trend decomposition using LOESS (period=24h)*")
        stl_data = eda_data.get("stl", {})
        if stl_data:
            stl_cols = st.columns(4)
            stl_cols[0].metric("Trend Strength", f"{stl_data.get('trend_strength', 0):.3f}")
            stl_cols[1].metric("Seasonal Strength", f"{stl_data.get('seasonal_strength', 0):.3f}")
            stl_cols[2].metric("Noise Ratio", f"{stl_data.get('noise_ratio', 0):.3f}")
            stl_cols[3].metric("Residual σ", f"{stl_data.get('residual_std', 0):.2f} µg/m³", help="Performance floor — mô hình không thể dưới giá trị này")
        img_stl = RESEARCH_DIR / "eda" / "05_stl_decomposition.png"
        if img_stl.exists():
            st.image(str(img_stl), caption="STL: Original → Trend → Seasonal → Residual", use_container_width=True)
        img_stl_zoom = RESEARCH_DIR / "eda" / "05a_stl_seasonal_zoom.png"
        if img_stl_zoom.exists():
            st.image(str(img_stl_zoom), caption="Seasonal Component zoom 1 tuần — 7 chu kỳ 24h rõ nét", use_container_width=True)
        if stl_data:
            insight_card("🔬 Phân Tích STL",
                        f"**Trend Strength = {stl_data.get('trend_strength', 0):.3f}** → Trend trung bình (có xu hướng nhẹ theo mùa). "
                        f"**Seasonal Strength = {stl_data.get('seasonal_strength', 0):.3f}** → Mùa vụ nhóm trung bình. "
                        f"**Residual σ = {stl_data.get('residual_std', 0):.2f} µg/m³** → Đây là 'sàn hiệu suất': "
                        "model mà đạt MAE ≈ Residual σ nghĩa là đã khai thác hết signal có thể.")

        # P1-2: Periodogram / PSD
        st.markdown("---")
        st.markdown("#### 📡 Periodogram — Xác Nhận Tần Số Chủ Đạo")
        st.markdown("*Ref: Huang Ch.7 — Power Spectral Density xác nhận frequencies mà Fourier features cần encode*")
        img_psd = RESEARCH_DIR / "eda" / "05c_periodogram.png"
        if img_psd.exists():
            st.image(str(img_psd), caption="PSD cho thấy dominant periods — validate Fourier feature design", use_container_width=True)
        spec_data = eda_data.get("spectral", {})
        if spec_data and spec_data.get("dominant_periods"):
            periods_str = ", ".join([f"{p['period_hours']}h" for p in spec_data["dominant_periods"][:5]])
            insight_card("📡 Spectral Analysis",
                        f"**Dominant periods (by power):** {periods_str}. "
                        "Top-5 là trend dài hạn (tháng/năm) — phản ánh xu hướng mùa. "
                        "Tín hiệu 24h daily cycle đã được xác nhận qua **STL Decomposition** (seasonal strength=0.343). "
                        "Fourier features (period=24) encode đúng chu kỳ mà STL cho thấy.")

        # P1-8: Expanding Window Stats
        st.markdown("---")
        st.markdown("#### 🌊 P1-8: Expanding Window Statistics — Kiểm tra phi tĩnh (Non-stationarity)")
        st.markdown("*Ref: Peixeiro Ch.4 — Thống kê mở rộng cho thấy Mean/Variance có hội tụ hay không*")
        
        if phase5_json.exists():
            # Loading is handled in tab1, so p5_data should exist if tab1 ran, but let's be safe
            if 'p5_data' not in locals():
                with open(phase5_json, "r") as f:
                    p5_data = json.load(f)
                    
            exp_data = p5_data.get("expanding_window")
            if exp_data:
                fig_exp = go.Figure()
                fig_exp.add_trace(go.Scatter(x=exp_data['dates'], y=exp_data['pm25_raw'], name='Raw PM2.5', opacity=0.3, line=dict(color='#8B95A5')))
                fig_exp.add_trace(go.Scatter(x=exp_data['dates'], y=exp_data['expanding_mean'], name='Expanding Mean', line=dict(color=COLORS['primary'], width=3)))
                fig_exp.add_trace(go.Scatter(x=exp_data['dates'], y=exp_data['expanding_std'], name='Expanding Std', line=dict(color=COLORS['accent'], width=3)))
                fig_exp.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_exp.update_layout(
                    title="Expanding Window Mean & Std",
                    xaxis_title="Thời gian (Date)",
                    yaxis_title="PM2.5 Statistics",
                    margin=dict(l=40, r=40, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_exp, use_container_width=True)
                
                insight_card("🌊 Expanding Window Stats",
                    "Expanding Mean (Đường màu Teal) dao động trong giai đoạn đầu và dần mượt hơn nhưng vẫn tiếp tục có xu hướng (trend). "
                    "Expanding Std (Đường màu Coral) tăng dần theo thời gian. "
                    "Cả hai đường đều **không đi ngang (non-flat)**, phản ánh hiện tượng **Không dừng (Non-stationarity)** về Mean và Variance "
                    "mà kiểm định ADF/KPSS đã phát hiện. Điều này ép các mô hình tuyến tính phải khác biệt hóa (differencing).")

    with tab4:
        st.markdown("### 4. Điểm Mạnh: Tự Tương Quan & Concept Drift")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔄 Autocorrelation Trap")
            st.markdown("""
            **Sức mạnh lừa dối:** Tại h=1h, tự tương quan r ≈ 0.97.
            Mô hình dễ mắc "Bẫy tự tương quan" - dự đoán h=1 rất tốt nhưng thực chất chỉ lấy giá trị giờ trước.
            """)
            img1 = EDA_DIR / "01a_autocorrelation_memory.png"
            if img1.exists():
                st.image(str(img1), caption="Bẫy tự tương quan: Trí nhớ dữ liệu (Memory) cực cao ở lag gần", use_container_width=True)
            img_disp = EDA_DIR / "01b_horizon_scatter_dispersion.png"
            if img_disp.exists():
                st.image(str(img_disp), caption="Scatter Dispersion: Error tăng nhanh khi horizon tăng", use_container_width=True)

        with c2:
            st.markdown("#### 🔀 Concept Drift Đa Biến")
            st.markdown("""
            Tương quan PM2.5 vs Nhiệt độ không tĩnh, dao động thay đổi từ mùa này sang mùa khác (-0.6 đến +0.6).
            Trạng thái này phá vỡ sự phỏng đoán của các mô hình tuyến tính cũ (như Ridge/Linear).
            """)
            img2 = EDA_DIR / "03b_hexbin_multivariate.png"
            if img2.exists():
                st.image(str(img2), caption="Concept drift: Tương quan phi tuyến tính thay đổi liên tục", use_container_width=True)
            img_roll = EDA_DIR / "03a_rolling_correlation.png"
            if img_roll.exists():
                st.image(str(img_roll), caption="Rolling Correlation: Tương quan thay đổi theo thời gian (60-day window)", use_container_width=True)

        # P1-7: Walk-Forward Stability
        st.markdown("---")
        st.markdown("#### 🚶 P1-7: Walk-Forward Stability (Monthly Volatility)")
        st.markdown("*Ref: Peixeiro Ch.8 — Đánh giá độ ổn định của Mean/Variance qua các block thời gian thực tế*")
        
        if phase5_json.exists():
            if 'p5_data' not in locals():
                with open(phase5_json, "r") as f:
                    p5_data = json.load(f)
            
            wf_data = p5_data.get("walk_forward")
            if wf_data:
                fig_wf = go.Figure()
                fig_wf.add_trace(go.Bar(x=wf_data['dates'], y=wf_data['mean'], name='Monthly Mean', marker_color='rgba(0, 212, 170, 0.6)'))
                fig_wf.add_trace(go.Scatter(x=wf_data['dates'], y=wf_data['std'], name='Monthly Std (Risk)', mode='lines+markers', line=dict(color=COLORS['accent'], width=2)))
                fig_wf.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_wf.update_layout(
                    title="Phân phối PM2.5 theo tháng (Mean vs Volatility)",
                    xaxis_title="Tháng",
                    yaxis_title="PM2.5 (µg/m³)",
                    margin=dict(l=40, r=40, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_wf, use_container_width=True)
                
                insight_card("🚶 Walk-Forward Stability",
                    "Volatility (Đường Coral) không nhất quán giữa các tháng, những tháng đỉnh điểm mùa khô có cả Mean và Std đều bật tăng mạnh. "
                    "Tính chất **Heteroskedasticity** (Phương sai thay đổi) này giải thích tại sao Walk-forward Validation (TimeSeriesSplit) lại ưu việt hơn các phương pháp k-Fold truyền thống.")

    with tab5:
        st.markdown("### 5. Tại sao tiếp cận Pipeline như vậy? (The 'Why')")
        st.markdown("Những nguyên lý cốt lõi trên giải thích lý do tại sao chúng ta thiết kế hệ thống ML Data Engineering:")

        st.info("**1. Xử lý Gaps (Thiếu hụt dữ liệu):** Vì missing data rớt theo chùm dài, các mô hình Linear Interpolation hỏng hoàn toàn. Chúng ta phải chia bậc: *Cubic Spline* (gaps ≤6h) -> *KNN* (6-24h) -> *Drop* (gaps >24h). Điều này vớt được tối đa dữ liệu mà vẫn giữ an toàn 100% Anti-Leakage.")
        st.info("**2. Xử lý Spikes (Mô hình Fat-Tailed):** PM2.5 có các đỉnh đột biến tàn phá loss function (MSE). Nên ta buộc dùng mô hình Deep Learning GRU kết hợp *Log Transform* hoặc áp dụng cơ chế *Quantile Regression* để đưa dự báo bao trùm được cận trên rủi ro (Upper Bound).")
        st.info("**3. Nắm bắt Mùa Vụ (Seasonality):** Chu kỳ đặc trưng buổi sáng (nghịch nhiệt) buộc ta phải ép thêm 110+ *Fourier features* và mã hóa Time-of-Day (v2) để DL học được quy luật vi khí hậu này.")
        st.info("**4. Thoát Bẫy Tự Tương Quan:** Vì r ≈ 0.97 ở 1 giờ, *MASE (Mean Absolute Scaled Error)* là metric sống còn. Mô hình phải đạt MASE < 1.0 thì mới được gọi là học đường nét mới thay vì chỉ copy giá trị cũ (Persistence).")
        st.info("**5. STL Residual σ = 'Sàn Hiệu Suất':** Phân tích STL cho thấy Residual σ ≈ 5.2 µg/m³ — model đạt MAE gần giá trị này nghĩa là đã khai thác hết signal. Đây là cơ sở đánh giá model đã tối ưu hay chưa.")
        
        st.markdown("#### 🚀 Phase 6: Thiết Kế Nâng Cao (v7)")
        phase6_json = EDA_DIR / "phase6_dashboard_data.json"
        
        box_cox_msg = "Kiểm định sự cần thiết của phép biến đổi phi tuyến tính (Log Transform)."
        sesd_msg = "Thuật toán phát hiện dị thường S-ESD (Seasonal Extreme Studentized Deviate) giúp nhận dạng Outliers chính xác trên chuỗi có tính mùa vụ cao."
        purging_msg = "Gap purging xử lý leakage ẩn giữa rollings của Train qua Test."
        
        if phase6_json.exists():
            import json as _json
            with open(phase6_json, "r") as f:
                p6_data = _json.load(f)
            
            box_cox = p6_data.get('box_cox', {})
            purging = p6_data.get('purging_gap', {})
            
            if 'optimal_lambda' in box_cox:
                box_cox_msg = f"**Box-Cox Optimal \u03bb = {box_cox['optimal_lambda']:.3f}** \u2192 Recommend: **{box_cox['interpretation']}**. Điều này chứng minh một cách khoa học việc chọn Log Transformation (`np.log1p`) ở các thiết lập DL v2 ban đầu là cực kỳ đúng đắn."
                
            if 'status' in purging:
                purging_msg = f"**{purging['concept']}**: {purging['definition']} \n\n\u2714\ufe0f **Status:** {purging['status']}"

        st.success(f"**6. Optimal Transform (Box-Cox):** {box_cox_msg}")
        st.success(f"**7. Robust Anomaly (S-ESD):** {sesd_msg} Nhờ áp dụng ngưỡng outlier IQR/Domain Bounds dựa trên MAD, ta đã loại bỏ nhiễu thành công mà không phá vỡ Seasonal Peaks.")
        st.success(f"**8. Purge Gap Validation:** {purging_msg}")

    with tab6:
        st.markdown("### 6. Deep Insights — Error Anatomy, Granger, Cross-Correlation (v7)")
        st.markdown("*Bổ sung theo khuyến nghị Manu Joseph Ch.7, Peixeiro Ch.10, Huang Ch.3*")

        # Load deep insights data
        di_path = RESEARCH_DIR / "eda" / "deep_insights_results.json"
        di_data = {}
        if di_path.exists():
            import json as _json
            with open(di_path) as f:
                di_data = _json.load(f)

        # ── P1-3: Error Anatomy ──
        st.markdown("---")
        st.markdown("#### 🔍 Error Anatomy — GRU v2+log @ h=6")
        st.markdown("*Lỗi dự báo phân bố theo giờ trong ngày và mức ô nhiễm như thế nào?*")

        ea = di_data.get("error_anatomy", {})
        if ea:
            ea_cols = st.columns(4)
            ea_cols[0].metric("Worst Hour", f"{ea.get('worst_hour', '?')}h",
                            delta=f"MAE={ea.get('worst_hour_mae', '?')}", delta_color="inverse")
            ea_cols[1].metric("Best Hour", f"{ea.get('best_hour', '?')}h",
                            delta=f"MAE={ea.get('best_hour_mae', '?')}", delta_color="normal")
            ea_cols[2].metric("Error ACF(1)", f"{ea.get('error_acf_lag1', '?')}",
                            help="Cao = errors structured (model missing patterns)")
            ea_cols[3].metric("Error ACF(24)", f"{ea.get('error_acf_lag24', '?')}",
                            help="Cao = daily pattern in errors")

        img_ea = EDA_DIR / "06_error_anatomy.png"
        if img_ea.exists():
            st.image(str(img_ea), caption="Error Anatomy: MAE by Hour, Bias by Hour, MAE by Level, Error ACF", use_container_width=True)

        if ea:
            insight_card("🔍 Phân Tích Error Anatomy",
                f"Model lỗi NHIỀU NHẤT lúc <b>{ea.get('worst_hour', '?')}h</b> (MAE={ea.get('worst_hour_mae', '?')}) "
                f"và ÍT NHẤT lúc <b>{ea.get('best_hour', '?')}h</b> (MAE={ea.get('best_hour_mae', '?')}). "
                f"Error ACF(1)={ea.get('error_acf_lag1', '?')} → errors có cấu trúc tự tương quan cao, "
                "cho thấy model vẫn chưa nắm bắt hết temporal patterns. "
                f"ACF(24)={ea.get('error_acf_lag24', '?')} → daily pattern trong errors đã được Fourier capture phần lớn.")

        # ── P1-4: Granger Causality ──
        st.markdown("---")
        st.markdown("#### 🧬 Granger Causality — Biến ngoại sinh có giúp predict PM2.5?")
        st.markdown("*Ref: Peixeiro Ch.10 — Kiểm định nhân quả Granger (F-test, α=0.05). Fitted trên TRAIN ONLY.*")

        gc = di_data.get("granger_causality", {})
        if gc:
            gc_rows = []
            for col, gr in gc.items():
                if "error" not in gr:
                    gc_rows.append({
                        "Variable": gr["label"],
                        "Best Lag": f"{gr['best_lag']}h",
                        "p-value": f"{gr['best_p_value']:.2e}" if gr['best_p_value'] > 0 else "< 1e-10",
                        "Significant (α=0.05)": "✅ Yes" if gr["significant_at_005"] else "❌ No",
                    })
            if gc_rows:
                st.dataframe(pd.DataFrame(gc_rows), use_container_width=True, hide_index=True)

        img_gc = EDA_DIR / "07_granger_causality.png"
        if img_gc.exists():
            st.image(str(img_gc), caption="Granger Causality p-values across lags", use_container_width=True)

        if gc:
            all_sig = all(gr.get("significant_at_005", False) for gr in gc.values() if "error" not in gr)
            insight_card("🧬 Granger Causality",
                "<b>Tất cả biến ngoại sinh đều Granger-cause PM2.5</b> (p < 0.05). "
                "Điều này xác nhận việc sử dụng Temperature, Humidity, CO2 "
                "làm input features là hợp lý — chúng THỰC SỰ cung cấp thông tin dự báo, "
                "không chỉ là noise. CO2 có best lag=6h → phù hợp horizon 6h." if all_sig else
                "Một số biến không significant → cần xem xét loại bỏ.")

        # ── P1-5: Cross-Correlation ──
        st.markdown("---")
        st.markdown("#### 📊 Cross-Correlation Lagged — PM2.5 vs External Variables")
        st.markdown("*Ref: Huang Ch.3 — Xác nhận lag nào có tương quan mạnh nhất, validate thiết kế lag features*")

        cc = di_data.get("cross_correlation", {})
        if cc:
            cc_rows = []
            for col, cr in cc.items():
                cc_rows.append({
                    "Variable": cr["label"],
                    "Best Lag": f"{cr['best_lag_hours']}h",
                    "Best r": f"{cr['best_correlation']:.4f}",
                    "r(lag=0)": f"{cr.get('cc_at_lag0', 'N/A')}",
                    "r(lag=6)": f"{cr.get('cc_at_lag6', 'N/A')}",
                    "r(lag=24)": f"{cr.get('cc_at_lag24', 'N/A')}",
                })
            if cc_rows:
                st.dataframe(pd.DataFrame(cc_rows), use_container_width=True, hide_index=True)

        img_cc = EDA_DIR / "08_cross_correlation.png"
        if img_cc.exists():
            st.image(str(img_cc), caption="Cross-Correlation: positive lag = external variable LEADS PM2.5", use_container_width=True)

        if cc:
            co2_r = cc.get("co2", {}).get("best_correlation", 0)
            insight_card("📊 Cross-Correlation",
                f"<b>CO2</b> có tương quan mạnh nhất với PM2.5 (r={co2_r:.4f}), "
                "xác nhận chúng chia sẻ nguồn phát thải (combustion). "
                "Temperature/Humidity tương quan yếu (~0.2) — chúng ảnh hưởng gián tiếp qua cơ chế khí quyển. "
                "Kết quả validate rằng lag features [1, 6, 12, 24, 48] bao phủ đủ các peak cross-correlation.")


# ══════════════════════════════════════════════════════════════════════
# Page: Hyperparameters
# ══════════════════════════════════════════════════════════════════════


def page_hyperparams(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">⚙️ Hyperparameter Configurations</h1>
    <p style="color: #8B95A5;">Chi tiết cấu hình tối ưu cho từng mô hình và horizon</p>
    """, unsafe_allow_html=True)

    configs = results.get("configs")

    # ── LightGBM ──
    section_header("🌲", "LightGBM (Optuna Bayesian)")
    lgbm_table = pd.DataFrame({
        "Tham số": [
            "n_estimators", "max_depth", "learning_rate",
            "num_leaves", "subsample", "colsample_bytree",
            "min_child_samples", "reg_alpha", "reg_lambda",
        ],
        "h=1": [500, 3, 0.013, 64, 0.8, 0.6, 30, 0.05, 0.5],
        "h=6": [637, 3, 0.012, 87, 0.85, 0.55, 25, 0.03, 0.7],
        "h=24": [450, 4, 0.015, 52, 0.75, 0.65, 35, 0.08, 0.4],
    })
    st.dataframe(lgbm_table, use_container_width=True, hide_index=True)
    st.caption("*Optuna TPE sampler, 100 trials/horizon, TimeSeriesSplit(5), minimize MAE*")

    # ── DL ──
    section_header("🧠", "GRU / LSTM")
    dl_table = pd.DataFrame({
        "Tham số": [
            "lookback", "hidden_dim", "num_layers", "dropout",
            "batch_size", "learning_rate", "epochs (max)",
            "early_stopping_patience", "optimizer", "scheduler",
        ],
        "Giá trị": [
            72, 64, 2, 0.2, 64, 0.001, 100, 10,
            "Adam", "ReduceLROnPlateau (factor=0.5, patience=5)",
        ],
    })
    st.dataframe(dl_table, use_container_width=True, hide_index=True)
    st.caption("*Features: pm25, nhiet_do, do_am, diem_suong, co2 | Device: MPS (M1 Pro GPU)*")

    # ── ARIMA/SARIMA ──
    section_header("📈", "ARIMA / SARIMA")
    arima_table = pd.DataFrame({
        "Mô hình": ["ARIMA", "SARIMA"],
        "Bậc (p,d,q)": ["(2, 1, 1)", "(1, 0, 0)"],
        "Seasonal (P,D,Q,s)": ["—", "(2, 1, 0, 24)"],
        "Phương pháp chọn": ["auto_arima (AIC)", "auto_arima (AIC)"],
        "Rolling window": [720, 720],
    })
    st.dataframe(arima_table, use_container_width=True, hide_index=True)

    # ── Raw configs ──
    if configs:
        section_header("📄", "Raw Configurations (JSON)")
        with st.expander("Xem chi tiết JSON"):
            st.json(configs)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    results = load_experiment_results()
    page = sidebar()

    # ── Pages defined in app.py (always available, no extra import) ──
    local_pages = {
        "🏠 Tổng Quan": page_overview,
        "📊 EDA & Khám Phá Dữ Liệu": page_eda,
        "⚙️ Cấu Hình & Hyperparameters": page_hyperparams,
        "📈 Kết Quả Multi-Horizon": page_multi_horizon,
        "🔍 Giải Thích Mô Hình (SHAP)": page_shap,
        "📊 Khoảng Tin Cậy Dự Báo": page_prediction_intervals,
    }

    if page in local_pages:
        local_pages[page](results)
        return

    # ── Lazy import: pages.py (numpy, plotly heavy) ──
    if page in ("🏋️ Huấn Luyện Mô Hình", "📋 Lịch Sử Thí Nghiệm",
                "📉 Actual vs Predicted", "🔮 Dự Báo PM2.5"):
        from pages import (
            page_actual_vs_predicted,
            page_experiment_runs,
            page_forecast,
            page_training,
        )
        pages_map = {
            "🏋️ Huấn Luyện Mô Hình": page_training,
            "📋 Lịch Sử Thí Nghiệm": page_experiment_runs,
            "📉 Actual vs Predicted": page_actual_vs_predicted,
            "🔮 Dự Báo PM2.5": page_forecast,
        }
        pages_map[page](results)
        return

    # ── Lazy import: chatbot (sentence_transformers ~4s first load) ──
    if page == "💬 Trợ Lý AI":
        from src.chatbot.chat_page import page_ai_assistant
        page_ai_assistant(results)
        return

    # Fallback
    page_overview(results)


if __name__ == "__main__":
    main()
