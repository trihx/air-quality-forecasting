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
        {kpi_card("Anti-Leakage Tests", "154/154", "✅ All passed")}
        {kpi_card("Models × Versions", "28 · v6", "6 snapshot versions")}
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
        <span class="highlight">[7]</span> Evaluate: <span class="warn">MAE</span> (primary) + <span class="warn">MASE</span> (mandatory) + RMSE + R² + ROC-AUC
    </div>
    """, unsafe_allow_html=True)

    # ── Rankings ──
    section_header("🏆", "Final Model Rankings — v6 Updated")

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
    section_header("📊", "MASE — So Sánh Toàn Bộ Mô Hình (v6 updated)")

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Tổng Quan", 
        "2. Gaps & Spikes (Điểm Yếu)", 
        "3. Tính Dừng & Mùa Vụ", 
        "4. Autocorr & Drift (Điểm Mạnh)",
        "5. The 'Why' (Actions)"
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
                
        with c2:
            st.markdown("#### ⚡ Fat-Tailed Spikes")
            st.markdown("""
            **Bản chất:** Dữ liệu có đỉnh đuôi dài (Fat-Tailed), vi phạm giả định phân phối chuẩn (Non-normal). 
            Đây là những khoảng rủi ro y tế cao nhất (đỉnh ô nhiễm dị thường).
            """)
            img2 = RESEARCH_DIR / "eda" / "03_distributions.png"
            if img2.exists():
                st.image(str(img2), caption="Phân phối không chuẩn của PM2.5 (đuôi dài lệch phải)", use_container_width=True)

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
                
        with c2:
            st.markdown("#### 🔀 Concept Drift Đa Biến")
            st.markdown("""
            Tương quan PM2.5 vs Nhiệt độ không tĩnh, dao động thay đổi từ mùa này sang mùa khác (-0.6 đến +0.6). 
            Trạng thái này phá vỡ sự phỏng đoán của các mô hình tuyến tính cũ (như Ridge/Linear).
            """)
            img2 = EDA_DIR / "03b_hexbin_multivariate.png"
            if img2.exists():
                st.image(str(img2), caption="Concept drift: Tương quan phi tuyến tính thay đổi liên tục", use_container_width=True)

    with tab5:
        st.markdown("### 5. Tại sao tiếp cận Pipeline như vậy? (The 'Why')")
        st.markdown("Những nguyên lý cốt lõi trên giải thích lý do tại sao chúng ta thiết kế hệ thống ML Data Engineering:")
        
        st.info("**1. Xử lý Gaps (Thiếu hụt dữ liệu):** Vì missing data rớt theo chùm dài, các mô hình Linear Interpolation hỏng hoàn toàn. Chúng ta phải chia bậc: *Cubic Spline* (gaps ≤6h) -> *KNN* (6-24h) -> *Drop* (gaps >24h). Điều này vớt được tối đa dữ liệu mà vẫn giữ an toàn 100% Anti-Leakage.")
        st.info("**2. Xử lý Spikes (Mô hình Fat-Tailed):** PM2.5 có các đỉnh đột biến tàn phá loss function (MSE). Nên ta buộc dùng mô hình Deep Learning GRU kết hợp *Log Transform* hoặc áp dụng cơ chế *Quantile Regression* để đưa dự báo bao trùm được cận trên rủi ro (Upper Bound).")
        st.info("**3. Nắm bắt Mùa Vụ (Seasonality):** Chu kỳ đặc trưng buổi sáng (nghịch nhiệt) buộc ta phải ép thêm 110+ *Fourier features* và mã hóa Time-of-Day (v2) để DL học được quy luật vi khí hậu này.")
        st.info("**4. Thoát Bẫy Tự Tương Quan:** Vì r ≈ 0.97 ở 1 giờ, *MASE (Mean Absolute Scaled Error)* là metric sống còn. Mô hình phải đạt MASE < 1.0 thì mới được gọi là học đường nét mới thay vì chỉ copy giá trị cũ (Persistence).")


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
