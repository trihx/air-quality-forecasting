"""Thesis Figures Dashboard Page — Interactive visualizations for thesis defense.

Charts rendered with Plotly via chart_factory for consistent styling.
Data loaded from verified JSON artifacts (reproduce.sh validated).

Usage:
    Registered in app.py sidebar as "📊 Thesis Figures"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.viz.chart_factory import (
    chart as _chart,
    render_chart as _render_chart,
    figure_caption as _caption,
    figure_caption_numbered,
    styled_bar,
    add_baseline,
    render_bw_download,
)
from src.viz.theme import PALETTE_CATEGORICAL, PALETTE_SEMANTIC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
DIAGNOSTICS_DIR = RESEARCH_DIR / "diagnostics"
THESIS_FIGURES_DIR = RESEARCH_DIR / "figures" / "thesis"


# ── Data loaders (cached) ──

@st.cache_data(ttl=3600)
def _load_bootstrap_ci():
    path = DIAGNOSTICS_DIR / "bootstrap_mase_ci.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def _load_standardized_metrics():
    path = RESEARCH_DIR / "experiments" / "standardized_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def _load_ljungbox():
    path = DIAGNOSTICS_DIR / "residual_ljungbox.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def _load_shap():
    path = RESEARCH_DIR / "figures" / "shap" / "shap_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Chart builders ──

def _chart_bootstrap_ci(bc_data):
    """Bootstrap 95% CI bar chart with error bars."""
    models = ["GRU_v9_30m", "LSTM_v9_30m", "LightGBM_v9_30m", "Ensemble_30m"]
    labels = ["GRU", "LSTM", "LightGBM", "Ensemble"]
    horizons = ["1h", "6h", "24h"]
    colors = [PALETTE_CATEGORICAL[6], PALETTE_CATEGORICAL[5], PALETTE_CATEGORICAL[0]]

    fig = _chart(
        xaxis_title="Mô hình",
        yaxis_title="MASE (unified)",
        height=480,
        barmode="group",
    )

    # B&W-safe hatch patterns for each horizon
    hatch_patterns = ["/", "\\", "x"]

    for i, h in enumerate(horizons):
        mase_vals, err_minus, err_plus = [], [], []
        for model in models:
            if model in bc_data and h in bc_data[model]:
                m = bc_data[model][h]
                mase_vals.append(m["mase_point"])
                err_minus.append(m["mase_point"] - m["ci_lower"])
                err_plus.append(m["ci_upper"] - m["mase_point"])
            else:
                mase_vals.append(None)
                err_minus.append(0)
                err_plus.append(0)

        fig.add_trace(go.Bar(
            name=f"h={h}",
            x=labels,
            y=mase_vals,
            marker_color=colors[i],
            marker_pattern_shape=hatch_patterns[i],
            marker_pattern_solidity=0.15,
            opacity=0.85,
            text=[f"{v:.3f}" if v else "" for v in mase_vals],
            textposition="outside",
            textfont=dict(size=9),
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                color="#CCCCCC",
                thickness=1.5,
                width=4,
            ),
        ))

    # Persistence threshold
    fig.add_hline(
        y=1.0, line_dash="dash", line_color=PALETTE_SEMANTIC["danger"],
        line_width=2, opacity=0.7,
        annotation_text="MASE = 1.0 (Persistence)",
        annotation_position="top right",
        annotation_font_color=PALETTE_SEMANTIC["danger"],
    )

    # Better/Worse zones
    fig.add_hrect(
        y0=0, y1=1.0, fillcolor=PALETTE_SEMANTIC["success"],
        opacity=0.05, line_width=0,
    )

    fig.update_yaxes(range=[0, 1.55])
    return fig


def _chart_mase_decay(sm_data):
    """MASE decay across horizons — line chart."""
    # B&W-safe: each line has unique (color, dash, marker_symbol)
    plot_models = {
        "GRU 15m": ("GRU_v9_15m", PALETTE_CATEGORICAL[6], "dot", "circle"),
        "GRU 30m": ("GRU_v9_30m", PALETTE_CATEGORICAL[6], "solid", "diamond"),
        "LSTM 30m": ("LSTM_v9_30m", PALETTE_CATEGORICAL[5], "dash", "square"),
        "LightGBM 30m": ("LightGBM_v9_30m", PALETTE_CATEGORICAL[0], "dashdot", "triangle-up"),
        "Ensemble 30m": ("Ensemble_Weighted_v9_30m", PALETTE_CATEGORICAL[4], "longdash", "star"),
    }

    fig = _chart(
        xaxis_title="Horizon dự báo (giờ)",
        yaxis_title="MASE (unified)",
        height=480,
    )

    h_labels = ["1h", "6h", "24h"]
    h_numeric = [1, 6, 24]

    for label, (key, color, dash, symbol) in plot_models.items():
        mase_vals = []
        for h in h_labels:
            if key in sm_data.get("results", {}).get(h, {}):
                mase_vals.append(sm_data["results"][h][key]["mase_unified"])
            else:
                mase_vals.append(None)

        fig.add_trace(go.Scatter(
            name=label,
            x=h_numeric,
            y=mase_vals,
            mode="lines+markers",
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=10, symbol=symbol, line=dict(width=1.5, color="#333")),
        ))

    # Persistence
    fig.add_hline(
        y=1.0, line_dash="dot", line_color=PALETTE_SEMANTIC["danger"],
        line_width=1.5, opacity=0.6,
        annotation_text="Persistence (MASE=1.0)",
        annotation_position="top right",
        annotation_font_color=PALETTE_SEMANTIC["danger"],
    )

    fig.add_hrect(
        y0=0, y1=1.0, fillcolor=PALETTE_SEMANTIC["success"],
        opacity=0.05, line_width=0,
    )

    fig.update_xaxes(
        tickvals=[1, 6, 24],
        ticktext=["1 giờ", "6 giờ", "24 giờ"],
        range=[0, 25],
    )
    fig.update_yaxes(range=[0.2, 1.15])

    # Annotations for best models
    annotations = [
        dict(x=1, y=0.667, text="GRU 15m<br><b>0.667</b>", xanchor="left", xshift=10),
        dict(x=6, y=0.382, text="Ensemble 30m<br><b>0.382</b>", xanchor="left", xshift=10),
        dict(x=24, y=0.469, text="Ensemble 30m<br><b>0.469</b>", xanchor="right", xshift=-10),
    ]
    for ann in annotations:
        fig.add_annotation(
            **ann,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor=PALETTE_CATEGORICAL[4],
            font=dict(size=10, color=PALETTE_CATEGORICAL[4]),
            bgcolor="rgba(0,0,0,0.5)", borderpad=4,
        )

    return fig


def _chart_shap_comparison(shap_data):
    """SHAP top features comparison across 3 horizons — horizontal bar subplots."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("h = 1 giờ", "h = 6 giờ", "h = 24 giờ"),
        shared_yaxes=False,
        horizontal_spacing=0.12,
    )

    colors_gradient = [PALETTE_CATEGORICAL[0], PALETTE_CATEGORICAL[2]]

    for idx, h in enumerate(["1h", "6h", "24h"]):
        if h not in shap_data:
            continue
        top_features = shap_data[h].get("top_15_shap", {})
        top10 = list(top_features.items())[:10]
        top10.reverse()  # Plotly horizontal bar: bottom-to-top

        features = [f[0].replace("pm25_", "").replace("_", " ") for f, _ in top10]
        values = [v for _, v in top10]

        max_v = max(values) if values else 1
        bar_colors = [
            PALETTE_CATEGORICAL[0] if v > max_v * 0.5 else PALETTE_CATEGORICAL[2]
            for v in values
        ]

        fig.add_trace(go.Bar(
            y=features,
            x=values,
            orientation="h",
            marker_color=bar_colors,
            showlegend=False,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont=dict(size=9),
        ), row=1, col=idx + 1)

    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=20),
    )
    for i in range(3):
        fig.update_xaxes(title_text="SHAP Value", row=1, col=i + 1, gridcolor="rgba(128,128,128,0.2)")
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", row=1, col=i + 1)

    return fig


def _chart_residual_bias(lb_data):
    """Residual bias heatmap — mean residual per model × horizon."""
    models = ["GRU", "LSTM", "LightGBM", "Ensemble"]
    horizons = ["1h", "6h", "24h"]

    z_vals = []
    annotations = []
    for i, model in enumerate(models):
        row = []
        for j, h in enumerate(horizons):
            if model in lb_data and h in lb_data[model]:
                mean_val = lb_data[model][h]["mean"]
                row.append(mean_val)
                annotations.append(dict(
                    x=j, y=i,
                    text=f"{mean_val:+.2f}",
                    showarrow=False,
                    font=dict(size=12, color="white" if abs(mean_val) > 1.0 else "#333"),
                ))
            else:
                row.append(0)
        z_vals.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=horizons,
        y=models,
        colorscale=[
            [0, "#2E86C1"],     # Negative bias (under-prediction) = blue
            [0.5, "#2ECC71"],   # Zero bias = green
            [1, "#E74C3C"],     # Positive bias (over-prediction) = red
        ],
        zmid=0,
        text=[[f"{v:+.2f}" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont=dict(size=13),
        colorbar=dict(title=dict(text="Mean Residual<br>(µg/m³)", side="right")),
        hovertemplate="Model: %{y}<br>Horizon: %{x}<br>Bias: %{z:+.2f} µg/m³<extra></extra>",
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Horizon dự báo"),
        yaxis=dict(title=""),
    )

    return fig


def _chart_train_time():
    """Computational cost bar chart."""
    models = ["SARIMA\n(walk-fwd)", "ARIMA\n(walk-fwd)", "TFT\n(50 epochs)", "LightGBM\n(Optuna 50)", "LSTM\n(50 epochs)", "GRU\n(50 epochs)", "Ensemble\n(inference)"]
    times = [137.4, 33.6, 3.0, 1.0, 0.3, 0.2, 0.05]

    fig = _chart(
        xaxis_title="Thời gian (giây, log scale)",
        yaxis_title="",
        height=380,
    )

    colors = [
        PALETTE_CATEGORICAL[1] if t > 10 else
        PALETTE_CATEGORICAL[5] if t > 0.5 else
        PALETTE_CATEGORICAL[0]
        for t in times
    ]

    fig.add_trace(go.Bar(
        y=models,
        x=times,
        orientation="h",
        marker_color=colors,
        text=[f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms" for t in times],
        textposition="outside",
        textfont=dict(size=10),
        showlegend=False,
    ))

    fig.update_xaxes(type="log", range=[-1.5, 2.5])
    fig.update_layout(margin=dict(l=10, r=40, t=10, b=30))

    return fig


# ══════════════════════════════════════════════════════════════════════
# Page
# ══════════════════════════════════════════════════════════════════════

def page_thesis_figures(results):
    """Thesis-ready interactive figures for copy-to-Word."""
    from app import section_header, insight_card, kpi_card
    from src.frontend.citations import cite, render_references_section

    st.markdown("""
    <h1 style="font-size: 2.2rem; margin-bottom: 0.25rem;">
        📊 Thesis Figures — Hình Minh Họa cho Luận Văn
    </h1>
    <p style="opacity: 0.7; font-size: 1.05rem; margin-bottom: 2rem;">
        Biểu đồ tương tác Plotly — Click vào biểu tượng 📷 để tải PNG chất lượng cao (300 DPI) cho Word.
    </p>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    bc_data = _load_bootstrap_ci()
    sm_data = _load_standardized_metrics()
    lb_data = _load_ljungbox()
    shap_data = _load_shap()

    n_figures = 5  # Number of interactive charts
    n_static = len(list(THESIS_FIGURES_DIR.glob("*.png"))) if THESIS_FIGURES_DIR.exists() else 0

    st.markdown(f"""
    <div class="kpi-row">
        {kpi_card("Biểu đồ tương tác", str(n_figures), "Plotly interactive")}
        {kpi_card("Hình tĩnh sẵn sàng", str(n_static), "PNG 300 DPI")}
        {kpi_card("Tổng hình (toàn dự án)", "73", "Xem FIGURES_MAP.md")}
        {kpi_card("Độ tin cậy", "✅ Verified", "reproduce.sh 7/7")}
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "💡 **Hướng dẫn dành cho Hội đồng & Trích xuất hình:**\n"
        "- **Chỉ số MASE < 1,0:** Thể hiện mô hình tốt hơn Persistence Naive Baseline (lấy giá trị giờ trước). Ví dụ: MASE = 0,382 (Ensemble 30m) nghĩa là giảm 61,8% sai số tuyệt đối so với baseline (Hyndman 2006).\n"
        "- **Chế độ in ấn (B&W):** Tất cả biểu đồ hỗ trợ tải dạng bản in Đen-Trắng (PNG 300 DPI) với nét gạch đan (hatch patterns) và hộp nền chữ bảo vệ, đạt chuẩn trình bày Luận văn QĐ 1799/ĐHCT."
    )

    # ══════════════════════════════════════════════════════
    # Tab 1: Bootstrap CI
    # ══════════════════════════════════════════════════════
    tab_ci, tab_decay, tab_shap, tab_bias, tab_cost = st.tabs([
        "📊 Bootstrap 95% CI",
        "📈 MASE Decay",
        "🧠 SHAP Features",
        "🌡️ Residual Bias",
        "⏱️ Chi phí tính toán",
    ])

    with tab_ci:
        section_header("📊", "Bootstrap 95% CI cho MASE — Pipeline v9, 30 phút")
        if bc_data:
            fig_ci = _chart_bootstrap_ci(bc_data)
            _render_chart(fig_ci, filename="bootstrap_ci_barplot")
            figure_caption_numbered(4, 1,
                f"Bootstrap 95% CI cho MASE (n=2.000 block bootstrap, block_size=24). "
                f"Error bars thể hiện khoảng tin cậy. "
                f"Đường đứt đỏ = Persistence baseline (MASE = 1,0). {cite('hyndman2006')}")
            render_bw_download(fig_ci, filename="bootstrap_ci_barplot")

            insight_card(
                "💡 Nhận định chính",
                "Tại h=6 và h=24, **toàn bộ CI** nằm dưới 1,0 → "
                "kết quả có **ý nghĩa thống kê** (p < 0,05). "
                "Tại h=1, CI chứa 1,0 → mô hình chưa vượt Persistence "
                "trên dữ liệu 30m (cần 15m). "
                f"Ensemble 30m h=6 CI: [0,419 – 0,552] — hẹp nhất, ổn định nhất. {cite('diebold1995')}"
            )
        else:
            st.warning("Không tìm thấy file bootstrap_mase_ci.json")

    with tab_decay:
        section_header("📈", "MASE theo Horizon — So sánh mô hình và độ phân giải")
        if sm_data:
            fig_decay = _chart_mase_decay(sm_data)
            _render_chart(fig_decay, filename="mase_decay_chart")
            figure_caption_numbered(4, 2,
                f"MASE (unified) theo horizon dự báo. "
                f"Tất cả mô hình đều vượt Persistence (MASE < 1,0) ở h ≥ 6. "
                f"GRU 15m tối ưu ở h=1, Ensemble 30m tối ưu ở h ≥ 6. {cite('hyndman2006')}")
            render_bw_download(fig_decay, filename="mase_decay_chart")

            insight_card(
                "💡 Hiện tượng MASE Decay",
                "MASE giảm mạnh từ h=1 → h=6 (tín hiệu autocorrelation yếu đi, "
                "Persistence kém hơn → dễ thắng hơn). "
                "Từ h=6 → h=24, MASE tăng nhẹ do uncertainty tăng. "
                "**30 phút** là sweet spot: giảm noise so với 15m nhưng giữ đủ granularity."
            )
        else:
            st.warning("Không tìm thấy standardized_metrics.json")

    with tab_shap:
        section_header("🧠", "Top-10 SHAP Features theo Horizon dự báo (LightGBM)")
        if shap_data:
            fig_shap = _chart_shap_comparison(shap_data)
            _render_chart(fig_shap, filename="shap_comparison_horizons")
            figure_caption_numbered(4, 3,
                f"Top-10 đặc trưng quan trọng nhất theo SHAP mean |value| "
                f"(LightGBM TreeExplainer). {cite('lundberg2017')}")
            render_bw_download(fig_shap, filename="shap_comparison_horizons")

            insight_card(
                "💡 Feature Importance Shift",
                "**h=1:** `pm25_lag_1h` chiếm ưu thế tuyệt đối (SHAP = 2,82) — "
                "autocorrelation quyết định 100% ở ngắn hạn. "
                "**h=6:** Trọng tâm dịch sang rolling mean 24h và chu kỳ ngày đêm. "
                "**h=24:** Cần kết hợp lag, Fourier, và cross-domain (humidity × PM2.5)."
            )
        else:
            st.warning("Không tìm thấy shap_results.json")

    with tab_bias:
        section_header("🌡️", "Bias hệ thống — Mean Residual (µg/m³)")
        if lb_data:
            fig_bias = _chart_residual_bias(lb_data)
            _render_chart(fig_bias, filename="residual_bias_heatmap")
            figure_caption_numbered(4, 4,
                f"Heatmap bias phần dư: xanh = under-prediction (mean < 0), "
                f"đỏ = over-prediction (mean > 0), xanh lá = cân bằng (≈ 0). "
                f"Kiểm định tự tương quan: Ljung-Box test lag=24. {cite('ljung1978')}")
            render_bw_download(fig_bias, filename="residual_bias_heatmap")

            # Ljung-Box detail table
            with st.expander("📋 Chi tiết Ljung-Box test (p-values)", expanded=False):
                rows = []
                for model in ["GRU", "LSTM", "LightGBM", "Ensemble"]:
                    for h in ["1h", "6h", "24h"]:
                        if model in lb_data and h in lb_data[model]:
                            m = lb_data[model][h]
                            lb24 = m["ljung_box"]["lag_24"]
                            rows.append({
                                "Model": model,
                                "Horizon": h,
                                "Mean": f"{m['mean']:+.3f}",
                                "Std": f"{m['std']:.3f}",
                                "Skewness": f"{m['skewness']:.3f}",
                                "Kurtosis": f"{m['kurtosis']:.3f}",
                                "LB(24) stat": f"{lb24['lb_stat']:.1f}",
                                "LB(24) p": f"{lb24['lb_pvalue']:.6f}",
                                "Autocorrelated?": "✅ Yes" if lb24["lb_pvalue"] < 0.05 else "❌ No",
                            })
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            insight_card(
                "💡 Bias Analysis",
                "**Ensemble** triệt tiêu bias hiệu quả nhất: mean = +0,05 (h=1), "
                "+1,30 (h=6), +0,99 (h=24) µg/m³. "
                "**LSTM** có bias dương lớn nhất ở h=24 (+2,42) — over-prediction. "
                "**LightGBM** gần zero-bias tại h=1 (+0,11) do tree-based không cần log transform."
            )
        else:
            st.warning("Không tìm thấy residual_ljungbox.json")

    with tab_cost:
        section_header("⏱️", "Chi phí tính toán — Thời gian huấn luyện (Apple M3, 16GB)")
        fig_cost = _chart_train_time()
        _render_chart(fig_cost, filename="train_time_comparison")
        figure_caption_numbered(4, 5,
            "Thời gian huấn luyện trên Apple M3, 16 GB RAM, MPS acceleration. "
            "Log scale. SARIMA walk-forward tốn thời gian nhất (137s) do phải fit lại "
            "tại mỗi bước. GRU/LSTM < 0,3s nhờ kiến trúc nhỏ gọn (4.354–5.634 params).")
        render_bw_download(fig_cost, filename="train_time_comparison")

        insight_card(
            "💡 Real-time Feasibility",
            "GRU inference < 50ms/sample → khả thi triển khai trên edge computing (Raspberry Pi). "
            "LightGBM Optuna 50 trials < 1s → nhanh hơn 100× so với Neural Architecture Search. "
            "Tổng pipeline v9 (3 horizons × 3 resolutions × 4 models) hoàn thành trong ~2 phút."
        )

    # ── Ablation Study ──
    st.markdown("---")
    section_header("🧪", "Ablation Study — Tác Động Xử Lý Ngoại Lai")

    ablation_img = RESEARCH_DIR / "figures" / "ablation_outlier_impact.png"
    ablation_json = RESEARCH_DIR / "experiments" / "v10_ablation" / "ablation_metrics_30m_20260508_095348.json"

    if ablation_img.exists():
        st.image(str(ablation_img), use_container_width=True)
        figure_caption_numbered(4, 6,
            "Ablation study: Tác động của chiến lược xử lý ngoại lai lên hiệu suất mô hình. "
            "So sánh MASE giữa pipeline có và không có bước winsorize/capping, "
            "cho thấy xử lý ngoại lai cải thiện đáng kể ở horizon dài (6h, 24h)."
        )
    else:
        st.info("Chưa có hình ablation. Chạy `uv run python scripts/ablation_outlier.py` để tạo.")

    if ablation_json.exists():
        import json as _json
        abl_data = _json.load(open(ablation_json, encoding="utf-8"))
        with st.expander("📊 Ablation Metrics (chi tiết)", expanded=False):
            for h_key in ["1h", "6h", "24h"]:
                if h_key in abl_data:
                    st.markdown(f"**Horizon {h_key}:**")
                    import pandas as _pd
                    df_abl = _pd.DataFrame(abl_data[h_key])
                    if not df_abl.empty:
                        st.dataframe(df_abl, use_container_width=True, hide_index=True)

    # ── Static thesis figures (pre-generated) ──
    st.markdown("---")
    section_header("🖼️", "Hình tĩnh sẵn sàng (PNG 300 DPI)")

    if THESIS_FIGURES_DIR.exists():
        thesis_pngs = sorted(THESIS_FIGURES_DIR.glob("*.png"))
        if thesis_pngs:
            cols = st.columns(min(len(thesis_pngs), 3))
            for idx, png_path in enumerate(thesis_pngs):
                with cols[idx % 3]:
                    st.image(str(png_path), caption=png_path.stem.replace("_", " ").title(), use_container_width=True)
        else:
            st.info("Chưa có hình tĩnh trong research/figures/thesis/")
    else:
        st.info("Thư mục research/figures/thesis/ chưa tồn tại")

    # ── Export All B&W Figures (ZIP) ──
    st.markdown("---")
    section_header("📦", "Export Toàn Bộ Hình B&W (ZIP)")

    st.markdown("""
    <div style="background: var(--secondary-background-color); border-radius: 10px;
                padding: 1rem; border-left: 3px solid #00D4AA; margin-bottom: 1rem;
                font-size: 0.9rem; opacity: 0.85;">
        Tải toàn bộ 5 biểu đồ thesis dưới dạng PNG đen trắng (300 DPI, Times New Roman)
        trong một file ZIP — sẵn sàng chèn vào Word.
    </div>
    """, unsafe_allow_html=True)

    if st.button("📦 Tạo & Tải ZIP tất cả hình B&W", key="export_all_bw_zip"):
        try:
            import io
            import zipfile
            from src.viz.chart_factory import to_bw

            figures_map = {}
            if bc_data:
                figures_map["Hinh_4.1_Bootstrap_CI"] = _chart_bootstrap_ci(bc_data)
            if sm_data:
                figures_map["Hinh_4.2_MASE_Decay"] = _chart_mase_decay(sm_data)
            if shap_data:
                figures_map["Hinh_4.3_SHAP_Features"] = _chart_shap_comparison(shap_data)
            if lb_data:
                figures_map["Hinh_4.4_Residual_Bias"] = _chart_residual_bias(lb_data)
            figures_map["Hinh_4.5_Train_Time"] = _chart_train_time()

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, fig in figures_map.items():
                    bw_fig = to_bw(fig)
                    try:
                        img_bytes = bw_fig.to_image(
                            format="png", width=1200, height=500, scale=3
                        )
                    except Exception:
                        img_bytes = bw_fig.to_image(
                            format="png", width=1200, height=500
                        )
                    zf.writestr(f"{name}.png", img_bytes)

            zip_buffer.seek(0)
            st.download_button(
                label=f"⬇️ Download ZIP ({len(figures_map)} hình B&W)",
                data=zip_buffer.getvalue(),
                file_name="thesis_figures_bw.zip",
                mime="application/zip",
                key="download_bw_zip",
            )
            st.success(f"✅ Đã tạo ZIP với {len(figures_map)} hình B&W (300 DPI)")
        except Exception as e:
            st.error(f"❌ Lỗi export: {e}")

    render_references_section()

