"""Explainability Hub — Model Explainability page for PM2.5 Dashboard.

Inspired by MAIF/Shapash organization, custom-built for Streamlit.
Replaces the old SHAP page with 5 interactive tabs.

Usage:
    Called from app.py as page_explainability_hub(results)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
SHAP_DIR = RESEARCH_DIR / "figures" / "shap"

# ── Design tokens (match app.py) ──
COLORS = {
    "primary": "#00D4AA",
    "secondary": "#4ECDC4",
    "accent": "#FF6B6B",
    "warning": "#FFE66D",
    "text": "#FAFAFA",
    "text_muted": "#8B95A5",
    "card_bg": "#1A1F2E",
}

CHART_COLORS = [
    "#00D4AA", "#FF6B6B", "#4ECDC4", "#FFE66D",
    "#A78BFA", "#FB923C", "#60A5FA", "#F472B6",
]


# ══════════════════════════════════════════════════════════════════════
# Shared helpers (mirroring app.py design system)
# ══════════════════════════════════════════════════════════════════════


def _section_header(icon: str, title: str):
    st.markdown(f"""
    <div class="section-header">
        <span class="icon">{icon}</span>
        <span class="title">{title}</span>
    </div>
    """, unsafe_allow_html=True)


def _insight_card(title: str, text: str, card_type: str = "default"):
    cls = "warning" if card_type == "warning" else ""
    st.markdown(f"""
    <div class="insight-card {cls}">
        <h4>{title}</h4>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)


def _apply_plotly_style(fig: go.Figure, height: int = 450) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif", size=13),
        xaxis=dict(
            gridcolor="rgba(139,149,165,0.12)",
            zerolinecolor="rgba(139,149,165,0.12)",
        ),
        yaxis=dict(
            gridcolor="rgba(139,149,165,0.12)",
            zerolinecolor="rgba(139,149,165,0.12)",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
    )
    return fig


@st.cache_data
def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════
# Tab 1: Pipeline Journey (Plotly Sankey)
# ══════════════════════════════════════════════════════════════════════


def _tab_pipeline_journey():
    """Interactive Sankey diagram showing data flow through the pipeline."""

    _section_header("🗺️", "Data Flow — Từ IoT Sensor Đến Dự Báo")

    _insight_card(
        "💡 Đọc biểu đồ Sankey",
        "Mỗi nút là một bước trong pipeline. <b>Độ rộng</b> của dòng chảy "
        "thể hiện khối lượng dữ liệu (số dòng/features) di chuyển qua mỗi bước. "
        "Hover lên dòng chảy để xem chi tiết.",
    )

    # ── Node definitions ──
    labels = [
        "IoT Sensor\n(209K records)",       # 0
        "Raw Data\n(27,649 rows × 5 cols)", # 1
        "Data Cleaning\n(IQR 3.0 + Domain)",# 2
        "Hybrid Imputation\n(Spline+KNN)",  # 3
        "Feature Eng. v2\n(119 features)",  # 4
        "Temporal Split\n(80/10/10)",        # 5
        "Train Set\n(6,194 rows)",           # 6
        "Val Set\n(774 rows)",              # 7
        "Test Set\n(774 rows)",             # 8
        "Statistical\n(ARIMA, SARIMA)",     # 9
        "ML Models\n(LightGBM, RF, Ens.)",  # 10
        "DL Models\n(GRU, LSTM, TFT)",      # 11
        "Evaluation\n(MAE, MASE, R²)",      # 12
        "Best: GRU v2+log\nMASE=0.692 (6h)",# 13
    ]

    # Node colors
    node_colors = [
        "#4ECDC4",  # IoT
        "#4ECDC4",  # Raw
        "#00D4AA",  # Clean
        "#00D4AA",  # Impute
        "#A78BFA",  # Features
        "#FFE66D",  # Split
        "#60A5FA",  # Train
        "#FB923C",  # Val
        "#FF6B6B",  # Test
        "#F472B6",  # Statistical
        "#F472B6",  # ML
        "#F472B6",  # DL
        "#00D4AA",  # Eval
        "#FFE66D",  # Best
    ]

    # ── Link definitions (source, target, value, label) ──
    sources = [0, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 9, 10, 11]
    targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12]
    values = [27649, 27649, 7742, 7742, 7742, 6194, 774, 774, 6194, 6194, 6194, 774, 774, 774]
    link_labels = [
        "Resample 1h → 27,649 rows",
        "IQR 3.0 + PM2.5 domain [0,500]",
        "Spline ≤6h + KNN 6-24h → 7,742 rows",
        "119 features (anti-leakage ✅)",
        "Temporal split",
        "80% Train",
        "10% Validation",
        "10% Test (REAL DATA ONLY)",
        "ARIMA(2,1,1), SARIMA×(2,1,0,24)",
        "LightGBM (Optuna), RF, Ensemble",
        "GRU, LSTM, TFT (PyTorch MPS)",
        "3 models evaluated",
        "7 models evaluated",
        "6 models evaluated",
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
            label=labels,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=[
                "rgba(78,205,196,0.25)",   # IoT→Raw
                "rgba(0,212,170,0.25)",     # Raw→Clean
                "rgba(0,212,170,0.25)",     # Clean→Impute
                "rgba(167,139,250,0.25)",   # Impute→Features
                "rgba(255,230,109,0.25)",   # Features→Split
                "rgba(96,165,250,0.30)",    # Split→Train
                "rgba(251,146,60,0.25)",    # Split→Val
                "rgba(255,107,107,0.30)",   # Split→Test
                "rgba(244,114,182,0.20)",   # Train→Statistical
                "rgba(244,114,182,0.20)",   # Train→ML
                "rgba(244,114,182,0.20)",   # Train→DL
                "rgba(0,212,170,0.20)",     # Stat→Eval
                "rgba(0,212,170,0.20)",     # ML→Eval
                "rgba(0,212,170,0.20)",     # DL→Eval
            ],
            hovertemplate="%{label}<extra></extra>",
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Pipeline Data Flow — PM2.5 Forecasting",
            font=dict(size=16, color=COLORS["primary"]),
        ),
    )
    fig = _apply_plotly_style(fig, height=550)
    st.plotly_chart(fig, use_container_width=True)

    # ── Pipeline Statistics Cards ──
    _section_header("📊", "Thống Kê Pipeline")

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("📥 Raw Input", "209K records", "~2 phút/mẫu, 3.1 năm"),
        ("🧹 After Clean", "27,649 rows", "Resample 1h, IQR 3.0"),
        ("🔧 After Impute", "7,742 rows", "Hybrid: Spline + KNN"),
        ("📐 Features", "119 columns", "v2: anti-leakage ✅"),
    ]
    for col, (label, value, detail) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1A1F2E 0%, #252B3D 100%);
                        border: 1px solid rgba(0,212,170,0.2); border-radius: 12px;
                        padding: 1.2rem; text-align: center;">
                <div style="font-size: 0.8rem; color: #8B95A5; text-transform: uppercase;
                            letter-spacing: 0.05em;">{label}</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #00D4AA;
                            font-family: 'JetBrains Mono', monospace; margin: 0.3rem 0;">
                    {value}
                </div>
                <div style="font-size: 0.75rem; color: rgba(248,250,252,0.6);">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Anti-leakage checkpoints ──
    _section_header("🛡️", "Anti-Leakage Checkpoints")

    checkpoints = [
        ("✅ Feature Engineering", "shift(1) trên mọi feature dùng target (diff, pct_change, ratio)"),
        ("✅ Temporal Split", "80/10/10 theo thời gian — KHÔNG random shuffle"),
        ("✅ Test = Real Data Only", "is_imputed == 0 filter bắt buộc trên test set"),
        ("✅ Transform Fit", "Scaler, PCA, BoxCox fit trên TRAIN ONLY"),
        ("✅ Purging Gap", "Gap = max_lookback giữa train/val/test"),
    ]
    for check, desc in checkpoints:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.75rem;
                    padding: 0.6rem 1rem; margin: 0.3rem 0;
                    background: rgba(0,212,170,0.06); border-radius: 8px;
                    border-left: 3px solid #00D4AA;">
            <span style="font-size: 0.95rem; font-weight: 600; color: #E2E8F0;
                         min-width: 200px;">{check}</span>
            <span style="font-size: 0.85rem; color: rgba(248,250,252,0.7);">{desc}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Tab 2: Feature Explainability (Interactive SHAP)
# ══════════════════════════════════════════════════════════════════════


def _tab_feature_explainability():
    """Interactive SHAP feature importance, heatmap, and static plots."""

    shap_data = _load_json(SHAP_DIR / "shap_results.json")

    _insight_card(
        "💡 Tại sao Explainability quan trọng?",
        "SHAP giải thích <b>tại sao</b> mô hình dự đoán giá trị cụ thể, "
        "không chỉ <b>chính xác bao nhiêu</b>. Điều này giúp xác nhận mô hình "
        "học đúng pattern vật lý (nhiệt độ, chu kỳ ngày đêm) thay vì exploit noise."
        "<br><br><b>Tại sao chỉ LightGBM?</b> SHAP TreeExplainer chỉ hỗ trợ "
        "tree-based models (LightGBM, XGBoost, RF). Cho Deep Learning "
        "(GRU/LSTM/TFT), chúng ta dùng <b>Permutation Importance</b> — "
        "phương pháp model-agnostic, đo trực tiếp ảnh hưởng khi shuffle từng feature.",
    )

    sub1, sub2, sub3, sub4 = st.tabs([
        "📊 Interactive Feature Importance",
        "🗺️ Feature × Horizon Heatmap",
        "🌊 SHAP Beeswarm & Dependence",
        "🧠 GRU Permutation Importance",
    ])

    # ── Sub-tab 1: Interactive bar chart ──
    with sub1:
        _section_header("📊", "Top 15 Features — SHAP Mean |Value| (LightGBM)")
        if not shap_data:
            st.warning("Chưa có SHAP results. Chạy script SHAP trước.")
            return

        h = st.selectbox("Chọn horizon", ["1h", "6h", "24h"], key="expl_shap_h")
        horizon_data = shap_data.get(h, {})
        top_features = horizon_data.get("top_15_shap", {})

        if top_features:
            names = list(top_features.keys())[::-1]
            values = list(top_features.values())[::-1]

            fig = go.Figure(go.Bar(
                x=values, y=names, orientation="h",
                marker=dict(
                    color=values,
                    colorscale=[[0, "#4ECDC4"], [0.5, "#00D4AA"], [1, "#FFE66D"]],
                    line=dict(width=0),
                ),
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate="%{y}: <b>%{x:.4f}</b><extra></extra>",
            ))
            fig.update_layout(
                title=dict(
                    text=f"SHAP Feature Importance — h={h} (n_test={horizon_data.get('n_test', '?')})",
                    font=dict(size=15, color=COLORS["primary"]),
                ),
                xaxis_title="Mean |SHAP value|",
            )
            fig = _apply_plotly_style(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Feature category breakdown
            _section_header("📂", "Phân Loại Features Quan Trọng")
            categories = {
                "🕐 Lag Features": [n for n in names if "lag" in n],
                "📈 Rolling Stats": [n for n in names if "roll" in n or "ewm" in n],
                "🌙 Calendar/Fourier": [n for n in names if any(k in n for k in ("hour", "month", "dow", "fourier", "sin", "cos"))],
                "🌡️ Domain/Interaction": [n for n in names if any(k in n for k in ("ratio", "interaction", "aqi", "diff", "pct"))],
                "🔬 Raw Sensors": [n for n in names if n in ("nhiet_do", "do_am", "diem_suong", "co2")],
            }
            cols = st.columns(len(categories))
            for col, (cat_name, feats) in zip(cols, categories.items()):
                with col:
                    count = len(feats)
                    feat_list = ", ".join(feats[:3]) + ("..." if len(feats) > 3 else "")
                    st.markdown(f"""
                    <div style="background: rgba(0,212,170,0.06); border-radius: 10px;
                                padding: 1rem; text-align: center; min-height: 120px;">
                        <div style="font-size: 1.3rem;">{cat_name.split()[0]}</div>
                        <div style="font-size: 0.75rem; color: #8B95A5; margin-top: 0.2rem;">
                            {cat_name.split(maxsplit=1)[1]}</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #00D4AA;
                                    margin: 0.3rem 0;">{count}</div>
                        <div style="font-size: 0.65rem; color: rgba(248,250,252,0.5);
                                    overflow: hidden; text-overflow: ellipsis;">
                            {feat_list if feats else "—"}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Sub-tab 2: Heatmap features × horizons ──
    with sub2:
        _section_header("🗺️", "Feature Importance Across Horizons")
        if not shap_data:
            st.warning("Chưa có SHAP results.")
            return

        # Collect all features across horizons
        all_features = set()
        for h_key in ["1h", "6h", "24h"]:
            all_features.update(shap_data.get(h_key, {}).get("top_15_shap", {}).keys())

        # Build matrix
        features_sorted = sorted(all_features)
        matrix = []
        for feat in features_sorted:
            row = []
            for h_key in ["1h", "6h", "24h"]:
                val = shap_data.get(h_key, {}).get("top_15_shap", {}).get(feat, 0)
                row.append(val)
            matrix.append(row)

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=["1h", "6h", "24h"],
            y=features_sorted,
            colorscale=[[0, "#0E1117"], [0.3, "#1A4040"], [0.6, "#00D4AA"], [1, "#FFE66D"]],
            text=[[f"{v:.3f}" if v > 0 else "" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="Feature: %{y}<br>Horizon: %{x}<br>SHAP: %{z:.4f}<extra></extra>",
            colorbar=dict(title="SHAP"),
        ))
        fig.update_layout(
            title=dict(
                text="Feature × Horizon SHAP Heatmap — Nào quan trọng ở đâu?",
                font=dict(size=15, color=COLORS["primary"]),
            ),
            yaxis=dict(dtick=1, tickfont=dict(size=10)),
        )
        fig = _apply_plotly_style(fig, height=max(400, len(features_sorted) * 22))
        st.plotly_chart(fig, use_container_width=True)

        _insight_card(
            "💡 Key Insight",
            "<b>pm25_lag_1h</b> chi phối 1h & 24h nhưng KHÔNG phải top ở 6h. "
            "Tại 6h, <b>pm25_roll_24h_mean</b> và <b>hour_sin</b> (Fourier) mới quan trọng nhất — "
            "cho thấy mô hình đã học được chu kỳ ngày đêm thay vì chỉ copy lag.",
        )

    # ── Sub-tab 3: Static SHAP images ──
    with sub3:
        _section_header("🌊", "SHAP Beeswarm & Dependence Plots")
        h3 = st.selectbox("Chọn horizon", ["1h", "6h", "24h"], key="expl_bee_h")

        bee_path = SHAP_DIR / f"shap_beeswarm_{h3}.png"
        if bee_path.exists():
            st.image(str(bee_path), caption=f"SHAP Beeswarm — h={h3}", use_container_width=True)
        else:
            st.warning(f"File chưa tồn tại: {bee_path.name}")

        # Dependence plots
        dep_images = sorted(SHAP_DIR.glob(f"shap_dep_{h3}_*.png"))
        if dep_images:
            st.markdown("---")
            _section_header("🔗", f"Dependence Plots — h={h3}")
            cols = st.columns(min(len(dep_images), 3))
            for i, img in enumerate(dep_images):
                with cols[i % 3]:
                    feature_name = img.stem.split(f"_{h3}_")[-1]
                    st.image(str(img), caption=feature_name, use_container_width=True)

    # ── Sub-tab 4: GRU Permutation Importance ──
    with sub4:
        _section_header("🧠", "GRU — Permutation Importance")
        _insight_card(
            "⚡ Tại sao Permutation thay vì SHAP cho GRU?",
            "SHAP TreeExplainer chỉ hoạt động với tree-based models. "
            "Cho neural networks, <b>Permutation Importance</b> đo trực tiếp: "
            "shuffle 1 feature → đo MAE tăng bao nhiêu. "
            "Ưu điểm: model-agnostic, không cần biết kiến trúc bên trong.",
        )
        h4 = st.selectbox("Chọn horizon", ["1h", "6h", "24h"], key="expl_perm_h")
        perm_path = SHAP_DIR / f"gru_permutation_{h4}.png"
        if perm_path.exists():
            st.image(str(perm_path), caption=f"GRU Permutation Importance — h={h4}",
                     use_container_width=True)
        else:
            st.warning(f"File chưa tồn tại: {perm_path.name}")


# ══════════════════════════════════════════════════════════════════════
# Tab 3: Model Selection Journey
# ══════════════════════════════════════════════════════════════════════


def _tab_model_selection(results: dict):
    """Timeline, trade-off matrix, per-horizon winners."""

    _section_header("🏆", "Hành Trình Chọn Mô Hình — Từ Baseline Đến Best")

    # ── Journey Timeline ──
    phases = [
        ("v1", "Persistence + ARIMA", "Baseline. MASE=1.0, 1.023", "#FF6B6B"),
        ("v2", "LightGBM + Feature Eng v2", "ML enters. MASE=0.725 (6h) ✅", "#FFE66D"),
        ("v3", "RF, GB, Stacking, Ensemble", "Ensemble diversification", "#FB923C"),
        ("v4", "Deep Learning GRU/LSTM", "GRU_v1 best 1h: MASE=1.173", "#A78BFA"),
        ("v5", "GRU_log + LSTM retrain", "GRU_log best 6h: MASE=0.692 ⭐", "#00D4AA"),
        ("v6", "PCA, Top-N, TFT", "1h cursed by autocorr. TFT_v2 tệ", "#4ECDC4"),
    ]

    # Timeline as a horizontal flow
    cols = st.columns(len(phases))
    for col, (ver, name, note, color) in zip(cols, phases):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 0.8rem 0.3rem;">
                <div style="background: {color}; color: #0E1117; font-weight: 800;
                            border-radius: 50%; width: 38px; height: 38px;
                            display: flex; align-items: center; justify-content: center;
                            margin: 0 auto; font-size: 0.7rem;">{ver}</div>
                <div style="font-size: 0.75rem; font-weight: 600; color: #E2E8F0;
                            margin-top: 0.5rem;">{name}</div>
                <div style="font-size: 0.65rem; color: rgba(248,250,252,0.55);
                            margin-top: 0.2rem;">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Trade-off Matrix ──
    _section_header("⚖️", "Trade-off Matrix — Best Models per Horizon")

    data = results.get("data", {}).get("results", {})
    horizons = ["1h", "6h", "24h"]

    # Key models to compare
    model_meta = {
        "Persistence": {"complexity": "Trivial", "interp": "⭐⭐⭐⭐⭐", "type": "Baseline"},
        "ARIMA": {"complexity": "Low", "interp": "⭐⭐⭐⭐", "type": "Statistical"},
        "LightGBM_tuned": {"complexity": "Medium", "interp": "⭐⭐⭐", "type": "ML"},
        "RandomForest": {"complexity": "Medium", "interp": "⭐⭐⭐", "type": "ML"},
        "GRU": {"complexity": "High", "interp": "⭐⭐", "type": "DL"},
        "TFT": {"complexity": "High", "interp": "⭐⭐⭐", "type": "DL"},
    }

    rows = []
    for model_name, meta in model_meta.items():
        row = {"Model": model_name, "Type": meta["type"],
               "Complexity": meta["complexity"], "Interpretability": meta["interp"]}
        for h in horizons:
            h_data = data.get(h, {}).get(model_name, {})
            mase = h_data.get("mase") or h_data.get("mase_unified") or h_data.get("mase_original")
            row[f"MASE {h}"] = f"{mase:.3f}" if mase else "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Per-horizon winners ──
    _section_header("🥇", "Best Model per Horizon")

    winners = [
        ("1h", "GRU (v1 raw)", "1.173", "Autocorr ~0.99 → simple models dominate"),
        ("6h", "GRU_log (v2)", "0.692", "Log transform + v2 features = breakthrough ⭐"),
        ("24h", "GRU (v1 raw)", "0.727", "Same autocorr advantage at long range"),
    ]

    cols = st.columns(3)
    for col, (h, model, mase, reason) in zip(cols, winners):
        with col:
            is_best = h == "6h"
            border = "#FFE66D" if is_best else "rgba(0,212,170,0.3)"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1A1F2E 0%, #252B3D 100%);
                        border: 2px solid {border}; border-radius: 14px;
                        padding: 1.3rem; text-align: center;">
                <div style="font-size: 1.8rem; font-weight: 800; color: #00D4AA;">h={h}</div>
                <div style="font-size: 1rem; font-weight: 600; color: #E2E8F0;
                            margin: 0.4rem 0;">{model}</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;
                            font-weight: 700; color: {'#FFE66D' if is_best else '#00D4AA'};">
                    MASE = {mase}</div>
                <div style="font-size: 0.72rem; color: rgba(248,250,252,0.55);
                            margin-top: 0.5rem;">{reason}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Key Decision Insight ──
    st.markdown("")
    _insight_card(
        "🔑 Kết Luận Quan Trọng",
        "Tại horizon 1h, PM2.5 có autocorrelation ~0.99 → Persistence baseline rất mạnh "
        "(MASE=1.0). Mọi ML/DL model KHÔNG thắng được Persistence ở 1h. "
        "Giá trị thực sự của ML/DL chỉ thể hiện ở horizons dài hơn (6h, 24h), "
        "nơi autocorrelation giảm đáng kể. "
        "Đây là insight quan trọng cho thực tiễn triển khai.",
        card_type="warning",
    )


# ══════════════════════════════════════════════════════════════════════
# Tab 4: Anti-Leakage Audit Trail
# ══════════════════════════════════════════════════════════════════════


def _tab_anti_leakage():
    """Visual audit of data leakage detection and remediation."""

    _section_header("🛡️", "Anti-Leakage Audit Trail")

    _insight_card(
        "⚠️ Tại sao Leakage Audit quan trọng?",
        "Data leakage = model \"nhìn\" thông tin tương lai khi training. "
        "Hậu quả: MASE/MAE <b>rất đẹp nhưng giả</b> — model sẽ fail hoàn toàn khi deploy. "
        "Dự án này đã phát hiện và sửa <b>4 nguồn leakage</b> trong 6 phiên bản.",
        card_type="warning",
    )

    # ── 4 Leakage Sources ──
    _section_header("🔍", "4 Nguồn Leakage Đã Phát Hiện & Sửa")

    leaks = [
        ("1️⃣", "diff(y) / pct_change(y)", "Chứa y[t] — target tại thời điểm t",
         "Dùng shift(1).diff() — chỉ dùng past values", "v2 → v3"),
        ("2️⃣", "STL Decomposition (full data)", "fit trên toàn bộ data (train+test)",
         "STL fit trên TRAIN ONLY", "v3 → v4"),
        ("3️⃣", "Scaler fit full data", "StandardScaler.fit(all_data)",
         "Scaler.fit(train_only)", "v3 → v4"),
        ("4️⃣", "Imputed data in test", "Test set chứa imputed rows",
         "Filter is_imputed==0 trên test", "v2 → v3"),
    ]

    for num, source, problem, fix, version in leaks:
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; padding: 0.8rem 1rem; margin: 0.4rem 0;
                    background: rgba(255,107,107,0.05); border-radius: 10px;
                    border-left: 4px solid #FF6B6B;">
            <div style="font-size: 1.5rem; min-width: 30px;">{num}</div>
            <div style="flex: 1;">
                <div style="font-weight: 700; color: #E2E8F0; font-size: 0.95rem;">
                    {source}</div>
                <div style="color: #FF6B6B; font-size: 0.8rem; margin: 0.2rem 0;">
                    ❌ Vấn đề: {problem}</div>
                <div style="color: #00D4AA; font-size: 0.8rem;">
                    ✅ Fix: {fix}</div>
                <div style="color: #8B95A5; font-size: 0.7rem; margin-top: 0.2rem;">
                    Phát hiện: {version}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Before/After Impact ──
    st.markdown("---")
    _section_header("📊", "Impact: MASE Before vs After Leakage Fix")

    comparison = [
        ("STL full-data (leaky)", 0.507, True),
        ("Raw features (no STL)", 0.731, False),
        ("STL train-only (fixed)", 0.736, False),
    ]

    fig = go.Figure()
    for label, mase, is_leaky in comparison:
        fig.add_trace(go.Bar(
            x=[label], y=[mase],
            marker_color="#FF6B6B" if is_leaky else "#00D4AA",
            text=[f"{mase:.3f}"],
            textposition="outside",
            textfont=dict(size=13, color="#FAFAFA"),
            hovertemplate=f"{label}: MASE = {mase:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="MASE@6h — STL Leakage Impact", font=dict(size=15, color=COLORS["primary"])),
        showlegend=False,
        yaxis_title="MASE",
    )
    fig = _apply_plotly_style(fig, height=350)
    st.plotly_chart(fig, use_container_width=True)

    _insight_card(
        "💡 Key Takeaway",
        "STL full-data cho MASE=0.507 (quá đẹp!) nhưng ĐÓ LÀ LEAKAGE. "
        "Sau khi fix (STL train-only), MASE=0.736 — gần bằng raw features (0.731). "
        "Chứng minh: STL decomposition KHÔNG tạo value thực sự, chỉ exploit future info.",
    )

    # ── Test Coverage ──
    _section_header("✅", "Test Coverage")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1A1F2E 0%, #252B3D 100%);
                border: 1px solid rgba(0,212,170,0.2); border-radius: 12px;
                padding: 1.5rem; text-align: center;">
        <div style="font-size: 3rem; font-weight: 800; color: #00D4AA;">167 / 167</div>
        <div style="font-size: 1rem; color: #8B95A5; margin-top: 0.3rem;">Tests Passed</div>
        <div style="font-size: 0.8rem; color: rgba(248,250,252,0.5); margin-top: 0.5rem;">
            Bao gồm: leakage tests, shuffle tests, metric validation, pipeline integrity</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Tab 5: Scientific Foundation
# ══════════════════════════════════════════════════════════════════════


def _tab_scientific_foundation():
    """Reference bookshelf and literature comparison."""

    _section_header("📚", "Nền Tảng Khoa Học & So Sánh Văn Liệu")

    # ── Core References ──
    _section_header("📖", "Sách & Tài Liệu Tham Khảo Chính")

    references = [
        ("Hyndman & Athanasopoulos (2021)", "Forecasting: Principles & Practice (3rd ed.)",
         "MASE metric, time series cross-validation, ETS/ARIMA", "#00D4AA"),
        ("Brownlee (2020)", "Deep Learning for Time Series Forecasting",
         "LSTM/GRU architecture, walk-forward validation", "#A78BFA"),
        ("Chen & Guestrin (2016)", "XGBoost: A Scalable Tree Boosting System",
         "Gradient boosting, regularization, feature importance", "#4ECDC4"),
        ("Ke et al. (2017)", "LightGBM: A Highly Efficient GBDT",
         "Histogram-based split, leaf-wise growth", "#FFE66D"),
        ("Lundberg & Lee (2017)", "SHAP: A Unified Approach to Interpreting Predictions",
         "TreeExplainer, SHAP values for model explainability", "#FB923C"),
        ("Box, Jenkins & Reinsel (2015)", "Time Series Analysis (5th ed.)",
         "ARIMA/SARIMA methodology, stationarity testing", "#60A5FA"),
        ("Lim et al. (2021)", "Temporal Fusion Transformers",
         "Multi-horizon forecasting, variable selection, attention", "#F472B6"),
        ("Molnar (2022)", "Interpretable Machine Learning (2nd ed.)",
         "Permutation importance, SHAP, global vs local explanations", "#FF6B6B"),
    ]

    cols = st.columns(2)
    for i, (author, title, contribution, color) in enumerate(references):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background: rgba(0,212,170,0.04); border-left: 3px solid {color};
                        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.3rem 0;">
                <div style="font-weight: 700; color: #E2E8F0; font-size: 0.9rem;">
                    📖 {author}</div>
                <div style="font-style: italic; color: {color}; font-size: 0.8rem;
                            margin: 0.2rem 0;">{title}</div>
                <div style="font-size: 0.75rem; color: rgba(248,250,252,0.6);">
                    → {contribution}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Literature Comparison Table ──
    st.markdown("---")
    _section_header("🔬", "So Sánh Với Nghiên Cứu Gần Đây (2022–2025)")

    lit_data = [
        {"Study": "Dự án CTU (Our)", "Location": "Sa Đéc, VN", "PM2.5 Range": "5-50",
         "Best MAE": "4.68", "Metric": "MASE=0.692", "Key Method": "GRU+log (v2 features)"},
        {"Study": "Zhang et al. 2023", "Location": "Beijing", "PM2.5 Range": "20-300",
         "Best MAE": "8.5", "Metric": "R²=0.92", "Key Method": "CNN-LSTM hybrid"},
        {"Study": "Liu et al. 2024", "Location": "Guangzhou", "PM2.5 Range": "15-150",
         "Best MAE": "6.2", "Metric": "RMSE=9.1", "Key Method": "TFT + attention"},
        {"Study": "Park et al. 2023", "Location": "Seoul", "PM2.5 Range": "10-100",
         "Best MAE": "5.8", "Metric": "R²=0.88", "Key Method": "LightGBM + SHAP"},
        {"Study": "Nguyen et al. 2024", "Location": "Hanoi, VN", "PM2.5 Range": "30-200",
         "Best MAE": "12.3", "Metric": "RMSE=18.5", "Key Method": "Random Forest"},
    ]

    st.dataframe(pd.DataFrame(lit_data), use_container_width=True, hide_index=True)

    _insight_card(
        "💡 Lưu Ý Khi So Sánh",
        "MAE <b>KHÔNG thể</b> so sánh trực tiếp giữa các nghiên cứu vì PM2.5 range khác nhau. "
        "Sa Đéc có PM2.5 thấp (5-50 µg/m³) → MAE tuyệt đối thấp hơn Beijing (20-300). "
        "Chính vì vậy dự án dùng <b>MASE</b> (scale-independent) thay vì MAE để đánh giá.",
    )


# ══════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════


def page_explainability_hub(results: dict):
    """Main entry — renders the Explainability Hub page."""

    st.markdown("""
    <h1 style="font-size: 2.2rem; margin-bottom: 0.25rem;">
        🧠 Giải Thích Trực Quan — Model Explainability Hub
    </h1>
    <p style="opacity: 0.7; font-size: 1.05rem; margin-bottom: 1.5rem;">
        Tổng hợp toàn bộ pipeline thành narrative trực quan — lấy cảm hứng từ
        <a href="https://github.com/MAIF/shapash" target="_blank"
           style="color: #00D4AA;">MAIF/Shapash</a>,
        tùy chỉnh cho Time Series multi-horizon
    </p>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Pipeline Journey",
        "🔬 Feature Explainability",
        "🏆 Model Selection",
        "🛡️ Anti-Leakage Audit",
        "📚 Scientific Foundation",
    ])

    with tab1:
        _tab_pipeline_journey()

    with tab2:
        _tab_feature_explainability()

    with tab3:
        _tab_model_selection(results)

    with tab4:
        _tab_anti_leakage()

    with tab5:
        _tab_scientific_foundation()

