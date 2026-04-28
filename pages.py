"""Dashboard Pages — New production features.

Contains:
  - page_forecast: 🔮 PM2.5 Forecasting with pre-filled input
  - page_actual_vs_predicted: 📊 Actual vs Predicted overlay chart
  - page_experiment_runs: 📋 Experiment history viewer
  - page_training: 🏋️ Interactive model training

Performance Optimizations:
  - Heavy data pipeline cached with @st.cache_data(ttl=600)
  - GRU inference uses MPS (Apple Silicon GPU) when available
  - Actual vs Predicted results cached per horizon
  - Chunked processing to prevent memory overflow
"""

from __future__ import annotations

# CRITICAL: Must be set BEFORE any import that loads OpenMP.
# Prevents OMP segfault (exit 139) when PyTorch + LightGBM coexist
# in the same Streamlit process on Apple Silicon.
# See LESSONS_LEARNED.md [2026-04-12].
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = PROJECT_ROOT / "research"
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"
CACHE_DIR = PROJECT_ROOT / "research" / "cache"

# Reuse design tokens from app.py
COLORS = {
    "primary": "#00D4AA",
    "accent": "#FF6B6B",
    "warning": "#FFE66D",
    "card_bg": "var(--secondary-background-color)",
    "text": "#FAFAFA",
    "text_muted": "#8B95A5",
}

CHART_COLORS = [
    "#00D4AA", "#FF6B6B", "#4ECDC4", "#FFE66D",
    "#A78BFA", "#FB923C", "#60A5FA", "#F472B6",
]

# WHO PM2.5 guidelines (µg/m³, 24h average)
WHO_LEVELS = [
    (0, 15, "Tốt", "#00D4AA"),
    (15, 25, "Trung bình", "#FFE66D"),
    (25, 50, "Kém", "#FB923C"),
    (50, 999, "Nguy hại", "#FF6B6B"),
]


def _apply_style(fig, height=450):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif", size=13),
        xaxis=dict(gridcolor="rgba(139,149,165,0.12)"),
        yaxis=dict(gridcolor="rgba(139,149,165,0.12)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
    )
    return fig


def _get_torch_device() -> str:
    """Detect best available device: MPS (Apple Silicon) > CUDA > CPU."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@st.cache_data(ttl=600, show_spinner="Đang tải và xử lý dữ liệu...")
def _cached_pipeline_data():
    """Cache the heavy data pipeline result (load → clean → impute).

    This is the most expensive operation (~10-20s on first run).
    Cached for 10 minutes to avoid re-running on every page switch.
    """
    from src.data.cleaner import (
        _clip_physical_bounds,
        _handle_outliers,
        _remove_duplicates,
        _resample,
        _set_datetime_index,
    )
    from src.data.imputer import impute_missing_data
    from src.data.loader import load_raw_data

    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")
    df_hybrid = impute_missing_data(
        df, strategy="hybrid",
        max_gap_interp=6, max_gap_ml=24, knn_neighbors=5, verbose=False,
    )
    # Free raw data immediately
    del df_raw, df
    gc.collect()
    return df_hybrid


def _pm25_color(value: float) -> tuple[str, str]:
    for lo, hi, label, color in WHO_LEVELS:
        if lo <= value < hi:
            return label, color
    return "Nguy hại", "#FF6B6B"


# ══════════════════════════════════════════════════════════════════════
# Page: Forecast
# ══════════════════════════════════════════════════════════════════════


def _detect_available_models() -> list[str]:
    """Scan models/exported/ to find which models can be used for inference."""
    models = []
    export_dir = PROJECT_ROOT / "models" / "exported"

    # GRU: check for .pt files
    if export_dir.exists() and any(export_dir.glob("gru_*h.pt")):
        models.append("GRU")
    # LightGBM: check for .txt files
    if export_dir.exists() and any(export_dir.glob("lgbm_*h.txt")):
        models.append("LightGBM")
    # Ensemble: available only if BOTH GRU and LightGBM are exported
    if "GRU" in models and "LightGBM" in models:
        models.append("Ensemble")
    # SARIMA: always available (fit on-the-fly, seasonal patterns)
    models.append("SARIMA")
    # ARIMA: always available (fit nhanh trên dữ liệu gần nhất)
    models.append("ARIMA")
    # Persistence: always available (baseline — copy y[t-1])
    models.append("Persistence")

    # Scan for user-trained models
    user_dir = PROJECT_ROOT / "models" / "user_trained"
    if user_dir.exists():
        for f in user_dir.glob("*.pt"):
            label = f"GRU (user: {f.stem})"
            if label not in models:
                models.append(label)
        for f in user_dir.glob("*.txt"):
            label = f"LightGBM (user: {f.stem})"
            if label not in models:
                models.append(label)

    return models if models else ["GRU", "LightGBM", "ARIMA"]


@st.cache_data(ttl=600)
def _load_model_rankings() -> dict:
    """Load MASE rankings per horizon from standardized_metrics.json."""
    path = RESEARCH_DIR / "experiments" / "standardized_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    rankings = {}
    for h_str, models_data in data.get("results", {}).items():
        h = int(h_str.replace("h", ""))
        sorted_models = []
        for name, info in models_data.items():
            mase = info.get("mase_unified") or info.get("mase")
            mae = info.get("mae")
            if mase is not None:
                sorted_models.append((name, round(mase, 3), round(mae, 2) if mae else None))
        sorted_models.sort(key=lambda x: x[1])
        rankings[h] = sorted_models[:5]
    return rankings


def _is_model_inferrable(name: str) -> bool:
    """Check if a model from standardized_metrics can be run for inference.

    Uses base-name matching: 'LightGBM_tuned' -> base 'LightGBM' -> inferrable.
    Ensemble variants are NOT inferrable (need multiple model exports).
    """
    base = name.split("_")[0]
    return base in {"GRU", "LightGBM", "ARIMA", "SARIMA", "Persistence"}


def _show_horizon_recommendation(horizon: int):
    """Show top-5 model ranking for selected horizon."""
    rankings = _load_model_rankings()
    if horizon not in rankings:
        return
    top5 = rankings[horizon]
    parts = []
    has_non_inferrable = False
    for name, mase, _ in top5:
        if _is_model_inferrable(name):
            parts.append(f"**{name}** ({mase})")
        else:
            parts.append(f"{name}¹ ({mase})")
            has_non_inferrable = True
    text = " → ".join(parts)
    st.caption(f"🏆 **Xếp hạng MASE tại {horizon}h:** {text}")
    if has_non_inferrable:
        st.caption("_¹ Ensemble — cần nhiều model exports, chưa hỗ trợ dự báo trực tiếp_")


@st.cache_data(ttl=300)
def _cached_sensor_preview():
    """Load latest sensor data — uses cached pipeline to avoid re-loading 209K rows."""
    try:
        df_hybrid = _cached_pipeline_data()
        return df_hybrid.tail(200)
    except Exception:
        return None


@st.cache_data(ttl=300)
def _cached_suggestion_values() -> dict:
    """Cached suggestion values — uses cached pipeline data."""
    try:
        df_hybrid = _cached_pipeline_data()
        row = df_hybrid.iloc[-1]
        return {
            "pm25": round(float(row.get("pm25", 10.0)), 1),
            "nhiet_do": round(float(row.get("nhiet_do", 28.0)), 1),
            "do_am": round(float(row.get("do_am", 75.0)), 1),
            "diem_suong": round(float(row.get("diem_suong", 24.0)), 1),
            "co2": round(float(row.get("co2", 400.0)), 1),
        }
    except Exception:
        return {
            "pm25": 10.0, "nhiet_do": 28.0, "do_am": 75.0,
            "diem_suong": 24.0, "co2": 400.0,
        }


# Column display names for sensor data
SENSOR_LABELS = {
    "pm25": ("PM2.5", "µg/m³"),
    "nhiet_do": ("Nhiệt độ", "°C"),
    "do_am": ("Độ ẩm", "%"),
    "diem_suong": ("Điểm sương", "°C"),
    "co2": ("CO₂", "ppm"),
}



# ── Mapping from standardized_metrics names → inference model key ──
# Only models with exported predictors are mapped.
# LightGBM_tuned is the exported version; LightGBM_default uses same file.
_METRICS_TO_INFERENCE = {
    "Persistence": "Persistence",
    "ARIMA": "ARIMA",
    "SARIMA": "SARIMA",
    "GRU": "GRU",
    "LightGBM_tuned": "LightGBM",
    "Ensemble_GRU": "Ensemble",
}

# Optimized weights from grid-search experiment (ensemble_20260404_204737.json).
# At 1h, Ensemble is just LightGBM (GRU weight=0), so not useful.
_ENSEMBLE_WEIGHTS = {
    1:  {"gru": 0.00, "lgbm": 1.00},
    6:  {"gru": 0.45, "lgbm": 0.55},
    24: {"gru": 0.70, "lgbm": 0.30},
}


@st.cache_data(ttl=600)
def _load_all_rankings() -> dict:
    """Load ALL model rankings per horizon (not just top 5)."""
    path = RESEARCH_DIR / "experiments" / "standardized_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    rankings = {}
    for h_str, models_data in data.get("results", {}).items():
        h = int(h_str.replace("h", ""))
        sorted_models = []
        for name, info in models_data.items():
            mase = info.get("mase_unified") or info.get("mase")
            mae = info.get("mae")
            if mase is not None:
                sorted_models.append((name, round(mase, 3), round(mae, 2) if mae else None))
        sorted_models.sort(key=lambda x: x[1])
        rankings[h] = sorted_models
    return rankings


def _get_smart_model_list(horizon: int) -> list[tuple[str, float | None, bool]]:
    """Get inferrable models sorted by MASE for the given horizon.

    Returns list of (model_key, mase, is_best) tuples.
    model_key is the clean key used by _run_prediction.
    is_best marks the top-ranked model.
    """
    rankings = _load_all_rankings()
    available = set(_detect_available_models())

    result = []
    seen_keys = set()

    # 1. Add models from rankings (sorted by MASE)
    for name, mase, _mae in rankings.get(horizon, []):
        key = _METRICS_TO_INFERENCE.get(name)
        if key and key in available and key not in seen_keys:
            seen_keys.add(key)
            result.append((key, mase, False))

    # 2. Add remaining available models not in rankings
    for model in available:
        if model not in seen_keys and not model.startswith(("GRU (user:", "LightGBM (user:")):
            result.append((model, None, False))
            seen_keys.add(model)

    # 3. Append user-trained models at the end
    for model in available:
        if model.startswith(("GRU (user:", "LightGBM (user:")):
            result.append((model, None, False))

    # Mark best model
    if result:
        result[0] = (result[0][0], result[0][1], True)

    return result


def _format_model_label(model_key: str, smart_models: list) -> str:
    """Format dropdown label: 'GRU ⭐ (MASE: 0.812)' or 'ARIMA'."""
    for key, mase, is_best in smart_models:
        if key == model_key:
            parts = []
            parts.append(key)
            if is_best:
                parts.append(" ⭐")
            if mase is not None:
                parts.append(f"  (MASE: {mase:.3f})")
            return "".join(parts)
    return model_key


def _show_smart_ranking_context(smart_models: list, selected: str):
    """Show compact ranking bar below dropdowns."""
    inferrable = [(k, m) for k, m, _ in smart_models if m is not None]
    if not inferrable:
        return

    parts = []
    for i, (key, mase) in enumerate(inferrable):
        if key == selected:
            parts.append(f"**▶ {key} ({mase})**")
        elif i == 0:
            parts.append(f"⭐ {key} ({mase})")
        else:
            parts.append(f"{key} ({mase})")

    text = " → ".join(parts)
    st.caption(f"🏆 **Xếp hạng MASE (thấp = tốt):** {text}")


def page_forecast(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">🔮 Dự Báo PM2.5</h1>
    <p style="opacity: 0.7;">Nhập dữ liệu cảm biến → Nhận dự báo nồng độ PM2.5</p>
    """, unsafe_allow_html=True)

    # ── Version-aware info cards ──
    from src.info_cards import cards_forecast, get_current_version, render_version_badge
    ver = get_current_version()
    render_version_badge(ver)
    cards_forecast(ver)

    # ── Horizon FIRST → determines model ranking ──
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.selectbox("⏱️ Khung dự báo", [1, 6, 24], index=1,
                               format_func=lambda x: f"{x} giờ")

    # ── Smart model list sorted by MASE for selected horizon ──
    smart_models = _get_smart_model_list(horizon)
    model_keys = [m[0] for m in smart_models]

    with col2:
        model_type = st.selectbox(
            "🧠 Chọn mô hình",
            model_keys,
            index=0,
            format_func=lambda k: _format_model_label(k, smart_models),
        )

    # ── Show ranking context ──
    _show_smart_ranking_context(smart_models, model_type)

    st.divider()

    # ── Input mode ──
    input_mode = st.radio(
        "📥 Phương thức nhập dữ liệu",
        ["⚡ Dữ liệu gần nhất (Demo)", "✏️ Nhập thủ công"],
        horizontal=True,
    )

    if input_mode == "⚡ Dữ liệu gần nhất (Demo)":
        _forecast_auto(model_type, horizon)
    else:
        _forecast_manual(model_type, horizon)



def _forecast_auto(model_type: str, horizon: int):
    """Forecast using latest data from dataset — show sensor preview first."""
    # ── Show sensor data preview ──
    defaults = _cached_suggestion_values()

    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--secondary-background-color) 100%);
                color: var(--text-color) !important;
                border-radius: 14px; padding: 1.2rem 1.5rem; margin: 1rem 0;
                border: 1px solid rgba(0,212,170,0.25);">
        <div style="font-size: 0.8rem; opacity: 0.65; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.8rem; font-weight: 600;">
            📡 Dữ liệu cảm biến gần nhất
        </div>
    """, unsafe_allow_html=True)

    # Sensor values in styled cards
    cols = st.columns(len(SENSOR_LABELS))
    for idx, (key, (label, unit)) in enumerate(SENSOR_LABELS.items()):
        val = defaults.get(key, 0.0)
        with cols[idx]:
            st.markdown(f"""
            <div style="text-align: center; padding: 0.6rem 0.4rem;
                        background: rgba(15, 20, 30, 0.85);
                        border-radius: 10px; border: 1px solid rgba(0,212,170,0.12);">
                <div style="font-size: 0.75rem; opacity: 0.65; font-weight: 600;
                            margin-bottom: 0.3rem;">{label}</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF;
                            font-family: 'JetBrains Mono', monospace;
                            text-shadow: 0 0 12px rgba(0,212,170,0.3);">{val:.1f}</div>
                <div style="font-size: 0.72rem; opacity: 0.65; margin-top: 0.2rem;">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Dự Báo Ngay", type="primary", use_container_width=True):
        with st.spinner(f"Đang dự báo {horizon}h bằng {model_type}..."):
            try:
                recent = _cached_sensor_preview()
                if recent is None:
                    recent = _cached_pipeline_data().tail(200)

                result = _run_prediction(model_type, horizon, recent)
                _show_forecast_result(result, recent)

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
                st.exception(e)


def _forecast_manual(model_type: str, horizon: int):
    """Forecast with user-entered values — textboxes pre-filled with latest data."""
    defaults = _cached_suggestion_values()

    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--secondary-background-color) 100%);
                border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.25);">
        <div style="font-size: 0.85rem; opacity: 0.7;">
            💡 Các giá trị bên dưới là dữ liệu gần nhất từ cảm biến. Bạn có thể sửa lại theo ý muốn.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        pm25_val = st.number_input("PM2.5 (µg/m³)", value=defaults["pm25"],
                                   min_value=0.0, max_value=500.0, step=0.5,
                                   key="manual_pm25")
    with c2:
        temp_val = st.number_input("Nhiệt độ (°C)", value=defaults["nhiet_do"],
                                   min_value=10.0, max_value=50.0, step=0.5,
                                   key="manual_temp")
    with c3:
        hum_val = st.number_input("Độ ẩm (%)", value=defaults["do_am"],
                                  min_value=0.0, max_value=100.0, step=1.0,
                                  key="manual_hum")
    with c4:
        dew_val = st.number_input("Điểm sương (°C)", value=defaults["diem_suong"],
                                  min_value=0.0, max_value=40.0, step=0.5,
                                  key="manual_dew")
    with c5:
        co2_val = st.number_input("CO₂ (ppm)", value=defaults["co2"],
                                  min_value=0.0, max_value=5000.0, step=10.0,
                                  key="manual_co2")

    if st.button("🚀 Dự Báo", type="primary", use_container_width=True):
        with st.spinner(f"Đang dự báo {horizon}h bằng {model_type}..."):
            try:
                recent = _cached_sensor_preview()
                if recent is None:
                    recent = _cached_pipeline_data().tail(200)

                # Override last row with user values
                recent = recent.copy()
                recent.iloc[-1, recent.columns.get_loc("pm25")] = pm25_val
                recent.iloc[-1, recent.columns.get_loc("nhiet_do")] = temp_val
                recent.iloc[-1, recent.columns.get_loc("do_am")] = hum_val
                recent.iloc[-1, recent.columns.get_loc("diem_suong")] = dew_val
                recent.iloc[-1, recent.columns.get_loc("co2")] = co2_val

                result = _run_prediction(model_type, horizon, recent)
                _show_forecast_result(result, recent)

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
                st.exception(e)


@st.cache_resource
def _get_gru_predictor(horizon: int, model_dir_str: str | None = None):
    from src.inference.predictor import GRUPredictor
    model_dir = Path(model_dir_str) if model_dir_str else None
    return GRUPredictor(horizon, model_dir=model_dir)

@st.cache_resource
def _get_gru_quantile_predictor(horizon: int):
    """Load GRU Quantile (CQR) predictor — returns None if model not exported yet."""
    try:
        from src.inference.predictor import GRUQuantilePredictor
        return GRUQuantilePredictor(horizon)
    except FileNotFoundError:
        return None

@st.cache_resource
def _get_lgbm_predictor(horizon: int, model_dir_str: str | None = None):
    from src.inference.predictor import LightGBMPredictor
    model_dir = Path(model_dir_str) if model_dir_str else None
    return LightGBMPredictor(horizon, model_dir=model_dir)

def _get_eval_metrics(model: str, horizon: int) -> dict:
    """Get evaluation metrics for a model+horizon combo.

    Priority: prediction_intervals JSON > standardized_metrics.json > fallback.
    """
    # ── Source 1: Prediction intervals (has confidence width) ──
    dir_path = PROJECT_ROOT / "research" / "experiments" / "prediction_intervals"
    if dir_path.exists():
        json_files = list(dir_path.glob("prediction_intervals_*.json"))
        if json_files:
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file) as f:
                data = json.load(f)

            best_match = None
            PRIORITY = {"cqr": 3, "conformal_prediction": 2, "quantile_regression": 1}
            for row in data:
                if row["model"] == model and row["horizon"] == horizon:
                    row_priority = PRIORITY.get(row["method"], 0)
                    best_priority = PRIORITY.get(best_match["method"], 0) if best_match else -1
                    if row_priority > best_priority:
                        best_match = row

            if best_match:
                return {
                    "mae": best_match["mae"],
                    "confidence_width": best_match.get("conformal_width") or best_match.get("avg_width", 0) / 2,
                    "coverage": best_match.get("coverage", 0.9),
                }

    # ── Source 2: standardized_metrics.json (accurate MAE per model+horizon) ──
    std_path = RESEARCH_DIR / "experiments" / "standardized_metrics.json"
    if std_path.exists():
        with open(std_path) as f:
            std_data = json.load(f)
        h_results = std_data.get("results", {}).get(f"{horizon}h", {})
        # Try exact match, then partial match (e.g. "GRU" matches "GRU")
        for name, info in h_results.items():
            if name == model or name.startswith(model):
                mae = info.get("mae", 5.0)
                return {
                    "mae": round(mae, 2),
                    "confidence_width": round(mae * 1.645, 1),
                    "coverage": 0.90,
                }

    # ── Source 3: Generic fallback ──
    fallback_mae = {1: 2.39, 6: 6.31, 24: 6.28}  # Persistence MAE
    mae = fallback_mae.get(horizon, 5.0)
    return {
        "mae": round(mae, 2),
        "confidence_width": round(mae * 1.645, 1),
        "coverage": 0.90,
    }


def _predict_ensemble(recent: pd.DataFrame, horizon: int) -> dict:
    """Weighted Ensemble: GRU × w_gru + LightGBM × w_lgbm.

    Weights are pre-optimized via grid-search (step=0.05).
    Source: research/experiments/ensemble/ensemble_20260404_204737.json
    """
    from src.features.builder import build_features

    w = _ENSEMBLE_WEIGHTS.get(horizon, {"gru": 0.50, "lgbm": 0.50})

    # ── GRU prediction ──
    q_predictor = _get_gru_quantile_predictor(horizon)
    if q_predictor is not None:
        gru_result = q_predictor.predict(recent)
    else:
        device = _get_torch_device()
        predictor = _get_gru_predictor(horizon)
        gru_result = predictor.predict(recent, device=device)
    gru_val = gru_result["predicted_pm25"]

    # ── LightGBM prediction ──
    df_feat = build_features(recent)
    lgbm_predictor = _get_lgbm_predictor(horizon)
    lgbm_result = lgbm_predictor.predict(df_feat)
    lgbm_val = lgbm_result["predicted_pm25"]

    # ── Weighted average ──
    ensemble_val = round(gru_val * w["gru"] + lgbm_val * w["lgbm"], 2)

    result = {
        "predicted_pm25": ensemble_val,
        "model": f"Ensemble (GRU×{w['gru']:.0%} + LightGBM×{w['lgbm']:.0%})",
        "horizon": horizon,
        "components": {
            "GRU": round(gru_val, 2),
            "LightGBM": round(lgbm_val, 2),
        },
    }

    # Propagate CQR intervals from GRU if available
    if gru_result.get("pi_method") == "cqr":
        result["pi_method"] = "cqr"
        result["pi_lower"] = gru_result.get("pi_lower", ensemble_val)
        result["pi_upper"] = gru_result.get("pi_upper", ensemble_val)
        result["pi_width"] = gru_result.get("pi_width", 0)
        result["coverage"] = gru_result.get("coverage", 0.9)
        result["quantile_raw_lower"] = gru_result.get("quantile_raw_lower", 0)
        result["quantile_raw_upper"] = gru_result.get("quantile_raw_upper", 0)
        result["conformal_adjustment"] = gru_result.get("conformal_adjustment", 0)

    return result


def _run_prediction(model_type: str, horizon: int, recent: pd.DataFrame) -> dict:
    """Run prediction based on model type.

    Uses MPS (Apple Silicon GPU) for GRU inference when available.
    Supports GRU, LightGBM, Ensemble, ARIMA, SARIMA, Persistence.

    IMPORTANT: torch is imported ONLY inside GRU branches to avoid
    OMP segfault when mixing PyTorch + LightGBM on Apple Silicon.
    See LESSONS_LEARNED.md [2026-04-12].
    """
    eval_metrics = _get_eval_metrics(model_type, horizon)

    if model_type == "Persistence":
        result = _predict_persistence(recent, horizon)

    elif model_type == "GRU":
        # Prefer CQR quantile model (provides prediction intervals)
        # Fallback to standard GRU if quantile model not exported
        q_predictor = _get_gru_quantile_predictor(horizon)
        if q_predictor is not None:
            result = q_predictor.predict(recent)
        else:
            device = _get_torch_device()
            predictor = _get_gru_predictor(horizon)
            result = predictor.predict(recent, device=device)

    elif model_type == "LightGBM":
        from src.features.builder import build_features
        df_feat = build_features(recent)
        predictor = _get_lgbm_predictor(horizon)
        result = predictor.predict(df_feat)

    elif model_type == "Ensemble":
        result = _predict_ensemble(recent, horizon)

    elif model_type == "ARIMA":
        result = _predict_arima(recent, horizon)

    elif model_type == "SARIMA":
        result = _predict_sarima(recent, horizon)

    elif model_type.startswith("GRU (user:"):
        device = _get_torch_device()
        stem = model_type.split("user: ")[1].rstrip(")")
        user_path = PROJECT_ROOT / "models" / "user_trained" / f"{stem}.pt"
        predictor = _get_gru_predictor(horizon, str(user_path.parent))
        result = predictor.predict(recent, device=device)

    elif model_type.startswith("LightGBM (user:"):
        from src.features.builder import build_features
        stem = model_type.split("user: ")[1].rstrip(")")
        user_path = PROJECT_ROOT / "models" / "user_trained" / f"{stem}.txt"
        df_feat = build_features(recent)
        predictor = _get_lgbm_predictor(horizon, str(user_path.parent))
        result = predictor.predict(df_feat)

    else:
        raise ValueError(f"Mô hình '{model_type}' chưa được hỗ trợ.")

    # Inject evaluation metrics into result
    result.update(eval_metrics)
    return result


def _predict_arima(recent: pd.DataFrame, horizon: int) -> dict:
    """Fit ARIMA on recent data and forecast."""
    from statsmodels.tsa.arima.model import ARIMA

    series = recent["pm25"].dropna().values
    # Fit ARIMA(2,1,1) — best order from experiments
    model = ARIMA(series, order=(2, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)
    pred = float(forecast[-1])

    return {
        "predicted_pm25": round(pred, 2),
        "model": "ARIMA(2,1,1)",
        "horizon": horizon,
        "timestamp": datetime.now().isoformat(),
        "input_rows": len(series),
        "last_pm25": round(float(series[-1]), 2),
    }


def _predict_persistence(recent: pd.DataFrame, horizon: int) -> dict:
    """Persistence baseline — predict PM2.5 = last known value."""
    last_pm25 = float(recent["pm25"].dropna().iloc[-1])
    return {
        "predicted_pm25": round(last_pm25, 2),
        "model": "Persistence (Baseline)",
        "horizon": horizon,
        "timestamp": datetime.now().isoformat(),
        "input_rows": len(recent),
        "last_pm25": round(last_pm25, 2),
    }


def _predict_sarima(recent: pd.DataFrame, horizon: int) -> dict:
    """Fit SARIMA on recent data and forecast."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = recent["pm25"].dropna().values
    # Best order from experiments: SARIMA(1,0,0)(2,1,0,24)
    model = SARIMAX(
        series, order=(1, 0, 0), seasonal_order=(2, 1, 0, 24),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=50)
    forecast = fitted.forecast(steps=horizon)
    pred = float(forecast[-1])

    return {
        "predicted_pm25": round(pred, 2),
        "model": "SARIMA(1,0,0)(2,1,0,24)",
        "horizon": horizon,
        "timestamp": datetime.now().isoformat(),
        "input_rows": len(series),
        "last_pm25": round(float(series[-1]), 2),
    }


def _show_forecast_result(result: dict, recent_data: pd.DataFrame):
    """Display forecast result with KPI card and chart.

    Supports CQR prediction intervals (adaptive width) when available.
    Falls back to symmetric confidence_width for non-CQR models.
    """
    pred = result["predicted_pm25"]
    level_label, level_color = _pm25_color(pred)
    is_cqr = result.get("pi_method") == "cqr"

    # ── Determine interval display ──
    if is_cqr:
        pi_lower = result.get("pi_lower", pred)
        pi_upper = result.get("pi_upper", pred)
        pi_width = result.get("pi_width", 0)
        coverage_pct = int(result.get('coverage', 0.9) * 100)
        method_label = "CQR"
        interval_text = f"[{pi_lower:.1f} — {pi_upper:.1f}] µg/m³"
    else:
        conf_width = result.get("confidence_width", 0)
        pi_lower = pred - conf_width
        pi_upper = pred + conf_width
        pi_width = conf_width * 2
        coverage_pct = int(result.get('coverage', 0.9) * 100)
        method_label = "CI"
        interval_text = f"± {conf_width:.1f} µg/m³"

    # ── Big KPI ──
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
        border: 2px solid {level_color};
        border-radius: 20px; padding: 2rem; text-align: center;
        margin: 1.5rem 0;
    ">
        <div style="font-size: 0.9rem; opacity: 0.7; text-transform: uppercase;
                    letter-spacing: 0.1em;">
            Dự Báo PM2.5 — {result['model']} ({result['horizon']}h)
        </div>
        <div style="font-size: 3.5rem; font-weight: 700;
                    font-family: 'JetBrains Mono', monospace;
                    color: {level_color}; margin: 0.5rem 0;">
            {pred:.1f} <span style="font-size: 1.2rem;">µg/m³</span>
        </div>
        <div style="font-size: 1rem; color: {level_color}; font-weight: 600; margin-bottom: 0.5rem;">
            {level_label}
        </div>
        <div style="font-size: 0.85rem; color: var(--text-color); opacity: 0.7; border-top: 1px solid rgba(128,128,128,0.2); padding-top: 0.5rem; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 0.5rem;">
            <span>🛡️ Khoảng {method_label} ({coverage_pct}%): <b>{interval_text}</b></span>
            <span>📉 Sai số MAE: <b>{result.get('mae', 0):.2f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CQR Detail expander ──
    if is_cqr:
        with st.expander("📐 Chi tiết Prediction Interval (CQR)", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Quantile thô (5%-95%)",
                          f"{result.get('quantile_raw_lower', 0):.1f} — {result.get('quantile_raw_upper', 0):.1f}")
            with c2:
                st.metric("Conformal Adjustment",
                          f"± {result.get('conformal_adjustment', 0):.2f} µg/m³")
            with c3:
                st.metric("Khoảng CQR cuối",
                          f"{pi_lower:.1f} — {pi_upper:.1f} µg/m³")
            st.caption(
                "_CQR = Conformalized Quantile Regression (Romano et al., 2019). "
                "Khoảng dự báo có chiều rộng **thích ứng** theo mức độ khó dự đoán, "
                "với đảm bảo toán học về coverage ≥ 90%._"
            )

    # ── Ensemble component detail ──
    if "components" in result:
        with st.expander("🧩 Chi tiết Ensemble Components", expanded=False):
            comps = result["components"]
            cols = st.columns(len(comps))
            for idx, (name, val) in enumerate(comps.items()):
                with cols[idx]:
                    st.metric(f"{name}", f"{val:.1f} µg/m³")
            st.caption(
                f"_Weighted Average: {result['model']}_"
            )

    # ── Chart: History + Prediction point with PI band ──
    if "pm25" in recent_data.columns and len(recent_data) > 10:
        fig = go.Figure()
        history = recent_data["pm25"].tail(72)
        fig.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=history.values,
            name="Lịch sử PM2.5",
            line=dict(color=COLORS["primary"], width=2),
        ))
        pred_x = len(history) + result["horizon"]

        # Prediction interval band (shaded area)
        if pi_width > 0:
            fig.add_trace(go.Scatter(
                x=[pred_x, pred_x],
                y=[pi_lower, pi_upper],
                mode="lines",
                name=f"Khoảng {method_label} {coverage_pct}%",
                line=dict(color=level_color, width=3),
                showlegend=True,
            ))
            # Add horizontal caps
            cap_w = 0.5
            fig.add_shape(type="line", x0=pred_x-cap_w, x1=pred_x+cap_w,
                          y0=pi_lower, y1=pi_lower,
                          line=dict(color=level_color, width=2))
            fig.add_shape(type="line", x0=pred_x-cap_w, x1=pred_x+cap_w,
                          y0=pi_upper, y1=pi_upper,
                          line=dict(color=level_color, width=2))

        # Prediction point
        fig.add_trace(go.Scatter(
            x=[pred_x],
            y=[pred],
            name=f"Dự báo ({result['horizon']}h)",
            mode="markers",
            marker=dict(size=16, color=level_color, symbol="star",
                        line=dict(width=2, color="white")),
        ))
        fig.update_layout(
            xaxis_title="Thời gian (giờ)", yaxis_title="PM2.5 (µg/m³)",
            title=f"72h Lịch Sử + Dự Báo {result['horizon']}h",
        )
        fig = _apply_style(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# Page: Actual vs Predicted (OPTIMIZED — loads from pre-computed cache)
# ══════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=3600)
def _load_avp_cache(horizon: int) -> dict | None:
    """Load pre-computed Actual vs Predicted data from cache file.

    CRITICAL: This replaces live model inference that caused SIGSEGV (exit 139).
    The heavy computation is done offline by scripts/precompute_avp.py
    and saved as lightweight JSON files (~100KB each).
    """
    cache_file = CACHE_DIR / f"avp_{horizon}h.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def page_actual_vs_predicted(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">📊 Actual vs Predicted</h1>
    <p style="opacity: 0.7;">So sánh giá trị thực tế và dự đoán từ các mô hình trên tập test</p>
    """, unsafe_allow_html=True)

    # ── Version-aware info cards ──
    from src.info_cards import cards_actual_vs_predicted, get_current_version, render_version_badge
    ver = get_current_version()
    render_version_badge(ver)
    cards_actual_vs_predicted(ver)

    horizon = st.selectbox("⏱️ Chọn horizon", [1, 6, 24], index=1,
                           format_func=lambda x: f"{x} giờ")

    # Try to load from cache first
    data = _load_avp_cache(horizon)

    if data is not None:
        _render_avp_chart(data, horizon)
    else:
        st.warning(
            f"⚠️ Chưa có dữ liệu cache cho horizon {horizon}h.\n\n"
            "Chạy lệnh sau để tạo cache:\n\n"
            "```\nuv run python scripts/precompute_avp.py\n```"
        )
        st.info(
            "💡 **Tại sao cần pre-compute?**\n\n"
            "Trang Actual vs Predicted yêu cầu chạy inference trên toàn bộ test set "
            "(~800 samples × 2 models). Quá trình này tốn ~30s và sử dụng nhiều bộ nhớ, "
            "có thể gây crash Streamlit server. Pre-compute chạy offline 1 lần và "
            "lưu kết quả (~100KB/horizon) để dashboard load tức thì."
        )


def _render_avp_chart(data: dict, horizon: int):
    """Render the Actual vs Predicted chart from cached data."""
    test_actuals = np.array(data["actuals"])
    test_persist = np.array(data["persistence"])

    # ── KPI Summary ──
    n_test = data.get("n_test", len(test_actuals))
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
                border: 1px solid rgba(0,212,170,0.2); border-radius: 12px;
                padding: 1rem 1.5rem; margin: 1rem 0;">
        <span style="opacity: 0.65; font-size: 0.85rem;">
            📊 Test samples: <b style="color:#00D4AA">{n_test}</b> (real data only) |
            Horizon: <b style="color:#00D4AA">{horizon}h</b> |
            Models: <b style="color:#00D4AA">{len(data.get('metrics', []))}</b>
        </span>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=list(range(len(test_actuals))),
        y=test_actuals,
        name="Actual",
        line=dict(color="#FAFAFA", width=2.5),
    ))

    # Persistence
    fig.add_trace(go.Scatter(
        x=list(range(len(test_persist))),
        y=test_persist,
        name="Persistence",
        line=dict(color="#8B95A5", width=1.5, dash="dash"),
    ))

    model_colors = {"GRU": "#00D4AA", "LightGBM": "#FF6B6B"}

    if data.get("gru_preds"):
        # Filter None values for clean rendering
        gru_preds = [p if p is not None else np.nan for p in data["gru_preds"]]
        fig.add_trace(go.Scatter(
            x=list(range(len(gru_preds))),
            y=gru_preds,
            name=f"GRU ({horizon}h)",
            line=dict(color=model_colors["GRU"], width=2),
        ))

    if data.get("lgbm_preds"):
        lgbm_preds = [p if p is not None else np.nan for p in data["lgbm_preds"]]
        fig.add_trace(go.Scatter(
            x=list(range(len(lgbm_preds))),
            y=lgbm_preds,
            name=f"LightGBM ({horizon}h)",
            line=dict(color=model_colors["LightGBM"], width=2),
        ))

    fig.update_layout(
        xaxis_title="Test Sample Index",
        yaxis_title="PM2.5 (µg/m³)",
        title=f"Actual vs Predicted — Horizon {horizon}h (Test Set, Real Data Only)",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    fig = _apply_style(fig, height=520)
    st.plotly_chart(fig, use_container_width=True)

    # ── Errors if any ──
    if data.get("gru_error"):
        st.warning(f"⚠️ GRU: {data['gru_error']}")
    if data.get("lgbm_error"):
        st.warning(f"⚠️ LightGBM: {data['lgbm_error']}")

    # ── Metrics summary ──
    if data.get("metrics"):
        st.markdown("### 📋 Tóm tắt Metrics")
        st.dataframe(
            pd.DataFrame(data["metrics"]),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════
# Page: Experiment Runs
# ══════════════════════════════════════════════════════════════════════


def page_experiment_runs(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">📋 Experiment Runs</h1>
    <p style="opacity: 0.7;">Lịch sử thí nghiệm và so sánh giữa các phiên bản pipeline</p>
    """, unsafe_allow_html=True)

    # ── Version-aware info cards ──
    from src.info_cards import cards_experiment_runs, get_current_version, render_version_badge
    ver = get_current_version()
    render_version_badge(ver)
    cards_experiment_runs(ver)

    tab1, tab2 = st.tabs(["📊 So Sánh Phiên Bản", "📋 Tất Cả Experiment Runs"])

    with tab1:
        _render_version_comparison()

    with tab2:
        _render_all_runs()


def _render_version_comparison():
    """Render version comparison from dashboard_runs/ snapshots."""
    runs_dir = RESEARCH_DIR / "experiments" / "dashboard_runs"
    if not runs_dir.exists() or not list(runs_dir.glob("*.json")):
        st.warning("Chưa có snapshot nào. Chạy `run_enhanced_pipeline.py` để tạo.")
        return

    # Load all snapshots
    snapshots = {}
    for jpath in sorted(runs_dir.glob("*.json")):
        try:
            with open(jpath) as f:
                data = json.load(f)
            version = data.get("version", jpath.stem)
            snapshots[version] = data
        except (json.JSONDecodeError, KeyError):
            continue

    if len(snapshots) < 2:
        st.info("Cần ít nhất 2 snapshots để so sánh. Hiện có: " + ", ".join(snapshots.keys()))
        if snapshots:
            v_name = list(snapshots.keys())[0]
            v_data = snapshots[v_name]
            st.json(v_data.get("feature_set", {}))
        return

    # ── Version selector ──
    versions = list(snapshots.keys())
    col1, col2 = st.columns(2)
    with col1:
        v1_name = st.selectbox("📌 Phiên bản cơ sở", versions, index=0)
    with col2:
        v2_idx = min(1, len(versions) - 1)
        v2_name = st.selectbox("🆕 Phiên bản mới", versions, index=v2_idx)

    if v1_name == v2_name:
        st.warning("Chọn 2 phiên bản khác nhau để so sánh.")
        return

    v1 = snapshots[v1_name]
    v2 = snapshots[v2_name]

    # ── Feature set comparison ──
    st.markdown("### 🧬 So Sánh Feature Set")
    v1_features = v1.get("feature_set", {})
    if not isinstance(v1_features, dict):
        v1_features = {"features": True} if v1_features else {}
        
    v2_features = v2.get("feature_set", {})
    if not isinstance(v2_features, dict):
        v2_features = {"features": True} if v2_features else {}

    feat_rows = []
    all_keys = sorted(set(list(v1_features.keys()) + list(v2_features.keys())))
    for key in all_keys:
        v1_val = v1_features.get(key, False)
        v2_val = v2_features.get(key, False)
        status = "✅ Mới" if v2_val and not v1_val else ("➖ Bỏ" if v1_val and not v2_val else ("✓" if v2_val else "✗"))
        feat_rows.append({
            "Feature": key,
            v1_name: "✓" if v1_val else "✗",
            v2_name: "✓" if v2_val else "✗",
            "Thay đổi": status,
        })
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

    # ── Version Info Cards (what/why/result) ──
    st.markdown("### 📋 Chi Tiết Phiên Bản")

    card_colors = {
        0: ("#4A90D9", "rgba(74,144,217,0.08)", "rgba(74,144,217,0.03)"),   # blue
        1: ("#00D4AA", "rgba(0,212,170,0.08)", "rgba(78,205,196,0.03)"),     # teal
        2: ("#A78BFA", "rgba(167,139,250,0.08)", "rgba(167,139,250,0.03)"),  # purple
        3: ("#FB923C", "rgba(251,146,60,0.08)", "rgba(251,146,60,0.03)"),    # orange
    }

    for idx, (v_name, v_data) in enumerate(snapshots.items()):
        changes = v_data.get("changes", {})
        accent, bg_start, bg_end = card_colors.get(idx, card_colors[0])
        timestamp = v_data.get("timestamp", "—")[:19]
        n_models = len(v_data.get("models_included", []))
        parent = v_data.get("parent_version", "—")

        what = changes.get("what", v_data.get("description", "—"))
        why = changes.get("why", "—")
        result = changes.get("result", "—")
        conclusion = changes.get("conclusion", "")

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {bg_start} 0%, {bg_end} 100%);
                    border-left: 4px solid {accent}; border-radius: 0 12px 12px 0;
                    padding: 1rem 1.2rem; margin: 0.8rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.1rem; font-weight: 700; color: {accent};">
                    {'📌' if idx == 0 else '🆕'} {v_name}
                </span>
                <span style="font-size: 0.75rem; opacity: 0.75;">
                    {timestamp} · {n_models} models · parent: {parent}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.3rem 0.8rem; font-size: 0.88rem;">
                <span style="opacity: 0.75; font-weight: 600;">📦 What</span>
                <span style="">{what}</span>
                <span style="opacity: 0.75; font-weight: 600;">💡 Why</span>
                <span style="">{why}</span>
                <span style="opacity: 0.75; font-weight: 600;">📊 Result</span>
                <span style="">{result}</span>
                {'<span style="opacity: 0.75; font-weight: 600;">✅ Conclusion</span><span style="">' + conclusion + '</span>' if conclusion else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── MASE Comparison Chart ──
    st.markdown("### 📊 MASE — So Sánh v1 vs v2")

    v1_results = v1.get("data", {}).get("results", {})
    v2_results = v2.get("data", {}).get("results", {})

    # Find common models across both versions
    horizons = ["1h", "6h", "24h"]
    comparison_rows = []

    for h in horizons:
        v1_h = v1_results.get(h, {})
        v2_h = v2_results.get(h, {})
        all_models = sorted(set(list(v1_h.keys()) + list(v2_h.keys())))

        for model in all_models:
            v1_m = v1_h.get(model, {})
            v2_m = v2_h.get(model, {})

            v1_mae = v1_m.get("mae", None)
            v2_mae = v2_m.get("mae", None)
            v1_mase = v1_m.get("mase", v1_m.get("mase_original", None))
            v2_mase = v2_m.get("mase", v2_m.get("mase_original", None))

            is_new = model not in v1_h
            mae_change = None
            if v1_mae and v2_mae and isinstance(v1_mae, (int, float)) and isinstance(v2_mae, (int, float)):
                mae_change = ((v2_mae - v1_mae) / v1_mae) * 100

            comparison_rows.append({
                "Horizon": h,
                "Model": model,
                f"MAE ({v1_name})": round(v1_mae, 3) if isinstance(v1_mae, (int, float)) else "—",
                f"MAE ({v2_name})": round(v2_mae, 3) if isinstance(v2_mae, (int, float)) else "—",
                f"MASE ({v1_name})": round(v1_mase, 4) if isinstance(v1_mase, (int, float)) else "—",
                f"MASE ({v2_name})": round(v2_mase, 4) if isinstance(v2_mase, (int, float)) else "—",
                "MAE Δ%": f"{mae_change:+.1f}%" if mae_change is not None else ("🆕" if is_new else "—"),
            })

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── MASE Bar Chart ──
    fig = go.Figure()

    # Collect models that exist in v2 for charting
    chart_models = []
    for h in horizons:
        for model in v2_results.get(h, {}):
            if model not in chart_models and model != "Persistence":
                chart_models.append(model)

    chart_colors = [
        "#00D4AA", "#FF6B6B", "#4ECDC4", "#FFE66D",
        "#A78BFA", "#FB923C", "#60A5FA", "#F472B6",
        "#34D399", "#F87171", "#818CF8", "#FBBF24",
    ]

    for i, model in enumerate(chart_models):
        v2_mases = []
        for h in horizons:
            v2_m = v2_results.get(h, {}).get(model, {})
            mase = v2_m.get("mase", v2_m.get("mase_original", None))
            v2_mases.append(mase if isinstance(mase, (int, float)) else None)

        fig.add_trace(go.Bar(
            name=model, x=horizons,
            y=[m if m else 0 for m in v2_mases],
            marker_color=chart_colors[i % len(chart_colors)],
            text=[f"{m:.3f}" if m else "—" for m in v2_mases],
            textposition="outside", textfont={"size": 10},
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="#FF6B6B",
                  annotation_text="Baseline (MASE = 1.0)",
                  annotation_font_color="#FF6B6B")
    fig.update_layout(
        barmode="group",
        yaxis_title="MASE (thấp hơn = tốt hơn)",
        xaxis_title="Forecast Horizon",
        title=f"MASE — {v2_name} (All Models)",
    )
    fig = _apply_style(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ── Nhắc nhở tối ưu tiếp ──
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255,230,109,0.08) 0%, rgba(255,107,107,0.04) 100%);
                border-left: 4px solid #FFE66D; border-radius: 0 12px 12px 0;
                padding: 1rem 1.2rem; margin: 1rem 0;">
        <b>🔔 Ghi nhớ tối ưu tiếp:</b><br>
        • Retrain DL (GRU/LSTM/TFT) với features mới + Fourier<br>
        • Thử CV feature (std/mean) với safeguard<br>
        • So sánh log transform vs raw target cho từng model family
    </div>
    """, unsafe_allow_html=True)


def _render_all_runs():
    """Render all experiment runs table."""
    exp_dir = RESEARCH_DIR / "experiments"
    all_jsons = sorted(exp_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not all_jsons:
        st.warning("Chưa có experiment nào.")
        return

    # ── Parse runs ──
    runs = []
    for jpath in all_jsons:
        try:
            with open(jpath) as f:
                data = json.load(f)

            if isinstance(data, dict):
                name = jpath.stem
                parent = jpath.parent.name

                if "timestamp" in data:
                    runs.append({
                        "File": jpath.name,
                        "Thư mục": parent,
                        "Timestamp": data.get("timestamp", "—"),
                        "Model": data.get("model", parent),
                        "Source": data.get("source", "script"),
                        "path": str(jpath),
                    })
                else:
                    for key in data:
                        if key.endswith("h") and isinstance(data[key], dict):
                            for model_name, metrics in data[key].items():
                                if isinstance(metrics, dict) and "mae" in metrics:
                                    runs.append({
                                        "File": jpath.name,
                                        "Thư mục": parent,
                                        "Timestamp": name,
                                        "Model": model_name,
                                        "Horizon": key,
                                        "MAE": metrics.get("mae"),
                                        "MASE": metrics.get("mase"),
                                        "Source": "script",
                                        "path": str(jpath),
                                    })
        except (json.JSONDecodeError, KeyError):
            continue

    if not runs:
        st.info("Không tìm thấy runs nào có format phù hợp.")
        return

    df_runs = pd.DataFrame(runs)

    # ── Filters ──
    col1, col2 = st.columns(2)
    with col1:
        dir_filter = st.multiselect(
            "📁 Lọc theo thư mục",
            options=sorted(df_runs["Thư mục"].unique()),
            default=sorted(df_runs["Thư mục"].unique()),
        )
    with col2:
        model_filter = st.multiselect(
            "🧠 Lọc theo model",
            options=sorted(df_runs["Model"].unique()),
            default=sorted(df_runs["Model"].unique()),
        )

    filtered = df_runs[
        df_runs["Thư mục"].isin(dir_filter) & df_runs["Model"].isin(model_filter)
    ]

    # ── Table ──
    display_cols = [c for c in filtered.columns if c != "path"]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.caption(f"Tổng: {len(filtered)} runs từ {len(all_jsons)} files")

    # ── Detail view ──
    st.divider()
    selected_file = st.selectbox(
        "🔎 Xem chi tiết JSON",
        options=[j.name for j in all_jsons],
    )
    if selected_file:
        selected_path = next((j for j in all_jsons if j.name == selected_file), None)
        if selected_path:
            with open(selected_path) as f:
                data = json.load(f)
            with st.expander(f"📄 {selected_file}", expanded=True):
                st.json(data)


# ══════════════════════════════════════════════════════════════════════
# Page: Training
# ══════════════════════════════════════════════════════════════════════


def page_training(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">🏋️ Huấn Luyện Mô Hình</h1>
    <p style="opacity: 0.7;">Tùy chỉnh hyperparameters → Train → Đánh giá → Lưu mô hình</p>
    """, unsafe_allow_html=True)

    # ── Version-aware info cards ──
    from src.info_cards import cards_training, get_current_version, render_version_badge
    ver = get_current_version()
    render_version_badge(ver)
    cards_training(ver)

    from src.training.trainer import get_default_params

    # ── Model selection ──
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("🧠 Mô hình", ["LightGBM", "GRU"], index=0,
                                  key="train_model")
    with col2:
        horizon = st.selectbox("⏱️ Horizon", [1, 6, 24], index=1,
                               format_func=lambda x: f"{x} giờ", key="train_horizon")

    st.divider()

    defaults = get_default_params(model_type)

    # ── Hyperparameter form ──
    st.markdown("""
    <div style="background: var(--secondary-background-color); color: var(--text-color) !important; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.2);">
        <div style="font-size: 0.85rem; opacity: 0.65;">
            💡 Các thông số bên dưới là cấu hình tối ưu (best params). Bạn có thể điều chỉnh trước khi huấn luyện.
        </div>
    </div>
    """, unsafe_allow_html=True)

    params = {}
    if model_type == "LightGBM":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["n_estimators"] = st.number_input("n_estimators", value=defaults["n_estimators"],
                                                     min_value=50, max_value=2000, step=50)
            params["max_depth"] = st.number_input("max_depth", value=defaults["max_depth"],
                                                  min_value=1, max_value=15, step=1)
            params["learning_rate"] = st.number_input("learning_rate", value=defaults["learning_rate"],
                                                      min_value=0.001, max_value=0.3, step=0.001,
                                                      format="%.3f")
        with c2:
            params["num_leaves"] = st.number_input("num_leaves", value=defaults["num_leaves"],
                                                   min_value=8, max_value=256, step=8)
            params["subsample"] = st.slider("subsample", 0.5, 1.0, defaults["subsample"], 0.05)
            params["colsample_bytree"] = st.slider("colsample_bytree", 0.3, 1.0,
                                                   defaults["colsample_bytree"], 0.05)
        with c3:
            params["min_child_samples"] = st.number_input("min_child_samples",
                                                          value=defaults["min_child_samples"],
                                                          min_value=5, max_value=100, step=5)
            params["reg_alpha"] = st.number_input("reg_alpha", value=defaults["reg_alpha"],
                                                  min_value=0.0, max_value=10.0, step=0.01,
                                                  format="%.3f")
            params["reg_lambda"] = st.number_input("reg_lambda", value=defaults["reg_lambda"],
                                                   min_value=0.0, max_value=10.0, step=0.01,
                                                   format="%.3f")
    else:  # GRU
        c1, c2 = st.columns(2)
        with c1:
            params["lookback"] = st.number_input("lookback (giờ)", value=defaults["lookback"],
                                                 min_value=12, max_value=168, step=12)
            params["hidden_dim"] = st.selectbox("hidden_dim", [32, 64, 128, 256],
                                                index=[32, 64, 128, 256].index(defaults["hidden_dim"]))
            params["num_layers"] = st.selectbox("num_layers", [1, 2, 3],
                                                index=[1, 2, 3].index(defaults["num_layers"]))
            params["dropout"] = st.slider("dropout", 0.0, 0.5, defaults["dropout"], 0.05)
        with c2:
            params["batch_size"] = st.selectbox("batch_size", [64, 128, 256, 512],
                                                index=[64, 128, 256, 512].index(defaults["batch_size"]))
            params["learning_rate"] = st.number_input("learning_rate", value=defaults["learning_rate"],
                                                      min_value=0.0001, max_value=0.01, step=0.0001,
                                                      format="%.4f", key="gru_lr")
            params["epochs"] = st.number_input("epochs (max)", value=defaults["epochs"],
                                               min_value=10, max_value=500, step=10)
            params["patience"] = st.number_input("early stopping patience",
                                                 value=defaults["patience"],
                                                 min_value=3, max_value=50, step=1)

    st.divider()

    # ── Training ──
    col_train, col_reset = st.columns([3, 1])
    with col_train:
        train_clicked = st.button("🚀 Bắt Đầu Huấn Luyện", type="primary",
                                  use_container_width=True)
    with col_reset:
        if st.button("🔄 Reset Params"):
            st.rerun()

    if train_clicked:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(step, total, msg):
            progress_bar.progress(step / total)
            status_text.markdown(f"**{msg}** ({step}/{total})")

        try:
            if model_type == "LightGBM":
                from src.training.trainer import LightGBMTrainer
                trainer = LightGBMTrainer(horizon, params)
            else:
                from src.training.trainer import GRUTrainer
                trainer = GRUTrainer(horizon, params)

            metrics = trainer.train(progress_callback=progress_callback)

            # ── Show results ──
            st.success(f"✅ Huấn luyện hoàn tất trong {metrics['training_time_s']:.0f}s!")

            # KPI cards
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("MAE (µg/m³)", f"{metrics['mae']:.2f}")
            col_b.metric("MASE", f"{metrics['mase']:.2f}",
                         delta="Tốt hơn Persistence" if metrics['mase'] < 1 else "Chưa vượt Persistence",
                         delta_color="normal" if metrics['mase'] < 1 else "inverse")
            col_c.metric("RMSE (µg/m³)", f"{metrics['rmse']:.2f}")
            col_d.metric("R²", f"{metrics['r2']:.3f}")

            st.markdown(f"""
            <div style="background: var(--secondary-background-color); color: var(--text-color) !important; border-radius: 12px; padding: 1rem; margin: 1rem 0;
                        border: 1px solid rgba(0,212,170,0.2);">
                <div style="font-size: 0.85rem; opacity: 0.65;">
                    📊 Persistence MAE: {metrics['persist_mae']:.2f} µg/m³ |
                    Test samples: {metrics['n_test']} |
                    ⏱️ Training time: {metrics['training_time_s']:.0f}s
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Save button ──
            st.divider()
            if st.button("💾 Lưu Mô Hình", type="secondary", use_container_width=True):
                save_path = trainer.save_model()
                st.success(f"✅ Đã lưu: `{save_path}`")
                st.balloons()

        except Exception as e:
            st.error(f"Lỗi khi huấn luyện: {e}")
            st.exception(e)
