"""PM2.5 Forecasting page with pre-filled inputs and multi-model inference."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from src.frontend.citations import cite, render_references_section
from src.frontend.components import (
    COLORS,
    PROJECT_ROOT,
    RESEARCH_DIR,
)
from src.info_cards import (
    cards_forecast,
    get_current_version,
    render_version_badge,
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

WHO_LEVELS = [
    (0, 15, "Tốt", "#00D4AA"),
    (15, 25, "Trung bình", "#FFE66D"),
    (25, 50, "Kém", "#FB923C"),
    (50, 999, "Nguy hại", "#FF6B6B"),
]

SENSOR_LABELS = {
    "pm25": ("PM2.5", "µg/m³"),
    "nhiet_do": ("Nhiệt độ", "°C"),
    "do_am": ("Độ ẩm", "%"),
    "diem_suong": ("Điểm sương", "°C"),
    "co2": ("CO₂", "ppm"),
}

_METRICS_TO_INFERENCE = {
    "Persistence": "Persistence",
    "ARIMA": "ARIMA",
    "SARIMA": "SARIMA",
    "GRU": "GRU",
    "LightGBM_tuned": "LightGBM",
    "Ensemble_GRU": "Ensemble",
}

_ENSEMBLE_WEIGHTS = {
    1: {"gru": 0.00, "lgbm": 1.00},
    6: {"gru": 0.45, "lgbm": 0.55},
    24: {"gru": 0.70, "lgbm": 0.30},
}


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
def _cached_pipeline_data() -> pd.DataFrame:
    """Cache heavy data pipeline result (load → clean → impute)."""
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
        df,
        strategy="hybrid",
        max_gap_interp=6,
        max_gap_ml=24,
        knn_neighbors=5,
        verbose=False,
    )
    del df_raw, df
    gc.collect()
    return df_hybrid


def _pm25_color(value: float) -> tuple[str, str]:
    for lo, hi, label, color in WHO_LEVELS:
        if lo <= value < hi:
            return label, color
    return "Nguy hại", "#FF6B6B"


def _detect_available_models() -> list[str]:
    """Scan models/exported/ to find which models can be used for inference."""
    models = []
    export_dir = PROJECT_ROOT / "models" / "exported"

    if export_dir.exists() and any(export_dir.glob("gru_*h.pt")):
        models.append("GRU")
    if export_dir.exists() and any(export_dir.glob("lgbm_*h.txt")):
        models.append("LightGBM")
    if "GRU" in models and "LightGBM" in models:
        models.append("Ensemble")
    models.append("SARIMA")
    models.append("ARIMA")
    models.append("Persistence")

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
def _load_model_rankings() -> dict[int, list[tuple[str, float, float | None]]]:
    """Load MASE rankings per horizon from standardized_metrics.json."""
    path = RESEARCH_DIR / "experiments" / "standardized_metrics.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rankings: dict[int, list[tuple[str, float, float | None]]] = {}
    for h_str, models_data in data.get("results", {}).items():
        h = int(h_str.replace("h", ""))
        sorted_models = []
        for name, info in models_data.items():
            from src.snapshot_adapter import extract_mase

            mase = extract_mase(info)
            mae = info.get("mae")
            if mase > 0:
                sorted_models.append(
                    (
                        name,
                        round(mase, 3),
                        round(mae, 2) if mae else None,
                    )
                )
        sorted_models.sort(key=lambda x: x[1])
        rankings[h] = sorted_models[:5]
    return rankings


def _is_model_inferrable(name: str) -> bool:
    base = name.split("_")[0]
    return base in {"GRU", "LightGBM", "ARIMA", "SARIMA", "Persistence"}


def _show_horizon_recommendation(horizon: int) -> None:
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
def _cached_sensor_preview() -> pd.DataFrame | None:
    try:
        df_hybrid = _cached_pipeline_data()
        return df_hybrid.tail(200)
    except Exception:
        return None


@st.cache_data(ttl=300)
def _cached_suggestion_values() -> dict[str, float]:
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
            "pm25": 10.0,
            "nhiet_do": 28.0,
            "do_am": 75.0,
            "diem_suong": 24.0,
            "co2": 400.0,
        }


@st.cache_data(ttl=600)
def _load_all_rankings() -> dict[int, list[tuple[str, float, float | None]]]:
    path = RESEARCH_DIR / "experiments" / "standardized_metrics.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rankings: dict[int, list[tuple[str, float, float | None]]] = {}
    for h_str, models_data in data.get("results", {}).items():
        h = int(h_str.replace("h", ""))
        sorted_models = []
        for name, info in models_data.items():
            from src.snapshot_adapter import extract_mase

            mase = extract_mase(info)
            mae = info.get("mae")
            if mase > 0:
                sorted_models.append(
                    (
                        name,
                        round(mase, 3),
                        round(mae, 2) if mae else None,
                    )
                )
        sorted_models.sort(key=lambda x: x[1])
        rankings[h] = sorted_models
    return rankings


def _get_smart_model_list(
    horizon: int,
) -> list[tuple[str, float | None, bool]]:
    rankings = _load_all_rankings()
    available = set(_detect_available_models())

    result: list[tuple[str, float | None, bool]] = []
    seen_keys = set()

    for name, mase, _mae in rankings.get(horizon, []):
        key = _METRICS_TO_INFERENCE.get(name)
        if key and key in available and key not in seen_keys:
            seen_keys.add(key)
            result.append((key, mase, False))

    for model in available:
        if model not in seen_keys and not model.startswith(
            (
                "GRU (user:",
                "LightGBM (user:",
            )
        ):
            result.append((model, None, False))
            seen_keys.add(model)

    for model in available:
        if model.startswith(("GRU (user:", "LightGBM (user:")):
            result.append((model, None, False))

    if result:
        result[0] = (result[0][0], result[0][1], True)

    return result


def _format_model_label(model_key: str, smart_models: list[tuple[str, float | None, bool]]) -> str:
    for key, mase, is_best in smart_models:
        if key == model_key:
            parts = [key]
            if is_best:
                parts.append(" ⭐")
            if mase is not None:
                parts.append(f"  (MASE: {mase:.3f})")
            return "".join(parts)
    return model_key


def _show_smart_ranking_context(smart_models: list[tuple[str, float | None, bool]], selected: str) -> None:
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


def page_forecast(results: dict[str, Any]) -> None:
    """Render interactive forecasting page."""
    st.markdown(
        """
    <h1 style="font-size: 2rem;">🔮 Dự Báo PM2.5</h1>
    <p style="opacity: 0.7;">Nhập dữ liệu cảm biến → Nhận dự báo nồng độ PM2.5</p>
    """,
        unsafe_allow_html=True,
    )

    ver = get_current_version()
    render_version_badge(ver)
    cards_forecast(ver)

    st.markdown(
        f"""
    <div style="background: var(--secondary-background-color); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.2); color: var(--text-color) !important;">
        <div style="font-size: 0.85rem; opacity: 0.65;">
            📌 Mô hình suy luận: LightGBM {cite("ke2017")} (gradient boosting),
            GRU {cite("cho2014")} (recurrent deep learning), và Weighted Ensemble {cite("peixeiro2022")}.
            Khoảng dự báo bằng Conformalized Quantile Regression (CQR) {cite("romano2019")}
            và Adaptive Conformal Inference (ACI) {cite("gibbs2021")}.
            Ngưỡng an toàn sức khỏe theo tiêu chuẩn WHO 2021 {cite("who2021")}.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col2:
        horizon = st.selectbox(
            "⏱️ Horizon dự báo",
            [1, 6, 24],
            index=1,
            format_func=lambda x: f"{x} giờ tới ({x}h)",
            key="fc_horizon",
        )

    smart_models = _get_smart_model_list(horizon)
    model_options = [m[0] for m in smart_models]

    with col1:
        model_type = st.selectbox(
            "🧠 Mô hình dự báo",
            model_options,
            index=0,
            format_func=lambda x: _format_model_label(x, smart_models),
            key="fc_model",
        )

    _show_smart_ranking_context(smart_models, model_type)

    st.divider()

    input_mode = st.radio(
        "📥 Phương thức nhập dữ liệu",
        ["⚡ Dữ liệu gần nhất (Demo)", "✏️ Nhập thủ công"],
        horizontal=True,
    )

    if input_mode == "⚡ Dữ liệu gần nhất (Demo)":
        _forecast_auto(model_type, horizon)
    else:
        _forecast_manual(model_type, horizon)

    render_references_section()


def _forecast_auto(model_type: str, horizon: int) -> None:
    """Forecast using latest data from dataset — show sensor preview first."""
    defaults = _cached_suggestion_values()

    st.markdown(
        """
    <div style="background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--secondary-background-color) 100%);
                color: var(--text-color) !important;
                border-radius: 14px; padding: 1.2rem 1.5rem; margin: 1rem 0;
                border: 1px solid rgba(0,212,170,0.25);">
        <div style="font-size: 0.95rem; font-weight: 600; color: #00D4AA; margin-bottom: 0.8rem;
                    display: flex; align-items: center; gap: 0.5rem;">
            <span>📡</span> Giá Trị Cảm Biến Gần Nhất (Auto-filled từ Dataset)
        </div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.8rem; text-align: center;">
            <div style="background: var(--background-color); border-radius: 10px; padding: 0.6rem; border: 1px solid rgba(0,212,170,0.15);">
                <div style="font-size: 0.75rem; opacity: 0.6;">PM2.5 Hiện Tại</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #00D4AA; font-family: 'JetBrains Mono', monospace;">"""
        + f"{defaults['pm25']:.1f}"
        + """</div>
                <div style="font-size: 0.7rem; opacity: 0.5;">µg/m³</div>
            </div>
            <div style="background: var(--background-color); border-radius: 10px; padding: 0.6rem; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.75rem; opacity: 0.6;">Nhiệt Độ</div>
                <div style="font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">"""
        + f"{defaults['nhiet_do']:.1f}"
        + """</div>
                <div style="font-size: 0.7rem; opacity: 0.5;">°C</div>
            </div>
            <div style="background: var(--background-color); border-radius: 10px; padding: 0.6rem; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.75rem; opacity: 0.6;">Độ Ẩm</div>
                <div style="font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">"""
        + f"{defaults['do_am']:.1f}"
        + """</div>
                <div style="font-size: 0.7rem; opacity: 0.5;">%</div>
            </div>
            <div style="background: var(--background-color); border-radius: 10px; padding: 0.6rem; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.75rem; opacity: 0.6;">Điểm Sương</div>
                <div style="font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">"""
        + f"{defaults['diem_suong']:.1f}"
        + """</div>
                <div style="font-size: 0.7rem; opacity: 0.5;">°C</div>
            </div>
            <div style="background: var(--background-color); border-radius: 10px; padding: 0.6rem; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.75rem; opacity: 0.6;">CO₂</div>
                <div style="font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">"""
        + f"{defaults['co2']:.0f}"
        + """</div>
                <div style="font-size: 0.7rem; opacity: 0.5;">ppm</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 Xem 24 giờ dữ liệu gần nhất", expanded=False):
        df_preview = _cached_sensor_preview()
        if df_preview is not None:
            display_cols = [c for c in ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"] if c in df_preview.columns]
            last_24 = df_preview[display_cols].tail(24).copy()
            last_24.columns = [
                f"{SENSOR_LABELS.get(c, (c, ''))[0]} ({SENSOR_LABELS.get(c, ('', ''))[1]})" for c in display_cols
            ]
            st.dataframe(last_24, use_container_width=True)

    if st.button("🚀 Chạy Dự Báo", type="primary", use_container_width=True, key="btn_auto"):
        with st.spinner("Đang chạy mô hình..."):
            try:
                recent = _cached_pipeline_data().tail(200)
                result = _run_prediction(model_type, horizon, recent)
                _show_forecast_result(result, recent)
            except Exception as e:
                st.error(f"❌ Lỗi dự báo: {e}")
                st.exception(e)


def _forecast_manual(model_type: str, horizon: int) -> None:
    """Manual input form with pre-filled default values."""
    defaults = _cached_suggestion_values()

    st.markdown("##### ✏️ Nhập giá trị cảm biến tại thời điểm hiện tại:")

    col_a, col_b = st.columns(2)
    with col_a:
        pm25 = st.number_input(
            "🌫️ PM2.5 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=float(defaults["pm25"]),
            step=0.5,
            format="%.1f",
        )
        nhiet_do = st.number_input(
            "🌡️ Nhiệt độ (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=float(defaults["nhiet_do"]),
            step=0.5,
            format="%.1f",
        )
        do_am = st.number_input(
            "💧 Độ ẩm (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults["do_am"]),
            step=1.0,
            format="%.1f",
        )

    with col_b:
        diem_suong = st.number_input(
            "🌫️ Điểm sương (°C)",
            min_value=-10.0,
            max_value=40.0,
            value=float(defaults["diem_suong"]),
            step=0.5,
            format="%.1f",
        )
        co2 = st.number_input(
            "🏭 CO₂ (ppm)",
            min_value=300.0,
            max_value=2000.0,
            value=float(defaults["co2"]),
            step=10.0,
            format="%.0f",
        )

    if st.button(
        "🚀 Chạy Dự Báo",
        type="primary",
        use_container_width=True,
        key="btn_manual",
    ):
        with st.spinner("Đang chạy mô hình..."):
            try:
                recent = _cached_pipeline_data().tail(200).copy()
                recent.iloc[-1, recent.columns.get_loc("pm25")] = pm25
                recent.iloc[-1, recent.columns.get_loc("nhiet_do")] = nhiet_do
                recent.iloc[-1, recent.columns.get_loc("do_am")] = do_am
                recent.iloc[-1, recent.columns.get_loc("diem_suong")] = diem_suong
                recent.iloc[-1, recent.columns.get_loc("co2")] = co2

                result = _run_prediction(model_type, horizon, recent)
                _show_forecast_result(result, recent)
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
                st.exception(e)


@st.cache_resource
def _get_gru_predictor(horizon: int, model_dir_str: str | None = None) -> Any:
    from src.inference.predictor import GRUPredictor

    model_dir = Path(model_dir_str) if model_dir_str else None
    return GRUPredictor(horizon, model_dir=model_dir)


@st.cache_resource
def _get_gru_quantile_predictor(horizon: int) -> Any:
    """Load GRU Quantile (CQR) predictor — returns None if model not exported yet."""
    try:
        from src.inference.predictor import GRUQuantilePredictor

        return GRUQuantilePredictor(horizon)
    except FileNotFoundError:
        return None


@st.cache_resource
def _get_lgbm_predictor(horizon: int, model_dir_str: str | None = None) -> Any:
    from src.inference.predictor import LightGBMPredictor

    model_dir = Path(model_dir_str) if model_dir_str else None
    return LightGBMPredictor(horizon, model_dir=model_dir)


def _get_eval_metrics(model: str, horizon: int) -> dict[str, Any]:
    """Get evaluation metrics for a model+horizon combo."""
    dir_path = PROJECT_ROOT / "research" / "experiments" / "prediction_intervals"
    if dir_path.exists():
        json_files = list(dir_path.glob("prediction_intervals_*.json"))
        if json_files:
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file, encoding="utf-8") as f:
                data = json.load(f)

            best_match = None
            priority = {
                "cqr": 3,
                "conformal_prediction": 2,
                "quantile_regression": 1,
            }
            for row in data:
                if row["model"] == model and row["horizon"] == horizon:
                    row_priority = priority.get(row["method"], 0)
                    best_priority = priority.get(best_match["method"], 0) if best_match else -1
                    if row_priority > best_priority:
                        best_match = row

            if best_match:
                return {
                    "mae": best_match["mae"],
                    "confidence_width": best_match.get("conformal_width") or best_match.get("avg_width", 0) / 2,
                    "coverage": best_match.get("coverage", 0.9),
                }

    std_path = RESEARCH_DIR / "experiments" / "standardized_metrics.json"
    if std_path.exists():
        with open(std_path, encoding="utf-8") as f:
            std_data = json.load(f)
        h_results = std_data.get("results", {}).get(f"{horizon}h", {})
        for name, info in h_results.items():
            if name == model or name.startswith(model):
                mae = info.get("mae", 5.0)
                return {
                    "mae": round(mae, 2),
                    "confidence_width": round(mae * 1.645, 1),
                    "coverage": 0.90,
                }

    fallback_mae = {1: 2.39, 6: 6.31, 24: 6.28}
    mae = fallback_mae.get(horizon, 5.0)
    return {
        "mae": round(mae, 2),
        "confidence_width": round(mae * 1.645, 1),
        "coverage": 0.90,
    }


def _predict_ensemble(recent: pd.DataFrame, horizon: int) -> dict[str, Any]:
    from src.features.builder import build_features

    w = _ENSEMBLE_WEIGHTS.get(horizon, {"gru": 0.50, "lgbm": 0.50})

    q_predictor = _get_gru_quantile_predictor(horizon)
    if q_predictor is not None:
        gru_result = q_predictor.predict(recent)
    else:
        device = _get_torch_device()
        predictor = _get_gru_predictor(horizon)
        gru_result = predictor.predict(recent, device=device)
    gru_val = gru_result["predicted_pm25"]

    lgbm_val = None
    try:
        df_feat = build_features(recent)
        lgbm_predictor = _get_lgbm_predictor(horizon)
        lgbm_result = lgbm_predictor.predict(df_feat)
        lgbm_val = lgbm_result["predicted_pm25"]
    except Exception as e:
        logger.warning(f"LightGBM prediction failed ({e}), falling back to GRU prediction for Ensemble.")
        lgbm_val = gru_val

    gc.collect()

    ensemble_val = round(gru_val * w["gru"] + lgbm_val * w["lgbm"], 2)

    result: dict[str, Any] = {
        "predicted_pm25": ensemble_val,
        "model": f"Ensemble (GRU×{w['gru']:.0%} + LightGBM×{w['lgbm']:.0%})",
        "horizon": horizon,
        "components": {
            "GRU": round(gru_val, 2),
            "LightGBM": round(lgbm_val, 2),
        },
    }

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


def _run_prediction(model_type: str, horizon: int, recent: pd.DataFrame) -> dict[str, Any]:
    try:
        from src.frontend.api_client import APIClient

        client = APIClient()
        api_result = client.predict(horizon=horizon, model_name=model_type.lower())
        if isinstance(api_result, dict) and "error" not in api_result:
            return api_result
    except Exception:  # noqa: S110
        pass

    eval_metrics = _get_eval_metrics(model_type, horizon)

    if model_type == "Persistence":
        result = _predict_persistence(recent, horizon)

    elif model_type == "GRU":
        q_predictor = _get_gru_quantile_predictor(horizon)
        if q_predictor is not None:
            result = q_predictor.predict(recent)
        else:
            device = _get_torch_device()
            predictor = _get_gru_predictor(horizon)
            result = predictor.predict(recent, device=device)

    elif model_type == "LightGBM":
        try:
            from src.features.builder import build_features

            df_feat = build_features(recent)
            predictor = _get_lgbm_predictor(horizon)
            result = predictor.predict(df_feat)
        except Exception as e:
            st.warning(f"⚠️ Thư viện C++ OpenMP (libgomp) chưa nạp sẵn: {e}. Hệ thống tự động chuyển sang mô hình GRU.")
            device = _get_torch_device()
            predictor = _get_gru_predictor(horizon)
            result = predictor.predict(recent, device=device)
            result["model"] = "GRU (Fallback từ LightGBM)"

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

    result.update(eval_metrics)
    return result


def _predict_arima(recent: pd.DataFrame, horizon: int) -> dict[str, Any]:
    from statsmodels.tsa.arima.model import ARIMA

    series = recent["pm25"].dropna().values
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


def _predict_persistence(recent: pd.DataFrame, horizon: int) -> dict[str, Any]:
    last_pm25 = float(recent["pm25"].dropna().iloc[-1])
    return {
        "predicted_pm25": round(last_pm25, 2),
        "model": "Persistence (Baseline)",
        "horizon": horizon,
        "timestamp": datetime.now().isoformat(),
        "input_rows": len(recent),
        "last_pm25": round(last_pm25, 2),
    }


def _predict_sarima(recent: pd.DataFrame, horizon: int) -> dict[str, Any]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = recent["pm25"].dropna().values
    model = SARIMAX(
        series,
        order=(1, 0, 0),
        seasonal_order=(2, 1, 0, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
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


def _show_forecast_result(result: dict[str, Any], recent_data: pd.DataFrame) -> None:
    pred = result["predicted_pm25"]
    level_label, level_color = _pm25_color(pred)
    is_cqr = result.get("pi_method") == "cqr"

    if is_cqr:
        pi_lower = result.get("pi_lower", pred)
        pi_upper = result.get("pi_upper", pred)
        pi_width = result.get("pi_width", 0)
        coverage_pct = int(result.get("coverage", 0.9) * 100)
        method_label = "CQR"
        interval_text = f"[{pi_lower:.1f} — {pi_upper:.1f}] µg/m³"
    else:
        conf_width = result.get("confidence_width", 0)
        pi_lower = pred - conf_width
        pi_upper = pred + conf_width
        pi_width = conf_width * 2
        coverage_pct = int(result.get("coverage", 0.9) * 100)
        method_label = "CI"
        interval_text = f"± {conf_width:.1f} µg/m³"

    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
        border: 2px solid {level_color};
        border-radius: 20px; padding: 2rem; text-align: center;
        margin: 1.5rem 0;
    ">
        <div style="font-size: 0.9rem; opacity: 0.7; text-transform: uppercase;
                    letter-spacing: 0.1em;">
            Dự Báo PM2.5 — {result["model"]} ({result["horizon"]}h)
        </div>
        <div style="font-size: 3.5rem; font-weight: 700;
                    font-family: 'JetBrains Mono', monospace;
                    color: {level_color}; margin: 0.5rem 0;
                    text-shadow: 1px 1px 3px rgba(0,0,0,0.3);">
            {pred:.1f} <span style="font-size: 1.2rem;">µg/m³</span>
        </div>
        <div style="font-size: 1rem; color: {level_color}; font-weight: 600; margin-bottom: 0.5rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
            {level_label}
        </div>
        <div style="font-size: 0.85rem; color: var(--text-color); opacity: 0.7; border-top: 1px solid rgba(128,128,128,0.2); padding-top: 0.5rem; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 0.5rem;">
            <span>🛡️ Khoảng {method_label} ({coverage_pct}%): <b>{interval_text}</b></span>
            <span>📉 Sai số MAE: <b>{result.get("mae", 0):.2f}</b></span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if is_cqr:
        with st.expander("📐 Chi tiết Prediction Interval (CQR)", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Quantile thô (5%-95%)",
                    f"{result.get('quantile_raw_lower', 0):.1f} — {result.get('quantile_raw_upper', 0):.1f}",
                )
            with c2:
                st.metric(
                    "Conformal Adjustment",
                    f"± {result.get('conformal_adjustment', 0):.2f} µg/m³",
                )
            with c3:
                st.metric(
                    "Khoảng CQR cuối",
                    f"{pi_lower:.1f} — {pi_upper:.1f} µg/m³",
                )
            st.caption(
                "_CQR = Conformalized Quantile Regression (Romano et al., 2019). "
                "Khoảng dự báo có chiều rộng **thích ứng** theo mức độ khó dự đoán, "
                "với đảm bảo toán học về coverage ≥ 90%._"
            )

    if "components" in result:
        with st.expander("🧩 Chi tiết Ensemble Components", expanded=False):
            comps = result["components"]
            cols = st.columns(len(comps))
            for idx, (name, val) in enumerate(comps.items()):
                with cols[idx]:
                    st.metric(f"{name}", f"{val:.1f} µg/m³")
            st.caption(f"_Weighted Average: {result['model']}_")

    if "pm25" in recent_data.columns and len(recent_data) > 10:
        fig = _chart(
            xaxis_title="Thời gian (giờ)",
            yaxis_title="PM2.5 (µg/m³)",
            height=380,
        )
        history = recent_data["pm25"].tail(72)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(history))),
                y=history.values,
                name="Lịch sử PM2.5",
                line={"color": COLORS["primary"], "width": 2},
            )
        )
        pred_x = len(history) + result["horizon"]

        if pi_width > 0:
            fig.add_trace(
                go.Scatter(
                    x=[pred_x, pred_x],
                    y=[pi_lower, pi_upper],
                    mode="lines",
                    name=f"Khoảng {method_label} {coverage_pct}%",
                    line={"color": level_color, "width": 3},
                    showlegend=True,
                )
            )
            cap_w = 0.5
            fig.add_shape(
                type="line",
                x0=pred_x - cap_w,
                x1=pred_x + cap_w,
                y0=pi_lower,
                y1=pi_lower,
                line={"color": level_color, "width": 2},
            )
            fig.add_shape(
                type="line",
                x0=pred_x - cap_w,
                x1=pred_x + cap_w,
                y0=pi_upper,
                y1=pi_upper,
                line={"color": level_color, "width": 2},
            )

        fig.add_trace(
            go.Scatter(
                x=[pred_x],
                y=[pred],
                name=f"Dự báo ({result['horizon']}h)",
                mode="markers",
                marker={
                    "size": 16,
                    "color": level_color,
                    "symbol": "star",
                    "line": {"width": 2, "color": "white"},
                },
            )
        )
        _render_chart(fig, filename="forecast_history")
        _caption(f"Lịch sử 72 giờ và dự báo PM2.5 tại horizon {result['horizon']}h")
