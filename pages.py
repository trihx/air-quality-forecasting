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
    "card_bg": "#1A1F2E",
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
    if not export_dir.exists():
        return ["GRU", "LightGBM"]

    # GRU: check for .pt files
    if any(export_dir.glob("gru_*h.pt")):
        models.append("GRU")
    # LightGBM: check for .txt files
    if any(export_dir.glob("lgbm_*h.txt")):
        models.append("LightGBM")
    # ARIMA: always available (fit nhanh trên dữ liệu gần nhất)
    models.append("ARIMA")

    # Scan for user-trained models
    user_dir = PROJECT_ROOT / "models" / "user_trained"
    if user_dir.exists():
        for f in user_dir.glob("*.pt"):
            name = f.stem  # e.g., gru_6h_20260405_123456
            label = f"GRU (user: {f.stem})"
            if label not in models:
                models.append(label)
        for f in user_dir.glob("*.txt"):
            label = f"LightGBM (user: {f.stem})"
            if label not in models:
                models.append(label)

    return models if models else ["GRU", "LightGBM"]


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


def page_forecast(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">🔮 Dự Báo PM2.5</h1>
    <p style="color: #CBD5E1;">Nhập dữ liệu cảm biến → Nhận dự báo nồng độ PM2.5</p>
    """, unsafe_allow_html=True)

    # ── Detect available models ──
    available_models = _detect_available_models()

    # ── Model & Horizon selection ──
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("🧠 Chọn mô hình", available_models, index=0)
    with col2:
        horizon = st.selectbox("⏱️ Khung dự báo", [1, 6, 24], index=1,
                               format_func=lambda x: f"{x} giờ")

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
    <div style="background: linear-gradient(135deg, #1A1F2E 0%, #1E2536 100%);
                border-radius: 14px; padding: 1.2rem 1.5rem; margin: 1rem 0;
                border: 1px solid rgba(0,212,170,0.25);">
        <div style="font-size: 0.8rem; color: #A0AEC0; text-transform: uppercase;
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
                <div style="font-size: 0.75rem; color: #A0AEC0; font-weight: 600;
                            margin-bottom: 0.3rem;">{label}</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF;
                            font-family: 'JetBrains Mono', monospace;
                            text-shadow: 0 0 12px rgba(0,212,170,0.3);">{val:.1f}</div>
                <div style="font-size: 0.72rem; color: #8B95A5; margin-top: 0.2rem;">{unit}</div>
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
    <div style="background: linear-gradient(135deg, #1A1F2E 0%, #1E2536 100%);
                border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.25);">
        <div style="font-size: 0.85rem; color: #CBD5E1;">
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


def _run_prediction(model_type: str, horizon: int, recent: pd.DataFrame) -> dict:
    """Run prediction based on model type.

    Uses MPS (Apple Silicon GPU) for GRU inference when available.
    Supports GRU, LightGBM, ARIMA.
    """
    device = _get_torch_device()

    if model_type == "GRU":
        from src.inference.predictor import GRUPredictor
        predictor = GRUPredictor(horizon)
        return predictor.predict(recent, device=device)

    elif model_type == "LightGBM":
        from src.features.builder import build_features
        from src.inference.predictor import LightGBMPredictor
        df_feat = build_features(recent)
        predictor = LightGBMPredictor(horizon)
        return predictor.predict(df_feat)

    elif model_type == "ARIMA":
        return _predict_arima(recent, horizon)

    elif model_type.startswith("GRU (user:"):
        from src.inference.predictor import GRUPredictor
        stem = model_type.split("user: ")[1].rstrip(")")
        user_path = PROJECT_ROOT / "models" / "user_trained" / f"{stem}.pt"
        predictor = GRUPredictor(horizon, model_dir=user_path.parent)
        return predictor.predict(recent, device=device)

    elif model_type.startswith("LightGBM (user:"):
        from src.features.builder import build_features
        from src.inference.predictor import LightGBMPredictor
        stem = model_type.split("user: ")[1].rstrip(")")
        user_path = PROJECT_ROOT / "models" / "user_trained" / f"{stem}.txt"
        df_feat = build_features(recent)
        predictor = LightGBMPredictor(horizon, model_dir=user_path.parent)
        return predictor.predict(df_feat)

    else:
        raise ValueError(f"Mô hình '{model_type}' chưa được hỗ trợ.")


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


def _show_forecast_result(result: dict, recent_data: pd.DataFrame):
    """Display forecast result with KPI card and chart."""
    pred = result["predicted_pm25"]
    level_label, level_color = _pm25_color(pred)

    # ── Big KPI ──
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3D 100%);
        border: 2px solid {level_color};
        border-radius: 20px; padding: 2rem; text-align: center;
        margin: 1.5rem 0;
    ">
        <div style="font-size: 0.9rem; color: #CBD5E1; text-transform: uppercase;
                    letter-spacing: 0.1em;">
            Dự Báo PM2.5 — {result['model']} ({result['horizon']}h)
        </div>
        <div style="font-size: 3.5rem; font-weight: 700;
                    font-family: 'JetBrains Mono', monospace;
                    color: {level_color}; margin: 0.5rem 0;">
            {pred:.1f} <span style="font-size: 1.2rem;">µg/m³</span>
        </div>
        <div style="font-size: 1rem; color: {level_color}; font-weight: 600;">
            {level_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart: History + Prediction point ──
    if "pm25" in recent_data.columns and len(recent_data) > 10:
        fig = go.Figure()
        history = recent_data["pm25"].tail(72)
        fig.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=history.values,
            name="Lịch sử PM2.5",
            line=dict(color=COLORS["primary"], width=2),
        ))
        # Prediction point
        fig.add_trace(go.Scatter(
            x=[len(history) + result["horizon"]],
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
    <p style="color: #CBD5E1;">So sánh giá trị thực tế và dự đoán từ các mô hình trên tập test</p>
    """, unsafe_allow_html=True)

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
    <div style="background: linear-gradient(135deg, #1A1F2E 0%, #252B3D 100%);
                border: 1px solid rgba(0,212,170,0.2); border-radius: 12px;
                padding: 1rem 1.5rem; margin: 1rem 0;">
        <span style="color: #8B95A5; font-size: 0.85rem;">
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
    <p style="color: #CBD5E1;">Lịch sử thí nghiệm và so sánh giữa các phiên bản pipeline</p>
    """, unsafe_allow_html=True)

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
    v2_features = v2.get("feature_set", {})

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
                <span style="font-size: 0.75rem; color: #94A3B8;">
                    {timestamp} · {n_models} models · parent: {parent}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.3rem 0.8rem; font-size: 0.88rem;">
                <span style="color: #94A3B8; font-weight: 600;">📦 What</span>
                <span style="color: #E2E8F0;">{what}</span>
                <span style="color: #94A3B8; font-weight: 600;">💡 Why</span>
                <span style="color: #E2E8F0;">{why}</span>
                <span style="color: #94A3B8; font-weight: 600;">📊 Result</span>
                <span style="color: #E2E8F0;">{result}</span>
                {'<span style="color: #94A3B8; font-weight: 600;">✅ Conclusion</span><span style="color: #E2E8F0;">' + conclusion + '</span>' if conclusion else ''}
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
    <p style="color: #CBD5E1;">Tùy chỉnh hyperparameters → Train → Đánh giá → Lưu mô hình</p>
    """, unsafe_allow_html=True)

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
    <div style="background: #1A1F2E; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.2);">
        <div style="font-size: 0.85rem; color: #8B95A5;">
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
            <div style="background: #1A1F2E; border-radius: 12px; padding: 1rem; margin: 1rem 0;
                        border: 1px solid rgba(0,212,170,0.2);">
                <div style="font-size: 0.85rem; color: #8B95A5;">
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
