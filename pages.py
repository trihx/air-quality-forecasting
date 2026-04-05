"""Dashboard Pages — New production features.

Contains:
  - page_forecast: 🔮 PM2.5 Forecasting with pre-filled input
  - page_actual_vs_predicted: 📊 Actual vs Predicted overlay chart
  - page_experiment_runs: 📋 Experiment history viewer
  - page_training: 🏋️ Interactive model training
"""

from __future__ import annotations

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
    """Load latest sensor data — cached 5 mins to avoid re-loading 209K rows."""
    try:
        from src.inference.predictor import get_latest_data
        return get_latest_data(200)
    except Exception:
        return None


@st.cache_data(ttl=300)
def _cached_suggestion_values() -> dict:
    """Cached suggestion values to avoid re-loading on every rerun."""
    try:
        from src.inference.predictor import get_suggestion_values
        return get_suggestion_values()
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
                    from src.inference.predictor import get_latest_data
                    recent = get_latest_data(200)

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
                    from src.inference.predictor import get_latest_data
                    recent = get_latest_data(200)

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
    """Run prediction based on model type. Supports GRU, LightGBM, ARIMA."""
    if model_type == "GRU":
        from src.inference.predictor import GRUPredictor
        predictor = GRUPredictor(horizon)
        return predictor.predict(recent)

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
        return predictor.predict(recent)

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
    from datetime import datetime
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
# Page: Actual vs Predicted
# ══════════════════════════════════════════════════════════════════════


def page_actual_vs_predicted(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">📊 Actual vs Predicted</h1>
    <p style="color: #CBD5E1;">So sánh giá trị thực tế và dự đoán từ các mô hình trên tập test</p>
    """, unsafe_allow_html=True)

    horizon = st.selectbox("⏱️ Chọn horizon", [1, 6, 24], index=1,
                           format_func=lambda x: f"{x} giờ")

    if st.button("📈 Tạo biểu đồ", type="primary"):
        with st.spinner("Đang tính toán predictions trên test set..."):
            _generate_actual_vs_predicted(horizon)


def _generate_actual_vs_predicted(horizon: int):
    """Generate actual vs predicted chart for all available models."""
    try:
        from sklearn.preprocessing import StandardScaler

        from src.data.cleaner import (
            _clip_physical_bounds,
            _handle_outliers,
            _remove_duplicates,
            _resample,
            _set_datetime_index,
        )
        from src.data.imputer import impute_missing_data
        from src.data.loader import TARGET_COL, load_raw_data
        from src.features.builder import build_features

        # Load data
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

        target = df_hybrid[TARGET_COL].values
        is_imputed = df_hybrid["is_imputed"].values
        n = len(target)
        val_end = int(n * 0.9)

        # Actual test values
        test_actuals = []
        test_persist = []
        test_indices = []
        for i in range(val_end, n - horizon):
            if is_imputed[i + horizon]:
                continue
            test_actuals.append(target[i + horizon])
            test_persist.append(target[i])
            test_indices.append(i + horizon)

        test_actuals = np.array(test_actuals)
        test_persist = np.array(test_persist)

        # Create figure
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

        model_colors = {
            "GRU": "#00D4AA",
            "LightGBM": "#FF6B6B",
        }

        # ── GRU Predictions ──
        gru_path = EXPORT_DIR / f"gru_{horizon}h.pt"
        scaler_path = EXPORT_DIR / f"scalers_{horizon}h.json"
        if gru_path.exists() and scaler_path.exists():
            import torch

            model_gru = torch.jit.load(str(gru_path), map_location="cpu")
            model_gru.eval()

            with open(scaler_path) as f:
                sc = json.load(f)

            feat_cols_dl = sc["features"]
            available = [c for c in feat_cols_dl if c in df_hybrid.columns]
            features = df_hybrid[available].values

            feat_scaler = StandardScaler()
            train_end = int(n * 0.8)
            feat_scaler.fit(features[:train_end])
            features_scaled = feat_scaler.transform(features)

            lb = sc.get("lookback", 72)
            gru_preds = []
            gru_valid = []
            for idx, i in enumerate(range(val_end, n - horizon)):
                if is_imputed[i + horizon]:
                    continue
                if i < lb:
                    gru_preds.append(np.nan)
                    gru_valid.append(idx)
                    continue
                window = features_scaled[i - lb + 1:i + 1]
                if len(window) < lb:
                    gru_preds.append(np.nan)
                    gru_valid.append(idx)
                    continue
                x = torch.FloatTensor(window).unsqueeze(0)
                with torch.no_grad():
                    pred_s = model_gru(x).item()
                pred = pred_s * sc["target_scaler_scale"] + sc["target_scaler_mean"]
                gru_preds.append(pred)

            fig.add_trace(go.Scatter(
                x=list(range(len(gru_preds))),
                y=gru_preds,
                name=f"GRU ({horizon}h)",
                line=dict(color=model_colors["GRU"], width=2),
            ))

        # ── LightGBM Predictions ──
        lgbm_path = EXPORT_DIR / f"lgbm_{horizon}h.txt"
        if lgbm_path.exists():
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(lgbm_path))
            df_feat = build_features(df_hybrid)
            feat_names_path = EXPORT_DIR / f"lgbm_{horizon}h_features.json"
            if feat_names_path.exists():
                with open(feat_names_path) as f:
                    feat_info = json.load(f)
                feat_cols = [c for c in feat_info["features"] if c in df_feat.columns]
            else:
                feat_cols = [c for c in df_feat.columns if c not in [TARGET_COL, "is_imputed"]]

            X_all = df_feat[feat_cols].values
            lgbm_preds = []
            idx_counter = 0
            for i in range(val_end, n - horizon):
                if is_imputed[i + horizon]:
                    continue
                if i < len(X_all):
                    pred = booster.predict(X_all[i:i + 1])[0]
                    lgbm_preds.append(pred)
                else:
                    lgbm_preds.append(np.nan)

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

        # ── Metrics summary ──
        st.markdown("### 📋 Tóm tắt Metrics")
        metrics_rows = [
            {"Mô hình": "Persistence",
             "MAE": f"{np.mean(np.abs(test_actuals - test_persist)):.2f}",
             "MASE": "1,00"}
        ]
        if gru_path.exists():
            gru_arr = np.array([p for p in gru_preds if not np.isnan(p)])
            gru_mae = np.mean(np.abs(test_actuals[:len(gru_arr)] - gru_arr))
            persist_mae = np.mean(np.abs(test_actuals - test_persist))
            metrics_rows.append({
                "Mô hình": "GRU",
                "MAE": f"{gru_mae:.2f}",
                "MASE": f"{gru_mae / persist_mae:.2f}",
            })
        if lgbm_path.exists():
            lgbm_arr = np.array(lgbm_preds)
            lgbm_mae = np.mean(np.abs(test_actuals[:len(lgbm_arr)] - lgbm_arr))
            persist_mae = np.mean(np.abs(test_actuals - test_persist))
            metrics_rows.append({
                "Mô hình": "LightGBM",
                "MAE": f"{lgbm_mae:.2f}",
                "MASE": f"{lgbm_mae / persist_mae:.2f}",
            })

        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════
# Page: Experiment Runs
# ══════════════════════════════════════════════════════════════════════


def page_experiment_runs(results):
    st.markdown("""
    <h1 style="font-size: 2rem;">📋 Experiment Runs</h1>
    <p style="color: #CBD5E1;">Lịch sử tất cả các lần chạy thí nghiệm</p>
    """, unsafe_allow_html=True)

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

            # Extract info based on structure
            if isinstance(data, dict):
                # Check for timestamp in filename
                name = jpath.stem
                parent = jpath.parent.name

                # Try to extract metrics
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
                    # Multi-horizon format (keys like "1h", "6h")
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
