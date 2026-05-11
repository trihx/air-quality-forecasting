"""
Pipeline Walkthrough page for PM2.5 Forecasting Dashboard.

Shows the complete research workflow step-by-step:
  1. Data Collection → 2. Cleaning → 3. EDA → 4. Feature Engineering
  5. Imputation → 6. Modeling & Evaluation → 7. Results & Conclusions

Uses pre-computed results for instant display + selective live demos.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.frontend.citations import cite, render_references_section
from src.viz.chart_factory import chart as _chart, render_chart as _render_chart, add_baseline

PROJECT_ROOT = Path(__file__).resolve().parent if Path(__file__).resolve().parent.name != "src" else Path(__file__).resolve().parent.parent

# Try to find project root properly
for candidate in [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
    if (candidate / "app.py").exists():
        PROJECT_ROOT = candidate
        break


def _load_standardized_metrics() -> dict:
    """Load pre-computed standardized metrics."""
    try:
        from src.snapshot_adapter import load_all_normalized
        snapshots = load_all_normalized()
        if "v9_multi_resolution" in snapshots:
            return snapshots["v9_multi_resolution"]
    except ImportError:
        pass
        
    path = PROJECT_ROOT / "research" / "experiments" / "standardized_metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(ttl=3600)
def _get_pipeline_metrics() -> dict:
    import os
    from datetime import datetime
    processed = PROJECT_ROOT / "dataset" / "processed"
    datasets = [
        ("marts_features.csv", "1h"),
        ("marts_features_30m.csv", "30m"),
        ("marts_features_15m.csv", "15m"),
    ]
    resolutions = {}
    features_count = 0
    for filename, label in datasets:
        path = processed / filename
        if not path.exists(): continue
        try:
            with open(path, encoding="utf-8") as f:
                header = f.readline()
                cols = len(header.strip().split(","))
                rows = sum(1 for _ in f)
            resolutions[label] = {"rows": rows, "cols": cols}
            if label == "1h": features_count = cols
        except Exception: continue
    return {"resolutions": resolutions, "features_count": features_count}

def _render_custom_metric(label, value, icon=""):
    """Render a custom metric card that prevents truncation and supports wrapping."""
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; background: var(--secondary-background-color); 
                    border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; padding: 1rem; height: 100%;">
            <span style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 0.3rem;">{icon} {label}</span>
            <span style="font-size: 1.25rem; font-weight: 600; line-height: 1.4; word-break: break-word; white-space: normal;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def _step_data_collection():
    """Step 1: Data Collection."""
    st.markdown(
        """
    <div style="background: var(--secondary-background-color); color: var(--text-color) !important;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.2);">
        <div style="font-size: 1.3rem; font-weight: 700; color: #00D4AA;">
            📥 Bước 1: Thu Thập Dữ Liệu
        </div>
        <div style="opacity: 0.7; margin-top: 0.5rem;">
            Dữ liệu IoT sensor thu thập liên tục tại Sa Đéc, Đồng Tháp
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1: _render_custom_metric("Tổng Records", "209,397", "📦")
    with col2: _render_custom_metric("Thời gian", "3.1 năm", "📅")
    with col3: _render_custom_metric("Tần suất", "~2 phút/lần", "⏱️")
    with col4: _render_custom_metric("Biến số", "5 (PM2.5, Nhiệt độ, Độ ẩm, Điểm sương, CO₂)", "📊")

    st.markdown("#### 📋 Mô tả biến")
    var_data = {
        "Biến": ["PM2.5", "Nhiệt độ", "Độ ẩm", "Điểm sương", "CO₂"],
        "Đơn vị": ["µg/m³", "°C", "%", "°C", "ppm"],
        "Vai trò": ["🎯 Target", "Feature", "Feature", "Feature", "Feature"],
    }
    st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)

    # Try to show sample raw data
    try:
        from src.data.loader import load_raw_data
        with st.expander("👀 Xem dữ liệu thô (100 dòng đầu)", expanded=False):
            df_raw = load_raw_data()
            st.dataframe(df_raw.head(100), use_container_width=True)
            st.caption(f"Tổng: {len(df_raw):,} dòng × {len(df_raw.columns)} cột")
    except Exception:
        st.info("💡 Dữ liệu thô sẽ hiển thị khi dataset có sẵn")


def _step_data_cleaning():
    """Step 2: Data Cleaning."""
    st.markdown(
        """
    <div style="background: var(--secondary-background-color); color: var(--text-color) !important;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid rgba(236,72,153,0.2);">
        <div style="font-size: 1.3rem; font-weight: 700; color: #EC4899;">
            🧹 Bước 2: Làm sạch Dữ Liệu
        </div>
        <div style="opacity: 0.7; margin-top: 0.5rem;">
            Xử lý outliers, duplicates, và domain constraints
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Cleaning pipeline steps
    steps = [
        ("1️⃣ Xóa duplicates", "Loại bỏ bản ghi trùng lặp theo timestamp"),
        (f"2️⃣ Domain Bounds {cite('who2021')}", "PM2.5 ∈ [0, 500] µg/m³ — Cắt ngưỡng vật lý theo WHO guidelines (AQI limit)"),
        (f"3️⃣ Outlier detection", "IQR 3.0 cho các biến môi trường phụ (Nhiệt độ, v.v.). ĐẶC BIỆT: KHÔNG DÙNG cho PM2.5 để tránh bẫy 'Outlier Removal Trap'."),
        (f"4️⃣ Resampling {cite('barkjohn2021')}", "Từ ~2 phút → đa độ phân giải (15m, 30m, 1h) bằng mean aggregation. v9 phát hiện 30m là tối ưu."),
    ]

    for title, desc in steps:
        st.markdown(
            f"""
        <div style="background: rgba(236,72,153,0.05); border-left: 3px solid #EC4899;
                    padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
            <strong>{title}</strong><br>
            <span style="opacity: 0.7; font-size: 0.9rem;">{desc}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns(3)
    metrics = _get_pipeline_metrics()
    c_1h = f"~{metrics['resolutions'].get('1h', {}).get('rows', 27649):,} (1h)"
    c_30m = f"~{metrics['resolutions'].get('30m', {}).get('rows', 55000):,} (30m)"
    c_15m = f"~{metrics['resolutions'].get('15m', {}).get('rows', 110000):,} (15m)"
    with col1: _render_custom_metric("Trước", "209,397 rows")
    with col2: _render_custom_metric("Sau cleaning", f"{c_1h} / {c_30m} / {c_15m}")
    with col3: _render_custom_metric("Missing", "~85% gaps (IoT sensor)")

    with st.expander("💡 Tại sao KHÔNG DÙNG IQR cho PM2.5?", expanded=False):
        st.markdown(
            """
        **Vấn đề (Outlier Removal Trap)**: Áp dụng phương pháp IQR chuẩn (loại bỏ các giá trị > `Q3 + 1.5*IQR`) cho phân phối fat-tailed như PM2.5 sẽ xóa nhầm các đỉnh ô nhiễm cực đoan (extreme events). Nếu mô hình học trên dữ liệu bị "cắt ngọn" này, nó sẽ bị hội chứng **"False Sense of Accuracy"**: điểm số MAE có vẻ thấp (vì đoán vùng dữ liệu dễ), nhưng mô hình bị mù hoàn toàn trước các đợt bùng phát ô nhiễm thật sự.

        **Giải pháp**: Với biến target (PM2.5), tuyệt đối không dùng filter thống kê (IQR/Z-score) mà bắt buộc dùng **Domain Bounds (0 - 500 µg/m³)**.
        """
        )


def _step_eda():
    """Step 3: Exploratory Data Analysis."""
    st.markdown(
        """
    <div style="background: var(--secondary-background-color); color: var(--text-color) !important;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid rgba(139,92,246,0.2);">
        <div style="font-size: 1.3rem; font-weight: 700; color: #8B5CF6;">
            🔬 Bước 3: Phân Tích Khám Phá (EDA)
        </div>
        <div style="opacity: 0.7; margin-top: 0.5rem;">
            Phát hiện patterns, seasonality, và mối quan hệ giữa các biến
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    findings = {
        f"🔄 Autocorrelation {cite('shumway2017')}": "PM2.5 có autocorrelation **rất cao** (~0.97 lag 1h) → Persistence baseline cực mạnh ở horizon ngắn",
        f"📈 Seasonality {cite('cleveland1990')}": "Chu kỳ ngày (24h) + tuần (168h) + năm — STL decomposition cho thấy trend nhẹ giảm",
        "🌡️ Correlations": "Nhiệt độ (-0.45), Độ ẩm (+0.38), CO₂ (+0.52) có tương quan có ý nghĩa với PM2.5",
        f"📊 Distribution {cite('boxcox1964')}": "PM2.5 right-skewed (thiên phải) → Box-Cox transform (λ≈-0.147) giúp normalize",
        "⏰ Diurnal Pattern": "PM2.5 cao nhất 6-8h sáng và 18-20h tối (giờ cao điểm giao thông)",
    }

    for key, finding in findings.items():
        st.markdown(f"**{key}**: {finding}", unsafe_allow_html=True)

    st.info(
        "💡 **Key insight**: Autocorrelation cao (~0.97) nghĩa là giá trị hiện tại gần giống giá trị 1 giờ trước. "
        "Persistence baseline (y_pred = y_last) rất khó beat ở horizon 1h. ML/DL chỉ vượt trội ở horizons dài hơn (6h, 24h)."
    )


def _get_dashboard_content():
    try:
        content_path = PROJECT_ROOT / "research" / "experiments" / "dashboard_content.json"
        with open(content_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _step_feature_engineering():
    """Step 4: Feature Engineering."""
    metrics = _get_pipeline_metrics()
    f_count = metrics.get('features_count', 121)
    
    content = _get_dashboard_content()
    feat_explanation = content.get("pipeline_walkthrough", {}).get("feature_engineering", {}).get(
        "feature_count_explanation", f"Tổng số cột đọc được: {f_count}"
    )

    st.markdown(
        f"""
    <div style="background: var(--secondary-background-color); color: var(--text-color) !important;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid rgba(245,158,11,0.2);">
        <div style="font-size: 1.3rem; font-weight: 700; color: #F59E0B;">
            🛠️ Bước 4: Feature Engineering {cite('christ2018')}
        </div>
        <div style="opacity: 0.7; margin-top: 0.5rem;">
            Từ 5 biến gốc → 119 Features ({f_count} tổng số cột - 1 Target - 1 Metadata)
        </div>
    </div>
    
    <div style="background: rgba(245,158,11,0.05); padding: 1rem; border-left: 3px solid #F59E0B; border-radius: 4px; margin-bottom: 1.5rem;">
        <span style="font-size: 0.95em;">💡 <b>Lưu ý học thuật:</b> {feat_explanation}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Feature groups
    feature_groups = {
        "⏰ Lag Features": ("24 features", "pm25_lag_1, pm25_lag_3, ..., pm25_lag_72"),
        "📊 Rolling Statistics": ("20 features", "rolling_mean_6h, rolling_std_24h, ..."),
        "📈 EWM (Exponential)": ("10 features", "ewm_mean_6h, ewm_std_12h, ..."),
        "📅 Calendar": ("10 features", "hour, dayofweek, month, is_weekend, ..."),
        "🌊 Fourier": ("8 features", "sin/cos(2π·hour/24), sin/cos(2π·day/7), ..."),
        "🔗 Interaction": ("15 features", "pm25_x_nhiet_do, temp_humidity_ratio, ..."),
        "Δ Change Features": ("12 features", "diff_1h, diff_6h, pct_change_1h, ... (shifted!)"),
        "🏠 Domain": ("20 features", "dew_point_depression, heat_index, ..."),
    }

    for group, (count, examples) in feature_groups.items():
        st.markdown(
            f"""
        <div style="display: flex; justify-content: space-between; align-items: center;
                    padding: 0.5rem 1rem; margin: 0.25rem 0;
                    background: rgba(245,158,11,0.05); border-radius: 8px;">
            <span><strong>{group}</strong></span>
            <span style="color: #F59E0B; font-weight: 600;">{count}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.caption(f"    Ví dụ: `{examples}`")

    st.markdown(
        f"""
        <div style="background: rgba(245,158,11,0.1); border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 4px; margin: 1rem 0;">
            <span style="font-size: 1.1em; font-weight: 600; color: #F59E0B;">⚠️ Anti-Leakage</span> {cite('hyndman2021')}<br><br>
            Tất cả features dùng target (diff, pct_change, ratio) đều áp dụng <code>shift(1)</code> — chỉ dùng dữ liệu QUÁ KHỨ.<br>
            Kiểm tra: <code>|corr(feature, target)| < 0.99</code> cho mọi feature.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Live demo - Feature Builder
    with st.expander("🟢 [LIVE] Chạy Feature Builder", expanded=False):
        if st.button("▶️ Build Features (CPU, ~5 giây)", key="run_feature_builder"):
            try:
                with st.spinner("Đang build features..."):
                    from src.data.cleaner import (
                        _clip_physical_bounds,
                        _handle_outliers,
                        _remove_duplicates,
                        _resample,
                        _set_datetime_index,
                    )
                    from src.data.imputer import impute_missing_data
                    from src.data.loader import load_raw_data
                    from src.features.builder import build_features

                    df_raw = load_raw_data()
                    df = _remove_duplicates(df_raw)
                    df = _set_datetime_index(df)
                    df, _ = _clip_physical_bounds(df)
                    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
                    df = _resample(df, freq="1h")
                    df_imp = impute_missing_data(df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24, verbose=False)
                    df_feat = build_features(df_imp)

                st.success(f"✅ {len(df_feat)} rows × {len(df_feat.columns)} features")
                st.dataframe(df_feat.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")


def _step_imputation():
    """Step 5: Missing Data Imputation."""
    st.markdown(
        """
    <div style="background: var(--secondary-background-color); color: var(--text-color) !important;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid rgba(6,182,212,0.2);">
        <div style="font-size: 1.3rem; font-weight: 700; color: #06B6D4;">
            🧪 Bước 5: Xử Lý Missing Data
        </div>
        <div style="opacity: 0.7; margin-top: 0.5rem;">
            Chiến lược Tiered Imputation cho dữ liệu IoT
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tiers = [
        (f"🟢 Gap ngắn (≤6 rows) {cite('moritz2015')}", "Cubic Spline Interpolation", "Tương đương: 6h (1h) | 3h (30m) | 1.5h (15m)"),
        (f"🟡 Gap trung bình (6-24 rows) {cite('troyanskaya2001')}", "KNN Multivariate (k=5)", "Tương đương: 24h (1h) | 12h (30m) | 6h (15m)"),
        (f"🔴 Gap dài (>24 rows) {cite('moritz2015')}", "DROP — Không recover", "Bỏ qua các đứt gãy quá lớn để tránh noise"),
    ]

    for tier, method, reason in tiers:
        col1, col2, col3 = st.columns([1.2, 1, 1.8])
        col1.markdown(f"**{tier}**", unsafe_allow_html=True)
        col2.markdown(f"`{method}`")
        col3.markdown(f"*{reason}*")
        
    content = _get_dashboard_content()
    imputation_strat = content.get("pipeline_walkthrough", {}).get("imputation_strategy", {})
    if imputation_strat:
        st.markdown(
            f"""
        <div style="background: rgba(245,158,11,0.05); padding: 1rem; border-left: 3px solid #F59E0B; border-radius: 4px; margin-top: 1rem; margin-bottom: 1.5rem;">
            <span style="font-size: 0.95em;">💡 <b>{imputation_strat.get('title', 'Cơ Sở Học Thuật')}:</b> {imputation_strat.get('explanation', '')}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: _render_custom_metric("Kết quả", "Tùy thuộc độ phân giải (15m/30m/1h)")
    with col2: _render_custom_metric("Tracking", "Cột `is_imputed` = 1/0")

    content = _get_dashboard_content()
    val_strategy = content.get("pipeline_walkthrough", {}).get("validation_strategy", {})
    val_title = val_strategy.get("title", "Chiến Lược Xác Thực (Validation Strategy) & Tính Toàn Vẹn Dữ Liệu")
    val_explanation = val_strategy.get("explanation", "Test set BẮT BUỘC chỉ dùng data thật (`is_imputed == 0`).")

    st.markdown(
        f"""
        <div style="background: rgba(6,182,212,0.05); border-left: 4px solid #06B6D4; padding: 1.2rem; border-radius: 4px; margin: 1.5rem 0;">
            <div style="font-size: 1.05em; font-weight: 700; color: #06B6D4; margin-bottom: 0.5rem;">
                🔬 {val_title}
            </div>
            <div style="font-size: 0.95em; line-height: 1.6;">
                {val_explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _step_modeling():
    """Step 6: Modeling & Evaluation."""
    st.markdown(
        """
    <div style="background: var(--secondary-background-color); color: var(--text-color) !important;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid rgba(16,185,129,0.2);">
        <div style="font-size: 1.3rem; font-weight: 700; color: #10B981;">
            📊 Bước 6: Huấn Luyện & Đánh Giá Mô Hình
        </div>
        <div style="opacity: 0.7; margin-top: 0.5rem;">
            Mô hình đa độ phân giải (15m, 30m, 1h) × 3 horizons — v9 pipeline
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Model families - dynamic best MASE for Ensemble
    metrics_data = _load_standardized_metrics()
    ens_6h_mase = 0.382  # fallback
    if metrics_data and "results" in metrics_data:
        h6 = metrics_data["results"].get("6h", {})
        for name, m in h6.items():
            if "Ensemble" in name:
                val = m.get("mase_unified", m.get("mase"))
                if val is not None and val < ens_6h_mase:
                    ens_6h_mase = val

    families = {
        "📏 Baseline": [f"Persistence (y_pred = y_last) — mỗi resolution riêng {cite('hyndman2021')}"],
        "📈 Statistical": [f"ARIMA {cite('shumway2017')}", f"SARIMAX {cite('box2015')}"],
        "🌲 ML (Tree-based)": [f"LightGBM (Optuna) {cite('ke2017')} {cite('akiba2019')}", "ElasticNet", f"Random Forest {cite('breiman2001')}", "Gradient Boosting", f"Stacking {cite('wolpert1992')}"],
        "🧠 Deep Learning": [f"LSTM {cite('hochreiter1997')}", f"GRU {cite('cho2014')}", f"TFT {cite('lim2021')}"],
        "🎯 Ensemble": [f"Ensemble_Weighted (Best 6h! MASE={ens_6h_mase:.3f}) {cite('lakshminarayanan2017')}", f"VotingEnsemble {cite('dietterich2000')}"],
    }

    for family, models in families.items():
        st.markdown(f"**{family}**: {', '.join(models)}", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background: rgba(16,185,129,0.05); padding: 1rem; border-left: 3px solid #10B981; border-radius: 4px; margin-top: 1.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.95em; line-height: 1.6;">
                <b>💡 Lý do chọn mô hình:</b> Hệ thống áp dụng 5 phân lớp mô hình để kiểm chứng chéo giả thuyết (Cross-Hypothesis Testing): 
                <b>(1) Baseline</b> cung cấp mức sàn tối thiểu; 
                <b>(2) Statistical</b> xử lý tuyến tính và xu hướng vĩ mô; 
                <b>(3) Tree-based ML</b> giải quyết tốt dữ liệu dạng bảng (tabular) với nhiều features; 
                <b>(4) Deep Learning</b> (đặc biệt RNN/Attention) bắt sóng chuỗi thời gian phi tuyến tính; và 
                <b>(5) Ensemble</b> triệt tiêu sai số phương sai (variance) bằng cách hợp nhất sức mạnh của tất cả các họ trên.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Results table from pre-computed data
    metrics = _load_standardized_metrics()
    if metrics and "results" in metrics:
        st.markdown("#### 🏆 Bảng Xếp Hạng (MASE — thấp hơn = tốt hơn)")

        for horizon in ["1h", "6h", "24h"]:
            h_data = metrics["results"].get(horizon, {})
            if not h_data:
                continue

            rows = []
            for model_name, m in h_data.items():
                mase = m.get("mase_unified", m.get("mase", m.get("mase_original", None)))
                mae = m.get("mae")
                if mase is not None:
                    rows.append({
                        "Model": model_name,
                        "MAE (µg/m³)": round(mae, 2) if mae else "—",
                        "MASE": round(mase, 3),
                        "vs Persistence": f"{(1 - mase) * 100:+.1f}%" if mase != 1.0 else "baseline",
                    })

            df_table = pd.DataFrame(rows).sort_values("MASE")
            st.markdown(f"**Horizon {horizon}**")
            st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Live demo - LightGBM Training
    with st.expander("🟢 [LIVE] Train LightGBM (CPU, ~15 giây)", expanded=False):
        demo_horizon = st.selectbox("Horizon", [1, 6, 24], index=1, key="demo_lgbm_h")
        if st.button("▶️ Train LightGBM", key="run_lgbm_demo"):
            try:
                from src.training.trainer import LightGBMTrainer, get_default_params

                params = get_default_params("LightGBM")
                trainer = LightGBMTrainer(horizon=demo_horizon, params=params)

                progress_bar = st.progress(0)
                status_text = st.empty()

                def callback(step, total, msg):
                    progress_bar.progress(step / total)
                    status_text.caption(f"⏳ {msg}")

                result = trainer.train(progress_callback=callback)
                progress_bar.progress(1.0)
                status_text.empty()

                st.success("✅ Training hoàn tất!")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{result['mae']:.3f}")
                col2.metric("MASE", f"{result['mase']:.3f}")
                col3.metric("R²", f"{result['r2']:.3f}")
                col4.metric("Thời gian", f"{result['training_time_s']:.1f}s")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")


def _step_results():
    """Step 7: Results & Conclusions."""
    st.markdown(
        """
    <div style="background: var(--secondary-background-color); color: var(--text-color) !important;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid rgba(234,179,8,0.2);">
        <div style="font-size: 1.3rem; font-weight: 700; color: #EAB308;">
            📈 Bước 7: Kết Quả & Kết Luận
        </div>
        <div style="opacity: 0.7; margin-top: 0.5rem;">
            Tổng hợp findings và bài học kinh nghiệm
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Key findings
    st.markdown("#### 🔑 Phát hiện chính")

    metrics_data = _load_standardized_metrics()
    best_6h_model = "Ensemble_Weighted_v9_30m"
    best_6h_mase = 0.382
    best_24h_model = "Ensemble_Weighted_v9_30m"
    best_24h_mase = 0.469
    
    if metrics_data and "results" in metrics_data:
        h6 = metrics_data["results"].get("6h", {})
        if h6:
            valid_6h = [(name, m.get("mase", 1.0)) for name, m in h6.items() if m.get("mase") is not None and "Persistence" not in name]
            if valid_6h:
                best_6h_model, best_6h_mase = min(valid_6h, key=lambda x: x[1])
                
        h24 = metrics_data["results"].get("24h", {})
        if h24:
            valid_24h = [(name, m.get("mase", 1.0)) for name, m in h24.items() if m.get("mase") is not None and "Persistence" not in name]
            if valid_24h:
                best_24h_model, best_24h_mase = min(valid_24h, key=lambda x: x[1])

    f_count = _get_pipeline_metrics().get('features_count', 119)
    findings = [
        ("🏆 Best Models", f"6h: {best_6h_model} (MASE={best_6h_mase:.3f}) | 24h: {best_24h_model} (MASE={best_24h_mase:.3f})"),
        (f"📏 Phá vỡ bẫy Autocorrelation {cite('hyndman2021')}", "Ở 1h (1h res.), Persistence thường thắng do autocorrelation ~0.97. Lần đầu tiên, mô hình GRU_v9_15m ở v9 đã chính thức phá vỡ giới hạn này (MASE < Persistence)!"),
        ("🌲 ML vs DL", "Fair DL Pipeline (tabular features cho DL) > Expert DL Pipeline (raw data). Ensemble methods tốt nhất."),
        ("🔧 Feature Engineering", f"{f_count} features (Fourier + interaction + domain). Ablation chứng minh tabular FE > raw data cho IoT."),
        (f"⚠️ Anti-Leakage {cite('tashman2000')}", "Phát hiện và xử lý 4 nguồn leakage từ diff/pct_change features → pipeline integrity 100%."),
    ]

    for title, desc in findings:
        st.markdown(
            f"""
        <div style="background: rgba(234,179,8,0.05); border-left: 3px solid #EAB308;
                    padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
            <strong>{title}</strong><br>
            <span style="opacity: 0.7;">{desc}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # MASE comparison chart
    st.markdown("#### 📊 So sánh MASE theo Horizon")
    metrics = _load_standardized_metrics()
    if metrics and "results" in metrics:
        chart_data = []
        for horizon in ["1h", "6h", "24h"]:
            h_data = metrics["results"].get(horizon, {})
            for model, m in h_data.items():
                mase = m.get("mase_unified", m.get("mase", None))
                if mase and model != "Persistence":
                    chart_data.append({
                        "Model": model,
                        "Horizon": horizon,
                        "MASE": round(mase, 3),
                    })

        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig = px.bar(
                df_chart,
                x="Model",
                y="MASE",
                color="Horizon",
                barmode="group",
                title="MASE theo Model × Horizon (thấp hơn = tốt hơn)",
                color_discrete_sequence=["#06B6D4", "#8B5CF6", "#F59E0B"],
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                         annotation_text="Persistence Baseline")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, Arial, sans-serif", size=10),
                height=450,
            )
            _render_chart(fig, filename="pipeline_mase_comparison")

    # Lessons learned
    st.markdown("#### 📝 Bài học kinh nghiệm")
    lessons = [
        "Data Leakage dễ xảy ra trong time series khi dùng `diff(y)`, `pct_change(y)` chứa y[t]. Luôn shift(1).",
        "IoT sensor missing data ~85% — phần lớn là gaps >1 tuần. Chỉ focus impute gaps ≤24h.",
        "Persistence baseline cực mạnh ở short horizon. Cần multi-horizon eval để thấy giá trị ML.",
        "Cubic Spline univariate không nên dùng cho gap >6h. KNN multivariate tốt hơn cho medium gaps.",
        "Box-Cox transformation (λ≈-0.147) giúp normalize PM2.5 skewed distribution → model tốt hơn.",
    ]
    for i, lesson in enumerate(lessons, 1):
        st.markdown(f"{i}. {lesson}")


def page_pipeline_walkthrough(results):
    """Render Pipeline Walkthrough page."""

    st.markdown(
        """
    <h1 style="font-size: 2.2rem; margin-bottom: 0.25rem;">
        📜 Quy Trình Pipeline — Từng Bước
    </h1>
    <p style="opacity: 0.7; font-size: 1.05rem; margin-bottom: 0.5rem;">
        Toàn bộ quy trình nghiên cứu từ thu thập dữ liệu đến kết luận
    </p>
    """,
        unsafe_allow_html=True,
    )

    # ── Version-aware info cards ──
    from src.info_cards import get_current_version, render_version_badge, render_info_card
    ver = get_current_version()
    render_version_badge(ver)
    render_info_card(
        "Hướng dẫn: Quy Trình Pipeline",
        "Trang này trình bày **7 bước nghiên cứu** từ thu thập dữ liệu đến kết luận:\n\n"
        "- Mỗi bước kèm **mục đích**, **phương pháp**, và **kết quả** cụ thể\n"
        "- Một số bước hỗ trợ **Live Demo** (Feature Builder, LightGBM training)\n"
        "- Metrics được pre-computed từ kết quả thực tế, không phải placeholder\n\n"
        "**💡 Tip**: Click vào từng step để xem chi tiết phương pháp luận.",
        icon="📖",
        collapsed=True,
    )

    # Pipeline progress bar
    steps = [
        "📥 Thu thập",
        "🧹 Làm sạch",
        "🔬 EDA",
        "🔧 Features",
        "🧪 Imputation",
        "📊 Modeling",
        "📈 Kết quả",
    ]

    # Step selector
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(
                f"""
            <div style="text-align: center; padding: 0.5rem;
                        background: rgba(0,212,170,0.1); border-radius: 8px;
                        border: 1px solid rgba(0,212,170,0.2);
                        font-size: 0.75rem; color: #00D4AA;">
                {step}
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Render all steps
    _step_data_collection()
    st.divider()
    _step_data_cleaning()
    st.divider()
    _step_eda()
    st.divider()
    _step_feature_engineering()
    st.divider()
    _step_imputation()
    st.divider()
    _step_modeling()
    st.divider()
    _step_results()

    render_references_section()
