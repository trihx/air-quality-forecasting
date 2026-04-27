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

PROJECT_ROOT = Path(__file__).resolve().parent if Path(__file__).resolve().parent.name != "src" else Path(__file__).resolve().parent.parent

# Try to find project root properly
for candidate in [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
    if (candidate / "app.py").exists():
        PROJECT_ROOT = candidate
        break


def _load_standardized_metrics() -> dict:
    """Load pre-computed standardized metrics."""
    path = PROJECT_ROOT / "research" / "experiments" / "standardized_metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _step_data_collection():
    """Step 1: Data Collection."""
    st.markdown(
        """
    <div style="background: #1A1F2E; color: #F8FAFC !important;
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
    col1.metric("📦 Tổng Records", "209,397")
    col2.metric("📅 Thời gian", "3.1 năm")
    col3.metric("⏱️ Tần suất", "~2 phút/lần")
    col4.metric("📊 Biến số", "5 (PM2.5, Nhiệt độ, Độ ẩm, Điểm sương, CO₂)")

    st.markdown("#### 📋 Mô tả biến")
    var_data = {
        "Biến": ["PM2.5", "Nhiệt độ", "Độ ẩm", "Điểm sương", "CO₂"],
        "Đơn vị": ["µg/m³", "°C", "%", "°C", "ppm"],
        "Vai trò": ["🎯 Target", "Feature", "Feature", "Feature", "Feature"],
        "Nguồn": ["PMS5003", "DHT22", "DHT22", "Tính toán", "MH-Z19B"],
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
    <div style="background: #1A1F2E; color: #F8FAFC !important;
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
        ("2️⃣ Domain clipping", "PM2.5 ∈ [0, 500] µg/m³ — theo WHO guidelines"),
        ("3️⃣ Outlier detection", "IQR 3.0 cho các biến ngoài PM2.5 (S-ESD cho PM2.5 để giữ seasonal peaks)"),
        ("4️⃣ Resampling", "Từ ~2 phút → 1 giờ (hourly) bằng mean aggregation"),
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
    col1.metric("Trước", "209,397 rows")
    col2.metric("Sau cleaning", "~27,000 hourly")
    col3.metric("Missing", "~85% gaps (IoT sensor)")

    with st.expander("💡 Tại sao S-ESD cho PM2.5?", expanded=False):
        st.markdown(
            """
        **Vấn đề**: IQR chuẩn loại bỏ các đỉnh PM2.5 theo mùa (seasonal peaks) — đây là dữ liệu thật, không phải outlier.

        **Giải pháp**: S-ESD (Seasonal Extreme Studentized Deviate) — phương pháp của Rosner (1983), tự động phân biệt outlier thật và seasonal variation.

        **Kết quả**: Giữ lại 100% seasonal peaks, chỉ loại bỏ sensor noise thực sự.
        """
        )


def _step_eda():
    """Step 3: Exploratory Data Analysis."""
    st.markdown(
        """
    <div style="background: #1A1F2E; color: #F8FAFC !important;
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
        "🔄 Autocorrelation": "PM2.5 có autocorrelation **rất cao** (~0.97 lag 1h) → Persistence baseline cực mạnh ở horizon ngắn",
        "📈 Seasonality": "Chu kỳ ngày (24h) + tuần (168h) + năm — STL decomposition cho thấy trend nhẹ giảm",
        "🌡️ Correlations": "Nhiệt độ (-0.45), Độ ẩm (+0.38), CO₂ (+0.52) có tương quan có ý nghĩa với PM2.5",
        "📊 Distribution": "PM2.5 right-skewed (thiên phải) → Box-Cox transform (λ≈-0.147) giúp normalize",
        "⏰ Diurnal Pattern": "PM2.5 cao nhất 6-8h sáng và 18-20h tối (giờ cao điểm giao thông)",
    }

    for key, finding in findings.items():
        st.markdown(f"**{key}**: {finding}")

    st.info(
        "💡 **Key insight**: Autocorrelation cao (~0.97) nghĩa là giá trị hiện tại gần giống giá trị 1 giờ trước. "
        "Persistence baseline (y_pred = y_last) rất khó beat ở horizon 1h. ML/DL chỉ vượt trội ở horizons dài hơn (6h, 24h)."
    )


def _step_feature_engineering():
    """Step 4: Feature Engineering."""
    st.markdown(
        """
    <div style="background: #1A1F2E; color: #F8FAFC !important;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid rgba(245,158,11,0.2);">
        <div style="font-size: 1.3rem; font-weight: 700; color: #F59E0B;">
            🔧 Bước 4: Feature Engineering
        </div>
        <div style="opacity: 0.7; margin-top: 0.5rem;">
            Từ 5 biến gốc → 119 features thông minh
        </div>
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

    st.warning(
        "⚠️ **Anti-Leakage**: Tất cả features dùng target (diff, pct_change, ratio) "
        "đều áp dụng `shift(1)` — chỉ dùng dữ liệu QUÁ KHỨ. "
        "Kiểm tra: `|corr(feature, target)| < 0.99` cho mọi feature."
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
    <div style="background: #1A1F2E; color: #F8FAFC !important;
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
        ("🟢 Gap ngắn (≤6h)", "Cubic Spline Interpolation", "Nhanh, chính xác cho gaps ngắn liên tục"),
        ("🟡 Gap trung bình (6-24h)", "KNN Multivariate (k=5)", "Dùng thông tin từ tất cả biến để dự đoán"),
        ("🔴 Gap dài (>24h)", "DROP — Không recover", "Impute gap >24h tạo noise, ảnh hưởng model"),
    ]

    for tier, method, reason in tiers:
        col1, col2, col3 = st.columns([1, 1, 2])
        col1.markdown(f"**{tier}**")
        col2.markdown(f"`{method}`")
        col3.markdown(f"*{reason}*")

    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Kết quả", "7,742 rows sau impute")
    col2.metric("Tracking", "Cột `is_imputed` = 1/0")

    st.info(
        "💡 **Test-on-Real-Only**: Test set BẮT BUỘC chỉ dùng data thật (`is_imputed == 0`). "
        "Data imputed chỉ dùng cho training."
    )


def _step_modeling():
    """Step 6: Modeling & Evaluation."""
    st.markdown(
        """
    <div style="background: #1A1F2E; color: #F8FAFC !important;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid rgba(16,185,129,0.2);">
        <div style="font-size: 1.3rem; font-weight: 700; color: #10B981;">
            📊 Bước 6: Huấn Luyện & Đánh Giá Mô Hình
        </div>
        <div style="opacity: 0.7; margin-top: 0.5rem;">
            28 mô hình × 3 horizons — so sánh công bằng
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Model families
    families = {
        "📏 Baseline": ["Persistence (y_pred = y_last)"],
        "📈 Statistical": ["ARIMA", "SARIMAX"],
        "🌲 ML (Tree-based)": ["LightGBM (Optuna 100 trials)", "XGBoost", "Random Forest", "Gradient Boosting"],
        "🧠 Deep Learning": ["LSTM", "GRU (Best 24h!)", "TFT (Transformer)"],
        "🎯 Ensemble": ["Stacking", "Weighted Ensemble (Best 6h!)"],
    }

    for family, models in families.items():
        st.markdown(f"**{family}**: {', '.join(models)}")

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
    <div style="background: #1A1F2E; color: #F8FAFC !important;
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

    findings = [
        ("🏆 Best Models", "6h: Ensemble Weighted (MASE=0.703) | 24h: LSTM v2+log (MASE=0.691)"),
        ("📏 Persistence dominance", "Ở 1h, Persistence là best model do autocorrelation ~0.97. ML/DL hiệu quả ở 6h-24h."),
        ("🌲 ML vs DL", "ML (LightGBM, RF) tốt hơn ở 6h. DL (GRU, LSTM) tốt hơn ở 24h khi cần capture long-term patterns."),
        ("🔧 Feature Engineering", "119 features (Fourier + interaction + domain) giúp giảm MAE 14% so với baseline features."),
        ("⚠️ Anti-Leakage", "Phát hiện và xử lý leakage từ diff/pct_change features → pipeline integrity 100%."),
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
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

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
