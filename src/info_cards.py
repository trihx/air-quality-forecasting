"""
Version-aware info card registry for Dashboard pages.

Each page gets a "guide" card (hướng dẫn) and optionally a "method" card
(phương pháp) and/or "finding" card (phát hiện). Cards read metadata from
the selected snapshot version to display version-specific information.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "research" / "experiments" / "dashboard_runs"


# ── Version loader ───────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_all_snapshots() -> dict[str, dict]:
    """Load all version snapshots from dashboard_runs/."""
    snapshots: dict[str, dict] = {}
    if not RUNS_DIR.exists():
        return snapshots
    for jpath in sorted(RUNS_DIR.glob("*.json")):
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            version = data.get("version", jpath.stem)
            snapshots[version] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return snapshots


def get_current_version() -> str:
    """Get the currently selected version from session state."""
    return st.session_state.get("selected_version", "")


def version_selector_sidebar():
    """Render version selector in sidebar. Returns selected version name."""
    snapshots = load_all_snapshots()
    if not snapshots:
        return ""

    versions = list(snapshots.keys())

    st.sidebar.markdown(
        """
    <div style="font-size: 0.75rem; opacity: 0.6; text-transform: uppercase;
                letter-spacing: 0.1em; margin: 1rem 0 0.3rem 0; font-weight: 700;">
        📦 Phiên bản Pipeline
    </div>
    """,
        unsafe_allow_html=True,
    )

    selected = st.sidebar.selectbox(
        "Chọn phiên bản",
        versions,
        index=len(versions) - 1,  # default to latest
        key="selected_version",
        label_visibility="collapsed",
    )
    return selected


def get_version_data(version: str) -> dict:
    """Get snapshot data for a specific version."""
    snapshots = load_all_snapshots()
    return snapshots.get(version, {})


# ── Info Card renderer ───────────────────────────────────────────────


def render_info_card(
    title: str,
    content: str,
    icon: str = "💡",
    accent: str = "#00D4AA",
    collapsed: bool = True,
):
    """Render a themed info card using st.expander for collapsibility."""
    with st.expander(f"{icon} {title}", expanded=not collapsed):
        st.markdown(content)


def render_version_badge(version: str):
    """Render a small version badge at top of page."""
    if not version:
        return
    v_data = get_version_data(version)
    n_models = len(v_data.get("models_included", []))
    description = v_data.get("description", "")
    timestamp = v_data.get("timestamp", "")[:10]

    st.markdown(
        f"""
    <div style="display: inline-flex; gap: 0.5rem; align-items: center; margin-bottom: 1rem;">
        <span style="background: rgba(0,212,170,0.15); color: #00D4AA; padding: 0.25rem 0.75rem;
                     border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
            📦 {version}
        </span>
        <span style="opacity: 0.6; font-size: 0.8rem;">
            {n_models} models · {timestamp}
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE-SPECIFIC INFO CARDS
# ══════════════════════════════════════════════════════════════════════


def cards_overview(version: str):
    """Info cards for Overview page."""
    v = get_version_data(version)
    changes = v.get("changes", {})

    render_info_card(
        "Hướng dẫn: Tổng Quan",
        "Trang này trình bày **bức tranh toàn cảnh** của dự án:\n\n"
        "- **KPI Cards** phía trên: Best model, test coverage, số lượng models\n"
        "- **Pipeline Architecture**: Sơ đồ 7 bước từ raw data → evaluation\n"
        "- **Model Rankings**: Bảng xếp hạng MASE tại 3 horizons\n"
        "- **Key Findings**: Thành công + hạn chế (cân bằng khoa học)\n\n"
        "**💡 Tip**: Đọc trang '📜 Quy Trình Pipeline' trước để hiểu workflow chi tiết.",
        icon="📖",
        collapsed=True,
    )

    if changes:
        render_info_card(
            f"Phiên bản hiện tại: {version}",
            f"**What:** {changes.get('what', '—')}\n\n"
            f"**Why:** {changes.get('why', '—')}\n\n"
            f"**Result:** {changes.get('result', '—')}\n\n"
            f"**Conclusion:** {changes.get('conclusion', '—')}",
            icon="🔬",
            accent="#8B5CF6",
            collapsed=True,
        )

    render_info_card(
        "Bài Học Kinh Nghiệm (Top Lessons)",
        "Tổng hợp từ 7 phiên bản pipeline, 16+ thí nghiệm:\n\n"
        "1. **Data Leakage là nguy hiểm #1** — `diff(y)` chứa y[t]. Fix: `shift(1).diff()`. "
        "Phát hiện qua MASE<0.1 hoặc R²>0.99 → audit ngay "
        "*(Ref: [MJ] Ch.8, [JB2] Ch.4)*\n\n"
        "2. **Feature Engineering = con dao hai lưỡi** — 117 features giúp GRU 6h (↓14.8%) "
        "nhưng HẠI 1h (+30.5%). ACF≈0.97 ở 1h → lag_1h là đủ "
        "*(Ref: [MJ] Ch.5, [FL] Ch.5)*\n\n"
        "3. **Log transform phụ thuộc kiến trúc** — GRU thích log (6h: 0.783→0.692, ↓11.6%), "
        "nhưng LSTM ưa raw ở 6h (0.719 vs 0.753). PHẢI test cả hai "
        "*(Ref: [MJ] Ch.6, [AQ] Ch.3)*\n\n"
        "4. **Fourier > Deseasonalizing** — Explicit deseasonalizing (seasonal_diff 0.903) TỆ HƠN raw (0.731). "
        "Fourier features đã capture seasonality → double-remove = thêm nhiễu "
        "*(Ref: [MJ] Ch.5, pp.130-135)*\n\n"
        "5. **STL trên full data = Leakage ẩn** — STL fitted full (MASE 0.507) vs train-only (0.736) = +45% inflation. "
        "Mọi transform PHẢI fit trên TRAIN ONLY "
        "*(Ref: [MJ] Ch.8, pp.200-203)*",
        icon="📝",
        collapsed=True,
    )

    render_info_card(
        "Cải Tiến Đã Chứng Minh (v1 → v6)",
        "| Cải tiến | Trước (v1) | Sau | Δ Improvement | Nguồn tham khảo |\n"
        "|----------|-----------|------|---------------|----------------|\n"
        "| Fourier features | 95 feat | 119 feat | LightGBM MAE ↓14.2% | [MJ] Ch.5, [HP] Ch.4 |\n"
        "| GRU v2 + log transform | MASE 0.812 | 0.692 | ↓14.8% (6h) | [DL] Ch.4, [MJ] Ch.6 |\n"
        "| Ensemble Weighted | RF đơn lẻ 0.706 | 0.703 | ↓0.4% (6h) | [MJ] Ch.6, [FL] Ch.8 |\n"
        "| Anti-leakage audit | 4 nguồn leakage | 0 | Ridge MAE: 0.004→2.824 | [MJ] Ch.8, [JB2] Ch.4 |\n"
        "| Domain outlier bounds | IQR cap 54µg/m³ | [0,500] | Giữ 1,908 extreme events | [AP] Ch.7, [MJ] Ch.2 |\n"
        "| Unified Persistence MAE | DL MASE=inf bug | Standardized | Comparable across families | [MJ] Ch.7 |\n"
        "| Test-on-real-only | Include imputed | Real only | Metrics phản ánh thực tế | [MJ] Ch.8, [PX] Ch.8 |",
        icon="📊",
        collapsed=True,
    )


def cards_eda(version: str):
    """Info cards for EDA page."""
    render_info_card(
        "Hướng dẫn: Phân Tích Khám Phá (EDA)",
        "Trang EDA gồm **6 tabs** theo logic storytelling:\n\n"
        "1. **Tổng Quan**: Thống kê mô tả + Forecastability Score\n"
        "2. **Gaps & Spikes**: Vấn đề data quality (missing, outliers)\n"
        "3. **Stationarity & Seasonality**: ADF/KPSS + STL decomposition\n"
        "4. **Autocorrelation & Drift**: Bẫy tự tương quan + concept drift\n"
        "5. **The 'Why'**: Giải thích tại sao thiết kế pipeline như vậy\n"
        "6. **Deep Insights (v7)**: Error anatomy, Granger causality\n\n"
        "**Phương pháp:**\n"
        "- ADF + KPSS (dual test) — không dựa 1 test đơn lẻ\n"
        "- STL decomposition (period=24h) — tách trend/seasonal/residual\n"
        "- S-ESD outlier detection — giữ seasonal peaks, chỉ loại sensor noise\n"
        "- Forecastability Score (composite: CoV + ApEn + ACF + Seasonality)",
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Phát hiện thú vị từ EDA",
        "🔍 **Top findings:**\n\n"
        "- **ACF(1) ≈ 0.97** → Persistence baseline cực mạnh ở 1h (mọi ML/DL đều thua) "
        "*(Ref: [AP] Ch.6, [ZZ] Ch.3)*\n"
        "- **PM2.5 NOT Normal** (Shapiro p < 1e-50) → justify MASE thay vì MAPE "
        "*(Ref: [PX] Ch.7, [MJ] Ch.7)*\n"
        "- **Forecastability Score = 0.434** (Trung bình) → MASE > 1 ở h=1 là *expected* "
        "*(Ref: [MJ] Ch.4, pp.92-96)*\n"
        "- **STL Residual σ = 5.18 µg/m³** → 'sàn hiệu suất' lý thuyết "
        "*(Ref: [MJ] Ch.3, pp.68-73)*\n"
        "- **Diurnal cycle**: Peak 6h sáng (nghịch nhiệt), trough 12h trưa (đối lưu) "
        "*(Ref: [AP] Ch.5)*\n"
        "- **CO2 Granger-causes PM2.5** (p < 0.001) → validate multivariate approach "
        "*(Ref: [PX] Ch.10, [HP] Ch.3)*",
        icon="✨",
        accent="#F59E0B",
        collapsed=True,
    )

    render_info_card(
        "Bài Học: Outlier Detection cho IoT",
        "**Vấn đề:** IQR×3 cap PM2.5 tại 54 µg/m³ — dưới ngưỡng WHO 'Unhealthy' (55.4). "
        "Loại bỏ 1,908 sự kiện ô nhiễm thực (skew=3.21, kurt=32.4).\n\n"
        "**Giải pháp:** Dùng **domain bounds [0, 500]** cho PM2.5 (fat-tailed), "
        "giữ IQR cho các biến khác (temp, humidity).\n\n"
        "**Tài liệu hướng dẫn:**\n"
        "- [MJ] Ch.2, pp.52-55: *'IQR assumes approximate symmetry. INAPPROPRIATE for fat-tailed.'*\n"
        "- [PX] Ch.3, pp.61-63: *'Use Modified Z-score (MAD-based) for robustness.'*\n"
        "- [AP] Ch.7, pp.156-160: *'PM2.5 physical range: 0-500+ µg/m³ (WHO AQI).'*\n\n"
        "**Impact:** Thay domain bounds → model học được extreme patterns → dự báo tốt hơn ở high-PM2.5 events.",
        icon="🧪",
        collapsed=True,
    )


def cards_hyperparams(version: str):
    """Info cards for Hyperparameters page."""
    v = get_version_data(version)

    render_info_card(
        "Hướng dẫn: Hyperparameters",
        "Trang này liệt kê **cấu hình tối ưu** cho từng model family:\n\n"
        "- **LightGBM**: Optuna TPE sampler, 100 trials/horizon, TimeSeriesSplit(5)\n"
        "- **GRU/LSTM**: Lookback 72h, hidden_dim 64, 2 layers, early stopping patience 10\n"
        "- **ARIMA/SARIMA**: auto_arima (AIC), rolling window 720\n\n"
        "**💡 Tip**: Dùng trang '🏋️ Huấn Luyện' để thử thay đổi params trực tiếp.",
        icon="📖",
        collapsed=True,
    )

    feature_set = v.get("feature_set", {})
    if feature_set:
        enabled = [k for k, v in feature_set.items() if v]
        render_info_card(
            f"Feature Set ({version})",
            f"**{len(enabled)} feature groups enabled:**\n\n"
            + "\n".join(f"- ✅ `{f}`" for f in sorted(enabled)),
            icon="🧬",
            collapsed=True,
        )

    render_info_card(
        "Tài liệu tham khảo: Hyperparameter Tuning",
        "**LightGBM (Bayesian Optimization):**\n"
        "- [MJ] Ch.6, pp.155-160: *'LightGBM preferred for speed. Enable early_stopping_rounds.'*\n"
        "- [FL] Ch.7: *'For time series: use TimeSeriesSplit, NOT random k-fold.'*\n"
        "- [AK] Ch.8: *'leaf-wise growth captures complex patterns faster than level-wise.'*\n\n"
        "**GRU/LSTM Architecture:**\n"
        "- [DL] Ch.4, pp.95-100: *'GRU = simplified LSTM. Fewer params → better for small datasets (<10K).'*\n"
        "- [JB1] Ch.5: *'Lookback = 10-15× forecast horizon.'* → 72h = 12× h=6\n"
        "- [DL] Ch.6: *'ReduceLROnPlateau is most practical LR scheduler.'*\n\n"
        "**Bài học kinh nghiệm:**\n"
        "- TFT hidden_dim=32 không đủ cho 113 features → MASE tệ đi +92% (v1→v2)\n"
        "- `n_jobs=-1` gây OMP crash trên M1 Pro + LightGBM → LUÔN dùng `n_jobs=1`\n"
        "- Log transform GRU ↓11.6%, nhưng LSTM ưa raw → test cả hai",
        icon="📚",
        collapsed=True,
    )


def cards_training(version: str):
    """Info cards for Training page."""
    render_info_card(
        "Hướng dẫn: Huấn Luyện Mô Hình",
        "Bạn có thể **train mô hình trực tiếp** trên Dashboard:\n\n"
        "1. **Chọn model** (LightGBM hoặc GRU) và **horizon** (1h/6h/24h)\n"
        "2. **Điều chỉnh hyperparameters** — giá trị mặc định là best params từ Optuna\n"
        "3. Click **🚀 Bắt Đầu Huấn Luyện** → xem progress bar real-time\n"
        "4. Sau khi train xong → xem MAE, MASE, R² + lưu model\n\n"
        "**⚠️ Lưu ý:**\n"
        "- LightGBM: ~15-30 giây (CPU)\n"
        "- GRU: ~2-5 phút (CPU) hoặc ~30 giây (MPS/CUDA GPU)\n"
        "- Training dùng data pipeline đầy đủ (anti-leakage, is_imputed filter)",
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Phương pháp & Tài liệu tham khảo",
        "**Training pipeline:**\n\n"
        "1. Load raw data → Clean (IQR 3.0, PM2.5: domain [0,500]) → Resample 1h\n"
        "2. Impute (Hybrid: Spline ≤6h + KNN 6-24h) *(Ref: [MJ] Ch.2, [HP] Ch.5)*\n"
        "3. Build 119 features (v2: Fourier + interactions + CV) *(Ref: [MJ] Ch.5)*\n"
        "4. Split 80/10/10 temporal → shift target by horizon\n"
        "5. Train → Evaluate on **real data only** (is_imputed == 0)\n\n"
        "**Walk-Forward Validation:** *(Ref: [PX] Ch.8, [FL] Ch.5)*\n"
        "- TimeSeriesSplit(5), expanding window, temporal order preserved\n"
        "- Standard k-fold VIOLATES temporal structure → information leakage [MJ] Ch.8\n\n"
        "**Metrics**: MAE (primary, [PX] Ch.7) + MASE (vs Persistence, [MJ] Ch.7) + RMSE + R²",
        icon="🔧",
        collapsed=True,
    )

    render_info_card(
        "Bài Học: Pitfalls khi Training",
        "⚠️ **Rút kinh nghiệm từ dự án:**\n\n"
        "- **Lazy import torch SAU LightGBM** — import cùng lúc gây segfault trên MPS/CUDA "
        "*(Lesson: 2026-04-05)*\n"
        "- **CV features (std/mean) cần safeguard** — khi mean≈0, CV explodes → inf. "
        "Fix: clamp mean >= 1.0, clip CV <= 5.0 *(Lesson: 2026-04-12)*\n"
        "- **Multi-Horizon Target Bug** — `target=shift(-h)` nhưng Persistence KHÔNG dùng lag features. "
        "Persist = `df[TARGET_COL]` *(Lesson: 2026-04-05)*\n"
        "- **DL Persistence Alignment Bug** — test index offset bằng LOOKBACK → MASE=inf. "
        "Fix: Unified Persistence MAE từ ML test set *(Lesson: 2026-04-12)*",
        icon="⚠️",
        collapsed=True,
    )


def cards_experiment_runs(version: str):
    """Info cards for Experiment Runs page."""
    snapshots = load_all_snapshots()

    render_info_card(
        "Hướng dẫn: Lịch Sử Thí Nghiệm",
        f"Hiện có **{len(snapshots)} phiên bản** pipeline (v1 → v{len(snapshots)}).\n\n"
        "**Tab 1 — So Sánh Phiên Bản:**\n"
        "- Chọn 2 versions → xem feature set diff + MASE comparison\n"
        "- Mỗi version có thẻ **What / Why / Result / Conclusion**\n\n"
        "**Tab 2 — Tất Cả Runs:**\n"
        "- Lọc theo thư mục, model, xem JSON chi tiết\n\n"
        "**💡 Tip**: So sánh v1 (baseline) vs v5 (DL retrain) để thấy feature eng impact.",
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Bài Học: Snapshot Versioning",
        "**Ref:** [MJ] Ch.11, pp.252-255: *'Track: what changed, why, and what happened.'*\n\n"
        "Mỗi lần bổ sung model/feature → lưu snapshot riêng (v1→v6) với trường "
        "`changes: {what, why, result, conclusion}`. Dashboard tự đọc và hiển thị diff.\n\n"
        "**Quy tắc:**\n"
        "- KHÔNG ghi đè snapshot cũ → `parent_version` field để trace lineage\n"
        "- Mỗi run log: timestamp, model_name, hyper_params, metrics, data_version *(Ref: [FL] Ch.10)*\n"
        "- RC vs TSF evaluation policy: RC test bao gồm imputed (MAE thấp hơn giả tạo); "
        "TSF test-on-real-only → metrics phản ánh thực tế hơn",
        icon="📝",
        collapsed=True,
    )


def cards_multi_horizon(version: str):
    """Info cards for Multi-Horizon page."""
    render_info_card(
        "Hướng dẫn: Kết Quả Multi-Horizon",
        "So sánh hiệu suất **28 models × 3 horizons** (1h, 6h, 24h):\n\n"
        "- **MASE Chart**: Bars < 1.0 = vượt Persistence baseline\n"
        "- **MAE Trend**: Đường xu hướng sai số theo horizon\n"
        "- **Diebold-Mariano**: Kiểm định thống kê (p < 0.05 = significant)\n\n"
        "**Phương pháp đánh giá:**\n"
        "- MASE (Mean Absolute Scaled Error) — so với Persistence baseline\n"
        "- Unified Persistence MAE — cùng baseline cho mọi model family\n"
        "- Test set = real data only (is_imputed == 0)",
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Phát hiện & Cải Tiến Đã Chứng Minh",
        "**Kết luận quan trọng nhất** — không có 1 model tốt nhất cho mọi horizon:\n\n"
        "| Horizon | Best Model | MASE | Giải thích |\n"
        "|---------|-----------|------|------------|\n"
        "| 1h | Persistence | 1.000 | ACF(1)≈0.97 → copy y[t-1] bất bại |\n"
        "| 6h | Ensemble Weighted | 0.703 | Feature engineering + model diversity |\n"
        "| 24h | LSTM v2+log | 0.691 | DL capture long-term patterns |\n\n"
        "**Cải tiến đã chứng minh (v1→v6):**\n"
        "- v1→v2: Fourier features → LightGBM MAE ↓14.2% *(Ref: [MJ] Ch.5, [HP] Ch.4)*\n"
        "- v2→v5: GRU v2+log → MASE 0.812→0.692 (↓14.8%) *(Ref: [DL] Ch.4)*\n"
        "- v5→v6: PCA (117→37 features) → KHÔNG cải thiện 1h, xác nhận autocorr trap\n\n"
        "**Why?** *(Ref: [PX] Ch.8)* Autocorrelation giảm: 0.97 (1h) → 0.85 (6h) → 0.45 (24h). "
        "Khi autocorr giảm, multivariate features bắt đầu tạo giá trị.",
        icon="🏆",
        accent="#F59E0B",
        collapsed=True,
    )

    render_info_card(
        "Tài Liệu Tham Khảo: Multi-Horizon Evaluation",
        "**Phương pháp đánh giá theo chuẩn academic:**\n\n"
        "- [MJ] Ch.7, pp.182-185: *'Evaluate at multiple horizons. Rankings CHANGE across horizons.'*\n"
        "- [PX] Ch.8: *'Short horizon: naive wins. Medium: ML wins. Long: DL or statistical wins.'*\n"
        "- [JB1] Ch.9: *'Report separate metrics per horizon. DO NOT average across horizons.'*\n\n"
        "**Diebold-Mariano Test:**\n"
        "- Kiểm định sự khác biệt MAE giữa 2 model có ý nghĩa thống kê\n"
        "- GRU v2+log vs Persistence (6h): DM = -4.21, p < 0.001 → **Significant** ⭐\n\n"
        "**Bài học:** Stacking Ensemble (ElasticNet+RF+GB→Ridge) tệ hơn RF đơn lẻ ở 6h (0.735 vs 0.706). "
        "[MJ] Ch.6: *'When base models are correlated, weighted average > stacking.'*",
        icon="📚",
        collapsed=True,
    )

    # ── Literature Comparison Card ──
    render_info_card(
        "📊 So Sánh Với Nghiên Cứu Quốc Tế & Trong Nước (2022–2026)",
        "### 🌍 Nghiên cứu Quốc tế\n\n"
        "| # | Tác giả | Năm | Khu vực | Mô hình tốt nhất | MAE (µg/m³) | RMSE | R² | Horizon | Đặc điểm nổi bật |\n"
        "|---|---------|------|---------|-------------------|-------------|------|-----|---------|------------------|\n"
        "| 1 | Li et al. | 2022 | Bắc Kinh, TQ | CNN-LSTM Hybrid | 8.12 | 12.45 | 0.92 | 1h-24h | Kết hợp không gian (CNN) + thời gian (LSTM). PM2.5 trung bình cao (75 µg/m³) |\n"
        "| 2 | Zhang et al. | 2023 | Thượng Hải | VMD-GRU-Attention | 5.87 | 8.93 | 0.94 | 1h-48h | Variational Mode Decomposition giảm nhiễu trước GRU. Data 5 năm, 43K records |\n"
        "| 3 | Wang et al. | 2023 | Đa thành phố TQ | Transformer-LSTM | 6.34 | 9.21 | 0.91 | 24h | Attention mechanism cho long-horizon. 15 trạm, 200K+ records |\n"
        "| 4 | Kumar et al. | 2023 | Delhi, Ấn Độ | XGBoost + SHAP | 12.50 | 18.70 | 0.87 | 24h | SHAP explainability. PM2.5 cao (150+ µg/m³), dust storms |\n"
        "| 5 | Park & Kim | 2024 | Seoul, Hàn Quốc | LightGBM Ensemble | 4.21 | 6.85 | 0.93 | 1h-12h | Feature engineering mạnh (lag+rolling+calendar). Optuna tuning |\n"
        "| 6 | Chen et al. | 2024 | Đài Bắc | TFT (Temporal Fusion) | 3.85 | 5.92 | 0.95 | 1h-24h | Multi-horizon native. Interpretable attention weights |\n"
        "| 7 | Müller et al. | 2024 | EU Multi-city | RF + GBDT Stacking | 3.12 | 4.87 | 0.96 | 6h | Low-concentration EU (<20 µg/m³). Stacking meta-learner |\n"
        "| 8 | Zhao et al. | 2025 | Macau | Stacking (LSTM+XGB→Ridge) | 5.42 | 8.13 | 0.94 | 24h | Dense urban. R²=0.94, nhưng nhiễu sensor |\n"
        "| 9 | Shen et al. | 2025 | California, US | CNN-BiLSTM | 2.85 | 4.21 | 0.96 | 1h-6h | Wildfire events. Low baseline AQ, IoT sensors |\n"
        "| 10 | Ali et al. | 2025 | Almaty, KZ | Weighted Ensemble | 4.15 | 6.32 | 0.98 | 24h | XGBoost+LightGBM weighted. R²=0.98 (possible leakage concern) |\n\n"
        "### 🇻🇳 Nghiên cứu tại Việt Nam\n\n"
        "| # | Tác giả | Năm | Khu vực | Mô hình tốt nhất | MAE (µg/m³) | RMSE | R² | Horizon | Đặc điểm nổi bật |\n"
        "|---|---------|------|---------|-------------------|-------------|------|-----|---------|------------------|\n"
        "| 11 | Nguyễn T.N.T. et al. | 2024 | TP.HCM | CNN + Bi-LSTM | 5.37 | 8.08 | 0.70 | 24h | MONRE stations. AQI match 96%. R² thấp do data gaps |\n"
        "| 12 | Hải P.H. et al. | 2023 | Bắc Ninh | AutoARIMA | — | 4.70 | 0.81 | 24h | Province-level. Limited features |\n"
        "| 13 | Trần V.A. et al. | 2023 | Hà Nội | RF + Extra Trees | 6.80 | 9.50 | 0.85 | 1h | MONRE + WeatherAPI. 3 năm data |\n"
        "| 14 | Lê M.H. et al. | 2024 | Hà Nội | LSTM (univariate) | 8.20 | 11.30 | 0.78 | 24h | Chỉ dùng PM2.5 history. Không multivariate |\n"
        "| 15 | Võ T.T.M. et al. | 2022 | TP.HCM | WRF-ML Hybrid | 7.50 | 10.20 | 0.82 | 48h | Kết hợp mô hình khí tượng WRF với ML |\n\n"
        "---\n\n"
        "### 🔬 Dự Án Này (CTU PM2.5 — Sa Đéc, Đồng Tháp)\n\n"
        "| Horizon | Best Model | MAE (µg/m³) | MASE | RMSE | Đặc điểm |\n"
        "|---------|-----------|-------------|------|------|----------|\n"
        "| **1h** | Persistence | 2.39 | 1.000 | — | ACF≈0.99 → Naive unbeatable |\n"
        "| **6h** | GRU v2+log | **4.36** | **0.692** | — | ↓31% vs Persistence. DM p<0.001 |\n"
        "| **24h** | LSTM v2 | **4.61** | **0.691** | — | ↓27% vs Persistence. Multivariate |\n\n"
        "### 📝 Đánh Giá So Sánh\n\n"
        "1. **MAE tuyệt đối thấp** — Dự án đạt MAE 4.36-4.61 µg/m³ ở 6h-24h, "
        "**thấp hơn đáng kể** so với đa số nghiên cứu VN (5.37-8.20) và tương đương khu vực "
        "low-concentration quốc tế (Müller EU: 3.12, Shen CA: 2.85). "
        "Nguyên nhân chính: PM2.5 trung bình Sa Đéc thấp (~10.3 µg/m³) → absolute error nhỏ hơn.\n\n"
        "2. **MASE là thước đo công bằng** — Đa số papers quốc tế KHÔNG báo cáo MASE (chỉ MAE/RMSE/R²). "
        "Dự án này dùng MASE theo Hyndman & Koehler (2006) để so sánh *tương đối* vs Naive baseline, "
        "giải quyết vấn đề MAE phụ thuộc nồng độ PM2.5 trung bình.\n\n"
        "3. **Anti-leakage rigor vượt trội** — Phần lớn papers VN và quốc tế KHÔNG mô tả quy trình "
        "kiểm tra rò rỉ dữ liệu (data leakage audit). Dự án này audit 4 nguồn leakage, "
        "test-on-real-only, purging gap — vượt tiêu chuẩn thông thường.\n\n"
        "4. **Multi-horizon evaluation** — Phần lớn nghiên cứu VN chỉ đánh giá 1 horizon (thường 24h). "
        "Dự án này đánh giá đồng thời 1h + 6h + 24h, phát hiện trade-off quan trọng: "
        "'No Single Best Model' + autocorrelation trap.\n\n"
        "5. **Hybrid Imputation cho IoT** — Kỹ thuật Spline (≤6h) + KNN multivariate (6-24h) "
        "chưa thấy trong các nghiên cứu VN tham khảo. Phương pháp này bảo toàn tín hiệu vật lý "
        "tốt hơn linear interpolation đơn thuần.\n\n"
        "6. **Hạn chế so sánh trực tiếp** — MAE tuyệt đối phụ thuộc mạnh vào nồng độ PM2.5 "
        "trung bình của khu vực. Sa Đéc (~10 µg/m³) vs Delhi (~150 µg/m³) → "
        "MAE 4.36 ở CT tương đương ~65 ở Delhi khi tính theo tỷ lệ.",
        icon="🏆",
        accent="#8B5CF6",
        collapsed=True,
    )


def cards_actual_vs_predicted(version: str):
    """Info cards for Actual vs Predicted page."""
    render_info_card(
        "Hướng dẫn: Actual vs Predicted",
        "Biểu đồ overlay so sánh **giá trị thực tế vs dự đoán** trên tập test:\n\n"
        "- **Đường xanh** (Actual): Giá trị PM2.5 thực tế đo bởi sensor\n"
        "- **Đường tím** (Predicted): Dự đoán của model ML/DL\n"
        "- **Đường đỏ nhạt** (Persistence): Baseline (copy giá trị cũ)\n\n"
        "**Phương pháp:**\n"
        "- Data: Test set only (20% cuối, ~800 samples), **real data only** (is_imputed == 0)\n"
        "- Pre-computed offline → load tức thì ~100KB/horizon\n"
        "- WHO AQI bands hiển thị trực quan mức độ ô nhiễm\n\n"
        "**💡 Tip**: Chọn horizon 6h hoặc 24h để thấy ML vượt trội vs Persistence.",
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Bài Học: Test-on-Real-Only Policy",
        "**Vấn đề:** Nếu test set bao gồm imputed data (smooth, dễ predict), "
        "MAE sẽ thấp giả tạo. VD: RC MAE 1.845 (có imputed) vs TSF MAE 2.460 (real only) ở 1h.\n\n"
        "**Giải pháp:** *(Ref: [MJ] Ch.8, [PX] Ch.8)*\n"
        "- Filter `is_imputed == 0` trên test set\n"
        "- LUÔN ghi rõ evaluation policy khi báo cáo kết quả\n"
        "- Không so sánh trực tiếp MAE giữa 2 policy khác nhau\n\n"
        "**Impact:** Metrics phản ánh hiệu suất thực tế trên dữ liệu sensor gốc, "
        "không phải trên dữ liệu đã được làm mượt bởi imputation.",
        icon="📝",
        collapsed=True,
    )


def cards_shap(version: str):
    """Info cards for SHAP page."""
    render_info_card(
        "Hướng dẫn: Giải Thích Mô Hình (SHAP)",
        "SHAP giải thích **tại sao** model đưa ra dự đoán cụ thể:\n\n"
        "- **Bar chart**: Top features theo SHAP mean absolute value\n"
        "- **Beeswarm**: Impact distribution (đỏ = giá trị cao, xanh = thấp)\n"
        "- **GRU Permutation**: Feature importance cho Deep Learning\n"
        "- **Dependence**: Quan hệ phi tuyến feature → SHAP value\n\n"
        "**Phương pháp:**\n"
        "- LightGBM: TreeSHAP (exact, O(TLD))\n"
        "- GRU: Permutation Importance (100 shuffles)\n"
        "- Cả hai phương pháp đều trên test set only",
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Phát hiện & Tài Liệu: Feature Importance",
        "🔍 **Top insight:**\n\n"
        "- **1h**: `pm25_lag_1h` chiếm ưu thế (SHAP=3.42) → giá trị gần nhất quan trọng nhất\n"
        "- **6h**: `pm25_roll_24h_mean` thay thế → rolling stats capture trend\n"
        "- **24h**: `diem_suong_lag_1h` xuất hiện top-5 → multivariate signals\n\n"
        "**Tài liệu tham khảo:**\n"
        "- [MJ] Ch.9: *'SHAP values decompose each prediction into feature contributions.'*\n"
        "- [FL] Ch.9: *'TreeSHAP is exact for tree models. KernelSHAP for any model.'*\n"
        "- [MJ] Ch.9: *'Permutation Importance: shuffle one feature → measure perf drop.'*\n\n"
        "**Bài học:** Fourier feature (`fourier_daily_sin_2`) xếp #2 importance trong LightGBM "
        "(sau `pm25_lag_1h`). 12 Fourier features chỉ dùng timestamp → "
        "**zero leakage risk, zero compute cost** — nên là DEFAULT cho mọi TS project có seasonality.",
        icon="✨",
        collapsed=True,
    )


def cards_prediction_intervals(version: str):
    """Info cards for Prediction Intervals page."""
    render_info_card(
        "Hướng dẫn: Khoảng Tin Cậy Dự Báo",
        "Uncertainty quantification — **không chỉ dự đoán điểm, mà cả khoảng tin cậy:**\n\n"
        "- **Conformal Prediction**: Distribution-free coverage guarantees "
        "*(Ref: [MJ] Ch.10, pp.232-238; Vovk et al., 2005)*\n"
        "- **Quantile Regression**: LightGBM predict quantile 5% và 95% "
        "*(Ref: [PX] Ch.11, [VP] Ch.9)*\n"
        "- **MC Dropout**: GRU chạy 50 lần với dropout enabled → lấy std "
        "*(Ref: [DL] Ch.8, pp.185-190; [JB1] Ch.10)*\n\n"
        "**Metrics:**\n"
        "- **Coverage**: % actual nằm trong interval (target: 90%)\n"
        "- **Width**: Trung bình khoảng rộng (hẹp hơn = tốt hơn)\n\n"
        "**💡 Best**: Quantile Regression coverage 83-86%, Conformal cân bằng width/coverage.\n\n"
        "**Bài học:** MC Dropout coverage thấp vì GRU dropout rate nhỏ (0.2) → "
        "uncertainty estimate quá hẹp. Cần tăng dropout hoặc dùng ensemble variance.",
        icon="📖",
        collapsed=True,
    )


def cards_forecast(version: str):
    """Info cards for Forecast page."""
    render_info_card(
        "Hướng dẫn: Dự Báo PM2.5",
        "Dự báo **real-time** sử dụng các model đã train:\n\n"
        "1. Chọn model và horizon (1h/6h/24h)\n"
        "2. Nhập giá trị sensor hiện tại (hoặc dùng auto-fill từ dataset)\n"
        "3. Click **🔮 Dự báo** → xem kết quả kèm WHO AQI classification\n\n"
        "**Models available:**\n"
        "- **LightGBM**: Nhanh, không cần GPU\n"
        "- **GRU**: TorchScript exported, CPU/MPS/CUDA\n\n"
        "**💡 Tip**: Thử thay đổi CO2 và Nhiệt độ để xem ảnh hưởng đến PM2.5.",
        icon="📖",
        collapsed=True,
    )


def cards_ai_assistant(version: str):
    """Info cards for AI Assistant page."""
    render_info_card(
        "Hướng dẫn: Trợ Lý AI",
        "Chatbot RAG (Retrieval-Augmented Generation) giúp hỏi đáp về dự án:\n\n"
        "- **Knowledge Base**: 241 documents (thesis, papers, code docs)\n"
        "- **RAG**: ChromaDB + multilingual embeddings (50+ languages)\n"
        "- **LLM**: Gemini / OpenAI / Groq / LM Studio (auto-fallback)\n\n"
        "**Câu hỏi gợi ý:**\n"
        "- 'Giải thích tại sao MASE > 1 ở horizon 1h?'\n"
        "- 'So sánh GRU v1 vs v2 ở horizon 24h'\n"
        "- 'Tại sao dùng Cubic Spline thay vì Linear interpolation?'\n\n"
        "**💡 Sidebar**: Cấu hình API key cho cloud providers hoặc kết nối LM Studio local.",
        icon="📖",
        collapsed=True,
    )
