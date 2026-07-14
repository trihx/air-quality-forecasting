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
    """Load all version snapshots from dashboard_runs/, normalized."""
    from src.snapshot_adapter import load_all_normalized
    return load_all_normalized()


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

    default_idx = len(versions) - 1
    if "v9_multi_resolution" in versions:
        default_idx = versions.index("v9_multi_resolution")

    selected = st.sidebar.selectbox(
        "Chọn phiên bản",
        versions,
        index=default_idx,
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
        st.markdown(content, unsafe_allow_html=True)


def render_version_badge(version: str):
    """Render a small version badge at top of page."""
    if not version:
        return
    v_data = get_version_data(version)
    n_models = len(v_data.get("models", []))
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

    from src.reporting.content import ContentManager
    content = ContentManager()
    
    render_info_card(
        "Hướng dẫn: Tổng Quan",
        content.get_info_card_text("overview_guide", "Trang này trình bày bức tranh toàn cảnh..."),
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
        content.get_info_card_text("overview_lessons", "Đang cập nhật..."),
        icon="📝",
        collapsed=True,
    )

    render_info_card(
        "Cải Tiến Đã Chứng Minh (v1 → v9)",
        content.get_info_card_text("overview_improvements", "Đang cập nhật..."),
        icon="📊",
        collapsed=True,
    )


def cards_eda(version: str):
    """Info cards for EDA page."""
    from src.reporting.content import ContentManager
    content = ContentManager()
    
    render_info_card(
        "Hướng dẫn: Phân Tích Khám Phá (EDA)",
        content.get_info_card_text("eda_guide", "Đang cập nhật..."),
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Phát hiện thú vị từ EDA",
        content.get_info_card_text("eda_findings", "Đang cập nhật..."),
        icon="✨",
        accent="#F59E0B",
        collapsed=True,
    )

    render_info_card(
        "Bài Học: Outlier Detection cho IoT",
        content.get_info_card_text("eda_lessons", "Đang cập nhật..."),
        icon="🧪",
        collapsed=True,
    )


def cards_hyperparams(version: str):
    """Info cards for Hyperparameters page."""
    v = get_version_data(version)
    from src.reporting.content import ContentManager
    content = ContentManager()

    render_info_card(
        "Hướng dẫn: Hyperparameters",
        content.get_info_card_text("hyperparams_guide", "Đang cập nhật..."),
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

    from src.reporting.content import ContentManager
    content = ContentManager()
    
    render_info_card(
        "Phương pháp & Tài liệu tham khảo",
        content.get_info_card_text("methodology_references", "Đang cập nhật..."),
        icon="🔧",
        collapsed=True,
    )

    render_info_card(
        "Bài Học: Pitfalls khi Training",
        content.get_info_card_text("pitfalls_lessons", "Đang cập nhật..."),
        icon="⚠️",
        collapsed=True,
    )


def cards_experiment_runs(version: str):
    """Info cards for Experiment Runs page."""
    snapshots = load_all_snapshots()

    from src.reporting.content import ContentManager
    content = ContentManager()
    
    render_info_card(
        "Hướng dẫn: Lịch Sử Thí Nghiệm",
        content.get_info_card_text("experiment_runs_guide", "Đang cập nhật...").format(len=len(snapshots)),
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Bài Học: Snapshot Versioning",
        content.get_info_card_text("experiment_runs_lessons", "Đang cập nhật..."),
        icon="📝",
        collapsed=True,
    )


def cards_multi_horizon(version: str):
    """Info cards for Multi-Horizon page."""
    from src.reporting.content import ContentManager
    content = ContentManager()
    
    render_info_card(
        "Hướng dẫn: Kết Quả Multi-Horizon",
        content.get_info_card_text("multi_horizon_guide", "Đang cập nhật..."),
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Phát hiện & Cải Tiến Đã Chứng Minh",
        content.get_info_card_text("multi_horizon_findings", "Đang cập nhật..."),
        icon="🏆",
        accent="#F59E0B",
        collapsed=True,
    )

    render_info_card(
        "Tài Liệu Tham Khảo: Multi-Horizon Evaluation",
        content.get_info_card_text("multi_horizon_references", "Đang cập nhật..."),
        icon="📚",
        collapsed=True,
    )

    # The literature tables are now rendered dynamically in app.py
    # But this summary comparison text can also be stored and loaded:
    render_info_card(
        "📝 Đánh Giá So Sánh",
        content.get_info_card_text("multi_horizon_comparison", "Đang cập nhật..."),
        icon="🏆",
        accent="#8B5CF6",
        collapsed=True,
    )


def cards_actual_vs_predicted(version: str):
    """Info cards for Actual vs Predicted page."""
    from src.reporting.content import ContentManager
    content = ContentManager()
    
    render_info_card(
        "Hướng dẫn: Actual vs Predicted",
        content.get_info_card_text("actual_vs_predicted_guide", "Đang cập nhật..."),
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Bài Học: Test-on-Real-Only Policy",
        content.get_info_card_text("actual_vs_predicted_lessons", "Đang cập nhật..."),
        icon="📝",
        collapsed=True,
    )


def cards_shap(version: str):
    """Info cards for SHAP page."""
    from src.reporting.content import ContentManager
    content = ContentManager()
    
    render_info_card(
        "Hướng dẫn: Giải Thích Mô Hình (SHAP)",
        content.get_info_card_text("shap_guide", "Đang cập nhật..."),
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Phát hiện & Tài Liệu: Feature Importance",
        content.get_info_card_text("shap_lessons", "Đang cập nhật..."),
        icon="✨",
        collapsed=True,
    )


def cards_prediction_intervals(version: str):
    """Info cards for Prediction Intervals page."""
    from src.reporting.content import ContentManager
    from src.frontend.citations import cite
    content = ContentManager()
    
    render_info_card(
        "Hướng dẫn: Khoảng Tin Cậy Dự Báo",
        content.get_info_card_text("pi_guide", "Đang cập nhật..."),
        icon="📖",
        collapsed=True,
    )

    render_info_card(
        "Tài Liệu Tham Khảo: Ước Lượng Bất Định (Uncertainty Estimation)",
        f"- **Adaptive Conformal Inference (ACI):** {cite('gibbs2021')}\n"
        f"- **Conformalized Quantile Regression (CQR):** {cite('romano2019')}\n"
        f"- **MC Dropout / Deep Ensembles:** {cite('gal2016')} & {cite('lakshminarayanan2017')} (Từng dùng làm Baseline)",
        icon="📚",
        collapsed=False,
    )


def cards_forecast(version: str):
    """Info cards for Forecast page."""
    from src.reporting.content import ContentManager
    content = ContentManager()
    
    render_info_card(
        "Hướng dẫn: Dự Báo PM2.5",
        content.get_info_card_text("forecast_guide", "Đang cập nhật..."),
        icon="📖",
        collapsed=True,
    )


def cards_ai_assistant(version: str):
    """Info cards for AI Assistant page."""
    from src.reporting.content import ContentManager
    content = ContentManager()
    
    render_info_card(
        "Hướng dẫn: Trợ Lý AI",
        content.get_info_card_text("ai_assistant_guide", "Đang cập nhật..."),
        icon="📖",
        collapsed=True,
    )
