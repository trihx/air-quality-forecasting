"""Actual vs Predicted performance comparison page across forecasting horizons."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.frontend.citations import cite, render_references_section
from src.frontend.components import PROJECT_ROOT, insight_card, section_header
from src.info_cards import (
    cards_actual_vs_predicted,
    get_current_version,
    get_version_data,
    render_version_badge,
)
from src.reporting.content import ContentManager
from src.snapshot_adapter import HORIZONS, TOP_N
from src.viz.chart_factory import (
    chart as _chart,
)
from src.viz.chart_factory import (
    figure_caption as _caption,
)
from src.viz.chart_factory import (
    render_chart as _render_chart,
)
from src.viz.theme import PALETTE_CATEGORICAL

CACHE_DIR = PROJECT_ROOT / "research" / "cache"


@st.cache_data(ttl=3600)
def _load_avp_cache(horizon: int) -> dict[str, Any] | None:
    """Load pre-computed Actual vs Predicted data from cache file."""
    cache_file = CACHE_DIR / f"avp_{horizon}h.json"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    return None


def page_actual_vs_predicted(results: dict[str, Any]) -> None:
    """Render Actual vs Predicted performance overlay page."""
    st.markdown(
        """
    <h1 style="font-size: 2rem;">📉 Thực Tế vs Dự Báo (Actual vs Predicted)</h1>
    <p style="opacity: 0.7;">Đối chuẩn so sánh giá trị thực tế và dự báo từ các mô hình trên tập kiểm thử mỏ neo</p>
    """,
        unsafe_allow_html=True,
    )

    ver = get_current_version()
    render_version_badge(ver)
    cards_actual_vs_predicted(ver)

    st.markdown(
        f"""
    <div style="background: var(--secondary-background-color); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.2); color: var(--text-color) !important;">
        <div style="font-size: 0.85rem; opacity: 0.85;">
            📌 Đánh giá trên <b>Anchor Test Set</b> (10% mỏ neo: 669h ở chuỗi 1h, 863 mẫu ở 30m, 1.836 mẫu ở 15m, 100% dữ liệu thực <code>is_imputed == 0</code>) {cite("tashman2000")}.
            Metrics chính: <b>MASE</b> {cite("hyndman2006")} (scale-independent), MAE {cite("willmott2005")},
            kiểm định Diebold-Mariano {cite("diebold1995")} và khoảng dự báo thích ứng ACI {cite("gibbs2021")}.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    horizon = st.selectbox(
        "⏱️ Chọn horizon",
        [1, 6, 24],
        index=1,
        format_func=lambda x: f"{x} giờ",
    )

    data = _load_avp_cache(horizon)

    if data is not None:
        _render_avp_chart(data, horizon, ver)
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

    render_references_section()


def _render_avp_chart(data: dict[str, Any], horizon: int, ver: str) -> None:
    """Render the Actual vs Predicted chart from cached data."""
    model_preds = data.get("model_preds", {})
    if not model_preds:
        if data.get("gru_preds"):
            model_preds["GRU"] = data["gru_preds"]
        if data.get("lgbm_preds"):
            model_preds["LightGBM"] = data["lgbm_preds"]

    v_data = get_version_data(ver)
    h_key = f"{horizon}h"

    top_models_info = v_data.get("top_n", {}).get(h_key, [])
    best_model_name = top_models_info[0]["model"] if top_models_info else ""
    if best_model_name.endswith("_15m"):
        default_res_idx = 0
    elif best_model_name.endswith("_30m"):
        default_res_idx = 1
    else:
        default_res_idx = 2

    res_options = ["15m", "30m", "1h"]
    res_labels = []
    for res in res_options:
        suffix = f"_{res}"
        res_models = [k for k in model_preds if k.endswith(suffix) and not k.startswith("Persistence_")]
        n_models = len(res_models)
        is_best = any(m["model"].endswith(suffix) for m in top_models_info[:1])
        label = f"{res} ({n_models} models)"
        if is_best:
            label += " ⭐ Best"
        res_labels.append(label)

    selected_res = st.radio(
        "📐 Độ phân giải dữ liệu",
        res_options,
        index=default_res_idx,
        format_func=lambda x: res_labels[res_options.index(x)],
        horizontal=True,
        help="Mỗi độ phân giải có tập Test riêng (số lượng điểm và khoảng thời gian khác nhau). "
        "Chọn resolution để xem mô hình được train tại tần suất đó.",
    )

    res_suffix = f"_{selected_res}"
    filtered_models = [k for k in model_preds if k.endswith(res_suffix) and not k.startswith("Persistence_")]

    actuals_multi = data.get("actuals_multi", {})
    if selected_res in actuals_multi and actuals_multi[selected_res]:
        test_actuals = np.array(actuals_multi[selected_res])
    else:
        test_actuals = np.array(data.get("actuals", []))

    persist_key = f"Persistence_{selected_res}"
    if persist_key in model_preds:
        test_persist = np.array(model_preds[persist_key])
    else:
        test_persist = np.array(data.get("persistence", []))

    snapshot_top_names = [m["model"] for m in top_models_info]
    default_selection = [m for m in snapshot_top_names if m in filtered_models]
    for m in filtered_models:
        if m not in default_selection:
            default_selection.append(m)
        if len(default_selection) >= TOP_N:
            break

    if filtered_models:
        selected_models = st.multiselect(
            "🎯 Chọn mô hình hiển thị trên biểu đồ",
            options=filtered_models,
            default=default_selection[:TOP_N],
            help=f"Hiển thị các mô hình được train trên dữ liệu {selected_res}. Mặc định: Top models theo MASE.",
        )
    else:
        selected_models = []
        st.info(f"ℹ️ Chưa có mô hình nào được train trên dữ liệu {selected_res} cho horizon {horizon}h.")

    n_test = len(test_actuals)
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
                border: 1px solid rgba(0,212,170,0.2); border-radius: 12px;
                padding: 1rem 1.5rem; margin: 1rem 0;">
        <span style="opacity: 0.65; font-size: 0.85rem;">
            📊 Test samples: <b style="color:#00D4AA">{n_test}</b> (real data only, {selected_res}) |
            Horizon: <b style="color:#00D4AA">{horizon}h</b> |
            Trực quan hóa: <b style="color:#00D4AA">{len(selected_models)}/{len(filtered_models)}</b> mô hình
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig = _chart(
        xaxis_title="Test Sample Index",
        yaxis_title="PM2.5 (µg/m³)",
        height=520,
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(n_test)),
            y=test_actuals,
            name=f"Actual ({selected_res})",
            line={"color": "#1E293B", "width": 2.5},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(test_persist))),
            y=test_persist,
            name=f"Persistence ({selected_res})",
            line={"color": "#94A3B8", "width": 1.5, "dash": "dash"},
        )
    )

    model_palette = {
        "GRU": "#00D4AA",
        "LightGBM": "#FF6B6B",
        "Ensemble_Stack": "#7C3AED",
        "Ensemble_GRU": "#F59E0B",
        "Ensemble_Weighted": "#EC4899",
        "TFT": "#3B82F6",
        "ARIMA": "#10B981",
        "SARIMA": "#6366F1",
        "LSTM": "#8B5CF6",
        "ElasticNet": "#F97316",
    }
    fallback_colors = PALETTE_CATEGORICAL

    for idx, model_name in enumerate(selected_models):
        preds = model_preds.get(model_name)
        if not preds:
            continue

        if len(preds) > n_test:
            preds = preds[-n_test:]

        preds_clean = [p if p is not None else np.nan for p in preds]
        base_name = model_name.rsplit("_v9_", 1)[0] if "_v9_" in model_name else model_name
        base_name_clean = base_name.replace("_expert", "").replace("_v9", "")
        color = model_palette.get(base_name_clean, fallback_colors[idx % len(fallback_colors)])
        fig.add_trace(
            go.Scatter(
                x=list(range(len(preds_clean))),
                y=preds_clean,
                name=f"{model_name} ({horizon}h)",
                line={"color": color, "width": 2},
            )
        )

    _render_chart(fig, filename=f"actual_vs_predicted_{horizon}h")
    _caption(f"Actual vs Predicted — Horizon {horizon}h (Test Set, {selected_res} Resolution)")

    unavailable_top = [m["model"] for m in top_models_info if m["model"] not in filtered_models]
    if unavailable_top:
        other_res = [r for r in res_options if r != selected_res]
        st.caption(
            f"*(ℹ️ Các mô hình {', '.join(unavailable_top)} nằm trong Top {TOP_N} nhưng thuộc resolution khác. "
            f"Chuyển sang tab {'/'.join(other_res)} để xem.)*"
        )
    else:
        st.caption(
            f"*(Ghi chú: Biểu đồ hiển thị mô hình được train trên dữ liệu {selected_res} cho horizon {horizon}h. "
            "Bảng Metrics bên dưới liệt kê toàn bộ các mô hình.)*"
        )

    if data.get("gru_error"):
        st.warning(f"⚠️ GRU: {data['gru_error']}")
    if data.get("lgbm_error"):
        st.warning(f"⚠️ LightGBM: {data['lgbm_error']}")

    rank_metric = st.radio(
        "📊 Xếp hạng theo",
        ["MAE", "MASE"],
        index=1,
        horizontal=True,
        help="MAE = sai số tuyệt đối (µg/m³). MASE = so với Persistence baseline (< 1.0 = tốt hơn).",
    )

    st.markdown(f"### 🏆 Top {TOP_N} Mô Hình Tốt Nhất (theo {rank_metric})")
    st.caption(
        f"*Persistence Baseline được loại khỏi xếp hạng. {'MASE < 1.0 = tốt hơn Persistence.' if rank_metric == 'MASE' else 'MAE thấp hơn = dự đoán chính xác hơn.'}*"
    )
    st.caption(
        "*(RMSE, R², DA là các chỉ số tham khảo bổ sung. **RMSE** nhạy cảm với outliers nên không dùng làm metric xếp hạng chính cho dữ liệu IoT PM2.5.)*"
    )

    cols = st.columns(len(HORIZONS))
    for idx, h_col_key in enumerate(HORIZONS):
        h_val = int(h_col_key.replace("h", ""))

        h_results_for_rank = v_data.get("results", {}).get(h_col_key, {})
        all_models_for_h = []
        for model_name, metrics in h_results_for_rank.items():
            if model_name.startswith("Persistence"):
                continue
            all_models_for_h.append(
                {
                    "model": model_name,
                    "mae": metrics.get("mae", float("inf")),
                    "mase": metrics.get("mase", float("inf")),
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
                    "da": metrics.get("da"),
                }
            )

        sort_key = "mase" if rank_metric == "MASE" else "mae"
        top_models = sorted(all_models_for_h, key=lambda x: x.get(sort_key, float("inf")))[:TOP_N]
        is_selected = h_val == horizon

        with cols[idx]:
            if is_selected:
                st.markdown(
                    f"""
                <style>
                    .avp-tab-selected {{
                        background: var(--text-color) !important;
                        border: 2px solid #00D4AA; border-radius: 10px; padding: 0.8rem; text-align: center;
                    }}
                </style>
                <div class="avp-tab-selected">
                    <b style="color: #00D4AA; font-size: 1.1rem;">⏱️ {h_col_key} (đang xem)</b>
                </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <style>
                    .avp-tab-unselected {{
                        background: var(--secondary-background-color) !important;
                        border: 1px solid rgba(128,128,128,0.3); border-radius: 10px; padding: 0.8rem; text-align: center;
                    }}
                </style>
                <div class="avp-tab-unselected">
                    <b style="color: var(--text-color); font-size: 1.1rem;">⏱️ {h_col_key}</b>
                </div>""",
                    unsafe_allow_html=True,
                )

            if top_models:
                medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                for rank, row in enumerate(top_models):
                    mase_color = "#00D4AA" if row["mase"] < 1.0 else "#FF6B6B"
                    medal = medals[rank] if rank < len(medals) else f"#{rank + 1}"
                    border_color = "#00D4AA" if rank == 0 else "#71717A"
                    rmse_display = f"{row['rmse']:.2f}" if row.get("rmse") else "—"
                    r2_display = f"{row['r2']:.3f}" if row.get("r2") is not None else "—"
                    da_display = f"{row['da']:.1f}%" if row.get("da") is not None else "—"
                    st.markdown(
                        f"""
                    <style>
                        .avp-model-card-{rank} {{
                            background: var(--text-color) !important; border-radius: 8px;
                            padding: 0.6rem 0.8rem; margin: 0.4rem 0;
                            border-left: 3px solid {border_color};
                            border-top: 1px solid rgba(128,128,128,0.2);
                            border-right: 1px solid rgba(128,128,128,0.2);
                            border-bottom: 1px solid rgba(128,128,128,0.2);
                        }}
                    </style>
                    <div class="avp-model-card-{rank}">
                        <div style="font-size: 0.75rem; color: var(--background-color); opacity: 0.8;">{medal} Top {rank + 1}</div>
                        <div style="font-weight: 600; font-size: 0.95rem; color: var(--background-color);">{row["model"]}</div>
                        <div style="font-size: 0.85rem; margin-top: 0.2rem; color: var(--background-color);">
                            MAE: <b>{row["mae"]:.2f}</b> · MASE: <b style="color: {mase_color};">{row["mase"]:.2f}</b>
                        </div>
                        <div style="font-size: 0.78rem; margin-top: 0.15rem; color: var(--background-color); opacity: 0.8;">
                            RMSE: <b>{rmse_display}</b> · R²: <b>{r2_display}</b> · DA: <b>{da_display}</b>
                        </div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"Không có dữ liệu cho {h_col_key}")

    h_selected_key = f"{horizon}h"
    h_results = v_data.get("results", {}).get(h_selected_key, {})
    if h_results:
        all_metrics = []
        for model_name, m in h_results.items():
            rmse_val = m.get("rmse", float("inf"))
            r2_val = m.get("r2")
            da_val = m.get("da")
            bias_val = m.get("forecast_bias")
            all_metrics.append(
                {
                    "Mô hình": model_name,
                    "MAE (µg/m³)": round(m["mae"], 2),
                    "RMSE (µg/m³)": (round(rmse_val, 2) if rmse_val != float("inf") else "—"),
                    "MASE": round(m["mase"], 4),
                    "R²": round(r2_val, 4) if r2_val is not None else "—",
                    "DA (%)": round(da_val, 1) if da_val is not None else "—",
                    "Bias": round(bias_val, 4) if bias_val is not None else "—",
                }
            )
        sort_col = "MASE" if rank_metric == "MASE" else "MAE (µg/m³)"
        all_metrics.sort(
            key=lambda x: x.get(sort_col, float("inf")) if isinstance(x.get(sort_col), (int, float)) else float("inf")
        )
        with st.expander(
            f"📋 Bảng đầy đủ tất cả mô hình — Horizon {horizon}h (xếp theo {rank_metric})",
            expanded=False,
        ):
            st.dataframe(
                pd.DataFrame(all_metrics),
                use_container_width=True,
                hide_index=True,
            )
    elif data.get("metrics"):
        st.markdown(f"### 📋 Tóm tắt Metrics (Reference Models - {horizon}h)")
        st.dataframe(
            pd.DataFrame(data["metrics"]),
            use_container_width=True,
            hide_index=True,
        )

    # ── Chẩn Đoán Thặng Dư & Kiểm Định Ljung-Box (Chương 4 §4.5 Đề Án) ──
    st.markdown("---")
    section_header("🔬", "Chẩn Đoán Thặng Dư & Kiểm Định Tự Tương Quan Ljung-Box (Chương 4 §4.5)")
    st.markdown(
        f"""
    <div style="background: var(--secondary-background-color); border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;
                border-left: 4px solid #00D4AA; font-size: 0.88rem; line-height: 1.65;">
        Phân tích chẩn đoán phần dư ($e_t = y_t - \\hat{{y}}_t$) là bước bắt buộc trong kinh tế lượng chuỗi thời gian
        nhằm kiểm tra các giả định kinh điển về <b>nhiễu trắng (White Noise)</b> {cite("ljung1978")},
        tính đối xứng sai số và độ bất định của dự báo.
    </div>
    """,
        unsafe_allow_html=True,
    )

    content_mgr = ContentManager()
    res_diag_list = content_mgr.get_residual_diagnostics()
    if res_diag_list:
        section_header("📋", "Bảng 4.4: Kết Quả Kiểm Định Tự Tương Quan Phần Dư Ljung-Box (lag=6)")
        df_res = pd.DataFrame(res_diag_list)
        st.dataframe(
            df_res,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Mean": st.column_config.NumberColumn(format="%+.3f µg/m³"),
                "Std": st.column_config.NumberColumn(format="%.3f µg/m³"),
                "Skewness": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(
            "*Ghi chú: Giá trị phần dư được tính theo công thức e_t = y_t - ŷ_t. "
            "Mô hình Persistence tại mốc 6 giờ được đưa vào làm mốc đối chứng chuẩn cho trạng thái suy thoái quán tính.*"
        )

    insight_card(
        "💡 3 Đặc Tính Thống Kê Nổi Bật Của Phần Dư (Chương 4.5 Đề Án)",
        f"<b>1. Đánh đổi Độ chệch - Phương sai (Bias-Variance Tradeoff):</b><br>"
        f"Các mô hình học máy (LightGBM) và học sâu (GRU) có Mean bias mang dấu âm nhẹ (-0,329 đến -2,350 µg/m³), "
        f"thể hiện xu hướng dự báo an toàn (hơi cao hơn thực tế nhằm tránh bỏ sót đỉnh ô nhiễm). "
        f"Trong khi đó, Persistence tại 6h có Mean gần bằng 0 (+0,029 µg/m³) nhưng phương sai cực lớn (<b>Std = 8,272</b>). "
        f"LightGBM 6h chấp nhận Mean = -1,656 µg/m³ nhưng <b>giảm 36,0% độ phân tán</b> (Std = 5,296), "
        f"khẳng định khả năng kiểm soát phương sai vượt trội của Machine Learning {cite('ke2017')}.<br><br>"
        f"<b>2. Phân phối lệch phải rõ rệt (Right-skewed Distribution & Fat Tails):</b><br>"
        f"Tất cả các mô hình ML/DL đều có hệ số độ lệch Skewness dương lớn (<b>1,173 – 1,552 > 0</b>), "
        f"chỉ ra phân phối phần dư có đuôi dày (Fat-tailed) về phía giá trị dương. "
        f"Điều này phản ánh các đỉnh ô nhiễm cực đoan xảy ra đột ngột do hoạt động giao thông giờ cao điểm hoặc đốt sinh khối cục bộ.<br><br>"
        f"<b>3. Bác bỏ giả thuyết nhiễu trắng & Nhu cầu Conformal Prediction:</b><br>"
        f"Kiểm định Ljung-Box tại độ trễ $k=6$ đều cho giá trị <b>p < 0,05</b>, chính thức bác bỏ giả thuyết $H_0$ rằng phần dư là nhiễu trắng thuần túy {cite('ljung1978')}. "
        f"Hệ quả phương pháp luận: <b>Các khoảng tin cậy truyền thống dựa trên giả định phân phối chuẩn đối xứng ($\\hat{{y}} \\pm 1,96\\sigma$) là hoàn toàn không hợp lệ</b>. "
        f"Đây là minh chứng khoa học đanh thép đòi hỏi phải áp dụng phương pháp <b>Conformalized Quantile Regression (CQR)</b> {cite('romano2019')} "
        f"và <b>Adaptive Conformal Inference (ACI)</b> {cite('gibbs2021')} trong Đề án.",
    )

    # ── Hiển thị 7 Hình ảnh Chẩn Đoán Thặng Dư Chuẩn Đề Án (Hình 4.6a-g) ──
    with st.expander("🖼️ Xem Biểu đồ Chẩn Đoán Thặng Dư Chuẩn Đề Án (Hình 4.6a–g)", expanded=False):
        diag_choice = st.selectbox(
            "Chọn biểu đồ chẩn đoán thặng dư:",
            [
                ("Hinh_4.6a_Diagnostics_GRU_1h.png", "Hình 4.6a: Phân tích thặng dư mô hình GRU tại mốc dự báo 1 giờ"),
                (
                    "Hinh_4.6b_Diagnostics_LightGBM_1h.png",
                    "Hình 4.6b: Phân tích thặng dư mô hình LightGBM tại mốc dự báo 1 giờ",
                ),
                ("Hinh_4.6c_Diagnostics_GRU_6h.png", "Hình 4.6c: Phân tích thặng dư mô hình GRU tại mốc dự báo 6 giờ"),
                (
                    "Hinh_4.6d_Diagnostics_LightGBM_6h.png",
                    "Hình 4.6d: Phân tích thặng dư mô hình LightGBM tại mốc dự báo 6 giờ",
                ),
                (
                    "Hinh_4.6e_Diagnostics_Persistence_6h.png",
                    "Hình 4.6e: Phân tích thặng dư mô hình Baseline Persistence tại mốc 6 giờ",
                ),
                (
                    "Hinh_4.6f_Diagnostics_GRU_24h.png",
                    "Hình 4.6f: Phân tích thặng dư mô hình GRU tại mốc dự báo 24 giờ",
                ),
                (
                    "Hinh_4.6g_Diagnostics_LightGBM_24h.png",
                    "Hình 4.6g: Phân tích thặng dư mô hình LightGBM tại mốc dự báo 24 giờ",
                ),
            ],
            format_func=lambda x: x[1],
            key="avp_diag_img_select",
        )
        if diag_choice:
            img_p = PROJECT_ROOT / "research" / "figures" / "thesis" / diag_choice[0]
            if img_p.exists():
                st.image(str(img_p), caption=diag_choice[1], use_container_width=True)
