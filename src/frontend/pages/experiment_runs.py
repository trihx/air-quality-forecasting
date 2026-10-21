"""Experiment Runs page — experiment run histories and version comparisons."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.frontend.citations import cite, render_references_section
from src.frontend.components import RESEARCH_DIR
from src.info_cards import (
    cards_experiment_runs,
    get_current_version,
    render_version_badge,
)
from src.reporting.content import ContentManager
from src.viz.chart_factory import (
    add_baseline,
    add_simple_bar_labels,
)
from src.viz.chart_factory import (
    figure_caption as _caption,
)
from src.viz.chart_factory import (
    render_chart as _render_chart,
)


def page_experiment_runs(results: dict[str, Any]) -> None:
    """Render experiment history and cross-version comparison page."""
    st.markdown(
        """
    <h1 style="font-size: 2rem;">📋 Nhật Ký Thực Nghiệm (Experiment Runs)</h1>
    <p style="opacity: 0.7;">Lịch sử các lượt thực nghiệm và đối chuẩn chéo giữa các phiên bản pipeline</p>
    """,
        unsafe_allow_html=True,
    )

    ver = get_current_version()
    render_version_badge(ver)
    cards_experiment_runs(ver)

    st.markdown(
        f"""
    <div style="background: var(--ds-bg-card); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.2); color: var(--ds-text-primary);">
        <div style="font-size: 0.85rem; color: var(--ds-text-secondary);">
            📌 Pipeline trải qua 9 phiên bản (v1→v9), tuân thủ quy trình nghiên cứu lặp lại {cite("makridakis2020")}.
            Mỗi version snapshot ghi nhận feature set, model configs, và metrics chuẩn hóa MASE {cite("hyndman2006")}.
            So sánh fair giữa các phiên bản sử dụng cùng temporal test set {cite("tashman2000")}.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📊 So Sánh Phiên Bản", "📋 Tất Cả Lượt Thực Nghiệm (Runs)"])

    with tab1:
        _render_version_comparison()

    with tab2:
        _render_all_runs()

    render_references_section()


def _render_version_comparison() -> None:
    """Render version comparison from dashboard_runs/ snapshots."""
    runs_dir = RESEARCH_DIR / "experiments" / "dashboard_runs"
    if not runs_dir.exists() or not list(runs_dir.glob("*.json")):
        st.warning("Chưa có snapshot nào. Chạy `run_enhanced_pipeline.py` để tạo.")
        return

    def _sort_key(p):
        m = re.match(r"^v(\d+)", p.stem)
        return (int(m.group(1)), p.stem) if m else (999, p.stem)

    snapshots: dict[str, dict[str, Any]] = {}
    for jpath in sorted(runs_dir.glob("*.json"), key=_sort_key):
        try:
            with open(jpath, encoding="utf-8") as f:
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
        feat_rows.append(
            {
                "Feature": key,
                v1_name: "✓" if v1_val else "✗",
                v2_name: "✓" if v2_val else "✗",
                "Thay đổi": status,
            }
        )
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

    st.markdown("### 📋 Chi Tiết Phiên Bản (Hành Trình Nâng Cấp Khoa Học)")

    content_mgr = ContentManager()
    evolution_matrix = content_mgr.get_evolution_matrix()

    view_mode = st.radio(
        "Chế độ hiển thị chi tiết phiên bản:",
        ["🔍 Phân tích chuyên sâu từng phiên bản (5 Trụ cột Thực nghiệm)", "📜 Xem tuần tự tất cả các phiên bản"],
        horizontal=True,
        key="exp_runs_version_view_mode",
    )

    if "từng phiên bản" in view_mode:
        snap_names = list(snapshots.keys())
        default_snap_idx = (
            snap_names.index("v9_multi_resolution") if "v9_multi_resolution" in snap_names else len(snap_names) - 1
        )
        sel_snap = st.selectbox(
            "📌 Chọn phiên bản muốn kiểm định chi tiết:",
            snap_names,
            index=default_snap_idx,
            key="exp_runs_sel_snap",
        )
        _render_detailed_version_card(sel_snap, snapshots[sel_snap], evolution_matrix.get(sel_snap, {}))
    else:
        for v_name, v_data in snapshots.items():
            _render_detailed_version_card(v_name, v_data, evolution_matrix.get(v_name, {}))

    st.markdown(f"### 📊 MASE — So Sánh {v1_name} vs {v2_name}")

    v1_results = v1.get(
        "results",
        v1.get("metrics", v1.get("data", {}).get("results", {})),
    )
    v2_results = v2.get(
        "results",
        v2.get("metrics", v2.get("data", {}).get("results", {})),
    )

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
            v1_mase = v1_m.get(
                "mase_unified",
                v1_m.get("mase", v1_m.get("mase_original", None)),
            )
            v2_mase = v2_m.get(
                "mase_unified",
                v2_m.get("mase", v2_m.get("mase_original", None)),
            )

            is_new = model not in v1_h
            mae_change = None
            if v1_mae and v2_mae and isinstance(v1_mae, (int, float)) and isinstance(v2_mae, (int, float)):
                mae_change = ((v2_mae - v1_mae) / v1_mae) * 100

            comparison_rows.append(
                {
                    "Horizon": h,
                    "Model": model,
                    f"MAE ({v1_name})": (round(v1_mae, 3) if isinstance(v1_mae, (int, float)) else "—"),
                    f"MAE ({v2_name})": (round(v2_mae, 3) if isinstance(v2_mae, (int, float)) else "—"),
                    f"MASE ({v1_name})": (round(v1_mase, 4) if isinstance(v1_mase, (int, float)) else "—"),
                    f"MASE ({v2_name})": (round(v2_mase, 4) if isinstance(v2_mase, (int, float)) else "—"),
                    "MAE Δ%": (f"{mae_change:+.1f}%" if mae_change is not None else ("✅ Mới" if is_new else "—")),
                }
            )

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown(f"### 📊 Biểu đồ so sánh — {v2_name} (Top 5 Models)")

    model_mase_avg: dict[str, list[float]] = {}
    for h in horizons:
        for model in v2_results.get(h, {}):
            if model == "Persistence":
                continue
            mase = v2_results[h][model].get(
                "mase_unified",
                v2_results[h][model].get("mase", v2_results[h][model].get("mase_original", None)),
            )
            if isinstance(mase, (int, float)):
                if model not in model_mase_avg:
                    model_mase_avg[model] = []
                model_mase_avg[model].append(float(mase))

    model_mase_mean = {m: sum(v) / len(v) for m, v in model_mase_avg.items()}
    chart_models = sorted(model_mase_mean.keys(), key=lambda k: model_mase_mean[k])[:5]

    chart_colors = [
        "#00D4AA",
        "#FF6B6B",
        "#4ECDC4",
        "#FFE66D",
        "#A78BFA",
        "#FB923C",
        "#60A5FA",
        "#F472B6",
        "#34D399",
        "#F87171",
        "#818CF8",
        "#FBBF24",
    ]

    fig_mase = go.Figure()
    fig_mae = go.Figure()

    for i, model in enumerate(chart_models):
        v2_mases = []
        v2_maes = []
        for h in horizons:
            v2_m = v2_results.get(h, {}).get(model, {})
            mase = v2_m.get(
                "mase_unified",
                v2_m.get("mase", v2_m.get("mase_original", None)),
            )
            mae = v2_m.get("mae", None)
            v2_mases.append(mase if isinstance(mase, (int, float)) else None)
            v2_maes.append(mae if isinstance(mae, (int, float)) else None)

        color = chart_colors[i % len(chart_colors)]

        fig_mase.add_trace(
            go.Bar(
                name=model,
                x=horizons,
                y=[m if m else 0 for m in v2_mases],
                marker_color=color,
                text=[f"{m:.3f}" if m else "—" for m in v2_mases],
            )
        )

        fig_mae.add_trace(
            go.Bar(
                name=model,
                x=horizons,
                y=[m if m else 0 for m in v2_maes],
                marker_color=color,
                text=[f"{m:.2f}" if m else "—" for m in v2_maes],
                showlegend=False,
            )
        )

    add_baseline(fig_mase, y=1.0, label="Baseline (MASE = 1.0)", color="#FF6B6B")
    fig_mase.update_layout(
        barmode="group",
        yaxis_title="MASE (thấp hơn = tốt hơn)",
        xaxis_title="Forecast Horizon",
    )

    fig_mae.update_layout(
        barmode="group",
        yaxis_title="MAE (µg/m³ - thấp hơn = tốt hơn)",
        xaxis_title="Forecast Horizon",
    )

    add_simple_bar_labels(fig_mase, orientation="v")
    add_simple_bar_labels(fig_mae, orientation="v")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        _render_chart(fig_mase, filename="exp_mase_comparison")
        _caption("MASE — So sánh giữa Baseline và Model")
    with col_c2:
        _render_chart(fig_mae, filename="exp_mae_comparison")
        _caption("MAE — So sánh giữa Baseline và Model")

    next_steps = v2.get("changes", {}).get("next_steps", "")
    if next_steps:
        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, rgba(255,230,109,0.08) 0%, rgba(255,107,107,0.04) 100%);
                    border-left: 4px solid #FFE66D; border-radius: 0 12px 12px 0;
                    padding: 1rem 1.2rem; margin: 1rem 0;">
            <b>🔔 Ghi nhớ tối ưu tiếp:</b><br>
            {next_steps}
        </div>
        """,
            unsafe_allow_html=True,
        )


def _render_all_runs() -> None:
    """Render all experiment runs table."""
    exp_dir = RESEARCH_DIR / "experiments"
    all_jsons = sorted(exp_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not all_jsons:
        st.warning("Chưa có experiment nào.")
        return

    runs = []
    for jpath in all_jsons:
        try:
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                name = jpath.stem
                parent = jpath.parent.name

                if "timestamp" in data:
                    runs.append(
                        {
                            "File": jpath.name,
                            "Thư mục": parent,
                            "Timestamp": data.get("timestamp", "—"),
                            "Model": data.get("model", parent),
                            "Source": data.get("source", "script"),
                            "path": str(jpath),
                        }
                    )
                else:
                    for key in data:
                        if key.endswith("h") and isinstance(data[key], dict):
                            for model_name, metrics in data[key].items():
                                if isinstance(metrics, dict) and "mae" in metrics:
                                    runs.append(
                                        {
                                            "File": jpath.name,
                                            "Thư mục": parent,
                                            "Timestamp": name,
                                            "Model": model_name,
                                            "Horizon": key,
                                            "MAE": metrics.get("mae"),
                                            "MASE": metrics.get("mase"),
                                            "Source": "script",
                                            "path": str(jpath),
                                        }
                                    )
        except (json.JSONDecodeError, KeyError):
            continue

    if not runs:
        st.info("Không tìm thấy runs nào có format phù hợp.")
        return

    df_runs = pd.DataFrame(runs)

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

    filtered = df_runs[df_runs["Thư mục"].isin(dir_filter) & df_runs["Model"].isin(model_filter)]

    display_cols = [c for c in filtered.columns if c != "path"]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.caption(f"Tổng: {len(filtered)} runs từ {len(all_jsons)} files")

    st.divider()
    selected_file = st.selectbox(
        "🔎 Xem chi tiết JSON",
        options=[j.name for j in all_jsons],
    )
    if selected_file:
        selected_path = next((j for j in all_jsons if j.name == selected_file), None)
        if selected_path:
            with open(selected_path, encoding="utf-8") as f:
                data = json.load(f)
            with st.expander(f"📄 {selected_file}", expanded=True):
                st.json(data)


def _render_detailed_version_card(v_name: str, v_data: dict, evo_info: dict) -> None:
    """Render a comprehensive version card with 5 academic pillars."""
    if evo_info:
        color = evo_info.get("color", "#00D4AA")
        badge = evo_info.get("badge", v_name)
        title = evo_info.get("title", v_name)
        date = evo_info.get("date", "")
        category = evo_info.get("category", "")
        models_cnt = evo_info.get("models_count", len(v_data.get("models", [])))

        st.markdown(
            f"""
            <div style="background: var(--ds-bg-card); color: var(--ds-text-primary);
                        border-left: 5px solid {color}; border-radius: 0 14px 14px 0;
                        border: 1px solid var(--ds-border); border-left-width: 5px;
                        padding: 1.25rem 1.5rem; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
                    <div>
                        <span style="background: {color}22; color: {color}; border: 1px solid {color}55;
                                     padding: 0.25rem 0.7rem; border-radius: 16px; font-size: 0.82rem; font-weight: 700;">
                            {badge}
                        </span>
                        <span style="font-size: 1.15rem; font-weight: 700; color: var(--ds-text-primary); margin-left: 0.6rem;">
                            {title}
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; color: var(--ds-text-muted);">
                        📅 <b>{date}</b> · 🏷️ <b>{category}</b> · 🤖 <b>{models_cnt} models</b>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr; gap: 0.6rem; font-size: 0.88rem; line-height: 1.55;">
                    <div style="background: var(--ds-bg-surface); padding: 0.7rem 0.9rem; border-radius: 6px; border-left: 3px solid #38BDF8;">
                        <span style="font-weight: 700; color: #38BDF8;">🏛️ 1. Bối cảnh & Giả thuyết:</span>
                        <span style="color: var(--ds-text-secondary);"> {evo_info.get("hypothesis", "—")}</span>
                    </div>
                    <div style="background: var(--ds-bg-surface); padding: 0.7rem 0.9rem; border-radius: 6px; border-left: 3px solid #A855F7;">
                        <span style="font-weight: 700; color: #A855F7;">⚙️ 2. Kỹ thuật nâng cấp lõi:</span>
                        <span style="color: var(--ds-text-secondary);"> {evo_info.get("technical_upgrades", "—")}</span>
                    </div>
                    <div style="background: var(--ds-bg-surface); padding: 0.7rem 0.9rem; border-radius: 6px; border-left: 3px solid #10B981;">
                        <span style="font-weight: 700; color: #10B981;">📊 3. Kết quả 3 chân trời:</span>
                        <span style="color: var(--ds-text-secondary);"> {evo_info.get("results_summary", "—")}</span>
                    </div>
                    <div style="background: var(--ds-bg-surface); padding: 0.7rem 0.9rem; border-radius: 6px; border-left: 3px solid #F59E0B;">
                        <span style="font-weight: 700; color: #F59E0B;">💡 4. Phát hiện chính & Rủi ro kỹ thuật:</span>
                        <span style="color: var(--ds-text-secondary);"> {evo_info.get("breakthrough_traps", "—")}</span>
                    </div>
                    <div style="background: var(--ds-bg-surface); padding: 0.7rem 0.9rem; border-radius: 6px; border-left: 3px solid #EC4899;">
                        <span style="font-weight: 700; color: #EC4899;">🎯 5. Bài học kế tiếp:</span>
                        <span style="color: var(--ds-text-secondary);"> {evo_info.get("takeaway_next", "—")}</span>
                    </div>
                </div>

                <div style="background: var(--ds-bg-teal-soft); border: 1px solid rgba(0,212,170,0.25); border-radius: 6px; padding: 0.7rem 0.9rem; margin-top: 0.6rem;">
                    <span style="font-weight: 700; color: #00D4AA; font-size: 0.85rem;">🔬 Đánh giá phương pháp luận:</span>
                    <span style="color: var(--ds-text-primary); font-size: 0.88rem; font-style: italic;"> "{evo_info.get("council_verdict", "—")}"</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Fallback to standard snapshot changes
        changes = v_data.get("changes", {})
        what = changes.get("what", v_data.get("description", "—"))
        why = changes.get("why", "—")
        result = changes.get("result", "—")
        conclusion = changes.get("conclusion", "")

        st.markdown(
            f"""
            <div style="background: var(--ds-bg-card); color: var(--ds-text-primary); border-left: 4px solid #00D4AA; border: 1px solid var(--ds-border); border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; margin: 0.8rem 0;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #00D4AA; margin-bottom: 0.4rem;">
                    📦 {v_name}
                </div>
                <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.3rem 0.8rem; font-size: 0.88rem; color: var(--ds-text-secondary);">
                    <span style="color: var(--ds-text-muted); font-weight: 600;">📦 What:</span><span>{what}</span>
                    <span style="color: var(--ds-text-muted); font-weight: 600;">💡 Why:</span><span>{why}</span>
                    <span style="color: var(--ds-text-muted); font-weight: 600;">📊 Result:</span><span>{result}</span>
                </div>
                {f'<div style="margin-top:0.4rem; padding-top:0.4rem; border-top:1px solid var(--ds-border); font-size:0.88rem; color: var(--ds-text-primary);"><span style="color:#00D4AA; font-weight:600;">✅ Conclusion:</span> {conclusion}</div>' if conclusion else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )
