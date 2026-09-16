"""Content Manager page — interactive editor for Info Cards, Dashboard JSON, and CSV metrics."""

from __future__ import annotations

import contextlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.frontend.api_client import APIClient
from src.frontend.components import (
    PROJECT_ROOT,
    _get_pipeline_metrics,
)
from src.info_cards import render_info_card


def page_content_manager(results: dict[str, Any]) -> None:
    """Multi-tab content management — Info Cards, JSON editor, CSV overview."""
    st.markdown("<h2 class='vt-title'>✏️ Quản Lý Nội Dung</h2>", unsafe_allow_html=True)
    st.markdown(
        "Quản lý tất cả nguồn nội dung Dashboard: thẻ hướng dẫn (Database), dữ liệu khoa học (JSON), và dữ liệu pipeline (CSV)."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📝 Info Cards (Database)",
            "📊 Nội Dung Khoa Học (JSON)",
            "📁 Dữ Liệu Pipeline (CSV)",
        ]
    )

    with tab1:
        _tab_info_cards()

    with tab2:
        _tab_dashboard_json()

    with tab3:
        _tab_data_overview()


def _tab_info_cards() -> None:
    """Info card editor backed by PostgreSQL."""
    st.markdown("Chỉnh sửa trực tiếp các thẻ hướng dẫn. Thay đổi lưu vào Database và cập nhật kiến thức Chatbot AI.")

    client = APIClient()
    cards_response = client.get_info_cards()
    if isinstance(cards_response, dict) and "error" in cards_response:
        st.error(f"Lỗi khi tải danh sách thẻ: {cards_response['error']}")
        return

    # Group cards by page
    cards_by_page: dict[str, list[dict[str, Any]]] = {}
    for card in cards_response:
        page_name = card["page"]
        if page_name not in cards_by_page:
            cards_by_page[page_name] = []
        cards_by_page[page_name].append(card)

    page_order = [
        "overview",
        "eda",
        "hyperparams",
        "training",
        "experiment_runs",
        "multi_horizon",
        "actual_vs_predicted",
        "shap",
        "prediction_intervals",
        "forecast",
        "audit",
        "ai_assistant",
    ]
    page_name_map = {
        "overview": "🏠 Tổng Quan",
        "eda": "📊 EDA & Khám Phá Dữ Liệu",
        "hyperparams": "⚙️ Cấu Hình & Hyperparameters",
        "training": "🏋️ Huấn Luyện Mô Hình",
        "experiment_runs": "📋 Lịch Sử Thí Nghiệm",
        "multi_horizon": "📈 Kết Quả Multi-Horizon",
        "actual_vs_predicted": "📉 Actual vs Predicted",
        "shap": "🧠 Giải Thích Trực Quan",
        "prediction_intervals": "📊 Khoảng Tin Cậy Dự Báo",
        "forecast": "🔮 Dự Báo PM2.5",
        "audit": "🔬 Scientific Audit",
        "ai_assistant": "💬 Trợ Lý AI",
    }

    if not cards_by_page:
        st.info("Chưa có thẻ nào trong database. Chạy `uv run python scripts/seed_info_cards.py` để tạo.")
        return

    available_pages = list(cards_by_page.keys())
    available_pages.sort(key=lambda x: page_order.index(x) if x in page_order else 999)

    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_page = st.selectbox(
            "Chọn trang (Page)",
            available_pages,
            format_func=lambda x: page_name_map.get(x, x),
            key="ic_page",
        )
    with col2:
        card_options = {c["card_key"]: f"{c['title']} ({c['card_key']})" for c in cards_by_page[selected_page]}
        selected_card_key = st.selectbox(
            "Chọn thẻ (Info Card)",
            list(card_options.keys()),
            format_func=lambda x: card_options[x],
            key="ic_card",
        )

    selected_card = next((c for c in cards_response if c["card_key"] == selected_card_key), None)

    if selected_card:
        st.markdown("---")

        col_edit, col_prev = st.columns(2)

        with col_edit:
            st.markdown("#### 📝 Editor")
            new_title = st.text_input("Tiêu đề (Title)", value=selected_card["title"], key="ic_title")
            new_content = st.text_area(
                "Nội dung Markdown",
                value=selected_card["content"],
                height=400,
                key="ic_content",
            )

            if st.button(
                "💾 Lưu thay đổi",
                type="primary",
                use_container_width=True,
                key="ic_save",
            ):
                with st.spinner("Đang lưu..."):
                    res = client.update_info_card(selected_card_key, title=new_title, content=new_content)
                    if isinstance(res, dict) and "error" in res:
                        st.error(f"Lỗi khi lưu: {res['error']}")
                    else:
                        st.success("Đã lưu thành công! Chatbot AI sẽ tự cập nhật kiến thức.")
                        import time

                        time.sleep(1)
                        st.rerun()

        with col_prev:
            st.markdown("#### 👁️ Preview")
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            render_info_card(new_title, new_content, icon="✨", collapsed=False)


def _tab_dashboard_json() -> None:
    """Structured editor for dashboard_content.json."""
    json_path = PROJECT_ROOT / "research" / "experiments" / "dashboard_content.json"

    if not json_path.exists():
        st.warning(f"File không tồn tại: `{json_path.relative_to(PROJECT_ROOT)}`")
        return

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Lỗi đọc JSON: {e}")
        return

    st.markdown(
        "Chỉnh sửa nội dung khoa học có cấu trúc. Thay đổi tự động lưu vào file JSON và cập nhật kiến thức Chatbot AI."
    )

    versions = list(data.get("versions", {}).keys())
    content_scope = st.radio(
        "Phạm vi nội dung",
        ["version_specific", "global"],
        format_func=lambda x: (
            "📌 Theo Version" if x == "version_specific" else "🌐 Dữ Liệu Chung (Literature, DM Test)"
        ),
        horizontal=True,
        key="json_scope",
    )

    changed = False

    if content_scope == "version_specific" and versions:
        selected_ver = st.selectbox("Chọn Version", versions, key="json_ver")
        ver_data = data["versions"].get(selected_ver, {})
        overview = ver_data.get("overview", {})

        section = st.selectbox(
            "Chọn Section",
            ["achievements", "limitations", "experiments"],
            format_func=lambda x: {
                "achievements": "🏆 Achievements",
                "limitations": "⚠️ Limitations",
                "experiments": "🧪 Experiments",
            }[x],
            key="json_section",
        )

        if section in ("achievements", "limitations"):
            items = overview.get(section, [])
            st.markdown(f"**{len(items)} mục hiện tại:**")

            updated_items = []
            for idx, item in enumerate(items):
                col_text, col_del = st.columns([10, 1])
                with col_text:
                    val = st.text_area(
                        f"#{idx + 1}",
                        value=item,
                        height=68,
                        key=f"json_{section}_{idx}",
                    )
                with col_del:
                    st.markdown(
                        "<div style='margin-top: 28px;'></div>",
                        unsafe_allow_html=True,
                    )
                    delete = st.button("🗑️", key=f"json_del_{section}_{idx}")
                if not delete:
                    updated_items.append(val)
                else:
                    changed = True

            new_item = st.text_area("➕ Thêm mục mới", value="", height=68, key=f"json_new_{section}")

            col_save, col_add = st.columns(2)
            with col_add:
                if st.button("➕ Thêm", key=f"json_add_{section}") and new_item.strip():
                    updated_items.append(new_item.strip())
                    changed = True

            if updated_items != items:
                changed = True

            save_clicked = col_save.button("💾 Lưu Section", type="primary", key=f"json_save_{section}")
            if changed or save_clicked:
                data["versions"][selected_ver]["overview"][section] = updated_items
                _save_dashboard_json(json_path, data)
                st.success(f"Đã lưu {section}!")
                st.rerun()

        elif section == "experiments":
            experiments = overview.get("experiments", [])
            st.markdown(f"**{len(experiments)} thí nghiệm:**")

            for idx, exp in enumerate(experiments):
                with st.expander(
                    f"🧪 {exp.get('title', f'Experiment {idx + 1}')}",
                    expanded=False,
                ):
                    exp["title"] = st.text_input(
                        "Title",
                        value=exp.get("title", ""),
                        key=f"exp_title_{idx}",
                    )
                    exp["why"] = st.text_area(
                        "Why",
                        value=exp.get("why", ""),
                        height=68,
                        key=f"exp_why_{idx}",
                    )
                    exp["how"] = st.text_area(
                        "How",
                        value=exp.get("how", ""),
                        height=68,
                        key=f"exp_how_{idx}",
                    )
                    exp["result"] = st.text_area(
                        "Result",
                        value=exp.get("result", ""),
                        height=68,
                        key=f"exp_result_{idx}",
                    )

            if st.button("💾 Lưu Experiments", type="primary", key="json_save_exp"):
                data["versions"][selected_ver]["overview"]["experiments"] = experiments
                _save_dashboard_json(json_path, data)
                st.success("Đã lưu experiments!")
                st.rerun()

    elif content_scope == "global":
        global_data = data.get("global", {}).get("multi_horizon", {})

        table_section = st.selectbox(
            "Chọn bảng dữ liệu",
            ["dm_test", "literature_intl", "literature_vn"],
            format_func=lambda x: {
                "dm_test": "📊 DM Test Results",
                "literature_intl": "📚 Literature (International)",
                "literature_vn": "📚 Literature (Việt Nam)",
            }[x],
            key="json_global_section",
        )

        table_data = global_data.get(table_section, [])

        if table_data:
            df = pd.DataFrame(table_data)
            st.markdown(f"**{len(df)} dòng** — chỉnh sửa trực tiếp trong bảng:")
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"json_table_{table_section}",
            )

            if st.button(
                "💾 Lưu bảng",
                type="primary",
                key=f"json_save_table_{table_section}",
            ):
                data["global"]["multi_horizon"][table_section] = edited_df.to_dict(orient="records")
                _save_dashboard_json(json_path, data)
                st.success(f"Đã lưu {table_section}!")
                st.rerun()
        else:
            st.info("Bảng trống. Thêm dữ liệu bằng cách click nút '+' bên dưới.")
            edited_df = st.data_editor(
                pd.DataFrame(),
                num_rows="dynamic",
                use_container_width=True,
                key=f"json_table_{table_section}_empty",
            )


def _save_dashboard_json(json_path: Path, data: dict[str, Any]) -> None:
    """Save dashboard_content.json with backup and trigger RAG re-index."""
    backup_name = f"dashboard_content.backup_{datetime.now():%Y%m%d_%H%M%S}.json"
    backup_path = json_path.parent / backup_name
    with contextlib.suppress(Exception):
        shutil.copy2(json_path, backup_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    flag_path = PROJECT_ROOT / ".chroma_db" / ".needs_reindex"
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.touch()


def _tab_data_overview() -> None:
    """Read-only overview of pipeline CSV data files."""
    st.markdown(
        "Tổng quan dữ liệu pipeline. Các file CSV là output của pipeline — cập nhật bằng cách chạy lại pipeline."
    )

    metrics = _get_pipeline_metrics()
    resolutions = metrics.get("resolutions", {})

    if not resolutions:
        st.warning("Không tìm thấy file dữ liệu trong `dataset/processed/`.")
        return

    rows = []
    for label, info in resolutions.items():
        rows.append(
            {
                "Dataset": info.get("filename", label),
                "Label": label,
                "Rows": f"{info['rows']:,}",
                "Columns": info["cols"],
                "Size": f"{info['size_mb']} MB",
                "Modified": info.get("modified", "—"),
            }
        )

    df_overview = pd.DataFrame(rows)
    st.dataframe(df_overview, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows (all resolutions)", f"{metrics['total_rows']:,}")
    with col2:
        st.metric("Features (1h dataset)", metrics["features_count"])
    with col3:
        n_files = len(resolutions)
        st.metric("Data Files", n_files)

    st.markdown("---")
    st.info(
        "💡 **Để cập nhật dữ liệu**, chạy lại pipeline:\n\n"
        "```bash\n"
        "uv run python scripts/v9_rebuild_data.py\n"
        "```\n\n"
        "Sau khi chạy xong, các số liệu trên Dashboard sẽ tự động cập nhật."
    )
