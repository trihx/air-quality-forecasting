"""Interactive model training and hyperparameter experimentation page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.frontend.citations import cite, render_references_section
from src.info_cards import (
    cards_training,
    get_current_version,
    render_version_badge,
)
from src.training.trainer import (
    GRUTrainer,
    LightGBMTrainer,
    get_default_params,
)


def page_training(results: dict[str, Any]) -> None:
    """Render interactive model training and evaluation page."""
    st.markdown(
        """
    <h1 style="font-size: 2rem;">🏋️ Huấn Luyện Mô Hình</h1>
    <p style="opacity: 0.7;">Tùy chỉnh hyperparameters → Train → Đánh giá → Lưu mô hình</p>
    """,
        unsafe_allow_html=True,
    )

    ver = get_current_version()
    render_version_badge(ver)
    cards_training(ver)

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("🧠 Mô hình", ["LightGBM", "GRU"], index=0, key="train_model")
    with col2:
        horizon = st.selectbox(
            "⏱️ Horizon",
            [1, 6, 24],
            index=1,
            format_func=lambda x: f"{x} giờ",
            key="train_horizon",
        )

    st.divider()

    defaults = get_default_params(model_type, horizon)

    st.markdown(
        f"""
    <div style="background: var(--secondary-background-color); border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,212,170,0.2); border-left: 4px solid #00D4AA; color: var(--text-color) !important;">
        <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.4rem; color: var(--text-color);">
            🔬 Quy Trình Huấn Luyện & Đánh Giá 5 Bước Chuẩn Khoa Học (Chương 3 §3.3–§3.5)
        </div>
        <div style="font-size: 0.85rem; opacity: 0.85; line-height: 1.6;">
            <b>1. Tiền xử lý & Điền khuyết phân tầng:</b> Cubic Spline cho gap ≤ 6h, KNN (K=5) cho gap 6–24h, loại bỏ triệt để 19.810h khuyết > 24h.<br>
            <b>2. Kỹ nghệ 119 đặc trưng phi rò rỉ:</b> 100% biến rolling, diff, ewm tuân thủ nghiêm ngặt kỷ luật trễ <code>shift(1)</code>.<br>
            <b>3. Phân chia mỏ neo 80/10/10 & Purging Gap:</b> Tập Train (80%), Val (10% dừng sớm), Test (10% mỏ neo niêm phong) kèm vùng đệm cách ly theo horizon.<br>
            <b>4. Hàm mất mát mục tiêu:</b> LightGBM {cite("ke2017")} tối ưu trực tiếp MAE (L1 loss) qua Optuna TPE {cite("akiba2019")}; GRU {cite("cho2014")} tối ưu MSELoss qua AdamW + ReduceLROnPlateau.<br>
            <b>5. Đánh giá kiểm chuẩn mỏ neo:</b> Chỉ đánh giá trên dữ liệu thực tế (<code>is_imputed == 0</code>), đối chuẩn bằng chỉ số MASE vô hướng {cite("hyndman2006")} so với Persistence.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    params: dict[str, Any] = {}
    if model_type == "LightGBM":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["n_estimators"] = st.number_input(
                "n_estimators",
                value=defaults["n_estimators"],
                min_value=50,
                max_value=2000,
                step=50,
            )
            params["max_depth"] = st.number_input(
                "max_depth",
                value=defaults["max_depth"],
                min_value=1,
                max_value=15,
                step=1,
            )
            params["learning_rate"] = st.number_input(
                "learning_rate",
                value=defaults["learning_rate"],
                min_value=0.001,
                max_value=0.3,
                step=0.001,
                format="%.3f",
            )
        with c2:
            params["num_leaves"] = st.number_input(
                "num_leaves",
                value=defaults["num_leaves"],
                min_value=8,
                max_value=256,
                step=8,
            )
            params["subsample"] = st.slider("subsample", 0.5, 1.0, defaults["subsample"], 0.05)
            params["colsample_bytree"] = st.slider("colsample_bytree", 0.3, 1.0, defaults["colsample_bytree"], 0.05)
        with c3:
            params["min_child_samples"] = st.number_input(
                "min_child_samples",
                value=defaults["min_child_samples"],
                min_value=5,
                max_value=100,
                step=5,
            )
            params["reg_alpha"] = st.number_input(
                "reg_alpha",
                value=defaults["reg_alpha"],
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                format="%.3f",
            )
            params["reg_lambda"] = st.number_input(
                "reg_lambda",
                value=defaults["reg_lambda"],
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                format="%.3f",
            )
    else:
        c1, c2 = st.columns(2)
        with c1:
            params["lookback"] = st.number_input(
                "lookback (giờ)",
                value=defaults["lookback"],
                min_value=12,
                max_value=168,
                step=12,
            )
            params["hidden_dim"] = st.selectbox(
                "hidden_dim",
                [32, 64, 128, 256],
                index=[32, 64, 128, 256].index(defaults["hidden_dim"]),
            )
            params["num_layers"] = st.selectbox(
                "num_layers",
                [1, 2, 3],
                index=[1, 2, 3].index(defaults["num_layers"]),
            )
            params["dropout"] = st.slider("dropout", 0.0, 0.5, defaults["dropout"], 0.05)
        with c2:
            params["batch_size"] = st.selectbox(
                "batch_size",
                [64, 128, 256, 512],
                index=[64, 128, 256, 512].index(defaults["batch_size"]),
            )
            params["learning_rate"] = st.number_input(
                "learning_rate",
                value=defaults["learning_rate"],
                min_value=0.0001,
                max_value=0.01,
                step=0.0001,
                format="%.4f",
                key="gru_lr",
            )
            params["epochs"] = st.number_input(
                "epochs (max)",
                value=defaults["epochs"],
                min_value=10,
                max_value=500,
                step=10,
            )
            params["patience"] = st.number_input(
                "early stopping patience",
                value=defaults["patience"],
                min_value=3,
                max_value=50,
                step=1,
            )

    st.divider()

    col_train, col_reset = st.columns([3, 1])
    with col_train:
        train_clicked = st.button("🚀 Bắt Đầu Huấn Luyện", type="primary", use_container_width=True)
    with col_reset:
        if st.button("🔄 Reset Params"):
            st.rerun()

    if train_clicked:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(step_num: int, total: int, msg: str) -> None:
            progress_bar.progress(step_num / total)
            status_text.markdown(f"**{msg}** ({step_num}/{total})")

        try:
            if model_type == "LightGBM":
                trainer: LightGBMTrainer | GRUTrainer = LightGBMTrainer(horizon, params)
            else:
                trainer = GRUTrainer(horizon, params)

            metrics = trainer.train(progress_callback=progress_callback)

            st.success(f"✅ Huấn luyện hoàn tất trong {metrics['training_time_s']:.0f}s!")

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("MAE (µg/m³)", f"{metrics['mae']:.2f}")
            col_b.metric(
                "MASE",
                f"{metrics['mase']:.2f}",
                delta=("Tốt hơn Persistence" if metrics["mase"] < 1 else "Chưa vượt Persistence"),
                delta_color=("normal" if metrics["mase"] < 1 else "inverse"),
            )
            col_c.metric("RMSE (µg/m³)", f"{metrics['rmse']:.2f}")
            col_d.metric("R²", f"{metrics['r2']:.3f}")

            st.markdown(
                f"""
            <div style="background: var(--secondary-background-color); color: var(--text-color) !important; border-radius: 12px; padding: 1rem; margin: 1rem 0;
                        border: 1px solid rgba(0,212,170,0.2);">
                <div style="font-size: 0.85rem; opacity: 0.65;">
                    📊 Persistence MAE: {metrics["persist_mae"]:.2f} µg/m³ |
                    Test samples: {metrics["n_test"]} |
                    ⏱️ Training time: {metrics["training_time_s"]:.0f}s
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.divider()
            if st.button("💾 Lưu Mô Hình", type="secondary", use_container_width=True):
                save_path = trainer.save_model()
                st.success(f"✅ Đã lưu: `{save_path}`")
                st.balloons()

        except Exception as e:
            st.error(f"Lỗi khi huấn luyện: {e}")
            st.exception(e)

    render_references_section()
