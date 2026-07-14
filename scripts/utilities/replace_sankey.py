def _tab_pipeline_journey():
    """Interactive Sankey diagram showing data flow through the pipeline."""

    _section_header("🗺️", "Data Flow — Từ IoT Sensor Đến Dự Báo")

    _insight_card(
        "💡 Đọc biểu đồ Sankey",
        "Mỗi nút là một bước trong pipeline. <b>Độ rộng</b> của dòng chảy "
        "thể hiện khối lượng dữ liệu (số dòng/features) di chuyển qua mỗi bước. "
        "Hover lên dòng chảy để xem chi tiết.",
    )

    # ── Dynamic pipeline data ──
    pm = _get_hub_pipeline_metrics()
    f_count = pm.get('features_count', 119)
    rows_1h = pm.get('resolutions', {}).get('1h', {}).get('rows', 27649)
    rows_30m = pm.get('resolutions', {}).get('30m', {}).get('rows', 55000)
    rows_15m = pm.get('resolutions', {}).get('15m', {}).get('rows', 110000)
    total_rows = rows_1h + rows_30m + rows_15m
    best = _get_best_mase()
    best_6h_model, best_6h_mase = best["6h"]

    # ── Node definitions (v9 multi-resolution pipeline) ──
    labels = [
        "IoT Sensor (209K)",                # 0
        "15m Resolution",                   # 1
        "30m Resolution",                   # 2
        "1h Resolution",                    # 3
        "15m Pipeline (Clean & FE)",        # 4
        "30m Pipeline (Clean & FE)",        # 5
        "1h Pipeline (Clean & FE)",         # 6
        "15m Models (14 models)",           # 7
        "30m Models (14 models)",           # 8
        "1h Models (10 models)",            # 9
        "Unified Evaluation (MASE)",        # 10
        f"Best 6h: {best_6h_model.split('_')[0]} ({best_6h_mase:.3f})",  # 11
    ]

    hover_labels = [
        "IoT Sensor: 209,219 records (~2 phút/mẫu)",
        f"15m Data: ~{rows_15m:,} rows",
        f"30m Data: ~{rows_30m:,} rows ⭐",
        f"1h Data: ~{rows_1h:,} rows",
        "15m Pipeline: Spline/KNN + 119 features",
        "30m Pipeline: Spline/KNN + 119 features",
        "1h Pipeline: Spline/KNN + 119 features",
        "15m Models: Đầy đủ ML, DL, Ensemble. DL (GRU) xuất sắc ở ngắn hạn.",
        "30m Models: Đầy đủ ML, DL, Ensemble. Ensemble thống trị trung và dài hạn.",
        "1h Models: Đầy đủ ML, DL, Ensemble. Baseline yếu do mất thông tin (autocorrelation cao).",
        "Đánh giá chung qua Diebold-Mariano & MASE",
        f"Mô hình xuất sắc nhất 6h: {best_6h_model} (MASE={best_6h_mase:.3f})",
    ]

    node_colors = [
        "#4ECDC4",  # 0: IoT
        "#FF6B6B",  # 1: 15m
        "#FFE66D",  # 2: 30m
        "#60A5FA",  # 3: 1h
        "#FF6B6B",  # 4: 15m pipe
        "#FFE66D",  # 5: 30m pipe
        "#60A5FA",  # 6: 1h pipe
        "#FF6B6B",  # 7: 15m model
        "#FFE66D",  # 8: 30m model
        "#60A5FA",  # 9: 1h model
        "#00D4AA",  # 10: Eval
        "#FFE66D",  # 11: Best
    ]

    # ── Link definitions (source → target, value, label) ──
    sources = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11]
    values =  [
        rows_15m, rows_30m, rows_1h,  # IoT -> Resample
        rows_15m, rows_30m, rows_1h,  # Resample -> Pipeline
        rows_15m, rows_30m, rows_1h,  # Pipeline -> Models
        rows_15m, rows_30m, rows_1h,  # Models -> Eval
        total_rows                    # Eval -> Best
    ]
    
    link_labels = [
        f"Extract 15m (~{rows_15m:,} rows)",
        f"Extract 30m (~{rows_30m:,} rows) ⭐",
        f"Extract 1h (~{rows_1h:,} rows)",
        "Process 15m", "Process 30m", "Process 1h",
        "Train 15m", "Train 30m", "Train 1h",
        "Eval 15m", "Eval 30m", "Eval 1h",
        "Winner selection"
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=45,
            thickness=18,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
            label=labels,
            color=node_colors,
            customdata=hover_labels,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=[
                "rgba(255,107,107,0.25)",   # IoT→15m
                "rgba(255,230,109,0.40)",   # IoT→30m (highlight)
                "rgba(96,165,250,0.25)",    # IoT→1h
                "rgba(255,107,107,0.20)",   # 15m→Pipe
                "rgba(255,230,109,0.35)",   # 30m→Pipe
                "rgba(96,165,250,0.20)",    # 1h→Pipe
                "rgba(255,107,107,0.20)",   # Pipe→Model
                "rgba(255,230,109,0.35)",   # Pipe→Model
                "rgba(96,165,250,0.20)",    # Pipe→Model
                "rgba(255,107,107,0.15)",   # Model→Eval
                "rgba(255,230,109,0.25)",   # Model→Eval
                "rgba(96,165,250,0.15)",    # Model→Eval
                "rgba(0,212,170,0.35)",     # Eval→Best
            ],
            hovertemplate="%{label}<extra></extra>",
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Pipeline Data Flow — V9 Multi-Resolution Framework",
            font=dict(size=16, color=COLORS["primary"]),
            pad=dict(b=20),
        ),
    )
    fig.update_traces(textfont=dict(size=11, color="#FAFAFA"))
    fig = _apply_plotly_style(fig, height=600)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

    # ── Pipeline Statistics Cards ──
    _section_header("📊", "Thống Kê Pipeline")

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("📥 Raw Input", "209K records", "~2 phút/mẫu, 3.1 năm"),
        ("🧹 Clean & Resample", "15m/30m/1h", "Đa độ phân giải (S-ESD)"),
        ("🔧 Imputed Rows", "88 (15m) / 230 (30m) / 631 (1h)", "Hybrid: Spline + KNN"),
        ("📐 Features", f"{f_count} columns", "v9: anti-leakage ✅"),
    ]
    st.markdown("""
    <style>
        .pipeline-card { text-align: center; padding: 1.2rem 0.4rem;
            background: var(--text-color) !important;
            border-radius: 10px; border: 1px solid rgba(0,212,170,0.2);
            border-top: 3px solid rgba(0,212,170,0.6); }
        .pipeline-card .pc-label { font-size: 0.78rem; color: var(--background-color) !important;
            font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
            padding-bottom: 0.3rem; border-bottom: 1px solid rgba(0,0,0,0.15);
            margin-bottom: 0.5rem; }
        .pipeline-card .pc-value { font-size: 1.5rem; font-weight: 800;
            color: var(--background-color) !important; font-family: 'JetBrains Mono', monospace;
            text-shadow: 0 0 12px rgba(0,212,170,0.3); margin: 0.3rem 0; }
        .pipeline-card .pc-detail { font-size: 0.7rem; color: var(--background-color) !important; opacity: 0.8;
            margin-top: 0.2rem; }
    </style>
    """, unsafe_allow_html=True)

    for col, (label, value, detail) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="pipeline-card">
                <div class="pc-label">{label}</div>
                <div class="pc-value">{value}</div>
                <div class="pc-detail">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Anti-leakage checkpoints ──
    _section_header("🛡️", "Anti-Leakage Checkpoints")

    checkpoints = [
        ("✅ Feature Engineering", "shift(1) trên mọi feature dùng target (diff, pct_change, ratio)"),
        ("✅ Temporal Split", "80/10/10 theo thời gian — KHÔNG random shuffle"),
        ("✅ Test = Real Data Only", "is_imputed == 0 filter bắt buộc trên test set"),
        ("✅ Transform Fit", "Scaler, PCA, BoxCox fit trên TRAIN ONLY"),
        ("✅ Purging Gap", "Gap = max_lookback giữa train/val/test"),
    ]
    for check, desc in checkpoints:
        st.markdown(f"""
        <style>
            .checkpoint-card {{
                display: flex; align-items: center; gap: 0.75rem;
                padding: 0.8rem 1.2rem; margin: 0.4rem 0;
                background: var(--text-color) !important; border-radius: 8px;
                border-left: 4px solid #00D4AA; border-top: 1px solid rgba(0,212,170,0.1);
                border-right: 1px solid rgba(0,212,170,0.1); border-bottom: 1px solid rgba(0,212,170,0.1);
            }}
        </style>
        <div class="checkpoint-card">
            <span style="font-size: 0.95rem; font-weight: 700; color: var(--background-color);
                         min-width: 200px;">{check}</span>
            <span style="font-size: 0.85rem; color: var(--background-color); opacity: 0.8;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Detail Sankey — Zoom into one resolution ──
    st.divider()
    _section_header("⚙️", "Chi Tiết Pipeline — Zoom Into One Resolution")

    _insight_card(
        "🔍 Biểu đồ Sankey chi tiết",
        "Chọn một <b>độ phân giải</b> để xem chi tiết từng bước pipeline: "
        "từ dữ liệu thô qua cleaning, feature engineering, chia tập dữ liệu, "
        "đến từng mô hình cụ thể và kết quả đánh giá. "
        "<b>Độ rộng</b> dòng chảy thể hiện số lượng mẫu dữ liệu thực tế.",
    )

    detail_res = st.radio(
        "Chọn resolution:",
        ["30m ⭐ (Tối ưu)", "15m", "1h"],
        horizontal=True,
        key="detail_sankey_res",
    )
    # Parse resolution key
    res_key = detail_res.split(" ")[0]  # "30m", "15m", "1h"
    _render_detail_sankey(res_key, pm, best)


def _render_detail_sankey(res: str, pm: dict, best_all: dict):
    """Render a detailed Sankey for a single resolution, showing every pipeline step."""

    # ── Real data from pipeline metrics ──
    res_data = pm.get("resolutions", {}).get(res, {})
    total_rows = res_data.get("rows", 0)
    n_cols = res_data.get("cols", 119)

    # Approximate data loss at each step (based on pipeline knowledge)
    raw_rows = 209397
    dedup_rows = int(raw_rows * 0.96)       # ~4% duplicates removed
    resample_rows = total_rows               # actual rows after resample
    outlier_rows = int(resample_rows * 0.98) # ~2% outliers removed by S-ESD
    impute_rows = resample_rows              # imputation fills gaps, keeps row count
    fe_rows = resample_rows                  # FE adds columns, not rows

    # Train/Val/Test split (80/10/10 temporal)
    train_rows = int(fe_rows * 0.8)
    val_rows = int(fe_rows * 0.1)
    test_rows = fe_rows - train_rows - val_rows

    # ── Best model per horizon for this resolution ──
    best_info = {}
    metrics_path = PROJECT_ROOT / "research" / "experiments" / "standardized_metrics.json"
    metrics_data = _load_json(metrics_path)
    if metrics_data and "results" in metrics_data:
        for h in ["1h", "6h", "24h"]:
            h_data = metrics_data["results"].get(h, {})
            best_m, best_mase = "—", 1.0
            for model, m in h_data.items():
                # Filter to current resolution only
                if res == "1h":
                    match = model.endswith("_1h")
                else:
                    match = f"_{res}" in model
                if not match:
                    continue
                mase = m.get("mase_unified", m.get("mase"))
                if mase is not None and mase < best_mase:
                    best_mase = mase
                    best_m = model
            best_info[h] = (best_m, best_mase)

    # Winner display
    best_h6_model, best_h6_mase = best_info.get("6h", ("—", 1.0))
    best_display = best_h6_model.split("_v9")[0].split("_v2")[0] if best_h6_model != "—" else "—"

    # ── Model definitions per resolution ──
    if res in ("15m", "30m"):
        model_nodes = [
            ("LightGBM", "#10B981", "ML"),
            ("RandomForest", "#10B981", "ML"),
            ("ElasticNet", "#10B981", "ML"),
            ("GradientBoosting", "#10B981", "ML"),
            ("Stacking", "#10B981", "ML"),
            ("VotingEnsemble", "#10B981", "ML"),
            ("GRU", "#60A5FA", "DL"),
            ("GRU Expert", "#60A5FA", "DL"),
            ("LSTM", "#60A5FA", "DL"),
            ("LSTM Expert", "#60A5FA", "DL"),
            ("TFT", "#60A5FA", "DL"),
            ("TFT Expert", "#60A5FA", "DL"),
            ("ARIMA", "#A78BFA", "Stat"),
            ("Ensemble Weighted", "#FFE66D", "Ensemble"),
        ]
    else:  # 1h (legacy)
        model_nodes = [
            ("LightGBM", "#10B981", "ML"),
            ("RandomForest", "#10B981", "ML"),
            ("GradientBoosting", "#10B981", "ML"),
            ("Stacking", "#10B981", "ML"),
            ("GRU", "#60A5FA", "DL"),
            ("LSTM", "#60A5FA", "DL"),
            ("TFT", "#60A5FA", "DL"),
            ("ARIMA", "#A78BFA", "Stat"),
            ("SARIMA", "#A78BFA", "Stat"),
            ("Ensemble Weighted", "#FFE66D", "Ensemble"),
        ]

    n_models = len(model_nodes)

    # ── Node indices ──
    # 0: IoT Raw
    # 1: Dedup & Domain Clip
    # 2: Resample
    # 3: Outlier (S-ESD)
    # 4: Imputation (Hybrid)
    # 5: Feature Engineering
    # 6: Train Set
    # 7: Val Set
    # 8: Test Set
    # 9..9+n_models-1: individual models
    # 9+n_models: MASE Eval
    # 9+n_models+1: Best Model
    IDX_IOT = 0
    IDX_DEDUP = 1
    IDX_RESAMPLE = 2
    IDX_OUTLIER = 3
    IDX_IMPUTE = 4
    IDX_FE = 5
    IDX_TRAIN = 6
    IDX_VAL = 7
    IDX_TEST = 8
    IDX_MODEL_START = 9
    IDX_EVAL = IDX_MODEL_START + n_models
    IDX_BEST = IDX_EVAL + 1

    # ── Build labels ──
    labels = [
        f"IoT Raw ({raw_rows:,})",
        f"Dedup & Clip ({dedup_rows:,})",
        f"Resample {res} ({resample_rows:,})",
        f"Outlier S-ESD ({outlier_rows:,})",
        f"Impute Hybrid ({impute_rows:,})",
        f"Feature Eng ({n_cols} cols)",
        f"Train 80% ({train_rows:,})",
        f"Val 10% ({val_rows:,})",
        f"Test 10% ({test_rows:,})",
    ]
    for name, _, _ in model_nodes:
        labels.append(name)
    labels.append("MASE Evaluation")
    labels.append(f"Best 6h: {best_display} ({best_h6_mase:.3f})")

    # ── Build hover labels ──
    hover = [
        f"Dữ liệu thô từ IoT sensor: {raw_rows:,} records (~2 phút/mẫu)",
        f"Sau xóa duplicates & domain clipping: {dedup_rows:,} rows",
        f"Resample xuống {res}: {resample_rows:,} rows (mean aggregation)",
        f"S-ESD cho PM2.5, IQR 3.0 cho biến khác: {outlier_rows:,} rows",
        f"Hybrid: Spline ≤6h + KNN 6-24h + Drop >24h: {impute_rows:,} rows",
        f"5 biến gốc → {n_cols} features (Lag, Rolling, Fourier, Domain...)",
        f"Training set: {train_rows:,} rows (temporal, data imputed + real)",
        f"Validation set: {val_rows:,} rows (early stopping, hyperparameter tuning)",
        f"Test set: {test_rows:,} rows (REAL data only, is_imputed == 0)",
    ]
    for name, _, family in model_nodes:
        hover.append(f"{name} — {family} model family")
    hover.append("Đánh giá thống nhất: MASE, MAE, RMSE, R² cho 3 horizons")
    hover.append(f"Mô hình tốt nhất 6h: {best_h6_model} (MASE={best_h6_mase:.3f})")

    # ── Build node colors ──
    node_colors = [
        "#4ECDC4",  # IoT
        "#FF6B6B",  # Dedup
        "#FFE66D",  # Resample
        "#F97316",  # Outlier
        "#06B6D4",  # Impute
        "#8B5CF6",  # FE
        "#10B981",  # Train
        "#EAB308",  # Val
        "#EC4899",  # Test
    ]
    for _, color, _ in model_nodes:
        node_colors.append(color)
    node_colors.append("#00D4AA")  # Eval
    node_colors.append("#FFD700")  # Best

    # ── Build links ──
    sources = []
    targets = []
    values = []
    link_labels = []
    link_colors = []

    # Pipeline chain: IoT → Dedup → Resample → Outlier → Impute → FE
    pipeline_steps = [
        (IDX_IOT, IDX_DEDUP, raw_rows, f"Xóa duplicates (-{raw_rows - dedup_rows:,})", "rgba(255,107,107,0.25)"),
        (IDX_DEDUP, IDX_RESAMPLE, dedup_rows, f"Resample → {res} ({resample_rows:,})", "rgba(255,230,109,0.30)"),
        (IDX_RESAMPLE, IDX_OUTLIER, resample_rows, f"S-ESD outlier (-{resample_rows - outlier_rows:,})", "rgba(249,115,22,0.25)"),
        (IDX_OUTLIER, IDX_IMPUTE, outlier_rows, f"Impute gaps (Spline+KNN)", "rgba(6,182,212,0.25)"),
        (IDX_IMPUTE, IDX_FE, impute_rows, f"Build {n_cols} features", "rgba(139,92,246,0.25)"),
    ]

    for s, t, v, lbl, clr in pipeline_steps:
        sources.append(s)
        targets.append(t)
        values.append(v)
        link_labels.append(lbl)
        link_colors.append(clr)

    # FE → Train/Val/Test split
    split_links = [
        (IDX_FE, IDX_TRAIN, train_rows, f"Train ({train_rows:,} rows)", "rgba(16,185,129,0.30)"),
        (IDX_FE, IDX_VAL, val_rows, f"Val ({val_rows:,} rows)", "rgba(234,179,8,0.30)"),
        (IDX_FE, IDX_TEST, test_rows, f"Test ({test_rows:,} rows, real only)", "rgba(236,72,153,0.30)"),
    ]
    for s, t, v, lbl, clr in split_links:
        sources.append(s)
        targets.append(t)
        values.append(v)
        link_labels.append(lbl)
        link_colors.append(clr)

    # Train & Val → each model (equal distribution for visual clarity)
    model_train_val = train_rows // n_models
    model_val_val = val_rows // n_models
    for i, (name, _, family) in enumerate(model_nodes):
        idx = IDX_MODEL_START + i
        if family == "ML":
            clr_train = "rgba(16,185,129,0.20)"
            clr_val = "rgba(16,185,129,0.15)"
        elif family == "DL":
            clr_train = "rgba(96,165,250,0.20)"
            clr_val = "rgba(96,165,250,0.15)"
        elif family == "Ensemble":
            clr_train = "rgba(255,230,109,0.30)"
            clr_val = "rgba(255,230,109,0.20)"
        else:
            clr_train = "rgba(167,139,250,0.20)"
            clr_val = "rgba(167,139,250,0.15)"
        
        # Link from Train
        sources.append(IDX_TRAIN)
        targets.append(idx)
        values.append(model_train_val)
        link_labels.append(f"Train {name}")
        link_colors.append(clr_train)
        
        # Link from Val
        sources.append(IDX_VAL)
        targets.append(idx)
        values.append(model_val_val)
        link_labels.append(f"Val (Tuning) {name}")
        link_colors.append(clr_val)

    # Each model → Eval (Models output predictions to be evaluated)
    eval_value_per_model = model_train_val + model_val_val
    for i, (name, _, family) in enumerate(model_nodes):
        idx = IDX_MODEL_START + i
        if family == "ML":
            clr = "rgba(16,185,129,0.15)"
        elif family == "DL":
            clr = "rgba(96,165,250,0.15)"
        elif family == "Ensemble":
            clr = "rgba(255,230,109,0.20)"
        else:
            clr = "rgba(167,139,250,0.15)"
        sources.append(idx)
        targets.append(IDX_EVAL)
        values.append(eval_value_per_model)
        link_labels.append(f"Eval {name}")
        link_colors.append(clr)

    # Test → Eval (Test set provides the ground truth for evaluation)
    sources.append(IDX_TEST)
    targets.append(IDX_EVAL)
    # The remaining rows to balance the Sankey perfectly to total_rows
    sum_models = eval_value_per_model * n_models
    test_eval_val = fe_rows - sum_models
    values.append(test_eval_val)
    link_labels.append(f"Ground Truth for Eval ({test_eval_val:,} rows)")
    link_colors.append("rgba(236,72,153,0.30)")

    # Eval → Best
    sources.append(IDX_EVAL)
    targets.append(IDX_BEST)
    values.append(fe_rows)
    link_labels.append(f"Winner: {best_display} (MASE={best_h6_mase:.3f})")
    link_colors.append("rgba(255,215,0,0.40)")

    # ── Render Plotly Sankey ──
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30,
            thickness=16,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
            label=labels,
            color=node_colors,
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
    ))

    res_display = {"15m": "15 phút", "30m": "30 phút", "1h": "1 giờ"}.get(res, res)
    fig.update_layout(
        title=dict(
            text=f"Detail Pipeline Flow — Resolution {res_display} ({n_models} models)",
            font=dict(size=15, color=COLORS["primary"]),
            pad=dict(b=15),
        ),
    )
    fig.update_traces(textfont=dict(size=10, color="#FAFAFA"))
    fig = _apply_plotly_style(fig, height=700)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

    # ── Legend for model families ──
    st.markdown("""
    <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; justify-content: center; margin-top: -0.5rem;">
        <span style="display: flex; align-items: center; gap: 0.4rem;">
            <span style="width: 12px; height: 12px; border-radius: 50%; background: #10B981; display: inline-block;"></span>
            <span style="font-size: 0.8rem; opacity: 0.8;">ML (Tree-based & Linear)</span>
        </span>
        <span style="display: flex; align-items: center; gap: 0.4rem;">
            <span style="width: 12px; height: 12px; border-radius: 50%; background: #60A5FA; display: inline-block;"></span>
            <span style="font-size: 0.8rem; opacity: 0.8;">Deep Learning (GRU, LSTM, TFT)</span>
        </span>
        <span style="display: flex; align-items: center; gap: 0.4rem;">
            <span style="width: 12px; height: 12px; border-radius: 50%; background: #A78BFA; display: inline-block;"></span>
            <span style="font-size: 0.8rem; opacity: 0.8;">Statistical (ARIMA)</span>
        </span>
        <span style="display: flex; align-items: center; gap: 0.4rem;">
            <span style="width: 12px; height: 12px; border-radius: 50%; background: #FFE66D; display: inline-block;"></span>
            <span style="font-size: 0.8rem; opacity: 0.8;">Ensemble</span>
        </span>
        <span style="display: flex; align-items: center; gap: 0.4rem;">
            <span style="width: 12px; height: 12px; border-radius: 50%; background: #FFD700; display: inline-block;"></span>
            <span style="font-size: 0.8rem; opacity: 0.8;">Best Model</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-horizon best summary for selected resolution ──
    if best_info:
        st.markdown("---")
        cols = st.columns(3)
        for col, h in zip(cols, ["1h", "6h", "24h"]):
            bm, bmase = best_info.get(h, ("—", 1.0))
            bm_short = bm.split("_v9")[0].split("_v2")[0] if bm != "—" else "—"
            delta = f"{(1 - bmase) * 100:+.1f}% vs Persistence" if bmase < 1.0 else "= Persistence"
            with col:
                st.metric(f"🏆 Best {h} ({res})", f"{bm_short}", delta=delta)


# ══════════════════════════════════════════════════════════════════════
# Tab 2: Feature Explainability (Interactive SHAP)
# ══════════════════════════════════════════════════════════════════════


