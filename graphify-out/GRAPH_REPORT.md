# Graph Report - .  (2026-07-21)

## Corpus Check
- Large corpus: 494 files · ~1,348,179 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 2447 nodes · 4791 edges · 293 communities (144 shown, 149 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 163 edges (avg confidence: 0.67)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Data Validation & Schema
- Streamlit Dashboard UI
- Leakage Audit & Hashing
- Multi-Horizon Experiments
- Baseline Models
- Data Pipeline & Imputation
- Info Cards & EDA
- GRU Deep Learning
- TFT Architecture
- Evaluation Metrics
- Database Engine
- FastAPI Backend
- Citation Management
- AI Chatbot (RAG/LLM)
- ORM & Experiment DB
- Feature Engineering v2
- Dashboard Hub & Charts
- API Smoke Tests
- Leakage Detection
- Visualization Components
- Module Group 20
- Module Group 21
- Module Group 22
- Module Group 23
- Module Group 24
- Module Group 25
- Module Group 26
- Module Group 27
- Module Group 28
- Module Group 29
- Module Group 30
- Module Group 31
- Module Group 32
- Module Group 33
- Module Group 34
- Module Group 35
- Module Group 36
- Module Group 37
- Module Group 38
- Module Group 39
- Module Group 40
- Module Group 41
- Module Group 42
- Module Group 43
- Module Group 44
- Module Group 45
- Module Group 46
- Module Group 47
- Module Group 48
- Module Group 49
- Module Group 50
- Module Group 51
- Module Group 52
- Module Group 53
- Module Group 54
- Module Group 55
- Module Group 56
- Module Group 57
- Module Group 58
- Module Group 59
- Module Group 60
- Module Group 61
- Module Group 62
- Module Group 63
- Module Group 64
- Module Group 65
- Module Group 66
- Module Group 67
- Module Group 68
- Module Group 69
- Module Group 70
- Module Group 71
- Module Group 72
- Module Group 73
- Module Group 74
- Module Group 75
- Module Group 76
- Module Group 77
- Module Group 78
- Module Group 79
- Module Group 80
- Module Group 81
- Module Group 82
- Module Group 83
- Module Group 84
- Module Group 85
- Module Group 86
- Module Group 87
- Module Group 88
- Module Group 89
- Module Group 90
- Module Group 91
- Module Group 92
- Module Group 93
- Module Group 94
- Module Group 95
- Module Group 96
- Module Group 97
- Module Group 98
- Module Group 99
- Module Group 100
- Module Group 101
- Module Group 102
- Module Group 103
- Module Group 104
- Module Group 105
- Module Group 106
- Module Group 107
- Module Group 108
- Module Group 109
- Module Group 110
- Module Group 111
- Module Group 112
- Module Group 113
- Module Group 114
- Module Group 115
- Module Group 116
- Module Group 117
- Module Group 118
- Module Group 119
- Module Group 120
- Module Group 121
- Module Group 122
- Module Group 123
- Module Group 124
- Module Group 125
- Module Group 130
- Module Group 131
- Module Group 132
- Module Group 133
- Module Group 134
- Module Group 135
- Module Group 136
- Module Group 137
- Module Group 138
- Module Group 139
- Module Group 140
- Module Group 141
- Module Group 142
- Module Group 143
- Module Group 144
- Module Group 145
- Module Group 146
- Module Group 147
- Module Group 148
- Module Group 149
- Module Group 150
- Module Group 151
- Module Group 152
- Module Group 153
- Module Group 154
- Module Group 155
- Module Group 156
- Module Group 157
- Module Group 158
- Module Group 159
- Module Group 160
- Module Group 161
- Module Group 162
- Module Group 163
- Module Group 164
- Module Group 165
- Module Group 166
- Module Group 167
- Module Group 168
- Module Group 169
- Module Group 170
- Module Group 171
- Module Group 172
- Module Group 173
- Module Group 174
- Module Group 175
- Module Group 176
- Module Group 177
- Module Group 178
- Module Group 179
- Module Group 180
- Module Group 181
- Module Group 182
- Module Group 183
- Module Group 184
- Module Group 185
- Module Group 186
- Module Group 187
- Module Group 188
- Module Group 189
- Module Group 190
- Module Group 191
- Module Group 192
- Module Group 193
- Module Group 194
- Module Group 195
- Module Group 196
- Module Group 197
- Module Group 198
- Module Group 199
- Module Group 200
- Module Group 201
- Module Group 202
- Module Group 203
- Module Group 204
- Module Group 205
- Module Group 206
- Module Group 207
- Module Group 208
- Module Group 209
- Module Group 210
- Module Group 211
- Module Group 212
- Module Group 213
- Module Group 214
- Module Group 215
- Module Group 216
- Module Group 217
- Module Group 218
- Module Group 219
- Module Group 220
- Module Group 221
- Module Group 222
- Module Group 223
- Module Group 224
- Module Group 225
- Module Group 226
- Module Group 227
- Module Group 228
- Module Group 229
- Module Group 230
- Module Group 231
- Module Group 232
- Module Group 233
- Module Group 234
- Module Group 235
- Module Group 236
- Module Group 237
- Module Group 238
- Module Group 239
- Module Group 240
- Module Group 241
- Module Group 242
- Module Group 243
- Module Group 244
- Module Group 245
- Module Group 246
- Module Group 247
- Module Group 248
- Module Group 249
- Module Group 250
- Module Group 251
- Module Group 252
- Module Group 253
- Module Group 254
- Module Group 255
- Module Group 256
- Module Group 257
- Module Group 258
- Module Group 259
- Module Group 260
- Module Group 261
- Module Group 262
- Module Group 263
- Module Group 264

## God Nodes (most connected - your core abstractions)
1. `load_raw_data()` - 91 edges
2. `impute_missing_data()` - 81 edges
3. `_remove_duplicates()` - 72 edges
4. `_clip_physical_bounds()` - 72 edges
5. `build_features()` - 72 edges
6. `_resample()` - 71 edges
7. `_handle_outliers()` - 70 edges
8. `_set_datetime_index()` - 68 edges
9. `DataValidator` - 49 edges
10. `evaluate_forecast_full()` - 39 edges

## Surprising Connections (you probably didn't know these)
- `main()` --indirect_call--> `page_actual_vs_predicted()`  [INFERRED]
  app.py → pages.py
- `main()` --calls--> `r2_score()`  [INFERRED]
  scripts/archive/compute_complete_metrics.py → src/evaluation/metrics.py
- `export_info_cards()` --indirect_call--> `InfoCard`  [INFERRED]
  scripts/export_db_to_json.py → src/api/models.py
- `export_experiments()` --indirect_call--> `Experiment`  [INFERRED]
  scripts/export_db_to_json.py → src/api/models.py
- `export_runs()` --indirect_call--> `Run`  [INFERRED]
  scripts/export_db_to_json.py → src/api/models.py

## Import Cycles
- None detected.

## Communities (293 total, 149 thin omitted)

### Community 0 - "Data Validation & Schema"
Cohesion: 0.08
Nodes (27): Enum, DataValidator, Any, DataFrame, Data validator — domain-specific quality checks for PM2.5 pipeline.  Validates d, Validate feature-engineered data (Marts layer).          Checks per SKILL.md §3., Check if any CRITICAL validation failed., Validation severity levels. (+19 more)

### Community 1 - "Streamlit Dashboard UI"
Cohesion: 0.08
Nodes (56): _count_tests(), _get_pipeline_metrics(), insight_card(), kpi_card(), load_experiment_results(), load_json(), main(), page_content_manager() (+48 more)

### Community 2 - "Leakage Audit & Hashing"
Cohesion: 0.07
Nodes (53): get_audit_report(), get_data_hashes(), get_model_weights(), _md5_file(), Path, Audit router — Data & model hash verification.  Endpoints:     GET /audit/dat, Generate full audit report (data + models)., Verify file integrity by comparing current MD5 with expected hashes from manifes (+45 more)

### Community 3 - "Multi-Horizon Experiments"
Cohesion: 0.07
Nodes (38): _evaluate_horizon(), _find_orders(), main(), _prepare_hybrid_data(), DataFrame, Series, ARIMA/SARIMA Multi-Horizon Evaluation — Level 1 Statistical Models.  Compare A, Load raw data and apply Hybrid imputation strategy. (+30 more)

### Community 4 - "Baseline Models"
Cohesion: 0.08
Nodes (28): get_all_baselines(), HourlyMeanModel, MeanModel, NaiveBaseline, PersistenceModel, DataFrame, ndarray, Series (+20 more)

### Community 5 - "Data Pipeline & Imputation"
Cohesion: 0.06
Nodes (34): get_data_summary(), Any, DataFrame, Run basic validation checks on dataset.      Args:         df: Input DataFrame., Check that all expected columns exist., Generate summary statistics about the dataset.      Args:         df: Input Data, _validate_columns(), validate_dataset() (+26 more)

### Community 6 - "Info Cards & EDA"
Cohesion: 0.07
Nodes (35): cards_ai_assistant(), cards_eda(), cards_experiment_runs(), cards_forecast(), cards_hyperparams(), cards_multi_horizon(), cards_overview(), cards_prediction_intervals() (+27 more)

### Community 7 - "GRU Deep Learning"
Cohesion: 0.06
Nodes (36): create_deseasonalized_targets(), GRUModel, DataFrame, DataLoader, Dataset, Module, ndarray, Series (+28 more)

### Community 8 - "TFT Architecture"
Cohesion: 0.06
Nodes (26): add_cv_features(), GatedLinearUnit, GatedResidualNetwork, GRUModel, LSTMModel, DataFrame, Dataset, Select numerical features, excluding target and metadata. (+18 more)

### Community 9 - "Evaluation Metrics"
Cohesion: 0.08
Nodes (29): classification_metrics(), _compute_roc_auc(), evaluate_forecast(), forecast_bias(), medae(), ndarray, Evaluation metrics for time series forecasting.  Implements SKILL.md §9 metrics:, Forecast Bias — over- or under-forecasting indicator.      Ref: Manu Joseph Ch.4 (+21 more)

### Community 10 - "Database Engine"
Cohesion: 0.07
Nodes (22): get_model_type(), DataFrame, Reporting Engine — Single Source of Truth for Dashboard metrics.  Loads normaliz, Get best model for all horizons.          Returns:             {"1h": {...}, "6h, Generate ranking table sorted by MAE, with star markers for best.          Args:, Get ranking table formatted for display with ⭐ markers.          Returns DataFra, Get MASE values formatted for chart plotting.          Args:             model_f, Get MAE values formatted for chart plotting.          Returns:             {"Per (+14 more)

### Community 11 - "FastAPI Backend"
Cohesion: 0.08
Nodes (32): FastAPI, Request, seed(), get_db(), Session, Database engine, session, and Base for SQLAlchemy 2.0.  Usage:     from src.api., FastAPI dependency — yields a DB session, auto-closes., FastAPI dependencies — DB session, model cache. (+24 more)

### Community 12 - "Citation Management"
Cohesion: 0.09
Nodes (33): cite(), _ensure_css(), IEEE Citation Popup System + Pipeline Step Framework.  Provides two complement, Render a full IEEE-formatted references list at the bottom of a page., Inject citation CSS. We inject it every time to ensure it persists across Stream, Return an inline HTML tooltip for a citation.      Args:         ref_id: Key, render_references_section(), _get_dashboard_content() (+25 more)

### Community 13 - "AI Chatbot (RAG/LLM)"
Cohesion: 0.11
Nodes (32): OpenAI, _ensure_index(), _get_knowledge_base(), AI Assistant page for PM2.5 Forecasting Dashboard.  Provides a chat interface wi, Lazy import and get knowledge base singleton., Ensure knowledge base is indexed, re-index if user content changed.      Checks, Render AI provider configuration in sidebar., _render_provider_config() (+24 more)

### Community 14 - "ORM & Experiment DB"
Cohesion: 0.11
Nodes (31): DeclarativeBase, _delete_experiment(), _load_json_safe(), main(), Path, Sync existing JSON experiment results → Database.  One-shot script that reads al, Load JSON with NaN/Infinity handling., Delete experiment by name (for --force mode). (+23 more)

### Community 15 - "Feature Engineering v2"
Cohesion: 0.08
Nodes (22): add_cv_features(), DataFrame, Tests for DL retrain v2 components — CV features and data pipeline., CV uses shift(1) on lag_1h → no access to current value., Test that DL feature selection works correctly., Feature selection should exclude target, is_imputed., Add Coefficient of Variation features with safeguard., Test log1p/expm1 inverse transform correctness. (+14 more)

### Community 16 - "Dashboard Hub & Charts"
Cohesion: 0.13
Nodes (33): _generate_shapash_html(), _get_best_mase(), _image_to_plotly(), _insight_card(), _load_json(), page_explainability_hub(), Figure, Path (+25 more)

### Community 17 - "API Smoke Tests"
Cohesion: 0.07
Nodes (16): override_get_db(), API tests — Experiments CRUD, Inference, Audit endpoints.  Uses FastAPI TestClie, Test /api/v1/runs endpoints., Test model logging and metric recording., Helper: create experiment → run → return run_id., Test /api/v1/audit endpoints., Override DB dependency for tests., Create tables before each test, drop after. (+8 more)

### Community 18 - "Leakage Detection"
Cohesion: 0.08
Nodes (23): feature_cols(), marts_df(), DataFrame, Data leakage detection tests for the forecasting pipeline.  These tests ensure n, Ensure temporal ordering and no future information., Verify lag features equal shifted target values., Rolling features should be computed on shifted (past) values only.          A ro, Randomization test: model should fail with shuffled target. (+15 more)

### Community 19 - "Visualization Components"
Cohesion: 0.10
Nodes (29): Bar, Scatter, plot_dm_test_heatmap(), plot_mae_trend(), plot_mae_trend_top5(), plot_mase_comparison(), plot_mase_comparison_top5(), Figure (+21 more)

### Community 20 - "Module Group 20"
Cohesion: 0.11
Nodes (13): ExperimentLogger, Any, Log a model within a run, return run_model_id., Log metrics for a model, return metric_id., Check if experiment with given name already exists., Parse a full JSON result dict → Experiment + Runs + Models + Metrics.          E, Convert NaN/Infinity to None for DB storage., Centralized experiment tracking — write to DB or API. (+5 more)

### Community 21 - "Module Group 21"
Cohesion: 0.10
Nodes (23): DataFrame, Split data temporally: oldest → train → val → test → newest.      NEVER shuffles, temporal_train_val_test_split(), main(), Run Baseline Experiment — Level 0 Models.  Usage:     uv run python -m src.mo, Run all Level 0 baselines and compare results., _get_lightgbm(), get_ml_models() (+15 more)

### Community 22 - "Module Group 22"
Cohesion: 0.10
Nodes (26): _cached_pipeline_data(), Cache the heavy data pipeline result (load → clean → impute).      This is the, main(), prepare_data(), P0-8 — Deseasonalizing Transform Experiment.  Hypothesis: PM2.5 has strong dai, Load → clean → impute → build_features (v2)., _prepare_hybrid_data(), DataFrame (+18 more)

### Community 23 - "Module Group 23"
Cohesion: 0.10
Nodes (20): Split DataFrame into real and imputed portions.      Use this to ensure test set, split_real_imputed(), clean_data(), hourly_data_with_gaps(), DataFrame, Tests for data imputer — multi-strategy missing data recovery.  Each test has ve, KNN feature matrix must NOT contain pm25 (anti-leakage)., ML imputation should fill gaps up to max_gap. (+12 more)

### Community 24 - "Module Group 24"
Cohesion: 0.15
Nodes (23): _save(), evaluate_tft(), main(), DataFrame, v9 Phase 5B-1h — TFT 1h Pipeline (OOM Fix).  Uses CPU fallback, base features, a, run_pipeline(), train_tft(), create_sequences_segment_aware() (+15 more)

### Community 25 - "Module Group 25"
Cohesion: 0.16
Nodes (25): _correlation_analysis(), _descriptive_stats(), _missing_analysis(), _plot_acf_pacf(), _plot_distributions(), _plot_time_series(), Any, DataFrame (+17 more)

### Community 26 - "Module Group 26"
Cohesion: 0.12
Nodes (21): main(), prepare_data(), DL Retrain v2 — GRU/LSTM with Enhanced Features + Log Transform Comparison.  3, Load → clean → impute → build_features (v2 enhanced) → add CV., Rebuild marts data and re-run ML pipeline after leakage fix.  Usage:     uv run, main(), prepare_data(), DataFrame (+13 more)

### Community 27 - "Module Group 27"
Cohesion: 0.09
Nodes (19): add_cv_features(), evaluate_horizon(), GRUModel, LSTMModel, DataFrame, DataLoader, Dataset, Module (+11 more)

### Community 28 - "Module Group 28"
Cohesion: 0.12
Nodes (21): main(), Quick verification of the metrics pipeline end-to-end., _compute_best_models(), _compute_top_n(), _derive_models(), _extract_results(), load_all_normalized(), _normalize_model_entry() (+13 more)

### Community 29 - "Module Group 29"
Cohesion: 0.14
Nodes (23): main(), plot_missing_barcode(), plot_recovery_bar(), DataFrame, Khoảng trống chất lượng (Data Quality Gaps) Mục tiêu: Kể câu chuyện về sự không, Vẽ mã vạch (barcode) thể hiện các thời điểm bị mất dữ liệu., Vẽ biểu đồ các loại gap., Unified Visualization Theme Framework (VTF) for PM2.5 Dashboard.  Provides centr (+15 more)

### Community 30 - "Module Group 30"
Cohesion: 0.13
Nodes (22): _apply_knn_imputation(), _build_knn_features(), _identify_gaps(), DataFrame, Drop all gaps > max_gap. Interpolate only very short gaps (≤2h)., KNN imputation using auxiliary features (temperature, humidity, CO2).      ANTI-, Tiered approach:     - Gap ≤ max_gap_interp: Cubic Spline interpolation     - Ga, Identify contiguous NaN gap segments.      Returns:         DataFrame with colum (+14 more)

### Community 31 - "Module Group 31"
Cohesion: 0.10
Nodes (15): mape(), r2_score(), R² (coefficient of determination)., Mean Absolute Percentage Error.      Skips near-zero values to avoid division by, create_naive_predictions(), ndarray, Series, Data splitter for time series — temporal split, NO shuffle.  Per SKILL.md §6.3: (+7 more)

### Community 32 - "Module Group 32"
Cohesion: 0.11
Nodes (15): mae(), Mean Absolute Error — Primary metric per SKILL.md., Root Mean Squared Error — penalizes large errors., rmse(), TestMAE, TestRMSE, Validation tests — cross-reference results with literature & best practices.  Sc, RMSE >= MAE always (Cauchy-Schwarz inequality). (+7 more)

### Community 33 - "Module Group 33"
Cohesion: 0.11
Nodes (11): APIClient, Get data file hashes for audit., Get model weight hashes for audit., Get full audit report., Verify file integrity against manifest.json expected MD5 hashes., List all info cards, optionally filtered by page., Thin HTTP client for the PM2.5 Forecasting API., List all experiments. (+3 more)

### Community 34 - "Module Group 34"
Cohesion: 0.09
Nodes (17): _evaluate_horizon(), GRUModel, LSTMModel, main(), DataLoader, Dataset, Module, ndarray (+9 more)

### Community 35 - "Module Group 35"
Cohesion: 0.12
Nodes (21): main(), DataFrame, v9 Phase 5B.3 — Retrain ARIMA/SARIMA.  Retrains Statsmodels ARIMA on high-resolu, Run ARIMA on the largest segment in the training set., run_arima(), eval_model(), main(), DataFrame (+13 more)

### Community 36 - "Module Group 36"
Cohesion: 0.12
Nodes (16): create_sequences_segment_aware(), evaluate_dl(), main(), DataFrame, v9 Phase 5B-Expert — DL Expert Pipeline (GRU/LSTM).  Uses the BASE dataset (more, Create sequences strictly within segments to prevent False Continuity., RNNModel, run_pipeline() (+8 more)

### Community 37 - "Module Group 37"
Cohesion: 0.14
Nodes (18): main(), Compare Data Strategies — Run all 4 imputation strategies and evaluate.  PRINC, Save results to JSON., _save_results(), main(), Compute R², RMSE, MAPE for ALL models across ALL horizons. Generates a complete, _handle_outliers_iqr_all(), main() (+10 more)

### Community 38 - "Module Group 38"
Cohesion: 0.16
Nodes (21): _count_label(), export_experiments(), export_feature_importances(), export_info_cards(), export_metrics(), export_run_models(), export_runs(), main() (+13 more)

### Community 39 - "Module Group 39"
Cohesion: 0.19
Nodes (20): _log(), main(), Automated Retraining Pipeline — MLOps Orchestrator.  Standardized wrapper that, Step 3: Standardize metrics with unified Persistence baseline., Step 4: Update SHAP explainability data., Step 2.5: Log training results to Database., Step 5: Precompute AVP cache for Dashboard., Run test suite to verify pipeline integrity. (+12 more)

### Community 40 - "Module Group 40"
Cohesion: 0.17
Nodes (19): compute_dm_hln(), compute_r2_multi_horizon(), compute_unified_mase(), get_actuals_for_preds(), load_predictions_per_source(), load_target_series(), main(), ndarray (+11 more)

### Community 41 - "Module Group 41"
Cohesion: 0.10
Nodes (13): Tests for DL v3 — PCA feature selection + TFT validation., Test TFT data preparation with v2 features., Static columns should be calendar cyclical features., Temporal features must NOT include target column., Verify log1p/expm1 and scaler inverse produce correct results., StandardScaler inverse should recover original values., Clipping should prevent negative PM2.5 predictions., Test top-N feature importance selection. (+5 more)

### Community 42 - "Module Group 42"
Cohesion: 0.14
Nodes (19): _get_eval_metrics(), _get_gru_quantile_predictor(), _get_torch_device(), _predict_arima(), _predict_ensemble(), _predict_persistence(), _predict_sarima(), DataFrame (+11 more)

### Community 43 - "Module Group 43"
Cohesion: 0.15
Nodes (15): _get_gru_predictor(), _get_lgbm_predictor(), predict(), Inference router — PM2.5 prediction endpoints.  Endpoints:     POST /predict — R, Predict PM2.5 for a given horizon using specified model.      Supported models:, PredictionRequest, PredictionResponse, Request body for PM2.5 prediction. (+7 more)

### Community 44 - "Module Group 44"
Cohesion: 0.14
Nodes (15): main(), prepare_data(), DL v3 — PCA Feature Selection for 1h + TFT Retrain with v2 Features.  Tasks:, Load → clean → impute → build_features (v2) → add CV., _load_latest(), main(), _prepare_hybrid_data(), DataFrame (+7 more)

### Community 45 - "Module Group 45"
Cohesion: 0.18
Nodes (18): _approximate_entropy(), generate_forecastability(), generate_hourly_boxplot(), generate_periodogram(), generate_qq_plot(), generate_stl_decomposition(), load_data(), main() (+10 more)

### Community 46 - "Module Group 46"
Cohesion: 0.21
Nodes (18): _evaluate_horizon(), _predict_and_eval(), DataFrame, Train → predict → evaluate → return metrics., Evaluate sklearn models at a specific forecast horizon., build_ensemble(), main(), DataFrame (+10 more)

### Community 47 - "Module Group 47"
Cohesion: 0.15
Nodes (16): build_pipeline(), main(), v9 Phase 5A — Rebuild data at high resolution with segment-aware features.  Bu, Run the full data pipeline at a given frequency.      Args:         freq: Res, get_segment_stats(), identify_contiguous_segments(), DataFrame, Data segmenter — Identify and manage contiguous data segments.  For IoT time ser (+8 more)

### Community 48 - "Module Group 48"
Cohesion: 0.13
Nodes (12): mase(), Mean Absolute Scaled Error — BẮT BUỘC benchmark per SKILL.md.      MASE < 1.0 →, TestMASE, Verify MASE = MAE_model / MAE_naive (Hyndman & Koehler 2006).      Reference: ht, Verify MASE matches manual calculation., Shuffle test: if we randomize targets, model should fail (MASE ≈ 1.0+).      Thi, After shuffling targets, MASE should be >> 1.0 (no signal)., MASE = 1.0 exactly when model = naive baseline. (+4 more)

### Community 49 - "Module Group 49"
Cohesion: 0.16
Nodes (13): main(), prepare_data(), Export TFT predictions for AVP dashboard integration.  Trains the Simplified T, Load → clean → impute → add static features (same as tft_multi_horizon.py)., _build_model_components(), main(), prepare_data(), Temporal Fusion Transformer (TFT) Multi-Horizon Evaluation — Level 5.  Simplif (+5 more)

### Community 50 - "Module Group 50"
Cohesion: 0.12
Nodes (11): _create_domain_features(), get_feature_columns(), DataFrame, Path, Domain-specific features for air quality per SKILL.md §5.3.      ANTI-LEAKAGE: U, Categorize feature columns by type for documentation.      Returns:         Dict, Save Marts-ready data with optional validation.      Args:         df: Feature-r, save_marts_data() (+3 more)

### Community 51 - "Module Group 51"
Cohesion: 0.15
Nodes (16): _format_model_label(), _load_avp_cache(), page_actual_vs_predicted(), Dashboard Pages — New production features.  Contains:   - page_forecast: 🔮 PM, Render the Actual vs Predicted chart from cached data.      Resolution-First D, Render version comparison from dashboard_runs/ snapshots., Format dropdown label: 'GRU ⭐ (MASE: 0.812)' or 'ARIMA'., Show compact ranking bar below dropdowns. (+8 more)

### Community 52 - "Module Group 52"
Cohesion: 0.21
Nodes (16): collect_metrics_from_files(), compute_directional_accuracy(), _load_actuals_from_dataset(), load_json(), main(), patch_standardized_metrics(), Path, Patch standardized_metrics.json with RMSE, R², Forecast Bias, and DA.  Reads exi (+8 more)

### Community 53 - "Module Group 53"
Cohesion: 0.15
Nodes (8): GatedLinearUnit, GatedResidualNetwork, InterpretableMultiHeadAttention, Multi-head attention with interpretable weights., GLU activation: splits input, applies sigmoid gate., GRN: core building block of TFT., VSN: learns which variables are important., VariableSelectionNetwork

### Community 54 - "Module Group 54"
Cohesion: 0.15
Nodes (14): create_ewm_features(), _ewm_col(), DataFrame, Series, Temporal feature engineering — lag, rolling, EWM features.  Implements SKILL.md, # NOTE: These are in TIME STEPS, not hours. At 1h freq → 1 step = 1h., Create Exponentially Weighted Moving features.      Shift(1) to prevent leakage., Shift a series, optionally within segments.      Args:         series: The serie (+6 more)

### Community 55 - "Module Group 55"
Cohesion: 0.18
Nodes (15): _evaluate_horizon(), _init_torch(), main(), _make_dataset_class(), _make_gru_model(), _prepare_hybrid_data(), Ensemble Stacking Multi-Horizon — Combine LightGBM + GRU for best performance., Create GRU model class with lazy torch import. (+7 more)

### Community 56 - "Module Group 56"
Cohesion: 0.21
Nodes (15): audit_mase_recompute(), audit_overfitting_check(), audit_persistence_baseline(), audit_statistical_methods_mase(), audit_test_set_consistency(), load_latest(), main(), mase_manual() (+7 more)

### Community 57 - "Module Group 57"
Cohesion: 0.18
Nodes (15): eval_model(), Train, predict, evaluate., LightGBM with Optuna-tuned hyperparameters., RF, GradientBoosting, Stacking, Ensemble., ARIMA/SARIMA with rolling window., GRU/LSTM v2+log with MPS acceleration., Create train/test split with anti-leakage., run_arima() (+7 more)

### Community 58 - "Module Group 58"
Cohesion: 0.20
Nodes (15): _diebold_mariano(), _diebold_mariano_tests(), _generate_predictions(), main(), _prepare_hybrid_data(), DataFrame, ndarray, Statistical Significance & Residual Diagnostics.  Generates: 1. Diebold-Maria (+7 more)

### Community 59 - "Module Group 59"
Cohesion: 0.19
Nodes (14): conformal_prediction_lgbm(), main(), mc_dropout_gru(), plot_intervals(), prepare_data(), prepare_ml_data(), quantile_regression_lgbm(), Prediction Intervals — Conformal Prediction + GRU Bootstrap.  Implements:   1 (+6 more)

### Community 60 - "Module Group 60"
Cohesion: 0.14
Nodes (14): _detect_available_models(), _get_smart_model_list(), _is_model_inferrable(), _load_all_rankings(), _load_model_rankings(), Scan models/exported/ to find which models can be used for inference., Load MASE rankings per horizon from standardized_metrics.json., Check if a model from standardized_metrics can be run for inference.      Uses (+6 more)

### Community 61 - "Module Group 61"
Cohesion: 0.18
Nodes (10): _evaluate_horizon(), DataFrame, Evaluate linear models at a specific forecast horizon., StandardScaler, Test PCA dimensionality reduction for DL pipeline., PCA with 95% threshold should capture >= 95% variance., PCA must be fit on training data only to prevent leakage., Simulated 117-feature data should reduce significantly. (+2 more)

### Community 62 - "Module Group 62"
Cohesion: 0.25
Nodes (13): evaluate_tuned_model(), main(), DataFrame, Series, Feature Selection + Optuna Hyperparameter Tuning.  Goal: Beat Persistence base, Tune LightGBM hyperparameters with Optuna., Train tuned model on full training set, evaluate on test., Feature Selection + Optuna Tuning Pipeline. (+5 more)

### Community 63 - "Module Group 63"
Cohesion: 0.19
Nodes (12): load_and_split_data(), main(), DataFrame, ndarray, V8 ACI Prediction Intervals — Compare CQR vs ACI.  Loads GRU Quantile models,, Load processed data and split into train/cal/test., Run GRU quantile model on all windows in df.      Returns:         (q_lower,, run_quantile_inference() (+4 more)

### Community 64 - "Module Group 64"
Cohesion: 0.22
Nodes (12): _chunk_text(), _load_dashboard_content_json(), _load_experiment_results(), _load_info_cards_from_db(), _load_markdown_docs(), RAG Knowledge Base for PM2.5 Project AI Assistant.  Indexes project documentatio, Load user-curated info cards for RAG indexing.      3-tier fallback:         Tie, Load structured dashboard content JSON as knowledge documents. (+4 more)

### Community 65 - "Module Group 65"
Cohesion: 0.21
Nodes (7): create_calendar_features(), _get_season(), DataFrame, Calendar feature engineering — time-based categorical features.  Implements SKIL, Create calendar/temporal categorical features from DatetimeIndex.      Features, Map month to season (Vietnam tropical climate).      0: Spring transition (Feb-M, TestCalendarFeatures

### Community 66 - "Module Group 66"
Cohesion: 0.23
Nodes (12): export_arima_horizon(), export_sarima_horizon(), find_orders(), main(), prepare_data(), DataFrame, Series, Export ARIMA and SARIMA predictions for AVP dashboard integration.  Runs rolli (+4 more)

### Community 67 - "Module Group 67"
Cohesion: 0.27
Nodes (11): load_cleaned_data(), main(), DataFrame, Phase 4: Deep Insights — P1-3, P1-4, P1-5.  P1-3: Error Anatomy (Error vs Hour, Granger Causality: do external vars help predict PM2.5?      Ref: Peixeiro Ch., Cross-correlation between PM2.5 and Temperature/Humidity.      Ref: Huang Ch.3, Load and clean data to get full feature set with timestamps., Error analysis by hour-of-day and PM2.5 level.      Uses GRU v2+log at h=6 as (+3 more)

### Community 68 - "Module Group 68"
Cohesion: 0.21
Nodes (9): main(), md5_file(), Path, Generate manifest.json with MD5 hashes for all model weights and key data files., Compute MD5 hash of a file., main(), Compile v8 snapshot and standardized metrics., main() (+1 more)

### Community 69 - "Module Group 69"
Cohesion: 0.24
Nodes (11): _create_v2_snapshot(), _evaluate_lgbm(), main(), _prepare_hybrid_data(), _print_comparison(), DataFrame, Enhanced Pipeline Runner — Re-evaluate LightGBM with new features.  Runs Light, Load raw data and apply Hybrid imputation strategy. (+3 more)

### Community 70 - "Module Group 70"
Cohesion: 0.26
Nodes (11): main(), plot_diagnostics(), Path, Series, Stationarity diagnostics for PM2.5 time series.  Tests:     - ADF (Augmented, Plot time series + ACF + PACF diagnostics., Run full stationarity diagnostics., Run Augmented Dickey-Fuller test.      H0: Series has a unit root (non-station (+3 more)

### Community 71 - "Module Group 71"
Cohesion: 0.24
Nodes (11): audit_knn_temporal_order(), build_knn_features(), identify_gaps(), main(), DataFrame, Series, V8 Audit: KNN Imputation Temporal Order Check.  Purpose: Verify whether KNNImp, Run the KNN temporal audit. (+3 more)

### Community 72 - "Module Group 72"
Cohesion: 0.26
Nodes (9): create_sequences_segment_aware(), evaluate_dl(), main(), DataFrame, v9 Phase 5B.4 — Retrain DL Models (GRU/LSTM).  Retrains PyTorch models (GRU, LST, Create sequences strictly within segments to prevent False Continuity., RNNModel, run_pipeline() (+1 more)

### Community 73 - "Module Group 73"
Cohesion: 0.21
Nodes (7): _clip_physical_bounds(), Clip values to physically valid ranges., _prepare_hybrid_data(), Interactive Model Training Service for Dashboard.  Provides trainers for Light, Train and evaluate LightGBM.          Args:             progress_callback: Ca, Load and prepare hybrid dataset., TestClipPhysicalBounds

### Community 74 - "Module Group 74"
Cohesion: 0.18
Nodes (9): _cubic_spline_fill(), Series, Extend interpolation window to fill longer gaps., Fill NaN gaps using Cubic Spline, respecting max_gap limit.      Only fills gaps, _strategy_extended_interp(), Extended interpolation should retain more rows than segment-only., Cubic spline should not modify known (non-NaN) values., Gaps longer than max_gap should NOT be filled. (+1 more)

### Community 75 - "Module Group 75"
Cohesion: 0.26
Nodes (5): create_lag_features(), Create lag features for target and optionally for feature columns.      Args:, Lag features must only use PAST data (shift ≥ 1)., Segment-aware lag should NOT leak across segment boundaries., TestLagFeatures

### Community 76 - "Module Group 76"
Cohesion: 0.26
Nodes (11): load_config(), load_model_config(), merge_configs(), Any, Path, Configuration loader — đọc YAML configs., Load YAML config file.      Args:         config_path: Path to YAML config fi, Load model-specific config by name.      Args:         model_name: Model name (+3 more)

### Community 77 - "Module Group 77"
Cohesion: 0.22
Nodes (11): _cached_sensor_preview(), _cached_suggestion_values(), _forecast_auto(), _forecast_manual(), _pm25_color(), Load latest sensor data — uses cached pipeline to avoid re-loading 209K rows., Cached suggestion values — uses cached pipeline data., Forecast using latest data from dataset — show sensor preview first. (+3 more)

### Community 78 - "Module Group 78"
Cohesion: 0.25
Nodes (10): _compute_ensemble_preds(), _detect_device(), _load_external_preds(), main(), precompute_horizon(), Pre-compute Actual vs Predicted data for all horizons.  Saves results as JSON, Compute Ensemble predictions from GRU + LightGBM using weights from ensemble JSO, Detect best available PyTorch device. (+2 more)

### Community 79 - "Module Group 79"
Cohesion: 0.29
Nodes (10): compute_unified_mase(), get_actuals(), _load_latest(), load_predictions(), load_targets(), main(), ndarray, V9 Statistical Methodology — Cross-Resolution MASE + DM-HLN + R².  Loads exist (+2 more)

### Community 80 - "Module Group 80"
Cohesion: 0.24
Nodes (6): KnowledgeBase, Vector-based knowledge base using ChromaDB + sentence-transformers., Lazy-init ChromaDB collection., Check if knowledge base has been indexed., Get number of indexed documents., Search for relevant context given a query.          Returns list of {content, so

### Community 81 - "Module Group 81"
Cohesion: 0.20
Nodes (6): GRUTrainer, Save model to user_trained directory., Train GRU with user-specified params., Train and evaluate GRU., Export GRU to TorchScript and save., Log run to dashboard_runs.

### Community 82 - "Module Group 82"
Cohesion: 0.27
Nodes (7): create_leakage_free_stl_target(), GRUModel, main(), ndarray, P0-8 FIX — Re-run STL Residual with TRAIN-ONLY STL fitting.  Issue: Previous r, Create STL residual target WITHOUT leakage.      Strategy:         1. Fit STL, train_model()

### Community 83 - "Module Group 83"
Cohesion: 0.29
Nodes (9): _evaluate_horizon(), main(), _prepare_hybrid_data(), DataFrame, Multi-Horizon Evaluation — Compare ML vs Persistence at 1h, 6h, 24h.  HYPOTHES, Load raw data and apply Hybrid imputation strategy., Evaluate all models at a specific forecast horizon.      Multi-step target: pm, Save multi-horizon results to JSON. (+1 more)

### Community 84 - "Module Group 84"
Cohesion: 0.38
Nodes (9): export_acf_pacf(), export_correlations(), export_distributions(), export_missing_barcode(), export_psd(), export_qq_plot(), export_stl(), load_data() (+1 more)

### Community 85 - "Module Group 85"
Cohesion: 0.29
Nodes (5): create_rolling_features(), Create rolling window statistics.      Shift(1) ensures no data leakage — window, Rolling window must use shift(1) to prevent leakage., Rolling should reset at segment boundaries., TestRollingFeatures

### Community 86 - "Module Group 86"
Cohesion: 0.20
Nodes (4): Run PM2.5 prediction., Create a new experiment., Log a model within a run., Log metrics for a model.

### Community 87 - "Module Group 87"
Cohesion: 0.20
Nodes (6): Cross-reference our results with known literature values.      Literature refere, Persistence MAE should be 1-10 µg/m³ for hourly indoor PM2.5., Theoretical expectations for MASE at different horizons.          Theory (autoco, If MASE < 0.1 for any model → almost certainly leakage., Literature expectations for model comparison.          Expected pattern (PM2.5 l, TestResultPlausibility

### Community 88 - "Module Group 88"
Cohesion: 0.31
Nodes (8): export_gru(), export_lgbm(), main(), prepare_data(), Model Export — Convert best models to portable formats.  Exports:   1. GRU →, Train and export LightGBM to native .txt format., Load and prepare hybrid data., Train GRU and export to TorchScript.

### Community 89 - "Module Group 89"
Cohesion: 0.28
Nodes (8): Interactive Sankey diagram showing data flow through the pipeline., Render a detailed Sankey for a single resolution, showing every pipeline step., _render_detail_sankey(), _tab_pipeline_journey(), _get_hub_pipeline_metrics(), Compute pipeline metrics from actual data files — zero hardcode., get_plotly_config(), Get standardized Plotly configuration for the dashboard.      .. deprecated::

### Community 90 - "Module Group 90"
Cohesion: 0.31
Nodes (8): extract_v9_model(), load_v10_metrics(), load_v9_metrics(), main(), v10 Ablation Study — Phase 3: Compare v9 (Domain Bounds) vs v10 (IQR).  Reads v9, Load v9 standardized metrics., Load latest v10 ablation metrics., Extract metrics for a model from v9 standardized structure.          Structure:

### Community 91 - "Module Group 91"
Cohesion: 0.36
Nodes (7): main(), _prepare_hybrid_data(), _print_summary(), Linear Models Multi-Horizon Evaluation — Lasso, ElasticNet, Ridge.  Integrates, Save results to JSON., Load raw data and apply Hybrid imputation strategy., _save_results()

### Community 92 - "Module Group 92"
Cohesion: 0.25
Nodes (7): cleanup_old_v7(), merge_v7(), Phase 1: Normalize snapshot versions — rename files, update version keys, merge, Remove old v7 files after merge., Rename file and update version key inside JSON., Merge v7_retrain (13 models) INTO v7_cqr (6 models).      Strategy:     - Start, rename_and_update()

### Community 93 - "Module Group 93"
Cohesion: 0.36
Nodes (7): main(), _prepare_hybrid_data(), _print_summary(), Sklearn Models Multi-Horizon Evaluation — RF, GradientBoosting, Stacking, Ensemb, Save results to JSON., Load raw data and apply Hybrid imputation strategy., _save_results()

### Community 94 - "Module Group 94"
Cohesion: 0.36
Nodes (7): main(), plot_autocorrelation_story(), plot_horizon_scatter(), DataFrame, Bẫy Tự Tương Quan (The Autocorrelation Trap) Mục tiêu: Chứng minh tại sao Baseli, Vẽ ACF/PACF để thấy tín hiệu giảm mạnh sau 24h., Vẽ Scatter matrix giữa y_t và y_{t-h} để thấy sự xói mòn theo horizon.

### Community 95 - "Module Group 95"
Cohesion: 0.36
Nodes (7): main(), plot_distribution(), plot_spikes_timeline(), DataFrame, Khó khăn do Đỉnh cực đoan (Erratic Spikes) Mục tiêu: Cho thấy sự phân bố đuôi dà, Vẽ phân phối Fat-Tailed của PM2.5., Vẽ chuỗi thời gian 1 tháng kèm cảnh báo mức độ.

### Community 96 - "Module Group 96"
Cohesion: 0.36
Nodes (7): main(), plot_hexbin_density(), plot_rolling_correlation(), DataFrame, Sự dịch chuyển của Đa Biến (Multivariate Shift) Mục tiêu: Cho thấy mối tương qua, Vẽ tương quan chạy (Rolling Correlation) theo Window = 14 ngày., Vẽ Hexbin 2D thay vì Scatter để xử lý overplotting.

### Community 97 - "Module Group 97"
Cohesion: 0.32
Nodes (5): create_diff_features(), Create rate-of-change / difference features.      Domain-specific: PM2.5 rate of, Diff uses shift(1) → diff_1s[t] = y[t-1] - y[t-2] (anti-leakage).          shift, Diff should not cross segment boundaries., TestDiffFeatures

### Community 98 - "Module Group 98"
Cohesion: 0.33
Nodes (6): fetch_aq_chunk(), fetch_weather_chunk(), DataFrame, Script: Fetch external data from Open-Meteo APIs for missing periods in the IoT, Fetch air quality data (PM2.5) for a date range using CAMS Global., Fetch historical weather data for a date range.

### Community 99 - "Module Group 99"
Cohesion: 0.29
Nodes (4): DataFrame, Predict PM2.5 with CQR prediction intervals.          Args:             recen, Predict PM2.5 from a single feature-engineered row.          Args:, Predict PM2.5 from recent data.          Args:             recent_data: DataF

### Community 100 - "Module Group 100"
Cohesion: 0.38
Nodes (6): Path, Path validation utilities — security rules from SKILL.md §1.7., Validate and resolve a file path within the project.      Args:         path: Fi, Validate a data file path (must be in dataset/ directory).      Args:         pa, validate_data_path(), validate_path()

### Community 101 - "Module Group 101"
Cohesion: 0.40
Nodes (5): audit_file(), main(), Path, Audit Dashboard — Quality Gate for hardcoded metrics detection.  Scans app.py, p, Scan a file for hardcoded metric patterns.      Returns list of violations: [{li

### Community 102 - "Module Group 102"
Cohesion: 0.47
Nodes (5): clean_and_format_file(), get_git_commits(), Lấy 5 commit gần nhất để phân tích., Đảm bảo file được lưu dưới dạng UTF-8 chuẩn và loại bỏ các ký tự lỗi., update_memory()

### Community 103 - "Module Group 103"
Cohesion: 0.47
Nodes (5): load_latest(), main(), mase_manual(), V9 Standardize Evaluation — Resolve MASE confound.  This script ensures that t, Compute MASE using a pre-calculated naive MAE denominator.

### Community 104 - "Module Group 104"
Cohesion: 0.40
Nodes (5): main(), Path, Snapshot Validator — Validates JSON schema for dashboard snapshots.  Ensures eve, Validate a single snapshot file against the schema contract.      Returns:, validate_snapshot()

### Community 105 - "Module Group 105"
Cohesion: 0.33
Nodes (5): Tests for feature engineering modules.  Per SKILL.md test spec: - Lag/rolling cr, Create a sample hourly DataFrame for testing., Create a sample DataFrame with segment IDs for testing segment-aware features., sample_df(), segmented_df()

### Community 106 - "Module Group 106"
Cohesion: 0.33
Nodes (4): Verify data pipeline produces valid outputs., Train data must be BEFORE test data (no temporal leakage)., Filtering imputed data should reduce test set size., TestDataIntegrity

### Community 107 - "Module Group 107"
Cohesion: 0.40
Nodes (4): data_files, generated_at, models, version

### Community 110 - "Module Group 110"
Cohesion: 0.50
Nodes (4): main(), Pre-compute Actual vs Predicted data — fully isolated version.  Each model typ, Run inline Python script in isolated subprocess., _run_script()

### Community 111 - "Module Group 111"
Cohesion: 0.50
Nodes (3): p(), Quick leakage audit — verifies specific leakage hypotheses., Print with immediate flush.

### Community 112 - "Module Group 112"
Cohesion: 0.67
Nodes (3): build_ensemble(), _load_latest(), V9 Ensemble Model Builder.  Combines the best Deep Learning model (LSTM_v9) an

### Community 116 - "Module Group 116"
Cohesion: 0.67
Nodes (3): DataFrame, Train LightGBM and evaluate on REAL data only.      Split: 80/10/10 temporal s, _train_and_evaluate()

## Knowledge Gaps
- **134 isolated node(s):** `version`, `generated_at`, `models`, `data_files`, `time-series-forecasting` (+129 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **149 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_features()` connect `Module Group 26` to `Multi-Horizon Experiments`, `Citation Management`, `Module Group 22`, `Module Group 37`, `Module Group 42`, `Module Group 43`, `Module Group 44`, `Module Group 47`, `Module Group 50`, `Module Group 51`, `Module Group 54`, `Module Group 55`, `Module Group 58`, `Module Group 59`, `Module Group 65`, `Module Group 67`, `Module Group 69`, `Module Group 73`, `Module Group 75`, `Module Group 78`, `Module Group 82`, `Module Group 83`, `Module Group 85`, `Module Group 88`, `Module Group 91`, `Module Group 93`, `Module Group 97`, `Module Group 105`?**
  _High betweenness centrality (0.133) - this node is a cross-community bridge._
- **Why does `load_raw_data()` connect `Module Group 22` to `Data Validation & Schema`, `Multi-Horizon Experiments`, `Data Pipeline & Imputation`, `Citation Management`, `Module Group 25`, `Module Group 26`, `Module Group 37`, `Module Group 44`, `Module Group 47`, `Module Group 49`, `Module Group 51`, `Module Group 55`, `Module Group 58`, `Module Group 59`, `Module Group 66`, `Module Group 67`, `Module Group 69`, `Module Group 71`, `Module Group 73`, `Module Group 78`, `Module Group 82`, `Module Group 83`, `Module Group 88`, `Module Group 91`, `Module Group 93`, `Module Group 100`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `APIClient` connect `Module Group 33` to `Module Group 64`, `Streamlit Dashboard UI`, `Module Group 132`, `Info Cards & EDA`, `Module Group 42`, `Module Group 80`, `Module Group 113`, `Module Group 51`, `Module Group 86`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **What connects `version`, `generated_at`, `models` to the rest of the system?**
  _134 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Data Validation & Schema` be split into smaller, more focused modules?**
  _Cohesion score 0.07552447552447553 - nodes in this community are weakly interconnected._
- **Should `Streamlit Dashboard UI` be split into smaller, more focused modules?**
  _Cohesion score 0.08166969147005444 - nodes in this community are weakly interconnected._
- **Should `Leakage Audit & Hashing` be split into smaller, more focused modules?**
  _Cohesion score 0.06818181818181818 - nodes in this community are weakly interconnected._