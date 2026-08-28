# Graph Report - .  (2026-08-11)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1686 nodes · 2744 edges · 254 communities (84 shown, 170 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9073a5b6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DataValidator
- pages.py
- PersistenceModel
- StandardScaler
- experiments.py
- schemas.py
- app.py
- explainability_hub.py
- test_smoke.py
- ReportingEngine
- main.py
- add_cv_features
- chart
- DataFrame
- ExperimentLogger
- DataFrame
- chart_factory.py
- _resample
- test_loader.py
- load_raw_data
- pipeline_walkthrough.py
- run_residual_diagnostics
- llm_client.py
- APIClient
- mae
- test_evaluation.py
- run_full_eda
- clean_data
- snapshot_adapter.py
- evaluate_forecast
- info_cards.py
- test_imputer.py
- build_features
- ContentManager
- render_references_section
- mase
- _build_knn_features
- impute_missing_data
- temporal.py
- theme.py
- metrics.py
- run_ml.py
- knowledge_base.py
- create_calendar_features
- KnowledgeBase
- TimeSeriesAugmenter
- forecast_bias
- create_lag_features
- load_config
- chat_page.py
- _cubic_spline_fill
- create_rolling_features
- ._post
- TestResultPlausibility
- loader.py
- segmenter.py
- temporal_train_val_test_split
- create_diff_features
- ChatGuardrails
- split_real_imputed
- builder.py
- load_all_snapshots
- get_version_data
- validate_snapshot
- TestTemporalSplit
- test_features.py
- TestDataIntegrity
- manifest.json
- reproduce.sh
- _identify_gaps
- adaptive_conformal_inference
- mape
- dependencies
- logging.py
- ._put
- run_dashboard.sh
- api/__init__.py
- init_database
- chatbot/__init__.py
- api_client.py
- frontend/__init__.py
- inference/__init__.py
- training/__init__.py
- Docker Publish
- Keep Render App Alive
- Any
- Path
- Ctu Logo
- ============================================================
- LSTM — PM2.5 Forecasting
- Random Forest — PM2.5 Forecasting
- Transformer — PM2.5 Forecasting
- XGBoost — PM2.5 Forecasting
- ═══════════════════════════════════════════
- ═══════════════════════════════════════════
- Figure
- Lgbm 1H
- Lgbm 24H
- Lgbm 6H
- ndarray
- DataFrame
- time-series-forecasting
- 🌫️ Time Series Forecasting — PM2.5 Air Quality Prediction
- Diagnostics Gru Deseas Raw H6
- Diagnostics Gru Deseas Seasonal Diff H6
- Diagnostics Gru Deseas Stl Residual H6
- Diagnostics Gru Stl Leakfree H6
- Stationarity 1St Diff (D=1)
- Stationarity Log 1St Diff
- Stationarity Log Pm2.5
- Stationarity Raw Pm2.5
- Stationarity Seasonal Diff (D=24H)
- Diagnostics 1H Gru
- Diagnostics 1H Lightgbm
- Diagnostics 1H Persistence
- Diagnostics 24H Gru
- Diagnostics 24H Lightgbm
- Diagnostics 24H Persistence
- Diagnostics 6H Gru
- Diagnostics 6H Lightgbm
- Diagnostics 6H Persistence
- 01 Descriptive Stats
- 02 All Timeseries
- 02 Pm25 Timeseries
- 03 Boxplots
- 03 Distributions
- 03C Qq Plot
- 04 Correlations
- 05 Stl Decomposition
- 05A Stl Seasonal Zoom
- 05B Boxplot Hourly
- 05C Periodogram
- 06 Acf Pacf
- 06 Error Anatomy
- 07 Granger Causality
- 07 Rolling Stats
- 07 Temporal Patterns
- 08 Cross Correlation
- 08 Missing Values
- 01A Autocorrelation Memory
- 01B Horizon Scatter Dispersion
- 02A Pm25 Fat Tailed Distribution
- 02B Pm25 Erratic Spikes
- 03A Rolling Correlation
- 03B Hexbin Multivariate
- 04A Missing Barcode
- 04B Recovery Limits
- Ablation Outlier Impact
- Pi Conformal Prediction Lightgbm 1H
- Pi Conformal Prediction Lightgbm 24H
- Pi Conformal Prediction Lightgbm 6H
- Gru Permutation 1H
- Gru Permutation 24H
- Gru Permutation 6H
- Project Info
- Shap Bar 1H
- Shap Bar 24H
- Shap Bar 6H
- Shap Beeswarm 1H
- Shap Beeswarm 24H
- Shap Beeswarm 6H
- Shap Dep 1H Co2
- Shap Dep 1H Fourier Daily Sin 2
- Shap Dep 1H Pm25 Lag 1H
- Shap Dep 1H Pm25 Roll 24H Mean
- Shap Dep 24H Fourier Daily Cos 2
- Shap Dep 24H Hour Cos
- Shap Dep 24H Pm25 Lag 1H
- Shap Dep 24H Pm25 Lag 24H
- Shap Dep 6H Hour Sin
- Shap Dep 6H Pm25 Roll 24H Min
- Shap Dep 6H Pm25 Roll 24S Min
- Shapash Report Lgbm 1H
- DataFrame
- Series
- Path
- DataFrame
- DataFrame
- DataFrame
- DataLoader
- Dataset
- Module
- ndarray
- Series
- Dataset
- ndarray
- DataFrame
- DataLoader
- Dataset
- Module
- ndarray
- DataFrame
- DataLoader
- Dataset
- Module
- ndarray
- DataFrame
- Dataset
- DataFrame
- ndarray
- DataFrame
- Series
- DataFrame
- DataFrame
- DataFrame
- DataFrame
- Path
- DataFrame
- Series
- DataFrame
- DataFrame
- DataFrame
- DataFrame
- DataFrame
- DataFrame
- Path
- Path
- DataFrame
- DataFrame
- DataFrame
- Path
- Series
- DataFrame
- ndarray
- Path
- DataFrame
- DataFrame
- DataFrame
- ndarray
- DataFrame
- Series
- ndarray
- DataFrame
- DataFrame
- DataFrame
- DataFrame
- DataFrame
- DataFrame
- DataFrame
- DataFrame
- ndarray

## God Nodes (most connected - your core abstractions)
1. `DataValidator` - 49 edges
2. `APIClient` - 29 edges
3. `ContentManager` - 28 edges
4. `ExperimentLogger` - 25 edges
5. `load_raw_data()` - 24 edges
6. `impute_missing_data()` - 22 edges
7. `evaluate_forecast()` - 22 edges
8. `ValidationResult` - 20 edges
9. `build_features()` - 20 edges
10. `cite()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `TestAudit` --uses--> `Base`  [INFERRED]
  tests/api/test_smoke.py → src/api/database.py
- `TestExperiments` --uses--> `Base`  [INFERRED]
  tests/api/test_smoke.py → src/api/database.py
- `TestHealthCheck` --uses--> `Base`  [INFERRED]
  tests/api/test_smoke.py → src/api/database.py
- `TestRunModelsAndMetrics` --uses--> `Base`  [INFERRED]
  tests/api/test_smoke.py → src/api/database.py
- `TestRuns` --uses--> `Base`  [INFERRED]
  tests/api/test_smoke.py → src/api/database.py

## Import Cycles
- None detected.

## Communities (254 total, 170 thin omitted)

### Community 0 - "DataValidator"
Cohesion: 0.08
Nodes (27): Enum, DataValidator, Any, DataFrame, Data validator — domain-specific quality checks for PM2.5 pipeline.  Validates d, Validate feature-engineered data (Marts layer).          Checks per SKILL.md §3., Check if any CRITICAL validation failed., Validation severity levels. (+19 more)

### Community 1 - "pages.py"
Cohesion: 0.06
Nodes (59): DataFrame, _cached_pipeline_data(), _cached_sensor_preview(), _cached_suggestion_values(), _detect_available_models(), _forecast_auto(), _forecast_manual(), _format_model_label() (+51 more)

### Community 2 - "PersistenceModel"
Cohesion: 0.07
Nodes (31): get_all_baselines(), HourlyMeanModel, MeanModel, NaiveBaseline, PersistenceModel, DataFrame, ndarray, Series (+23 more)

### Community 3 - "StandardScaler"
Cohesion: 0.05
Nodes (29): GRUTrainer, LightGBMTrainer, Train and evaluate LightGBM.          Args:             progress_callback: Ca, Save model to user_trained directory., Log run to dashboard_runs., Train GRU with user-specified params., Train and evaluate GRU., Export GRU to TorchScript and save. (+21 more)

### Community 4 - "experiments.py"
Cohesion: 0.09
Nodes (43): Base, Centralized experiment tracking — write to DB or API.  Provides ExperimentLogger, Initialize database — create all tables.  Usage:     # From project root:     uv, Experiment, FeatureImportance, Metric, ORM models for experiment tracking.  Schema design: Relational + JSONB (hybrid)., Evaluation metrics for a model.      Fixed columns for fast queries (MAE, RMSE, (+35 more)

### Community 5 - "schemas.py"
Cohesion: 0.09
Nodes (38): get_audit_report(), get_data_hashes(), get_model_weights(), _md5_file(), Path, Audit router — Data & model hash verification.  Endpoints:     GET /audit/dat, Generate full audit report (data + models)., Verify file integrity by comparing current MD5 with expected hashes from manifes (+30 more)

### Community 6 - "app.py"
Cohesion: 0.12
Nodes (36): _count_tests(), _get_pipeline_metrics(), insight_card(), kpi_card(), load_experiment_results(), load_json(), main(), page_content_manager() (+28 more)

### Community 7 - "explainability_hub.py"
Cohesion: 0.11
Nodes (36): _generate_shapash_html(), _get_best_mase(), _get_hub_pipeline_metrics(), _image_to_plotly(), _insight_card(), _load_json(), page_explainability_hub(), Figure (+28 more)

### Community 8 - "test_smoke.py"
Cohesion: 0.07
Nodes (19): DeclarativeBase, Base, Declarative base for all ORM models., override_get_db(), API tests — Experiments CRUD, Inference, Audit endpoints.  Uses FastAPI TestClie, Test /api/v1/runs endpoints., Test model logging and metric recording., Helper: create experiment → run → return run_id. (+11 more)

### Community 9 - "ReportingEngine"
Cohesion: 0.07
Nodes (19): DataFrame, Reporting Engine — Single Source of Truth for Dashboard metrics.  Loads normaliz, Get best model for all horizons.          Returns:             {"1h": {...}, "6h, Generate ranking table sorted by MAE, with star markers for best.          Args:, Get ranking table formatted for display with ⭐ markers.          Returns DataFra, Get MASE values formatted for chart plotting.          Args:             model_f, Get MAE values formatted for chart plotting.          Returns:             {"Per, Get best model per family for clean Top-5 charts.          Selects the model wit (+11 more)

### Community 10 - "main.py"
Cohesion: 0.09
Nodes (31): FastAPI, Request, get_db(), Session, Database engine, session, and Base for SQLAlchemy 2.0.  Usage:     from src.api., FastAPI dependency — yields a DB session, auto-closes., FastAPI dependencies — DB session, model cache., health_check() (+23 more)

### Community 11 - "add_cv_features"
Cohesion: 0.08
Nodes (22): add_cv_features(), DataFrame, Tests for DL retrain v2 components — CV features and data pipeline., CV uses shift(1) on lag_1h → no access to current value., Test that DL feature selection works correctly., Feature selection should exclude target, is_imputed., Add Coefficient of Variation features with safeguard., Test log1p/expm1 inverse transform correctness. (+14 more)

### Community 12 - "chart"
Cohesion: 0.10
Nodes (32): Bar, Scatter, plot_dm_test_heatmap(), plot_mae_trend(), plot_mae_trend_top5(), plot_mase_comparison(), plot_mase_comparison_top5(), Figure (+24 more)

### Community 13 - "DataFrame"
Cohesion: 0.08
Nodes (23): feature_cols(), marts_df(), DataFrame, Data leakage detection tests for the forecasting pipeline.  These tests ensure n, Ensure temporal ordering and no future information., Verify lag features equal shifted target values., Rolling features should be computed on shifted (past) values only.          A ro, Randomization test: model should fail with shuffled target. (+15 more)

### Community 14 - "ExperimentLogger"
Cohesion: 0.11
Nodes (13): ExperimentLogger, Any, Log a model within a run, return run_model_id., Log metrics for a model, return metric_id., Check if experiment with given name already exists., Parse a full JSON result dict → Experiment + Runs + Models + Metrics.          E, Convert NaN/Infinity to None for DB storage., Centralized experiment tracking — write to DB or API. (+5 more)

### Community 15 - "DataFrame"
Cohesion: 0.15
Nodes (14): _clip_physical_bounds(), _handle_outliers(), Clip values to physically valid ranges., Detect and replace outliers with NaN (will be interpolated later).      Strategy, get_default_params(), _prepare_hybrid_data(), Interactive Model Training Service for Dashboard.  Provides trainers for Light, Load and prepare hybrid dataset. (+6 more)

### Community 16 - "chart_factory.py"
Cohesion: 0.11
Nodes (27): _chart_bootstrap_ci(), _chart_mase_decay(), _chart_residual_bias(), _chart_shap_comparison(), _chart_train_time(), _load_bootstrap_ci(), _load_ljungbox(), _load_sensitivity() (+19 more)

### Community 17 - "_resample"
Cohesion: 0.09
Nodes (19): Resample to regular frequency using mean aggregation., _resample(), get_latest_data(), get_suggestion_values(), GRUPredictor, GRUQuantilePredictor, LightGBMPredictor, DataFrame (+11 more)

### Community 18 - "test_loader.py"
Cohesion: 0.11
Nodes (20): get_data_summary(), Any, Run basic validation checks on dataset.      Args:         df: Input DataFrame., Generate summary statistics about the dataset.      Args:         df: Input Data, validate_dataset(), DataFrame, Tests for data loader module., Should detect duplicated timestamps. (+12 more)

### Community 19 - "load_raw_data"
Cohesion: 0.10
Nodes (17): load_raw_data(), DataFrame, Path, Check that all expected columns exist., Load raw PM2.5 dataset from CSV.      Args:         path: Path to CSV file. Defa, _validate_columns(), Integration tests against the real dataset., Real dataset should load successfully. (+9 more)

### Community 20 - "pipeline_walkthrough.py"
Cohesion: 0.13
Nodes (27): cite(), Return an inline HTML tooltip for a citation.      Args:         ref_id: Key, get_current_version(), Get the currently selected version from session state., _get_dashboard_content(), _get_pipeline_metrics(), _load_standardized_metrics(), page_pipeline_walkthrough() (+19 more)

### Community 21 - "run_residual_diagnostics"
Cohesion: 0.11
Nodes (22): dm_test_hln(), _ensure_statsmodels(), _generate_diagnostic_chart(), _ljung_box_test(), Any, ndarray, Path, Residual Diagnostics — Post-model residual analysis.  Implements Peixeiro Ch.6 (+14 more)

### Community 22 - "llm_client.py"
Cohesion: 0.14
Nodes (23): OpenAI, _build_client(), chat_stream(), check_connection(), get_available_models(), Multi-LLM Client for PM2.5 AI Assistant.  Supports tiered fallback across multip, List models available on a provider., Attempt to stream from a single provider. Returns None on failure. (+15 more)

### Community 23 - "APIClient"
Cohesion: 0.11
Nodes (12): APIClient, Get metrics for a model., Get data file hashes for audit., Get model weight hashes for audit., Get full audit report., Verify file integrity against manifest.json expected MD5 hashes., List all info cards, optionally filtered by page., Get a single info card by key. (+4 more)

### Community 24 - "mae"
Cohesion: 0.11
Nodes (15): mae(), Mean Absolute Error — Primary metric per SKILL.md., Root Mean Squared Error — penalizes large errors., rmse(), TestMAE, TestRMSE, Validation tests — cross-reference results with literature & best practices.  Sc, RMSE >= MAE always (Cauchy-Schwarz inequality). (+7 more)

### Community 25 - "test_evaluation.py"
Cohesion: 0.10
Nodes (13): r2_score(), R² (coefficient of determination)., Winkler Score for interval prediction evaluation (Winkler 1972).      Penalizes, Symmetric MAPE — safer than MAPE for near-zero values.      Per evaluation-metri, smape(), winkler_score(), Tests for evaluation metrics and data splitter., TestNaivePredictions (+5 more)

### Community 26 - "run_full_eda"
Cohesion: 0.20
Nodes (22): _correlation_analysis(), _descriptive_stats(), _missing_analysis(), _plot_acf_pacf(), _plot_distributions(), _plot_time_series(), Any, DataFrame (+14 more)

### Community 27 - "clean_data"
Cohesion: 0.09
Nodes (23): clean_data(), get_cleaning_stats(), _interpolate_gaps(), Any, DataFrame, Data cleaner — clean, handle missing values, outliers, and resample.  Pipeline:, Set datetime column as index., Interpolate missing values, respecting max gap size.      Args:         df: Data (+15 more)

### Community 28 - "snapshot_adapter.py"
Cohesion: 0.13
Nodes (21): _compute_best_models(), _compute_top_n(), _derive_models(), extract_mase(), _extract_results(), load_all_normalized(), _normalize_model_entry(), _normalize_results() (+13 more)

### Community 29 - "evaluate_forecast"
Cohesion: 0.12
Nodes (14): evaluate_forecast(), medae(), Median Absolute Error — robust to outliers.      More robust than MAE for fat-ta, Evaluate a forecast with all metrics.      Args:         y_true: Actual values., Tests for P0-3: Forecast Bias, P0-4: RMSE/MAE Ratio, P2-4: MedAE, and P0-5: Resi, P2-4: Median Absolute Error — robust to outliers., MedAE should be less affected by outliers than MAE., P0-4: RMSE/MAE Ratio — outlier detection. (+6 more)

### Community 30 - "info_cards.py"
Cohesion: 0.14
Nodes (20): cards_actual_vs_predicted(), cards_ai_assistant(), cards_eda(), cards_forecast(), cards_hyperparams(), cards_multi_horizon(), cards_prediction_intervals(), cards_shap() (+12 more)

### Community 31 - "test_imputer.py"
Cohesion: 0.11
Nodes (17): get_imputation_stats(), Get statistics about real vs imputed data in a DataFrame., clean_data(), hourly_data_with_gaps(), DataFrame, Tests for data imputer — multi-strategy missing data recovery.  Each test has ve, KNN feature matrix must NOT contain pm25 (anti-leakage)., ML imputation should fill gaps up to max_gap. (+9 more)

### Community 32 - "build_features"
Cohesion: 0.15
Nodes (13): build_features(), _create_domain_features(), get_feature_columns(), DataFrame, Path, Domain-specific features for air quality per SKILL.md §5.3.      ANTI-LEAKAGE: U, Categorize feature columns by type for documentation.      Returns:         Dict, Save Marts-ready data with optional validation.      Args:         df: Feature-r (+5 more)

### Community 33 - "ContentManager"
Cohesion: 0.16
Nodes (6): ContentManager, Get content specific to a snapshot version., Get global content that is shared across versions (e.g. literature, general info, Manager class to handle loading and providing textual content for the dashboard., Lazy-load info cards from JSON export file (Tier 2 fallback).          Caches th, Get info card content with 3-tier fallback.          Tier 1: API (PostgreSQL via

### Community 34 - "render_references_section"
Cohesion: 0.15
Nodes (16): page_conclusion(), Page: Kết Luận & Hướng Phát Triển — Thesis conclusion dashboard.  Designed for a, Research limitations — honest scientific assessment., Main entry point for the Conclusion & Future Work page., Future research directions with feasibility assessment., Research summary — what was accomplished., _render_future_work(), _render_limitations() (+8 more)

### Community 35 - "mase"
Cohesion: 0.14
Nodes (11): mase(), Mean Absolute Scaled Error — BẮT BUỘC benchmark per SKILL.md.      MASE < 1.0 →, TestMASE, Verify MASE = MAE_model / MAE_naive (Hyndman & Koehler 2006).      Reference: ht, Verify MASE matches manual calculation., Shuffle test: if we randomize targets, model should fail (MASE ≈ 1.0+).      Thi, After shuffling targets, MASE should be >> 1.0 (no signal)., MASE = 1.0 exactly when model = naive baseline. (+3 more)

### Community 36 - "_build_knn_features"
Cohesion: 0.18
Nodes (11): _build_knn_features(), Build feature matrix for KNN imputation.      ANTI-LEAKAGE: Uses only auxiliary, _create_test_series(), DataFrame, Test KNN imputation temporal order — no future data leakage.  Verifies that the, Integration test: run full hybrid imputation and verify temporal safety., Create a synthetic hourly time series with a known gap., Verify KNN imputation uses only past data. (+3 more)

### Community 37 - "impute_missing_data"
Cohesion: 0.22
Nodes (15): _apply_knn_imputation(), impute_missing_data(), DataFrame, Data imputer — Multi-strategy missing data recovery for IoT time series.  Strate, Drop all gaps > max_gap. Interpolate only very short gaps (≤2h)., Extend interpolation window to fill longer gaps., KNN imputation using auxiliary features (temperature, humidity, CO2).      ANTI-, Tiered approach:     - Gap ≤ max_gap_interp: Cubic Spline interpolation     - Ga (+7 more)

### Community 38 - "temporal.py"
Cohesion: 0.17
Nodes (13): create_ewm_features(), _ewm_col(), Series, Temporal feature engineering — lag, rolling, EWM features.  Implements SKILL.md, # NOTE: These are in TIME STEPS, not hours. At 1h freq → 1 step = 1h., Create Exponentially Weighted Moving features.      Shift(1) to prevent leakage., Shift a series, optionally within segments.      Args:         series: The serie, Apply rolling aggregation, optionally within segments.      Args:         series (+5 more)

### Community 39 - "theme.py"
Cohesion: 0.22
Nodes (14): Unified Visualization Theme Framework (VTF) for PM2.5 Dashboard.  Provides centr, annotation_bbox(), apply_mpl_theme(), apply_plotly_style(), get_plotly_config(), get_plotly_template(), get_theme(), Unified Visualization Theme — Single Source of Truth.  Centralized design toke (+6 more)

### Community 40 - "metrics.py"
Cohesion: 0.21
Nodes (14): classification_metrics(), _compute_roc_auc(), evaluate_forecast_full(), mase_hyndman(), nmpiw(), pollution_event_f1(), ndarray, Evaluation metrics for time series forecasting.  Implements SKILL.md §9 metrics: (+6 more)

### Community 41 - "run_ml.py"
Cohesion: 0.19
Nodes (14): _get_lightgbm(), get_ml_models(), _get_xgboost(), main(), DataFrame, Series, Level 2-3: ML Models with Walk-Forward Validation.  Usage:     uv run python, Train with walk-forward CV then evaluate on test set.      Per SKILL.md §6.3: (+6 more)

### Community 42 - "knowledge_base.py"
Cohesion: 0.22
Nodes (12): _chunk_text(), _load_dashboard_content_json(), _load_experiment_results(), _load_info_cards_from_db(), _load_markdown_docs(), RAG Knowledge Base for PM2.5 Project AI Assistant.  Indexes project documentatio, Load user-curated info cards for RAG indexing.      3-tier fallback:         Tie, Load structured dashboard content JSON as knowledge documents. (+4 more)

### Community 43 - "create_calendar_features"
Cohesion: 0.21
Nodes (7): create_calendar_features(), _get_season(), DataFrame, Calendar feature engineering — time-based categorical features.  Implements SKIL, Create calendar/temporal categorical features from DatetimeIndex.      Features, Map month to season (Vietnam tropical climate).      0: Spring transition (Feb-M, TestCalendarFeatures

### Community 44 - "KnowledgeBase"
Cohesion: 0.19
Nodes (8): get_knowledge_base(), KnowledgeBase, Vector-based knowledge base using ChromaDB + sentence-transformers., Lazy-init ChromaDB collection., Check if knowledge base has been indexed., Get number of indexed documents., Search for relevant context given a query.          Returns list of {content, so, Get or create singleton knowledge base.

### Community 45 - "TimeSeriesAugmenter"
Cohesion: 0.20
Nodes (7): Time Series Data Augmentation Module.  Implements techniques to artificially aug, Applies data augmentation to time series sequences., Args:             technique (str): Augmentation technique to apply ('jitter' or, Apply augmentation to a batch of sequences.          Args:             X (np.nda, Add Gaussian noise to the sequences.         This helps models become robust aga, Scale the sequences by a random factor.         This helps models generalize acr, TimeSeriesAugmenter

### Community 46 - "forecast_bias"
Cohesion: 0.21
Nodes (8): forecast_bias(), Forecast Bias — over- or under-forecasting indicator.      Ref: Manu Joseph Ch.4, P0-3: Forecast Bias metric — Manu Joseph Ch.4 p.80., FB ≈ 0 when predictions match actuals., FB > 0 when model over-predicts., FB < 0 when model under-predicts (dangerous for PM2.5)., Returns NaN when total actual is near zero., TestForecastBias

### Community 47 - "create_lag_features"
Cohesion: 0.26
Nodes (5): create_lag_features(), Create lag features for target and optionally for feature columns.      Args:, Lag features must only use PAST data (shift ≥ 1)., Segment-aware lag should NOT leak across segment boundaries., TestLagFeatures

### Community 48 - "load_config"
Cohesion: 0.26
Nodes (11): load_config(), load_model_config(), merge_configs(), Any, Path, Configuration loader — đọc YAML configs., Load YAML config file.      Args:         config_path: Path to YAML config fi, Load model-specific config by name.      Args:         model_name: Model name (+3 more)

### Community 49 - "chat_page.py"
Cohesion: 0.27
Nodes (9): _ensure_index(), _get_knowledge_base(), page_ai_assistant(), AI Assistant page for PM2.5 Forecasting Dashboard.  Provides a chat interface wi, Render AI Assistant chatbot page., Lazy import and get knowledge base singleton., Ensure knowledge base is indexed, re-index if user content changed.      Checks, Render AI provider configuration in sidebar. (+1 more)

### Community 50 - "_cubic_spline_fill"
Cohesion: 0.22
Nodes (7): _cubic_spline_fill(), Series, Fill NaN gaps using Cubic Spline, respecting max_gap limit.      Only fills gaps, Extended interpolation should retain more rows than segment-only., Cubic spline should not modify known (non-NaN) values., Gaps longer than max_gap should NOT be filled., TestExtendedInterp

### Community 51 - "create_rolling_features"
Cohesion: 0.29
Nodes (5): create_rolling_features(), Create rolling window statistics.      Shift(1) ensures no data leakage — window, Rolling window must use shift(1) to prevent leakage., Rolling should reset at segment boundaries., TestRollingFeatures

### Community 52 - "._post"
Cohesion: 0.20
Nodes (4): Run PM2.5 prediction., Create a new experiment., Log a model within a run., Log metrics for a model.

### Community 53 - "TestResultPlausibility"
Cohesion: 0.20
Nodes (6): Cross-reference our results with known literature values.      Literature refere, Persistence MAE should be 1-10 µg/m³ for hourly indoor PM2.5., Theoretical expectations for MASE at different horizons.          Theory (autoco, If MASE < 0.1 for any model → almost certainly leakage., Literature expectations for model comparison.          Expected pattern (PM2.5 l, TestResultPlausibility

### Community 54 - "loader.py"
Cohesion: 0.31
Nodes (7): Data loader — load, validate, and provide basic info about PM2.5 dataset., Path, Path validation utilities — security rules from SKILL.md §1.7., Validate and resolve a file path within the project.      Args:         path: Fi, Validate a data file path (must be in dataset/ directory).      Args:         pa, validate_data_path(), validate_path()

### Community 55 - "segmenter.py"
Cohesion: 0.28
Nodes (8): get_segment_stats(), identify_contiguous_segments(), DataFrame, Data segmenter — Identify and manage contiguous data segments.  For IoT time ser, Validate that no segment contains a time gap larger than allowed.      This catc, Assign segment IDs to contiguous non-NaN blocks in the target column.      Each, Get summary statistics about segments in a DataFrame.      Args:         df: Dat, validate_segment_boundaries()

### Community 56 - "temporal_train_val_test_split"
Cohesion: 0.25
Nodes (8): create_naive_predictions(), DataFrame, ndarray, Series, Data splitter for time series — temporal split, NO shuffle.  Per SKILL.md §6.3:, Split data temporally: oldest → train → val → test → newest.      NEVER shuffles, Create naive baseline predictions for MASE calculation.      Per SKILL.md Level, temporal_train_val_test_split()

### Community 57 - "create_diff_features"
Cohesion: 0.28
Nodes (6): create_diff_features(), DataFrame, Create rate-of-change / difference features.      Domain-specific: PM2.5 rate of, Diff uses shift(1) → diff_1s[t] = y[t-1] - y[t-2] (anti-leakage).          shift, Diff should not cross segment boundaries., TestDiffFeatures

### Community 59 - "split_real_imputed"
Cohesion: 0.33
Nodes (5): Split DataFrame into real and imputed portions.      Use this to ensure test set, split_real_imputed(), Split should not lose any rows., After imputation, splitting test to real-only should work., TestSplitRealImputed

### Community 60 - "builder.py"
Cohesion: 0.33
Nodes (5): Feature builder — orchestrates all feature creation and produces Marts-ready dat, create_fourier_features(), DataFrame, Fourier feature engineering — capture daily and weekly seasonality.  Fourier fea, Create Fourier sin/cos features for daily and weekly seasonality.      For each

### Community 61 - "load_all_snapshots"
Cohesion: 0.33
Nodes (6): cards_experiment_runs(), load_all_snapshots(), Load all version snapshots from dashboard_runs/, normalized., Info cards for Experiment Runs page., Render version selector in sidebar. Returns selected version name., version_selector_sidebar()

### Community 62 - "get_version_data"
Cohesion: 0.33
Nodes (6): cards_overview(), get_version_data(), Info cards for Overview page., Get snapshot data for a specific version., Render a small version badge at top of page., render_version_badge()

### Community 63 - "validate_snapshot"
Cohesion: 0.40
Nodes (5): main(), Path, Snapshot Validator — Validates JSON schema for dashboard snapshots.  Ensures eve, Validate a single snapshot file against the schema contract.      Returns:, validate_snapshot()

### Community 65 - "test_features.py"
Cohesion: 0.33
Nodes (5): Tests for feature engineering modules.  Per SKILL.md test spec: - Lag/rolling cr, Create a sample hourly DataFrame for testing., Create a sample DataFrame with segment IDs for testing segment-aware features., sample_df(), segmented_df()

### Community 66 - "TestDataIntegrity"
Cohesion: 0.33
Nodes (4): Verify data pipeline produces valid outputs., Train data must be BEFORE test data (no temporal leakage)., Filtering imputed data should reduce test set size., TestDataIntegrity

### Community 67 - "manifest.json"
Cohesion: 0.40
Nodes (4): data_files, generated_at, models, version

### Community 68 - "reproduce.sh"
Cohesion: 0.70
Nodes (4): err(), log(), reproduce.sh script, warn()

### Community 69 - "_identify_gaps"
Cohesion: 0.40
Nodes (4): _identify_gaps(), Identify contiguous NaN gap segments.      Returns:         DataFrame with colum, Gap identifier should find correct gap segments., TestGapIdentification

### Community 70 - "adaptive_conformal_inference"
Cohesion: 0.40
Nodes (4): adaptive_conformal_inference(), ndarray, Adaptive Conformal Inference (ACI) for non-stationary time series.  Implements A, Run ACI on test predictions using calibration residuals.      Unlike static CQR

### Community 71 - "mape"
Cohesion: 0.50
Nodes (3): mape(), Mean Absolute Percentage Error.      Skips near-zero values to avoid division by, TestMAPE

### Community 72 - "dependencies"
Cohesion: 0.50
Nodes (3): docx, dependencies, docx

## Knowledge Gaps
- **91 isolated node(s):** `run_dashboard.sh script`, `Docker Publish`, `Keep Render App Alive`, `🌫️ Time Series Forecasting — PM2.5 Air Quality Prediction`, `============================================================` (+86 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **170 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `APIClient` connect `APIClient` to `ContentManager`, `knowledge_base.py`, `._put`, `KnowledgeBase`, `api_client.py`, `._post`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `ContentManager` connect `ContentManager` to `get_version_data`, `load_all_snapshots`, `info_cards.py`, `APIClient`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `DataValidator` connect `DataValidator` to `build_features`, `load_raw_data`, `clean_data`, `builder.py`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `DataValidator` (e.g. with `TestIntermediateValidation` and `TestMartsValidation`) actually correct?**
  _`DataValidator` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `APIClient` (e.g. with `KnowledgeBase` and `ContentManager`) actually correct?**
  _`APIClient` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `ExperimentLogger` (e.g. with `Experiment` and `FeatureImportance`) actually correct?**
  _`ExperimentLogger` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `run_dashboard.sh script`, `Docker Publish`, `Keep Render App Alive` to the rest of the system?**
  _91 weakly-connected nodes found - possible documentation gaps or missing edges._