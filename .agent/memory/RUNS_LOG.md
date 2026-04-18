# EXPERIMENT RUNS LOG

> Bảng ghi lại tất cả các lần chạy thí nghiệm. Cập nhật sau **MỖI** lần chạy.

| Run ID | Timestamp | Model | Config Summary | MAE | RMSE | R² | MASE | Notes |
|--------|-----------|-------|----------------|-----|------|----|------|-------|
| baseline-001 | 2026-03-29 | Persistence | lag_1h | 1.821 | 3.306 | — | 1.000 | Level 0 benchmark ✅ |
| baseline-002 | 2026-03-29 | SeasonalNaive | lag_24h | 4.197 | 6.418 | — | 2.305 | Level 0 |
| baseline-003 | 2026-03-29 | HistoricalMean | global mean | 7.715 | 8.502 | — | 4.237 | Level 0 |
| baseline-004 | 2026-03-29 | HourlyMean | hour mean | 6.628 | 7.693 | — | 3.640 | Level 0 |
| ~~ml-001~~ | ~~2026-03-29~~ | ~~Ridge~~ | — | ~~0.004~~ | ~~0.006~~ | ~~1.000~~ | ~~0.002~~ | ❌ INVALIDATED: leakage |
| ~~ml-002~~ | ~~2026-03-29~~ | ~~Lasso~~ | — | ~~0.106~~ | ~~0.132~~ | ~~0.999~~ | ~~0.058~~ | ❌ INVALIDATED: leakage |
| ~~ml-003~~ | ~~2026-03-29~~ | ~~ElasticNet~~ | — | ~~0.371~~ | ~~0.535~~ | ~~0.989~~ | ~~0.204~~ | ❌ INVALIDATED: leakage |
| ~~ml-004~~ | ~~2026-03-29~~ | ~~RandomForest~~ | — | ~~0.143~~ | ~~0.383~~ | ~~0.995~~ | ~~0.078~~ | ❌ INVALIDATED: leakage |
| ~~ml-005~~ | ~~2026-03-29~~ | ~~XGBoost~~ | — | ~~0.189~~ | ~~0.395~~ | ~~0.994~~ | ~~0.104~~ | ❌ INVALIDATED: leakage |
| ~~ml-006~~ | ~~2026-03-29~~ | ~~LightGBM~~ | — | ~~0.221~~ | ~~0.384~~ | ~~0.995~~ | ~~0.121~~ | ❌ INVALIDATED: leakage |
| ml-007 | 2026-04-04 16:45 | Lasso | α=0.1, post-fix | **1.915** | 3.187 | — | 1.052 | ✅ Clean, MASE>1 |
| ml-008 | 2026-04-04 16:45 | ElasticNet | α=0.1, l1=0.5, post-fix | 2.037 | 3.339 | — | 1.118 | ✅ Clean, MASE>1 |
| ml-009 | 2026-04-04 16:45 | LightGBM | n=500, depth=8, post-fix | 2.276 | 3.266 | — | 1.250 | ✅ Clean, MASE>1 |
| ml-010 | 2026-04-04 16:45 | RandomForest | n=200, depth=12, post-fix | 2.666 | 3.668 | — | 1.464 | ✅ Clean, MASE>1 |
| ml-011 | 2026-04-04 16:45 | Ridge | α=1.0, post-fix | 2.824 | 4.591 | — | 1.551 | ✅ Clean, MASE>1 |
| ml-012 | 2026-04-04 16:45 | XGBoost | n=500, depth=6, post-fix | 3.364 | 4.345 | — | 1.847 | ✅ Clean, MASE>1 |
| ml-013 | 2026-04-04 17:13 | LightGBM_tuned_all | Optuna 150t, all 94 feats | **1.874** | 3.070 | — | **1.029** | ✅ Closest! Gap=2.9% |
| ml-014 | 2026-04-04 17:13 | LightGBM_tuned | Optuna 150t, top-20 feats | 1.937 | 3.103 | — | 1.063 | ✅ Feature-selected |
| ml-015 | 2026-04-04 17:13 | Lasso_tuned_all | Optuna 100t, all 94 feats | 1.946 | 3.149 | — | 1.069 | ✅ |
| ml-016 | 2026-04-04 17:13 | Lasso_tuned | Optuna 100t, top-20 feats | 2.050 | 3.213 | — | 1.126 | ✅ |
| strategy-001 | 2026-04-04 18:45 | LightGBM_segment | max_gap=2h, 7335→7167 rows | 2.619 | 3.562 | — | 1.113 | Strategy A (baseline) |
| strategy-002 | 2026-04-04 18:45 | LightGBM_ext_interp | CubicSpline, max=12h, 7386→7217 | **2.977** | 3.760 | — | **1.321** | Strategy B ⚠️ worst |
| strategy-003 | 2026-04-04 18:45 | LightGBM_ml_impute | KNN k=5, max=24h, 7742→7574 | 2.606 | 3.690 | — | 1.084 | Strategy C |
| strategy-004 | 2026-04-04 18:45 | LightGBM_hybrid | Spline≤6h+KNN 6-24h, 7742→7574 | **2.562** | 3.593 | — | **1.066** | Strategy D ⭐ best |
| ~~mh-001~~ | ~~2026-04-04~~ | ~~Persistence_1h~~ | ⚠️ BUG: target=y[t] | ~~2.390~~ | — | — | — | ❌ INVALID (v1) |
| ~~mh-002~~ | ~~2026-04-04~~ | ~~LightGBM_tuned_1h~~ | ⚠️ BUG: target=y[t] | ~~2.419~~ | — | — | ~~1.012~~ | ❌ INVALID (v1) |
| mh-v2-001 | 2026-04-04 21:52 | Persistence_1h | **FIXED** target=y[t+1] | 2.493 | 4.011 | — | 1.000 | Baseline |
| mh-v2-002 | 2026-04-04 21:52 | LightGBM_tuned_1h | Optuna 100t, h=1, FIXED | 3.720 | 5.078 | — | 1.492 | ❌ ML thua Persist |
| mh-v2-003 | 2026-04-04 21:52 | Persistence_6h | **FIXED** persist=y[t] | 6.773 | 9.120 | — | 1.000 | Baseline |
| mh-v2-004 | 2026-04-04 21:52 | **LightGBM_tuned_6h** | Optuna 100t, h=6, FIXED | **5.046** | 6.584 | — | **0.745** | ✅ **-25.5%** ⭐ |
| mh-v2-005 | 2026-04-04 21:52 | Persistence_24h | **FIXED** persist=y[t] | 6.153 | 8.629 | — | 1.000 | Baseline |
| mh-v2-006 | 2026-04-04 21:52 | **LightGBM_tuned_24h** | Optuna 100t, h=24, FIXED | **5.178** | 6.705 | — | **0.842** | ✅ **-15.8%** |
| ar-001 | 2026-04-04 19:59 | ARIMA(2,1,1)_1h | Rolling w=720, h=1 | 2.564 | 3.925 | — | 1.023 | ❌ |
| ar-002 | 2026-04-04 19:59 | ARIMA(2,1,1)_6h | Rolling w=720, h=6 | 5.843 | 7.685 | — | 0.856 | ✅ -14% |
| ar-003 | 2026-04-04 19:59 | ARIMA(2,1,1)_24h | Rolling w=720, h=24 | 5.598 | 7.325 | — | 0.913 | ✅ -9% |
| sr-001 | 2026-04-04 19:59 | SARIMA(1,0,0)×(2,1,0,24)_1h | s=24, h=1 | 3.214 | 4.682 | — | 1.283 | ❌ worst |
| sr-002 | 2026-04-04 19:59 | **SARIMA_6h** | s=24, h=6 | **5.207** | 6.908 | — | **0.762** | ✅ -24% |
| sr-003 | 2026-04-04 19:59 | **SARIMA_24h** | s=24, h=24 | **4.981** | 6.510 | — | **0.813** | ✅ **≈ LightGBM!** |
| dl-001 | 2026-04-04 20:12 | LSTM_1h | lookback=72, h=1 | 3.730 | 4.794 | — | 1.560 | ❌ worst |
| dl-002 | 2026-04-04 20:12 | GRU_1h | lookback=72, h=1 | 2.805 | 3.971 | — | 1.173 | ❌ |
| dl-003 | 2026-04-04 20:12 | LSTM_6h | lookback=72, h=6 | 5.765 | 7.717 | — | 0.914 | ✅ -9% |
| dl-004 | 2026-04-04 20:12 | GRU_6h | lookback=72, h=6 | 5.119 | 6.547 | — | 0.812 | ✅ -19% |
| dl-005 | 2026-04-04 20:12 | LSTM_24h | lookback=72, h=24 | 5.211 | 6.810 | — | 0.830 | ✅ -17% |
| dl-006 | 2026-04-04 20:12 | **GRU_24h** | lookback=72, h=24 | **4.562** | 6.766 | — | **0.727** | ✅ **BEST EVER!** ⭐⭐ |
| ens-001 | 2026-04-04 21:01 | LightGBM_ens_1h | n=300, h=1 | 3.912 | — | — | 1.569 | ❌ |
| ens-002 | 2026-04-04 21:01 | GRU_ens_1h | lookback=72, h=1 | 2.723 | — | — | 1.092 | ❌ |
| ens-003 | 2026-04-04 21:01 | **Stack_1h** | Ridge meta | 3.099 | — | — | 1.354 | ❌ |
| ens-004 | 2026-04-04 21:01 | LightGBM_ens_6h | n=300, h=6 | 4.936 | — | — | 0.729 | ✅ -27% |
| ens-005 | 2026-04-04 21:01 | **GRU_ens_6h** | lookback=72, h=6 | **4.729** | — | — | **0.698** | ✅ **-30%** ⭐ |
| ens-006 | 2026-04-04 21:01 | Weighted_6h | w_lgbm=0.75 | 4.873 | — | — | 0.720 | ✅ -28% |
| ens-007 | 2026-04-04 21:01 | LightGBM_ens_24h | n=300, h=24 | 6.144 | — | — | 0.998 | ≈ Persistence |
| ens-008 | 2026-04-04 21:01 | **GRU_ens_24h** | lookback=72, h=24 | **4.492** | — | — | **0.730** | ✅ **-27%** |
| ens-009 | 2026-04-04 21:01 | **Stack_24h** | Ridge meta | **4.367** | — | — | **0.784** | ✅ -22% |
| tft-001 | 2026-04-05 03:05 | TFT_1h | hidden=32, heads=4, 25K params | **2.573** | — | — | **1.029** | ⭐ Best ML/DL at h=1! |
| tft-002 | 2026-04-05 03:05 | TFT_6h | hidden=32, heads=4 | 5.565 | — | — | 0.822 | ✅ |
| tft-003 | 2026-04-05 03:05 | TFT_24h | hidden=32, heads=4 | 4.999 | — | — | 0.812 | ✅ |
| pi-001 | 2026-04-05 02:45 | Conformal_PI | LightGBM, α=0.1 | — | — | — | — | Coverage: 80.5% (1h), 76% (6h) |
| pi-002 | 2026-04-05 02:45 | Quantile_PI | LightGBM, α=[0.05,0.95] | — | — | — | — | Coverage: 86.2% (1h) ⭐ best |
| pi-003 | 2026-04-05 02:45 | MC_Dropout_PI | GRU, n=50 forward | — | — | — | — | Coverage: 36.8% (1h) ⚠️ needs cal |

> ✅ **[2026-04-04]** Leakage fix verified. Runs ml-001→ml-006 INVALIDATED. Clean: ml-007→ml-016.
> 🔥 **[2026-04-04]** Optuna tuning: LightGBM_tuned_all MAE=1.874, **MASE=1.029** — gap chỉ còn 2.9% vs Persistence.
> 📊 Feature insight: pm25_lag_1h chiếm 83.8% importance. All-features > top-20 selection.
> 📊 **[2026-04-04 18:45]** Strategy comparison: Hybrid best (MASE=1.066), ext_interp worst (1.321). Test=REAL data only.
> ⚠️ Key finding: Cubic spline alone gây noise → tệ hơn. KNN impute tốt hơn vì multivariate context.
> 🔧 **[2026-04-04 21:52]** PIPELINE AUDIT & FIX v2: Fixed 2 critical bugs in multi_horizon_eval.py
> 🏆🏆 **[2026-04-04 20:12]** GRU 24h = **MASE 0.727** — BEST! Vượt LightGBM (0.842).
> 🧠 **[2026-04-05 03:05]** TFT: MASE 1h=1.029 (⭐ best ML/DL), 6h=0.822, 24h=0.812.
> ✅ **[2026-04-05 10:48]** ALL 5 PHASES COMPLETE. Tests: 133/133.

---

## RC Integration Phase (v2→v3) — 2026-04-11 → 2026-04-12

### v2_enhanced (2026-04-11) — Fourier + Interactions + Linear + Log1p

**Lý do**: RC cho thấy Fourier + log1p + interaction features mang lại MAE thấp hơn.
**Bổ sung**: 12 Fourier, 6 rolling range, 6 interaction features, log1p transform, 3 linear models.

| Run ID | Model | h | MAE | MASE | Notes |
|--------|-------|---|-----|------|-------|
| rc-v2-001 | LightGBM_tuned | 1h | 3.193 | 1.281 | ↓14.2% vs v1 (3.720). Fourier=#2 importance |
| rc-v2-002 | LightGBM_tuned | 6h | 4.911 | 0.725 | ↓2.7% vs v1 |
| rc-v2-003 | LightGBM_tuned | 24h | 4.957 | 0.806 | ↓4.3% vs v1 |
| rc-v2-004 | ElasticNet | 6h | 4.846 | 0.715 | ✅ Best linear, bests=0.01/l1=0.5 |
| rc-v2-005 | LassoCV | 1h | 3.641 | 1.461 | ❌ Linear thua Persistence ở 1h |
| rc-v2-006 | ElasticNet | 24h | 5.020 | 0.816 | ✅ Beats Persistence 24h |

### v3_sklearn_ensemble (2026-04-12) — RF + GB + Stacking + Ensemble

**Lý do**: RC Ensemble (weighted) đạt MAE thấp nhất (1.845 single-step). Cần kiểm chứng trên TSF.
**Bổ sung**: RandomForest, GradientBoosting, Stacking (ElasticNet+RF+GB→Ridge), VotingRegressor, Weighted Ensemble.

| Run ID | Model | h | MAE | MASE | Notes |
|--------|-------|---|-----|------|-------|
| rc-v3-001 | RandomForest | 1h | 3.140 | 1.260 | ❌ All sklearn MASE>1 at 1h |
| rc-v3-002 | Ensemble_Weighted | 1h | 3.114 | 1.249 | ❌ Best sklearn but still >1 |
| rc-v3-003 | RandomForest | 6h | 4.782 | 0.706 | ✅ Ngang ElasticNet |
| rc-v3-004 | **Ensemble_Weighted** | **6h** | **4.777** | **0.705** | ✅ RF=80%+GB=20%. Best sklearn |
| rc-v3-005 | GradientBoosting | 6h | 4.880 | 0.721 | ✅ |
| rc-v3-006 | RandomForest | 24h | 4.913 | 0.798 | ✅ |
| rc-v3-007 | **Ensemble_Weighted** | **24h** | **4.907** | **0.797** | ✅ RF=80%+GB=20% |
| rc-v3-008 | Stacking | 24h | 5.272 | 0.857 | ✅ Stacking tệ nhất trong group |

> 📊 **[v3 Conclusion]**: Ensemble_Weighted (RF 80% + GB 20%) = best sklearn ở cả 6h+24h, cạnh tranh LightGBM_tuned. Stacking KHÔNG mang lợi ích rõ rệt. DL (GRU) vẫn dẫn đầu ở 24h.
> 🔑 **Key insight**: Weights search luôn cho RF=80% → RF là backbone chính, GB chỉ diversify. Stacking weight=0%.

---

### v5 — DL Retrain with v2 Features + CV + Log Transform (2026-04-12)

**What**: Retrain GRU/LSTM với 117 features (v2 enhanced: Fourier, interactions, CV) thay vì 5 raw features v1. Compare log1p vs raw target.
**Why**: v2 features đã cải thiện LightGBM ↓14.2%. Test xem DL có benefit tương tự. CV features capture PM2.5 volatility.

| Run ID | Model | Hz | MAE | MASE | Log? | Note |
|--------|-------|----|-----|------|------|------|
| dl-v2-001 | GRU_raw | 1h | 4.118 | 1.723 | ❌ | ❌ Tệ hơn v1 (1.173). 117 features quá nhiều cho 1h |
| dl-v2-002 | LSTM_raw | 1h | 4.513 | 1.888 | ❌ | ❌ |
| dl-v2-003 | GRU_log | 1h | 3.659 | 1.531 | ✅ | ❌ Log giúp nhưng vẫn MASE>1 |
| dl-v2-004 | LSTM_log | 1h | 4.243 | 1.775 | ✅ | ❌ |
| dl-v2-005 | GRU_raw | 6h | 4.936 | 0.783 | ❌ | ✅ v1=0.812 → ↓3.6% |
| dl-v2-006 | **GRU_log** | **6h** | **4.363** | **0.692** | ✅ | ✅ **NEW BEST EVER!** v1=0.812 → ↓14.8% |
| dl-v2-007 | **LSTM_raw** | **6h** | **4.534** | **0.719** | ❌ | ✅ v1=0.914 → ↓21.3% huge gain! |
| dl-v2-008 | LSTM_log | 6h | 4.746 | 0.753 | ✅ | ✅ |
| dl-v2-009 | GRU_raw | 24h | 4.875 | 0.776 | ❌ | ✅ v1=0.727 → +6.8% slight worse |
| dl-v2-010 | GRU_log | 24h | 4.876 | 0.776 | ✅ | ✅ Same as raw |
| dl-v2-011 | LSTM_raw | 24h | 4.683 | 0.746 | ❌ | ✅ v1=0.830 → ↓10.1% |
| dl-v2-012 | **LSTM_log** | **24h** | **4.611** | **0.734** | ✅ | ✅ v1=0.830 → ↓11.5% |

> 📊 **[v5 Conclusion]**:
> - **6h**: GRU_log (MASE=0.692) = NEW BEST model toàn pipeline, thắng cả Ensemble_Weighted (0.705).
> - **24h**: LSTM_log (0.734) > GRU v1 (0.727). v2 features giúp LSTM nhiều hơn GRU ở 24h.
> - **1h**: DL tệ hơn v1 → curse of dimensionality, 117 features quá nhiều cho short horizon.
> - **Log transform**: GRU thích log (especially 6h Δ=9.1%). LSTM ưa raw ở 6h nhưng log ở 24h.
> 🔑 **Key insight**: DL benefits từ v2 features ở medium/long horizons. Tuy nhiên, feature selection/PCA nên thử cho 1h.

---

### v6 — PCA/Feature Selection (1h) + TFT Retrain v2 (2026-04-12)

**What**: PCA (117→37, 95% var) + Top-N feature selection (10/20/40) @ 1h. TFT retrain (113 tempo + 4 static) @ all horizons.
**Why**: v5 cho thấy curse of dimensionality ở 1h. Giả thuyết: PCA/selection giảm noise. TFT cần test v2.

#### PCA/Feature Selection @ 1h

| Run ID | Model | Hz | MAE | MASE | Dim | Note |
|--------|-------|----|-----|------|-----|------|
| v3-001 | GRU_pca | 1h | 3.757 | 1.572 | 37 | ❌ PCA 95% → 37 comp. Better than 117 but < v1 |
| v3-002 | LSTM_pca | 1h | 4.293 | 1.796 | 37 | ❌ |
| v3-003 | GRU_top10 | 1h | 5.986 | 2.505 | 10 | ❌ Quá ít features |
| v3-004 | GRU_top20 | 1h | 4.685 | 1.960 | 20 | ❌ |
| v3-005 | **GRU_top40** | **1h** | **3.577** | **1.497** | 40 | ❌ Best v2 selection nhưng v1 (1.173) vẫn thắng |

#### TFT Retrain with v2 Features

| Run ID | Model | Hz | MAE | MASE | Params | vs v1 |
|--------|-------|----|-----|------|--------|-------|
| v3-006 | TFT_v2 | 1h | 4.722 | 1.976 | 28,545 | ❌ v1=1.029 → +92.0% |
| v3-007 | TFT_v2 | 6h | 5.357 | 0.850 | 28,545 | ✅ v1=0.821 → +3.5% |
| v3-008 | TFT_v2 | 24h | 5.561 | 0.886 | 28,545 | ✅ v1=0.811 → +9.2% |

> 📊 **[v6 Conclusion]**:
> - **PCA/TopN KHÔNG giải quyết 1h**: GRU_top40 (1.497) < GRU_pca (1.572) < v2_full (1.723), nhưng TẤT CẢ đều tệ hơn v1 (1.173).
> - **1h = autocorrelation trap**: Persistence gần perfect ở lag-1. Bất kỳ feature engineering nào cũng thêm noise. Simple model (5 raw features) tốt nhất.
> - **TFT_v2 tệ hơn v1**: 113 temporal features quá nhiều cho hidden_dim=32 (28K params). TFT cần feature selection trước hoặc hidden_dim lớn hơn.
> - **Ranking final": 1h → GRU_v1(1.173). 6h → GRU_log_v2(0.692). 24h → GRU_v1(0.727).
> 🔑 **Key insight**: Feature engineering là CON DAO HAI LƯỠI cho DL: giúp multi-step (6h/24h) nhưng hại single-step (1h). TFT cần dataset >50K rows để handle 113+ features.

### v7 — Pipeline Audit & Enhancement (2026-04-12)

**What**: Audit toàn bộ pipeline đối chiếu 8+ sách chuyên ngành (Manu Joseph, Peixeiro, Brownlee, Vishwas & Patel, Huang). Bổ sung visualization + evaluation metrics thiếu.
**Why**: Đảm bảo pipeline đạt chuẩn academic cho luận văn. Phát hiện 5 chart thiếu và 3 metric thiếu so với khuyến nghị sách.

#### Phase 1: EDA Foundation

| Item | What | Result |
|------|------|--------|
| P0-1 | STL Decomposition (period=24h, LOESS) | Trend=0.647, Seasonal=0.343, Residual σ=5.18 µg/m³ |
| P0-2 | Forecastability Assessment | Score=0.434 (Trung bình), CoV=0.72, ApEn=1.46, ACF(1)=0.90 |
| P0-6 | Box Plot per Hour (24 boxes) | Peak 6h (19.7 µg/m³), Trough 12h (6.3 µg/m³) |
| P0-7 | Mount 4 unmounted charts | rolling_corr, fat_tail, dispersion, recovery → Dashboard |
| P1-1 | Q-Q Plot (normality) | Shapiro-Wilk raw p=1.4e-50 → NOT Normal |
| P1-2 | Periodogram/PSD | Dominant 24h cycle confirmed → validates Fourier |

#### Phase 2: Evaluation Upgrade

| Item | What | Result |
|------|------|--------|
| P0-3 | Forecast Bias metric | `forecast_bias()` → detects over/under-forecasting |
| P0-4 | RMSE/MAE Ratio | Added → >√2 = outlier errors |
| P0-5 | Residual Diagnostics | Ljung-Box + Q-Q + ACF + 4-panel chart |
| P2-4 | MedAE (Median Absolute Error) | `medae()` → robust to fat-tailed PM2.5 |

> 📊 **[v7 Metrics]**:
> - STL Residual σ = 5.18 µg/m³ → best-case MAE floor
> - Forecastability Score = 0.434 → "Trung bình" → explains MASE>1 at h=1
> - PM2.5 NOT Normal (p=1.4e-50) → justifies MASE over MAPE
> - Tests: 167/167 passed ✅
> 🔑 **Key insight**: STL Residual σ (5.18) thiết lập "sàn hiệu suất" — model đạt MAE ≈ 5.2 tức đã khai thác hết signal. GRU v2+log MAE=4.35 tại 6h → đang nằm dưới noise floor → excellent.

### v7-exp — Deseasonalizing Transform Experiment (2026-04-12)

**What**: Thí nghiệm 3 biến thể target: raw, seasonal_diff (y-y[t-24]), STL residual.
**Why**: Manu Joseph Ch.7 khuyến nghị. PM2.5 seasonal strength = 0.343 → deseasonalizing có thể giúp.

| Run ID | Transform | Hz | MAE | MASE | FB | MedAE | Leakage |
|--------|-----------|-----|-----|------|-----|-------|---------|
| v7-001 | raw (baseline) | 6h | 4.608 | 0.731 | +0.016 | 3.547 | ✅ Clean |
| v7-002 | seasonal_diff | 6h | 5.692 | 0.903 | -0.111 | 4.240 | ✅ Clean |
| v7-003 | stl_residual | 6h | 3.198 | 0.507 | -0.016 | 1.825 | ⚠️ STL fitted on full data |

> 📊 **[v7-exp Conclusion]**:
> - **seasonal_diff (0.903) TỆ HƠN raw (0.731)**: Fourier features ALREADY capture seasonality → deseasonalizing thêm nhiễu.
> - **STL residual (0.507) SUSPECT**: STL fitted on entire series incl. test → LOOK-AHEAD BIAS. Kết quả INVALID, cần re-run với train-only STL.
> - **Forecast Bias**: seasonal_diff under-forecasts 11% (FB=-0.111) → dangerous for PM2.5 alerting.
> - **Residual Diagnostics**: Tất cả 3 variants FAIL Ljung-Box → residuals still autocorrelated (normal for DL).
> 🔑 **Key insight**: **Fourier features khiến deseasonalizing explicit TRỞ NÊN DƯ THỪA (redundant)**. Model đã "remove" seasonal via Fourier → thêm seasonal_diff = double-remove → performance giảm. Đây là finding quan trọng cho thesis.

#### v7-exp FIX: STL Leakage Corrected (2026-04-12)

| Run ID | Transform | Hz | MAE | MASE | FB | Leakage |
|--------|-----------|-----|-----|------|-----|---------|
| v7-004 | stl_leakfree | 6h | 4.643 | 0.736 | -0.246 | ✅ Clean (train-only STL) |

> ⚠️ **LEAKAGE CONFIRMED**: STL full-data (0.507) → STL train-only (0.736) = **+45% inflation**.
> STL leak-free (0.736) ≈ raw (0.731) → **deseasonalizing NOT helpful with Fourier features**.
> 📊 **Final v7-exp conclusion**: GRU v2+log (MASE=0.692) remains BEST at 6h. Fourier features make explicit deseasonalizing redundant.

### v7-P4 — Deep Insights Phase 4 (2026-04-12)

**P1-3: Error Anatomy** (GRU v2+log @ h=6, data from avp_6h.json)
- Worst hour: **9h** (MAE=6.880) — morning rush + inversion
- Best hour: **16h** (MAE=3.486) — afternoon convection disperses PM2.5
- Error ACF(1) = 0.8253 → errors highly autocorrelated (model missing temporal patterns)
- Error ACF(24) = 0.0317 → daily pattern in errors minimal (Fourier working)

**P1-4: Granger Causality** (train-only, n=5547, maxlag=24)
- Temperature → PM2.5: ✅ best_lag=1, p<1e-10
- Humidity → PM2.5: ✅ best_lag=1, p<1e-10
- CO2 → PM2.5: ✅ best_lag=6, p<1e-10
- **All 3 external vars Granger-cause PM2.5** — validates feature design

**P1-5: Cross-Correlation** (train-only)
- CO2: best correlation r=0.3668 @ lag=0h (shared emission source)
- Temperature: r=-0.1993 @ lag=0h (weak negative, via atmospheric mechanism)
- Humidity: r=-0.2217 @ lag=-56h (indirect, complex mechanism)
- Dew Point: r=-0.1971 @ lag=49h (weak)

> All data saved to `research/eda/deep_insights_results.json` for cross-reference.
> Dashboard Tab 6 reads ALL values from JSON dynamically.

### v7-outlier — Outlier Strategy Fix (2026-04-12)

**Problem**: IQR×3 cap PM2.5 at 54 µg/m³ — below WHO "Unhealthy" (55.4)
- 1,908 real pollution events removed (0.9% of data)
- PM2.5 skew=3.21, kurt=32.4 — IQR inappropriate for fat-tailed distributions

**Fix**: PM2.5 uses DOMAIN bounds [0, 500] (WHO AQI scale + sensor spec)
- Other variables (temp, humidity, CO2) keep IQR×3
- 1,908 real data points NOW PRESERVED

**Refs**: Manu Joseph Ch.2 (domain > statistical), Peixeiro Ch.3 (context first)
**Impact**: All model results computed with old strategy. Full retrain needed.

### v7-refs — Pipeline References Document (2026-04-12)

Created `docs/PIPELINE_REFERENCES.md` — IEEE format citation mapping:
- 8 sections: Data, EDA, Features, Models, Evaluation, PI, Experiments, Dashboard
- Every decision linked to [Book] Ch.X, pp.Y-Z
- 9 reference books indexed (MJ, PX, JB, VP, HP, TM, DL, AP, AQ)

### v7-retrain — Full Pipeline Retrain (2026-04-12)

**Trigger**: Outlier fix (PM2.5 domain bounds [0,500] thay IQR×3)
**Duration**: 89 min | **Models**: 12 configs × 3 horizons = 36 experiments
**Results**: `research/experiments/v7_retrain/`

#### Summary Table (v7-retrain)
| Horizon | Model | MAE | MASE |
|---------|-------|-----|------|
| **1h** | Persistence | 2.596 | 1.000 |
| **1h** | LSTM_v2_log | 3.242 | - |
| **1h** | Ens_Weighted | 3.216 | 1.239 |
| **1h** | LightGBM | 3.298 | 1.270 |
| **6h** | Ens_Weighted | **4.871** | **0.703** ⭐ |
| **6h** | RandomForest | 4.886 | 0.705 |
| **6h** | LightGBM | 5.081 | 0.733 |
| **6h** | GRU_v2_log | 5.053 | 0.899 |
| **24h** | LSTM_v2_log | **4.461** | **0.740** ⭐ |
| **24h** | ARIMA(2,1,1) | 4.564 | 0.764 |
| **24h** | Ens_Weighted | 4.978 | 0.788 |
| **24h** | RandomForest | 4.999 | 0.791 |

#### Key Findings
1. Persistence MAE tăng (2.60 vs 1.82 cũ) — real pollution spikes giữ lại
2. Ens_Weighted (RF+GB) thay GRU_v2_log ở 6h (0.703 vs 0.899)
3. LSTM_v2_log vượt trội ở 24h (0.740)
4. DL 1h MASE=inf: Persistence MAE~0 trên subset nhỏ → cần standardize
5. ARIMA surprisingly competitive ở 24h (0.764)

### v7-standardize — DL MASE Fix + Model Literature (2026-04-12)

**DL MASE=inf fix**: persist_test trùng target do index offset
- Root cause: DL sequencing offset khiến Persistence=Actual → MAE=0
- Fix: Unified Persistence MAE (2.49, 6.42, 6.45) from ML-style test set
- DL 1h: GRU MASE inf→1.343, LSTM inf→1.300
- DL 6h: GRU MASE 0.899→0.787, LSTM 1.047→0.916
- DL 24h: GRU MASE 0.937→0.875, LSTM 0.740→0.691 ⭐

**Final v7-retrain Rankings (Standardized)**:
| h | Best | MASE |
|---|------|------|
| 1h | Persistence | 1.000 |
| 6h | Ens_Weighted | 0.703 ⭐ |
| 24h | LSTM_v2_log | 0.691 ⭐ |

**Model Selection Literature** added to PIPELINE_REFERENCES.md §10:
- Why each model, why NOT others, data-model fit, results vs literature prediction
