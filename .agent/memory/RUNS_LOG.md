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
| mh-001 | 2026-04-04 19:37 | Persistence_1h | Hybrid data, h=1 | 2.390 | 3.852 | — | 1.000 | Baseline |
| mh-002 | 2026-04-04 19:37 | LightGBM_tuned_1h | Optuna 100t, h=1 | 2.419 | 3.568 | — | 1.012 | ❌ gap=1.2% |
| mh-003 | 2026-04-04 19:37 | Persistence_6h | Hybrid data, h=6 | 6.942 | 9.500 | — | 1.000 | Baseline |
| mh-004 | 2026-04-04 19:37 | **LightGBM_tuned_6h** | Optuna 100t, h=6 | **5.071** | 6.629 | — | **0.730** | ✅ **BEATS -27%** ⭐ |
| mh-005 | 2026-04-04 19:37 | Persistence_24h | Hybrid data, h=24 | 6.357 | 8.679 | — | 1.000 | Baseline |
| mh-006 | 2026-04-04 19:37 | **LightGBM_tuned_24h** | Optuna 100t, h=24 | **5.160** | 6.698 | — | **0.812** | ✅ **BEATS -19%** |
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

> ✅ **[2026-04-04]** Leakage fix verified. Runs ml-001→ml-006 INVALIDATED. Clean: ml-007→ml-016.
> 🔥 **[2026-04-04]** Optuna tuning: LightGBM_tuned_all MAE=1.874, **MASE=1.029** — gap chỉ còn 2.9% vs Persistence.
> 📊 Feature insight: pm25_lag_1h chiếm 83.8% importance. All-features > top-20 selection.
> 📊 **[2026-04-04 18:45]** Strategy comparison: Hybrid best (MASE=1.066), ext_interp worst (1.321). Test=REAL data only.
> ⚠️ Key finding: Cubic spline alone gây noise → tệ hơn. KNN impute tốt hơn vì multivariate context.
> 🏆 **[2026-04-04 19:37]** MULTI-HORIZON LightGBM: 6h=MASE 0.730 (-27%), 24h=MASE 0.812 (-19%).
> 📊 Feature shift: 1h→pm25_lag_1h dominant. 6h→hour_sin+rolling_mean. 24h→multivariate (diem_suong, nhiet_do).
> 🏆 **[2026-04-04 19:59]** ARIMA/SARIMA: SARIMA 24h ≈ LightGBM (0.813 vs 0.812)! Seasonal s=24 rất mạnh.
> 🏆🏆 **[2026-04-04 20:12]** GRU 24h = **MASE 0.727** — NEW BEST! Vượt LightGBM 10.5%. GRU > LSTM toàn diện.
> 📊 Final ranking 6h: LightGBM(0.730) > SARIMA(0.762) > GRU(0.812) > ARIMA(0.856) > LSTM(0.914).
> 📊 Final ranking 24h: **GRU(0.727)** > LightGBM(0.812) ≈ SARIMA(0.813) > LSTM(0.830) > ARIMA(0.913).
> 📊 1h: Persistence(1.000) > LightGBM(1.012) > ARIMA(1.023) > GRU(1.173) > SARIMA(1.283) > LSTM(1.560).
> ⏭️ Next: Ensemble (stacking), SHAP Explainability, Thesis write-up (QĐ 1799 CTU, IEEE).
