# TODO - Công việc cần làm

> Cập nhật: **2026-04-05 10:48**. Tất cả 5 Phase chính đã hoàn thành.

## ✅ ALL 5 PHASES COMPLETE

## 🟡 Có thể làm thêm (chiều nay)
- [ ] Review lại thesis draft — kiểm tra số liệu consistency
- [ ] Chạy lại dashboard để demo trước khi nộp
- [ ] Cập nhật PROJECT_WALKTHROUGH.md với TFT + PI
- [ ] Chạy thêm experiments nếu cần (TFT tuning, PI calibration)
- [ ] Git commit toàn bộ changes

## ✅ Đã Hoàn Thành
- [x] **Phase 5: TFT Experiment ✅** — MASE: 1h=1.029 | 6h=0.822 | 24h=0.812
- [x] **Phase 4: Model Export ✅** — GRU→TorchScript, LightGBM→Native (.txt)
- [x] **Phase 3: Streamlit Dashboard ✅** — Scientific Observatory + TFT charts
- [x] **Phase 2: Prediction Intervals ✅** — Conformal + Quantile + MC Dropout
- [x] **Phase 1: Thesis Draft ✅** — Ch1-5 + TFT + [13] Lim et al. citation
- [x] Infrastructure + Dev tools + Testing (133/133 passed)
- [x] Data pipeline: Raw → Clean → Intermediate → Marts (7,574 × 95)
- [x] EDA 8 sections + Stationarity check (ADF+KPSS)
- [x] Feature engineering 95 features (anti-leakage ✅)
- [x] Baseline Level 0: Persistence MAE=1.821 (cleaned_hourly), 2.493 (hybrid h=1)
- [x] Leakage Audit & Fix ✅ — 4 nguồn fix
- [x] ML Level 2-3: LightGBM/Lasso/Ridge/RF/XGB
- [x] Optuna Tuning ✅ — LightGBM best across horizons
- [x] Multi-horizon LightGBM (Optuna) ✅ — h=6: 0.745, h=24: 0.842
- [x] ARIMA/SARIMA ✅ — SARIMA 24h MASE=0.813
- [x] DL (GRU/LSTM) ✅ — GRU 24h **MASE=0.727** ⭐⭐ BEST
- [x] Ensemble (LightGBM+GRU stacking) ✅ — GRU_ens 6h MASE=0.698 ⭐
- [x] SHAP Explainability ✅ — LightGBM SHAP + GRU Permutation
- [x] Diebold-Mariano Test ✅ — GRU significant ở 6h & 24h
- [x] Residual Diagnostics ✅
- [x] EDA Data Storytelling ✅
- [x] Literature Comparison ✅ — 4 papers (2023-2025) + TFT [13]
- [x] Lint fixed ✅ — scripts clean
