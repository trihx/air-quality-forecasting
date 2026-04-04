# TODO - Công việc cần làm

> Cập nhật: **2026-04-04 17:15**. Ưu tiên từ trên xuống dưới.

## 🔥 Ưu Tiên Cao (Ngay)
- [ ] Multi-horizon evaluation (6h, 24h) → ML lợi thế ở horizon dài
- [ ] Verbose progress output cho training scripts (user feedback)

## 🟡 Ưu Tiên Trung Bình
- [ ] Statistical models (ARIMA, SARIMA) — Level 1
- [ ] Diebold-Mariano test cho model comparison (LightGBM vs Persistence)
- [ ] Residual diagnostics (Ljung-Box, Q-Q plot)
- [ ] Deep Learning (LSTM, GRU) — Level 4

## 🔵 Ưu Tiên Thấp
- [ ] Ensemble (Stacking Lasso + LightGBM) — Level 5
- [ ] SHAP explainability
- [ ] Confidence Intervals / Prediction Intervals
- [ ] Streamlit dashboard cập nhật

## ✅ Đã Hoàn Thành (Archived)
<!-- Chi tiết xem docs/PROJECT_WALKTHROUGH.md -->
- [x] Infrastructure + Dev tools + Testing (106/106 passed)
- [x] Data pipeline: Raw → Clean → Intermediate → Marts (6,689 × 95)
- [x] EDA 8 sections + Stationarity check (ADF+KPSS)
- [x] Feature engineering 95 features (anti-leakage ✅)
- [x] Baseline Level 0: Persistence MAE=1.821
- [x] Leakage Audit & Fix ✅ — 4 nguồn fix, 106/106 tests
- [x] ML Level 2-3 re-run ✅ — Clean results (ml-007→ml-012)
- [x] **Optuna Tuning ✅** — LightGBM MASE=1.029, gap 2.9% vs Persistence
- [x] **Feature Selection ✅** — RF importance, top-20 search, all-features won
- [x] PROJECT_WALKTHROUGH.md + Literature Review ✅
