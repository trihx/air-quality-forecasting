# TECHNICAL DECISIONS LOG

> Ghi lại các quyết định kỹ thuật quan trọng để team và AI assistant tham chiếu.

---

## Archived (2026-03-28 → 2026-03-29) — Tóm tắt
<!-- Chi tiết: xem git log hoặc conversation 30cab957 -->
- **Cấu trúc thư mục**: Custom based on Cookiecutter DS + `memory/` cho context engineering
- **SKILL.md**: Hybrid approach (inline rules + separate guides). 777+ dòng → tách 7 guide files
- **Logging**: loguru (research DX > enterprise). Migrate structlog khi production.
- **Visualization**: Data Storytelling approach. 12 chart types, WHO reference line bắt buộc.
- **Feature Engineering**: 95 features, anti-leakage (shift ≥ 1). ⚠️ diff/domain CÓ leakage → ĐÃ FIX.
- **Leakage Audit**: 4 nguồn fix. MASE>1 cho tất cả ML. ✅ HOÀN THÀNH 2026-04-04.
- **Stationarity**: PM2.5 = trend-stationary. ML dùng raw, ARIMA cần d=1.

---

## [2026-04-04] Memory Optimization Strategy
- HOT (mỗi phiên): CONTEXT.md + TODO.md ≤ 100 dòng
- WARM (khi cần): DECISIONS.md, LESSONS_LEARNED.md, RUNS_LOG.md
- COLD (1 lần): SKILL.md, guides/*
- **Compact trigger**: LESSONS_LEARNED > 80 dòng → archive entries > 7 ngày

## [2026-04-04] EDA Strategy — Hybrid Imputation
- So sánh 4 strategies: Segment(1.113) < ML(1.084) < Hybrid(**1.066**) < ExtInterp(1.321)
- ⭐ **Hybrid** (Spline≤6h + KNN 6-24h) = best. Cubic spline alone gây noise.
- **Rule**: Test set = REAL data ONLY

## [2026-04-04] IEEE Citation Style
- Dùng chuẩn IEEE cho trích dẫn (theo QĐ 1799 CTU)

## [2026-04-04 21:52] Multi-Horizon — ML thắng ở horizon dài
- **v2 (FIXED)**: 1h=MASE **1.492** (❌), 6h=MASE **0.745** (-25.5% ✅), 24h=MASE **0.842** (-15.8% ✅)
- Feature shift: 1h→lag, 6h→temporal patterns, 24h→multivariate
- Giả thuyết ĐÚNG: ML tạo giá trị khi autocorrelation giảm

## [2026-04-04 21:52] Pipeline Audit v2 — 2 Critical Bugs Fixed
- **Bug #1**: h=1 target=y[t] → fix: `shift(-h)` cho MỌI horizon
- **Bug #2**: Persistence=lag_Xh=y[t-X] → fix: dùng `df[TARGET_COL]` trực tiếp
- **Impact**: LightGBM h=1 MASE: 1.012 → 1.492. Persistence MAE nhất quán across scripts.
- **Rule**: LUÔN `shift(-h)` cho target + `df[TARGET_COL]` cho Persistence. KHÔNG dùng lag features.

## [2026-04-04] statsmodels + gappy DatetimeIndex
- Drop NaN → mất freq → dùng `.values` (numpy). `forecast()` trả numpy → `[-1]` not `iloc[-1]`.

## [2026-04-05] Hyperparameter Tuning Strategy
- **ML**: Optuna Bayesian (TPE), 50-100 trials, TimeSeriesSplit(5), minimize MAE. Mỗi horizon tune riêng.
- **DL**: Manual config + Early Stopping (patience=10). Optuna DL quá tốn kém với 7742-row dataset.
- **Lưu trữ**: `research/best_models_configs.json` = single source of truth → dùng cho bảng biểu luận văn.
- **Fine-tune**: ML → sửa search space trong `multi_horizon_eval.py`. DL → sửa constants trong `dl_multi_horizon.py`.

## [2026-04-05] SHAP Explainability
- **LightGBM**: TreeExplainer (exact, <0.3s/horizon). Top feature: pm25_lag_1h (1h), pm25_roll_24h_mean (6h).
- **GRU**: Permutation importance (5 features × 5 rounds). MPS GPU + mini-batch (256). 50 epochs + ReduceLROnPlateau.
- **Insight**: SHAP rank ≠ built-in rank → SHAP chính xác hơn vì tính interaction effects.
- **Plots**: `research/figures/shap/` — bar, beeswarm, dependence (LightGBM) + permutation bar (GRU).

## [2026-04-05] TFT — Simplified Temporal Fusion Transformer
- **Quyết định**: Implement Simplified TFT trong pure PyTorch (không dùng pytorch-forecasting) để control architecture.
- **Architecture**: GRN + GLU + Multi-head Attention (4 heads) + Static Encoder. 25,089 params.
- **Sizing**: hidden_dim=32 (nhỏ hơn GRU 64) vì dataset 7.5K rows không đủ cho TFT lớn.
- **Kết quả**: Best ML/DL tại h=1 (MASE=1.029), competitive tại h=6/24 (0.822/0.812).
- **Kết luận**: Attention giúp khai thác short-term patterns tốt hơn GRU, nhưng cần >50K rows để full potential.

## [2026-04-05] Dashboard — Scientific Observatory Design
- **Theme**: Dark (#0E1117) + teal accent (#00D4AA) + glassmorphism KPI cards.
- **Pages**: 6 (Overview, Multi-Horizon, SHAP, PI, EDA, Hyperparameters).
- **Charts**: Plotly (bar, scatter, line) với custom theme. TFT integrated vào tất cả charts.

## [2026-04-05] Prediction Intervals Strategy
- **3 methods**: Conformal (agnostic), Quantile Regression (LightGBM native), MC Dropout (GRU).
- **Best**: Quantile coverage 86.2% (1h). MC Dropout quá narrow (36.8%) do dropout=0.2 nhỏ.
- **Kết luận**: Quantile = recommended. MC Dropout cần calibration hoặc tăng dropout rate.
