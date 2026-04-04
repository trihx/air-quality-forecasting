# PROJECT CONTEXT

## Trạng thái hiện tại
- **Phase**: DL + Ensemble ✅ — GRU BEST tại 24h!
- **Best model (1h)**: Persistence (MAE=2.390, MASE=1.000) — KHÔNG model nào vượt
- **Best model (6h)**: ⭐ LightGBM_tuned (MAE=5.071, **MASE=0.730**) — giảm 27% vs Persistence!
- **Best model (24h)**: ⭐⭐ GRU (MAE=4.562, **MASE=0.727**) — giảm 27.3%!
- **Strategy tốt nhất**: Hybrid (Spline≤6h + KNN 6-24h)
- **Baseline**: Persistence (MAE=1.821, MASE=1.000) — vẫn BEST
- **Dữ liệu**: `dataset/raw/final_dataset.csv` - 209,594 bản ghi, 6 cột
- **Cleaned**: `dataset/interim/cleaned_hourly.csv` - 6,857 rows (hourly)
- **Marts**: `dataset/processed/marts_features.csv` - 6,689 rows × 95 cols ✅ REBUILT 2026-04-04
- **Thời gian dữ liệu**: 2022-03-25 → 2025-05-11 (~3.1 năm)
- **Tests**: 106/106 passed ✅ (including 9 leakage tests)

## ✅ Data Leakage — ĐÃ FIX & VERIFY (2026-04-04)

### 4 nguồn leakage đã fix:
1. ✅ `pm25_diff_1h`: `shift(1).diff(1)` thay `diff(1)`
2. ✅ `pm25_pct_change_1h`: `shift(1).pct_change(1)` thay `pct_change(1)`
3. ✅ `co2_pm25_ratio`: dùng `pm25_lag_1h` thay `pm25[t]`
4. ✅ `pm25_aqi_cat`: dùng `pm25_lag_1h` thay `pm25[t]`

### Verification:
- ✅ Leakage audit script: 6/6 checks 🟢
- ✅ Leakage test suite: 9/9 passed
- ✅ Top feature correlation: pm25_lag_1h (0.89) — hợp lý
- ✅ Full test suite: 106/106 passed

## Kết quả ML Post-Fix (2026-04-04)

| Model | MAE | MASE | Status |
|-------|-----|------|--------|
| Persistence | **1.821** | **1.000** | ✅ Best |
| Lasso | 1.915 | 1.052 | ❌ MASE>1 |
| LightGBM | 2.276 | 1.250 | ❌ MASE>1 |
| RandomForest | 2.666 | 1.464 | ❌ MASE>1 |
| Ridge | 2.824 | 1.551 | ❌ MASE>1 |
| XGBoost | 3.364 | 1.847 | ❌ MASE>1 |

## Pipeline Architecture
```
Raw (209K rows, ~2min)
  → Clean (dedup, bounds, outliers IQR 3.0, resample 1h)
  → Impute (4 strategies: segment/interp/KNN/hybrid) — is_imputed tracking
  → Intermediate (7,742 rows hourly with hybrid)
  → Features (lag, rolling, EWM, calendar, diff, domain) — anti-leakage ✅
  → Marts (7,574 rows × 95 cols, validated)
  → Split (80/10/10 temporal, NO shuffle, TEST = REAL DATA ONLY)
  → Model → Evaluate (MAE primary, MASE mandatory)
```

## ✅ Result Validation Protocol (BẮT BUỘC)

### Nguyên tắc: Validate-Before-Trust
1. **Cross-reference literature**: MAE PM2.5 phải trong 1.5-15 µg/m³ (hourly)
2. **MASE sanity**: MASE < 0.1 → 🚨 LEAKAGE! MASE ≈ 1.0 ở 1h → ✅ expected (autocorr = 0.97)
3. **Shuffle test**: Shuffle target → retrain → MASE phải >> 1.0
4. **Persistence monotone**: Error tăng theo horizon (1h < 6h < 24h)
5. **GRU ≤ LSTM**: Literature confirms GRU more efficient
6. **Test suite**: `tests/validation/test_result_validation.py` — 15+ tests

### Reference papers:
- Hyndman & Koehler (2006) — MASE definition
- Kapoor & Narayanan (2023) — Shuffle test, leakage detection
- WHO PM2.5 Guidelines (2021) — plausible concentration ranges

## Next Steps
1. → ✅ Ensemble script debug (torch lazy import fix)
2. → Ensemble (stacking LightGBM + GRU)
3. → SHAP Explainability
4. → Walkthrough restructure theo QĐ 1799 CTU (IEEE citation)
5. → Final thesis write-up với kết quả multi-horizon

## Tài liệu quan trọng
- `docs/PROJECT_WALKTHROUGH.md` — Walkthrough toàn bộ dự án cho luận văn
- `.agent/memory/RUNS_LOG.md` — Lịch sử tất cả experiment runs
- `.agent/memory/LESSONS_LEARNED.md` — Bài học kinh nghiệm
