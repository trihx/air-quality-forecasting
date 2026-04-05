# LESSONS LEARNED

> Ghi lại **MỌI** lỗi gặp phải và cách khắc phục. Đọc file này TRƯỚC KHI implement để tránh lặp lại lỗi cũ.

---

## Archived (2026-03-29) — Tóm tắt 1 dòng
<!-- Chi tiết: xem git log hoặc conversation logs -->
- **Data Leakage**: `diff/pct_change/ratio/aqi_cat` chứa y[t] → fix: `shift(1).diff()`, dùng `pm25_lag_1h`. R²>0.99 = red flag.
- **CSV Loading chậm**: Dùng `usecols`, cân nhắc parquet. Luôn `flush=True`.
- **Stationarity**: PM2.5 = trend-stationary (ADF ✅, KPSS ❌). Luôn chạy CẢ hai.
- **Unit test phải update**: Thay đổi logic → update test assertions. `shift(1).diff()` NaN tại iloc[1].
- **Kill processes**: `pkill -f "python.*scripts/"` trước pipeline run.

---

## [2026-04-04] Leakage Impact — Kết quả thực tế sau fix
- **Impact**: Ridge MAE: 0.004 → 2.824 (+700x). Persistence vẫn BEST ở h=1.
- **Rule**: Luôn audit TRƯỚC khi celebrate. MASE<0.1 = 🚨 leakage.
- **Reference**: Kapoor & Narayanan (2023), Hyndman & Koehler (2006)

## [2026-04-04] Result Validation Protocol — Validate-Before-Trust
- MAE PM2.5 phải trong 1.5-15 µg/m³ (hourly)
- MASE ≈ 1.0 ở 1h → ✅ expected (autocorr=0.97)
- MASE < 1.0 ở 6h-24h → ✅ expected
- Shuffle test: randomize target → MASE >> 1.0
- Test suite: `tests/validation/test_result_validation.py`

## [2026-04-04] PyTorch MPS + LightGBM = Silent Segfault
- **Nguyên nhân**: `import torch` top-level + MPS Metal → xung đột OpenMP với LightGBM
- **Fix**: Lazy import torch trong hàm GRU, SAU KHI LightGBM train xong
- **Rule**: KHÔNG import torch ở top-level trong script đa mô hình

## [2026-04-04 21:52] Pipeline Audit — Multi-Horizon Target & Persistence Bugs
- **Bug #1**: `multi_horizon_eval.py` h=1 dùng `TARGET_COL` (y[t]) → PHẢI `shift(-1)` (y[t+1])
- **Bug #2**: Persistence dùng `pm25_lag_Xh` (= y[t-X]) → PHẢI dùng `df[TARGET_COL]` (y[t])
- **Impact**: LightGBM h=1 MASE: 1.012 → **1.492**. Persistence MAE giờ consistent.
- **Rule**: Khi tính multi-horizon forecast:
  1. `target = df[TARGET_COL].shift(-h)` → y[t+h]
  2. `persist = df[TARGET_COL]` → y[t] (LƯU TRƯỚC khi shift)
  3. Dùng cùng persist_mae cho tất cả models cùng horizon
  4. KHÔNG dùng lag features cho Persistence
- **Scripts đúng từ đầu**: dl_multi_horizon.py, arima_multi_horizon.py
- **Baseline-001** (MAE=1.821): HỢP LỆ cho h=1, nhưng trên cleaned_hourly (khác hybrid data)

## [2026-04-05] MPS GPU + GRU Training Performance
- **Lỗi**: GRU train ~6000 samples trong 1 batch trên CPU → treo >5 phút, không progress.
- **Fix**: `torch.device("mps")` + DataLoader(batch_size=256) + ReduceLROnPlateau → 27s/horizon.
- **Lỗi 2**: `ReduceLROnPlateau(verbose=False)` → TypeError (PyTorch mới bỏ param `verbose`). Xóa param.
- **Lỗi 3**: GRU chỉ có 4 env features (thiếu pm25 target history) → MAE=13.8 (1h), quá tệ.
  → Fix: thêm `pm25` vào input features → MAE converge hợp lý.
- **Rule**: GRU permutation importance CHỈ có nghĩa khi model đã converge tốt (loss < 0.3).
  Nếu Δ MAE âm toàn bộ → model chưa fit đủ → tăng epochs / thêm features.
- **Hardware**: Apple Silicon M1 Pro — LUÔN dùng `torch.device("mps")`. Tốc độ ~5x so CPU.

## [2026-04-05] Thesis Review — Đối Chiếu Bài Báo & Metric Consistency
- **Lỗi 1**: Ghi "Cao nhất" cho R² bài Nam et al. thay vì giá trị cụ thể (R²=0.70) → LUÔN tìm số liệu gốc.
- **Lỗi 2**: MAE h=1 ghi 1,874 (giá trị cũ từ pipeline leaky) thay vì 3,720 (pipeline v2 fixed).
  → **Rule**: Sau pipeline audit, CẬP NHẬT TẤT CẢ tài liệu tham chiếu, không chỉ code.
- **Lỗi 3**: Ensemble attribution sai — MASE 0,698 thuộc GRU đơn lẻ (trong ensemble run), không phải "GRU Ensemble (Stacking)".
  → **Rule**: Luôn ghi rõ nguồn gốc metric: model nào, run nào, file JSON nào.
- **Bài học**: Bảng trong thesis PHẢI cross-reference với JSON experiment files, không ghi từ trí nhớ.

## [2026-04-05] TFT Sizing — Small Dataset Attention Trap
- **Lỗi tiềm ẩn**: TFT hidden_dim=64 (giống GRU) → overfit nhanh trên 7.5K rows.
- **Fix**: Giảm hidden_dim=32 + num_heads=4 → 25K params (vs GRU 41K). Stable training.
- **Rule**: Transformer trên dataset <10K rows → hidden ≤ 32, heads ≤ 4. Tăng patience (15 vs 10).
- **Insight**: TFT best tại short-horizon (h=1) nhờ Attention, yếu hơn GRU ở long-horizon do thiếu data.

## [2026-04-05] Plotly + Streamlit — CHART_COLORS phải đủ cho N models
- **Lỗi**: Thêm TFT (model thứ 7) nhưng CHART_COLORS chỉ có 6 → IndexError.
- **Fix**: Đảm bảo CHART_COLORS list ≥ số models. Dùng `CHART_COLORS[i % len(CHART_COLORS)]` an toàn hơn.
