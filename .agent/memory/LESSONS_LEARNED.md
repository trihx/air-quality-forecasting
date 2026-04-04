# LESSONS LEARNED

> Ghi lại **MỌI** lỗi gặp phải và cách khắc phục. Đọc file này TRƯỚC KHI implement để tránh lặp lại lỗi cũ.

---

## [2026-03-29] Data Leakage trong Feature Engineering
- **Triệu chứng**: Ridge regression đạt MAE=0.004, R²=1.0 — kết quả "hoàn hảo" bất thường
- **Nguyên nhân gốc rễ**: Nhiều features chứa thông tin target tại thời điểm t:
  - `diff(1) = y[t] - y[t-1]` → chứa y[t]
  - `pct_change(1)` → chứa y[t]
  - `co2/pm25[t]` → chia cho target
  - `pd.cut(pm25[t])` → binning target
- **Giải pháp**: Sử dụng shifted values: `shift(1).diff()`, domain features dùng `pm25_lag_1h`
- **Phòng tránh**:
  - **Quy tắc vàng**: KHÔNG BAO GIỜ dùng giá trị target tại thời điểm t trong features
  - Luôn chạy `tests/validation/test_leakage.py` sau khi thêm features mới
  - R² > 0.99 = red flag → audit ngay
  - Kiểm tra: `feature + other_feature = target` → leakage
- **Files liên quan**: `src/features/temporal.py`, `src/features/builder.py`
- **Đã cập nhật SKILL.md**: Chưa (cần bổ sung anti-leakage rules)

## [2026-03-29] CSV Loading quá chậm cho large marts data
- **Triệu chứng**: `pd.read_csv()` treo >3 phút khi load 6689 rows × 95 cols
- **Nguyên nhân gốc rễ**: Parse 95 cột float + datetime index, không dùng `usecols`
- **Giải pháp**: Dùng `usecols` chỉ load cột cần thiết, hoặc convert sang parquet
- **Phòng tránh**:
  - Script audit/debug: luôn dùng `usecols` để load subset
  - Production pipeline: cân nhắc parquet format cho tốc độ
  - Luôn thêm `print(..., flush=True)` cho progress indication
- **Files liên quan**: `scripts/leakage_audit.py`

## [2026-03-29] Stationarity — PM2.5 là trend-stationary
- **Triệu chứng**: ADF nói stationary nhưng KPSS nói non-stationary
- **Nguyên nhân gốc rễ**: PM2.5 có trend nhẹ nhưng không có unit root → trend-stationary
- **Giải pháp**: Differencing (d=1 hoặc d=24) loại bỏ trend → fully stationary
- **Phòng tránh**: Luôn chạy CẢ ADF + KPSS, không dựa vào 1 test đơn lẻ
- **Files liên quan**: `scripts/stationarity_check.py`

## [2026-04-04] Leakage Impact — Kết quả thực tế sau fix
- **Triệu chứng**: Sau fix leakage, KHÔNG ML model nào beat Persistence (MASE > 1.0)
- **Nguyên nhân**: PM2.5 autocorrelation lag 1h = 0.89 → Persistence rất khó bị vượt
- **Lesson**: Kết quả "tệ hơn" sau fix KHÔNG phải thất bại — đây là **kết quả trung thực**
- **So sánh impact**: Ridge MAE tăng từ 0.004 → 2.824 (+700x). Leakage che giấu hoàn toàn performance thực.
- **Phòng tránh**: Luôn chạy leakage audit TRƯỚC khi celebrate kết quả tốt
- **Reference**: Kapoor & Narayanan (2023) — M4 Competition cũng cho thấy simple methods often unbeatable
- **Files liên quan**: `scripts/rebuild_and_rerun.py`, `docs/PROJECT_WALKTHROUGH.md`

## [2026-04-04] Unit test cần update khi thay đổi logic
- **Triệu chứng**: `test_diff_value_correct` FAIL sau khi fix diff logic
- **Nguyên nhân**: Test assert kết quả tại iloc[1], nhưng shift(1).diff(1) cho NaN tại iloc[1]
- **Giải pháp**: Update test assert iloc[2] (= y[1] - y[0]) thay vì iloc[1]
- **Lesson**: Khi thay đổi logic code → PHẢI update tất cả test liên quan
- **Files liên quan**: `tests/unit/test_features.py`

## [2026-04-04] Kill processes trước khi test/build
- **Triệu chứng**: Tiến trình Python treo (CSV load timeout) gây conflict khi chạy tiếp
- **Giải pháp**: `pkill -f "python.*scripts/" && pkill -f "pytest"` trước mỗi pipeline run
- **Phòng tránh**: Thêm vào workflow standard

## [2026-04-04] Result Validation Protocol — Validate-Before-Trust
- **Triệu chứng**: Kết quả tốt bất thường (MASE=0.002) → leakage; hoặc kết quả "tệ" (MASE≈1.0 ở 1h) → chưa chắc là lỗi
- **Nguyên tắc**: LUÔN đối chiếu kết quả với literature TRƯỚC khi kết luận
- **Checklist validation**:
  1. MAE PM2.5 phải trong 1.5-15 µg/m³ (hourly) — theo literature review
  2. MASE < 0.1 → 🚨 RED FLAG: gần như chắc chắn leakage
  3. MASE ≈ 1.0 ở 1h → ✅ Expected (autocorr=0.97, Persistence rất mạnh)
  4. MASE < 1.0 ở 6h-24h → ✅ Expected (autocorr giảm, ML tìm được pattern)
  5. Shuffle test: randomize target → retrain → MASE phải >> 1.0
  6. GRU ≤ LSTM → ✅ Consistent with literature (simpler = less overfitting)
- **Reference papers**:
  - Hyndman & Koehler (2006): MASE formula
  - Kapoor & Narayanan (2023): Shuffle test, leakage detection
  - WHO PM2.5 Guidelines: plausible value range
- **Test suite**: `tests/validation/test_result_validation.py` — 15+ tests
- **Phòng tránh**: Chạy validation tests SAU MỖI experiment run

## [2026-04-04] PyTorch MPS + LightGBM = Silent Segfault
- **Triệu chứng**: Script chết tĩnh (không exception, không traceback) khi LightGBM fit() sau torch MPS init
- **Nguyên nhân**: `import torch` + `torch.device("mps")` ở top-level → khởi tạo MPS Metal backend → xung đột OpenMP/memory với LightGBM C++ backend
- **Giải pháp**: **Lazy import torch** — chỉ import trong hàm GRU, SAU KHI LightGBM đã train xong
- **Debug approach**: Viết script tách riêng từng step, chạy isolate → xác định bước crash
- **Phòng tránh**:
  - KHÔNG import torch ở top-level trong script đa mô hình
  - Dùng factory pattern: `_make_gru_model()`, `_make_dataset_class()` → return class
  - Kiểm tra import order khi kết hợp PyTorch + tree-based libraries
- **Files liên quan**: `scripts/ensemble_multi_horizon.py`

