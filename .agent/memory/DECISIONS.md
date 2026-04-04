# TECHNICAL DECISIONS LOG

> Ghi lại các quyết định kỹ thuật quan trọng để team và AI assistant tham chiếu.

---

## [2026-03-28] Chọn kiến trúc thư mục dự án
- **Bối cảnh**: Cần thiết lập cấu trúc chuẩn cho dự án Time Series Forecasting PM2.5
- **Các phương án**: (1) Flat structure, (2) Cookiecutter Data Science, (3) Custom structure kết hợp
- **Quyết định**: Custom structure dựa trên Cookiecutter Data Science, bổ sung `memory/` cho context engineering
- **Lý do**: Cookiecutter là chuẩn công nghiệp, `memory/` giúp AI và team duy trì context xuyên suốt dự án
- **Hệ quả**: Mọi code mới phải tuân theo cấu trúc trong SKILL.md Section 2

## [2026-03-29] Tách guides thành file riêng, bundle skills vào SKILL.md
- **Bối cảnh**: Cần bổ sung nhiều chức năng từ 9 skills
- **Quyết định**: Tách thành 7 guide files, SKILL.md chứa summary (bảng tóm tắt + link)
- **Lý do**: SKILL.md đã 777+ dòng, tách giúp navigate + cập nhật độc lập
- **Data Quality**: Great Expectations cho standard checks, custom validators cho domain-specific (leakage, temporal)
- **Hệ quả**: Agent ĐỌC guide chi tiết khi cần, không cần memorize toàn bộ

## [2026-03-29 PM] Chọn loguru cho logging
- **Quyết định**: `loguru` — zero config, color output, structured JSON trivial
- **Lý do**: ML research — DX > enterprise compatibility
- **Reconsider khi**: Scale lên production serving cần OTEL → migrate sang `structlog`

## [2026-03-29 PM] Visualization — Data Storytelling approach
- **Quyết định**: 12 chart types chia 3 phases (Understanding, Patterns, Model Storytelling)
- **Key decisions**: Violin > Histogram (PM2.5 skewed), Heatmap (hour×month), Polar chart cho error-by-hour
- **Rule**: Mỗi chart = 1 câu hỏi, WHO reference line bắt buộc

## [2026-03-29 PM] SKILL.md — Hybrid approach (inline rules + separate guides)
- **Quyết định**: Critical rules inline (~30 dòng), detailed templates tách 4 guides mới
- **New rules**: Scaling table, Walk-Forward, DL 8 rules, Multi-horizon eval, DM test, CI, MAPE warning

## [2026-03-29 PM] Feature Engineering — 95 features, Anti-leakage design
- **Bối cảnh**: Cần tạo features cho PM2.5 forecasting từ 5 sensor columns
- **Quyết định**: 95 features = 40 lag + 24 rolling + 6 EWM + 13 calendar + 4 diff + 3 domain + 5 raw
- **Anti-leakage**: Lag sử dụng `shift()`, Rolling sử dụng `shift(1).rolling()`, EWM sử dụng `shift(1).ewm()`
- **⚠️ Phát hiện sau**: diff và domain features CÓ leakage (xem entry 2026-03-29 PM Leakage Audit)

## [2026-03-29 PM] Data Leakage Audit — Phát hiện leakage trong pipeline
- **Bối cảnh**: Ridge model đạt MAE=0.004, R²=1.0 — bất thường nghiêm trọng
- **Phân tích**: 4 nguồn leakage tiềm năng:
  1. `pm25_diff_1h = pm25[t] - pm25[t-1]` → chứa target y[t]
  2. `pm25_pct_change_1h = (pm25[t] - pm25[t-1]) / pm25[t-1]` → chứa y[t]
  3. `co2_pm25_ratio = co2 / pm25[t]` → chia cho target
  4. `pm25_aqi_cat = pd.cut(pm25[t])` → binning target
- **Quyết định**: Cần fix tất cả → dùng shifted values thay vì current target
- **Fix plan**: diff → `shift(1).diff()`, domain → dùng `pm25_lag_1h` thay pm25 raw
- **Status**: ✅ HOÀN THÀNH (2026-04-04). Kết quả: MASE > 1.0 cho tất cả ML — Persistence vẫn best.

## [2026-03-29 PM] Stationarity Analysis
- **Bối cảnh**: Cần xác định tính dừng của PM2.5 trước khi chọn model
- **Kết quả**: PM2.5 raw = trend-stationary (ADF ✅, KPSS ❌). Differencing (d=1 hoặc d=24) → fully stationary
- **Quyết định**: ML models OK dùng raw features. ARIMA cần d=1 hoặc seasonal differencing d=24

## [2026-04-04] Đồng bộ kiến thức Global Playbook ↔ Project Memory
- **Quyết định**: Cập nhật 2 chiều giữa Global AGENTS.md và Project Memory

## [2026-04-04] Memory Optimization Strategy
- **Bối cảnh**: Tổng memory = 135 KB (~34K tokens). SKILL.md chiếm 38.6 KB = 28% tổng.
- **Quyết định**: Áp dụng Tiered Memory (HOT/WARM/COLD) + định kỳ compact
- **Quy tắc**:
  - HOT (đọc mỗi phiên): CONTEXT.md + TODO.md ≤ 100 dòng mỗi file
  - WARM (đọc khi cần): DECISIONS.md, LESSONS_LEARNED.md, RUNS_LOG.md
  - COLD (đọc 1 lần): SKILL.md, guides/* — chỉ đọc section liên quan
  - **Compact trigger**: LESSONS_LEARNED > 80 dòng → archive entries > 30 ngày
  - **TODO cleanup**: Completed items → 1-line archive + link đến walkthrough

## [2026-04-04 PM] EDA Strategy Comparison — Missing Data Handling
- **Bối cảnh**: Dataset mất 74% hourly data. Gap analysis cho thấy 85% missing nằm trong gaps >1 tuần (unrecoverable). Cần chiến lược tối ưu.
- **Thí nghiệm**: So sánh 4 strategies (LightGBM, test = real data only):
  - A. Segment-Only (max_gap=2h): 7,335 rows → MASE=1.113
  - B. Extended Interp (CubicSpline, max=12h): 7,386 rows → MASE=1.321 ⚠️ **TỆ NHẤT**
  - C. ML Impute (KNN k=5, max=24h): 7,742 rows → MASE=1.084
  - D. **Hybrid** (Spline≤6h + KNN 6-24h): 7,742 rows → MASE=**1.066** ⭐
- **Quyết định**: Dùng **Hybrid** làm strategy chính
- **Lý do**: Cubic spline alone tạo noise (MASE worst). KNN multivariate context tốt hơn vì dùng nhiet_do/do_am/co2 đã correlate.
- **Nguyên tắc quan trọng**: **Test set = REAL data ONLY** — không bao giờ dùng data imputed để đánh giá
- **Hạn chế**: Tất cả strategies vẫn MASE > 1 ở 1h horizon → PM2.5 autocorrelation lag_1h = 0.97
- **Hướng đi tiếp**: Multi-horizon eval (6h, 24h) — Persistence sẽ yếu ở horizon dài, ML nên thắng

## [2026-04-04 PM] IEEE Citation Style
- **Quyết định**: Dùng chuẩn IEEE cho trích dẫn (theo QĐ 1799 CTU)
- **Lý do**: Dự án thuộc lĩnh vực kỹ thuật, IEEE phổ biến hơn APA cho CS/EE

## [2026-04-04 19:37] Multi-Horizon Evaluation — ML THẮNG!
- **Kết quả**: LightGBM Optuna beats Persistence ở 6h (MASE=0.730, -27%) và 24h (MASE=0.812, -19%)
- **1h**: MASE=1.012 — gap chỉ 1.2%, Persistence vẫn gần bằng (autocorr=0.97)
- **Feature shift theo horizon**: 1h→lag dominant, 6h→temporal patterns, 24h→multivariate features
- **Kết luận**: Giả thuyết ĐÚNG — ML tạo giá trị ở horizons dài khi autocorrelation giảm

## [2026-04-04 19:48] BUG: statsmodels + DatetimeIndex sau drop gaps
- **Vấn đề**: Khi drop NaN rows → DatetimeIndex mất frequency info → statsmodels ARIMA/SARIMAX đổ warning
- **Fix**: Dùng `.values` (numpy array) khi truyền vào model
- **Trap thêm**: Khi dùng numpy input, `forecast()` trả numpy array → dùng `[-1]`, KHÔNG `iloc[-1]`
- **Bài học**: Đã cập nhật vào Global Playbook Known Traps
