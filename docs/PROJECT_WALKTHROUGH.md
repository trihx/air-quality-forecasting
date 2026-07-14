# 📖 PROJECT WALKTHROUGH — Dự Báo PM2.5 Bằng Machine Learning

> **Mục đích**: Tài liệu chi tiết toàn bộ quá trình thực hiện dự án, từ thu thập dữ liệu đến đánh giá mô hình.
> **Thiết kế cho luận văn**: Tài liệu này đóng vai trò cơ sở kỹ thuật chi tiết. Bản dựng luận văn chính thức chuẩn ĐH Cần Thơ (QĐ 1799) đã được đồng bộ sang file [THESIS_DRAFT_CTU_1799.md](THESIS_DRAFT_CTU_1799.md).
> **Triết lý**: Quá trình quan trọng hơn kết quả. Cách làm tốt hơn giúp con đường đi tốt hơn.

---

## Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Thu Thập & Khám Phá Dữ Liệu](#2-thu-thập--khám-phá-dữ-liệu)
3. [Data Cleaning Pipeline](#3-data-cleaning-pipeline)
4. [Phân Tích Khám Phá (EDA)](#4-phân-tích-khám-phá-eda)
5. [Feature Engineering](#5-feature-engineering)
6. [Data Leakage — Phát Hiện & Sửa Lỗi](#6-data-leakage--phát-hiện--sửa-lỗi)
7. [Model Training — Baseline & ML](#7-model-training--baseline--ml)
8. [Deep Learning — LSTM & GRU](#8-deep-learning--lstm--gru)
9. [Temporal Fusion Transformer (TFT)](#9-temporal-fusion-transformer-tft)
10. [Ensemble Methods](#10-ensemble-methods)
11. [Prediction Intervals — Khoảng Tin Cậy Dự Báo](#11-prediction-intervals--khoảng-tin-cậy-dự-báo)
12. [Evaluation Methodology](#12-evaluation-methodology)
13. [Kết Quả So Sánh Trước/Sau Fix Leakage](#13-kết-quả-so-sánh-trướcsau-fix-leakage)
14. [Bài Học Kinh Nghiệm](#14-bài-học-kinh-nghiệm)
15. [Kết Luận & Hướng Phát Triển](#15-kết-luận--hướng-phát-triển)
16. [Tài Liệu Tham Khảo](#16-tài-liệu-tham-khảo)

---

## 1. Tổng Quan Dự Án

### 1.1 Bối Cảnh & Động Lực

Ô nhiễm không khí là vấn đề nghiêm trọng, đặc biệt tại các đô thị Việt Nam. **PM2.5** (bụi mịn có đường kính ≤ 2.5µm) là chỉ số quan trọng nhất do khả năng xâm nhập sâu vào phổi. Dự báo PM2.5 chính xác giúp:
- Cảnh báo sớm cho cộng đồng
- Hỗ trợ ra quyết định chính sách
- Bảo vệ sức khỏe người dân

### 1.2 Mục Tiêu

1. Xây dựng pipeline end-to-end: từ dữ liệu IoT sensor → dự báo PM2.5
2. So sánh hiệu quả các phương pháp: Baseline → Statistical → ML → DL
3. Đảm bảo tính đúng đắn khoa học: anti-leakage, proper validation
4. Đánh giá dựa trên metrics chuẩn quốc tế (MAE, MASE, RMSE)

### 1.3 Công Nghệ Sử Dụng

| Thành phần | Công cụ |
|-----------|---------|
| Ngôn ngữ | Python 3.11 |
| Package Manager | `uv` (thay thế pip, nhanh hơn 10x) |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Time Series | statsmodels (ADF/KPSS/ARIMA) |
| Visualization | matplotlib, seaborn |
| Testing | pytest + pytest-cov |
| Code Quality | ruff (lint), bandit (security), mypy (types) |
| Logging | loguru (structured logging) |

### 1.4 Kiến Trúc Pipeline

```
IoT Sensor (~2 phút/mẫu, 209K records, 3.1 năm)
    │
    ▼
[1] Raw Data (dataset/raw/)
    │ Load + Validate schema
    ▼
[2] Staging — Validate format, bounds
    │ 7-step cleaning pipeline
    ▼
[3] Intermediate (dataset/interim/) — 6,857 rows hourly
    │ Feature engineering (lag, rolling, calendar, domain)
    ▼
[4] Marts (dataset/processed/) — 6,689 rows × 95 cols
    │ Temporal split 80/10/10
    ▼
[5] Train/Val/Test → Model Training → Evaluation
    │
    ▼
[6] Results → RUNS_LOG.md → Visualization → Report
```

---

## 2. Thu Thập & Khám Phá Dữ Liệu

### 2.1 Nguồn Dữ Liệu

Dữ liệu từ **IoT sensor** đặt tại vị trí cố định, thu thập liên tục từ 2022-03-16 đến 2025-05-11 (~3.1 năm).

| Thuộc tính | Giá trị |
|-----------|---------|
| Tổng records | 209,594 |
| Tần suất | ~2 phút/mẫu |
| Cột dữ liệu | 6 (5 features + 1 timestamp) |
| Dung lượng | ~10 MB |

### 2.2 Các Biến Đo Lường

| Biến | Ý Nghĩa | Đơn Vị | Range | Median | Vai Trò |
|------|---------|--------|-------|--------|---------|
| `nhiet_do` | Nhiệt độ | °C | 22–38 | 28.3 | Feature |
| `do_am` | Độ ẩm | % | 36–98 | 78.1 | Feature |
| `diem_suong` | Điểm sương | °C | 22–29 | 26.0 | Feature |
| `co2` | Nồng độ CO₂ | ppm | 74–1385 | 405 | Feature |
| `pm25` | Bụi mịn PM2.5 | µg/m³ | 1.1–54 | 10.3 | **Target** |
| `ngay_tao` | Thời gian | datetime | 3.1 năm | — | Index |

### 2.3 Nhận Xét Ban Đầu

- **PM2.5 skewed** (skewness = 1.45): Phần lớn giá trị thấp, ít giá trị cao → ảnh hưởng đến MAPE
- **Tất cả biến NON-normal** (Shapiro p < 0.001): Cần lưu ý khi chọn model/metrics
- **Correlation với PM2.5 yếu**: `nhiet_do` (r = -0.21), `co2` (r = 0.12) → cần lag/temporal features

### 2.4 Cách Thực Hiện

```python
# Load data với validation 3 lớp
from src.data.loader import load_raw_data
from src.data.validator import DataValidator

df = load_raw_data("dataset/raw/final_dataset.csv")
validator = DataValidator()
validator.validate_staging(df)
# → 8/8 checks passed ✅
```

**Quyết định thiết kế**: Dùng 3-layer validation (Staging → Intermediate → Marts), lấy cảm hứng từ dbt data pipeline methodology. Mỗi layer có critical checks riêng.

---

## 3. Data Cleaning Pipeline

### 3.1 Tổng Quan 7-Step Pipeline

Thứ tự xử lý **QUAN TRỌNG** — thay đổi thứ tự sẽ cho kết quả khác:

| Bước | Thao Tác | Input | Output | Lý Do Thứ Tự |
|------|----------|-------|--------|---------------|
| 1/7 | Remove duplicates | 209,591 rows | Không có duplicate | Loại data lặp trước |
| 2/7 | Set datetime index | DataFrame | DatetimeIndex | Cần index thời gian cho bước sau |
| 3/7 | Clip physical bounds | Values ngoài range | Clipped | Loại giá trị bất hợp lý trước IQR |
| 4/7 | Handle outliers (IQR 3.0) | Normal data | 23,336 outliers → NaN | Dùng NaN, không xóa hàng |
| 5/7 | Resample 1h | 209,591 rows | 27,649 rows | Đưa về tần suất đều |
| 6/7 | Interpolate (linear, max 2h) | NaN gaps | 1,193 values filled | Chỉ fill gap ≤ 2h |
| 7/7 | Drop remaining NaN | Rows có NaN | 6,857 rows | Gap > 2h không tin cậy |

### 3.2 Quyết Định Kỹ Thuật Quan Trọng

#### a) IQR Threshold = 3.0 (thay vì 1.5)

**Vấn đề**: IQR 1.5 (mặc định) loại quá nhiều data PM2.5, vì PM2.5 tự nhiên có biến động lớn (spike khi ô nhiễm).

**Phân tích**:
- IQR 1.5: Loại ~30% data → mất quá nhiều thông tin
- IQR 3.0: Loại ~11% → cân bằng giữa data quality và data quantity
- Domain knowledge: PM2.5 spike (ô nhiễm) là real signal, không phải noise

**Quyết định**: IQR 3.0 → chỉ loại extreme outliers, giữ spike tự nhiên.

#### b) Max Interpolation Gap = 2h

**Vấn đề**: Gap lớn (> 2h) → interpolation không đáng tin cậy cho time series.

**Cách tiếp cận**:
- Gap < 30 phút: Linear interpolation (biến đổi tuyến tính)
- Gap 30 phút – 2h: Linear (đủ tốt cho hourly data)
- Gap > 2h: **KHÔNG interpolate** → drop row

#### c) Outlier Strategy: Replace → NaN → Interpolate (không xóa)

**Lý do**: Xóa outlier tạo gap trong time series → mất temporal continuity. Thay outlier bằng NaN → interpolate → giữ liên tục thời gian.

### 3.3 Kết Quả Cleaning

```
Input:  209,591 rows (2022-03-16 → 2025-05-11)
Output:   6,857 rows (3.3% retained, hourly frequency)

Report:
  - Duplicates removed: 0
  - Values clipped: 0
  - Outliers → NaN: 23,336
  - Resampled: 209,591 → 27,649 rows
  - Interpolated: 1,193 values
  - Dropped NaN: 20,792 rows
```

**Phân tích tỷ lệ 3.3%**: Tỷ lệ thấp vì (1) resample từ 2-phút → 1h giảm 30x, (2) nhiều gap > 2h trong sensor data. Đây là đặc điểm của dữ liệu IoT thực tế — sensor offline, bảo trì, mất điện.

### 3.4 Validation Intermediate Layer

```
--- INTERMEDIATE Validation: 4 passed, 1 failed ---
  ✅ [CRITICAL] no_nan: Data complete
  ✅ [CRITICAL] monotonic_index: Time monotonically increasing
  ❌ [WARNING] regular_frequency: Irregular — có gaps
  ✅ [WARNING] sufficient_data: 6,857 rows (min: 1,000)
  ✅ [INFO] stats_sanity: Reasonable variance
```

**Ghi chú**: Warning `irregular_frequency` chấp nhận được — do drop gap > 2h tạo segments không liên tục. Không ảnh hưởng feature engineering (lag/rolling tự xử lý NaN).

---

## 4. Phân Tích Khám Phá (EDA)

### 4.1 EDA Pipeline (8 Section)

| Section | Phân tích | Phát hiện chính |
|---------|-----------|-----------------|
| 1/8 | Descriptive Statistics | PM2.5 mean=13.2, skewed (1.45) |
| 2/8 | Time Series Plots | Seasonal Pattern rõ: PM2.5 cao ban đêm/sáng sớm |
| 3/8 | Distribution Analysis | Tất cả biến non-normal (Shapiro p < 0.001) |
| 4/8 | Correlation Analysis | Target correlations yếu (max |r| = 0.21) |
| 5/8 | Stationarity Tests | PM2.5 = trend-stationary (ADF ✅, KPSS ❌) |
| 6/8 | ACF/PACF Analysis | Strong autocorrelation → lag features quan trọng |
| 7/8 | Temporal Patterns | Peak: 6h sáng, Trough: 12h trưa, Peak month: tháng 4 |
| 8/8 | Missing Values | 0% missing sau cleaning |

### 4.2 Khám Phá Qua Lăng Kính "Data Storytelling" (New)

Thay vì chỉ báo cáo thống kê thuần túy, EDA được tái cấu trúc thành 4 câu chuyện dữ liệu nhằm làm nổi bật **độ khó đặc thù** của bài toán dự báo PM2.5 (các kịch bản được lưu tại `scripts/eda/` và biểu đồ tại `research/eda/visualizations/`).

#### 1. Bẫy Tự Tương Quan (The Autocorrelation Trap)
- **Insight**: Khả năng "nhớ" của PM2.5 giảm cực nhanh. Ở trễ 1h (lag 1h), tự tương quan r=0.97 (gần như tuyến tính hoàn hảo). Nhưng đến 6h hay 24h, biểu diễn Hexbin Scatter Dispersion phân tán thành một đám mây rời rạc.
- **Ý nghĩa**: Model ML trông có vẻ "rất thiên tài" ở h=1 nhưng bế tắc ở h=24. Đây là nguyên nhân vì sao Persistence Baseline (copy $y_{t-1}$) vô đối ở h=1.

#### 2. Đỉnh Dị Thường & Phân Phối Đuôi Dài (Erratic Spikes)
- **Insight**: Biểu đồ phân phối cho thấy PM2.5 không phải normal distribution mà là Fat-Tailed (Đuôi dài bất đối xứng cực đoan). Dải an toàn WHO (dưới 15 hoặc 35 $\mu g/m^3$) thường bị phá vỡ bởi các "đỉnh ô nhiễm" tăng vọt lên 100 $\mu g/m^3$ chỉ trong 1-2h (do kẹt xe cục bộ, đốt rác).
- **Ý nghĩa**: Các hàm Loss như MSE/MAE của ML mặc định thích hội tụ về giá trị trung bình (mean) → Mô hình luôn bị under-estimate các đỉnh rủi ro y tế này (coi chúng như nhiễu/outliers).

#### 3. Sự Xê Dịch Quy Luật Đa Biến (Concept Drift in Multivariates)
- **Insight**: Vẽ tương quan chạy (14-day Rolling Spearman Correlation) giữa PM2.5 vs Nhiệt Độ/Độ Ẩm cho thấy sự dao động từ âm sâu (-0.6) sang dương đậm (+0.6).
- **Ý nghĩa**: Rule "trời nắng nóng thì bụi nhiều" không tĩnh. Nó thay đổi theo mùa, gió, và cấu trúc nghịch nhiệt. Do đó, các mô hình hồi quy truyền thống (Linear) sẽ thất bại vì relationship bị drift liên tục.

#### 4. Khoảng Trống Chất Lượng Cảm Biến IoT (Data Quality Gaps)
- **Insight**: Biểu đồ Missing Data Barcode cho thấy cảm biến rớt mạng theo từng chùm dài hàng ngày/tuần chứ không rớt ngẫu nhiên 1-2 giờ.
- **Ý nghĩa**: Vùng gap ≤ 24h (chiếm khoảng ~10-15%) có thể cứu bằng Interpolation/KNN. Phần còn lại bắt buộc drop để giữ Anti-Leakage và Data Integrity (Test-on-Real-Only). Nhấn mạnh tầm quan trọng của Data Engineering trong hệ thống thực tế.

---

## 5. Feature Engineering

### 5.1 Thiết Kế Feature Groups

Tổng cộng **95 features** được tạo theo 6 nhóm:

| Nhóm | Số Features | Công Thức | Lý Do |
|------|-------------|-----------|-------|
| **Lag** | 40 | `shift(k)` cho 5 cols × 8 lags | Capture autocorrelation |
| **Rolling** | 24 | `shift(1).rolling(w).agg(f)` | Smooth trends, detect volatility |
| **EWM** | 6 | `shift(1).ewm(span).agg(f)` | Recent values weight more |
| **Calendar** | 13 | hour, month, sin/cos encoding | Seasonal patterns |
| **Diff** | 4 | `shift(1).diff()`, `shift(1).pct_change()` | Rate of change |
| **Domain** | 3 | CO2/PM2.5 ratio, AQI category, T×RH | Domain knowledge |
| **Original** | 5 | raw sensor values | Direct features |

### 5.2 Anti-Leakage Design

**Nguyên tắc cốt lõi**: Feature tại thời điểm t **KHÔNG ĐƯỢC** chứa bất kỳ thông tin nào của target y[t].

| Feature Type | Cách Tính Đúng | Giải Thích |
|-------------|----------------|------------|
| Lag | `y.shift(k)` | Dùng giá trị quá khứ y[t-k] |
| Rolling | `y.shift(1).rolling(w)` | shift(1) trước → window dùng [t-w-1, t-1] |
| EWM | `y.shift(1).ewm(span)` | Tương tự rolling |
| Diff | `y.shift(1).diff(1)` | = y[t-1] - y[t-2], chỉ past values |
| Domain | Dùng `pm25_lag_1h` thay `pm25[t]` | Ratio và binning dùng giá trị quá khứ |

### 5.3 Quyết Định Lag Selection

Lags được chọn dựa trên **ACF/PACF analysis**:
```
LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 168]
```
- 1, 2, 3: Short-term dependencies (autocorrelation cao)
- 6, 12: Intra-day patterns (nửa ngày)
- 24: Daily cycle (mạnh thứ 2 sau lag 1)
- 48: 2-day pattern
- 168: Weekly pattern (7 × 24)

### 5.4 Cyclical Encoding Cho Calendar Features

**Vấn đề**: `hour = 0` và `hour = 23` gần nhau về mặt thời gian, nhưng xa nhau trong encoding linear.

**Giải pháp**: Sin/Cos encoding:
```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
```
→ hour 0 và hour 23 gần nhau trong feature space ✅

### 5.5 Output Marts Data

```
Marts Data: 6,689 rows × 95 columns
  - Dropped 168 warmup rows (lag 168h cần 168 rows đầu)
  - Validated: No NaN, monotonic index, reasonable variance
```

---

## 6. Data Leakage — Phát Hiện & Sửa Lỗi

> **Đây là phần quan trọng nhất** — minh họa quá trình phát hiện, phân tích, và sửa lỗi data leakage. Bài học này áp dụng cho BẤT KỲ dự án ML nào.

### 6.1 Triệu Chứng — "Quá Tốt Để Là Thật"

Khi chạy ML models lần đầu (2026-03-29), kết quả:

| Model | MAE | RMSE | R² | MASE |
|-------|-----|------|----|------|
| **Ridge** | **0.004** | 0.006 | **1.000** | 0.002 |
| Lasso | 0.106 | 0.132 | 0.999 | 0.058 |
| ElasticNet | 0.371 | 0.535 | 0.989 | 0.204 |
| RandomForest | 0.143 | 0.383 | 0.995 | 0.078 |
| XGBoost | 0.189 | 0.395 | 0.994 | 0.104 |
| LightGBM | 0.221 | 0.384 | 0.995 | 0.121 |

**Red flags**:
- Ridge MAE = 0.004 µg/m³ → sai số trung bình chỉ 0.004 → **gần như perfect**
- R² = 1.000 → model giải thích 100% variance → **bất thường nghiêm trọng**
- Persistence baseline MAE = 1.821 → Ridge tốt hơn **450 lần** → **không thực tế**

**Câu hỏi**: Giá trị PM2.5 dao động 1–54 µg/m³. Không model nào đạt MAE = 0.004 trong thực tế. Tại sao?

### 6.2 Phân Tích — Tìm Nguyên Nhân Gốc Rễ

#### Bước 1: Kiểm tra correlation matrix

Chạy correlation giữa tất cả features và target PM2.5:
```python
corrs = df[feature_cols].corrwith(df["pm25"]).abs().sort_values(ascending=False)
```

**Phát hiện**: Nhiều features có |correlation| > 0.95 với target → red flag.

#### Bước 2: Phân tích từng feature nghi ngờ

**Feature 1**: `pm25_diff_1h`

```
Công thức ban đầu: diff_1h = pm25[t] - pm25[t-1]
Cho biết: pm25[t] = diff_1h + pm25[t-1] = diff_1h + pm25_lag_1h
→ Model có thể tái tạo CHÍNH XÁC target từ 2 features!
→ LEAKAGE: diff_1h chứa pm25[t]
```

**Kiểm chứng bằng code**:
```python
recon = df["pm25_diff_1h"] + df["pm25_lag_1h"]  # = pm25[t]
np.allclose(recon, df["pm25"])  # → True → LEAKAGE CONFIRMED
```

**Feature 2**: `pm25_pct_change_1h`

```
Công thức: pct_change = (pm25[t] - pm25[t-1]) / pm25[t-1]
→ pm25[t] = pm25[t-1] * (1 + pct_change) = lag_1h * (1 + pct_change)
→ LEAKAGE: chứa pm25[t]
```

**Feature 3**: `co2_pm25_ratio`

```
Công thức ban đầu: ratio = co2 / pm25[t]
→ pm25[t] = co2 / ratio
→ LEAKAGE: sử dụng pm25[t] (target) trong tính toán
```

**Feature 4**: `pm25_aqi_cat`

```
Công thức ban đầu: pd.cut(pm25[t], bins=[0, 12, 35.4, ...])
→ Binning trực tiếp từ target → category encode target
→ LEAKAGE: mã hóa thông tin target
```

### 6.3 Tại Sao Leakage Xảy Ra

**Nguyên nhân gốc**: Thiếu phân biệt giữa `pm25[t]` (target) và `pm25[t-1]` (past value).

Trong time series, `diff()` của pandas tính:
```python
df["pm25"].diff(1)  # = pm25[t] - pm25[t-1] → CHỨA pm25[t]!
```

Đây KHÔNG phải lỗi pandas. Đây là **hiểu sai về temporal dependency**:
- ở thời điểm t, ta cần dự đoán pm25[t]
- Feature tại t chỉ được dùng thông tin ĐÃ BIẾT (tức ≤ t-1)
- `diff(1)` dùng pm25[t] → vi phạm nguyên tắc

### 6.4 Cách Sửa — Anti-Leakage Pattern

**Nguyên tắc**: Luôn `shift(1)` TRƯỚC khi tính toán bất kỳ gì liên quan đến target.

#### Fix diff/pct_change (temporal.py):

```python
# ❌ SAI (leakage):
df["pm25_diff_1h"] = df["pm25"].diff(1)           # = y[t] - y[t-1]

# ✅ ĐÚNG (anti-leakage):
shifted = df["pm25"].shift(1)
df["pm25_diff_1h"] = shifted.diff(1)               # = y[t-1] - y[t-2]
df["pm25_pct_change_1h"] = shifted.pct_change(1)   # = (y[t-1]-y[t-2])/y[t-2]
```

#### Fix domain features (builder.py):

```python
# ❌ SAI:
df["co2_pm25_ratio"] = df["co2"] / df["pm25"]      # dùng target
df["pm25_aqi_cat"] = pd.cut(df["pm25"], ...)        # binning target

# ✅ ĐÚNG:
df["co2_pm25_ratio"] = df["co2"] / df["pm25_lag_1h"]   # dùng past value
df["pm25_aqi_cat"] = pd.cut(df["pm25_lag_1h"], ...)     # binning past value
```

### 6.5 Kiểm Chứng Fix — Leakage Audit

Script `leakage_audit.py` kiểm tra 6 tiêu chí:

```
[1/6] diff_1h: pm25 = diff_1h + lag_1h? → 🟢 OK (không tái tạo được target)
[2/6] pct_change_1h: pm25 = lag_1h * (1+pct)? → 🟢 OK
[3/6] co2_pm25_ratio: dùng pm25[t]? → 🟢 OK (dùng lag)
[4/6] pm25_aqi_cat: correlation cao? → 🟢 OK (< 0.95)
[5/6] Key correlations: tất cả < 0.95 → 🟢 OK
[6/6] Autocorrelation: hợp lý → 🟢 OK
```

### 6.6 Leakage Test Suite

9 unit tests tự động phát hiện leakage:

| Test | Kiểm tra | Pass Criteria |
|------|---------|---------------|
| `test_no_perfect_correlation` | |corr| < 0.99 | Không feature nào có corr 1:1 |
| `test_no_feature_equals_target` | Feature ≠ target | Không có copy trực tiếp |
| `test_diff_features_use_shifted` | Diff corr < 0.95 | Diff không encode target |
| `test_domain_features_no_current` | Domain corr < 0.95 | Domain dùng past value |
| `test_lag_features_are_lagged` | lag_1h = shift(1) | Lag đúng |
| `test_rolling_features_use_past` | Rolling corr < 0.99 | Rolling không dùng y[t] |
| `test_ridge_fails_shuffled` | Shuffle → R² ≈ 0 | Features không encode target |
| `test_no_raw_target_in_features` | pm25 ∉ features | Target không trong X |
| `test_suspicious_patterns` | Flag tên nghi ngờ | Warning cho review |

### 6.7 Bài Học Data Leakage

> **Quy tắc vàng (Golden Rule)**: Trong time series, KHÔNG BAO GIỜ dùng giá trị target tại thời điểm t trong features. Luôn dùng `shift(1)` trước.

**Cách phát hiện sớm**:
1. R² > 0.99 → **red flag** → audit ngay
2. Kiểm tra: có 2 features nào cộng/nhân/chia cho ra target không
3. Luôn chạy **shuffle test**: shuffle target → R² phải ≈ 0
4. Correlation matrix: |r| > 0.95 với target → kiểm tra kỹ

**Tham khảo**: Kapoor & Narayanan (2023) — _"Leakage and the Reproducibility Crisis in ML-Based Science"_ — survey 17 lĩnh vực, phát hiện leakage phổ biến, ảnh hưởng hàng trăm papers.

---

## 7. Model Training — Baseline & ML

### 7.1 Model Progression Strategy

```
Level 0: Naive Baselines     ← BẮT BUỘC làm trước
Level 1: Statistical         ← ARIMA, SARIMA (chờ implement)
Level 2: Linear ML           ← Ridge, Lasso, ElasticNet
Level 3: Tree-based ML       ← RandomForest, XGBoost, LightGBM
Level 4: Deep Learning       ← LSTM, GRU (chờ implement)
Level 5: Ensemble            ← Stacking, Blending (chờ implement)
Level 6: AutoML              ← Optuna optimization (chờ implement)
```

**Triết lý**: Luôn bắt đầu từ model đơn giản nhất. Model phức tạp phải CHỨNG MINH tốt hơn baseline. Nếu không → complexity không justified.

### 7.2 Level 0: Naive Baselines

| Model | Cách Tính | MAE | RMSE | MASE |
|-------|-----------|-----|------|------|
| **Persistence** | ŷ(t) = y(t-1) | **1.821** | 3.306 | **1.000** |
| SeasonalNaive | ŷ(t) = y(t-24) | 4.197 | 6.418 | 2.305 |
| HourlyMean | ŷ(t) = mean(y|hour) | 6.628 | 7.693 | 3.640 |
| HistoricalMean | ŷ(t) = mean(y_train) | 7.715 | 8.502 | 4.237 |

**Phân tích**:
- **Persistence** (dùng giá trị 1h trước) tốt nhất trong baselines
- PM2.5 có autocorrelation lag 1h rất cao (r=0.97) → persistence khó bị đánh bại
- **Đây là target**: Model ML phải đạt MAE < 1.821 (MASE < 1.0) mới có giá trị

### 7.3 Walk-Forward Cross-Validation

**KHÔNG dùng K-Fold** cho time series (sẽ shuffle temporal order → leakage).

**Phương pháp: Expanding Window (TimeSeriesSplit, 5 folds)**:

```
Fold 1: [Train: 1,005 rows] → [Val: 1,003 rows]
Fold 2: [Train: 2,008 rows] → [Val: 1,003 rows]
Fold 3: [Train: 3,011 rows] → [Val: 1,003 rows]
Fold 4: [Train: 4,014 rows] → [Val: 1,003 rows]
Fold 5: [Train: 5,017 rows] → [Val: 1,003 rows]
```

**Final evaluation**: Train trên 80% đầu, test trên 10% cuối.

### 7.4 Temporal Split

```
Train: 5,351 rows [2022-03-24 → 2023-06-10]
Val:     669 rows [2023-06-11 → 2023-08-18]
Test:    669 rows [2023-08-18 → 2025-03-15]
```

**Quan trọng**: Val và Test LUÔN nằm SAU Train trong thời gian. Không shuffle.

### 7.5 Scaling Strategy

| Model | Cần Scaling? | Lý Do |
|-------|-------------|-------|
| Ridge/Lasso/ElasticNet | ✅ Yes (StandardScaler) | Regularization penalizes large coefficients |
| RandomForest | ❌ No | Tree-based, invariant to scale |
| XGBoost/LightGBM | ❌ No | Tree-based |

**Anti-leakage**: Scaler nằm TRONG Pipeline, fit chỉ trên train data.

---

## 8. Deep Learning — LSTM & GRU

### 8.1 Động Lực

Các mô hình ML (LightGBM) đã chứng minh hiệu quả ở horizons trung-dài (6h, 24h). Tuy nhiên, ML feature-based yêu cầu thiết kế features thủ công và không khai thác trực tiếp sequential patterns trong chuỗi thời gian. Deep Learning (LSTM/GRU) được triển khai để đánh giá khả năng tự động học temporal dependencies từ raw sequences.

### 8.2 Kiến Trúc

**GRU (Gated Recurrent Unit)**:
- Input: Sequence `[t-72, ..., t-1]` × 5 features (pm25, nhiet_do, do_am, diem_suong, co2)
- GRU: 2 layers, hidden_dim=64, dropout=0.2
- FC Head: Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1)
- Parameters: **40,705**

**LSTM (Long Short-Term Memory)**:
- Kiến trúc tương tự GRU nhưng dùng LSTM cells
- Parameters: **53,569** (nhiều hơn GRU ~32% do forget gate riêng)

### 8.3 Hyperparameters

| Parameter | Giá trị | Lý do |
|-----------|---------|-------|
| Lookback window | 72h (3 ngày) | Capture 3 daily cycles |
| Hidden dim | 64 | Cân bằng capacity và dataset size (7,742 rows) |
| Layers | 2 | Đủ sâu cho temporal patterns, tránh overfit |
| Dropout | 0.2 | Regularization cho dataset nhỏ |
| Batch size | 64 | Ổn định training trên MPS |
| Learning rate | 1e-3 → ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | patience=10 |
| Optimizer | Adam |
| Loss | MSELoss |
| Device | Apple MPS (Metal Performance Shaders) |

### 8.4 Training Pipeline

1. Load Hybrid dataset (7,742 rows, 5 raw features)
2. Split temporal: Train 80% / Val 10% / Test 10%
3. StandardScaler fit on train only → transform all
4. Sliding window → sequences (lookback=72, horizon=h)
5. Train with early stopping (monitor val_loss)
6. **Test on REAL data only** (filter `is_imputed == 0` → 604 test points)

### 8.5 Kết Quả

| Model | 1h MASE | 6h MASE | 24h MASE | Params | Train time |
|-------|---------|---------|----------|--------|------------|
| Persistence | 1.000 | 1.000 | 1.000 | — | — |
| LSTM | 1.560 | 0.914 | 0.830 | 53,569 | 54-28s |
| **GRU** | **1.173** | **0.812** | **0.727** ⭐ | 40,705 | 217-113s |

**Phân tích**:
- GRU **vượt LSTM toàn diện** ở cả 3 horizons — ít parameters hơn nhưng hội tụ tốt hơn
- h=1: Cả hai DL chưa thắng Persistence (autocorr=0.97) nhưng GRU tiệm cận nhất trong các model "heavy"
- h=24: GRU đạt **MASE=0.727** — giảm **27.3%** lỗi so với Persistence → **NEW BEST model** toàn pipeline
- Apple MPS tăng tốc training ~3x so với CPU trên M-series

---

## 9. Temporal Fusion Transformer (TFT)

### 9.1 Động Lực

Transformer architecture đã cách mạng hóa NLP (BERT, GPT) và đang được áp dụng cho time series. Temporal Fusion Transformer (Lim et al., 2021) được thiết kế đặc biệt cho multi-horizon forecasting với cơ chế:
- **Variable Selection Network (VSN)**: Tự động chọn features quan trọng
- **Multi-head Attention**: Capture long-range dependencies
- **Gated Residual Network (GRN)**: Flexible nonlinear processing

### 9.2 Kiến Trúc Simplified TFT

Dự án triển khai **Simplified TFT** bằng PyTorch thuần (không phụ thuộc pytorch-forecasting):

```
Input (72 timesteps × 5 features)
  ↓
Variable Selection Network (GRN per feature)
  ↓
GRU Encoder (2 layers, hidden=32)
  ↓
Interpretable Multi-Head Attention (4 heads)
  ↓  
Gated Linear Unit (GLU)
  ↓
Output: PM2.5 prediction at horizon h
```

### 9.3 Cấu Hình

| Parameter | Giá trị | Ghi chú |
|-----------|---------|---------|
| hidden_dim | 32 | Nhỏ hơn GRU (64) — phù hợp dataset nhỏ |
| num_heads | 4 | Multi-head interpretable attention |
| dropout | 0.1 | Thấp hơn GRU vì TFT có VSN regularization |
| batch_size | 64 | |
| learning_rate | 5e-4 | Thấp hơn GRU do Attention cần ổn định |
| patience | 15 | Dài hơn GRU — TFT hội tụ chậm hơn |
| **Parameters** | **25,089** | Nhỏ hơn GRU (40K) 39% |

### 9.4 Kết Quả

| Metric | 1h | 6h | 24h |
|--------|-----|-----|-----|
| MAE | 2.460 | 5.181 | 5.093 |
| MASE | **1.029** | 0.822 | 0.812 |
| Train time | 38s | 52s | 21s |
| Test points | 604 | 604 | 604 |

**Phân tích**:
- **h=1: TFT đạt MASE=1.029** — tiệm cận Persistence nhất trong tất cả models! Cơ chế Attention giúp khai thác ngắn hạn hiệu quả hơn GRU (1.173) và LightGBM (1.492)
- h=6, h=24: TFT kém hơn GRU (~10-12%) — dataset nhỏ (7,742 rows) hạn chế khả năng học Attention patterns ở horizons dài
- **Efficiency**: Chỉ 25K params nhưng cạnh tranh tốt → cho thấy Attention mechanism có tiềm năng lớn nếu dataset lớn hơn
- TFT train nhanh nhất (21-52s) nhờ lightweight architecture

### 9.5 So Sánh Architectures

| Model | Params | Best horizon | Mechanism |
|-------|--------|-------------|-----------|
| LightGBM | ~500 trees | 6h (0.745) | Feature engineering + gradient boosting |
| GRU | 40,705 | **24h (0.727)** ⭐ | Temporal gates + sequential processing |
| TFT | 25,089 | **1h (1.029)** ⭐ | Attention + Variable Selection |
| LSTM | 53,569 | 24h (0.830) | Memory cells — outperformed by GRU |

> **Kết luận**: Không có "one-size-fits-all". TFT mạnh nhất short-term, GRU mạnh nhất long-term. Horizon quyết định model choice.

---

## 10. Ensemble Methods

### 10.1 Mục Tiêu

Kết hợp predictions từ nhiều models để giảm variance và khai thác complementary strengths:
- **LightGBM**: Mạnh ở feature-based patterns
- **GRU**: Mạnh ở temporal sequential patterns

### 10.2 Phương Pháp

**1. Stacking (Ridge Regression)**:
- Meta-learner: Ridge regression
- Base learners: LightGBM + GRU predictions as features
- Train meta-learner on validation set predictions

**2. Simple Weighted Average**:
- Grid search weight w ∈ [0.0, 0.05, ..., 1.0]
- Minimize MAE: `pred = w × LightGBM + (1-w) × GRU`

### 10.3 Kết Quả

| Method | 6h MASE | 24h MASE | Weights |
|--------|---------|----------|---------|
| LightGBM (solo) | 0.729 | 0.998 | — |
| GRU (solo) | **0.698** ⭐ | **0.730** ⭐ | — |
| Stack (Ridge) | 0.809 | 0.784 | LightGBM=0.72, GRU=-0.08 |
| Weighted Avg | 0.720 | 0.802 | LightGBM=75%, GRU=25% |

**Phân tích**:
- **Ensemble KHÔNG vượt GRU đơn lẻ** — kết quả đáng chú ý!
- Ridge stacking có intercept lớn (3.97-4.19) → bias correction thay vì combination thực sự
- GRU weight bị negative trong stacking → Ridge coi GRU predictions là noise → overfitting trên calibration set
- **Nguyên nhân**: Dataset nhỏ (604 test points) không đủ để meta-learner ước lượng tốt complementary patterns

> **Bài học**: Ensemble không phải lúc nào cũng cải thiện. Khi base models đã strong và dataset nhỏ, meta-learner dễ overfit calibration data.

---

## 11. Prediction Intervals — Khoảng Tin Cậy Dự Báo

### 11.1 Mục Tiêu

Point forecast (dự báo điểm) không đủ cho ra quyết định — cần **uncertainty quantification**. Dự án triển khai 3 phương pháp:

### 11.2 Phương Pháp

**1. Conformal Prediction (LightGBM)**:
- Distribution-free: Không giả định phân phối residuals
- Sử dụng calibration set (757 samples) để tính conformal score
- Đảm bảo coverage guarantee (lý thuyết: 1-α ≥ 90%)

**2. Quantile Regression (LightGBM)**:
- Train 2 models riêng cho quantile 0.05 và 0.95
- Trực tiếp ước lượng khoảng tin cậy 90%
- Không cần calibration set riêng

**3. MC Dropout (GRU)**:
- Monte Carlo sampling (50 forward passes) với dropout active
- Mean → point prediction, Std → uncertainty
- PI = mean ± z₀.₉₅ × std

### 11.3 Kết Quả

| Method | h=1 | h=6 | h=24 |
|--------|-----|-----|------|
| **Conformal** Coverage | 80.5% | 76.0% | 77.8% |
| Conformal Width (µg/m³) | 10.8 | 14.2 | 14.9 |
| **Quantile** Coverage | 86.2% | 83.2% | 79.1% |
| Quantile Width (µg/m³) | 16.0 | 18.8 | 18.6 |
| **MC Dropout** Coverage | 36.8% | 7.6% | 25.7% |
| MC Dropout Width (µg/m³) | 2.7 | 1.7 | 3.0 |

**Phân tích**:
- **Quantile Regression tốt nhất**: Coverage cao (79-86%), tuy width rộng hơn Conformal nhưng đáng tin cậy hơn
- **Conformal Prediction tốt thứ 2**: Coverage thấp hơn target 90% (76-81%) nhưng width hẹp hơn → trade-off hợp lý
- **MC Dropout thất bại**: Coverage quá thấp (7.6-36.8%) → GRU dropout variance quá nhỏ, không phản ánh uncertainty thật
- Coverage < 90% target: Do data IoT non-stationary (phân phối test khác train), violating exchangeability assumption

> **Khuyến nghị**: Sử dụng **Quantile Regression** cho production. Conformal cho real-time alerting (width nhỏ, nhanh). MC Dropout cần redesign (variational inference hoặc ensemble dropout).

---

## 12. Evaluation Methodology

### 12.1 Metrics System

| Metric | Vai Trò | Cơ Sở Khoa Học |
|--------|---------|----------------|
| **MAE** | **Primary** — sai số trung bình | Willmott & Matsuura (2005) |
| **MASE** | **Mandatory** — so với baseline | Hyndman & Koehler (2006) |
| **RMSE** | Secondary — phạt outlier | Chai & Draxler (2014) |
| **R²** | Overall fit | Standard |
| **sMAPE** | % error (thay MAPE) | MAPE undefined khi y=0 |

### 12.2 Tại Sao MAE Là Primary (Không Phải RMSE)

Theo Willmott & Matsuura (2005):
- MAE = thước đo **trực giác, không mơ hồ** của sai số trung bình
- RMSE phụ thuộc 3 yếu tố: average error + error variance + √n
- Trong context PM2.5 (µg/m³), MAE = X có nghĩa rõ: "trung bình sai X µg/m³"
- RMSE > MAE luôn đúng → tỷ lệ RMSE/MAE cho biết có outlier errors không

### 12.3 MASE — Tại Sao Bắt Buộc

Theo Hyndman & Koehler (2006):
- MASE < 1.0: model tốt hơn naive → **có giá trị** ✅
- MASE = 1.0: bằng naive → model vô ích
- MASE > 1.0: tệ hơn naive → **model tệ** ❌

MAE = 2.0 là tốt hay xấu? Không biết — phụ thuộc scale. MASE = 0.8 → luôn biết: tốt hơn naive 20%.

### 12.4 Validation Methodology

Dựa trên Tashman (2000) — _"Out-of-sample tests of forecasting accuracy"_:
1. ✅ Temporal split (no shuffle)
2. ✅ Walk-forward (expanding window, 5 folds)
3. ✅ Report mean ± std qua folds
4. ✅ Baseline comparison bắt buộc
5. ⬜ Diebold-Mariano test (khi MAE diff < 10%)

### 12.5 So Sánh Với Literature

Benchmark PM2.5 hourly prediction từ nghiên cứu quốc tế (2023-2025):

| Model Type | MAE Typical (µg/m³) | R² Typical |
|-----------|---------------------|------------|
| Linear Models | 10–25+ | 0.5–0.7 |
| Tree-Based (XGBoost, RF) | 2.5–12 | 0.80–0.95 |
| Deep Learning (LSTM, CNN) | 2.0–10 | 0.85–0.96 |
| Hybrid (CNN-LSTM + Meta) | 2.0–8 | 0.88–0.97 |

**Lưu ý**: Dự án này có MAE baseline = 1.821 (thấp hơn literature) vì PM2.5 concentration thấp (median ≈ 10 µg/m³) — indoor sensor hoặc khu vực ít ô nhiễm.

---

## 13. Kết Quả So Sánh Trước/Sau Fix Leakage

### 13.1 Kết Quả TRƯỚC Fix (⚠️ LEAKAGE)

| Model | MAE | RMSE | R² | MASE | Ghi Chú |
|-------|-----|------|----|------|---------|
| Persistence | 1.821 | 3.306 | — | 1.000 | Baseline ✅ |
| Ridge | 0.004 | 0.006 | 1.000 | 0.002 | ⚠️ LEAKAGE |
| Lasso | 0.106 | 0.132 | 0.999 | 0.058 | ⚠️ LEAKAGE |
| RandomForest | 0.143 | 0.383 | 0.995 | 0.078 | ⚠️ LEAKAGE |
| XGBoost | 0.189 | 0.395 | 0.994 | 0.104 | ⚠️ LEAKAGE |
| LightGBM | 0.221 | 0.384 | 0.995 | 0.121 | ⚠️ LEAKAGE |

### 13.2 Kết Quả SAU Fix (2026-04-04) ✅

| Model | MAE | RMSE | R² | MASE | Beats Baseline? |
|-------|-----|------|----|------|-----------------|
| **Persistence** | **1.821** | 3.306 | — | **1.000** | Baseline |
| Lasso | 1.915 | 3.187 | — | 1.052 | ❌ (MASE > 1) |
| ElasticNet | 2.037 | 3.339 | — | 1.118 | ❌ |
| LightGBM | 2.276 | 3.266 | — | 1.250 | ❌ |
| RandomForest | 2.666 | 3.668 | — | 1.464 | ❌ |
| Ridge | 2.824 | 4.591 | — | 1.551 | ❌ |
| XGBoost | 3.364 | 4.345 | — | 1.847 | ❌ |

### 13.3 Phân Tích Kết Quả

**Phát hiện quan trọng**: Sau khi loại bỏ leakage, **KHÔNG model ML nào đánh bại Persistence baseline** (MASE > 1.0 cho tất cả).

Đây **KHÔNG phải** kết quả thất bại — đây là **kết quả trung thực**:

1. **Tại sao Persistence mạnh**: PM2.5 có autocorrelation lag 1h = 0.89 → rất khó đánh bại "dùng giá trị 1h trước"
2. **ML models chưa tối ưu**: Chưa hyperparameter tuning (Optuna), chưa feature selection
3. **Validation đúng**: Walk-Forward CV cho kết quả realistic hơn random split
4. **Consistent với literature**: M4 Competition (Makridakis 2020) cho thấy simple methods often competitive

**So sánh trước/sau fix**:

| Model | MAE Trước (⚠️) | MAE Sau (✅) | Thay Đổi |
|-------|-----------------|-------------|----------|
| Ridge | 0.004 | 2.824 | **+706x** (leakage bị loại) |
| Lasso | 0.106 | 1.915 | **+18x** |
| RandomForest | 0.143 | 2.666 | **+19x** |
| XGBoost | 0.189 | 3.364 | **+18x** |
| LightGBM | 0.221 | 2.276 | **+10x** |

**Bài học**: Kết quả "trước fix" tốt gấp 10-700 lần → leakage impact cực lớn. Nếu không audit, kết quả publishable nhưng SAI.

### 13.4 Feature Selection + Optuna Tuning (2026-04-04) ✅

#### Phương pháp

1. **Feature Selection**: Random Forest importance → search top_k ∈ {15, 20, 25, 30, 40, 50}
2. **Optuna Tuning**: 100 trials (Lasso), 150 trials (LightGBM) trên Walk-Forward CV
3. **So sánh**: Tuned models on selected features vs all features

#### Feature Importance (RF-based)

| Rank | Feature | Importance | Nhóm |
|------|---------|-----------|------|
| 1 | **pm25_lag_1h** | **0.838** | Lag |
| 2 | pm25_lag_24h | 0.006 | Lag |
| 3 | is_rush_hour | 0.005 | Calendar |
| 4 | pm25_pct_change_1h | 0.005 | Diff |
| 5 | do_am_lag_3h | 0.004 | Lag |
| 6–10 | do_am, hour_cos/sin, co2, roll_24h | ~0.004 each | Mixed |

**Phát hiện**: `pm25_lag_1h` chiếm **83.8%** tổng importance → model chủ yếu dựa vào giá trị 1h trước (giống Persistence). Các features khác đóng góp rất ít.

**Feature count search**: top_k=20 cho CV MAE thấp nhất (3.039). Nhiều features hơn → overfitting trên noise.

#### Kết Quả Sau Tuning

| Model | Features | MAE | RMSE | MASE | Cải Thiện vs Default |
|-------|----------|-----|------|------|---------------------|
| **LightGBM_tuned_all** | All (94) | **1.874** | 3.070 | **1.029** | ↓17.6% từ 2.276 |
| LightGBM_tuned | Top-20 | 1.937 | 3.103 | 1.063 | ↓14.9% |
| Lasso_tuned_all | All (94) | 1.946 | 3.149 | 1.069 | +1.6% |
| Lasso_tuned | Top-20 | 2.050 | 3.213 | 1.126 | +7.0% |

#### Phân Tích

- **LightGBM_tuned_all MAE=1.874** — gap chỉ còn **2.9%** vs Persistence (1.821)
- **All features > Top-20**: Feature selection loại mất context phụ trợ
- **LightGBM cải thiện đáng kể** (2.276 → 1.874), Lasso thay đổi ít (1.915 → 1.946)
- **Optuna best params**: LightGBM cần max_depth=3 (nông), n_estimators=637, learning_rate=0.013 → regularized model

#### Bài Học Từ Tuning

1. **Diminishing returns**: Từ default → tuned: cải thiện 17.6%. Nhưng vẫn chưa beat baseline.
2. **Feature dominance**: Khi 1 feature chiếm 84% importance, model khó tốt hơn "predict = lag_1h"
3. **All features > Selected**: Trong low-signal regime, mỗi bit thông tin phụ trợ đều quan trọng
4. **Gap 2.9%** → cần multi-horizon hoặc DL để vượt qua

### 13.5 Hướng Cải Thiện (Next Steps)

Để đánh bại Persistence (MASE < 1.0), cần:
1. **Multi-horizon** (6h, 24h) — Persistence yếu ở horizon dài, ML có lợi thế
2. **LSTM/GRU** — capture non-linear temporal dependencies
3. **Ensemble** — kết hợp Lasso + LightGBM

---

## 14. Bài Học Kinh Nghiệm

### 14.1 Data Leakage (Quan Trọng Nhất)

**Bài học**: Kết quả "quá tốt" (R² > 0.99) gần như luôn sai. Luôn nghi ngờ trước khi celebrate.

**Quy tắc phòng tránh**:
1. `shift(1)` trước MỌI phép tính liên quan target
2. Shuffle test bắt buộc sau mỗi feature set mới
3. R² > 0.99 = red flag → audit ngay
4. Kiểm tra: 2 features có tái tạo target không

**Reference**: Kapoor & Narayanan (2023, _Patterns_)

### 14.2 Stationarity Testing

**Bài học**: KHÔNG bao giờ dựa vào 1 test đơn lẻ. ADF và KPSS có null hypothesis **ngược nhau**.

- ADF H₀: non-stationary
- KPSS H₀: stationary
- Chạy CẢ HAI và so sánh kết quả

### 14.3 CSV Loading Performance

**Bài học**: Large DataFrames (95+ cols) → `pd.read_csv()` chậm.

**Giải pháp**:
- Script audit: dùng `usecols` chỉ load cột cần
- Production: dùng parquet format (10x nhanh hơn CSV)
- Luôn thêm `print(..., flush=True)` cho progress indication

### 14.4 IoT Sensor Data Characteristics

**Bài học**: Dữ liệu IoT thực tế KHÁC dữ liệu benchmark:
- Nhiều gaps (sensor offline, bảo trì)
- Outliers do sensor malfunction
- Tần suất không đều
→ Cleaning pipeline cần robust, không assume dữ liệu sạch

### 14.5 Process > Results

**Bài học**: Kết quả "tốt" (MAE=0.004) nhưng sai methodology → vô giá trị.
Quá trình nghiêm ngặt (proper validation, anti-leakage, baseline comparison) cho kết quả "kém hơn" nhưng **đáng tin cậy** → có giá trị khoa học.

### 14.6 Workflow — Kill Processes

**Bài học**: Tiến trình Python treo (đặc biệt khi CSV load timeout) gây conflict khi chạy test/build tiếp.

**Phòng tránh**: Luôn `pkill -f "python.*scripts/"` trước khi chạy pipeline mới.

### 14.7 Multi-Horizon Evaluation — Thay Đổi Kết Luận

**Bài học**: Đánh giá mô hình ở **một forecast horizon duy nhất** có thể dẫn đến kết luận sai lệch.

**Thực tế nghiên cứu**:
- Nếu chỉ đánh giá ở **h=1h**: kết luận "ML vô dụng, Persistence tốt nhất" (MASE=1.012 ❌)
- Mở rộng sang **h=6h**: ML giảm **27% lỗi** so với Persistence (MASE=0.730 ✅)
- Tiếp tục **h=24h**: ML giảm **19% lỗi** (MASE=0.812 ✅)

**Nguyên nhân**: PM2.5 có autocorrelation rất cao ở lag-1h (~0.97) → "copy giá trị giờ trước" gần như unbeatable. Nhưng khi horizon tăng, autocorrelation giảm nhanh → ML tận dụng multivariate features để vượt trội.

> **Quy tắc**: LUÔN đánh giá ở nhiều forecast horizons (short, medium, long) trước khi kết luận về hiệu quả mô hình [9].

### 14.8 Naive Baseline Trước — Mô Hình Phức Tạp Sau

**Bài học**: Persistence model (lấy giá trị gần nhất làm dự báo) là baseline **bắt buộc** cho mọi bài toán time series.

**Tại sao**: Persistence tận dụng autocorrelation — đặc tính cơ bản nhất của time series. Nếu một mô hình ML không thắng được Persistence, mô hình đó **không tạo giá trị thực** [5].

**Kết quả dự án**: MASE (Mean Absolute Scaled Error) sử dụng Persistence làm mẫu số. MASE < 1.0 = mô hình tốt hơn naive baseline. Cách tiếp cận này tuân theo khuyến nghị của Hyndman & Koehler (2006) [1].

### 14.9 Missing Data Handling — Imputation Strategy Quyết Định Chất Lượng Mô Hình

**Bài học**: Với dữ liệu IoT có 74% missing, cách xử lý missing data ảnh hưởng **lớn hơn** việc chọn mô hình ML.

**So sánh 4 strategies** (cùng LightGBM, cùng features):

| Strategy | MASE | Ghi chú |
|----------|------|---------|
| Segment Only (drop all gaps) | 1.085 | Ít data nhất, nhưng "sạch" |
| Extended Interpolation (Spline) | 1.321 | ⚠️ Worst — Cubic Spline tạo noise cho gap >6h |
| ML Imputation (KNN) | 1.084 | KNN multivariate fill tốt hơn |
| **Hybrid (Spline ≤6h + KNN 6-24h)** | **1.066** | ⭐ Best — kết hợp ưu điểm cả hai |

**Quy tắc**:
1. Cubic Spline chỉ hữu ích cho gaps ngắn (≤6h). Gaps dài → tạo giá trị giả, gây noise.
2. KNN multivariate imputation tốt hơn vì sử dụng context từ các biến liên quan (nhiệt độ, độ ẩm, CO2).
3. Gaps >24h: **KHÔNG CỐ GẮNG** recover — drop để giữ data integrity.

### 14.10 Feature Importance Thay Đổi Theo Forecast Horizon

**Bài học**: Cùng một bộ features, nhưng **cơ chế dự đoán hoàn toàn khác** ở mỗi horizon.

| Horizon | Top Feature | Cơ chế |
|---------|------------|--------|
| **1h** | `pm25_lag_1h` (importance=834) | Copy giá trị gần nhất — autocorrelation dominates |
| **6h** | `hour_sin` (608), `pm25_roll_24h_mean` (517) | Temporal patterns — xu hướng trong ngày |
| **24h** | `hour_cos`, `diem_suong`, `nhiet_do` (phân bố đều) | Multivariate — cần nhiều nguồn thông tin |

**Ý nghĩa khoa học**: Khi horizon ngắn, target nặng về inertia (quán tính). Khi horizon dài, cần hiểu **cơ chế vật lý** (nhiệt độ ảnh hưởng đối lưu, độ ẩm ảnh hưởng kết tụ bụi). Đây là lý do ML multivariate thắng Persistence ở horizons dài.

### 14.11 Test-on-Real-Only — Nguyên Tắc Toàn Vẹn Dữ Liệu

**Bài học**: Dữ liệu imputed (được điền bởi thuật toán) **KHÔNG BAO GIỜ** được dùng trong tập test/evaluation.

**Tại sao**: Nếu test trên dữ liệu imputed → model học dự đoán output **của thuật toán imputation** chứ không phải giá trị thực → kết quả "tốt" nhưng **không có ý nghĩa thực tế**.

**Thiết kế trong dự án**:
1. Thêm cột `is_imputed` (boolean) vào DataFrame sau mỗi bước imputation
2. Khi chia train/val/test: train có thể chứa imputed data, nhưng **test set BẮT BUỘC chỉ chứa real data**
3. Filter: `test_real = test_df[test_df['is_imputed'] == False]`

**Kết quả**: Từ 758 test rows → 601 real rows (79.3%). Mất 21% data nhưng đảm bảo evaluation **trung thực**.

---

## 15. Kết Luận & Hướng Phát Triển

### 15.1 Kết Luận

1. Đã xây dựng pipeline end-to-end từ IoT sensor → dự báo PM2.5 đa horizon (1h, 6h, 24h)
2. **Phát hiện và sửa data leakage** — bài học quan trọng nhất của dự án
3. **Xử lý 74% missing data** — thiết kế và so sánh 4 strategies, chứng minh Hybrid (Spline + KNN) tối ưu nhất
4. **Multi-horizon evaluation** — chứng minh ML/DL tạo giá trị thực ở horizons dài:
   - 6h: GRU (ensemble run) giảm **30%** lỗi so với Persistence (MASE=0.698)
   - 24h: GRU đơn lẻ giảm **27.3%** lỗi (MASE=0.727) — **BEST toàn pipeline**
5. **Deep Learning vượt ML**: GRU (40K params) vượt LightGBM (tuned by Optuna) ở cả 6h và 24h
6. **TFT Transformer**: Tiệm cận Persistence nhất ở h=1 (MASE=1.029) với chỉ 25K params — cho thấy Attention mechanism tiềm năng lớn trên dataset lớn hơn
7. **Ensemble thất bại mang tính insight**: Stacking (Ridge) KHÔNG vượt GRU đơn lẻ — dataset nhỏ không đủ cho meta-learner
8. **Prediction Intervals**: Quantile Regression đạt coverage 79-86%, Conformal 76-81%. MC Dropout thất bại (coverage <37%)
9. **Test-on-Real-Only** — đảm bảo evaluation methodology trung thực, không bias từ imputed data
10. Anti-leakage test suite tự động (11 tests) → phòng tránh cho tương lai
11. **Dashboard & AI Chatbot** — 12 pages + RAG assistant (241 documents, multilingual embedding)

### 15.2 So Sánh Tổng Hợp Các Mô Hình

| Mô hình | Loại | 1h (MASE) | 6h (MASE) | 24h (MASE) | Nhận xét |
|---------|------|-----------|-----------|------------|----------|
| Persistence | Baseline | **1.000** | 1.000 | 1.000 | Best ở 1h (autocorr=0.97) |
| ARIMA(2,1,1) | Statistical | 1.023 | 0.856 | 0.913 | Univariate, nhanh |
| SARIMA×(2,1,0,24) | Statistical | 1.283 | 0.762 | 0.813 | Seasonal s=24 rất mạnh |
| LightGBM (Optuna) | ML | 1.492 | 0.745 | 0.842 | ML vượt tại 6h, 24h |
| LSTM | Deep Learning | 1.560 | 0.914 | 0.830 | Yếu hơn GRU toàn diện |
| **GRU** | Deep Learning | 1.173 | 0.812 | **0.727** ⭐⭐ | **BEST ở 24h!** |
| GRU (Ensemble run) | Ensemble | — | **0.698** ⭐ | 0.730 | Best ở 6h |
| Stack (Ridge) | Ensemble | — | 0.809 | 0.784 | Stacking kém hơn GRU đơn |

### 15.3 Hướng Phát Triển

| Ưu tiên | Hạng mục | Trạng thái | Kết quả |
|---------|---------|-----------|---------|
| ~~Cao~~ | ~~Re-run ML với clean features~~ | ✅ Done | MASE=1.029 (gap 2.9%) |
| ~~Cao~~ | ~~Optuna tuning~~ | ✅ Done | LightGBM best params found |
| ~~Cao~~ | ~~Multi-horizon (6h, 24h)~~ | ✅ Done | ML beats Persistence ở 6h, 24h |
| ~~Cao~~ | ~~Level 1: ARIMA/SARIMA~~ | ✅ Done | SARIMA ≈ LightGBM ở 24h |
| ~~Cao~~ | ~~Level 4: LSTM, GRU~~ | ✅ Done | GRU 24h MASE=0.727 — NEW BEST |
| ~~Cao~~ | ~~Level 5: Ensemble (stacking)~~ | ✅ Done | GRU_ens 6h MASE=0.698 ⭐ |
| ~~Cao~~ | ~~SHAP Explainability~~ | ✅ Done | SHAP (LightGBM) + Permutation (GRU) |
| ~~Cao~~ | ~~Diebold-Mariano Test~~ | ✅ Done | GRU significant ở 6h & 24h |
| ~~Cao~~ | ~~Residual Diagnostics~~ | ✅ Done | Ljung-Box, Shapiro-Wilk, Jarque-Bera |
| ~~Cao~~ | ~~Thesis Draft CTU 1799~~ | ✅ Done | Ch1-5 + bảng + hyperparams |
| ~~Trung bình~~ | ~~Confidence Intervals~~ | ✅ Done | Conformal + Quantile Reg + MC Dropout |
| ~~Trung bình~~ | ~~Streamlit Dashboard~~ | ✅ Done | 12 pages + AI Chatbot (RAG) |
| ~~Thấp~~ | ~~Model Export (ONNX/TorchScript)~~ | ✅ Done | GRU .pt + LightGBM .txt (13 files) |
| ~~Thấp~~ | ~~Temporal Fusion Transformer~~ | ✅ Done | MASE=1.029 (h=1 best!), 25K params |

---

## 16. Tài Liệu Tham Khảo

### Academic Papers

| # | Tác Giả | Tiêu Đề | Journal | Năm | Chủ Đề |
|---|---------|---------|---------|-----|--------|
| 1 | Hyndman & Koehler | Another look at measures of forecast accuracy | _Int. J. Forecasting_ | 2006 | MASE metric |
| 2 | Willmott & Matsuura | Advantages of MAE over RMSE | _Climate Research_ | 2005 | MAE vs RMSE |
| 3 | Diebold & Mariano | Comparing Predictive Accuracy | _J. Business & Econ. Stat._ | 1995 | DM test |
| 4 | Tashman | Out-of-sample tests of forecasting accuracy | _Int. J. Forecasting_ | 2000 | Walk-forward |
| 5 | Hyndman & Athanasopoulos | Forecasting: Principles & Practice (3e) | OTexts | 2021 | Textbook |
| 6 | Chai & Draxler | RMSE or MAE? | _Geosci. Model Dev._ | 2014 | Metric debate |
| 7 | Gneiting & Raftery | Strictly Proper Scoring Rules | _JASA_ | 2007 | CRPS |
| 8 | Kapoor & Narayanan | Leakage and Reproducibility Crisis | _Patterns_ | 2023 | Data leakage |
| 9 | Makridakis et al. | The M4 Competition | _Int. J. Forecasting_ | 2020 | Benchmarking |

### Books (trong docs/)

| Sách | Tác Giả | Chủ Đề |
|------|---------|--------|
| Modern Time Series Forecasting with Python | Manu Joseph | ML/DL Pipeline |
| Time Series Forecasting in Python | Marco Peixeiro | ARIMA → DL |
| Deep Learning for Time Series Cookbook | Vitor Cerqueira | LSTM, CNN |
| Time Series Analysis with R Examples | Shumway & Stoffer | Statistical Foundation |
| Air Pollution Modeling | — | Domain Knowledge |

---

> 📌 **Lưu ý**: Tài liệu này đã hoàn thành với tất cả phases thí nghiệm.
> Phiên bản hiện tại: 2026-04-11
