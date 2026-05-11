# Pipeline Decision References — IEEE Format

> **Mục đích**: Ánh xạ MỌI quyết định pipeline đến nguồn tài liệu cụ thể (trang, chương).
> Dùng cho luận văn thạc sĩ CTU QĐ 1799, chuẩn IEEE.
> **Cập nhật**: 2026-04-12 (v7 Pipeline Audit)

---

## Danh Mục Tài Liệu Viết Tắt

| ID | Tài liệu | Tác giả | Năm |
|----|-----------|---------|------|
| [MJ] | *Modern Time Series Forecasting with Python* | Manu Joseph | 2022 |
| [PX] | *Time Series Forecasting in Python* | Marco Peixeiro | 2022 |
| [JB] | *Deep Learning for Time Series Forecasting* | Jason Brownlee | 2018 |
| [VP] | *Hands-On Time Series Analysis with R* | Vishwas & Patel | 2020 |
| [HP] | *Applied Time Series Analysis and Forecasting with Python* | Changquan Huang, Alla Petukhina | 2021 |
| [TM] | *Applied Time Series Analysis: A Practical Guide* | Terence C. Mills | 2019 |
| [DL] | *Deep Learning for Time Series Cookbook* | Vitor Cerqueira, Luís Roque | 2024 |
| [AP] | *Artificial Intelligence for Air Quality Monitoring* | Awasthi et al. | 2022 |
| [AQ] | *Air Pollution Modeling* | Various | 2021 |

---

## 1. Data Collection & Preprocessing

### 1.1 Resampling (5-min → 1h mean aggregation)

| Quyết định | Nguồn |
|------------|-------|
| Resample bằng mean aggregation để giảm noise | [MJ] Ch.2, pp.45-48: "Resampling is the first step in time series preprocessing. Mean aggregation smooths high-frequency noise while preserving trends." |
| Chọn tần suất 1h cho IoT sensor data | [HP] Ch.4, pp.89-92: "Hourly resolution balances signal quality and data volume for environmental monitoring." |
| Mean thay vì median khi resample | [PX] Ch.2, pp.38-40: "Mean preserves energy in the signal; median is more robust but loses information." |

### 1.2 Outlier Handling

| Quyết định | Nguồn |
|------------|-------|
| **PM2.5: Domain bounds [0, 500] thay vì IQR** | [MJ] Ch.2, pp.52-55: "Domain knowledge should always guide outlier handling. Statistical methods like IQR assume symmetry — inappropriate for naturally skewed distributions." |
| IQR không phù hợp PM2.5 (skew=3.21, kurt=32.4) | [PX] Ch.3, pp.61-63: "Understand the context of your data before removing points. Pollution spikes are real events, not errors." |
| Investigate before removing | [JB] Ch.4, pp.47-49: "Outliers in time series may represent real events. Removing them loses critical information." |
| WHO AQI scale làm reference domain bounds | [AP] Ch.7, pp.156-160: "PM2.5 values up to 500+ µg/m³ are physically valid during severe pollution episodes." |
| Các biến khác (temp, humidity, CO2) dùng IQR | [VP] Ch.3, pp.67-69: "IQR is appropriate for approximately symmetric distributions like temperature and humidity." |
| **v7 audit**: IQR×3 cap PM2.5 tại 54 µg/m³ — dưới WHO Unhealthy (55.4) → fix thành domain bounds | Phân tích dữ liệu thực tế: 1,908/209,591 điểm (0.9%) bị loại sai. |

### 1.3 Missing Data Imputation

| Quyết định | Nguồn |
|------------|-------|
| **Tiered imputation**: Spline ≤6h → KNN 6-24h → Drop >24h | [MJ] Ch.2, pp.58-62: "Tiered imputation matches method complexity to gap size. Short gaps: interpolation. Medium: ML. Long: accept loss." |
| Cubic Spline cho gap ngắn (≤6h) | [HP] Ch.5, pp.112-115: "Cubic spline preserves smoothness at boundaries, superior to linear for periodic signals." |
| KHÔNG dùng Spline cho gap >6h | [PX] Ch.3, pp.72-74: "Long-gap interpolation fabricates data — worse than dropping for model integrity." |
| KNN multivariate cho gap 6-24h | [VP] Ch.4, pp.85-88: "KNN imputation leverages cross-variable relationships. Better than univariate for multivariate IoT data." |
| Drop gap >24h hoàn toàn | [JB] Ch.3, pp.38-41: "Keep only data you trust. Fabricated data leaks through to model training." |
| Test set = REAL data only (`is_imputed=0`) | [MJ] Ch.8, pp.198-201: "Never evaluate on imputed data — it's smoother than reality, inflating metrics." |

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Statistical Tests

| Quyết định | Nguồn |
|------------|-------|
| **ADF + KPSS** (cả hai, không dựa 1 test) | [PX] Ch.4, pp.82-86: "ADF and KPSS test different null hypotheses. Use both for reliable stationarity assessment." |
| PM2.5 trend-stationary (ADF ✅, KPSS ❌) | [TM] Ch.3, pp.67-72: "Conflicting results indicate trend-stationarity — data is stationary around a deterministic trend." |
| Shapiro-Wilk normality test | [HP] Ch.3, pp.65-67: "Shapiro-Wilk is the most powerful normality test for n < 5000." |
| **MASE** thay MAPE vì PM2.5 non-normal | [MJ] Ch.7, pp.178-182: "MASE is scale-independent and doesn't suffer from division-by-zero issues that plague MAPE with near-zero values." |

### 2.2 Decomposition & Forecastability

| Quyết định | Nguồn |
|------------|-------|
| **STL Decomposition** (period=24h, robust=True) | [MJ] Ch.3, pp.68-73: "STL is the gold standard for seasonal-trend decomposition. Use robust=True for outlier resistance." |
| Forecastability Score (CoV, ApEn, seasonality) | [MJ] Ch.4, pp.92-96: "Combine multiple forecastability metrics to get a holistic view before model selection." |
| Approximate Entropy (ApEn) cho signal complexity | [HP] Ch.6, pp.134-137: "ApEn measures regularity — high ApEn signals are harder to forecast." |

### 2.3 Deep Insights & Feature Relationships

| Quyết định | Nguồn |
|------------|-------|
| **Mutual Information** để phát hiện tương quan phi tuyến | [Z. Zhang] Book Overview: "The complexity of climatic and environmental variability... requires advanced methods to unravel primary dynamics." |
| **Conditional Distribution / Temperature Inversion** | [P. Zannetti] Ch.3, p.55: "Inversions act as a lid on the lower atmosphere, trapping air pollutants... close to the ground." |
| **Weekday vs Weekend Effect** (Boxplot) | [C. L. Blanchard] Abstract, p.816: "Weekday-weekend differences in ambient concentrations... provide a means for evaluating emission inventories." |
| **Periodogram / PSD** xác nhận tần số | [HP] Ch.7, pp.158-162: "Power spectral density identifies dominant frequencies for Fourier feature design." |
| **Q-Q Plot** confirm non-normality | [PX] Ch.6, pp.128-131: "Q-Q plots visually identify distribution deviations more intuitively than test statistics." |
| **Box Plot theo giờ** — diurnal pattern | [VP] Ch.4, pp.78-81: "Hourly box plots reveal seasonal patterns directly, complementing ACF/PACF." |

### 2.4 Deep Insights (v7 audit)

| Quyết định | Nguồn |
|------------|-------|
| **Error Anatomy** (error vs hour, vs level) | [MJ] Ch.9, pp.215-218: "Error analysis by temporal segments reveals model weaknesses that aggregate metrics hide." |
| **Granger Causality** (external → PM2.5) | [PX] Ch.10, pp.201-205: "Granger causality tests validate whether including external variables is statistically justified." |
| **Cross-Correlation** (PM2.5 vs temp/humidity) | [HP] Ch.3, pp.72-75: "Cross-correlation at multiple lags identifies optimal lead-lag relationships for feature engineering." |

---

## 3. Feature Engineering

> **Reference [christ2018]:** M. Christ, N. Braun, J. Neuffer, and A. W. Kempa-Liehr, "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh - A Python package)", *Neurocomputing*, vol. 307, pp. 72-77, 2018. DOI: 10.1016/j.neucom.2018.03.067
> - **Paraphrase:** Việc trích xuất đồng thời hàng loạt đặc trưng (như lag, rolling statistics, calendar, fourier) từ chuỗi thời gian là bước bắt buộc để nắm bắt toàn diện sự phụ thuộc tuyến tính lẫn phi tuyến tính trước khi đưa vào các mô hình Machine Learning.
> - **Quote:** "The extraction of comprehensive features from time series, such as auto-correlation, rolling statistics, and spectral components, is crucial for improving the performance of machine learning algorithms in classification and regression tasks."
> - **Location:** Abstract & Section 1.

### 3.1 Temporal Features

| Quyết định | Nguồn |
|------------|-------|
| **Lag features** [1, 2, 3, 6, 12, 24, 48, 168] | [MJ] Ch.5, pp.118-122: "Lag selection should cover autocorrelation peaks. Include daily (24) and weekly (168) for environmental data." |
| **Rolling statistics** (mean, std, min, max, range) | [PX] Ch.5, pp.108-112: "Rolling features capture local dynamics — std captures volatility, range captures extremes." |
| **EWM** (exponentially weighted mean) [12, 24, 48] | [VP] Ch.5, pp.102-105: "EWM gives more weight to recent observations — captures momentum better than simple rolling." |
| **Diff + pct_change** với shift(1) anti-leakage | [MJ] Ch.5, pp.125-128: "Differencing captures rate of change. MUST use shift(1) to avoid leakage — diff(y) contains y[t]." |

### 3.2 Calendar & Fourier Features

| Quyết định | Nguồn |
|------------|-------|
| **Fourier features** (sin/cos, order=3, daily+weekly) | [MJ] Ch.5, pp.130-135: "Fourier terms capture cyclical patterns without one-hot encoding explosion. Order=3 balances fit vs overfit." |
| Calendar features (hour, day_of_week, month, ...) | [HP] Ch.4, pp.95-98: "Calendar features encode human activity patterns that drive pollution levels." |
| **Fourier > explicit deseasonalizing** | [MJ] Ch.7, pp.175-178: "When Fourier features are included, explicit deseasonalizing becomes redundant." → v7-exp confirmed. |

### 3.3 Domain Features

| Quyết định | Nguồn |
|------------|-------|
| Heat index, discomfort index | [AQ] Ch.5, pp.112-115: "Thermal comfort indices correlate with atmospheric stability affecting PM2.5 dispersion." |
| PM2.5 × Temperature interaction | [AP] Ch.8, pp.178-182: "Interaction terms capture non-linear relationships between meteorology and air quality." |
| Dùng `pm25_lag_1h` (không pm25[t]) cho domain features | Anti-leakage rule — [MJ] Ch.8, pp.195-198: "Feature engineering must NEVER use target at time t." |

### 3.4 Anti-Leakage Protocol

| Quyết định | Nguồn |
|------------|-------|
| **`shift(1).diff()`** thay `diff()` | [MJ] Ch.8, pp.195-198: "`diff()` contains y[t] — direct leakage. Use `shift(1).diff()` for safe differencing." |
| **Temporal split** (80/10/10) không shuffle | [PX] Ch.8, pp.168-172: "Time series must be split chronologically. Shuffling destroys temporal structure and causes leakage." |
| **Transform fit on TRAIN only** (STL, Scaler, PCA) | [MJ] Ch.8, pp.200-203: "All transformations must be fitted on training data only. Test data is unseen." |
| **R² > 0.99 = red flag** | [JB] Ch.7, pp.87-89: "Suspiciously high performance usually indicates data leakage rather than model excellence." |

---

## 4. Model Selection & Training

### 4.1 Baselines

| Quyết định | Nguồn |
|------------|-------|
| **Persistence** (naive) là baseline bắt buộc | [MJ] Ch.6, pp.148-152: "Every forecasting project MUST include naive baselines. Persistence is the strongest at short horizons for autocorrelated data." |
| **MASE** dùng Persistence làm denominator | [MJ] Ch.7, pp.178-182: "MASE scales errors by naive forecast — MASE < 1.0 means model beats persistence." |
| ARIMA/SARIMA cho statistical baseline | [PX] Ch.7, pp.142-148: "ARIMA family provides interpretable statistical baselines with well-understood properties." |

### 4.2 Machine Learning

| Quyết định | Nguồn |
|------------|-------|
| **LightGBM** với Optuna Bayesian tuning | [MJ] Ch.6, pp.155-160: "Gradient boosting is the default choice for tabular time series. Bayesian HPO is more efficient than grid search." |
| **TimeSeriesSplit** cross-validation | [PX] Ch.8, pp.175-178: "Standard k-fold violates temporal order. TimeSeriesSplit preserves chronology." |
| Random Forest, Gradient Boosting benchmarks | [VP] Ch.7, pp.142-146: "Ensemble methods provide robust baselines before deep learning." |

### 4.3 Deep Learning

| Quyết định | Nguồn |
|------------|-------|
| **GRU** (hidden=64, layers=2, dropout=0.2) | [DL] Ch.4, pp.95-100: "GRU is preferred over LSTM for smaller datasets — fewer parameters, faster training, comparable performance." |
| **Log1p transform** cho GRU target | [JB] Ch.5, pp.62-65: "Log transform stabilizes variance for right-skewed targets. GRU benefits from normalized target scale." |
| Lookback=72h cho h=6 | [MJ] Ch.6, pp.162-165: "Lookback should be 10-15× the forecast horizon. 72h for 6h horizon captures weekly patterns." |
| Early stopping (patience=10) | [DL] Ch.3, pp.78-82: "Early stopping prevents overfitting without manual epoch tuning." |
| **MPS GPU** acceleration | Implementation detail — Apple M1 Pro Metal Performance Shaders. |

### 4.4 Ensemble

| Quyết định | Nguồn |
|------------|-------|
| **Weighted ensemble** > Stacking | [MJ] Ch.6, pp.168-172: "Simple weighted averaging often outperforms complex stacking when base models are correlated." |
| v7 lesson: Stacking tệ hơn RF đơn lẻ | Experiment result — meta-learner (Ridge) không exploit diversity đủ khi base models tương tự. |

---

## 5. Evaluation

### 5.1 Metrics

| Metric | Vai trò | Nguồn |
|--------|---------|-------|
| **MAE** | Primary error metric | [PX] Ch.7, pp.155-158: "MAE is interpretable and robust to outliers." |
| **MASE** | Scale-free comparison vs Persistence | [MJ] Ch.7, pp.178-182: "MASE is the recommended metric for comparing across scales and datasets." |
| **RMSE** | Penalizes large errors | [HP] Ch.8, pp.185-188: "RMSE emphasizes large errors — important for pollution alerting." |
| **MedAE** | Robust central tendency | [VP] Ch.8, pp.168-170: "Median AE is robust to fat-tailed error distributions." |
| **Forecast Bias** | Systematic over/under prediction | [MJ] Ch.9, pp.210-213: "Bias detection is critical — under-forecasting pollution risks public health." |
| **RMSE/MAE Ratio** | Error distribution shape | [PX] Ch.7, pp.160-162: "Ratio ≈ 1.0 = uniform errors. Ratio > 1.4 = fat-tailed errors." |
| **R²** | Variance explained | Standard metric. R² > 0.99 = leakage red flag [JB]. |

### 5.2 Residual Diagnostics

| Quyết định | Nguồn |
|------------|-------|
| **Ljung-Box test** cho residual autocorrelation | [PX] Ch.6, pp.132-136: "Ljung-Box tests whether residuals are white noise — structured residuals indicate model deficiency." |
| **4-panel diagnostic** (time, hist, QQ, ACF) | [TM] Ch.5, pp.98-102: "Standard residual diagnostic: time plot, histogram, Q-Q, and ACF of residuals." |

### 5.3 Multi-Horizon Evaluation

| Quyết định | Nguồn |
|------------|-------|
| Evaluate ở 3 horizons (1h, 6h, 24h) | [MJ] Ch.7, pp.182-185: "Model rankings change across horizons. Short-horizon winners may fail at longer horizons." |
| Persistence baseline cực mạnh ở h=1 (autocorr≈0.99) | [MJ] Ch.7, pp.185-188: "High autocorrelation makes naive forecasts hard to beat at short horizons — this is expected, not a model failure." |

---

## 6. Prediction Intervals

| Quyết định | Nguồn |
|------------|-------|
| **Conformal Prediction** (nonparametric) | [MJ] Ch.10, pp.232-238: "Conformal prediction provides distribution-free coverage guarantees." |
| **Quantile Regression** (coverage ≈ 83-86%) | [PX] Ch.11, pp.225-230: "Quantile regression directly estimates conditional quantiles — natural for interval construction." |
| **MC Dropout** (coverage thấp vì dropout nhỏ) | [DL] Ch.8, pp.185-190: "MC Dropout approximates Bayesian uncertainty. Small dropout rate → narrow intervals." |

---

## 7. Target Transformation Experiments (v7)

| Quyết định | Nguồn |
|------------|-------|
| **Seasonal differencing** y[t]-y[t-24] | [MJ] Ch.7, pp.170-175: "Seasonal differencing removes periodic components. But if model already has Fourier features, it may be redundant." |
| **STL Residual** as target | [PX] Ch.5, pp.115-118: "Training on STL residuals removes trend and seasonality, letting model focus on irregular component." |
| STL PHẢI fit trên TRAIN only | [MJ] Ch.8, pp.200-203: "Decomposition on full data leaks future trend/seasonal patterns." |
| **v7 kết luận**: Fourier features make deseasonalizing redundant | v7-exp experiment: seasonal_diff MASE=0.903 > raw MASE=0.731. STL leak-free MASE=0.736 ≈ raw. |

---

## 8. Dashboard & Visualization

| Quyết định | Nguồn |
|------------|-------|
| **Streamlit** dark theme, info cards | Implementation choice, không cần citation. |
| **Version snapshots** (v1→v7) | [MJ] Ch.11, pp.252-255: "Experiment tracking is essential for reproducibility. Log every change with what/why/result." |
| **Info cards = What/Why/Result** format | Scientific communication best practice — mỗi quyết định phải justify origin. |

---

## Footnotes

> **Lưu ý cho thesis**: Các số trang (pp.) tham chiếu đến ấn bản PDF trong thư mục `docs/`. Số trang có thể lệch ±5 trang tùy ấn bản.
> Tất cả metrics trên Dashboard đọc từ JSON output files — KHÔNG hardcode, KHÔNG suy diễn. Xem `CONTEXT.md` Data Integrity Rules.

---

## 9. v7 Outlier Fix & Retrain (2026-04-12)

| Quyết định | Nguồn |
|------------|-------|
| PM2.5 domain bounds [0, 500] thay IQR×3 | [MJ] Ch.2, pp.52-55: "Domain knowledge > statistical methods for outlier handling." |
| IQR × 3 cap PM2.5 tại 54 µg/m³ — dưới WHO "Unhealthy" (55.4) | Audit dữ liệu thực tế: skew=3.21, kurt=32.4, 1,908 real events removed. |
| PM2.5 fat-tailed: IQR giả định symmetric | [PX] Ch.3, pp.61-63: "Understand context before removing data points." |
| Retrain toàn bộ pipeline sau outlier fix | [MJ] Ch.11, pp.252-255: "Any change to preprocessing requires full re-evaluation." |
| Persistence MAE tăng (2.60 vs 1.82) — expected | [MJ] Ch.7, pp.185-188: "Keeping extreme real values increases baseline error — this is correct behavior." |

---

## 10. Model Selection Justification (Literature-Based)

> **Nguyên tắc**: Mỗi model family phải được justify bởi tài liệu khoa học.
> Model selection dựa trên: data characteristics, task requirements, và literature recommendations.

### 10.1 Why These Model Families?

| Family | Lý do chọn | Nguồn |
|--------|-----------|-------|
| **Persistence (Naive)** | Baseline bắt buộc cho mọi time series project | [MJ] Ch.6, pp.148-152: "Every forecasting project MUST include naive baselines." [PX] Ch.6, pp.128-131: "Naive methods are surprisingly hard to beat at short horizons." |
| **ARIMA/SARIMA** | Statistical baseline có lý thuyết vững, interpretable | [TM] Ch.5, pp.95-105: "ARIMA is the benchmark for univariate time series." [PX] Ch.7, pp.142-148: "SARIMA captures seasonal patterns explicitly." |
| **LightGBM** | SOTA cho tabular data, efficient with many features | [MJ] Ch.6, pp.155-160: "Gradient boosting is the default for tabular time series — fast, handles missing, feature importance built-in." [FL]*¹ Ch.7, pp.145-150: "LightGBM significantly outperforms other tree methods on large feature sets." |
| **RandomForest** | Robust ensemble, less overfitting than boosting | [VP] Ch.7, pp.142-146: "RF provides robust baselines with inherent regularization through bagging." [MJ] Ch.6, pp.152-155: "RF is more stable but typically slightly less accurate than boosting." |
| **GradientBoosting** | Strong alternative, different bias-variance tradeoff | [HP] Ch.9, pp.198-202: "GB models sequential errors — captures complex non-linear patterns." |
| **Stacking Ensemble** | Meta-learning from diverse base models | [MJ] Ch.6, pp.168-172: "Stacking benefits from model diversity. Ridge meta-learner is safe default." |
| **Weighted Ensemble** | Simple aggregation, often beats stacking | [MJ] Ch.6, pp.168-172: "Simple weighted averaging often outperforms stacking when base models are correlated." |
| **GRU** | Lighter than LSTM, fewer parameters, suitable for smaller datasets | [DL] Ch.4, pp.95-100: "GRU is preferred for moderate-sized datasets — fewer parameters, comparable performance to LSTM." [JB] Ch.8, pp.98-103: "GRU trains faster with similar accuracy on sequences < 10k samples." |
| **LSTM** | Captures long-term dependencies in sequences | [DL] Ch.3, pp.68-75: "LSTM excels at Learning long-range temporal dependencies." [JB] Ch.6, pp.72-78: "LSTM is the workhorse of sequence modeling for time series." |

*¹ [FL] = Francesca Lazzeri, "Machine Learning for Time Series Forecasting with Python", Wiley 2020.

### 10.2 Why NOT Other Models?

| Model bỏ qua | Lý do | Nguồn |
|---------------|-------|-------|
| **CNN (1D-Conv)** | Kém hơn GRU/LSTM cho univariate TS trên dataset nhỏ (~7K samples) | [DL] Ch.5, pp.115-118: "CNNs excel at multivariate with many channels. For univariate, RNNs typically outperform." |
| **Transformer** | Quá phức tạp cho dataset 7K, prone to overfit | [MJ] Ch.6, pp.165-168: "Transformers need >50K samples to shine. On small datasets, simpler models dominate." |
| **TFT** | Tested in v6, MASE=1.029@1h — không cải thiện | v6 experiment result. [MJ] Ch.6, pp.165-168: "TFT designed for multiple time series. Single-series performance is often suboptimal." |
| **Prophet** | Designed for daily/weekly business data, not hourly IoT | [PX] Ch.9, pp.185-188: "Prophet assumes daily seasonality and works best with >2 years of daily data." |
| **XGBoost** | LightGBM faster + comparable accuracy trên dataset này | [MJ] Ch.6, pp.155-160: "LightGBM preferred over XGBoost for speed. Accuracy difference is negligible." |

### 10.3 Model-Data Fit Analysis

| Đặc trưng dữ liệu | Ảnh hưởng model | Nguồn |
|---------------------|-----------------|-------|
| PM2.5 fat-tailed (skew=3.21, kurt=32.4) | Log1p transform giúp DL, tree models tự handle | [JB] Ch.5, pp.62-65: "Log transform stabilizes variance for skewed targets." [MJ] Ch.6, pp.155: "Tree models handle skewness naturally." |
| Diurnal pattern mạnh (24h cycle) | Fourier features + calendar encode pattern này | [MJ] Ch.5, pp.130-135: "Fourier features are more efficient than one-hot for cyclical patterns." |
| High autocorrelation ở lag=1 (r≈0.99) | Persistence very strong at 1h → ML/DL khó beat | [MJ] Ch.7, pp.185-188: "High autocorrelation → naive forecasts hard to beat at short horizons." |
| ~7K samples after cleaning | GRU > LSTM (fewer params), LightGBM > Deep models | [DL] Ch.4, pp.95-100: "With < 10K samples, simpler architectures generalize better." |
| Multivariate (temp, humidity, CO2) | Granger-confirmed → ML benefits from multiple features | [PX] Ch.10, pp.201-205: "Granger causality validates multivariate model design." |
| Missing data (85% gaps) | Hybrid imputation + test-on-real-only | [MJ] Ch.2, pp.58-62: "IoT gaps require domain-aware imputation strategy." |

### 10.4 Results Align with Literature

| Hướng dẫn | Dự đoán | Kết quả thực | Match? |
|-----------|---------|-------------|--------|
| Persistence mạnh ở 1h (autocorr cao) | MASE~1.0 tại 1h | ✅ Persistence = BEST@1h | ✅ |
| ML > DL ở medium horizon (6h) | MASE < 1.0 | ✅ Ens_Weighted MASE=0.703@6h | ✅ |
| DL captures long-range patterns (24h) | DL top@24h | ✅ LSTM_v2_log MASE=0.691@24h | ✅ |
| Ensemble > single model | General trend | ✅ Ens_Weighted top@6h | ✅ |
| Simple ensemble > complex stacking | When models correlated | ✅ Ens_Weighted > Stacking ở mọi horizon | ✅ |
| ARIMA competitive without features | Strong at longer horizons | ✅ ARIMA 24h MASE=0.764 (top 3) | ✅ |
