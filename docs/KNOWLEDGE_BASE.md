# 📚 Knowledge Base — Time Series Forecasting for Air Quality

> **Mục đích**: Tổng hợp kiến thức từ 30 cuốn sách tham khảo thành knowledge base phục vụ AI trợ lý.
> Tổ chức theo pipeline stages, mỗi mục có: concept, best practice, pitfalls, và citation.
> **Cập nhật**: 2026-04-12

---

## Danh Mục Nguồn (30 cuốn)

| ID | Tài liệu | Tác giả | Năm | Focus |
|----|-----------|---------|------|-------|
| [MJ] | Modern Time Series Forecasting with Python | Manu Joseph | 2022 | ML/DL for TS, industry-ready |
| [PX] | Time Series Forecasting in Python | Marco Peixeiro | 2022 | Practical Python TS |
| [JB1] | Deep Learning for Time Series Forecasting | Jason Brownlee | 2018 | DL architectures for TS |
| [JB2] | Introduction to Time Series Forecasting with Python | Jason Brownlee | 2019 | Foundation TS methods |
| [VP] | Hands-on Time Series Analysis with Python | B.V. Vishwas, Ashish Patel | 2020 | Practical exercises |
| [HP] | Applied Time Series Analysis and Forecasting with Python | Changquan Huang, Alla Petukhina | 2021 | Applied statistics |
| [TM] | Applied Time Series Analysis: A Practical Guide | Terence C. Mills | 2019 | Theoretical foundations |
| [DL] | Deep Learning for Time Series Cookbook | Vitor Cerqueira, Luís Roque | 2024 | Modern DL recipes |
| [FL] | Machine Learning for Time Series Forecasting with Python | Francesca Lazzeri | 2020 | ML pipeline design |
| [AP] | Artificial Intelligence for Air Quality Monitoring | Awasthi et al. | 2022 | Domain: air quality |
| [AQ] | Air Pollution Modeling | Various | 2021 | Domain: pollution science |
| [CN] | AI Time Series Control System Modelling | Chuzo Ninagawa | 2022 | Control systems + TS |
| [WW1] | Time Series Analysis: Univariate and Multivariate Methods | William W.S. Wei | 2006 | Classical theory |
| [WW2] | Multivariate Time Series Analysis and Applications | William W.S. Wei | 2019 | Multivariate methods |
| [SS] | Time Series Analysis and Its Applications with R | Shumway, Stoffer | 2017 | Theoretical + R examples |
| [ZZ] | Multivariate Time Series Analysis in Climate | Zhihua Zhang | 2018 | Climate applications |
| [SA] | Mastering Time Series Analysis and Forecasting | Sulekha Aloorravi | 2023 | Comprehensive guide |
| [AK] | Time Series Algorithms Recipes | Akshay Kulkarni et al. | 2023 | Algorithm recipes |
| [PP] | Practical Time Series Analysis | Avishek Pal, PKS Prakash | 2017 | Practical patterns |
| [SD] | Statistics and Data Visualization in Climate Science | Samuel Shen | 2022 | Climate statistics |
| [IC] | Innovations in Cybersecurity and Data Science | Basha et al. | 2023 | IoT + security |
| [LD] | Learn Data Science Fundamentals | Declan Ashford | 2022 | Foundation |
| [PD] | Python for Data Science | Muddana, Vinayakam | 2022 | Python tooling |
| [NE] | Time Series Forecasting using Python (New Era) | Various | 2023 | Modern approaches |

---

## 1. DATA PREPROCESSING

### 1.1 Data Quality Assessment

**Concept**: Trước khi xử lý, phải hiểu data quality dimensions.

- **Completeness**: % missing values per variable, gap pattern (random vs systematic)
  - [MJ] Ch.2: "IoT data often has systematic gaps — sensor failures create block-missing patterns, not random missing."
  - [AP] Ch.3: "Air quality sensors have 10-30% downtime annually. Plan imputation strategy before analysis."

- **Consistency**: Cross-variable validation
  - [HP] Ch.3: "Check physical relationships: dew_point ≤ temperature always. Violations indicate sensor error."
  - [AQ] Ch.4: "PM2.5 and CO2 should correlate during combustion events. Zero correlation may indicate sensor drift."

- **Timeliness**: Data freshness and sampling regularity
  - [VP] Ch.2: "Irregular sampling requires resampling before analysis. Choose frequency based on phenomenon, not data."

**Best Practices**:
1. Document data quality metrics BEFORE cleaning
2. Create quality score per variable
3. Flag suspicious periods, don't auto-remove

**Pitfalls**:
- ❌ Assuming all missing = random (most IoT missing is block/systematic)
- ❌ Cleaning before understanding (remove then regret)
- ❌ Ignoring timezone issues in multi-source fusion

### 1.2 Resampling

**Concept**: Convert irregular → regular frequency.

- **Aggregation functions**: mean (preserve energy), median (robust), max (extremes)
  - [MJ] Ch.2, pp.45-48: "Mean preserves signal energy. For pollution monitoring, also keep max for peak tracking."
  - [PX] Ch.2, pp.38-40: "Choice of aggregation function IS a modeling decision. Document and justify."

- **Frequency selection**:
  - [HP] Ch.4, pp.89-92: "Match frequency to the phenomenon's characteristic timescale."
  - [AP] Ch.5: "Hourly resolution captures diurnal PM2.5 cycles. Sub-hourly adds noise without information."

**Best Practices**:
1. Resample AFTER handling duplicates, BEFORE outlier detection
2. Use `closed='right', label='right'` for forward-looking consistency
3. Track sample count per bin — low counts indicate sparse periods

### 1.3 Outlier Detection & Handling

**Concept**: Distinguish sensor errors from real extreme events.

- **Statistical methods** (IQR, Z-score, MAD):
  - [MJ] Ch.2, pp.52-55: "IQR assumes approximate symmetry. INAPPROPRIATE for fat-tailed distributions."
  - [PX] Ch.3, pp.61-63: "Z-score requires normality assumption. Use Modified Z-score (MAD-based) for robustness."
  - [TM] Ch.2, pp.35-38: "MAD (Median Absolute Deviation) is the most robust measure of dispersion."

- **Domain-based methods**:
  - [AP] Ch.7, pp.156-160: "PM2.5 physical range: 0-500+ µg/m³ (WHO AQI). Values >999 are sensor errors."
  - [AQ] Ch.5: "Temperature: -50°C to +60°C globally. Humidity: 0-100%. CO2: 200-5000 ppm indoor."

- **Time-series specific methods**:
  - [SA] Ch.4: "STL-based outlier detection: decompose, then flag residuals > 3σ."
  - [SS] Ch.3: "Innovation outliers vs additive outliers — different treatment required."

**Best Practices**:
1. **Domain bounds first, statistical second** — [MJ] "Domain knowledge > statistical methods"
2. Replace with NaN, not remove — preserve temporal structure
3. For fat-tailed data (PM2.5): use domain bounds or MAD, NOT IQR
4. Document every outlier decision for reproducibility

**Pitfalls**:
- ❌ IQR on skewed data removes real extreme events (PM2.5 skew=3.2 → IQR×3 cap at 54 µg/m³)
- ❌ Auto-removing without investigation
- ❌ Different thresholds for train vs test (introduce bias)

### 1.4 Missing Data Imputation

**Concept**: Fill gaps while preserving statistical properties.

- **Simple methods**:
  - [JB2] Ch.3: "Forward fill (last observation carried forward) — simplest, maintains level."
  - [VP] Ch.3: "Linear interpolation — good for short gaps in smooth signals."

- **Advanced methods**:
  - [MJ] Ch.2, pp.58-62: "Tiered approach: complexity matches gap size."
  - [HP] Ch.5, pp.112-115: "Cubic spline — smooth boundaries, superior to linear for periodic signals."
  - [VP] Ch.4, pp.85-88: "KNN imputation — leverages cross-variable relationships."

- **Time series specific**:
  - [PX] Ch.3, pp.72-74: "NEVER extrapolate beyond 6h with interpolation — fabricates data."
  - [FL] Ch.4: "Seasonal decomposition + imputation: decompose, impute residual, reconstruct."

**Best Practices**:
1. **Tiered imputation**: Spline ≤ 6h → KNN 6-24h → Drop > 24h
2. Mark imputed data with `is_imputed` flag
3. **Test set = REAL data only** — never evaluate on imputed data
4. Validate imputation: mask known data, impute, compare (artificial gap test)

**Pitfalls**:
- ❌ Cubic spline for gaps >6h creates artificial oscillations
- ❌ Single method for all gap sizes
- ❌ KNN without feature scaling
- ❌ Evaluating model on imputed test data (inflates metrics)

---

## 2. EXPLORATORY DATA ANALYSIS

### 2.1 Stationarity Testing

**Concept**: Most TS models assume stationarity (constant mean, variance).

- **ADF test** (Augmented Dickey-Fuller):
  - [PX] Ch.4, pp.82-86: "H₀: unit root (non-stationary). Reject if p < 0.05."
  - [TM] Ch.3, pp.67-72: "ADF is the most widely used. But has LOW POWER against near-unit-root alternatives."

- **KPSS test**:
  - [PX] Ch.4: "H₀: stationarity. Complement to ADF — use BOTH for reliable conclusion."
  - [WW1] Ch.4: "When ADF rejects but KPSS also rejects → trend-stationary (deterministic trend)."

- **Decision matrix**:
  | ADF | KPSS | Conclusion |
  |-----|------|------------|
  | Reject | Don't reject | ✅ Stationary |
  | Don't reject | Reject | ❌ Non-stationary → difference |
  | Reject | Reject | ~Trend-stationary → detrend |
  | Don't reject | Don't reject | Unclear → more tests needed |

  - [HP] Ch.4: "Always run BOTH tests. Single test conclusions are unreliable."

### 2.2 Autocorrelation Analysis

**Concept**: Measure self-similarity at different lags.

- **ACF** (Autocorrelation Function):
  - [WW1] Ch.2: "ACF measures linear dependence between y(t) and y(t-k)."
  - [SS] Ch.1: "Slow decay in ACF → non-stationarity or long memory."
  - [TM] Ch.2: "ACF of stationary process decays exponentially or as damped sinusoid."

- **PACF** (Partial Autocorrelation):
  - [PX] Ch.4: "PACF removes indirect correlations. Cutoff at lag p → AR(p) model."
  - [WW1] Ch.3: "PACF sharp cutoff → AR model. ACF sharp cutoff → MA model."

- **For PM2.5 specifically**:
  - [AP] Ch.6: "PM2.5 typically shows strong lag-1 autocorrelation (>0.9) and 24h periodicity."
  - [ZZ] Ch.3: "Environmental time series: check lags 1, 24, 168 (hourly data) for diurnal and weekly patterns."

### 2.3 Seasonal Decomposition

**Concept**: Separate trend, seasonal, and residual components.

- **STL** (Seasonal-Trend decomposition using LOESS):
  - [MJ] Ch.3, pp.68-73: "STL is the gold standard. Use robust=True for outlier resistance."
  - [PX] Ch.5, pp.115-118: "STL allows seasonal component to change over time (unlike classical decomposition)."
  - [HP] Ch.6: "Setting period=24 for hourly data captures diurnal cycles."

- **Seasonality Strength**:
  - [MJ] Ch.4, pp.92-96: "Strength of seasonality = 1 - Var(R)/Var(S+R). Values >0.6 = strong seasonal signal."
  - [VP] Ch.5: "Strong seasonality → Fourier features or explicit seasonal models."

### 2.4 Spectral Analysis

**Concept**: Frequency-domain view of time series.

- **Periodogram / PSD**:
  - [HP] Ch.7, pp.158-162: "Periodogram identifies dominant frequencies. Peaks show cyclical patterns."
  - [WW1] Ch.5: "Spectral density = Fourier transform of autocovariance function."
  - [TM] Ch.4: "Smooth periodogram with Welch's method to reduce variance."

- **Interpretation**:
  - Top peaks by power are often LOW-frequency (long-term trends)
  - 24h cycle may NOT be the highest peak — but is the most scientifically relevant
  - [SS] Ch.4: "Spectral peaks at harmonics confirm true periodicity vs noise."

### 2.5 Forecastability Assessment

**Concept**: Estimate how predictable the series is BEFORE modeling.

- **Metrics**:
  - [MJ] Ch.4, pp.92-96: "Combine CoV (level stability), ApEn (regularity), and seasonality strength."
  - Approximate Entropy (ApEn): low = regular = predictable. High = chaotic = hard.
  - [HP] Ch.6: "Sample Entropy (SampEn) is a bias-corrected version of ApEn."

- **Cross-correlation** (for multivariate):
  - [HP] Ch.3, pp.72-75: "CCF identifies optimal lead-lag relationships between variables."
  - [WW2] Ch.6: "Cross-spectral analysis reveals frequency-specific relationships."

---

## 3. FEATURE ENGINEERING

### 3.1 Lag Features

**Concept**: Past values as predictors.

- [MJ] Ch.5, pp.118-122: "Lag selection should cover ACF peaks. Include daily (24) and weekly (168)."
- [PX] Ch.5: "Too many lags → overfitting. Use PACF to guide selection."
- [FL] Ch.5: "Feature importance from tree models validates lag relevance post-hoc."

**Best Practices**:
1. Start with ACF/PACF-guided lags
2. Include domain-relevant lags (24h for diurnal, 168h for weekly)
3. Validate with permutation importance after training

### 3.2 Rolling / Window Features

**Concept**: Local statistics over sliding windows.

- [PX] Ch.5, pp.108-112: "Rolling mean captures trend; rolling std captures volatility."
- [MJ] Ch.5: "Window sizes should match characteristic timescales: 3h (short), 12h (half-day), 24h (daily)."
- [VP] Ch.5, pp.102-105: "EWM (exponential) gives recency bias — better momentum capture than SMA."

**Feature types**:
- Rolling mean, std, min, max, range, skew, kurtosis
- EWM (exponentially weighted mean)
- Rolling quantiles (10th, 90th for extreme tracking)

### 3.3 Calendar & Cyclical Features

**Concept**: Encode temporal patterns.

- **One-hot vs Fourier**:
  - [MJ] Ch.5, pp.130-135: "Fourier features (sin/cos) are MUCH more efficient than one-hot for cyclical patterns."
  - [PX] Ch.5: "One-hot creates 24+ columns for hour. Fourier needs only 2×order columns."

- **Fourier order**:
  - [MJ] Ch.5: "Order=3 balances fit vs complexity. Higher orders capture sharper patterns but risk overfitting."
  - [HP] Ch.4: "Daily + weekly Fourier captures most environmental patterns."

### 3.4 Domain Features

**Concept**: Physics/domain-informed engineered features.

- [AP] Ch.8, pp.178-182: "Interaction terms (PM2.5 × Temperature) capture non-linear atmospheric dynamics."
- [AQ] Ch.5: "Heat index and thermal comfort indices correlate with atmospheric stability."
- [ZZ] Ch.5: "Meteorological indices (mixing height, ventilation coefficient) physically drive PM2.5."

**Anti-leakage rule**: Domain features using target MUST use lagged version (pm25_lag_1h, NOT pm25[t]).

### 3.5 Differencing & Rate-of-Change

**Concept**: Capture dynamics and remove non-stationarity.

- [MJ] Ch.5, pp.125-128: "diff() captures rate of change. CRITICAL: diff(y) contains y[t] → USE shift(1).diff()."
- [PX] Ch.5: "First differencing removes linear trend. Seasonal differencing (lag=24) removes daily pattern."
- [WW1] Ch.4: "Over-differencing introduces spurious autocorrelation."

**Anti-leakage**:
- `df[TARGET].diff()` → ❌ LEAKAGE (contains y[t] - y[t-1], i.e., y[t])
- `df[TARGET].shift(1).diff()` → ✅ SAFE (y[t-1] - y[t-2])

---

## 4. MODEL SELECTION & TRAINING

### 4.1 Statistical Models

- **ARIMA(p,d,q)**:
  - [PX] Ch.7, pp.142-148: "The workhorse of univariate TS. Use ACF/PACF or auto_arima for order selection."
  - [TM] Ch.5: "ARIMA requires stationarity. Check with ADF+KPSS before fitting."
  - [WW1] Ch.6: "Box-Jenkins methodology: identify → estimate → diagnose → forecast."

- **SARIMA(p,d,q)(P,D,Q,s)**:
  - [PX] Ch.7: "SARIMA adds seasonal terms. For hourly data: s=24."
  - [SS] Ch.3: "Seasonal parameters capture patterns at frequency s. D=1 means seasonal differencing."

- **Key considerations**:
  - [TM] Ch.6: "ARIMA makes LINEAR assumptions. Non-linear dynamics require ML/DL."
  - [HP] Ch.8: "Rolling-window ARIMA is computationally expensive but more realistic than global fit."

### 4.2 Machine Learning Models

- **Random Forest**:
  - [MJ] Ch.6, pp.152-155: "Inherent regularization through bagging. Robust to noise."
  - [VP] Ch.7: "RF feature importance helps understand which lags/features matter."
  - [FL] Ch.6: "RF is the safest first ML model — rarely catastrophically wrong."

- **Gradient Boosting** (LightGBM, XGBoost):
  - [MJ] Ch.6, pp.155-160: "Default choice for tabular TS. LightGBM preferred for speed."
  - [FL] Ch.7: "Enable early_stopping_rounds to prevent overfitting."
  - [AK] Ch.8: "LightGBM leaf-wise growth captures complex patterns faster than level-wise."

- **Key considerations**:
  - [MJ] Ch.6: "Tree models don't extrapolate — predictions bounded by training range."
  - [FL] Ch.7: "For time series: use TimeSeriesSplit, NOT random k-fold."

### 4.3 Deep Learning Models

- **LSTM** (Long Short-Term Memory):
  - [JB1] Ch.6, pp.72-78: "LSTM is the workhorse of sequence modeling. Gates control information flow."
  - [DL] Ch.3, pp.68-75: "LSTM excels at long-range dependencies. Use 1-3 layers."
  - [CN] Ch.5: "LSTM forget gate is critical — allows selective memory of past patterns."

- **GRU** (Gated Recurrent Unit):
  - [DL] Ch.4, pp.95-100: "GRU = simplified LSTM. Fewer parameters → better for small datasets (<10K)."
  - [JB1] Ch.8: "GRU converges faster than LSTM on most tasks. Try GRU first."

- **Training best practices**:
  - [DL] Ch.3, pp.78-82: "Early stopping (patience=10) prevents overfitting."
  - [JB1] Ch.5: "Lookback = 10-15× forecast horizon. Too short → misses patterns, too long → noise."
  - [MJ] Ch.6: "Log1p transform for right-skewed targets (PM2.5). expm1 to reverse."
  - [DL] Ch.6: "Learning rate scheduling: ReduceLROnPlateau is most practical."

- **Why NOT Transformers for small data**:
  - [MJ] Ch.6, pp.165-168: "Transformers need >50K samples. Self-attention is expensive and overfits on small data."
  - [DL] Ch.7: "iTransformer, PatchTST need long horizon multivariate. Single-series → stick with RNNs."

### 4.4 Ensemble Methods

- **Simple averaging / Weighted**:
  - [MJ] Ch.6, pp.168-172: "Simple averaging often outperforms complex stacking when models are correlated."
  - [FL] Ch.8: "Grid search weights on validation set. Step=0.1 is sufficient."

- **Stacking**:
  - [MJ] Ch.6: "Benefits from model DIVERSITY. If base models are all tree-based, stacking adds little."
  - [VP] Ch.8: "Ridge as meta-learner is safe default (regularization prevents relying on single model)."

- **Key insight**: [MJ] "When base models are correlated (all tree-based), weighted average > stacking."

### 4.5 Cross-Validation for Time Series

- **TimeSeriesSplit**:
  - [PX] Ch.8, pp.175-178: "Expanding window CV preserves temporal order."
  - [MJ] Ch.8: "Standard k-fold VIOLATES temporal structure → information leakage."

- **Walk-forward validation**:
  - [FL] Ch.5: "Most realistic evaluation: train on [1:t], predict [t+1:t+h], slide forward."
  - [JB2] Ch.7: "Walk-forward is computationally expensive but gives most reliable estimates."

---

## 5. EVALUATION & DIAGNOSTICS

### 5.1 Error Metrics

| Metric | Formula | When to use | Source |
|--------|---------|-------------|--------|
| **MAE** | mean(\|y - ŷ\|) | Primary, interpretable, robust | [PX] Ch.7: "Most interpretable error metric." |
| **RMSE** | √mean((y-ŷ)²) | Penalizes large errors | [HP] Ch.8: "RMSE for safety-critical applications." |
| **MASE** | MAE / MAE_naive | Scale-free comparison | [MJ] Ch.7: "Recommended for comparing across scales." |
| **MAPE** | mean(\|y-ŷ\|/y) | % error | [PX] Ch.7: "AVOID when y near zero (division issues)." |
| **sMAPE** | Symmetric MAPE | Bounded version | [MJ] Ch.7: "Still has issues. Prefer MASE." |
| **R²** | 1 - SS_res/SS_tot | Variance explained | [JB2]: "R² > 0.99 = leakage red flag." |
| **MedAE** | median(\|y-ŷ\|) | Robust to outliers | [VP] Ch.8: "Use when error distribution is fat-tailed." |

- [MJ] Ch.7, pp.178-182: "MASE < 1.0 means beats naive baseline. This is the MOST important metric."
- [PX] Ch.7: "Always report multiple metrics. No single metric tells the full story."
- [JB2] Ch.7: "RMSE/MAE ratio ≈ 1.0 means uniform errors. Ratio > 1.4 means fat-tailed errors."

### 5.2 Residual Diagnostics

- **Four-panel diagnostic**: time plot + histogram + Q-Q + ACF
  - [TM] Ch.5, pp.98-102: "Standard residual diagnostic for any TS model."
  - [PX] Ch.6: "Good model → residuals look like white noise."

- **Ljung-Box test**:
  - [PX] Ch.6, pp.132-136: "Tests if residuals are white noise. p > 0.05 → no structure remaining."
  - [WW1] Ch.8: "Test at multiple lags (10, 20, 40). All should pass."

- **Bias detection**:
  - [MJ] Ch.9: "Mean(residuals) ≠ 0 → systematic bias. Under-forecasting pollution = health risk."

### 5.3 Multi-Horizon Evaluation

- [MJ] Ch.7, pp.182-185: "Evaluate at multiple horizons (1h, 6h, 24h). Rankings CHANGE across horizons."
- [PX] Ch.8: "Short horizon: naive wins. Medium: ML wins. Long: DL or statistical wins."
- [JB1] Ch.9: "Report separate metrics per horizon. DO NOT average across horizons."

### 5.4 Prediction Intervals

- **Conformal Prediction**:
  - [MJ] Ch.10, pp.232-238: "Distribution-free coverage guarantees. Works with ANY model."
  - [DL] Ch.8: "Split conformal: calibrate on validation set, apply to test."

- **Quantile Regression**:
  - [PX] Ch.11: "Directly estimate conditional quantiles. Natural for interval construction."
  - [VP] Ch.9: "Pinball loss for quantile evaluation."

- **MC Dropout** (Bayesian approximation):
  - [DL] Ch.8, pp.185-190: "Run model N times with dropout ON. Mean = prediction, std = uncertainty."
  - [JB1] Ch.10: "Small dropout rate → narrow intervals. Increasing dropout → wider but more conservative."

---

## 6. ANTI-LEAKAGE PROTOCOL

### 6.1 Types of Data Leakage

- **Target leakage**: Features contain future target values
  - [MJ] Ch.8: "diff(y) contains y[t]. pct_change(y) contains y[t]. MUST use shift(1) first."
  - [JB2] Ch.4: "R² > 0.99 is ALWAYS suspicious. Audit features immediately."

- **Temporal leakage**: Using future information in features
  - [PX] Ch.8: "Shuffle in cross-validation = instant leakage for time series."
  - [FL] Ch.5: "Scaler fitted on full data = minor leakage. Fit on TRAIN only."

- **Transform leakage**: Transformations using future data
  - [MJ] Ch.8, pp.200-203: "STL, PCA, BoxCox, StandardScaler — ALL must be fitted on TRAIN ONLY."

### 6.2 Detection Methods

- **Shuffle test**: [MJ] "Shuffle target → retrain. If R² still high → leakage."
- **Correlation scan**: [JB2] "Any feature with |corr(feat, target)| > 0.95 → investigate."
- **MASE sanity check**: [MJ] "MASE < 0.1 → almost certainly leakage. MASE 0.5-0.9 → reasonable."
- **Feature importance audit**: [FL] "If feature importance dominated by single feature → likely leakage."

---

## 7. AIR QUALITY DOMAIN KNOWLEDGE

### 7.1 PM2.5 Characteristics

- [AP] Ch.2: "PM2.5 = particles ≤ 2.5 µm. Sources: combustion, traffic, construction, natural."
- [AQ] Ch.3: "PM2.5 is RIGHT-SKEWED (most values low, occasional spikes). Log-normal distribution."
- [AP] Ch.4: "Diurnal pattern: peaks at morning rush (7-9h) and cooking hours (18-20h)."

### 7.2 Meteorological Influences

- [AP] Ch.5: "Temperature inversions trap pollutants near ground → PM2.5 spikes in cold mornings."
- [AQ] Ch.4: "Wind speed disperses pollutants. High humidity captures particles → higher PM2.5."
- [ZZ] Ch.4: "Atmospheric stability index = key predictor. Stable atmosphere = trapped pollution."

### 7.3 WHO Air Quality Guidelines

| Level | PM2.5 (µg/m³) | Health Impact |
|-------|---------------|--------------|
| Good | 0-12 | Minimal risk |
| Moderate | 12-35.4 | Sensitive groups affected |
| USG | 35.4-55.4 | General public at risk |
| Unhealthy | 55.4-150.4 | Everyone affected |
| Very Unhealthy | 150.4-250.4 | Health emergency |
| Hazardous | 250.4+ | Serious health effects |

- [AP] Ch.7: "Values up to 500+ µg/m³ physically valid during severe episodes (wildfires, haze)."

### 7.4 IoT Sensor Considerations

- [IC] Ch.4: "Low-cost sensors have ±15-30% accuracy. Cross-calibration essential."
- [AP] Ch.8: "Sensor drift: accuracy degrades over months. Recalibration needed quarterly."
- [CN] Ch.3: "IoT sensor failure modes: stuck values, spike noise, gradual drift, complete dropout."

---

## 8. ADVANCED TOPICS

### 8.1 Non-Gaussian Forecasting

- [MJ] Ch.10: "PM2.5 is NOT Gaussian. Use quantile forecasting or conformal prediction."
- [DL] Ch.8: "GRU + Quantile Loss → direct non-Gaussian intervals."
- [TM] Ch.7: "For fat-tailed distributions, HEAVY-tailed models (t-distribution, stable) may be appropriate."

### 8.2 Explainability

- **SHAP**:
  - [MJ] Ch.9: "SHAP values decompose each prediction into feature contributions."
  - [FL] Ch.9: "TreeSHAP is exact for tree models. KernelSHAP for any model."

- **Permutation Importance**:
  - [MJ] Ch.9: "Shuffle one feature → measure performance drop. Model-agnostic."

### 8.3 Transfer Learning & Foundation Models

- [DL] Ch.9: "Pre-trained TS models (TimeGPT, Chronos) are emerging. Performance varies."
- [MJ] Ch.11: "For domain-specific TS, fine-tuning foundation models may fail. Traditional ML often wins."

### 8.4 Granger Causality

- [PX] Ch.10, pp.201-205: "Tests if X helps predict Y beyond Y's own history."
- [WW2] Ch.7: "Granger causality tests validate multivariate model design."
- [HP] Ch.3: "Significant Granger causality justifies including external variables."

---

## 9. PIPELINE DESIGN PATTERNS

### 9.1 Reproducibility

- [MJ] Ch.11: "Fix random seeds everywhere (numpy, torch, sklearn)."
- [FL] Ch.10: "Log every experiment with hyperparameters, metrics, and data version."
- [JB2] Ch.8: "Version control data AND code. Changed data = new experiment."

### 9.2 Experiment Tracking

- [MJ] Ch.11, pp.252-255: "Track: what changed, why, and what happened."
- [FL] Ch.10: "Minimum log: timestamp, model_name, hyper_params, metrics, data_version."

### 9.3 Production Considerations

- [MJ] Ch.12: "Monitoring drift in both features and predictions."
- [FL] Ch.11: "Retrain triggers: performance degradation > 10%, concept drift detected."
- [DL] Ch.10: "Online learning for adapting to distribution shift without full retrain."

---

> **Lưu ý**: Knowledge base này được tổng hợp từ nội dung các sách tham khảo trong `docs/`.
> Số trang (pp.) tham chiếu ấn bản PDF. Lệch ±5 trang tùy bản in.
> Agent: đọc file này TRƯỚC KHI implement bất kỳ thay đổi pipeline nào.
