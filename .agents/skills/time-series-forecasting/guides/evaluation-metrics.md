# Evaluation & Metrics Guide — Time Series Forecasting

> Bổ sung chi tiết cho SKILL.md §9. Multi-horizon evaluation, statistical tests, confidence intervals.

---

## 1. Multi-Horizon Evaluation Matrix

> [!IMPORTANT]
> **PHẢI đánh giá TỪNG horizon riêng biệt.** Model tốt ở 1h có thể tệ ở 24h.

### Evaluation Table Template

| Metric | Horizon 1h | Horizon 6h | Horizon 24h |
|--------|-----------|-----------|------------|
| MAE | — | — | — |
| RMSE | — | — | — |
| MASE | — | — | — |
| R² | — | — | — |

### Implementation

```python
import pandas as pd
from typing import Dict, List

def evaluate_multi_horizon(
    y_true_dict: Dict[int, np.ndarray],
    y_pred_dict: Dict[int, np.ndarray],
    y_naive_dict: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """Evaluate model across multiple forecast horizons.

    Args:
        y_true_dict: {horizon: y_true_array}
        y_pred_dict: {horizon: y_pred_array}
        y_naive_dict: {horizon: naive_predictions}

    Returns:
        DataFrame with metrics × horizons
    """
    results = []
    for horizon in sorted(y_true_dict.keys()):
        y_true = y_true_dict[horizon]
        y_pred = y_pred_dict[horizon]
        y_naive = y_naive_dict[horizon]

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mase = mae / np.mean(np.abs(y_true - y_naive))
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

        # MAPE — only when y_true > threshold (avoid div/0)
        mask = y_true > 1.0  # Skip near-zero values
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float("nan")

        results.append({
            "Horizon": f"{horizon}h",
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MASE": round(mase, 4),
            "MAPE": round(mape, 2) if not np.isnan(mape) else "N/A",
            "R²": round(r2, 4),
            "Pass": "✅" if mase < 1.0 else "❌",
        })

    return pd.DataFrame(results).set_index("Horizon")
```

---

## 2. Diebold-Mariano Test

> [!IMPORTANT]
> **Khi MAE difference < 10% giữa 2 models → PHẢI chạy DM test để xác nhận significance.**

### Khi nào dùng?

- Model A: MAE = 12.5, Model B: MAE = 13.0
- Difference = 3.8% → **Có thể do noise!**
- DM test: p < 0.05 → Model A thực sự tốt hơn ✅
- DM test: p ≥ 0.05 → Không đủ evidence, chọn model đơn giản hơn

### Implementation

```python
from scipy import stats
import numpy as np

def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    loss_func: str = "mae",
    h: int = 1,
) -> Dict[str, float]:
    """Diebold-Mariano test for forecast comparison.

    H0: Two forecasts have the same accuracy.
    H1: Forecast 1 is more accurate than Forecast 2.

    Args:
        y_true: Actual values
        y_pred_1: Predictions from model 1 (expected better)
        y_pred_2: Predictions from model 2
        loss_func: "mae" or "mse"
        h: Forecast horizon (for HAC correction)

    Returns:
        {"DM_statistic": float, "p_value": float, "significant": bool}
    """
    if loss_func == "mae":
        d = np.abs(y_true - y_pred_1) - np.abs(y_true - y_pred_2)
    elif loss_func == "mse":
        d = (y_true - y_pred_1)**2 - (y_true - y_pred_2)**2
    else:
        raise ValueError(f"Unknown loss: {loss_func}")

    n = len(d)
    d_mean = np.mean(d)

    # HAC variance (Newey-West for h-step ahead)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += gamma_k
    d_var = (gamma_0 + 2 * gamma_sum) / n

    dm_stat = d_mean / np.sqrt(max(d_var, 1e-10))
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n-1))  # two-sided

    return {
        "DM_statistic": round(dm_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "better_model": "Model 1" if d_mean < 0 else "Model 2",
    }
```

### Usage

```python
result = diebold_mariano_test(y_true, y_pred_xgb, y_pred_rf, loss_func="mae", h=24)
logger.info(f"DM Test: stat={result['DM_statistic']}, p={result['p_value']}")

if result["significant"]:
    logger.info(f"  → {result['better_model']} is significantly better ✅")
else:
    logger.info(f"  → No significant difference → choose simpler model")
```

---

## 3. Confidence Intervals for Forecasts

> [!IMPORTANT]
> **Mọi forecast PHẢI kèm 95% Confidence Interval.** Point forecast alone là KHÔNG ĐỦ.

### Bootstrap CI (Universal — works with any model)

```python
def forecast_with_ci(
    model,
    X_test: np.ndarray,
    n_bootstrap: int = 100,
    ci: float = 0.95,
) -> Dict[str, np.ndarray]:
    """Generate predictions with bootstrap confidence intervals.

    Returns:
        {"mean": array, "lower": array, "upper": array}
    """
    y_pred = model.predict(X_test)

    # Bootstrap residuals from training set
    residuals = model.train_residuals_  # Stored during training
    alpha = (1 - ci) / 2

    bootstrap_predictions = []
    for _ in range(n_bootstrap):
        noise = np.random.choice(residuals, size=len(y_pred), replace=True)
        bootstrap_predictions.append(y_pred + noise)

    bootstrap_predictions = np.array(bootstrap_predictions)

    return {
        "mean": y_pred,
        "lower": np.percentile(bootstrap_predictions, alpha * 100, axis=0),
        "upper": np.percentile(bootstrap_predictions, (1 - alpha) * 100, axis=0),
    }
```

### Statsmodels CI (for ARIMA/SARIMAX)

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y_train, order=(1, 1, 1))
results = model.fit()

# Forecast with built-in CI
forecast = results.get_forecast(steps=24)
forecast_df = forecast.summary_frame(alpha=0.05)
# Columns: mean, mean_se, mean_ci_lower, mean_ci_upper
```

---

## 4. MAPE Limitations

> [!WARNING]
> **MAPE undefined khi y_true = 0.** PM2.5 dataset có min=0, median=10 → MAPE sẽ inflate near-zero values.

| Trường hợp | MAPE | Giải pháp |
|------------|------|----------|
| y_true = 0 | Division by zero | Skip hoặc dùng sMAPE |
| y_true = 1, y_pred = 2 | **100%** (misleading!) | Dùng MAE thay vì MAPE |
| y_true = 100, y_pred = 110 | 10% (hợp lý) | MAPE OK |

### Symmetric MAPE (sMAPE) — thay thế an toàn

```python
def smape(y_true, y_pred):
    """Symmetric MAPE — handles near-zero values better."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid 0/0
    mask = denominator > 0
    return np.mean(numerator[mask] / denominator[mask]) * 100
```

---

## 5. Ensemble Validation Rules

> [!WARNING]
> **Stacking ensemble PHẢI dùng Out-of-Fold predictions.** Không train meta-learner trên cùng data.

### Out-of-Fold Stacking

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def generate_oof_predictions(models, X, y, n_splits=5):
    """Generate out-of-fold predictions for stacking."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_predictions = np.zeros((len(X), len(models)))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        for i, model in enumerate(models):
            model.fit(X_train, y_train)
            oof_predictions[val_idx, i] = model.predict(X_val)

    return oof_predictions  # Use as features for meta-learner
```

### Ensemble Selection Criteria

| Điều kiện | Check | Threshold |
|-----------|-------|-----------|
| Model diversity | corr(pred_i, pred_j) | < 0.7 |
| Minimum models | count | ≥ 3 cho stacking |
| Individual performance | MASE per model | < 1.5 (không quá tệ) |
| Ensemble improvement | MASE_ensemble vs best_single | < 0.95 × best |

```python
def check_ensemble_diversity(predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Check if models are diverse enough for ensemble."""
    pred_df = pd.DataFrame(predictions_dict)
    corr = pred_df.corr()

    high_corr = []
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if abs(corr.iloc[i, j]) > 0.7:
                high_corr.append(f"{corr.index[i]}-{corr.columns[j]}: {corr.iloc[i,j]:.2f}")

    if high_corr:
        logger.warning(f"High correlation pairs (>0.7): {high_corr}")
        logger.warning("Consider removing one model from each pair")
    else:
        logger.info("All model pairs have correlation < 0.7 ✅ — good diversity")

    return corr
```

---

## 6. Statistical Test Interpretation Table (EDA)

| Test | Null Hypothesis | p < 0.05 nghĩa là | Action |
|------|----------------|-------------------|--------|
| **ADF** (Augmented Dickey-Fuller) | Series has unit root (non-stationary) | Stationary ✅ → có thể dùng ARMA | Nếu p > 0.05 → difference (d=1) |
| **KPSS** | Series is stationary | Non-stationary ❌ → cần difference | Nếu p > 0.05 → stationary ✅ |
| **ADF + KPSS** cùng nói stationary | — | Strong evidence of stationarity | Proceed with ARMA |
| **Ljung-Box** | No autocorrelation in residuals | Residuals ARE correlated ❌ | Model cần thêm AR/MA terms |
| **Jarque-Bera** | Residuals are normal | NOT normal ❌ | Consider robust methods |
| **Granger Causality** | X does NOT Granger-cause Y | X helps predict Y ✅ | Include X as feature |
| **Breusch-Pagan** | Homoscedastic errors | Heteroscedastic ❌ | Use robust SEs or transform |

> **Nhớ**: ADF và KPSS có null hypothesis **NGƯỢC NHAU**. Chạy CẢ HAI để confirm.

---

## 7. Explainable AI (SHAP)

Sau mỗi model tốt nhất → chạy SHAP để giải thích:
1. Summary plot (feature importance tổng thể)
2. Dependence plot (ảnh hưởng từng feature)
3. Force plot (giải thích 1 dự báo cụ thể)

---

## 8. Tổng quan Metrics

| Metric | Ý nghĩa | Vai trò |
|--------|---------|---------|
| **MAE** | Sai số tuyệt đối trung bình | **Primary metric** |
| **RMSE** | Phạt nặng outlier errors | Secondary |
| **MAPE** | % sai số | Interpretability |
| **MASE** | So với naive baseline | **BẮT BUỘC** benchmark |
| **R²** | % variance giải thích | Overall fit |

**Quy tắc**:
- MASE < 1.0 ✅ (tốt hơn naive) | MASE ≥ 1.0 ❌ (cần cải thiện)
- **Multi-Horizon**: PHẢI đánh giá TỪNG horizon (1h, 6h, 24h) riêng biệt
- **Confidence Intervals**: Mọi forecast PHẢI kèm **95% CI** (bootstrap hoặc built-in)
- **Diebold-Mariano Test**: Khi MAE difference < 10% giữa 2 models → PHẢI chạy DM test (p < 0.05)
