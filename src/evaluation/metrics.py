"""Evaluation metrics for time series forecasting.

Implements SKILL.md §9 metrics: MAE, RMSE, MAPE, MASE, R².
Plus sMAPE for near-zero safety (evaluation-metrics.md §4).
"""

import numpy as np
from loguru import logger


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error — Primary metric per SKILL.md."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — penalizes large errors."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.0) -> float:
    """Mean Absolute Percentage Error.

    Skips near-zero values to avoid division by zero (evaluation-metrics.md §4).

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        threshold: Minimum y_true to include (avoid near-zero inflation).

    Returns:
        MAPE in percentage.
    """
    mask = y_true > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — safer than MAPE for near-zero values.

    Per evaluation-metrics.md §4.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(numerator[mask] / denominator[mask]) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
    """Mean Absolute Scaled Error — BẮT BUỘC benchmark per SKILL.md.

    MASE < 1.0 → model is better than naive baseline.
    MASE = 1.0 → same as naive.
    MASE > 1.0 → worse than naive (model is useless).

    Args:
        y_true: Actual values.
        y_pred: Model predictions.
        y_naive: Naive baseline predictions.

    Returns:
        MASE score.
    """
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))
    if mae_naive < 1e-10:
        logger.warning("MASE: naive MAE is near zero, returning inf")
        return float("inf")
    return float(mae_model / mae_naive)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Forecast Bias — over- or under-forecasting indicator.

    Ref: Manu Joseph Ch.4 p.80.
    FB > 0 → model over-forecasts (predicts too high)
    FB < 0 → model under-forecasts (predicts too low) ← dangerous for PM2.5
    FB ≈ 0 → unbiased (ideal)

    Returns:
        Forecast bias as fraction.
    """
    total_actual = np.sum(y_true)
    if abs(total_actual) < 1e-10:
        return float("nan")
    return float((np.sum(y_pred) - total_actual) / total_actual)


def medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Error — robust to outliers.

    More robust than MAE for fat-tailed distributions like PM2.5.
    """
    return float(np.median(np.abs(y_true - y_pred)))


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray,
    model_name: str = "Model",
    horizon: int | None = None,
) -> dict[str, float | str]:
    """Evaluate a forecast with all metrics.

    Args:
        y_true: Actual values.
        y_pred: Model predictions.
        y_naive: Naive baseline predictions.
        model_name: Name for logging.
        horizon: Forecast horizon (h) for logging.

    Returns:
        Dictionary of metric name → value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_naive = np.asarray(y_naive, dtype=float)

    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)

    results: dict[str, float | str] = {
        "model": model_name,
        "mae": round(mae_val, 4),
        "rmse": round(rmse_val, 4),
        "mape": round(mape(y_true, y_pred), 2),
        "smape": round(smape(y_true, y_pred), 2),
        "mase": round(mase(y_true, y_pred, y_naive), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
        "medae": round(medae(y_true, y_pred), 4),
        "forecast_bias": round(forecast_bias(y_true, y_pred), 4),
        "rmse_mae_ratio": round(rmse_val / mae_val, 4) if mae_val > 1e-10 else float("nan"),
    }

    if horizon is not None:
        results["horizon"] = horizon

    mase_val = results["mase"]
    pass_str = "✅" if isinstance(mase_val, float) and mase_val < 1.0 else "❌"
    results["pass_naive"] = pass_str

    h_str = f" (h={horizon})" if horizon else ""
    logger.info(
        f"  {model_name}{h_str}: MAE={results['mae']}, RMSE={results['rmse']}, MASE={results['mase']} {pass_str}"
    )

    return results


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 35.0,
) -> dict[str, float]:
    """Classification metrics for threshold exceedance.

    Evaluates how well the model predicts PM2.5 exceeding a threshold.
    Inspired by RC's multi-threshold classification (25, 35, 50 µg/m³).

    Includes ROC-AUC via pseudo-probability (sigmoid-scaled distance from
    threshold), matching RC's approach without requiring sklearn.

    Args:
        y_true: Actual PM2.5 values (µg/m³).
        y_pred: Predicted PM2.5 values (µg/m³).
        threshold: PM2.5 threshold for exceedance classification.

    Returns:
        Dictionary with precision, recall, f1, brier_score, roc_auc, n_exceed.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true_class = (y_true > threshold).astype(int)
    y_pred_class = (y_pred > threshold).astype(int)
    n_exceed = int(y_true_class.sum())

    results: dict[str, float] = {"threshold": threshold, "n_exceed": n_exceed}

    if n_exceed == 0:
        logger.warning(f"No samples exceed threshold {threshold} µg/m³")
        results.update({
            "precision": float("nan"), "recall": float("nan"),
            "f1": float("nan"), "brier_score": float("nan"),
            "roc_auc": float("nan"),
        })
        return results

    # Precision, Recall, F1 — manual calc to avoid sklearn import
    tp = int(((y_true_class == 1) & (y_pred_class == 1)).sum())
    fp = int(((y_true_class == 0) & (y_pred_class == 1)).sum())
    fn = int(((y_true_class == 1) & (y_pred_class == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Brier Score — mean squared error of binary classification
    brier = float(np.mean((y_true_class - y_pred_class) ** 2))

    # ROC-AUC — manual computation using pseudo-probabilities
    # Convert regression outputs to probabilities via sigmoid of distance from threshold
    roc_auc = _compute_roc_auc(y_true_class, y_pred, threshold)

    results.update({
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4),
    })
    return results


def _compute_roc_auc(
    y_true_class: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> float:
    """Compute ROC-AUC from regression predictions.

    Uses sigmoid-scaled distance from threshold as pseudo-probability,
    then computes AUC via trapezoidal rule on the ROC curve.

    Args:
        y_true_class: Binary labels (0/1).
        y_pred: Raw regression predictions (µg/m³).
        threshold: Classification threshold.

    Returns:
        ROC-AUC score (0-1). Returns NaN if only one class present.
    """
    n_pos = int(y_true_class.sum())
    n_neg = int(len(y_true_class) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Pseudo-probability: sigmoid of (pred - threshold), scaled by std
    distance = y_pred - threshold
    scale = max(np.std(y_pred), 1.0)  # prevent division by zero
    prob = 1.0 / (1.0 + np.exp(-distance / scale))

    # Sort by descending probability
    sorted_idx = np.argsort(-prob)
    y_sorted = y_true_class[sorted_idx]

    # Trapezoidal ROC-AUC
    tp_count = 0
    fp_count = 0
    auc = 0.0
    prev_fpr = 0.0

    for label in y_sorted:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
            tpr = tp_count / n_pos
            fpr = fp_count / n_neg
            auc += (fpr - prev_fpr) * tpr
            prev_fpr = fpr

    return float(auc)


def evaluate_forecast_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray,
    model_name: str = "Model",
    horizon: int | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, float | str | dict]:
    """Full evaluation: regression + classification metrics.

    Extends evaluate_forecast() with threshold classification metrics.

    Args:
        y_true: Actual values.
        y_pred: Model predictions.
        y_naive: Naive baseline predictions.
        model_name: Name for logging.
        horizon: Forecast horizon (h).
        thresholds: PM2.5 thresholds for classification. Default: [25, 35, 50].

    Returns:
        Dictionary with regression metrics + classification per threshold.
    """
    if thresholds is None:
        thresholds = [25.0, 35.0, 50.0]

    # Regression metrics
    results = evaluate_forecast(
        y_true=y_true,
        y_pred=y_pred,
        y_naive=y_naive,
        model_name=model_name,
        horizon=horizon,
    )

    # Classification metrics per threshold
    clf_results = {}
    for thresh in thresholds:
        clf = classification_metrics(y_true, y_pred, threshold=thresh)
        clf_results[f"threshold_{int(thresh)}"] = clf
        if clf["n_exceed"] > 0:
            logger.info(
                f"    {model_name} @ {thresh}µg/m³: "
                f"P={clf['precision']:.3f} R={clf['recall']:.3f} F1={clf['f1']:.3f} "
                f"Brier={clf['brier_score']:.4f} AUC={clf['roc_auc']:.3f} (n={clf['n_exceed']})"
            )

    results["classification"] = clf_results
    return results
