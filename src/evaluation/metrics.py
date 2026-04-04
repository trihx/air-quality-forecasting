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

    results: dict[str, float | str] = {
        "model": model_name,
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 2),
        "smape": round(smape(y_true, y_pred), 2),
        "mase": round(mase(y_true, y_pred, y_naive), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
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
