"""Adaptive Conformal Inference (ACI) for non-stationary time series.

Implements ACI (Gibbs & Candès, 2021) which adaptively adjusts the
conformal quantile level to maintain target coverage despite
distribution shift in time series data.

References:
    [31] I. Gibbs and E. Candès, "Adaptive Conformal Inference Under
         Distribution Shift," NeurIPS 2021.
         DOI: 10.48550/arXiv.2106.00170

    [32] M. Zaffran, O. Féron, Y. Goude, J. Josse, and A. Dieuleveut,
         "Adaptive Conformal Predictions for Time Series," ICML 2022.
         DOI: 10.48550/arXiv.2202.07282
"""

import numpy as np
from loguru import logger


def adaptive_conformal_inference(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    y_cal_true: np.ndarray,
    y_cal_lower: np.ndarray,
    y_cal_upper: np.ndarray,
    alpha: float = 0.10,
    gamma: float = 0.005,
) -> dict:
    """Run ACI on test predictions using calibration residuals.

    Unlike static CQR which uses a fixed conformal adjustment,
    ACI adapts the quantile level αₜ at each step:
        αₜ₊₁ = αₜ + γ · (α - errₜ)

    where errₜ = 1 if yₜ ∉ [lower_t, upper_t], else 0.

    Args:
        y_true: Test actual values (n_test,).
        y_pred_lower: Test lower quantile predictions (n_test,).
        y_pred_upper: Test upper quantile predictions (n_test,).
        y_cal_true: Calibration actual values (n_cal,).
        y_cal_lower: Calibration lower quantile predictions (n_cal,).
        y_cal_upper: Calibration upper quantile predictions (n_cal,).
        alpha: Target miscoverage rate (0.10 = 90% coverage target).
        gamma: Step size for adaptive update (smaller = more stable).

    Returns:
        Dict with coverage, avg_width, coverage_trajectory, etc.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_lower = np.asarray(y_pred_lower, dtype=float)
    y_pred_upper = np.asarray(y_pred_upper, dtype=float)
    y_cal_true = np.asarray(y_cal_true, dtype=float)
    y_cal_lower = np.asarray(y_cal_lower, dtype=float)
    y_cal_upper = np.asarray(y_cal_upper, dtype=float)

    n_test = len(y_true)
    n_cal = len(y_cal_true)

    # Compute calibration nonconformity scores
    # Score = max(lower - y, y - upper) → positive means outside interval
    cal_scores = np.maximum(y_cal_lower - y_cal_true, y_cal_true - y_cal_upper)
    cal_scores_sorted = np.sort(cal_scores)

    # Initialize adaptive alpha
    alpha_t = alpha
    alpha_trajectory = [alpha_t]
    coverage_trajectory = []
    width_trajectory = []

    covered = np.zeros(n_test, dtype=bool)
    aci_lower = np.zeros(n_test)
    aci_upper = np.zeros(n_test)

    for t in range(n_test):
        # Compute conformal quantile at current alpha_t
        # Clamp alpha_t to valid range
        alpha_clamped = np.clip(alpha_t, 0.001, 0.999)

        # Quantile from calibration scores
        q_level = np.ceil((1 - alpha_clamped) * (n_cal + 1)) / n_cal
        q_level = np.clip(q_level, 0.0, 1.0)
        q_idx = int(np.clip(q_level * n_cal, 0, n_cal - 1))
        q_hat = cal_scores_sorted[q_idx]

        # Adjusted prediction interval
        aci_lower[t] = y_pred_lower[t] - q_hat
        aci_upper[t] = y_pred_upper[t] + q_hat

        # Check coverage
        covered[t] = (y_true[t] >= aci_lower[t]) and (y_true[t] <= aci_upper[t])
        err_t = 0.0 if covered[t] else 1.0

        # Adaptive update
        alpha_t = alpha_t + gamma * (alpha - err_t)
        alpha_trajectory.append(alpha_t)

        # Track running coverage
        coverage_so_far = np.mean(covered[: t + 1])
        coverage_trajectory.append(coverage_so_far)
        width_trajectory.append(aci_upper[t] - aci_lower[t])

    # Final metrics
    final_coverage = float(np.mean(covered))
    avg_width = float(np.mean(aci_upper - aci_lower))

    logger.info(
        f"ACI complete: coverage={final_coverage:.4f} "
        f"(target={1 - alpha:.2f}), avg_width={avg_width:.2f}, "
        f"gamma={gamma}, n_test={n_test}, n_cal={n_cal}"
    )

    return {
        "method": "aci",
        "alpha": alpha,
        "gamma": gamma,
        "coverage": round(final_coverage, 4),
        "avg_width": round(avg_width, 4),
        "n_test": n_test,
        "n_calibration": n_cal,
        "coverage_trajectory": [round(c, 4) for c in coverage_trajectory],
        "alpha_trajectory": [round(a, 4) for a in alpha_trajectory],
        "aci_lower": aci_lower.tolist(),
        "aci_upper": aci_upper.tolist(),
    }
