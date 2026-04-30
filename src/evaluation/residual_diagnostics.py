"""Residual Diagnostics — Post-model residual analysis.

Implements Peixeiro Ch.6 modeling procedure:
    1. Residual time plot (trend check)
    2. Residual histogram (normality)
    3. Q-Q Plot of residuals (normality visual)
    4. ACF of residuals (independence check)
    5. Ljung-Box test (statistical independence test)

Reference: Peixeiro Ch.6 pp.116-124 (Fig 6.6, 6.13)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Lazy imports for statsmodels
_sm_loaded = False


def _ensure_statsmodels():
    """Lazy import statsmodels to avoid import conflicts."""
    global _sm_loaded
    if not _sm_loaded:
        import statsmodels  # noqa: F401
        _sm_loaded = True


# ── VTF: Centralized theme ──
from src.viz.theme import apply_mpl_theme, annotation_bbox, ACCENT_COLORS

ACCENT_BLUE = ACCENT_COLORS["blue"]
ACCENT_ORANGE = ACCENT_COLORS["orange"]
ACCENT_GREEN = ACCENT_COLORS["green"]
ACCENT_RED = ACCENT_COLORS["red"]


def run_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    horizon: int = 1,
    output_dir: str | Path | None = None,
    max_lags: int = 48,
) -> dict[str, Any]:
    """Run full residual diagnostics suite.

    Args:
        y_true: Actual values.
        y_pred: Model predictions.
        model_name: Model name for titles.
        horizon: Forecast horizon.
        output_dir: Directory to save charts (None = no save).
        max_lags: Maximum lags for ACF and Ljung-Box tests.

    Returns:
        Dictionary with diagnostic results:
            - ljung_box: {lag: p_value} — p > 0.05 = uncorrelated (good)
            - normality: {shapiro_p, mean, std, skew, kurtosis}
            - independence: {acf_values for first N lags}
            - verdict: str — "PASS" or "FAIL" with reason
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "model": model_name,
        "horizon": horizon,
    }

    # 1. Residual statistics
    results["residual_stats"] = {
        "mean": round(float(np.mean(residuals)), 4),
        "std": round(float(np.std(residuals)), 4),
        "skew": round(float(stats.skew(residuals)), 4),
        "kurtosis": round(float(stats.kurtosis(residuals)), 4),
        "median": round(float(np.median(residuals)), 4),
    }

    # 2. Ljung-Box test
    lb_results = _ljung_box_test(residuals, max_lags=max_lags)
    results["ljung_box"] = lb_results

    # 3. Normality test
    sample = residuals[:min(5000, len(residuals))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, shapiro_p = stats.shapiro(sample)
    results["normality"] = {
        "shapiro_p": round(float(shapiro_p), 6),
        "is_normal": bool(shapiro_p > 0.05),
    }

    # 4. Verdict
    lb_fails = sum(1 for v in lb_results.values() if v < 0.05)
    lb_total = len(lb_results)
    lb_pass_rate = 1 - lb_fails / lb_total if lb_total > 0 else 0

    if lb_pass_rate > 0.8 and abs(results["residual_stats"]["mean"]) < 1.0:
        verdict = "PASS — Residuals approximately independent, near zero-mean"
    elif lb_pass_rate > 0.5:
        verdict = "PARTIAL — Some autocorrelation remains in residuals"
    else:
        verdict = "FAIL — Significant autocorrelation in residuals (model misspecification)"

    results["verdict"] = verdict
    results["lb_pass_rate"] = round(lb_pass_rate, 3)

    # 5. Generate 4-panel diagnostic chart
    if output_dir is not None:
        _generate_diagnostic_chart(
            residuals=residuals,
            model_name=model_name,
            horizon=horizon,
            lb_results=lb_results,
            results=results,
            output_dir=output_dir,
        )

    return results


def _ljung_box_test(
    residuals: np.ndarray,
    max_lags: int = 48,
) -> dict[str, float]:
    """Run Ljung-Box test at multiple lags.

    Ref: Peixeiro Ch.6 — "If p > 0.05, residuals are uncorrelated."

    Returns:
        {lag_N: p_value} dictionary.
    """
    _ensure_statsmodels()
    from statsmodels.stats.diagnostic import acorr_ljungbox

    test_lags = [1, 6, 12, 24, min(max_lags, len(residuals) // 3)]
    test_lags = sorted(set(l for l in test_lags if l > 0 and l < len(residuals) // 2))

    results = {}
    for lag in test_lags:
        try:
            lb = acorr_ljungbox(residuals, lags=[lag], return_df=True)
            p_val = float(lb["lb_pvalue"].iloc[0])
            results[f"lag_{lag}"] = round(p_val, 6)
        except Exception:
            results[f"lag_{lag}"] = float("nan")

    return results


def _generate_diagnostic_chart(
    residuals: np.ndarray,
    model_name: str,
    horizon: int,
    lb_results: dict,
    results: dict,
    output_dir: Path,
) -> None:
    """Generate 4-panel residual diagnostic chart.

    Layout (Peixeiro Fig 6.6 style):
        [1] Residual vs Time  | [2] Histogram + KDE
        [3] Q-Q Plot          | [4] ACF of Residuals
    """
    apply_mpl_theme("light")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: Residual vs Time
    ax1 = axes[0, 0]
    ax1.plot(residuals, color=ACCENT_BLUE, linewidth=0.5, alpha=0.7)
    ax1.axhline(y=0, color=ACCENT_RED, linestyle="--", linewidth=1)
    ax1.axhline(y=np.mean(residuals), color=ACCENT_ORANGE, linestyle=":", linewidth=1)
    ax1.set_title("1. Residuals vs Time", fontweight="bold")
    ax1.set_xlabel("Observation")
    ax1.set_ylabel("Residual (µg/m³)")
    ax1.grid(True, alpha=0.3)
    mean_text = f"Mean = {np.mean(residuals):.3f}"
    ax1.text(0.02, 0.95, mean_text, transform=ax1.transAxes, fontsize=8,
            verticalalignment="top", bbox=annotation_bbox("light"))

    # Panel 2: Histogram + KDE
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=50, density=True, color=ACCENT_BLUE, alpha=0.6, edgecolor="none")
    # Overlay normal fit
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    normal_pdf = stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals))
    ax2.plot(x_range, normal_pdf, color=ACCENT_RED, linewidth=2, label="Normal fit")
    ax2.set_title("2. Residual Distribution", fontweight="bold")
    ax2.set_xlabel("Residual (µg/m³)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    skew_text = f"Skew={results['residual_stats']['skew']:.2f} Kurt={results['residual_stats']['kurtosis']:.2f}"
    ax2.text(0.02, 0.95, skew_text, transform=ax2.transAxes, fontsize=8,
            verticalalignment="top", bbox=annotation_bbox("light"))

    # Panel 3: Q-Q Plot
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title("3. Q-Q Plot (Normality)", fontweight="bold")
    ax3.get_lines()[0].set(color=ACCENT_BLUE, markersize=2, alpha=0.5)
    ax3.get_lines()[1].set(color=ACCENT_RED, linewidth=2)
    ax3.grid(True, alpha=0.3)

    # Panel 4: ACF of Residuals
    ax4 = axes[1, 1]
    _ensure_statsmodels()
    from statsmodels.tsa.stattools import acf

    max_lag = min(48, len(residuals) // 3)
    acf_vals = acf(residuals, nlags=max_lag, fft=True)
    lags = np.arange(len(acf_vals))

    ax4.bar(lags[1:], acf_vals[1:], color=ACCENT_BLUE, width=0.6, alpha=0.7)
    # Confidence bounds (95%)
    ci = 1.96 / np.sqrt(len(residuals))
    ax4.axhline(y=ci, color=ACCENT_RED, linestyle="--", linewidth=1, alpha=0.7)
    ax4.axhline(y=-ci, color=ACCENT_RED, linestyle="--", linewidth=1, alpha=0.7)
    ax4.axhline(y=0, color="#555", linewidth=0.5)
    ax4.set_title("4. ACF of Residuals (should be within bounds)", fontweight="bold")
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("Autocorrelation")
    ax4.grid(True, alpha=0.3)

    # Overall verdict
    verdict = results["verdict"]
    color = ACCENT_GREEN if "PASS" in verdict else (ACCENT_ORANGE if "PARTIAL" in verdict else ACCENT_RED)
    fig.suptitle(
        f"Residual Diagnostics — {model_name} (h={horizon})\n{verdict}",
        fontsize=13, fontweight="bold", color=color, y=1.02
    )

    plt.tight_layout()
    fname = f"diagnostics_{model_name.lower().replace(' ', '_')}_h{horizon}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Diagnostics] Saved: {out_path}", flush=True)


def run_all_diagnostics(
    predictions_dir: str | Path,
    output_dir: str | Path,
) -> list[dict]:
    """Run diagnostics for all available model predictions.

    Args:
        predictions_dir: Directory containing prediction CSVs.
        output_dir: Directory to save diagnostic outputs.

    Returns:
        List of diagnostic result dicts.
    """
    import pandas as pd

    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for csv_file in sorted(predictions_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            # Expect columns: y_true, y_pred (or actual, predicted)
            if "y_true" in df.columns and "y_pred" in df.columns:
                y_true = df["y_true"].values
                y_pred = df["y_pred"].values
            elif "actual" in df.columns and "predicted" in df.columns:
                y_true = df["actual"].values
                y_pred = df["predicted"].values
            else:
                continue

            # Extract model name and horizon from filename
            name = csv_file.stem
            result = run_residual_diagnostics(
                y_true=y_true,
                y_pred=y_pred,
                model_name=name,
                output_dir=output_dir,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  [Diagnostics] Skipped {csv_file.name}: {e}", flush=True)

    # Save summary
    if all_results:
        summary_path = output_dir / "diagnostics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  [Diagnostics] Summary saved: {summary_path}", flush=True)

    return all_results
