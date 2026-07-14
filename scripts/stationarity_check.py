"""Stationarity diagnostics for PM2.5 time series.

Tests:
    - ADF (Augmented Dickey-Fuller): H0 = unit root (non-stationary)
    - KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = stationary

Usage:
    uv run python scripts/stationarity_check.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss


def run_adf_test(series: pd.Series, name: str) -> dict:
    """Run Augmented Dickey-Fuller test.

    H0: Series has a unit root (non-stationary).
    Reject H0 if p < 0.05 → series IS stationary.
    """
    series_clean = series.dropna()
    result = adfuller(series_clean, autolag="AIC")

    adf_stat, p_value, used_lag, n_obs, critical_values, _ = result
    is_stationary = p_value < 0.05

    out = {
        "test": "ADF",
        "series": name,
        "statistic": round(float(adf_stat), 4),
        "p_value": round(float(p_value), 6),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {k: round(float(v), 4) for k, v in critical_values.items()},
        "is_stationary": bool(is_stationary),
        "interpretation": (
            f"STATIONARY (p={p_value:.4f} < 0.05, reject H0)"
            if is_stationary
            else f"NON-STATIONARY (p={p_value:.4f} >= 0.05, fail to reject H0)"
        ),
    }

    emoji = "✅" if is_stationary else "❌"
    logger.info(f"  ADF [{name}]: {emoji} {out['interpretation']}")
    return out


def run_kpss_test(series: pd.Series, name: str, regression: str = "c") -> dict:
    """Run KPSS test.

    H0: Series IS stationary.
    Reject H0 if p < 0.05 → series is NON-stationary.
    """
    series_clean = series.dropna()
    # KPSS may warn about p-value bounds; that's expected
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=InterpolationWarning)
        try:
            stat, p_value, n_lags, critical_values = kpss(series_clean, regression=regression, nlags="auto")
        except Exception:
            # Fallback without nlags="auto"
            stat, p_value, n_lags, critical_values = kpss(series_clean, regression=regression)

    is_stationary = p_value >= 0.05  # KPSS: H0 = stationary

    out = {
        "test": "KPSS",
        "series": name,
        "regression": regression,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "n_lags": int(n_lags),
        "critical_values": {k: round(float(v), 4) for k, v in critical_values.items()},
        "is_stationary": bool(is_stationary),
        "interpretation": (
            f"STATIONARY (p={p_value:.4f} >= 0.05, fail to reject H0)"
            if is_stationary
            else f"NON-STATIONARY (p={p_value:.4f} < 0.05, reject H0)"
        ),
    }

    emoji = "✅" if is_stationary else "❌"
    logger.info(f"  KPSS [{name}]: {emoji} {out['interpretation']}")
    return out


# Handle InterpolationWarning that may not exist in all Python versions
try:
    from statsmodels.tools.sm_exceptions import InterpolationWarning
except ImportError:
    InterpolationWarning = UserWarning  # type: ignore[misc,assignment]


def plot_diagnostics(
    series: pd.Series,
    name: str,
    output_dir: Path,
    n_lags: int = 72,
) -> Path:
    """Plot time series + ACF + PACF diagnostics."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Time series plot
    axes[0].plot(series.index, series.values, linewidth=0.5, color="#2196F3")
    axes[0].set_title(f"Time Series: {name}", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Value")
    axes[0].axhline(y=series.mean(), color="red", linestyle="--", alpha=0.5, label=f"Mean={series.mean():.2f}")
    axes[0].legend()

    # ACF
    plot_acf(series.dropna(), ax=axes[1], lags=n_lags, alpha=0.05)
    axes[1].set_title(f"ACF: {name}", fontsize=12, fontweight="bold")

    # PACF
    plot_pacf(series.dropna(), ax=axes[2], lags=n_lags, alpha=0.05, method="ywm")
    axes[2].set_title(f"PACF: {name}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / f"stationarity_{name.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  📊 Saved plot: {out_path}")
    return out_path


def main() -> None:
    """Run full stationarity diagnostics."""
    logger.info("=" * 60)
    logger.info("📊 Stationarity Diagnostics for PM2.5")
    logger.info("=" * 60)

    # Load cleaned hourly data
    data_path = Path("dataset/interim/cleaned_hourly.csv")
    if not data_path.exists():
        logger.error(f"❌ Data not found: {data_path}")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    pm25 = df["pm25"]
    logger.info(f"Loaded: {len(pm25):,} hourly observations")
    logger.info(f"  Range: {pm25.index.min()} → {pm25.index.max()}")
    logger.info(f"  Stats: mean={pm25.mean():.2f}, std={pm25.std():.2f}, min={pm25.min():.2f}, max={pm25.max():.2f}")

    # Prepare variants
    variants = {
        "Raw PM2.5": pm25,
        "1st Diff (d=1)": pm25.diff(1).dropna(),
        "Seasonal Diff (d=24h)": pm25.diff(24).dropna(),
        "Log PM2.5": np.log1p(pm25),  # log(1+x) to handle zeros
        "Log 1st Diff": np.log1p(pm25).diff(1).dropna(),
    }

    # Output
    output_dir = Path("research/diagnostics/stationarity")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for name, series in variants.items():
        logger.info(f"\n--- {name} ({len(series):,} obs) ---")

        # ADF test
        adf = run_adf_test(series, name)
        all_results.append(adf)

        # KPSS test
        kpss_result = run_kpss_test(series, name)
        all_results.append(kpss_result)

        # Joint interpretation
        if adf["is_stationary"] and kpss_result["is_stationary"]:
            logger.info(f"  🟢 CONCLUSIVE: {name} is STATIONARY (both tests agree)")
        elif not adf["is_stationary"] and not kpss_result["is_stationary"]:
            logger.info(f"  🔴 CONCLUSIVE: {name} is NON-STATIONARY (both tests agree)")
        elif adf["is_stationary"] and not kpss_result["is_stationary"]:
            logger.info(f"  🟡 TREND-STATIONARY: {name} (ADF=stationary, KPSS=non-stationary)")
        else:
            logger.info(f"  🟡 INCONCLUSIVE: {name} (ADF=non-stationary, KPSS=stationary)")

        # Plot diagnostics
        plot_diagnostics(series, name, output_dir)

    # Save results
    json_path = output_dir / "stationarity_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_observations": len(pm25),
                "date_range": {"start": str(pm25.index.min()), "end": str(pm25.index.max())},
                "results": all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"\n💾 Results saved: {json_path}")

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("📋 STATIONARITY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Series':<25} {'ADF':<15} {'KPSS':<15} {'Conclusion':<20}")
    logger.info("-" * 75)
    for i in range(0, len(all_results), 2):
        adf_r = all_results[i]
        kpss_r = all_results[i + 1]
        name_str = adf_r["series"]
        adf_str = "Stationary" if adf_r["is_stationary"] else "Non-stat"
        kpss_str = "Stationary" if kpss_r["is_stationary"] else "Non-stat"
        if adf_r["is_stationary"] and kpss_r["is_stationary"]:
            conclusion = "✅ STATIONARY"
        elif not adf_r["is_stationary"] and not kpss_r["is_stationary"]:
            conclusion = "❌ NON-STAT"
        else:
            conclusion = "🟡 MIXED"
        logger.info(f"  {name_str:<25} {adf_str:<15} {kpss_str:<15} {conclusion:<20}")

    logger.info("=" * 60)
    logger.info("✅ Stationarity Check Complete!")


if __name__ == "__main__":
    main()
