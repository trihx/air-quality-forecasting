"""EDA module — Exploratory Data Analysis for PM2.5 time series.

Follows SKILL.md §4 checklist and visualization-storytelling.md guide.
Generates comprehensive EDA report with charts saved to research/eda/.
"""

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats as sp_stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

from src.data.loader import FEATURE_COLS, TARGET_COL
from src.viz.theme import apply_mpl_theme

# Use non-interactive backend for saving plots
matplotlib.use("Agg")

# WHO PM2.5 guidelines (µg/m³)
WHO_PM25_ANNUAL = 5.0
WHO_PM25_24H = 15.0
VN_PM25_24H = 50.0  # Vietnam QCVN 05:2023

# Color palette
PALETTE = {
    "pm25": "#e74c3c",
    "nhiet_do": "#e67e22",
    "do_am": "#3498db",
    "diem_suong": "#2ecc71",
    "co2": "#9b59b6",
    "who_line": "#c0392b",
    "vn_line": "#f39c12",
}


def run_full_eda(
    df: pd.DataFrame,
    output_dir: str | Path = "research/eda",
) -> dict[str, Any]:
    """Run complete EDA pipeline per SKILL.md §4 checklist.

    Args:
        df: Cleaned DataFrame with DatetimeIndex.
        output_dir: Directory to save charts & report.

    Returns:
        Dictionary with EDA results and statistics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    apply_mpl_theme("light")
    sns.set_palette("husl")

    logger.info("=" * 60)
    logger.info("EDA Pipeline Started")
    logger.info("=" * 60)

    results: dict[str, Any] = {}

    # 1. Descriptive Statistics
    results["descriptive"] = _descriptive_stats(df, output_path)

    # 2. Time Series Plots
    _plot_time_series(df, output_path)

    # 3. Distribution Analysis
    results["distributions"] = _plot_distributions(df, output_path)

    # 4. Correlation Analysis
    results["correlations"] = _correlation_analysis(df, output_path)

    # 5. Stationarity Tests (ADF + KPSS)
    results["stationarity"] = _stationarity_tests(df)

    # 6. ACF/PACF Analysis
    _plot_acf_pacf(df, output_path)

    # 7. Seasonality & Temporal Patterns
    results["temporal"] = _temporal_patterns(df, output_path)

    # 8. Missing Values Analysis
    results["missing"] = _missing_analysis(df, output_path)

    plt.close("all")

    logger.info("=" * 60)
    logger.info(f"EDA Complete — {len(list(output_path.glob('*.png')))} charts saved to {output_path}")
    logger.info("=" * 60)

    return results


# ============================================================
# 1. Descriptive Statistics
# ============================================================


def _descriptive_stats(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Generate descriptive statistics summary."""
    logger.info("[1/8] Descriptive Statistics")

    desc = df.describe().round(2)
    desc_dict: dict[str, Any] = {}
    for col in df.columns:
        col_data = df[col].dropna()
        desc_dict[col] = {
            "count": int(col_data.count()),
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "q25": float(col_data.quantile(0.25)),
            "median": float(col_data.median()),
            "q75": float(col_data.quantile(0.75)),
            "max": float(col_data.max()),
            "skewness": float(col_data.skew()),
            "kurtosis": float(col_data.kurtosis()),
        }
        logger.debug(f"  {col}: mean={desc_dict[col]['mean']:.2f}, std={desc_dict[col]['std']:.2f}")

    # Save as table image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    table = ax.table(
        cellText=desc.values,
        colLabels=desc.columns,
        rowLabels=desc.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title("Descriptive Statistics", fontsize=14, fontweight="bold", pad=20)
    fig.savefig(output_path / "01_descriptive_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return desc_dict


# ============================================================
# 2. Time Series Plots
# ============================================================


def _plot_time_series(df: pd.DataFrame, output_path: Path) -> None:
    """Plot time series for all variables with WHO reference lines."""
    logger.info("[2/8] Time Series Plots")

    # Full time series — PM2.5
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df.index, df[TARGET_COL], color=PALETTE["pm25"], linewidth=0.5, alpha=0.7)
    ax.axhline(
        y=WHO_PM25_24H,
        color=PALETTE["who_line"],
        linestyle="--",
        linewidth=1.5,
        label=f"WHO 24h ({WHO_PM25_24H})",
    )
    ax.axhline(
        y=VN_PM25_24H,
        color=PALETTE["vn_line"],
        linestyle="--",
        linewidth=1.5,
        label=f"VN QCVN ({VN_PM25_24H})",
    )
    ax.set_title("PM2.5 Concentration Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper right")
    fig.savefig(output_path / "02_pm25_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # All features subplot
    fig, axes = plt.subplots(len(FEATURE_COLS) + 1, 1, figsize=(16, 3 * (len(FEATURE_COLS) + 1)), sharex=True)
    all_cols = FEATURE_COLS + [TARGET_COL]
    for ax, col in zip(axes, all_cols, strict=False):
        ax.plot(df.index, df[col], color=PALETTE.get(col, "#333"), linewidth=0.5, alpha=0.7)
        ax.set_ylabel(col)
        ax.set_title(col, fontsize=11, fontweight="bold")
    axes[-1].set_xlabel("Date")
    fig.suptitle("All Variables Over Time", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path / "02_all_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. Distribution Analysis
# ============================================================


def _plot_distributions(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Plot histograms + boxplots + normality tests."""
    logger.info("[3/8] Distribution Analysis")

    all_cols = FEATURE_COLS + [TARGET_COL]
    dist_results: dict[str, Any] = {}

    # Histograms with KDE
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()
    for i, col in enumerate(all_cols):
        ax = axes_flat[i]
        data = df[col].dropna()
        ax.hist(data, bins=50, alpha=0.7, color=PALETTE.get(col, "#333"), density=True, edgecolor="white")
        data.plot.kde(ax=ax, color="black", linewidth=1.5)
        ax.set_title(f"{col} Distribution", fontweight="bold")
        ax.set_xlabel(col)

        # Shapiro-Wilk test (on sample if too large)
        sample = data.sample(min(5000, len(data)), random_state=42)
        stat, p_value = sp_stats.shapiro(sample)
        dist_results[col] = {
            "shapiro_stat": float(round(stat, 4)),
            "shapiro_pvalue": float(p_value),
            "is_normal": p_value > 0.05,
        }
        normality_text = f"Shapiro p={p_value:.2e}\n{'Normal ✅' if p_value > 0.05 else 'Non-normal ❌'}"
        ax.text(
            0.95,
            0.95,
            normality_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    # Hide unused subplot
    if len(all_cols) < len(axes_flat):
        for j in range(len(all_cols), len(axes_flat)):
            axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "03_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Boxplots
    fig, axes = plt.subplots(1, len(all_cols), figsize=(16, 5))
    for ax, col in zip(axes, all_cols, strict=False):
        bp = ax.boxplot(df[col].dropna(), patch_artist=True)
        bp["boxes"][0].set_facecolor(PALETTE.get(col, "#333"))
        bp["boxes"][0].set_alpha(0.7)
        ax.set_title(col, fontweight="bold")
    fig.suptitle("Boxplots — Outlier Detection", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "03_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dist_results


# ============================================================
# 4. Correlation Analysis
# ============================================================


def _correlation_analysis(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Pearson + Spearman correlation matrices."""
    logger.info("[4/8] Correlation Analysis")

    all_cols = FEATURE_COLS + [TARGET_COL]
    df_numeric = df[all_cols]

    # Pearson
    pearson_corr = df_numeric.corr(method="pearson")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0], square=True)
    axes[0].set_title("Pearson Correlation", fontweight="bold")

    # Spearman
    spearman_corr = df_numeric.corr(method="spearman")
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[1], square=True)
    axes[1].set_title("Spearman Correlation", fontweight="bold")

    fig.suptitle("Correlation Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "04_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Correlation with target
    target_corr = {
        col: {
            "pearson": float(round(pearson_corr.loc[col, TARGET_COL], 4)),
            "spearman": float(round(spearman_corr.loc[col, TARGET_COL], 4)),
        }
        for col in FEATURE_COLS
    }
    for col, corrs in target_corr.items():
        logger.info(f"  {col} → PM2.5: pearson={corrs['pearson']:.3f}, spearman={corrs['spearman']:.3f}")

    return {"target_correlations": target_corr}


# ============================================================
# 5. Stationarity Tests
# ============================================================


def _stationarity_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Run ADF + KPSS tests per SKILL.md §4 interpretation guide."""
    logger.info("[5/8] Stationarity Tests (ADF + KPSS)")

    results: dict[str, Any] = {}
    all_cols = FEATURE_COLS + [TARGET_COL]

    for col in all_cols:
        data = df[col].dropna()

        # ADF Test (H0: non-stationary)
        adf_result = adfuller(data, autolag="AIC")
        adf_stat, adf_pval = float(adf_result[0]), float(adf_result[1])

        # KPSS Test (H0: stationary) — suppress warnings for truncation
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(data, regression="c", nlags="auto")
        kpss_stat, kpss_pval = float(kpss_result[0]), float(kpss_result[1])

        # Interpretation per SKILL.md table
        adf_stationary = adf_pval < 0.05
        kpss_stationary = kpss_pval > 0.05

        if adf_stationary and kpss_stationary:
            verdict = "Stationary ✅"
        elif not adf_stationary and not kpss_stationary:
            verdict = "Non-stationary ❌ — needs differencing"
        else:
            verdict = "Inconclusive ⚠️ — conflicting results"

        results[col] = {
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pval,
            "adf_stationary": adf_stationary,
            "kpss_stat": kpss_stat,
            "kpss_pvalue": kpss_pval,
            "kpss_stationary": kpss_stationary,
            "verdict": verdict,
        }
        logger.info(f"  {col}: ADF p={adf_pval:.4f}, KPSS p={kpss_pval:.4f} → {verdict}")

    return results


# ============================================================
# 6. ACF/PACF
# ============================================================


def _plot_acf_pacf(df: pd.DataFrame, output_path: Path) -> None:
    """Plot ACF and PACF for PM2.5."""
    logger.info("[6/8] ACF/PACF Analysis")

    data = df[TARGET_COL].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(data, lags=72, ax=axes[0], title="Autocorrelation (ACF) — PM2.5")
    plot_pacf(data, lags=72, ax=axes[1], title="Partial Autocorrelation (PACF) — PM2.5", method="ywm")
    fig.suptitle("ACF & PACF — PM2.5 (72 lags = 3 days)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "06_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. Temporal Patterns
# ============================================================


def _temporal_patterns(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Analyze hourly, daily, monthly patterns."""
    logger.info("[7/8] Temporal Patterns")

    results: dict[str, Any] = {}

    # Hourly pattern
    hourly = df.groupby(df.index.hour)[TARGET_COL].agg(["mean", "std"])
    results["peak_hour"] = int(hourly["mean"].idxmax())
    results["trough_hour"] = int(hourly["mean"].idxmin())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Hourly
    axes[0].bar(
        hourly.index, hourly["mean"], yerr=hourly["std"], capsize=2, color=PALETTE["pm25"], alpha=0.7, edgecolor="white"
    )
    axes[0].axhline(y=WHO_PM25_24H, color=PALETTE["who_line"], linestyle="--", linewidth=1, label="WHO 24h")
    axes[0].set_title("PM2.5 by Hour of Day", fontweight="bold")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].legend()

    # Day of week
    daily = df.groupby(df.index.dayofweek)[TARGET_COL].agg(["mean", "std"])
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(
        range(7), daily["mean"], yerr=daily["std"], capsize=2, color=PALETTE["pm25"], alpha=0.7, edgecolor="white"
    )
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names)
    axes[1].set_title("PM2.5 by Day of Week", fontweight="bold")
    axes[1].set_xlabel("Day")

    # Monthly
    monthly = df.groupby(df.index.month)[TARGET_COL].agg(["mean", "std"])
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    available_months = [i - 1 for i in monthly.index]
    axes[2].bar(
        range(len(monthly)),
        monthly["mean"],
        yerr=monthly["std"],
        capsize=2,
        color=PALETTE["pm25"],
        alpha=0.7,
        edgecolor="white",
    )
    axes[2].set_xticks(range(len(monthly)))
    axes[2].set_xticklabels([month_names[i] for i in available_months], rotation=45)
    axes[2].set_title("PM2.5 by Month", fontweight="bold")
    axes[2].set_xlabel("Month")

    results["peak_month"] = int(monthly["mean"].idxmax())
    results["trough_month"] = int(monthly["mean"].idxmin())

    fig.suptitle("Temporal Patterns — PM2.5", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "07_temporal_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Rolling statistics (7-day window)
    fig, ax = plt.subplots(figsize=(16, 6))
    rolling_mean = df[TARGET_COL].rolling(window=24 * 7).mean()
    rolling_std = df[TARGET_COL].rolling(window=24 * 7).std()
    ax.plot(df.index, df[TARGET_COL], alpha=0.3, color=PALETTE["pm25"], linewidth=0.5, label="Raw")
    ax.plot(df.index, rolling_mean, color="black", linewidth=1.5, label="7-day MA")
    ax.fill_between(
        df.index,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.15,
        color=PALETTE["pm25"],
        label="± 1 std",
    )
    ax.axhline(y=WHO_PM25_24H, color=PALETTE["who_line"], linestyle="--", linewidth=1, label="WHO 24h")
    ax.set_title("PM2.5 — 7-Day Rolling Mean ± Std", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper right")
    fig.savefig(output_path / "07_rolling_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Peak hour: {results['peak_hour']}h, Trough: {results['trough_hour']}h")
    logger.info(f"  Peak month: {results['peak_month']}, Trough: {results['trough_month']}")

    return results


# ============================================================
# 8. Missing Values Analysis
# ============================================================


def _missing_analysis(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Analyze missing value patterns."""
    logger.info("[8/8] Missing Values Analysis")

    missing_counts = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df) * 100).round(2)

    results: dict[str, Any] = {}
    for col in df.columns:
        results[col] = {
            "count": int(missing_counts[col]),
            "pct": float(missing_pct[col]),
        }

    # Plot missing values
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE.get(col, "#333") for col in df.columns]
    bars = ax.barh(df.columns, missing_pct.values, color=colors, alpha=0.7, edgecolor="white")
    ax.set_xlabel("Missing Values (%)")
    ax.set_title("Missing Values by Feature", fontsize=14, fontweight="bold")
    for bar, pct in zip(bars, missing_pct.values, strict=False):
        if pct > 0:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path / "08_missing_values.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results
