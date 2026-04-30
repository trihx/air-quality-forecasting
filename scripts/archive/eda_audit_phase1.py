"""Phase 1 EDA Audit — Generate missing charts per reference book recommendations.

Generates:
    1. STL Decomposition (Manu Joseph Ch.3)
    2. Forecastability Assessment (Manu Joseph Ch.4)
    3. Box Plot per Hour (Vishwas & Patel Ch.4)
    4. Q-Q Plot (Peixeiro Ch.6) — bonus P1 item done early
    5. Periodogram / PSD (Huang Ch.7) — bonus P1 item done early

Output directory: research/eda/
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.tsa.seasonal import STL

# ── Config ──
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
EDA_DIR = PROJECT_ROOT / "research" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROJECT_ROOT / "dataset" / "interim" / "cleaned_hourly.csv"

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)

# ── VTF: Centralized theme (Light mode for publication) ──
from src.viz.theme import apply_mpl_theme, annotation_bbox, ACCENT_COLORS

THEME_MODE = "light"
apply_mpl_theme(THEME_MODE)

ACCENT_BLUE = ACCENT_COLORS["blue"]
ACCENT_ORANGE = ACCENT_COLORS["orange"]
ACCENT_GREEN = ACCENT_COLORS["green"]
ACCENT_RED = ACCENT_COLORS["red"]
ACCENT_PURPLE = ACCENT_COLORS["purple"]


def load_data() -> pd.DataFrame:
    """Load cleaned hourly data."""
    print(f"[Phase1] Loading data from {DATASET_PATH}...", flush=True)
    df = pd.read_csv(DATASET_PATH, parse_dates=["ngay_tao"], index_col="ngay_tao")
    df.index.name = "timestamp"
    print(f"  Shape: {df.shape}, Range: {df.index.min()} → {df.index.max()}", flush=True)
    print(f"  PM2.5 NaN: {df['pm25'].isna().sum()}", flush=True)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. STL Decomposition (P0-1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_stl_decomposition(df: pd.DataFrame) -> dict:
    """Generate STL decomposition chart.

    Reference: Manu Joseph Ch.3 pp.64-65, Brownlee Ch.12
    """
    print("\n[P0-1] STL Decomposition...", flush=True)

    pm25 = df["pm25"].dropna()

    # STL with daily period (24h)
    stl = STL(pm25, period=24, robust=True)
    result = stl.fit()

    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid

    # Calculate component strengths
    total_var = np.var(pm25)
    trend_strength = max(0, 1 - np.var(resid) / np.var(trend + resid))
    seasonal_strength = max(0, 1 - np.var(resid) / np.var(seasonal + resid))
    noise_ratio = np.var(resid) / total_var

    print(f"  Trend strength: {trend_strength:.3f}", flush=True)
    print(f"  Seasonal strength: {seasonal_strength:.3f}", flush=True)
    print(f"  Noise ratio: {noise_ratio:.3f}", flush=True)
    print(f"  Residual std: {np.std(resid):.3f} µg/m³ (performance floor)", flush=True)

    # ── 4-Panel Chart ──
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Original
    axes[0].plot(pm25.index, pm25.values, color=ACCENT_BLUE, linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel("PM2.5\n(µg/m³)")
    axes[0].set_title("STL Decomposition — PM2.5 (period=24h, LOESS)", fontsize=14, fontweight="bold", pad=10)
    axes[0].legend(["Original"], loc="upper right", fontsize=8)

    # Panel 2: Trend
    axes[1].plot(trend.index, trend.values, color=ACCENT_ORANGE, linewidth=1.5)
    axes[1].set_ylabel("Trend")
    strength_text = f"Trend Strength = {trend_strength:.3f}"
    axes[1].text(0.98, 0.85, strength_text, transform=axes[1].transAxes, ha="right",
                fontsize=9, bbox=annotation_bbox(THEME_MODE))

    # Panel 3: Seasonal
    axes[2].plot(seasonal.index, seasonal.values, color=ACCENT_GREEN, linewidth=0.8)
    axes[2].set_ylabel("Seasonal")
    axes[2].axhline(y=0, color="#999", linestyle="--", linewidth=0.5)
    strength_text = f"Seasonal Strength = {seasonal_strength:.3f}"
    axes[2].text(0.98, 0.85, strength_text, transform=axes[2].transAxes, ha="right",
                fontsize=9, bbox=annotation_bbox(THEME_MODE))

    # Panel 4: Residual
    axes[3].plot(resid.index, resid.values, color=ACCENT_RED, linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel("Residual")
    axes[3].axhline(y=0, color="#999", linestyle="--", linewidth=0.5)
    noise_text = f"Residual σ = {np.std(resid):.2f} µg/m³"
    axes[3].text(0.98, 0.85, noise_text, transform=axes[3].transAxes, ha="right",
                fontsize=9, bbox=annotation_bbox(THEME_MODE))

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = EDA_DIR / "05_stl_decomposition.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)

    # ── Zoomed Seasonal (1 week) ──
    fig2, ax = plt.subplots(figsize=(12, 4))
    week_slice = seasonal.iloc[:168]  # 7 days
    ax.plot(range(len(week_slice)), week_slice.values, color=ACCENT_GREEN, linewidth=1.5)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Seasonal Component (µg/m³)")
    ax.set_title("Seasonal Component — 1 Week Detail (7×24h cycles)", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="#999", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Mark day boundaries
    for d in range(1, 8):
        ax.axvline(x=d * 24, color="#999", linestyle=":", linewidth=0.5)

    plt.tight_layout()
    out_path2 = EDA_DIR / "05a_stl_seasonal_zoom.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_path2}", flush=True)

    return {
        "trend_strength": round(float(trend_strength), 4),
        "seasonal_strength": round(float(seasonal_strength), 4),
        "noise_ratio": round(float(noise_ratio), 4),
        "residual_std": round(float(np.std(resid)), 4),
        "residual_mean": round(float(np.mean(resid)), 4),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Forecastability Assessment (P0-2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _approximate_entropy(ts: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Approximate Entropy (ApEn) — measures time series complexity/regularity.

    Higher ApEn = more complex/random = harder to forecast.
    Lower ApEn = more regular/predictable.

    Args:
        ts: Time series values (1D array).
        m: Embedding dimension.
        r_factor: Tolerance as fraction of std.
    """
    n = len(ts)
    r = r_factor * np.std(ts)

    def _phi(m_val: int) -> float:
        patterns = np.array([ts[i:i + m_val] for i in range(n - m_val + 1)])
        count = np.zeros(len(patterns))
        for i, p in enumerate(patterns):
            count[i] = np.sum(np.max(np.abs(patterns - p), axis=1) <= r)
        return np.sum(np.log(count / len(patterns))) / len(patterns)

    return abs(_phi(m) - _phi(m + 1))


def generate_forecastability(df: pd.DataFrame, stl_results: dict) -> dict:
    """Generate forecastability assessment metrics.

    Reference: Manu Joseph Ch.4 pp.102-108
    """
    print("\n[P0-2] Forecastability Assessment...", flush=True)

    pm25 = df["pm25"].dropna().values

    # 1. Coefficient of Variation (CoV)
    cov = float(np.std(pm25) / np.mean(pm25))
    print(f"  CoV = {cov:.4f} (higher = harder)", flush=True)

    # 2. Approximate Entropy (ApEn) — use subsample for speed
    subsample = pm25[::6][:1000]  # every 6th point, max 1000
    apen = _approximate_entropy(subsample)
    print(f"  ApEn = {apen:.4f} (higher = more complex)", flush=True)

    # 3. Seasonality Strength (from STL)
    ss = stl_results["seasonal_strength"]
    print(f"  Seasonality Strength = {ss:.4f} (from STL)", flush=True)

    # 4. Autocorrelation at lag-1
    acf_1 = float(pd.Series(pm25).autocorr(lag=1))
    print(f"  ACF(1) = {acf_1:.4f}", flush=True)

    # Composite score (heuristic: higher = easier to forecast)
    # High seasonality + high autocorrelation + low entropy = easier
    forecastability_score = (ss * 0.35 + acf_1 * 0.35 + (1 - min(apen, 1.0)) * 0.3)
    print(f"  Forecastability Score = {forecastability_score:.4f} (composite)", flush=True)

    interpretation = (
        "Dễ dự báo" if forecastability_score > 0.7
        else "Trung bình" if forecastability_score > 0.4
        else "Khó dự báo"
    )
    print(f"  Interpretation: {interpretation}", flush=True)

    return {
        "cov": round(cov, 4),
        "approximate_entropy": round(apen, 4),
        "seasonality_strength": round(ss, 4),
        "acf_lag1": round(acf_1, 4),
        "forecastability_score": round(forecastability_score, 4),
        "interpretation": interpretation,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Box Plot per Hour (P0-6)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_hourly_boxplot(df: pd.DataFrame) -> None:
    """Generate box plot of PM2.5 per hour of day.

    Reference: Vishwas & Patel Ch.4
    Shows seasonal pattern (peak ~6h, trough ~12h) directly.
    """
    print("\n[P0-6] Box Plot per Hour...", flush=True)

    pm25_hourly = df[["pm25"]].dropna().copy()
    pm25_hourly["hour"] = pm25_hourly.index.hour

    fig, ax = plt.subplots(figsize=(14, 6))

    # Group data by hour
    hour_data = [pm25_hourly[pm25_hourly["hour"] == h]["pm25"].values for h in range(24)]

    bp = ax.boxplot(
        hour_data,
        positions=range(24),
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markerfacecolor=ACCENT_RED, markersize=2, alpha=0.3),
        medianprops=dict(color=ACCENT_ORANGE, linewidth=2),
        whiskerprops=dict(color="#8B95A5"),
        capprops=dict(color="#8B95A5"),
    )

    # Color boxes by median level
    medians = [np.median(d) for d in hour_data]
    max_median = max(medians)
    min_median = min(medians)

    for i, (patch, median) in enumerate(zip(bp["boxes"], medians)):
        # Color gradient: blue (low) → red (high)
        ratio = (median - min_median) / (max_median - min_median) if max_median > min_median else 0
        r = int(74 + ratio * 171)  # 74 → 245
        g = int(158 - ratio * 84)  # 158 → 74
        b = int(245 - ratio * 171)  # 245 → 74
        patch.set_facecolor(f"#{r:02x}{g:02x}{b:02x}")
        patch.set_alpha(0.7)

    # Mark peak/trough
    peak_hour = int(np.argmax(medians))
    trough_hour = int(np.argmin(medians))
    ax.annotate(
        f"Peak: {medians[peak_hour]:.1f} µg/m³",
        xy=(peak_hour, medians[peak_hour]),
        xytext=(peak_hour + 2, medians[peak_hour] + 5),
        arrowprops=dict(arrowstyle="->", color=ACCENT_RED),
        fontsize=9, color=ACCENT_RED, fontweight="bold",
    )
    ax.annotate(
        f"Trough: {medians[trough_hour]:.1f} µg/m³",
        xy=(trough_hour, medians[trough_hour]),
        xytext=(trough_hour + 2, medians[trough_hour] - 6),
        arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN),
        fontsize=9, color=ACCENT_GREEN, fontweight="bold",
    )

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=12)
    ax.set_title(
        "PM2.5 Distribution by Hour — Daily Seasonal Pattern (Diurnal Cycle)",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}h" for h in range(24)])
    ax.grid(True, axis="y", alpha=0.3)

    # WHO guideline
    ax.axhline(y=25, color=ACCENT_ORANGE, linestyle="--", linewidth=1, alpha=0.6, label="WHO 24h guideline (25 µg/m³)")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = EDA_DIR / "05b_boxplot_hourly.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Peak hour: {peak_hour}h (median={medians[peak_hour]:.1f})", flush=True)
    print(f"  Trough hour: {trough_hour}h (median={medians[trough_hour]:.1f})", flush=True)
    print(f"  Saved: {out_path}", flush=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Q-Q Plot (P1-1 — done early)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_qq_plot(df: pd.DataFrame) -> dict:
    """Generate Q-Q plot for PM2.5 — check normality assumption.

    Reference: Peixeiro Ch.6 pp.116-124
    """
    print("\n[P1-1] Q-Q Plot (normality check)...", flush=True)

    pm25 = df["pm25"].dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Q-Q Plot (raw)
    stats.probplot(pm25, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot — PM2.5 (Raw)", fontsize=12, fontweight="bold")
    axes[0].get_lines()[0].set(color=ACCENT_BLUE, markersize=2, alpha=0.5)
    axes[0].get_lines()[1].set(color=ACCENT_RED, linewidth=2)

    # Panel 2: Q-Q Plot (log-transformed)
    pm25_log = np.log1p(pm25)
    stats.probplot(pm25_log, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot — PM2.5 (Log-Transformed)", fontsize=12, fontweight="bold")
    axes[1].get_lines()[0].set(color=ACCENT_GREEN, markersize=2, alpha=0.5)
    axes[1].get_lines()[1].set(color=ACCENT_RED, linewidth=2)

    plt.tight_layout()
    out_path = EDA_DIR / "03c_qq_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Shapiro-Wilk test (subsample for speed — max 5000)
    sample = pm25[: min(5000, len(pm25))]
    _, p_raw = stats.shapiro(sample)
    _, p_log = stats.shapiro(np.log1p(sample))

    print(f"  Shapiro-Wilk (raw): p={p_raw:.2e} → {'Normal' if p_raw > 0.05 else 'NOT Normal'}", flush=True)
    print(f"  Shapiro-Wilk (log): p={p_log:.2e} → {'Normal' if p_log > 0.05 else 'NOT Normal'}", flush=True)
    print(f"  Saved: {out_path}", flush=True)

    return {
        "shapiro_p_raw": float(f"{p_raw:.6e}"),
        "shapiro_p_log": float(f"{p_log:.6e}"),
        "is_normal_raw": bool(p_raw > 0.05),
        "is_normal_log": bool(p_log > 0.05),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Periodogram / PSD (P1-2 — done early)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_periodogram(df: pd.DataFrame) -> dict:
    """Generate Periodogram / Power Spectral Density.

    Reference: Huang Ch.7, Manu Joseph Ch.3 (Fourier)
    Validates that Fourier features capture the right frequencies.
    """
    print("\n[P1-2] Periodogram / PSD...", flush=True)

    pm25 = df["pm25"].dropna().values

    # Compute periodogram (fs=1 sample/hour)
    freqs, psd = signal.periodogram(pm25, fs=1.0, scaling="density")

    # Convert frequency to period (hours)
    with np.errstate(divide="ignore"):
        periods = 1.0 / freqs
    periods[0] = np.inf  # DC component

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Panel 1: PSD vs Frequency
    axes[0].semilogy(freqs[1:], psd[1:], color=ACCENT_BLUE, linewidth=0.8, alpha=0.8)
    axes[0].set_xlabel("Frequency (cycles/hour)")
    axes[0].set_ylabel("Power Spectral Density")
    axes[0].set_title("Power Spectral Density — PM2.5", fontsize=13, fontweight="bold", pad=10)
    axes[0].grid(True, alpha=0.3)

    bbox_props = annotation_bbox(THEME_MODE)

    # Mark dominant frequencies
    # Daily: f = 1/24
    axes[0].axvline(x=1 / 24, color=ACCENT_ORANGE, linestyle="--", linewidth=1.5, alpha=0.8)
    axes[0].text(1 / 24 + 0.005, 0.95, "24h\n(daily)", color=ACCENT_ORANGE, fontsize=9,
                 transform=axes[0].get_xaxis_transform(), bbox=bbox_props)

    # 12h harmonic
    axes[0].axvline(x=1 / 12, color=ACCENT_GREEN, linestyle="--", linewidth=1, alpha=0.6)
    axes[0].text(1 / 12 + 0.005, 0.80, "12h\n(semi-daily)", color=ACCENT_GREEN, fontsize=8,
                 transform=axes[0].get_xaxis_transform(), bbox=bbox_props)

    # 8h harmonic
    axes[0].axvline(x=1 / 8, color=ACCENT_PURPLE, linestyle="--", linewidth=1, alpha=0.5)
    axes[0].text(1 / 8 + 0.005, 0.65, "8h", color=ACCENT_PURPLE, fontsize=8,
                 transform=axes[0].get_xaxis_transform(), bbox=bbox_props)

    # Panel 2: PSD vs Period (more intuitive)
    mask = (periods > 1) & (periods < 200) & np.isfinite(periods)
    axes[1].plot(periods[mask], psd[mask], color=ACCENT_BLUE, linewidth=0.8, alpha=0.8)
    axes[1].set_xlabel("Period (hours)")
    axes[1].set_ylabel("Power Spectral Density")
    axes[1].set_title("PSD vs Period — Dominant Cycles", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    # Mark key periods
    for period, label, color in [
        (24, "24h (daily)", ACCENT_ORANGE),
        (12, "12h (semi-daily)", ACCENT_GREEN),
        (168, "168h (weekly)", ACCENT_RED),
        (8, "8h (tri-daily)", ACCENT_PURPLE),
    ]:
        axes[1].axvline(x=period, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
        axes[1].text(period * 1.03, 1.02, label, color=color, fontsize=8, rotation=90, va="bottom",
                     transform=axes[1].get_xaxis_transform())

    plt.tight_layout()
    out_path = EDA_DIR / "05c_periodogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Find top 5 dominant periods
    top_indices = np.argsort(psd[1:])[::-1][:5] + 1
    dominant = []
    for idx in top_indices:
        if np.isfinite(periods[idx]) and periods[idx] > 1:
            dominant.append({
                "period_hours": round(float(periods[idx]), 1),
                "frequency": round(float(freqs[idx]), 6),
                "power": round(float(psd[idx]), 2),
            })

    print(f"  Top dominant periods: {[d['period_hours'] for d in dominant[:5]]}", flush=True)
    print(f"  Saved: {out_path}", flush=True)

    return {"dominant_periods": dominant}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    print("=" * 60, flush=True)
    print("Phase 1: EDA Audit — Generating Missing Charts", flush=True)
    print("=" * 60, flush=True)

    df = load_data()

    # P0-1: STL Decomposition
    stl_results = generate_stl_decomposition(df)

    # P0-2: Forecastability Assessment
    forecast_results = generate_forecastability(df, stl_results)

    # P0-6: Box Plot per Hour
    generate_hourly_boxplot(df)

    # P1-1: Q-Q Plot (bonus — done early)
    qq_results = generate_qq_plot(df)

    # P1-2: Periodogram (bonus — done early)
    psd_results = generate_periodogram(df)

    # ── Save all metrics to JSON ──
    audit_metrics = {
        "stl_decomposition": stl_results,
        "forecastability": forecast_results,
        "normality": qq_results,
        "spectral_analysis": psd_results,
    }

    out_json = EDA_DIR / "audit_phase1_metrics.json"
    with open(out_json, "w") as f:
        json.dump(audit_metrics, f, indent=2, default=str)
    print(f"\n[Phase1] Metrics saved: {out_json}", flush=True)

    # ── Update existing eda_results.json ──
    existing_json = EDA_DIR / "eda_results.json"
    if existing_json.exists():
        with open(existing_json) as f:
            existing = json.load(f)
        existing["stl"] = stl_results
        existing["forecastability"] = forecast_results
        existing["normality"] = qq_results
        existing["spectral"] = psd_results
        with open(existing_json, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        print(f"[Phase1] Updated: {existing_json}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Phase 1 COMPLETE — All charts generated!", flush=True)
    print("=" * 60, flush=True)

    return audit_metrics


if __name__ == "__main__":
    main()
