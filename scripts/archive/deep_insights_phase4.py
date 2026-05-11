"""Phase 4: Deep Insights — P1-3, P1-4, P1-5.

P1-3: Error Anatomy (Error vs Hour-of-Day, Error vs PM2.5 Level)
P1-4: Granger Causality Test (do external vars help predict PM2.5?)
P1-5: Cross-Correlation Lagged (PM2.5 vs Temperature/Humidity)

All numbers output to JSON for Dashboard cross-reference.

Usage:
    uv run python scripts/deep_insights_phase4.py 2>&1 | tee research/logs/deep_insights.log
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.cleaner import (
    _clip_physical_bounds,
    _handle_outliers,
    _remove_duplicates,
    _resample,
    _set_datetime_index,
)
from src.data.imputer import impute_missing_data
from src.data.loader import TARGET_COL, load_raw_data
from src.features.builder import build_features

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
EDA_DIR = RESEARCH_DIR / "eda"
OUTPUT_JSON = EDA_DIR / "deep_insights_results.json"

# Style — VTF: Centralized theme (Light mode for publication)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.viz.theme import apply_mpl_theme

THEME_MODE = "light"
apply_mpl_theme(THEME_MODE)


def load_cleaned_data() -> pd.DataFrame:
    """Load and clean data to get full feature set with timestamps."""
    print("  Loading and cleaning...", flush=True)
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")

    df_hybrid = impute_missing_data(
        df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24,
        knn_neighbors=5, verbose=True,
    )

    is_imputed = df_hybrid["is_imputed"].copy()
    df_for_feat = df_hybrid.drop(columns=["is_imputed"])

    df_feat = build_features(df_for_feat, include_fourier=True, fourier_order=3)
    df_feat["is_imputed"] = is_imputed.reindex(df_feat.index).fillna(False)

    print(f"  Shape: {df_feat.shape}", flush=True)
    return df_feat


# ══════════════════════════════════════════════════════════════════════
# P1-3: Error Anatomy
# ══════════════════════════════════════════════════════════════════════


def run_error_anatomy(df_feat: pd.DataFrame, results: dict) -> None:
    """Error analysis by hour-of-day and PM2.5 level.

    Uses GRU v2+log at h=6 as the best model. Generates predictions
    via the same pipeline to get per-point errors with timestamps.
    """
    print("\n[P1-3] Error Anatomy...", flush=True)

    # Load precomputed AVP cache for 6h
    avp_path = RESEARCH_DIR / "cache" / "avp_6h.json"
    if not avp_path.exists():
        print("  ⚠️ avp_6h.json not found, skipping", flush=True)
        return

    with open(avp_path, encoding="utf-8") as f:
        avp = json.load(f)

    actuals = np.array(avp["actuals"])
    gru_preds = np.array(avp["gru_preds"])
    n_test = len(actuals)

    # Errors (signed and absolute)
    errors = gru_preds - actuals  # positive = over-predict
    abs_errors = np.abs(errors)

    # Get test set timestamps from data
    n_data = len(df_feat)
    val_end = int(n_data * 0.9)
    # AVP has n_test points from test set; align to timestamps
    test_idx = df_feat.index[val_end:val_end + n_test]
    hours = test_idx.hour

    # ── Error vs Hour-of-Day ──
    hour_mae = {}
    hour_bias = {}
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hour_mae[h] = float(np.mean(abs_errors[mask]))
            hour_bias[h] = float(np.mean(errors[mask]))

    # ── Error vs PM2.5 Level ──
    # Bin actuals into WHO levels
    bins = [0, 12, 35.4, 55.4, 100]
    labels = ["Good (0-12)", "Moderate (12-35)", "USG (35-55)", "Unhealthy (55+)"]
    actual_bins = pd.cut(actuals, bins=bins, labels=labels, right=True)

    level_mae = {}
    level_bias = {}
    level_count = {}
    for label in labels:
        mask = actual_bins == label
        if mask.sum() > 0:
            level_mae[label] = float(np.mean(abs_errors[mask]))
            level_bias[label] = float(np.mean(errors[mask]))
            level_count[label] = int(mask.sum())

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. MAE by Hour
    ax = axes[0, 0]
    sorted_hours = sorted(hour_mae.keys())
    mae_vals = [hour_mae[h] for h in sorted_hours]
    colors = ["#FF6B6B" if v > np.mean(mae_vals) else "#4ECDC4" for v in mae_vals]
    ax.bar(sorted_hours, mae_vals, color=colors, alpha=0.85, edgecolor="none")
    ax.axhline(np.mean(mae_vals), color="#FFD93D", ls="--", lw=1.5, label=f"Mean MAE={np.mean(mae_vals):.2f}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("MAE by Hour of Day (GRU v2+log, h=6)")
    ax.legend(fontsize=9)
    ax.set_xticks(range(0, 24, 2))

    # 2. Forecast Bias by Hour
    ax = axes[0, 1]
    bias_vals = [hour_bias[h] for h in sorted_hours]
    colors = ["#FF6B6B" if v < 0 else "#4ECDC4" for v in bias_vals]
    ax.bar(sorted_hours, bias_vals, color=colors, alpha=0.85, edgecolor="none")
    ax.axhline(0, color="#FFD93D", ls="--", lw=1)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Error (µg/m³)")
    ax.set_title("Forecast Bias by Hour (negative = under-forecast)")
    ax.set_xticks(range(0, 24, 2))

    # 3. MAE by PM2.5 Level
    ax = axes[1, 0]
    valid_labels = [l for l in labels if l in level_mae]
    mae_by_level = [level_mae[l] for l in valid_labels]
    counts = [level_count.get(l, 0) for l in valid_labels]
    bars = ax.bar(range(len(valid_labels)), mae_by_level, color=["#4ECDC4", "#FFD93D", "#FF6B6B", "#FF4757"],
                  alpha=0.85, edgecolor="none")
    ax.set_xticks(range(len(valid_labels)))
    ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(valid_labels, counts)], fontsize=8)
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("MAE by PM2.5 Level — Model tệ ở level nào?")

    # 4. Error autocorrelation
    ax = axes[1, 1]
    max_lag = min(48, len(errors) - 1)
    acf_vals = [np.corrcoef(errors[:-lag], errors[lag:])[0, 1] for lag in range(1, max_lag + 1)]
    ax.bar(range(1, max_lag + 1), acf_vals, color="#7C83FD", alpha=0.7, edgecolor="none")
    ax.axhline(1.96 / np.sqrt(len(errors)), color="#FFD93D", ls="--", lw=1, label="95% CI")
    ax.axhline(-1.96 / np.sqrt(len(errors)), color="#FFD93D", ls="--", lw=1)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Error Autocorrelation (structured = model missing pattern)")
    ax.legend(fontsize=9)

    plt.suptitle("P1-3: Error Anatomy — GRU v2+log @ h=6", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = EDA_DIR / "06_error_anatomy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}", flush=True)

    # Find worst/best hours
    worst_hour = max(hour_mae, key=hour_mae.get)
    best_hour = min(hour_mae, key=hour_mae.get)

    results["error_anatomy"] = {
        "model": "GRU_v2_log",
        "horizon": 6,
        "n_test": n_test,
        "hour_mae": {str(k): round(v, 3) for k, v in hour_mae.items()},
        "hour_bias": {str(k): round(v, 3) for k, v in hour_bias.items()},
        "level_mae": {k: round(v, 3) for k, v in level_mae.items()},
        "level_bias": {k: round(v, 3) for k, v in level_bias.items()},
        "level_count": level_count,
        "worst_hour": worst_hour,
        "best_hour": best_hour,
        "worst_hour_mae": round(hour_mae[worst_hour], 3),
        "best_hour_mae": round(hour_mae[best_hour], 3),
        "error_acf_lag1": round(acf_vals[0], 4) if acf_vals else None,
        "error_acf_lag24": round(acf_vals[23], 4) if len(acf_vals) > 23 else None,
    }
    print(f"  Worst hour: {worst_hour}h (MAE={hour_mae[worst_hour]:.3f})", flush=True)
    print(f"  Best hour: {best_hour}h (MAE={hour_mae[best_hour]:.3f})", flush=True)
    print(f"  Error ACF(1): {acf_vals[0]:.4f}, ACF(24): {acf_vals[23]:.4f}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# P1-4: Granger Causality Test
# ══════════════════════════════════════════════════════════════════════


def run_granger_causality(df_feat: pd.DataFrame, results: dict) -> None:
    """Granger Causality: do external vars help predict PM2.5?

    Ref: Peixeiro Ch.10 pp.201-203.
    Tests if lagged values of temp/humidity/co2 improve PM2.5 forecasts.
    """
    print("\n[P1-4] Granger Causality...", flush=True)
    from statsmodels.tsa.stattools import grangercausalitytests

    # Use cleaned hourly data (real values only)
    real_mask = ~df_feat["is_imputed"].astype(bool)
    df_real = df_feat[real_mask].copy()

    # Need continuous block for Granger — take the longest segment
    # Use train portion only (first 80%) to avoid test contamination
    n = len(df_real)
    train_end = int(n * 0.8)
    df_train = df_real.iloc[:train_end]

    # Variables to test
    external_vars = {
        "nhiet_do_lag_1h": "Temperature (lag 1h)",
        "do_am_lag_1h": "Humidity (lag 1h)",
        "co2_lag_1h": "CO2 (lag 1h)",
    }

    max_lag = 24
    granger_results = {}

    for col, label in external_vars.items():
        if col not in df_train.columns:
            print(f"  ⚠️ {col} not found, skipping", flush=True)
            continue

        test_df = df_train[[TARGET_COL, col]].dropna()
        if len(test_df) < max_lag * 3:
            print(f"  ⚠️ {col}: too few data points ({len(test_df)})", flush=True)
            continue

        print(f"  Testing: {label} → PM2.5 (n={len(test_df)}, maxlag={max_lag})...", flush=True)

        try:
            gc_result = grangercausalitytests(test_df, maxlag=max_lag, verbose=False)

            # Extract p-values for F-test
            p_values = {}
            for lag in range(1, max_lag + 1):
                p = gc_result[lag][0]["ssr_ftest"][1]
                p_values[lag] = round(p, 6)

            # Find best lag (lowest p-value)
            best_lag = min(p_values, key=p_values.get)
            best_p = p_values[best_lag]
            significant = best_p < 0.05

            granger_results[col] = {
                "label": label,
                "best_lag": best_lag,
                "best_p_value": best_p,
                "significant_at_005": significant,
                "p_values_selected": {
                    str(k): p_values[k] for k in [1, 6, 12, 24] if k in p_values
                },
            }

            status = "✅ SIGNIFICANT" if significant else "❌ NOT significant"
            print(f"    {status}: best_lag={best_lag}, p={best_p:.6f}", flush=True)

        except Exception as e:
            print(f"    ⚠️ Error: {e}", flush=True)
            granger_results[col] = {"label": label, "error": str(e)}

    results["granger_causality"] = granger_results

    # ── Plot Granger p-values ──
    fig, ax = plt.subplots(figsize=(12, 5))
    lags = list(range(1, max_lag + 1))

    for col, gr in granger_results.items():
        if "error" in gr:
            continue
        # Re-extract all p values
        label = gr["label"]
        # We need to re-run to get all lags — use stored selected ones + interpolate
        # Better: just plot the selected lags
        p_sel = gr.get("p_values_selected", {})
        if not p_sel:
            continue
        sel_lags = sorted([int(k) for k in p_sel.keys()])
        # Thêm floor 1e-15 để tránh lỗi log(0) trên trục Y
        sel_ps = [max(p_sel[str(l)], 1e-15) for l in sel_lags]
        ax.plot(sel_lags, sel_ps, "o-", label=f"{label}", markersize=6, lw=2, alpha=0.85)

    ax.axhline(0.05, color="#FF6B6B", ls="--", lw=1.5, label="α = 0.05")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("p-value (F-test)")
    ax.set_title("P1-4: Granger Causality — Do external vars help predict PM2.5?")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    out_path = EDA_DIR / "07_granger_causality.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# P1-5: Cross-Correlation Lagged
# ══════════════════════════════════════════════════════════════════════


def run_cross_correlation(df_feat: pd.DataFrame, results: dict) -> None:
    """Cross-correlation between PM2.5 and Temperature/Humidity.

    Ref: Huang Ch.3 — shows at which lag external vars most correlate with PM2.5.
    Useful to validate lag feature design.
    """
    print("\n[P1-5] Cross-Correlation...", flush=True)

    real_mask = ~df_feat["is_imputed"].astype(bool)
    df_real = df_feat[real_mask].copy()

    # Use train only
    n = len(df_real)
    train_end = int(n * 0.8)
    df_train = df_real.iloc[:train_end]

    pm25 = df_train[TARGET_COL].dropna().values
    max_lag = 72  # 3 days

    cc_results = {}
    variables = {
        "nhiet_do": "Temperature",
        "do_am": "Humidity",
        "co2": "CO2",
        "diem_suong": "Dew Point",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, (col, label) in enumerate(variables.items()):
        if col not in df_train.columns:
            print(f"  ⚠️ {col} not found", flush=True)
            continue

        # Align and drop NaN
        combined = df_train[[TARGET_COL, col]].dropna()
        pm25_vals = combined[TARGET_COL].values
        ext_vals = combined[col].values

        # Normalize
        pm25_norm = (pm25_vals - pm25_vals.mean()) / pm25_vals.std()
        ext_norm = (ext_vals - ext_vals.mean()) / ext_vals.std()

        # Cross-correlation at each lag
        cc_vals = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                cc = np.corrcoef(pm25_norm[lag:], ext_norm[:len(ext_norm) - lag])[0, 1] if lag < len(pm25_norm) else 0
            else:
                alag = abs(lag)
                cc = np.corrcoef(pm25_norm[:len(pm25_norm) - alag], ext_norm[alag:])[0, 1] if alag < len(ext_norm) else 0
            cc_vals.append(cc)

        lags = list(range(-max_lag, max_lag + 1))

        # Find optimal lag (highest absolute correlation)
        abs_cc = [abs(c) for c in cc_vals]
        best_idx = np.argmax(abs_cc)
        best_lag = lags[best_idx]
        best_cc = cc_vals[best_idx]

        cc_results[col] = {
            "label": label,
            "best_lag_hours": best_lag,
            "best_correlation": round(best_cc, 4),
            "cc_at_lag0": round(cc_vals[max_lag], 4),  # lag=0 index
            "cc_at_lag1": round(cc_vals[max_lag + 1], 4) if max_lag + 1 < len(cc_vals) else None,
            "cc_at_lag6": round(cc_vals[max_lag + 6], 4) if max_lag + 6 < len(cc_vals) else None,
            "cc_at_lag24": round(cc_vals[max_lag + 24], 4) if max_lag + 24 < len(cc_vals) else None,
        }

        print(f"  {label}: best_lag={best_lag}h, r={best_cc:.4f}, r(lag0)={cc_vals[max_lag]:.4f}", flush=True)

        # Plot
        ax = axes_flat[idx]
        ax.plot(lags, cc_vals, color="#7C83FD", lw=1.5, alpha=0.8)
        ax.axvline(best_lag, color="#FF6B6B", ls="--", lw=1, alpha=0.8, label=f"Best lag={best_lag}h")
        ax.axvline(0, color="#FFD93D", ls=":", lw=1, alpha=0.5)
        ax.axhline(0, color="#8B95A5", ls="-", lw=0.5)
        ax.fill_between(lags, -1.96 / np.sqrt(len(pm25_norm)), 1.96 / np.sqrt(len(pm25_norm)),
                        color="#FFD93D", alpha=0.1, label="95% CI")
        ax.set_xlabel("Lag (hours) — positive = ext leads PM2.5")
        ax.set_ylabel("Cross-Correlation")
        ax.set_title(f"PM2.5 × {label} (r={best_cc:.3f} @ lag {best_lag}h)")
        ax.legend(fontsize=8)
        ax.set_xlim(-max_lag, max_lag)

    plt.suptitle("P1-5: Cross-Correlation — PM2.5 vs External Variables", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = EDA_DIR / "08_cross_correlation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}", flush=True)

    results["cross_correlation"] = cc_results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 60, flush=True)
    print("PHASE 4: DEEP INSIGHTS (P1-3, P1-4, P1-5)", flush=True)
    print("=" * 60, flush=True)

    print("\n[0] Loading data...", flush=True)
    df_feat = load_cleaned_data()

    results = {}

    # P1-3: Error Anatomy
    run_error_anatomy(df_feat, results)

    # P1-4: Granger Causality
    run_granger_causality(df_feat, results)

    # P1-5: Cross-Correlation
    run_cross_correlation(df_feat, results)

    # Save all results
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"\n  Results saved: {OUTPUT_JSON}", flush=True)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    ea = results.get("error_anatomy", {})
    if ea:
        print(f"  [P1-3] Worst hour: {ea['worst_hour']}h (MAE={ea['worst_hour_mae']})")
        print(f"  [P1-3] Best hour: {ea['best_hour']}h (MAE={ea['best_hour_mae']})")
        print(f"  [P1-3] Error ACF(1)={ea.get('error_acf_lag1')}, ACF(24)={ea.get('error_acf_lag24')}")

    gc = results.get("granger_causality", {})
    for col, gr in gc.items():
        if "error" not in gr:
            sig = "✅" if gr["significant_at_005"] else "❌"
            print(f"  [P1-4] {gr['label']}: Granger {sig} (best lag={gr['best_lag']}, p={gr['best_p_value']:.6f})")

    cc = results.get("cross_correlation", {})
    for col, cr in cc.items():
        print(f"  [P1-5] {cr['label']}: best lag={cr['best_lag_hours']}h, r={cr['best_correlation']}")

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
