"""Export ARIMA and SARIMA predictions for AVP dashboard integration.

Runs rolling forecast for ARIMA (fast) and SARIMA (slow, eval_step=1) at each
horizon and saves aligned prediction arrays for precompute_avp.py to merge.

IMPORTANT: Uses the EXACT same data pipeline as arima_multi_horizon.py.
SARIMA eval_step=1 will take ~2-3 hours total.

Usage:
    export OMP_NUM_THREADS=1
    uv run python scripts/export_arima_preds.py 2>&1 | tee research/logs/export_arima_preds.log
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*frequency information.*")
warnings.filterwarnings("ignore", message=".*No supported index.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "research" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 6, 24]
ARIMA_WINDOW = 720  # 30 days rolling window


def prepare_data() -> pd.DataFrame:
    """Load → clean → impute (same as arima_multi_horizon.py)."""
    from src.data.cleaner import (
        _clip_physical_bounds,
        _handle_outliers,
        _remove_duplicates,
        _resample,
        _set_datetime_index,
    )
    from src.data.imputer import impute_missing_data
    from src.data.loader import load_raw_data

    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")

    df_hybrid = impute_missing_data(
        df,
        strategy="hybrid",
        max_gap_interp=6,
        max_gap_ml=24,
        knn_neighbors=5,
        verbose=True,
    )
    return df_hybrid


def find_orders(pm25: pd.Series) -> tuple:
    """Auto-detect ARIMA and SARIMA orders (same as arima_multi_horizon.py)."""
    import pmdarima as pm

    n_train = int(len(pm25) * 0.8)
    train_subset = pm25.iloc[max(0, n_train - 2000) : n_train]

    print(f"  Auto-ARIMA on {len(train_subset)} samples...", flush=True)
    t0 = time.time()

    auto_arima = pm.auto_arima(
        train_subset,
        start_p=0, start_q=0, max_p=5, max_q=5,
        d=None, seasonal=False, stepwise=True,
        suppress_warnings=True, error_action="ignore", trace=False,
    )
    arima_order = auto_arima.order
    print(f"  ARIMA order: {arima_order} (AIC={auto_arima.aic():.1f}) [{time.time() - t0:.1f}s]", flush=True)

    print(f"  Auto-SARIMA (s=24) on {min(len(train_subset), 1000)} samples...", flush=True)
    t0 = time.time()
    sarima_subset = train_subset.iloc[-1000:]

    try:
        auto_sarima = pm.auto_arima(
            sarima_subset,
            start_p=0, start_q=0, max_p=3, max_q=3,
            d=None, seasonal=True, m=24,
            start_P=0, start_Q=0, max_P=2, max_Q=2, D=1,
            stepwise=True, suppress_warnings=True,
            error_action="ignore", trace=False, n_fits=30,
        )
        sarima_order = auto_sarima.order
        sarima_seasonal = auto_sarima.seasonal_order
        print(
            f"  SARIMA order: {sarima_order}×{sarima_seasonal} "
            f"(AIC={auto_sarima.aic():.1f}) [{time.time() - t0:.1f}s]",
            flush=True,
        )
    except Exception as e:
        print(f"  ⚠️ SARIMA auto-fit failed: {e}", flush=True)
        sarima_order = (1, 1, 1)
        sarima_seasonal = (1, 1, 0, 24)

    return arima_order, sarima_order, sarima_seasonal


def export_arima_horizon(
    pm25: pd.Series,
    is_imputed: pd.Series,
    horizon: int,
    arima_order: tuple,
) -> dict | None:
    """Export ARIMA predictions for a single horizon (eval_step=1)."""
    from statsmodels.tsa.arima.model import ARIMA

    from src.data.loader import TARGET_COL

    print(f"\n{'=' * 60}")
    print(f"  ARIMA Export: Horizon {horizon}h (eval_step=1)")
    print(f"{'=' * 60}", flush=True)

    n = len(pm25)
    val_end = int(n * 0.9)

    # Build AVP-aligned test index list
    avp_test_indices = []
    for i in range(val_end, n - horizon):
        if is_imputed.iloc[i + horizon]:
            continue
        avp_test_indices.append(i)

    print(f"  Test points (real only): {len(avp_test_indices)}", flush=True)

    # Rolling forecast
    t0 = time.time()
    preds_dict = {}  # i -> predicted value

    for count, pos in enumerate(avp_test_indices):
        window_start = max(0, pos - ARIMA_WINDOW)
        train_window = pm25.iloc[window_start : pos + 1].dropna()

        if len(train_window) < 50:
            continue

        train_vals = train_window.values

        try:
            model = ARIMA(train_vals, order=arima_order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=horizon)
            pred = float(forecast[-1])

            actual = pm25.iloc[pos + horizon]
            if not np.isnan(actual) and not np.isnan(pred):
                preds_dict[pos] = pred
        except Exception:
            continue

        if (count + 1) % 100 == 0 or count == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (count + 1)) * (len(avp_test_indices) - count - 1)
            print(
                f"    [{count + 1}/{len(avp_test_indices)}] "
                f"{elapsed:.0f}s elapsed, ETA ~{eta:.0f}s",
                flush=True,
            )

    arima_time = time.time() - t0

    # Align with AVP cache
    aligned_preds = []
    aligned_actuals = []
    for pos in avp_test_indices:
        if pos in preds_dict:
            aligned_preds.append(preds_dict[pos])
            aligned_actuals.append(float(pm25.iloc[pos + horizon]))
        else:
            aligned_preds.append(None)
            aligned_actuals.append(float(pm25.iloc[pos + horizon]))

    n_filled = sum(1 for p in aligned_preds if p is not None)

    # Compute metrics on valid predictions
    valid_preds = np.array([p for p in aligned_preds if p is not None])
    valid_actuals = np.array([
        a for p, a in zip(aligned_preds, aligned_actuals) if p is not None
    ])
    mae = float(np.mean(np.abs(valid_actuals - valid_preds))) if len(valid_preds) > 0 else None
    rmse = float(np.sqrt(np.mean((valid_actuals - valid_preds) ** 2))) if len(valid_preds) > 0 else None

    print(
        f"  ✅ ARIMA {horizon}h: MAE={mae:.3f} | RMSE={rmse:.3f} | "
        f"n={n_filled}/{len(aligned_preds)} ({arima_time:.0f}s)",
        flush=True,
    )

    # Save
    out_path = CACHE_DIR / f"arima_preds_{horizon}h.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "ARIMA",
            "horizon": horizon,
            "order": str(arima_order),
            "predictions": aligned_preds,
            "n_total": len(aligned_preds),
            "n_valid": n_filled,
            "mae": round(mae, 4) if mae else None,
            "rmse": round(rmse, 4) if rmse else None,
            "time_s": round(arima_time, 1),
        }, f)
    print(f"  ✅ Saved: {out_path}", flush=True)

    return {"mae": mae, "rmse": rmse, "n": n_filled}


def export_sarima_horizon(
    pm25: pd.Series,
    is_imputed: pd.Series,
    horizon: int,
    sarima_order: tuple,
    sarima_seasonal: tuple,
) -> dict | None:
    """Export SARIMA predictions for a single horizon (eval_step=1, SLOW)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    print(f"\n{'=' * 60}")
    print(f"  SARIMA Export: Horizon {horizon}h (eval_step=1, SLOW)")
    print(f"  Order: {sarima_order}×{sarima_seasonal}")
    print(f"{'=' * 60}", flush=True)

    n = len(pm25)
    val_end = int(n * 0.9)

    # Build AVP-aligned test index list
    avp_test_indices = []
    for i in range(val_end, n - horizon):
        if is_imputed.iloc[i + horizon]:
            continue
        avp_test_indices.append(i)

    print(f"  Test points (real only): {len(avp_test_indices)}", flush=True)
    print(f"  ⏳ This will take ~{len(avp_test_indices) * 0.8 / 60:.0f}-{len(avp_test_indices) * 2.0 / 60:.0f} min", flush=True)

    # Rolling forecast
    t0 = time.time()
    preds_dict = {}

    for count, pos in enumerate(avp_test_indices):
        window_start = max(0, pos - ARIMA_WINDOW)
        train_window = pm25.iloc[window_start : pos + 1].dropna()

        if len(train_window) < 72:  # Need at least 3 days for seasonality
            continue

        train_vals = train_window.values

        try:
            model = SARIMAX(
                train_vals,
                order=sarima_order,
                seasonal_order=sarima_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=50)
            forecast = fitted.forecast(steps=horizon)
            pred = float(forecast[-1])

            actual = pm25.iloc[pos + horizon]
            if not np.isnan(actual) and not np.isnan(pred) and abs(pred) < 500:
                preds_dict[pos] = pred
        except Exception:
            continue

        if (count + 1) % 50 == 0 or count == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (count + 1)) * (len(avp_test_indices) - count - 1)
            print(
                f"    [{count + 1}/{len(avp_test_indices)}] "
                f"{elapsed:.0f}s elapsed, ETA ~{eta:.0f}s",
                flush=True,
            )

    sarima_time = time.time() - t0

    # Align
    aligned_preds = []
    aligned_actuals = []
    for pos in avp_test_indices:
        if pos in preds_dict:
            aligned_preds.append(preds_dict[pos])
            aligned_actuals.append(float(pm25.iloc[pos + horizon]))
        else:
            aligned_preds.append(None)
            aligned_actuals.append(float(pm25.iloc[pos + horizon]))

    n_filled = sum(1 for p in aligned_preds if p is not None)
    valid_preds = np.array([p for p in aligned_preds if p is not None])
    valid_actuals = np.array([
        a for p, a in zip(aligned_preds, aligned_actuals) if p is not None
    ])
    mae = float(np.mean(np.abs(valid_actuals - valid_preds))) if len(valid_preds) > 0 else None
    rmse = float(np.sqrt(np.mean((valid_actuals - valid_preds) ** 2))) if len(valid_preds) > 0 else None

    print(
        f"  ✅ SARIMA {horizon}h: MAE={mae:.3f} | RMSE={rmse:.3f} | "
        f"n={n_filled}/{len(aligned_preds)} ({sarima_time:.0f}s)",
        flush=True,
    )

    out_path = CACHE_DIR / f"sarima_preds_{horizon}h.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "SARIMA",
            "horizon": horizon,
            "order": str(sarima_order),
            "seasonal_order": str(sarima_seasonal),
            "predictions": aligned_preds,
            "n_total": len(aligned_preds),
            "n_valid": n_filled,
            "mae": round(mae, 4) if mae else None,
            "rmse": round(rmse, 4) if rmse else None,
            "time_s": round(sarima_time, 1),
        }, f)
    print(f"  ✅ Saved: {out_path}", flush=True)

    return {"mae": mae, "rmse": rmse, "n": n_filled}


def main():
    t_start = time.time()
    print("=" * 60)
    print("  EXPORT ARIMA + SARIMA PREDICTIONS FOR AVP DASHBOARD")
    print(f"  Horizons: {HORIZONS}")
    print("  SARIMA eval_step=1 (full coverage, SLOW)")
    print("=" * 60, flush=True)

    # Prepare data
    print("\n[1/4] Preparing data...", flush=True)
    df_hybrid = prepare_data()
    from src.data.loader import TARGET_COL
    pm25 = df_hybrid[TARGET_COL].copy()
    is_imputed = df_hybrid["is_imputed"].copy()
    print(f"  Data: {len(pm25)} rows", flush=True)

    # Find orders
    print("\n[2/4] Auto-detecting ARIMA/SARIMA orders...", flush=True)
    arima_order, sarima_order, sarima_seasonal = find_orders(pm25)

    # ARIMA export (fast)
    print("\n[3/4] Exporting ARIMA predictions...", flush=True)
    for h in HORIZONS:
        export_arima_horizon(pm25, is_imputed, h, arima_order)

    # SARIMA export (SLOW — last)
    print("\n[4/4] Exporting SARIMA predictions (SLOW)...", flush=True)
    for h in HORIZONS:
        export_sarima_horizon(pm25, is_imputed, h, sarima_order, sarima_seasonal)

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  All done in {total:.0f}s ({total / 60:.1f} min)")
    print(f"  Files saved in: {CACHE_DIR}")
    for model in ["arima", "sarima"]:
        for h in HORIZONS:
            p = CACHE_DIR / f"{model}_preds_{h}h.json"
            if p.exists():
                print(f"    ✅ {p.name} ({p.stat().st_size / 1024:.0f} KB)")
            else:
                print(f"    ❌ {p.name} (not created)")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
