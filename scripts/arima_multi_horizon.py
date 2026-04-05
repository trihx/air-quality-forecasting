"""ARIMA/SARIMA Multi-Horizon Evaluation — Level 1 Statistical Models.

Compare ARIMA/SARIMA with Persistence and LightGBM (from multi_horizon_eval)
at 1h, 6h, 24h horizons.

ARIMA = univariate (only PM2.5 history)
SARIMA = univariate + seasonal component (period=24h)

Strategy: Hybrid imputation | Test = REAL data only.

Usage:
    uv run python scripts/arima_multi_horizon.py 2>&1 | tee research/logs/arima_multi_horizon.log
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.data.cleaner import (
    _clip_physical_bounds,
    _handle_outliers,
    _remove_duplicates,
    _resample,
    _set_datetime_index,
)
from src.data.imputer import impute_missing_data
from src.data.loader import TARGET_COL, load_raw_data

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*frequency information.*")
warnings.filterwarnings("ignore", message=".*No supported index.*")

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "arima"
HORIZONS = [1, 6, 24]

# Rolling forecast window: last N hours for ARIMA fit
ARIMA_WINDOW = 720  # 30 days — balance between accuracy and speed
# How many test points to evaluate (subsample for speed)
# SARIMA is very slow — evaluate every Nth test point
SARIMA_EVAL_STEP = 6  # evaluate every 6th test point for SARIMA
ARIMA_EVAL_STEP = 1   # evaluate every test point for ARIMA (fast)


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("ARIMA/SARIMA MULTI-HORIZON EVALUATION", flush=True)
    print(f"Horizons: {HORIZONS}h | Window: {ARIMA_WINDOW}h", flush=True)
    print("RULE: Test set = REAL data only", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Prepare Hybrid dataset ──
    print("\n[1/5] Preparing Hybrid dataset...", flush=True)
    df_hybrid = _prepare_hybrid_data()
    pm25 = df_hybrid[TARGET_COL].copy()
    is_imputed = df_hybrid["is_imputed"].copy()
    print(f"  PM2.5 series: {len(pm25):,} rows", flush=True)

    # ── Step 2: Auto-ARIMA order selection ──
    print("\n[2/5] Finding best ARIMA/SARIMA orders...", flush=True)
    arima_order, sarima_order, sarima_seasonal = _find_orders(pm25)

    # ── Step 3: Multi-horizon evaluation ──
    all_results = {}

    for h in HORIZONS:
        print(f"\n{'═' * 70}", flush=True)
        print(f"[3/5] HORIZON = {h}h", flush=True)
        print(f"{'═' * 70}", flush=True)

        results_h = _evaluate_horizon(pm25, is_imputed, h, arima_order, sarima_order, sarima_seasonal)
        all_results[f"{h}h"] = results_h

    # ── Step 4: Summary ──
    print(f"\n{'═' * 70}", flush=True)
    print("[4/5] ARIMA/SARIMA MULTI-HORIZON SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)

    # Include LightGBM results from previous experiment for comparison
    # Reference from multi_horizon v2 (post-audit ground truth)
    lgbm_results = {
        "1h": {"mae": 3.720, "mase": 1.492},
        "6h": {"mae": 5.046, "mase": 0.745},
        "24h": {"mae": 5.179, "mase": 0.842},
    }

    print(f"\n{'Horizon':<10} {'Model':<25} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'Status':>12}", flush=True)
    print("─" * 80, flush=True)

    for h in HORIZONS:
        h_key = f"{h}h"
        # Persistence
        p = all_results[h_key].get("Persistence", {})
        print(f"{h}h{'':<7} {'Persistence':<25} {p.get('mae', 0):>8.3f} {p.get('rmse', 0):>8.3f} {'1.000':>8} {'baseline':>12}", flush=True)

        # ARIMA
        a = all_results[h_key].get("ARIMA", {})
        a_status = "✅ BEATS!" if a.get("mase", 99) < 1.0 else "❌ MASE>1"
        print(f"{h}h{'':<7} {'ARIMA':<25} {a.get('mae', 0):>8.3f} {a.get('rmse', 0):>8.3f} {a.get('mase', 0):>8.3f} {a_status:>12}", flush=True)

        # SARIMA
        s = all_results[h_key].get("SARIMA", {})
        if s:
            s_status = "✅ BEATS!" if s.get("mase", 99) < 1.0 else "❌ MASE>1"
            print(f"{h}h{'':<7} {'SARIMA':<25} {s.get('mae', 0):>8.3f} {s.get('rmse', 0):>8.3f} {s.get('mase', 0):>8.3f} {s_status:>12}", flush=True)

        # LightGBM (reference from previous experiment)
        l = lgbm_results[h_key]
        l_status = "✅ BEATS!" if l["mase"] < 1.0 else "❌ MASE>1"
        print(f"{h}h{'':<7} {'LightGBM_tuned (ref)':<25} {l['mae']:>8.3f} {'—':>8} {l['mase']:>8.3f} {l_status:>12}", flush=True)

        print("─" * 80, flush=True)

    # ── Step 5: Save ──
    print(f"\n[5/5] Saving results...", flush=True)
    _save_results(all_results)

    total_time = time.time() - t_start
    print(f"\n{'═' * 70}", flush=True)
    print(f"COMPLETE — Total: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print(f"{'═' * 70}", flush=True)


def _prepare_hybrid_data() -> pd.DataFrame:
    """Load raw data and apply Hybrid imputation strategy."""
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")

    df_hybrid = impute_missing_data(
        df, strategy="hybrid",
        max_gap_interp=6, max_gap_ml=24, knn_neighbors=5,
        verbose=True,
    )
    return df_hybrid


def _find_orders(pm25: pd.Series) -> tuple:
    """Use auto_arima to find best ARIMA and SARIMA orders."""
    import pmdarima as pm

    # Use a subset for order selection (last 2000 points of training data)
    n_train = int(len(pm25) * 0.8)
    train_subset = pm25.iloc[max(0, n_train - 2000):n_train]

    print(f"  Auto-ARIMA on {len(train_subset)} samples...", flush=True)
    t0 = time.time()

    # Non-seasonal ARIMA
    auto_arima = pm.auto_arima(
        train_subset,
        start_p=0, start_q=0, max_p=5, max_q=5,
        d=None,  # auto detect
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    arima_order = auto_arima.order
    print(f"  ARIMA order: {arima_order} (AIC={auto_arima.aic():.1f}) [{time.time() - t0:.1f}s]", flush=True)

    # Seasonal SARIMA (s=24 for hourly)
    print(f"  Auto-SARIMA (s=24) on {min(len(train_subset), 1000)} samples...", flush=True)
    t0 = time.time()

    # Use smaller subset for SARIMA (very slow with s=24)
    sarima_subset = train_subset.iloc[-1000:]

    try:
        auto_sarima = pm.auto_arima(
            sarima_subset,
            start_p=0, start_q=0, max_p=3, max_q=3,
            d=None,
            seasonal=True, m=24,
            start_P=0, start_Q=0, max_P=2, max_Q=2, D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            n_fits=30,  # limit fits for speed
        )
        sarima_order = auto_sarima.order
        sarima_seasonal = auto_sarima.seasonal_order
        print(f"  SARIMA order: {sarima_order}×{sarima_seasonal} (AIC={auto_sarima.aic():.1f}) [{time.time() - t0:.1f}s]", flush=True)
    except Exception as e:
        print(f"  ⚠️ SARIMA auto-fit failed: {e}", flush=True)
        print(f"  Using default SARIMA(1,1,1)(1,1,0,24)", flush=True)
        sarima_order = (1, 1, 1)
        sarima_seasonal = (1, 1, 0, 24)

    return arima_order, sarima_order, sarima_seasonal


def _evaluate_horizon(
    pm25: pd.Series,
    is_imputed: pd.Series,
    horizon: int,
    arima_order: tuple,
    sarima_order: tuple,
    sarima_seasonal: tuple,
) -> dict:
    """Evaluate ARIMA and SARIMA at a specific forecast horizon."""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    results = {}

    # ── Temporal split ──
    n = len(pm25)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    test_series = pm25.iloc[val_end:]
    test_imputed = is_imputed.iloc[val_end:]

    # Real test indices only
    real_mask = ~test_imputed.values
    real_test_idx = test_series.index[real_mask]

    # For multi-step: we need target = pm25[t + horizon]
    # So we evaluate at test points where pm25[t + horizon] exists
    valid_indices = []
    for i, idx in enumerate(real_test_idx):
        future_idx = pm25.index.get_loc(idx)
        if future_idx + horizon < len(pm25):
            valid_indices.append((i, idx, future_idx))

    print(f"\n  Test points (real only): {len(valid_indices)}/{len(test_series)}", flush=True)

    if len(valid_indices) < 5:
        print(f"  ⚠️ Too few valid test points, skipping horizon {horizon}h", flush=True)
        return results

    # ── A. Persistence Baseline ──
    print(f"  Evaluating Persistence baseline ({horizon}h)...", flush=True)
    y_true_list = []
    y_persist_list = []

    for _, idx, pos in valid_indices:
        actual = pm25.iloc[pos + horizon] if pos + horizon < len(pm25) else np.nan
        persist = pm25.iloc[pos]  # last known value
        if not np.isnan(actual) and not np.isnan(persist):
            y_true_list.append(actual)
            y_persist_list.append(persist)

    y_true = np.array(y_true_list)
    y_persist = np.array(y_persist_list)

    persist_mae = float(np.mean(np.abs(y_true - y_persist)))
    persist_rmse = float(np.sqrt(np.mean((y_true - y_persist) ** 2)))
    results["Persistence"] = {"mae": round(persist_mae, 4), "rmse": round(persist_rmse, 4), "mase": 1.0}
    print(f"    Persistence {horizon}h: MAE={persist_mae:.3f}, RMSE={persist_rmse:.3f}", flush=True)

    # ── B. ARIMA (rolling window) ──
    print(f"\n  ARIMA{arima_order} rolling forecast ({horizon}h)...", flush=True)
    t0 = time.time()

    y_arima_list = []
    y_true_arima = []
    n_eval = len(valid_indices)
    eval_points = list(range(0, n_eval, ARIMA_EVAL_STEP))

    for count, eval_i in enumerate(eval_points):
        _, idx, pos = valid_indices[eval_i]

        # Sliding window for training
        window_start = max(0, pos - ARIMA_WINDOW)
        train_window = pm25.iloc[window_start:pos + 1].dropna()

        if len(train_window) < 50:
            continue

        # Reset index to avoid statsmodels frequency warnings
        train_vals = train_window.values

        try:
            model = ARIMA(train_vals, order=arima_order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=horizon)
            pred = float(forecast[-1])

            actual = pm25.iloc[pos + horizon]
            if not np.isnan(actual) and not np.isnan(pred):
                y_arima_list.append(pred)
                y_true_arima.append(actual)
        except Exception:
            continue

        if (count + 1) % 50 == 0 or count == 0:
            elapsed = time.time() - t0
            print(f"    [{count + 1}/{len(eval_points)}] {elapsed:.0f}s elapsed", flush=True)

    arima_time = time.time() - t0

    if len(y_arima_list) > 0:
        y_true_a = np.array(y_true_arima)
        y_pred_a = np.array(y_arima_list)
        arima_mae = float(np.mean(np.abs(y_true_a - y_pred_a)))
        arima_rmse = float(np.sqrt(np.mean((y_true_a - y_pred_a) ** 2)))
        arima_mase = round(arima_mae / persist_mae, 4) if persist_mae > 0 else float("inf")

        results["ARIMA"] = {
            "mae": round(arima_mae, 4),
            "rmse": round(arima_rmse, 4),
            "mase": arima_mase,
            "order": str(arima_order),
            "n_eval": len(y_arima_list),
            "time_s": round(arima_time, 1),
        }
        print(f"    ✅ ARIMA {horizon}h: MAE={arima_mae:.3f}, MASE={arima_mase:.3f} ({arima_time:.0f}s, {len(y_arima_list)} points)", flush=True)
    else:
        print(f"    ⚠️ ARIMA failed all evaluations", flush=True)

    # ── C. SARIMA (subsampled rolling) ──
    print(f"\n  SARIMA{sarima_order}×{sarima_seasonal} rolling forecast ({horizon}h)...", flush=True)
    print(f"    (evaluating every {SARIMA_EVAL_STEP}th point for speed)", flush=True)
    t0 = time.time()

    y_sarima_list = []
    y_true_sarima = []
    sarima_eval_points = list(range(0, n_eval, SARIMA_EVAL_STEP))

    for count, eval_i in enumerate(sarima_eval_points):
        _, idx, pos = valid_indices[eval_i]

        # Larger window for SARIMA (needs seasonal data)
        window_start = max(0, pos - ARIMA_WINDOW)
        train_window = pm25.iloc[window_start:pos + 1].dropna()

        if len(train_window) < 72:  # at least 3 days for seasonality
            continue

        # Reset index to avoid statsmodels frequency warnings
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
                y_sarima_list.append(pred)
                y_true_sarima.append(actual)
        except Exception:
            continue

        if (count + 1) % 20 == 0 or count == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (count + 1)) * (len(sarima_eval_points) - count - 1)
            print(f"    [{count + 1}/{len(sarima_eval_points)}] {elapsed:.0f}s elapsed, ETA ~{eta:.0f}s", flush=True)

    sarima_time = time.time() - t0

    if len(y_sarima_list) > 0:
        y_true_s = np.array(y_true_sarima)
        y_pred_s = np.array(y_sarima_list)
        sarima_mae = float(np.mean(np.abs(y_true_s - y_pred_s)))
        sarima_rmse = float(np.sqrt(np.mean((y_true_s - y_pred_s) ** 2)))
        sarima_mase = round(sarima_mae / persist_mae, 4) if persist_mae > 0 else float("inf")

        results["SARIMA"] = {
            "mae": round(sarima_mae, 4),
            "rmse": round(sarima_rmse, 4),
            "mase": sarima_mase,
            "order": str(sarima_order),
            "seasonal_order": str(sarima_seasonal),
            "n_eval": len(y_sarima_list),
            "time_s": round(sarima_time, 1),
        }
        print(f"    ✅ SARIMA {horizon}h: MAE={sarima_mae:.3f}, MASE={sarima_mase:.3f} ({sarima_time:.0f}s, {len(y_sarima_list)} points)", flush=True)
    else:
        print(f"    ⚠️ SARIMA failed all evaluations", flush=True)

    return results


def _save_results(all_results: dict) -> None:
    """Save results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"arima_multi_horizon_{timestamp}.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"  Results saved: {json_path}", flush=True)


if __name__ == "__main__":
    main()
