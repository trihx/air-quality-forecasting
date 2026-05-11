"""Multi-Horizon Evaluation — Compare ML vs Persistence at 1h, 6h, 24h.

HYPOTHESIS: Persistence dominates at 1h (autocorr=0.97) but ML should win
at longer horizons where autocorrelation drops significantly.

Strategy: Use Hybrid imputation (best EDA strategy) + Optuna LightGBM.
Test data = REAL only.

Usage:
    uv run python scripts/multi_horizon_eval.py
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.data.cleaner import (
    _clip_physical_bounds,
    _handle_outliers,
    _remove_duplicates,
    _resample,
    _set_datetime_index,
)
from src.data.imputer import impute_missing_data
from src.data.loader import TARGET_COL, load_raw_data
from src.evaluation.metrics import evaluate_forecast
from src.features.builder import build_features

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "multi_horizon"
HORIZONS = [1, 6, 24]  # hours ahead
OPTUNA_TRIALS = 100  # per horizon


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("MULTI-HORIZON EVALUATION", flush=True)
    print(f"Horizons: {HORIZONS}h | Strategy: Hybrid | Model: LightGBM (Optuna)", flush=True)
    print("RULE: Test set = REAL data only", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Prepare Hybrid dataset (best EDA strategy) ──
    print("\n[1/5] Preparing Hybrid dataset...", flush=True)
    df_hybrid = _prepare_hybrid_data()
    print(f"  Hybrid data: {len(df_hybrid):,} rows", flush=True)

    # ── Step 2: Build features ──
    print("\n[2/5] Building features...", flush=True)
    is_imputed_col = df_hybrid["is_imputed"].copy()
    df_for_features = df_hybrid.drop(columns=["is_imputed"])

    df_feat = build_features(df_for_features)
    df_feat["is_imputed"] = is_imputed_col.reindex(df_feat.index).fillna(False)
    print(f"  Features: {len(df_feat):,} rows × {len(df_feat.columns)} cols", flush=True)

    # ── Step 3: Multi-horizon evaluation ──
    all_results = {}

    for h in HORIZONS:
        print(f"\n{'═' * 70}", flush=True)
        print(f"[3/5] HORIZON = {h}h", flush=True)
        print(f"{'═' * 70}", flush=True)

        results_h = _evaluate_horizon(df_feat, horizon=h)
        all_results[f"{h}h"] = results_h

    # ── Step 4: Summary ──
    print(f"\n{'═' * 70}", flush=True)
    print("[4/5] MULTI-HORIZON COMPARISON SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)

    print(f"\n{'Horizon':<10} {'Model':<25} {'MAE':>8} {'MASE':>8} {'RMSE':>8} {'Status':>12}", flush=True)
    print("─" * 75, flush=True)

    for h in HORIZONS:
        for model_name, metrics in all_results[f"{h}h"].items():
            mae = metrics.get("mae", float("nan"))
            mase = metrics.get("mase", float("nan"))
            rmse = metrics.get("rmse", float("nan"))
            if model_name == "Persistence":
                status = "baseline"
            elif isinstance(mase, float) and mase < 1.0:
                status = "✅ BEATS!"
            else:
                status = "❌ MASE>1"
            print(f"{h}h{'':<7} {model_name:<25} {mae:>8.3f} {mase:>8.3f} {rmse:>8.3f} {status:>12}", flush=True)
        print("─" * 75, flush=True)

    # ── Step 5: Save ──
    print("\n[5/5] Saving results...", flush=True)
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
        df,
        strategy="hybrid",
        max_gap_interp=6,
        max_gap_ml=24,
        knn_neighbors=5,
        verbose=True,
    )
    return df_hybrid


def _evaluate_horizon(df_feat: pd.DataFrame, horizon: int) -> dict:
    """Evaluate all models at a specific forecast horizon.

    Multi-step target: pm25[t + horizon] using features known at time t.
    """
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = {}

    # ── Create multi-step target ──
    print(f"\n  Creating {horizon}h-ahead target...", flush=True)
    df = df_feat.copy()

    # FIXED: ALWAYS shift target for ALL horizons including h=1
    # At row t: features are from time t, target = pm25[t + horizon]
    df[f"target_{horizon}h"] = df[TARGET_COL].shift(-horizon)
    target_col = f"target_{horizon}h"

    # Save pm25[t] for Persistence BEFORE any filtering
    # Persistence: ŷ[t+h] = y[t] (use current value to predict future)
    df["_persist_value"] = df[TARGET_COL]  # pm25[t]

    # Drop rows where shifted target is NaN (end of series)
    df = df.dropna(subset=[target_col])
    print(f"  Dataset after shift: {len(df):,} rows", flush=True)

    # ── Features and target ──
    exclude_cols = ["is_imputed", TARGET_COL, "_persist_value", f"target_{horizon}h"] + [
        f"target_{h}h" for h in HORIZONS
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ("float64", "float32", "int64")]

    X = df[feature_cols].fillna(0)
    y = df[target_col]
    is_imputed = df["is_imputed"]
    persist_values = df["_persist_value"]  # pm25[t] for Persistence

    # ── Temporal split (80/10/10) ──
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    test_imputed = is_imputed.iloc[val_end:]
    test_persist = persist_values.iloc[val_end:]  # pm25[t] for test set

    # Filter test to REAL data only
    test_real_mask = ~test_imputed.values
    X_test_real = X_test[test_real_mask]
    y_test_real = y_test[test_real_mask]
    persist_test_real = test_persist[test_real_mask]  # pm25[t] for real test

    print(f"  Train: {len(X_train):,} rows", flush=True)
    print(f"  Test (real only): {len(X_test_real):,}/{len(X_test):,} rows", flush=True)

    if len(X_test_real) < 10:
        print("  ⚠️ Too few real test samples, using all test data", flush=True)
        X_test_real = X_test
        y_test_real = y_test
        persist_test_real = test_persist

    # ── A. Persistence Baseline ──
    # FIXED: Persistence = |y[t+h] - y[t]| (predict future = current value)
    # y_test_real = y[t+h] (shifted target), persist_test_real = y[t] (current)
    print(f"\n  Evaluating Persistence baseline ({horizon}h)...", flush=True)
    y_persistence = persist_test_real.values  # pm25[t] = prediction

    persist_mae = float(np.mean(np.abs(y_test_real.values - y_persistence)))
    persist_rmse = float(np.sqrt(np.mean((y_test_real.values - y_persistence) ** 2)))

    # Naive reference for MASE = Persistence itself

    results["Persistence"] = {
        "mae": round(persist_mae, 4),
        "rmse": round(persist_rmse, 4),
        "mase": 1.0,  # By definition
        "horizon": horizon,
    }
    print(f"    Persistence {horizon}h: MAE={persist_mae:.3f}, RMSE={persist_rmse:.3f}", flush=True)
    print(f"    (y_true=pm25[t+{horizon}], y_persist=pm25[t], n={len(y_test_real)})", flush=True)

    # ── B. LightGBM Default ──
    print("\n  Training LightGBM (default params)...", flush=True)
    lgbm_default = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    lgbm_default.fit(X_train, y_train)
    y_pred_default = lgbm_default.predict(X_test_real)

    default_metrics = evaluate_forecast(
        y_true=y_test_real.values,
        y_pred=y_pred_default,
        y_naive=y_persistence,
        model_name=f"LightGBM_default_{horizon}h",
        horizon=horizon,
    )
    results["LightGBM_default"] = default_metrics
    print(f"    LightGBM default: MAE={default_metrics['mae']}, MASE={default_metrics['mase']}", flush=True)

    # ── C. LightGBM Optuna-tuned ──
    print(f"\n  Tuning LightGBM with Optuna ({OPTUNA_TRIALS} trials)...", flush=True)
    t0 = time.time()

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        model = lgb.LGBMRegressor(**params)
        cv_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_vl = X_train.iloc[val_idx]
            y_vl = y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_vl)
            cv_maes.append(float(np.mean(np.abs(y_vl.values - y_pred))))

        return float(np.mean(cv_maes))

    study = optuna.create_study(direction="minimize", study_name=f"lgbm_{horizon}h")

    # Progress callback
    def _progress_callback(study, trial):
        if (trial.number + 1) % 20 == 0 or trial.number == 0:
            print(
                f"    Trial {trial.number + 1}/{OPTUNA_TRIALS}: MAE={trial.value:.4f} (best={study.best_value:.4f})",
                flush=True,
            )

    study.optimize(objective, n_trials=OPTUNA_TRIALS, callbacks=[_progress_callback])
    tune_time = time.time() - t0

    best_params = study.best_params.copy()
    best_params["random_state"] = 42
    best_params["verbose"] = -1
    best_params["n_jobs"] = -1

    print(f"    Best CV MAE: {study.best_value:.4f} ({tune_time:.1f}s)", flush=True)
    print(f"    Best params: {json.dumps(study.best_params, indent=2)}", flush=True)

    # Train best model on full training set
    tuned_lgbm = lgb.LGBMRegressor(**best_params)
    tuned_lgbm.fit(X_train, y_train)
    y_pred_tuned = tuned_lgbm.predict(X_test_real)

    tuned_metrics = evaluate_forecast(
        y_true=y_test_real.values,
        y_pred=y_pred_tuned,
        y_naive=y_persistence,
        model_name=f"LightGBM_tuned_{horizon}h",
        horizon=horizon,
    )
    tuned_metrics["optuna_trials"] = OPTUNA_TRIALS
    tuned_metrics["optuna_best_params"] = study.best_params
    tuned_metrics["optuna_best_cv_mae"] = study.best_value
    tuned_metrics["tune_time_s"] = round(tune_time, 1)

    results["LightGBM_tuned"] = tuned_metrics
    print(f"\n    ✅ LightGBM tuned {horizon}h: MAE={tuned_metrics['mae']}, MASE={tuned_metrics['mase']}", flush=True)

    # ── D. Feature importance for this horizon ──
    importances = pd.Series(tuned_lgbm.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False).head(10)
    print(f"\n  Top-10 features for {horizon}h:", flush=True)
    for feat, imp in top_features.items():
        print(f"    {feat:<35s} importance={imp}", flush=True)

    return results


def _save_results(all_results: dict) -> None:
    """Save multi-horizon results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"multi_horizon_{timestamp}.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"  Results saved: {json_path}", flush=True)


if __name__ == "__main__":
    main()
