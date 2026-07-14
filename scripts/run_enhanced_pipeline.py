"""Enhanced Pipeline Runner — Re-evaluate LightGBM with new features.

Runs LightGBM (Optuna) on the v2 enhanced feature set:
  - Fourier features (12 sin/cos)
  - Rolling range (6 features)
  - Interaction features (6 features)
  - Log1p target transform

Then compiles all v2 results (linear + LightGBM) into a dashboard snapshot.

Usage:
    uv run python scripts/run_enhanced_pipeline.py
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
from src.evaluation.metrics import evaluate_forecast_full
from src.features.builder import build_features

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "multi_horizon"
DASHBOARD_RUNS_DIR = PROJECT_ROOT / "research" / "experiments" / "dashboard_runs"
HORIZONS = [1, 6, 24]
OPTUNA_TRIALS = 100
USE_LOG_TRANSFORM = True


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("ENHANCED PIPELINE — LightGBM + v2 Features", flush=True)
    print(f"Features: Fourier + rolling_range + interactions", flush=True)
    print(f"Log Transform: {USE_LOG_TRANSFORM} | Optuna trials: {OPTUNA_TRIALS}", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Prepare data ──
    print("\n[1/5] Preparing Hybrid dataset...", flush=True)
    df_hybrid = _prepare_hybrid_data()
    print(f"  Hybrid data: {len(df_hybrid):,} rows", flush=True)

    # ── Step 2: Build v2 features ──
    print("\n[2/5] Building v2 features (Fourier + interactions + range)...", flush=True)
    is_imputed_col = df_hybrid["is_imputed"].copy()
    df_for_features = df_hybrid.drop(columns=["is_imputed"])
    df_feat = build_features(df_for_features, include_fourier=True, fourier_order=3)
    df_feat["is_imputed"] = is_imputed_col.reindex(df_feat.index).fillna(False)

    # Print feature summary
    feature_groups = {
        "fourier": [c for c in df_feat.columns if c.startswith("fourier_")],
        "interaction": [c for c in df_feat.columns if c.startswith("pm25_x_") or c in ["temp_dew_diff", "pm25_relative_24h"]],
        "range": [c for c in df_feat.columns if "_roll_" in c and "_range" in c],
    }
    print(f"  Total: {len(df_feat):,} rows × {len(df_feat.columns)} cols", flush=True)
    for name, cols in feature_groups.items():
        print(f"  {name}: {len(cols)} features", flush=True)

    # ── Step 3: Multi-horizon LightGBM ──
    lgbm_results = {}
    for h in HORIZONS:
        print(f"\n{'═' * 70}", flush=True)
        print(f"[3/5] HORIZON = {h}h — LightGBM Optuna", flush=True)
        print(f"{'═' * 70}", flush=True)
        lgbm_results[f"{h}h"] = _evaluate_lgbm(df_feat, horizon=h)

    # ── Step 4: Load linear results & create v2 snapshot ──
    print(f"\n{'═' * 70}", flush=True)
    print("[4/5] Creating v2 dashboard snapshot...", flush=True)
    print(f"{'═' * 70}", flush=True)
    _create_v2_snapshot(lgbm_results)

    # ── Step 5: Summary ──
    print(f"\n{'═' * 70}", flush=True)
    print("[5/5] FINAL SUMMARY — v2 Enhanced Results", flush=True)
    print(f"{'═' * 70}", flush=True)
    _print_comparison(lgbm_results)

    total_time = time.time() - t_start
    print(f"\nCOMPLETE — Total: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)


def _prepare_hybrid_data() -> pd.DataFrame:
    """Load raw data and apply Hybrid imputation strategy."""
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
    return df_hybrid


def _evaluate_lgbm(df_feat: pd.DataFrame, horizon: int) -> dict:
    """Evaluate LightGBM with v2 features at a specific horizon."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Create multi-step target ──
    df = df_feat.copy()
    df[f"target_{horizon}h"] = df[TARGET_COL].shift(-horizon)
    target_col = f"target_{horizon}h"
    df["_persist_value"] = df[TARGET_COL]
    df = df.dropna(subset=[target_col])

    # ── Features and target ──
    exclude_cols = [
        "is_imputed", TARGET_COL, "_persist_value", f"target_{horizon}h"
    ] + [f"target_{h}h" for h in HORIZONS]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and df[c].dtype in ("float64", "float32", "int64")
    ]

    X = df[feature_cols].fillna(0)
    y_raw = df[target_col]
    persist_values = df["_persist_value"]
    is_imputed = df["is_imputed"]

    # Log transform
    if USE_LOG_TRANSFORM:
        y = np.log1p(y_raw.values)
    else:
        y = y_raw.values

    # ── Temporal split (80/10/10) ──
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_test = X.iloc[val_end:]
    y_test_raw = y_raw.iloc[val_end:]
    test_imputed = is_imputed.iloc[val_end:]
    test_persist = persist_values.iloc[val_end:]

    # Filter REAL only
    test_real_mask = ~test_imputed.values
    X_test_real = X_test[test_real_mask]
    y_test_real = y_test_raw[test_real_mask].values
    persist_test_real = test_persist[test_real_mask].values

    print(f"  Train: {len(X_train):,} | Test (real): {len(X_test_real):,}/{len(X_test):,}", flush=True)

    # ── Persistence baseline ──
    persist_metrics = evaluate_forecast_full(
        y_true=y_test_real, y_pred=persist_test_real, y_naive=persist_test_real,
        model_name="Persistence", horizon=horizon,
    )
    persist_metrics["mase"] = 1.0

    # ── LightGBM Default ──
    print(f"\n  Training LightGBM Default...", flush=True)
    lgbm_default = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    lgbm_default.fit(X_train, y_train)
    pred_default = lgbm_default.predict(X_test_real)
    if USE_LOG_TRANSFORM:
        pred_default = np.clip(np.expm1(pred_default), 0, None)
    else:
        pred_default = np.clip(pred_default, 0, None)

    default_metrics = evaluate_forecast_full(
        y_true=y_test_real, y_pred=pred_default, y_naive=persist_test_real,
        model_name=f"LightGBM_default_{horizon}h", horizon=horizon,
    )

    # ── LightGBM Optuna ──
    print(f"\n  Tuning LightGBM Optuna ({OPTUNA_TRIALS} trials)...", flush=True)
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
            "random_state": 42, "verbose": -1, "n_jobs": -1,
        }
        model = lgb.LGBMRegressor(**params)
        cv_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_vl, y_vl = X_train.iloc[val_idx], y_train[val_idx]
            model.fit(X_tr, y_tr)
            y_pred_cv = model.predict(X_vl)
            if USE_LOG_TRANSFORM:
                y_pred_orig = np.expm1(y_pred_cv)
                y_vl_orig = np.expm1(y_vl)
                cv_maes.append(float(np.mean(np.abs(y_vl_orig - y_pred_orig))))
            else:
                cv_maes.append(float(np.mean(np.abs(y_vl - y_pred_cv))))
        return float(np.mean(cv_maes))

    study = optuna.create_study(direction="minimize", study_name=f"lgbm_v2_{horizon}h")

    def _progress(study, trial):
        if (trial.number + 1) % 20 == 0:
            print(f"    Trial {trial.number + 1}/{OPTUNA_TRIALS}: MAE={trial.value:.4f} (best={study.best_value:.4f})", flush=True)

    study.optimize(objective, n_trials=OPTUNA_TRIALS, callbacks=[_progress])
    tune_time = time.time() - t0

    best_params = study.best_params.copy()
    best_params.update({"random_state": 42, "verbose": -1, "n_jobs": -1})
    print(f"    Best CV MAE: {study.best_value:.4f} ({tune_time:.1f}s)", flush=True)

    # Train best model
    tuned_lgbm = lgb.LGBMRegressor(**best_params)
    tuned_lgbm.fit(X_train, y_train)
    pred_tuned = tuned_lgbm.predict(X_test_real)
    if USE_LOG_TRANSFORM:
        pred_tuned = np.clip(np.expm1(pred_tuned), 0, None)
    else:
        pred_tuned = np.clip(pred_tuned, 0, None)

    tuned_metrics = evaluate_forecast_full(
        y_true=y_test_real, y_pred=pred_tuned, y_naive=persist_test_real,
        model_name=f"LightGBM_tuned_{horizon}h", horizon=horizon,
    )
    tuned_metrics["optuna_trials"] = OPTUNA_TRIALS
    tuned_metrics["optuna_best_cv_mae"] = study.best_value
    tuned_metrics["optuna_best_params"] = study.best_params
    tuned_metrics["tune_time_s"] = round(tune_time, 1)

    print(f"    ✅ LightGBM tuned: MAE={tuned_metrics['mae']}, MASE={tuned_metrics['mase']}", flush=True)

    # Feature importance
    importances = pd.Series(tuned_lgbm.feature_importances_, index=feature_cols)
    top_10 = importances.sort_values(ascending=False).head(10)
    print(f"\n  Top-10 features ({horizon}h):", flush=True)
    for feat, imp in top_10.items():
        print(f"    {feat:<40s} importance={imp}", flush=True)

    return {
        "Persistence": persist_metrics,
        "LightGBM_default": default_metrics,
        "LightGBM_tuned": tuned_metrics,
    }


def _create_v2_snapshot(lgbm_results: dict) -> None:
    """Create v2 dashboard snapshot combining linear + LightGBM results."""
    DASHBOARD_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Load linear model results
    linear_dir = PROJECT_ROOT / "research" / "experiments" / "linear_models"
    linear_jsons = sorted(linear_dir.glob("*.json")) if linear_dir.exists() else []
    linear_data = {}
    if linear_jsons:
        with open(linear_jsons[-1], encoding="utf-8") as f:
            linear_data = json.load(f)
        print(f"  Loaded linear results: {linear_jsons[-1].name}", flush=True)

    # Load v1 baseline for DL models (GRU, LSTM, TFT) — unchanged
    v1_path = DASHBOARD_RUNS_DIR / "v1_baseline_20260411.json"
    v1_data = {}
    if v1_path.exists():
        with open(v1_path, encoding="utf-8") as f:
            v1_data = json.load(f)
        print(f"  Loaded v1 baseline for DL models", flush=True)

    # Build combined results
    v2_results = {"results": {}}
    for h in HORIZONS:
        key = f"{h}h"
        combined = {}

        # Persistence from LightGBM run
        if key in lgbm_results and "Persistence" in lgbm_results[key]:
            combined["Persistence"] = lgbm_results[key]["Persistence"]

        # LightGBM from v2 run
        if key in lgbm_results:
            for model_name in ["LightGBM_default", "LightGBM_tuned"]:
                if model_name in lgbm_results[key]:
                    combined[model_name] = lgbm_results[key][model_name]

        # Linear models from separate run
        if key in linear_data:
            for model_name in ["LassoCV", "ElasticNet", "RidgeCV"]:
                if model_name in linear_data[key]:
                    combined[model_name] = linear_data[key][model_name]

        # DL models from v1 (not retrained)
        v1_results = v1_data.get("data", {}).get("results", {})
        if key in v1_results:
            for model_name in ["ARIMA", "SARIMA", "LSTM", "GRU", "TFT", "Ensemble_GRU", "Ensemble_Stack"]:
                if model_name in v1_results[key]:
                    model_data = v1_results[key][model_name].copy()
                    model_data["source"] = "v1_baseline (not retrained)"
                    combined[model_name] = model_data

        v2_results["results"][key] = combined

    # Metadata
    snapshot = {
        "version": "v2_enhanced",
        "timestamp": datetime.now().isoformat(),
        "description": (
            "Enhanced features: Fourier (12), rolling range (6), interactions (6). "
            "Log1p target transform. Linear models added (LassoCV, ElasticNet, RidgeCV). "
            "DL models (GRU, LSTM, TFT) from v1 — NOT retrained."
        ),
        "feature_set": {
            "lag": True, "rolling": True, "ewm": True, "diff": True,
            "calendar": True, "domain": True, "fourier": True,
            "interaction": True, "log_transform": True,
            "rolling_range": True,
        },
        "models_included": list(set(
            m for h_data in v2_results["results"].values() for m in h_data.keys()
        )),
        "data": v2_results,
    }

    output_path = DASHBOARD_RUNS_DIR / "v2_enhanced_20260411.json"
    
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"  Saved v2 snapshot: {output_path}", flush=True)
    print(f"  Models: {len(snapshot['models_included'])}", flush=True)


def _print_comparison(lgbm_results: dict) -> None:
    """Print LightGBM v1 vs v2 comparison from saved data."""
    # Load v1 metrics
    v1_path = DASHBOARD_RUNS_DIR / "v1_baseline_20260411.json"
    if not v1_path.exists():
        print("  No v1 baseline to compare", flush=True)
        return

    with open(v1_path, encoding="utf-8") as f:
        v1 = json.load(f)

    v1_results = v1.get("data", {}).get("results", {})

    print(f"\n{'Horizon':<8} {'Metric':<8} {'v1 (before)':>12} {'v2 (after)':>12} {'Change':>10}", flush=True)
    print("─" * 55, flush=True)

    for h in HORIZONS:
        key = f"{h}h"
        v1_lgbm = v1_results.get(key, {}).get("LightGBM_tuned", {})
        v2_lgbm = lgbm_results.get(key, {}).get("LightGBM_tuned", {})

        v1_mae = v1_lgbm.get("mae", float("nan"))
        v2_mae = v2_lgbm.get("mae", float("nan"))
        v1_mase = v1_lgbm.get("mase", v1_lgbm.get("mase_original", float("nan")))
        v2_mase = v2_lgbm.get("mase", float("nan"))

        if isinstance(v1_mae, (int, float)) and isinstance(v2_mae, (int, float)):
            mae_change = ((v2_mae - v1_mae) / v1_mae) * 100
            mase_change = ((v2_mase - v1_mase) / v1_mase) * 100 if isinstance(v1_mase, (int, float)) else float("nan")
            print(f"{key:<8} {'MAE':<8} {v1_mae:>12.3f} {v2_mae:>12.3f} {mae_change:>+9.1f}%", flush=True)
            print(f"{'':<8} {'MASE':<8} {v1_mase:>12.3f} {v2_mase:>12.3f} {mase_change:>+9.1f}%", flush=True)
        print("─" * 55, flush=True)


if __name__ == "__main__":
    main()
