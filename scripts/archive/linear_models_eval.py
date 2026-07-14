"""Linear Models Multi-Horizon Evaluation — Lasso, ElasticNet, Ridge.

Integrates linear model approaches from Research Code (RC) into TSF pipeline.
Key RC insight: ElasticNet + Fourier features + log1p target = MAE 1.85.

Uses same pipeline as multi_horizon_eval.py:
  Hybrid data → build_features (with Fourier) → temporal split → test-on-real-only

Usage:
    uv run python scripts/linear_models_eval.py
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

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
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "linear_models"
HORIZONS = [1, 6, 24]
USE_LOG_TRANSFORM = True  # RC's key advantage for linear models


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("LINEAR MODELS MULTI-HORIZON EVALUATION", flush=True)
    print(f"Horizons: {HORIZONS}h | Log Transform: {USE_LOG_TRANSFORM}", flush=True)
    print(f"Models: Lasso(CV), ElasticNet, Ridge(CV)", flush=True)
    print("RULE: Test set = REAL data only", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Prepare data ──
    print("\n[1/4] Preparing Hybrid dataset...", flush=True)
    df_hybrid = _prepare_hybrid_data()
    print(f"  Hybrid data: {len(df_hybrid):,} rows", flush=True)

    # ── Step 2: Build features (with Fourier) ──
    print("\n[2/4] Building features (with Fourier + interactions)...", flush=True)
    is_imputed_col = df_hybrid["is_imputed"].copy()
    df_for_features = df_hybrid.drop(columns=["is_imputed"])

    df_feat = build_features(df_for_features, include_fourier=True, fourier_order=3)
    df_feat["is_imputed"] = is_imputed_col.reindex(df_feat.index).fillna(False)
    print(f"  Features: {len(df_feat):,} rows × {len(df_feat.columns)} cols", flush=True)

    # Count feature groups
    fourier_cols = [c for c in df_feat.columns if c.startswith("fourier_")]
    interaction_cols = [c for c in df_feat.columns if c.startswith("pm25_x_") or c in ["temp_dew_diff", "pm25_relative_24h"]]
    range_cols = [c for c in df_feat.columns if "_roll_" in c and "_range" in c]
    print(f"  Fourier features: {len(fourier_cols)}", flush=True)
    print(f"  Interaction features: {len(interaction_cols)}", flush=True)
    print(f"  Rolling range features: {len(range_cols)}", flush=True)

    # ── Step 3: Multi-horizon evaluation ──
    all_results = {}
    for h in HORIZONS:
        print(f"\n{'═' * 70}", flush=True)
        print(f"[3/4] HORIZON = {h}h", flush=True)
        print(f"{'═' * 70}", flush=True)
        results_h = _evaluate_horizon(df_feat, horizon=h)
        all_results[f"{h}h"] = results_h

    # ── Step 4: Summary + Save ──
    print(f"\n{'═' * 70}", flush=True)
    print("[4/4] SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)
    _print_summary(all_results)
    _save_results(all_results)

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
        df,
        strategy="hybrid",
        max_gap_interp=6,
        max_gap_ml=24,
        knn_neighbors=5,
        verbose=True,
    )
    return df_hybrid


def _evaluate_horizon(df_feat: pd.DataFrame, horizon: int) -> dict:
    """Evaluate linear models at a specific forecast horizon."""

    results = {}

    # ── Create multi-step target ──
    print(f"\n  Creating {horizon}h-ahead target...", flush=True)
    df = df_feat.copy()
    df[f"target_{horizon}h"] = df[TARGET_COL].shift(-horizon)
    target_col = f"target_{horizon}h"
    df["_persist_value"] = df[TARGET_COL]

    df = df.dropna(subset=[target_col])
    print(f"  Dataset after shift: {len(df):,} rows", flush=True)

    # ── Features and target ──
    exclude_cols = [
        "is_imputed", TARGET_COL, "_persist_value", f"target_{horizon}h"
    ] + [f"target_{h}h" for h in HORIZONS]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and df[c].dtype in ("float64", "float32", "int64")
    ]

    X = df[feature_cols].fillna(0)
    y_raw = df[target_col]  # raw µg/m³
    is_imputed = df["is_imputed"]
    persist_values = df["_persist_value"]

    # ── Log transform target if enabled ──
    if USE_LOG_TRANSFORM:
        y = np.log1p(y_raw.values)
        print(f"  Target: log1p transformed", flush=True)
    else:
        y = y_raw.values

    # ── Temporal split (80/10/10) ──
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train = X.iloc[:train_end]
    y_train = y[:train_end]
    X_test = X.iloc[val_end:]
    y_test_raw = y_raw.iloc[val_end:]
    y_test = y[val_end:]
    test_imputed = is_imputed.iloc[val_end:]
    test_persist = persist_values.iloc[val_end:]

    # Filter test to REAL data only
    test_real_mask = ~test_imputed.values
    X_test_real = X_test[test_real_mask]
    y_test_real = y_test_raw[test_real_mask].values  # always evaluate in original scale
    y_test_log_real = y_test[test_real_mask]
    persist_test_real = test_persist[test_real_mask].values

    print(f"  Train: {len(X_train):,} rows", flush=True)
    print(f"  Test (real only): {len(X_test_real):,}/{len(X_test):,} rows", flush=True)

    if len(X_test_real) < 10:
        print("  ⚠️ Too few real test samples, using all test data", flush=True)
        X_test_real = X_test
        y_test_real = y_test_raw.values
        y_test_log_real = y_test
        persist_test_real = test_persist.values

    # ── Scale features (required for linear models) ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_real)

    # ── A. Persistence Baseline ──
    print(f"\n  Evaluating Persistence baseline ({horizon}h)...", flush=True)
    y_persistence = persist_test_real

    persist_metrics = evaluate_forecast_full(
        y_true=y_test_real,
        y_pred=y_persistence,
        y_naive=y_persistence,
        model_name="Persistence",
        horizon=horizon,
    )
    persist_metrics["mase"] = 1.0
    results["Persistence"] = persist_metrics

    # ── B. LassoCV ──
    print("\n  Training LassoCV...", flush=True)
    t0 = time.time()
    lasso = LassoCV(
        cv=5, random_state=42, max_iter=10000, n_jobs=-1,
    )
    lasso.fit(X_train_scaled, y_train[:len(X_train_scaled)])
    lasso_pred = lasso.predict(X_test_scaled)

    if USE_LOG_TRANSFORM:
        lasso_pred_orig = np.clip(np.expm1(lasso_pred), 0, None)
    else:
        lasso_pred_orig = np.clip(lasso_pred, 0, None)

    lasso_time = time.time() - t0
    lasso_metrics = evaluate_forecast_full(
        y_true=y_test_real,
        y_pred=lasso_pred_orig,
        y_naive=y_persistence,
        model_name="LassoCV",
        horizon=horizon,
    )
    lasso_metrics["train_time_s"] = round(lasso_time, 2)
    lasso_metrics["best_alpha"] = float(lasso.alpha_)
    results["LassoCV"] = lasso_metrics
    print(f"    LassoCV: MAE={lasso_metrics['mae']}, MASE={lasso_metrics['mase']} ({lasso_time:.1f}s)", flush=True)

    # ── C. ElasticNet ──
    print("\n  Training ElasticNet...", flush=True)
    t0 = time.time()
    # Grid search for best alpha/l1_ratio
    best_enet_mae = float("inf")
    best_enet = None
    for alpha in [0.0001, 0.001, 0.01, 0.1]:
        for l1_ratio in [0.1, 0.5, 0.9, 0.95]:
            enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
            enet.fit(X_train_scaled, y_train[:len(X_train_scaled)])
            enet_pred = enet.predict(X_test_scaled)
            if USE_LOG_TRANSFORM:
                enet_pred_orig = np.clip(np.expm1(enet_pred), 0, None)
            else:
                enet_pred_orig = np.clip(enet_pred, 0, None)
            enet_mae = float(np.mean(np.abs(y_test_real - enet_pred_orig)))
            if enet_mae < best_enet_mae:
                best_enet_mae = enet_mae
                best_enet = enet
                best_enet_pred = enet_pred_orig

    enet_time = time.time() - t0
    enet_metrics = evaluate_forecast_full(
        y_true=y_test_real,
        y_pred=best_enet_pred,
        y_naive=y_persistence,
        model_name="ElasticNet",
        horizon=horizon,
    )
    enet_metrics["train_time_s"] = round(enet_time, 2)
    enet_metrics["best_alpha"] = float(best_enet.alpha)
    enet_metrics["best_l1_ratio"] = float(best_enet.l1_ratio)
    results["ElasticNet"] = enet_metrics
    print(f"    ElasticNet: MAE={enet_metrics['mae']}, MASE={enet_metrics['mase']} ({enet_time:.1f}s)", flush=True)
    print(f"    Best: alpha={best_enet.alpha}, l1_ratio={best_enet.l1_ratio}", flush=True)

    # ── D. RidgeCV ──
    print("\n  Training RidgeCV...", flush=True)
    t0 = time.time()
    ridge = RidgeCV(
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
        cv=5,
    )
    ridge.fit(X_train_scaled, y_train[:len(X_train_scaled)])
    ridge_pred = ridge.predict(X_test_scaled)

    if USE_LOG_TRANSFORM:
        ridge_pred_orig = np.clip(np.expm1(ridge_pred), 0, None)
    else:
        ridge_pred_orig = np.clip(ridge_pred, 0, None)

    ridge_time = time.time() - t0
    ridge_metrics = evaluate_forecast_full(
        y_true=y_test_real,
        y_pred=ridge_pred_orig,
        y_naive=y_persistence,
        model_name="RidgeCV",
        horizon=horizon,
    )
    ridge_metrics["train_time_s"] = round(ridge_time, 2)
    ridge_metrics["best_alpha"] = float(ridge.alpha_)
    results["RidgeCV"] = ridge_metrics
    print(f"    RidgeCV: MAE={ridge_metrics['mae']}, MASE={ridge_metrics['mase']} ({ridge_time:.1f}s)", flush=True)

    return results


def _print_summary(all_results: dict) -> None:
    """Print summary table."""
    print(f"\n{'Horizon':<10} {'Model':<20} {'MAE':>8} {'MASE':>8} {'RMSE':>8} {'Status':>12}", flush=True)
    print("─" * 70, flush=True)

    for h in HORIZONS:
        key = f"{h}h"
        if key not in all_results:
            continue
        for model_name, metrics in all_results[key].items():
            m_mae = metrics.get("mae", float("nan"))
            m_mase = metrics.get("mase", float("nan"))
            m_rmse = metrics.get("rmse", float("nan"))
            if model_name == "Persistence":
                status = "baseline"
            elif isinstance(m_mase, (int, float)) and m_mase < 1.0:
                status = "✅ BEATS!"
            else:
                status = "❌ MASE>1"
            print(f"{h}h{'':<7} {model_name:<20} {m_mae:>8.3f} {m_mase:>8.3f} {m_rmse:>8.3f} {status:>12}", flush=True)
        print("─" * 70, flush=True)


def _save_results(all_results: dict) -> None:
    """Save results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"linear_models_{timestamp}.json"

    metadata = {
        "_metadata": {
            "script": "linear_models_eval.py",
            "timestamp": datetime.now().isoformat(),
            "log_transform": USE_LOG_TRANSFORM,
            "horizons": HORIZONS,
            "models": ["LassoCV", "ElasticNet", "RidgeCV"],
            "features": "Fourier + interaction + rolling_range (v2 enhanced)",
            "test_policy": "real-only",
        }
    }
    metadata.update(all_results)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"\n  Results saved: {json_path}", flush=True)


if __name__ == "__main__":
    main()
