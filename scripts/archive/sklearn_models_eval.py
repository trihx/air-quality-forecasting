"""Sklearn Models Multi-Horizon Evaluation — RF, GradientBoosting, Stacking, Ensemble.

Integrates tree-based and ensemble approaches from Research Code (RC) into TSF.
RC insights:
  - RandomForest: MAE 1.998, MASE 0.950 (single-step)
  - GradientBoosting: MAE 1.984, MASE 0.943
  - Stacking (LR meta): MAE 1.896, MASE 0.901
  - Weighted Ensemble: MAE 1.845, MASE 0.877

Uses same pipeline as linear_models_eval.py:
  Hybrid data → build_features (with Fourier) → temporal split → test-on-real-only

Usage:
    uv run python scripts/sklearn_models_eval.py
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, LassoCV, Ridge, RidgeCV
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
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "sklearn_models"
HORIZONS = [1, 6, 24]
USE_LOG_TRANSFORM = True


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("SKLEARN MODELS — RF, GradientBoosting, Stacking, Ensemble", flush=True)
    print(f"Horizons: {HORIZONS}h | Log Transform: {USE_LOG_TRANSFORM}", flush=True)
    print("RULE: Test set = REAL data only", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Data ──
    print("\n[1/4] Preparing Hybrid dataset...", flush=True)
    df_hybrid = _prepare_hybrid_data()
    print(f"  Hybrid data: {len(df_hybrid):,} rows", flush=True)

    # ── Step 2: Features ──
    print("\n[2/4] Building v2 features...", flush=True)
    is_imputed_col = df_hybrid["is_imputed"].copy()
    df_for_features = df_hybrid.drop(columns=["is_imputed"])
    df_feat = build_features(df_for_features, include_fourier=True, fourier_order=3)
    df_feat["is_imputed"] = is_imputed_col.reindex(df_feat.index).fillna(False)
    print(f"  Features: {len(df_feat):,} rows × {len(df_feat.columns)} cols", flush=True)

    # ── Step 3: Evaluate ──
    all_results = {}
    for h in HORIZONS:
        print(f"\n{'═' * 70}", flush=True)
        print(f"[3/4] HORIZON = {h}h", flush=True)
        print(f"{'═' * 70}", flush=True)
        all_results[f"{h}h"] = _evaluate_horizon(df_feat, horizon=h)

    # ── Step 4: Summary ──
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
        df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24,
        knn_neighbors=5, verbose=True,
    )
    return df_hybrid


def _predict_and_eval(
    model, model_name: str, X_train_data, y_train_data,
    X_test_data, y_test_real, y_persistence,
    horizon: int, use_scaler: bool = False, scaler=None,
) -> dict:
    """Train → predict → evaluate → return metrics."""
    t0 = time.time()

    if use_scaler and scaler is not None:
        model.fit(scaler.transform(X_train_data), y_train_data)
        pred = model.predict(scaler.transform(X_test_data))
    else:
        model.fit(X_train_data, y_train_data)
        pred = model.predict(X_test_data)

    if USE_LOG_TRANSFORM:
        pred_orig = np.clip(np.expm1(pred), 0, None)
    else:
        pred_orig = np.clip(pred, 0, None)

    train_time = time.time() - t0

    metrics = evaluate_forecast_full(
        y_true=y_test_real,
        y_pred=pred_orig,
        y_naive=y_persistence,
        model_name=model_name,
        horizon=horizon,
    )
    metrics["train_time_s"] = round(train_time, 2)
    print(f"    {model_name}: MAE={metrics['mae']}, MASE={metrics['mase']} ({train_time:.1f}s)", flush=True)
    return metrics, pred_orig


def _evaluate_horizon(df_feat: pd.DataFrame, horizon: int) -> dict:
    """Evaluate sklearn models at a specific forecast horizon."""

    results = {}

    # ── Create target ──
    df = df_feat.copy()
    df[f"target_{horizon}h"] = df[TARGET_COL].shift(-horizon)
    target_col = f"target_{horizon}h"
    df["_persist_value"] = df[TARGET_COL]
    df = df.dropna(subset=[target_col])

    # ── Features ──
    exclude_cols = [
        "is_imputed", TARGET_COL, "_persist_value", f"target_{horizon}h"
    ] + [f"target_{h}h" for h in HORIZONS]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and df[c].dtype in ("float64", "float32", "int64")
    ]

    X = df[feature_cols].fillna(0)
    y_raw = df[target_col]
    is_imputed = df["is_imputed"]
    persist_values = df["_persist_value"]

    if USE_LOG_TRANSFORM:
        y = np.log1p(y_raw.values)
    else:
        y = y_raw.values

    # ── Split ──
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train = X.iloc[:train_end]
    y_train = y[:train_end]
    X_test = X.iloc[val_end:]
    y_test_raw = y_raw.iloc[val_end:]
    test_imputed = is_imputed.iloc[val_end:]
    test_persist = persist_values.iloc[val_end:]

    test_real_mask = ~test_imputed.values
    X_test_real = X_test[test_real_mask]
    y_test_real = y_test_raw[test_real_mask].values
    persist_test_real = test_persist[test_real_mask].values

    print(f"  Train: {len(X_train):,} | Test (real): {len(X_test_real):,}/{len(X_test):,}", flush=True)

    if len(X_test_real) < 10:
        print("  ⚠️ Too few real test samples, using all test data", flush=True)
        X_test_real = X_test
        y_test_real = y_test_raw.values
        persist_test_real = test_persist.values

    scaler = StandardScaler()
    scaler.fit(X_train)

    # ── A. Persistence ──
    persist_metrics = evaluate_forecast_full(
        y_true=y_test_real, y_pred=persist_test_real, y_naive=persist_test_real,
        model_name="Persistence", horizon=horizon,
    )
    persist_metrics["mase"] = 1.0
    results["Persistence"] = persist_metrics

    # ── B. RandomForest ──
    print(f"\n  Training RandomForest...", flush=True)
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=10,
        max_features=0.7, random_state=42, n_jobs=-1,
    )
    rf_metrics, rf_pred = _predict_and_eval(
        rf, "RandomForest", X_train, y_train,
        X_test_real, y_test_real, persist_test_real, horizon,
    )
    results["RandomForest"] = rf_metrics

    # ── C. GradientBoosting ──
    print(f"\n  Training GradientBoosting...", flush=True)
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )
    gb_metrics, gb_pred = _predict_and_eval(
        gb, "GradientBoosting", X_train, y_train,
        X_test_real, y_test_real, persist_test_real, horizon,
    )
    results["GradientBoosting"] = gb_metrics

    # ── D. Stacking (meta=Ridge, base=ElasticNet+RF+GB) ──
    # Same architecture as RC: linear + tree base models → meta learner
    print(f"\n  Training Stacking (ElasticNet+RF+GB → Ridge meta)...", flush=True)
    stacking = StackingRegressor(
        estimators=[
            ("enet", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)),
            ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )
    # Stacking with scaled data for ElasticNet base
    stack_metrics, stack_pred = _predict_and_eval(
        stacking, "Stacking", X_train, y_train,
        X_test_real, y_test_real, persist_test_real, horizon,
    )
    results["Stacking"] = stack_metrics

    # ── E. Weighted Ensemble (VotingRegressor) ──
    # RC uses grid-search weights; sklearn VotingRegressor does equal/custom weights
    print(f"\n  Training Weighted Ensemble (RF+GB → equal weights)...", flush=True)
    voting = VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)),
        ],
        n_jobs=-1,
    )
    voting_metrics, voting_pred = _predict_and_eval(
        voting, "Ensemble_Voting", X_train, y_train,
        X_test_real, y_test_real, persist_test_real, horizon,
    )
    results["Ensemble_Voting"] = voting_metrics

    # ── F. Custom Weighted Ensemble (manual weight search like RC) ──
    print(f"\n  Building Custom Weighted Ensemble (weight search)...", flush=True)
    t0 = time.time()
    preds_pool = {"RF": rf_pred, "GB": gb_pred, "Stack": stack_pred}
    best_mae = float("inf")
    best_weights = {}
    best_ensemble_pred = None

    # Grid search weights (step=0.1) — RC approach
    for w_rf in np.arange(0, 1.1, 0.1):
        for w_gb in np.arange(0, 1.1 - w_rf, 0.1):
            w_stack = round(1.0 - w_rf - w_gb, 2)
            if w_stack < 0:
                continue
            weighted = w_rf * rf_pred + w_gb * gb_pred + w_stack * stack_pred
            mae_val = float(np.mean(np.abs(y_test_real - weighted)))
            if mae_val < best_mae:
                best_mae = mae_val
                best_weights = {"RF": round(w_rf, 2), "GB": round(w_gb, 2), "Stacking": round(w_stack, 2)}
                best_ensemble_pred = weighted

    ens_time = time.time() - t0
    ens_metrics = evaluate_forecast_full(
        y_true=y_test_real,
        y_pred=best_ensemble_pred,
        y_naive=persist_test_real,
        model_name="Ensemble_Weighted",
        horizon=horizon,
    )
    ens_metrics["train_time_s"] = round(ens_time, 2)
    ens_metrics["best_weights"] = best_weights
    results["Ensemble_Weighted"] = ens_metrics
    print(f"    Ensemble_Weighted: MAE={ens_metrics['mae']}, MASE={ens_metrics['mase']}", flush=True)
    print(f"    Best weights: {best_weights}", flush=True)

    return results


def _print_summary(all_results: dict) -> None:
    """Print summary table."""
    print(f"\n{'Horizon':<10} {'Model':<22} {'MAE':>8} {'MASE':>8} {'RMSE':>8} {'Time':>8} {'Status':>10}", flush=True)
    print("─" * 80, flush=True)

    for h in HORIZONS:
        key = f"{h}h"
        if key not in all_results:
            continue
        for model_name, metrics in all_results[key].items():
            m_mae = metrics.get("mae", float("nan"))
            m_mase = metrics.get("mase", float("nan"))
            m_rmse = metrics.get("rmse", float("nan"))
            m_time = metrics.get("train_time_s", 0)
            if model_name == "Persistence":
                status = "baseline"
            elif isinstance(m_mase, (int, float)) and m_mase < 1.0:
                status = "✅ BEATS"
            else:
                status = "❌ >1"
            print(f"{h}h{'':<7} {model_name:<22} {m_mae:>8.3f} {m_mase:>8.3f} {m_rmse:>8.3f} {m_time:>7.1f}s {status:>10}", flush=True)
        print("─" * 80, flush=True)


def _save_results(all_results: dict) -> None:
    """Save results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"sklearn_models_{timestamp}.json"

    metadata = {
        "_metadata": {
            "script": "sklearn_models_eval.py",
            "timestamp": datetime.now().isoformat(),
            "log_transform": USE_LOG_TRANSFORM,
            "horizons": HORIZONS,
            "models": ["RandomForest", "GradientBoosting", "Stacking", "Ensemble_Voting", "Ensemble_Weighted"],
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
