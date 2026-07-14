"""v9 Phase 5B.1 — Retrain LightGBM.

Retrains LightGBM models on high-resolution segment-aware data.
Runs on BOTH 15min and 30min datasets, converting physical horizons
(1h, 6h, 24h) to the correct number of steps.

Usage:
    uv run python scripts/v9_retrain_lgbm.py 2>&1 | tee research/logs/v9_retrain_lgbm.log
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import TARGET_COL
from src.evaluation.metrics import evaluate_forecast_full

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
USE_LOG_TRANSFORM = True

# Target physical horizons in hours
HORIZONS_HOURS = [1, 6, 24]

# Tuned Hyperparameters from previous Optuna runs (we reuse them as a strong baseline)
LGBM_PARAMS = {
    1: dict(
        n_estimators=500, max_depth=3, learning_rate=0.013, num_leaves=64,
        subsample=0.8, colsample_bytree=0.6, min_child_samples=30,
        reg_alpha=0.05, reg_lambda=0.5, random_state=42, verbose=-1
    ),
    6: dict(
        n_estimators=637, max_depth=3, learning_rate=0.012, num_leaves=87,
        subsample=0.85, colsample_bytree=0.55, min_child_samples=25,
        reg_alpha=0.03, reg_lambda=0.7, random_state=42, verbose=-1
    ),
    24: dict(
        n_estimators=450, max_depth=4, learning_rate=0.015, num_leaves=52,
        subsample=0.75, colsample_bytree=0.65, min_child_samples=35,
        reg_alpha=0.08, reg_lambda=0.4, random_state=42, verbose=-1
    ),
}


def _save(data: dict, name: str, freq: str) -> None:
    """Save JSON output."""
    out_dir = PROJECT_ROOT / "research" / "experiments" / "v9_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{freq}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def split_data_segment_aware(df_feat: pd.DataFrame, steps: int) -> tuple:
    """Create train/test split with strict anti-leakage (segment-aware)."""
    df = df_feat.copy()

    # CRITICAL FIX: The target must be shifted WITHIN segments to prevent False Continuity.
    # We shift negative steps so target[t] = pm25[t + steps]
    target_col_name = f"target_{steps}s"
    df[target_col_name] = df.groupby("segment_id")[TARGET_COL].shift(-steps)
    df["_persist"] = df[TARGET_COL]  # Persistence baseline uses value at time t

    # Drop rows where target couldn't be computed (end of segments)
    df = df.dropna(subset=[target_col_name])

    # Feature selection
    exclude = ["is_imputed", "segment_id", TARGET_COL, "_persist", target_col_name]
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "float32", "int64")]

    X = df[feat_cols].fillna(0)
    y_raw = df[target_col_name]
    is_imp = df["is_imputed"]
    persist = df["_persist"]

    # Chronological split
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train_raw = X.iloc[:train_end], y_raw.iloc[:train_end]

    # For testing, we only use real data
    X_test = X.iloc[val_end:]
    y_test_raw = y_raw.iloc[val_end:]
    test_imp = is_imp.iloc[val_end:]
    test_persist = persist.iloc[val_end:]

    real_mask = ~test_imp.values
    X_test_real = X_test[real_mask]
    y_test_real = y_test_raw[real_mask].values
    persist_real = test_persist[real_mask].values

    if USE_LOG_TRANSFORM:
        y_train = np.log1p(y_train_raw.values)
    else:
        y_train = y_train_raw.values

    return X_train, y_train, X_test_real, y_test_real, persist_real, train_end, val_end, n


def eval_model(model, name: str, X_train, y_train, X_test, y_true, y_naive, h: int) -> tuple:
    """Train, predict, evaluate."""
    t0 = time.time()

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if USE_LOG_TRANSFORM:
        pred_orig = np.clip(np.expm1(pred), 0, None)
    else:
        pred_orig = np.clip(pred, 0, None)

    elapsed = time.time() - t0
    metrics = evaluate_forecast_full(y_true, pred_orig, y_naive, name, h)
    metrics["train_time_s"] = round(elapsed, 2)
    print(f"    {name}: MAE={metrics['mae']}, MASE={metrics['mase']} ({elapsed:.1f}s)", flush=True)

    return metrics, pred_orig


def run_pipeline(freq: str, df_feat: pd.DataFrame):
    """Run the entire LightGBM pipeline for a given frequency."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"[ML] LightGBM - {freq} Dataset", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Determine steps per hour
    steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(freq.replace("m", "min")))

    results = {}
    preds = {}

    for h_hours in HORIZONS_HOURS:
        steps = h_hours * steps_per_hour
        print(f"\n  ── Horizon {h_hours}h ({steps} steps) ──", flush=True)

        # Segment-aware train/test split
        X_train, y_train, X_test, y_true, y_naive, _, _, _ = split_data_segment_aware(df_feat, steps)

        print(f"    Train size: {len(X_train):,} | Test size (real only): {len(X_test):,}")

        # Train model using params tuned for that physical horizon
        model = lgb.LGBMRegressor(**LGBM_PARAMS[h_hours])
        m, p = eval_model(model, f"LightGBM_v9", X_train, y_train, X_test, y_true, y_naive, h_hours)

        results[f"{h_hours}h"] = {"LightGBM_v9": m}
        preds[f"{h_hours}h"] = {
            "LightGBM_v9": p.tolist(),
            "Persistence": y_naive.tolist(),
            "Actuals": y_true.tolist()
        }

    # Save outputs
    _save(results, "lgbm_metrics", freq)
    _save(preds, "lgbm_preds", freq)
    print(f"\n[v9] ✅ LightGBM ({freq}) saved.", flush=True)


def main():
    print("=" * 70, flush=True)
    print(f"v9 LIGHTGBM RETRAIN — Segment-Aware", flush=True)
    print(f"Timestamp: {TIMESTAMP}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()

    data_dir = PROJECT_ROOT / "dataset" / "processed"

    # Process 30m
    path_30m = data_dir / "marts_features_30m.csv"
    if path_30m.exists():
        df_30m = pd.read_csv(path_30m, index_col=0, parse_dates=True)
        run_pipeline("30m", df_30m)
    else:
        print(f"⚠️ {path_30m} not found.")

    # Process 15m
    path_15m = data_dir / "marts_features_15m.csv"
    if path_15m.exists():
        df_15m = pd.read_csv(path_15m, index_col=0, parse_dates=True)
        run_pipeline("15m", df_15m)
    else:
        print(f"⚠️ {path_15m} not found.")

    print(f"\n✅ Total LightGBM time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
