"""v9 Phase 5B.2 — Retrain Sklearn Suite.

Retrains RF, GB, ElasticNet, and Stacking/Voting Regressors
on high-resolution segment-aware data (15min and 30min).
"""

from __future__ import annotations

import json
import sys
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
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v9_retrain_lgbm import split_data_segment_aware, _save
from src.evaluation.metrics import evaluate_forecast_full

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
USE_LOG_TRANSFORM = True

# Target physical horizons in hours
HORIZONS_HOURS = [1, 6, 24]


def eval_model(model, name: str, X_train, y_train, X_test, y_true, y_naive, h: int, scaler=None) -> tuple:
    """Train, predict, evaluate."""
    t0 = time.time()

    if scaler:
        model.fit(scaler.transform(X_train), y_train)
        pred = model.predict(scaler.transform(X_test))
    else:
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
    """Run the Sklearn pipeline for a given frequency."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"[ML] Sklearn Suite - {freq} Dataset", flush=True)
    print(f"{'=' * 70}", flush=True)

    steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(freq.replace("m", "min")))

    results = {}
    preds = {}

    for h_hours in HORIZONS_HOURS:
        steps = h_hours * steps_per_hour
        print(f"\n  ── Horizon {h_hours}h ({steps} steps) ──", flush=True)

        # Segment-aware split
        X_train, y_train, X_test, y_true, y_naive, _, _, _ = split_data_segment_aware(df_feat, steps)

        h_results = {}
        h_preds = {"Actuals": y_true.tolist(), "Persistence": y_naive.tolist()}

        # 1. Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=10,
                                   n_jobs=-1, random_state=42)
        m, p = eval_model(rf, "RandomForest_v9", X_train, y_train, X_test, y_true, y_naive, h_hours)
        h_results["RandomForest_v9"] = m
        h_preds["RandomForest_v9"] = p.tolist()

        # 2. Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5,
                                       subsample=0.8, random_state=42)
        m, p = eval_model(gb, "GradientBoosting_v9", X_train, y_train, X_test, y_true, y_naive, h_hours)
        h_results["GradientBoosting_v9"] = m
        h_preds["GradientBoosting_v9"] = p.tolist()

        # 3. ElasticNet (Needs scaling)
        scaler = StandardScaler()
        scaler.fit(X_train)
        en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        m, p = eval_model(en, "ElasticNet_v9", X_train, y_train, X_test, y_true, y_naive, h_hours, scaler=scaler)
        h_results["ElasticNet_v9"] = m
        h_preds["ElasticNet_v9"] = p.tolist()

        # 4. Ensemble (Voting)
        voting = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42))
        ])
        m, p = eval_model(voting, "VotingEnsemble_v9", X_train, y_train, X_test, y_true, y_naive, h_hours)
        h_results["VotingEnsemble_v9"] = m
        h_preds["VotingEnsemble_v9"] = p.tolist()

        # 5. Stacking
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42))
        ]
        stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=3, n_jobs=-1)
        m, p = eval_model(stacking, "Stacking_v9", X_train, y_train, X_test, y_true, y_naive, h_hours)
        h_results["Stacking_v9"] = m
        h_preds["Stacking_v9"] = p.tolist()

        results[f"{h_hours}h"] = h_results
        preds[f"{h_hours}h"] = h_preds

    # Save
    _save(results, "sklearn_metrics", freq)
    _save(preds, "sklearn_preds", freq)
    print(f"\n[v9] ✅ Sklearn Suite ({freq}) saved.", flush=True)


def main():
    print("=" * 70, flush=True)
    print(f"v9 SKLEARN SUITE RETRAIN — Segment-Aware", flush=True)
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

    print(f"\n✅ Total Sklearn time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
