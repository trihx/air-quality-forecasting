"""Compare Data Strategies — Run all 4 imputation strategies and evaluate.

PRINCIPLE: Train can use imputed data, Test MUST be real data only.

Usage:
    uv run python scripts/compare_data_strategies.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from src.data.cleaner import (
    _clip_physical_bounds,
    _handle_outliers,
    _remove_duplicates,
    _resample,
    _set_datetime_index,
)
from src.data.imputer import (
    Strategy,
    get_imputation_stats,
    impute_missing_data,
)
from src.data.loader import TARGET_COL, load_raw_data
from src.evaluation.metrics import evaluate_forecast
from src.features.builder import build_features

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "eda"
RESULTS_FILE = OUTPUT_DIR / "strategy_comparison.json"


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("STRATEGY COMPARISON — PM2.5 Missing Data Handling", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Load and pre-clean raw data (shared across strategies) ──
    print("\n[1/6] Loading and pre-cleaning raw data...", flush=True)
    df_raw = load_raw_data()
    print(f"  Raw: {len(df_raw):,} rows", flush=True)

    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, n_clipped = _clip_physical_bounds(df)
    df, n_outliers = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")
    print(f"  After pre-clean + resample: {len(df):,} hourly rows", flush=True)
    print(f"  PM2.5 NaN: {df[TARGET_COL].isna().sum():,}", flush=True)

    # ── Step 2: Run all 4 strategies ──
    strategies: list[tuple[Strategy, dict]] = [
        ("segment_only", {"max_gap_interp": 2}),
        ("extended_interp", {"max_gap_interp": 12}),
        ("ml_impute", {"max_gap_ml": 24, "knn_neighbors": 5}),
        ("hybrid", {"max_gap_interp": 6, "max_gap_ml": 24, "knn_neighbors": 5}),
    ]

    results = {}
    datasets = {}

    for i, (strategy, params) in enumerate(strategies, 1):
        print(f"\n{'─' * 70}", flush=True)
        print(f"[2/6] Strategy {i}/4: {strategy.upper()}", flush=True)
        print(f"{'─' * 70}", flush=True)
        print(f"  Parameters: {params}", flush=True)

        t0 = time.time()
        df_imputed = impute_missing_data(
            df.copy(),
            strategy=strategy,
            verbose=True,
            **params,
        )
        elapsed = time.time() - t0

        stats = get_imputation_stats(df_imputed)
        stats["time_seconds"] = round(elapsed, 1)
        stats["strategy"] = strategy
        stats["params"] = params

        results[strategy] = stats
        datasets[strategy] = df_imputed

        print(
            f"\n  ✅ {strategy}: {stats['total']:,} rows "
            f"({stats['real']:,} real + {stats['imputed']:,} imputed) "
            f"in {elapsed:.1f}s",
            flush=True,
        )

    # ── Step 3: Build features for each dataset ──
    print(f"\n{'─' * 70}", flush=True)
    print("[3/6] Building features for each strategy...", flush=True)
    print(f"{'─' * 70}", flush=True)

    feature_datasets = {}
    for strategy, df_s in datasets.items():
        print(f"\n  Building features for {strategy}...", flush=True)
        t0 = time.time()

        # Remove is_imputed before feature building, preserve it
        is_imputed_col = df_s["is_imputed"].copy() if "is_imputed" in df_s.columns else None
        df_for_features = df_s.drop(columns=["is_imputed"], errors="ignore")

        try:
            df_feat = build_features(df_for_features)
            # Re-attach is_imputed (aligned by index)
            if is_imputed_col is not None:
                df_feat["is_imputed"] = is_imputed_col.reindex(df_feat.index).fillna(False)
            else:
                df_feat["is_imputed"] = False

            elapsed = time.time() - t0
            feature_datasets[strategy] = df_feat
            print(f"  ✅ {strategy}: {len(df_feat):,} rows × {len(df_feat.columns)} cols ({elapsed:.1f}s)", flush=True)
            results[strategy]["n_features_rows"] = len(df_feat)
            results[strategy]["n_features_cols"] = len(df_feat.columns)
        except Exception as e:
            print(f"  ❌ {strategy}: Feature building failed: {e}", flush=True)
            results[strategy]["error"] = str(e)

    # ── Step 4: Train and evaluate LightGBM on each dataset ──
    print(f"\n{'─' * 70}", flush=True)
    print("[4/6] Training LightGBM on each dataset...", flush=True)
    print("  RULE: Test set = REAL data ONLY", flush=True)
    print(f"{'─' * 70}", flush=True)

    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        print("  ❌ LightGBM not installed. Skipping model comparison.", flush=True)
        _save_results(results)
        return

    for strategy, df_feat in feature_datasets.items():
        print(f"\n  Training on {strategy} dataset...", flush=True)
        t0 = time.time()

        try:
            metrics = _train_and_evaluate(df_feat, strategy, verbose=True)
            results[strategy]["metrics"] = metrics
            elapsed = time.time() - t0
            print(f"  ✅ {strategy}: MAE={metrics['mae']:.3f}, MASE={metrics['mase']:.3f} ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"  ❌ {strategy}: Training failed: {e}", flush=True)
            results[strategy]["error"] = str(e)

    # ── Step 5: Summary comparison ──
    print(f"\n{'═' * 70}", flush=True)
    print("[5/6] STRATEGY COMPARISON SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)
    print(
        f"\n{'Strategy':<20} {'Total':>7} {'Real':>7} {'Imputed':>8} {'MAE':>8} {'MASE':>8} {'Status':>10}", flush=True
    )
    print("─" * 70, flush=True)

    for strategy, stats in results.items():
        mae = stats.get("metrics", {}).get("mae", float("nan"))
        mase = stats.get("metrics", {}).get("mase", float("nan"))
        status = "✅ PASS" if mase < 1.0 else "❌ MASE>1"
        print(
            f"{strategy:<20} {stats['total']:>7,} {stats['real']:>7,} "
            f"{stats['imputed']:>8,} {mae:>8.3f} {mase:>8.3f} {status:>10}",
            flush=True,
        )

    # Persistence baseline
    print(f"{'persistence':<20} {'─':>7} {'─':>7} {'─':>8} {'1.821':>8} {'1.000':>8} {'baseline':>10}", flush=True)

    # ── Step 6: Save results ──
    print("\n[6/6] Saving results...", flush=True)
    _save_results(results)

    total_time = time.time() - t_start
    print(f"\n{'═' * 70}", flush=True)
    print(f"COMPLETE — Total time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print(f"{'═' * 70}", flush=True)


def _train_and_evaluate(
    df: pd.DataFrame,
    strategy_name: str,
    verbose: bool = True,
) -> dict:
    """Train LightGBM and evaluate on REAL data only.

    Split: 80/10/10 temporal split.
    Training: Uses ALL data (real + imputed).
    Validation/Test: Uses REAL data ONLY.
    """
    import lightgbm as lgb

    def _log(msg):
        return print(f"    [{strategy_name}] {msg}", flush=True) if verbose else None

    # Separate features and target
    exclude_cols = ["is_imputed", TARGET_COL]
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ("float64", "float32", "int64")]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    is_imputed = df["is_imputed"].copy() if "is_imputed" in df.columns else pd.Series(False, index=df.index)

    # Temporal split
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # Filter validation and test to REAL data only
    val_real_mask = ~is_imputed.iloc[train_end:val_end].values
    test_real_mask = ~is_imputed.iloc[val_end:].values

    X_val_real = X_val[val_real_mask]
    y_val_real = y_val[val_real_mask]
    X_test_real = X_test[test_real_mask]
    y_test_real = y_test[test_real_mask]

    _log(f"Train: {len(X_train):,} rows (real+imputed)")
    _log(f"Val:   {len(X_val_real):,}/{len(X_val):,} real rows")
    _log(f"Test:  {len(X_test_real):,}/{len(X_test):,} real rows")

    if len(X_test_real) < 10:
        _log("⚠️ Too few real test samples, using all test data")
        X_test_real = X_test
        y_test_real = y_test

    # Handle NaN in features
    X_train = X_train.fillna(0)
    X_val_real = X_val_real.fillna(0)
    X_test_real = X_test_real.fillna(0)

    # Train LightGBM with reasonable defaults
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_estimators": 500,
    }

    _log("Training LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val_real, label=y_val_real, reference=train_data)

    callbacks = [lgb.log_evaluation(period=100)]

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=callbacks,
    )

    # Predict on REAL test data only
    y_pred = model.predict(X_test_real)

    # Persistence baseline (y_naive = previous hour value)
    y_naive = y_test_real.shift(1).bfill().values

    # Calculate metrics
    metrics = evaluate_forecast(
        y_true=y_test_real.values,
        y_pred=y_pred,
        y_naive=y_naive,
        model_name=f"LightGBM_{strategy_name}",
    )
    _log(f"Results (REAL test data only): MAE={metrics['mae']}, MASE={metrics['mase']}")

    return metrics


def _save_results(results: dict) -> None:
    """Save results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_results = json.loads(json.dumps(results, default=_convert))

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
