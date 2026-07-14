"""Level 2-3: ML Models with Walk-Forward Validation.

Usage:
    uv run python -m src.models.run_ml

Per SKILL.md §6.1:
    Level 2: Linear ML  → Ridge, Lasso, ElasticNet
    Level 3: Tree-based → Random Forest, XGBoost, LightGBM

Per SKILL.md §6.2:
    Tree-based: NO scaling
    Linear: YES scaling (inside Pipeline to prevent leakage)

Per SKILL.md §6.3:
    Walk-Forward with Expanding Window (TimeSeriesSplit)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import TARGET_COL
from src.evaluation.metrics import evaluate_forecast
from src.evaluation.splitter import temporal_train_val_test_split
from src.utils.logging import setup_logging

# Lazy imports for optional dependencies
_xgb = None
_lgb = None


def _get_xgboost():
    global _xgb
    if _xgb is None:
        import xgboost as xgb

        _xgb = xgb
    return _xgb


def _get_lightgbm():
    global _lgb
    if _lgb is None:
        import lightgbm as lgb

        _lgb = lgb
    return _lgb


def get_ml_models() -> list[tuple[str, object, bool]]:
    """Get all ML models to evaluate.

    Returns:
        List of (name, model_or_pipeline, needs_scaling) tuples.
    """
    xgb = _get_xgboost()
    lgb = _get_lightgbm()

    models = [
        # Level 2: Linear ML (need scaling in Pipeline)
        (
            "Ridge",
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            True,
        ),
        (
            "Lasso",
            Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1, max_iter=5000))]),
            True,
        ),
        (
            "ElasticNet",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
                ]
            ),
            True,
        ),
        # Level 3: Tree-based ML (NO scaling needed)
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
            False,
        ),
        (
            "XGBoost",
            xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "LightGBM",
            lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            ),
            False,
        ),
    ]
    return models


def walk_forward_evaluate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    n_splits: int = 5,
) -> dict:
    """Train with walk-forward CV then evaluate on test set.

    Per SKILL.md §6.3: Expanding Window validation.

    Args:
        model: Fitted sklearn-compatible model.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        model_name: Name for logging.
        n_splits: Number of CV folds.

    Returns:
        Dict with cv_scores and test_metrics.
    """
    # Walk-Forward Cross-Validation on training data
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_vl = X_train.iloc[val_idx]
        y_vl = y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred_val = model.predict(X_vl)
        fold_mae = float(np.mean(np.abs(y_vl.values - y_pred_val)))
        cv_maes.append(fold_mae)
        logger.debug(f"  {model_name} Fold {fold + 1}: MAE={fold_mae:.4f} (train={len(X_tr)}, val={len(X_vl)})")

    cv_mean = float(np.mean(cv_maes))
    cv_std = float(np.std(cv_maes))
    logger.info(f"  {model_name} CV: MAE={cv_mean:.4f} ± {cv_std:.4f}")

    # Final training on full training set → test evaluation
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    # Naive reference: persistence (lag_1h)
    if "pm25_lag_1h" in X_test.columns:
        y_naive = X_test["pm25_lag_1h"].values
    else:
        y_naive = np.full(len(y_test), float(y_train.mean()))

    test_metrics = evaluate_forecast(
        y_true=y_test.values,
        y_pred=y_pred_test,
        y_naive=y_naive,
        model_name=model_name,
    )
    test_metrics["cv_mae_mean"] = round(cv_mean, 4)
    test_metrics["cv_mae_std"] = round(cv_std, 4)

    return test_metrics


def main() -> None:
    """Run all Level 2-3 ML models with walk-forward validation."""
    setup_logging(level="INFO", log_dir="research/runs")
    logger.info("🚀 Level 2-3: ML Models Experiment")

    # 1. Load Marts data
    marts_path = Path("dataset/processed/marts_features.csv")
    if not marts_path.exists():
        logger.error(f"❌ Marts data not found: {marts_path}")
        return

    df = pd.read_csv(marts_path, index_col=0, parse_dates=True)
    logger.info(f"📊 Loaded: {len(df):,} rows × {len(df.columns)} cols")

    # 2. Temporal Split
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(df, target_col=TARGET_COL)

    # Combine train + val for final training (use CV for validation)
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    logger.info(f"  Train+Val: {len(X_train_full):,} rows, Test: {len(X_test):,} rows")

    # 3. Load baseline results for comparison
    baseline_path = Path("research/experiments/baselines/latest_results.json")
    baseline_mae = None
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            baseline_data = json.load(f)
        baseline_mae = baseline_data.get("best_mae")
        logger.info(f"  📏 Baseline target: MAE < {baseline_mae}")

    # 4. Train and evaluate all models
    models = get_ml_models()
    results = []

    logger.info("=" * 60)
    logger.info("Training & Evaluating ML Models (Walk-Forward CV)")
    logger.info("=" * 60)

    for name, model, _needs_scaling in models:
        logger.info(f"\n--- {name} ---")
        try:
            metrics = walk_forward_evaluate(
                model=model,
                X_train=X_train_full,
                y_train=y_train_full,
                X_test=X_test,
                y_test=y_test,
                model_name=name,
            )
            results.append(metrics)
        except Exception as e:
            logger.error(f"  ❌ {name} failed: {e}")

    # 5. Results comparison
    results_df = pd.DataFrame(results).sort_values("mae")

    logger.info("\n" + "=" * 60)
    logger.info("📋 ML Model Results (sorted by MAE)")
    logger.info("=" * 60)
    for _, row in results_df.iterrows():
        beat = ""
        if baseline_mae and isinstance(row["mae"], float) and row["mae"] < baseline_mae:
            beat = " 🏆 BEATS BASELINE"
        logger.info(
            f"  {row['model']:<14} | MAE={row['mae']:<8} | "
            f"RMSE={row['rmse']:<8} | MASE={row['mase']:<8} "
            f"{row['pass_naive']}{beat}"
        )

    # 6. Save results
    output_dir = Path("research/experiments/ml_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"ml_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)

    json_path = output_dir / "latest_results.json"
    best = results_df.iloc[0]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "n_train": len(X_train_full),
                "n_test": len(X_test),
                "baseline_mae": baseline_mae,
                "results": results,
                "best_model": str(best["model"]),
                "best_mae": float(best["mae"]),
                "beats_baseline": bool(baseline_mae and float(best["mae"]) < baseline_mae),
            },
            f,
            indent=2,
        )

    logger.info(f"\n💾 Results: {csv_path}")
    logger.info(f"💾 JSON: {json_path}")

    # 7. Summary
    logger.info("\n" + "=" * 60)
    best_model = results_df.iloc[0]
    if baseline_mae and float(best_model["mae"]) < baseline_mae:
        improvement = (1 - float(best_model["mae"]) / baseline_mae) * 100
        logger.info(
            f"🏆 Best ML: {best_model['model']} (MAE={best_model['mae']}) — {improvement:.1f}% better than baseline"
        )
    else:
        logger.info(f"📊 Best ML: {best_model['model']} (MAE={best_model['mae']})")
    logger.info("✅ ML Experiment Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
