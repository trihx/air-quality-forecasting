"""Feature Selection + Optuna Hyperparameter Tuning.

Goal: Beat Persistence baseline (MAE=1.821, MASE=1.000)
Strategy:
  1. Feature importance → select top-K features (remove noise)
  2. Optuna tuning for Lasso + LightGBM (best candidates from ml-007~ml-012)
  3. Walk-Forward CV → eval on test set

Usage:
    uv run python scripts/tune_and_select.py
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.loader import TARGET_COL
from src.evaluation.metrics import evaluate_forecast
from src.evaluation.splitter import temporal_train_val_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Step 1: Feature Selection — Importance-based
# ============================================================


def select_features_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 30,
) -> list[str]:
    """Select top-K features using Random Forest importances.

    Why RF: Non-linear, captures interactions, robust feature ranking.
    Why not correlation: Misses non-linear relationships.
    """
    logger.info(f"🔍 Feature Selection: ranking {len(X_train.columns)} features...")

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    selected = importances.head(top_k).index.tolist()

    logger.info(f"  Top {top_k} features (cumulative importance = {importances.head(top_k).sum():.3f}):")
    for i, (feat, imp) in enumerate(importances.head(10).items()):
        logger.info(f"    {i + 1:2d}. {feat:<35s} {imp:.4f}")
    if top_k > 10:
        logger.info(f"    ... + {top_k - 10} more features")

    # Log what was dropped
    dropped = importances.tail(len(importances) - top_k)
    logger.info(f"  ❌ Dropped {len(dropped)} low-importance features (sum imp = {dropped.sum():.4f})")

    return selected


# ============================================================
# Step 2: Optuna Tuning
# ============================================================


def tune_lasso_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_splits: int = 5,
) -> dict:
    """Tune Lasso regularization strength with Optuna."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
        max_iter = trial.suggest_int("max_iter", 1000, 10000)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=alpha, max_iter=max_iter)),
            ]
        )

        cv_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_vl)
            mae = float(np.mean(np.abs(y_vl.values - y_pred)))
            cv_maes.append(mae)

        return float(np.mean(cv_maes))

    study = optuna.create_study(direction="minimize", study_name="lasso_tune")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"  🏆 Lasso best: MAE={study.best_value:.4f}, params={study.best_params}")
    return {
        "best_params": study.best_params,
        "best_cv_mae": study.best_value,
        "n_trials": n_trials,
    }


def tune_lightgbm_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_splits: int = 5,
) -> dict:
    """Tune LightGBM hyperparameters with Optuna."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=n_splits)

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
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_vl)
            mae = float(np.mean(np.abs(y_vl.values - y_pred)))
            cv_maes.append(mae)

        return float(np.mean(cv_maes))

    study = optuna.create_study(direction="minimize", study_name="lgbm_tune")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"  🏆 LightGBM best: MAE={study.best_value:.4f}, params={study.best_params}")
    return {
        "best_params": study.best_params,
        "best_cv_mae": study.best_value,
        "n_trials": n_trials,
    }


# ============================================================
# Step 3: Final Evaluation
# ============================================================


def evaluate_tuned_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Train tuned model on full training set, evaluate on test."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Naive reference: persistence (lag_1h)
    if "pm25_lag_1h" in X_test.columns:
        y_naive = X_test["pm25_lag_1h"].values
    else:
        y_naive = np.full(len(y_test), float(y_train.mean()))

    metrics = evaluate_forecast(
        y_true=y_test.values,
        y_pred=y_pred,
        y_naive=y_naive,
        model_name=model_name,
    )
    return metrics


def main() -> None:
    """Feature Selection + Optuna Tuning Pipeline."""
    from src.utils.logging import setup_logging

    setup_logging(level="INFO", log_dir="research/runs")
    logger.info("🚀 Feature Selection + Optuna Hyperparameter Tuning")
    logger.info("=" * 60)

    # 1. Load Marts data
    marts_path = Path("dataset/processed/marts_features.csv")
    df = pd.read_csv(marts_path, index_col=0, parse_dates=True)
    logger.info(f"📊 Loaded: {len(df):,} rows × {len(df.columns)} cols")

    # 2. Temporal Split
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(df, target_col=TARGET_COL)
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    logger.info(f"  Train+Val: {len(X_train_full):,} rows, Test: {len(X_test):,} rows")

    # 3. Feature Selection
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Feature Selection")
    logger.info("=" * 60)

    # Try multiple top-K values to find optimal
    best_overall = {"mae": float("inf")}
    all_results = []

    for top_k in [15, 20, 25, 30, 40, 50]:
        selected_features = select_features_by_importance(X_train_full, y_train_full, top_k=top_k)

        # Ensure pm25_lag_1h is always included (needed for naive baseline)
        if "pm25_lag_1h" not in selected_features:
            selected_features.append("pm25_lag_1h")

        X_train_sel = X_train_full[selected_features]
        X_test_sel = X_test[selected_features]

        # Quick Lasso eval with default params on selected features
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.1, max_iter=5000)),
            ]
        )

        tscv = TimeSeriesSplit(n_splits=5)
        cv_maes = []
        for train_idx, val_idx in tscv.split(X_train_sel):
            X_tr = X_train_sel.iloc[train_idx]
            y_tr = y_train_full.iloc[train_idx]
            X_vl = X_train_sel.iloc[val_idx]
            y_vl = y_train_full.iloc[val_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_vl)
            cv_maes.append(float(np.mean(np.abs(y_vl.values - y_pred))))

        avg_mae = float(np.mean(cv_maes))
        logger.info(f"  top_k={top_k:3d}: Lasso CV MAE = {avg_mae:.4f}")
        all_results.append({"top_k": top_k, "cv_mae": avg_mae, "features": selected_features})

        if avg_mae < best_overall["mae"]:
            best_overall = {"mae": avg_mae, "top_k": top_k, "features": selected_features}

    best_k = best_overall["top_k"]
    best_features = best_overall["features"]
    logger.info(f"\n  ✅ Best feature count: top_k={best_k} (CV MAE={best_overall['mae']:.4f})")

    X_train_sel = X_train_full[best_features]
    X_test_sel = X_test[best_features]

    # 4. Optuna Tuning
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2: Optuna Tuning (features={best_k})")
    logger.info("=" * 60)

    # 4a. Tune Lasso
    logger.info("\n--- Lasso Tuning (100 trials) ---")
    lasso_result = tune_lasso_optuna(X_train_sel, y_train_full, n_trials=100)

    # 4b. Tune LightGBM
    logger.info("\n--- LightGBM Tuning (150 trials) ---")
    lgbm_result = tune_lightgbm_optuna(X_train_sel, y_train_full, n_trials=150)

    # 5. Final Evaluation — tuned models on test set
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Final Evaluation on Test Set")
    logger.info("=" * 60)

    import lightgbm as lgb

    results = []

    # Tuned Lasso
    tuned_lasso = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(**lasso_result["best_params"])),
        ]
    )
    lasso_metrics = evaluate_tuned_model(tuned_lasso, "Lasso_tuned", X_train_sel, y_train_full, X_test_sel, y_test)
    lasso_metrics["tuning"] = lasso_result
    results.append(lasso_metrics)

    # Tuned LightGBM
    lgbm_params = lgbm_result["best_params"].copy()
    lgbm_params["random_state"] = 42
    lgbm_params["verbose"] = -1
    lgbm_params["n_jobs"] = -1
    tuned_lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm_metrics = evaluate_tuned_model(tuned_lgbm, "LightGBM_tuned", X_train_sel, y_train_full, X_test_sel, y_test)
    lgbm_metrics["tuning"] = lgbm_result
    results.append(lgbm_metrics)

    # Also evaluate with ALL features (no selection) for comparison
    logger.info("\n--- Comparison: Tuned models on ALL features ---")

    tuned_lasso_all = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(**lasso_result["best_params"])),
        ]
    )
    lasso_all_metrics = evaluate_tuned_model(
        tuned_lasso_all, "Lasso_tuned_all", X_train_full, y_train_full, X_test, y_test
    )
    results.append(lasso_all_metrics)

    tuned_lgbm_all = lgb.LGBMRegressor(**lgbm_params)
    lgbm_all_metrics = evaluate_tuned_model(
        tuned_lgbm_all, "LightGBM_tuned_all", X_train_full, y_train_full, X_test, y_test
    )
    results.append(lgbm_all_metrics)

    # 6. Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 FINAL RESULTS — Feature Selection + Tuning")
    logger.info("=" * 60)
    logger.info("  Persistence baseline:  MAE=1.821, MASE=1.000")

    for r in sorted(results, key=lambda x: x["mae"]):
        beat = "🏆 BEATS BASELINE" if r["mase"] < 1.0 else "❌ MASE>1"
        logger.info(f"  {r['model']:<22s} | MAE={r['mae']:<8} | MASE={r['mase']:<8} | {beat}")

    # 7. Save results
    output_dir = Path("research/experiments/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"tuning_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "feature_selection": {
                    "method": "RandomForest importance",
                    "best_top_k": best_k,
                    "selected_features": best_features,
                    "feature_search": [{"top_k": r["top_k"], "cv_mae": r["cv_mae"]} for r in all_results],
                },
                "lasso_tuning": lasso_result,
                "lightgbm_tuning": lgbm_result,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"\n💾 Results saved: {json_path}")
    logger.info("✅ Tuning Complete!")


if __name__ == "__main__":
    main()
