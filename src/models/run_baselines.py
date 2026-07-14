"""Run Baseline Experiment — Level 0 Models.

Usage:
    uv run python -m src.models.run_baselines

Pipeline:
    Load Marts data → Temporal Split → Fit & Predict → Evaluate → Save results
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.loader import TARGET_COL
from src.evaluation.metrics import evaluate_forecast
from src.evaluation.splitter import temporal_train_val_test_split
from src.models.baselines import get_all_baselines
from src.utils.logging import setup_logging


def main() -> None:
    """Run all Level 0 baselines and compare results."""
    setup_logging(level="INFO", log_dir="research/runs")
    logger.info("🚀 Level 0: Baseline Models Experiment")

    # 1. Load Marts data
    marts_path = Path("dataset/processed/marts_features.csv")
    if not marts_path.exists():
        logger.error(f"❌ Marts data not found: {marts_path}")
        logger.error("   Run feature engineering first: uv run python -m src.features.run_features")
        return

    df = pd.read_csv(marts_path, index_col=0, parse_dates=True)
    logger.info(f"📊 Loaded Marts data: {len(df):,} rows × {len(df.columns)} cols")

    # 2. Temporal Split
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(df, target_col=TARGET_COL)

    # 3. Fit and evaluate all baselines
    baselines = get_all_baselines()
    results = []

    logger.info("=" * 60)
    logger.info("Training & Evaluating Baselines")
    logger.info("=" * 60)

    # Use persistence as the naive reference for MASE
    persistence = baselines[0]
    persistence.fit(y_train)
    y_naive_test = persistence.predict(X_test)

    for model in baselines:
        model.fit(y_train)

        # Evaluate on test set
        y_pred_test = model.predict(X_test)

        result = evaluate_forecast(
            y_true=y_test.values,
            y_pred=y_pred_test,
            y_naive=y_naive_test,
            model_name=model.name,
        )
        results.append(result)

    # 4. Results comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")

    logger.info("=" * 60)
    logger.info("📋 Baseline Results (sorted by MAE)")
    logger.info("=" * 60)
    for _, row in results_df.iterrows():
        logger.info(
            f"  {row['model']:<16} | MAE={row['mae']:<8} | "
            f"RMSE={row['rmse']:<8} | MASE={row['mase']:<8} {row['pass_naive']}"
        )

    # 5. Save results
    output_dir = Path("research/experiments/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = output_dir / f"baseline_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"💾 Results CSV: {csv_path}")

    # JSON (for programmatic access)
    json_path = output_dir / "latest_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "results": results,
                "best_baseline": results_df.iloc[0]["model"],
                "best_mae": float(results_df.iloc[0]["mae"]),
            },
            f,
            indent=2,
        )
    logger.info(f"💾 Results JSON: {json_path}")

    # 6. Summary
    best = results_df.iloc[0]
    logger.info("=" * 60)
    logger.info(f"🏆 Best Baseline: {best['model']} (MAE={best['mae']})")
    logger.info(f"   Any Level 1+ model must beat MAE < {best['mae']} to be useful")
    logger.info("✅ Baseline Experiment Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
