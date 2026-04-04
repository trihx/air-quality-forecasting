"""Run EDA pipeline — entry point script.

Usage:
    uv run python -m src.eda.run_eda
"""

import json
from pathlib import Path

from loguru import logger

from src.data.cleaner import clean_data
from src.data.loader import load_raw_data
from src.data.validator import DataValidator
from src.eda.analyzer import run_full_eda
from src.utils.logging import setup_logging


def main() -> None:
    """Run full EDA pipeline: load → validate → clean → validate → analyze."""
    setup_logging(level="INFO", log_dir="research/runs")

    logger.info("🚀 Starting EDA Pipeline")

    # 1. Load raw data
    df_raw = load_raw_data()

    # 2. Validate staging (raw data)
    validator = DataValidator()
    validator.validate_staging(df_raw)
    if validator.has_critical_failures():
        logger.error("❌ Staging validation FAILED — aborting")
        return

    # 3. Clean data
    df_clean = clean_data(df_raw)

    # 4. Validate intermediate (cleaned data)
    validator = DataValidator()
    validator.validate_intermediate(df_clean)
    if validator.has_critical_failures():
        logger.error("❌ Intermediate validation FAILED — check cleaning pipeline")
        return

    # 5. Save cleaned data to interim
    interim_path = Path("dataset/interim/cleaned_hourly.csv")
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(interim_path)
    logger.info(f"💾 Cleaned data saved to {interim_path} ({len(df_clean):,} rows)")

    # 6. Run EDA
    results = run_full_eda(df_clean, output_dir="research/eda")

    # 7. Save results as JSON
    results_path = Path("research/eda/eda_results.json")

    # Convert non-serializable values
    def _serialize(obj: object) -> object:
        if hasattr(obj, "item"):
            return obj.item()  # type: ignore[union-attr]
        if hasattr(obj, "__float__"):
            return float(obj)  # type: ignore[arg-type]
        return str(obj)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    logger.info(f"📊 EDA results saved to {results_path}")

    logger.info("✅ EDA Pipeline Complete!")


if __name__ == "__main__":
    main()
