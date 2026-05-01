"""Automated Retraining Pipeline — MLOps Orchestrator.

Standardized wrapper that runs the full retraining pipeline:
  1. Data preparation (Clean → Impute → Features)
  2. Model training (LightGBM, sklearn, ARIMA, GRU/LSTM)
  3. Metric evaluation & snapshot creation
  4. SHAP explainability update
  5. Dashboard cache precomputation

Usage:
    # Full retrain (all models):
    uv run python scripts/train_pipeline.py --all 2>&1 | tee research/logs/retrain.log

    # Selective retrain:
    uv run python scripts/train_pipeline.py --models lightgbm gru

    # Data prep only (no training):
    uv run python scripts/train_pipeline.py --data-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
LOG_DIR = RESEARCH_DIR / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        _log(f"Script not found: {script_path}", "WARN")
        return False

    _log(f"Starting: {description}")
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # Let output stream to console
            text=True,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            _log(f"✅ {description} completed ({elapsed:.0f}s)")
            return True
        else:
            _log(f"❌ {description} failed (exit code {result.returncode})", "ERROR")
            return False
    except Exception as e:
        _log(f"❌ {description} crashed: {e}", "ERROR")
        return False


def step_data_prep() -> bool:
    """Step 1: Data preparation (shared by all models)."""
    _log("=" * 60)
    _log("STEP 1/5: Data Preparation (Clean → Impute → Features)")
    _log("=" * 60)

    # Data prep is embedded in retrain_v7_full.py's prepare_data()
    # We verify data files exist
    raw_data = PROJECT_ROOT / "dataset" / "raw" / "final_dataset.csv"
    if not raw_data.exists():
        _log(f"Raw data not found: {raw_data}", "ERROR")
        return False

    _log(f"Raw data: {raw_data} ✅")
    _log("Data prep will run as part of model training.")
    return True


def step_train_models(models: list[str]) -> dict[str, bool]:
    """Step 2: Train selected models."""
    _log("=" * 60)
    _log(f"STEP 2/5: Model Training ({', '.join(models)})")
    _log("=" * 60)

    # retrain_v7_full.py handles all model families
    # For selective training, we'd need to uncomment specific sections
    # For now, run the full retrain script
    success = _run_script("retrain_v7_full.py", "Full Pipeline Retrain (v7)")
    return {m: success for m in models}


def step_standardize_metrics() -> bool:
    """Step 3: Standardize metrics with unified Persistence baseline."""
    _log("=" * 60)
    _log("STEP 3/5: Standardize Metrics (Unified MASE)")
    _log("=" * 60)

    std_metrics = RESEARCH_DIR / "experiments" / "standardized_metrics.json"
    if std_metrics.exists():
        _log(f"Standardized metrics exist: {std_metrics}")
        # Load and print summary
        with open(std_metrics) as f:
            data = json.load(f)
        results = data.get("results", {})
        for h_key in ["1h", "6h", "24h"]:
            h_data = results.get(h_key, {})
            best_model = min(
                ((k, v.get("mase_unified", v.get("mase", 999)))
                 for k, v in h_data.items() if isinstance(v, dict)),
                key=lambda x: x[1],
                default=("N/A", 0),
            )
            _log(f"  {h_key}: Best = {best_model[0]} (MASE = {best_model[1]:.3f})")
        return True
    else:
        _log("Standardized metrics not found — run standardize_metrics.py", "WARN")
        return False


def step_shap_explainability() -> bool:
    """Step 4: Update SHAP explainability data."""
    _log("=" * 60)
    _log("STEP 4/5: SHAP Explainability Update")
    _log("=" * 60)

    return _run_script("shap_explainability.py", "SHAP Feature Importance")


def step_log_to_db() -> bool:
    """Step 2.5: Log training results to Database."""
    _log("=" * 60)
    _log("STEP 2.5/5: Log Results to Database")
    _log("=" * 60)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.api.experiment_logger import ExperimentLogger

        logger = ExperimentLogger(mode="db")
        exp_dir = RESEARCH_DIR / "experiments" / "v8_final"

        if not exp_dir.exists():
            _log("Experiment directory not found — skipping DB log", "WARN")
            return False

        # Find the latest JSON files (by modification time)
        json_files = sorted(
            [f for f in exp_dir.glob("*.json") if "_preds_" not in f.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not json_files:
            _log("No JSON result files found", "WARN")
            return False

        # Group by model family (take latest of each)
        seen_families = set()
        files_to_log = []
        for f in json_files:
            # Extract family: "dl_20260501_071715.json" → "dl"
            parts = f.stem.split("_")
            family = parts[0] if parts else f.stem
            if family == "tft":
                family = "tft"  # tft_multi_horizon → tft
            if family not in seen_families:
                seen_families.add(family)
                files_to_log.append(f)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        synced = 0

        for filepath in files_to_log:
            # Read JSON (handle NaN/Infinity)
            text = filepath.read_text()
            text = text.replace(": NaN", ": null")
            text = text.replace(": Infinity", ": null")
            text = text.replace(": -Infinity", ": null")

            import json as json_mod
            try:
                data = json_mod.loads(text)
            except Exception as e:
                _log(f"  ❌ JSON parse error in {filepath.name}: {e}", "ERROR")
                continue

            family = filepath.stem.split("_")[0]
            exp_name = f"v8_{family}_{timestamp}"

            exp_id = logger.log_full_result(
                experiment_name=exp_name,
                result_dict=data,
                description=f"Auto-logged from train_pipeline ({filepath.name})",
                version="v8",
                source_file=filepath.name,
            )

            if exp_id is not None:
                _log(f"  ✅ Logged {filepath.name} → experiment_id={exp_id}")
                synced += 1
            else:
                _log(f"  ⏭️  {filepath.name} — already exists")

        _log(f"  Total synced: {synced} experiments")
        return synced > 0

    except Exception as e:
        _log(f"DB logging failed: {e}", "ERROR")
        return False


def step_precompute_cache() -> bool:
    """Step 5: Precompute AVP cache for Dashboard."""
    _log("=" * 60)
    _log("STEP 5/5: Dashboard Cache Precomputation")
    _log("=" * 60)

    return _run_script("precompute_avp_safe.py", "AVP Cache Precomputation")


def run_tests() -> bool:
    """Run test suite to verify pipeline integrity."""
    _log("=" * 60)
    _log("VERIFICATION: Running Test Suite")
    _log("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        text=True,
    )
    if result.returncode == 0:
        _log("✅ All tests passed!")
        return True
    else:
        _log("❌ Some tests failed!", "ERROR")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="PM2.5 Forecasting — Automated Retraining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all models (LightGBM, sklearn, ARIMA, GRU, LSTM)",
    )
    parser.add_argument(
        "--models", nargs="+", default=[],
        choices=["lightgbm", "sklearn", "arima", "gru", "lstm", "tft"],
        help="Select specific models to retrain",
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Only run data preparation (no training)",
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Skip test suite verification",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP explainability update",
    )
    parser.add_argument(
        "--skip-db", action="store_true",
        help="Skip logging results to database",
    )
    args = parser.parse_args()

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    _log("╔" + "═" * 58 + "╗")
    _log("║  PM2.5 Forecasting — Automated Retraining Pipeline       ║")
    _log("║  Author: trihx | CTU Master's Thesis (QĐ 1799)          ║")
    _log("╚" + "═" * 58 + "╝")
    _log(f"Timestamp: {datetime.now().isoformat()}")
    _log(f"Python: {sys.version.split()[0]}")

    t_total = time.time()
    report: dict[str, str] = {}

    # Step 1: Data prep
    if step_data_prep():
        report["data_prep"] = "✅"
    else:
        report["data_prep"] = "❌"
        _log("Data prep failed — aborting.", "ERROR")
        sys.exit(1)

    if args.data_only:
        _log("--data-only flag set. Skipping training.")
        return

    # Step 2: Train
    models = args.models if args.models else ["lightgbm", "sklearn", "arima", "gru", "lstm"]
    if args.all:
        models = ["lightgbm", "sklearn", "arima", "gru", "lstm"]

    train_results = step_train_models(models)
    for m, ok in train_results.items():
        report[f"train_{m}"] = "✅" if ok else "❌"

    # Step 2.5: Log to DB
    if not args.skip_db:
        if step_log_to_db():
            report["db_log"] = "✅"
        else:
            report["db_log"] = "⚠️"

    # Step 3: Standardize metrics
    if step_standardize_metrics():
        report["standardize"] = "✅"
    else:
        report["standardize"] = "⚠️"

    # Step 4: SHAP
    if not args.skip_shap:
        if step_shap_explainability():
            report["shap"] = "✅"
        else:
            report["shap"] = "⚠️"

    # Step 5: Cache
    if step_precompute_cache():
        report["cache"] = "✅"
    else:
        report["cache"] = "⚠️"

    # Verification
    if not args.skip_tests:
        if run_tests():
            report["tests"] = "✅"
        else:
            report["tests"] = "❌"

    # Summary
    total_time = time.time() - t_total
    _log("")
    _log("╔" + "═" * 58 + "╗")
    _log("║  PIPELINE SUMMARY                                       ║")
    _log("╚" + "═" * 58 + "╝")
    for step, status in report.items():
        _log(f"  {step:<20} {status}")
    _log(f"\n  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    _log("Done.")


if __name__ == "__main__":
    main()
