"""Patch standardized_metrics.json with RMSE, R², Forecast Bias, and DA.

Reads existing *_metrics_*.json files for RMSE/R²/Bias (already computed),
and computes Directional Accuracy (DA) from *_preds_*.json files.

Usage:
    uv run python scripts/patch_v9_metrics.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "research" / "experiments" / "v9_final"
STD_PATH = PROJECT_ROOT / "research" / "experiments" / "standardized_metrics.json"

HORIZONS = ["1h", "6h", "24h"]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_freq(filename: str) -> str:
    """Extract frequency from filename like 'lgbm_metrics_15m_xxx.json'."""
    if "_15m_" in filename:
        return "15m"
    if "_30m_" in filename:
        return "30m"
    return "1h"


def collect_metrics_from_files() -> dict:
    """Scan all *_metrics_*.json in v9_final and collect RMSE, R², Bias per model per horizon.

    Returns:
        {horizon: {model_name_with_freq: {"rmse": float, "r2": float, "forecast_bias": float}}}
    """
    result: dict[str, dict] = {h: {} for h in HORIZONS}
    metrics_files = sorted(EXP_DIR.glob("*_metrics_*.json"))

    print(f"📂 Found {len(metrics_files)} metrics files in v9_final/")

    for mf in metrics_files:
        freq = _resolve_freq(mf.name)
        data = load_json(mf)

        for h in HORIZONS:
            h_data = data.get(h, {})
            for model_name, metrics in h_data.items():
                if not isinstance(metrics, dict):
                    continue
                # Build the key as it appears in standardized_metrics.json
                # e.g., "LightGBM_v9" from metrics file → "LightGBM_v9_15m" in standardized
                std_key = f"{model_name}_{freq}"

                rmse = metrics.get("rmse")
                r2 = metrics.get("r2")
                bias = metrics.get("forecast_bias")

                if rmse is not None:
                    result[h][std_key] = {
                        "rmse": round(rmse, 4),
                        "r2": round(r2, 4) if r2 is not None else None,
                        "forecast_bias": round(bias, 4) if bias is not None else None,
                    }
                    # Also store without freq suffix for fallback matching
                    result[h][model_name] = result[h][std_key]

    for h in HORIZONS:
        print(f"  {h}: collected metrics for {len(result[h])} model variants")

    return result


def compute_directional_accuracy() -> dict:
    """Compute Directional Accuracy from prediction files.

    DA = % of times the model correctly predicts the direction of change.
    Direction at time t: sign(actual[t] - actual[t-1]) vs sign(pred[t] - pred[t-1])

    Returns:
        {horizon: {model_name_with_freq: da_percentage}}
    """
    result: dict[str, dict] = {h: {} for h in HORIZONS}
    preds_files = sorted(EXP_DIR.glob("*_preds_*.json"))

    print(f"\n📂 Found {len(preds_files)} prediction files for DA calculation")

    for pf in preds_files:
        freq = _resolve_freq(pf.name)
        data = load_json(pf)

        for h in HORIZONS:
            h_data = data.get(h, {})
            for model_name, preds in h_data.items():
                if not isinstance(preds, list) or len(preds) < 3:
                    continue

                std_key = f"{model_name}_{freq}"

                # For DA we compare consecutive prediction changes
                # Since we don't have actuals separately, we use the predictions
                # to at least compute self-consistency.
                # However, we CAN cross-reference with Persistence predictions
                # which ARE the actuals (shifted).
                result[h][std_key] = None  # placeholder

    # For proper DA, we need actuals. Let's check if Persistence preds exist
    # Persistence_1h predictions = actuals shifted by 1 step = actuals at t-1
    # So actuals[t] ≈ we need to reconstruct from the metrics files

    # Alternative: compute DA using pairs of prediction files where one is Persistence
    # Persistence preds ARE the last-known actual values, so:
    # actual[t] can be reconstructed as: pred[t] + error[t]
    # But we don't have individual errors...

    # Best approach: Use the prediction arrays directly.
    # For each model, compute direction changes in predictions,
    # then compare with a reference (Persistence = actuals lagged).
    # Since Persistence IS "use last value", Persistence preds are actuals[t-h].

    # Let's find if we have any ensemble or persistence preds
    ensemble_preds = {}
    for pf in preds_files:
        freq = _resolve_freq(pf.name)
        data = load_json(pf)
        for h in HORIZONS:
            h_data = data.get(h, {})
            for model_name, preds in h_data.items():
                if isinstance(preds, list):
                    std_key = f"{model_name}_{freq}"
                    if h not in ensemble_preds:
                        ensemble_preds[h] = {}
                    ensemble_preds[h][std_key] = preds

    # Without separate actuals file, we cannot compute true DA.
    # We'll mark DA as needing actuals and skip for now,
    # OR we can load actuals from the dataset files.
    print("  ⚠️  DA computation requires actuals — attempting to load from dataset...")

    # Try loading actuals from the processed dataset
    actuals = _load_actuals_from_dataset()
    if not actuals:
        print("  ❌ Could not load actuals. DA will be skipped.")
        return {h: {} for h in HORIZONS}

    # Now compute DA for each model
    for h in HORIZONS:
        h_preds_all = ensemble_preds.get(h, {})
        h_actuals = actuals.get(h)

        if h_actuals is None:
            continue

        for model_key, preds in h_preds_all.items():
            # Align lengths (take tail)
            n = min(len(preds), len(h_actuals))
            if n < 3:
                continue
            p = preds[-n:]
            a = h_actuals[-n:]

            # DA: direction of change
            correct = 0
            total = 0
            for i in range(1, len(a)):
                actual_dir = a[i] - a[i - 1]
                pred_dir = p[i] - p[i - 1]
                if (actual_dir > 0 and pred_dir > 0) or \
                   (actual_dir < 0 and pred_dir < 0) or \
                   (actual_dir == 0 and pred_dir == 0):
                    correct += 1
                total += 1

            if total > 0:
                da = round((correct / total) * 100, 2)
                result[h][model_key] = da

        counted = sum(1 for v in result[h].values() if v is not None)
        print(f"  {h}: computed DA for {counted} models")

    return result


def _load_actuals_from_dataset() -> dict:
    """Try to reconstruct actuals per horizon from the processed dataset.

    We look for Persistence predictions — since Persistence just predicts
    the last known value, the actual values at those timesteps can be
    reconstructed by finding the metrics: actual = pred + error.

    Alternative: load the raw test split from the parquet/CSV files.
    """
    actuals = {}

    # Strategy: find sklearn or lgbm metrics files which contain both
    # predictions and metrics. Then use: actuals = preds + residuals.
    # But we don't have residuals per-sample.

    # Better strategy: Look for Persistence predictions in ensemble_preds.
    # Ensemble preds file stores "Actuals" key directly in some formats.
    for pf in sorted(EXP_DIR.glob("ensemble_preds_*.json")):
        freq = _resolve_freq(pf.name)
        data = load_json(pf)
        for h in HORIZONS:
            h_data = data.get(h, {})
            # Check for "Actuals" key
            if "Actuals" in h_data:
                actuals[h] = h_data["Actuals"]
                print(f"    ✅ Found Actuals in ensemble_preds ({freq}) for {h}: {len(actuals[h])} samples")
                continue

            # Check for keys containing "Actual" or "actual"
            for key in h_data:
                if "actual" in key.lower():
                    actuals[h] = h_data[key]
                    print(f"    ✅ Found actuals key '{key}' in ensemble_preds ({freq}) for {h}")
                    break

    if len(actuals) == len(HORIZONS):
        return actuals

    # Fallback: try loading from dataset CSVs
    # Look for test split data
    dataset_dir = PROJECT_ROOT / "dataset"
    for freq_label in ["30min", "1h"]:
        csv_path = dataset_dir / f"processed_{freq_label}.csv"
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, usecols=["pm25"], nrows=5)
                print(f"    📊 Found {csv_path.name} — but need test split info to extract actuals")
            except Exception:
                pass

    # If we still don't have actuals, check if unified_mase_all.json
    # or any other file stores them
    print("    ⚠️  No 'Actuals' key found in prediction files.")
    print("    💡 DA will be computed post-hoc if actuals become available.")

    return actuals


def patch_standardized_metrics(collected: dict, da_data: dict) -> None:
    """Patch standardized_metrics.json with RMSE, R², Bias, and DA."""
    std = load_json(STD_PATH)
    results = std.get("results", {})

    patched_count = 0
    for h in HORIZONS:
        h_results = results.get(h, {})
        for model_name, model_data in h_results.items():
            # Try exact match first, then without freq suffix
            metrics = collected.get(h, {}).get(model_name)
            if metrics is None:
                # Try stripping _v9 suffix variations
                for key in collected.get(h, {}):
                    if model_name.startswith(key.split("_v9")[0]):
                        metrics = collected[h][key]
                        break

            if metrics:
                model_data["rmse"] = metrics["rmse"]
                model_data["r2"] = metrics["r2"]
                model_data["forecast_bias"] = metrics.get("forecast_bias")
                patched_count += 1

            # Add DA if available
            da_val = da_data.get(h, {}).get(model_name)
            if da_val is not None:
                model_data["da"] = da_val

    # Update metadata
    from datetime import datetime
    std["_metadata"]["patched"] = datetime.now().isoformat()
    std["_metadata"]["patched_fields"] = ["rmse", "r2", "forecast_bias", "da"]

    with open(STD_PATH, "w", encoding="utf-8") as f:
        json.dump(std, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Patched {patched_count} model entries in standardized_metrics.json")
    print(f"   Path: {STD_PATH}")


def verify_patch():
    """Quick verification of patched data."""
    std = load_json(STD_PATH)
    results = std.get("results", {})

    print("\n── Verification ──")
    for h in HORIZONS:
        h_results = results.get(h, {})
        total = len(h_results)
        has_rmse = sum(1 for m in h_results.values() if m.get("rmse") is not None and m["rmse"] != "null")
        has_r2 = sum(1 for m in h_results.values() if m.get("r2") is not None)
        has_da = sum(1 for m in h_results.values() if m.get("da") is not None)
        has_bias = sum(1 for m in h_results.values() if m.get("forecast_bias") is not None)

        print(f"  {h}: {total} models | RMSE: {has_rmse}/{total} | R²: {has_r2}/{total} | DA: {has_da}/{total} | Bias: {has_bias}/{total}")

    # Show a sample
    sample_h = "6h"
    sample_models = list(results.get(sample_h, {}).items())[:3]
    print(f"\n  Sample ({sample_h}):")
    for name, data in sample_models:
        print(f"    {name}: MAE={data.get('mae')}, RMSE={data.get('rmse')}, "
              f"R²={data.get('r2')}, DA={data.get('da')}, Bias={data.get('forecast_bias')}")


def main():
    print("=" * 60)
    print("🔧 Patch v9 Standardized Metrics")
    print("   Adding: RMSE, R², Forecast Bias, Directional Accuracy")
    print("=" * 60)

    if not EXP_DIR.exists():
        print(f"❌ Directory not found: {EXP_DIR}")
        sys.exit(1)

    if not STD_PATH.exists():
        print(f"❌ File not found: {STD_PATH}")
        sys.exit(1)

    # Step 1: Collect RMSE, R², Bias from metrics files
    print("\n── Step 1: Collect RMSE/R²/Bias from metrics files ──")
    collected = collect_metrics_from_files()

    # Step 2: Compute DA from prediction files
    print("\n── Step 2: Compute Directional Accuracy ──")
    da_data = compute_directional_accuracy()

    # Step 3: Patch standardized_metrics.json
    print("\n── Step 3: Patch standardized_metrics.json ──")
    patch_standardized_metrics(collected, da_data)

    # Step 4: Verify
    verify_patch()

    print("\n🏁 Done!")


if __name__ == "__main__":
    main()
