"""Compile v9 snapshot and standardized metrics."""
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "research" / "experiments" / "v9_final"

def main():
    # 1. Load unified MASE
    with open(EXP_DIR / "unified_mase_all.json") as f:
        unified_mase = json.load(f)

    # 3. Create standardized metrics format
    std_metrics = {
        "_metadata": {
            "description": "Standardized metrics with multi-resolution v9 models",
            "generated": datetime.now().isoformat(),
            "unified_test": "val_end to end, anchored by test set",
        },
        "unified_persistence": {},
        "results": {}
    }

    horizons = ["1h", "6h", "24h"]
    for h in horizons:
        std_metrics["results"][h] = {}
        
        if h in unified_mase:
            # Add persistence baseline for 30m as the main one for metrics UI tracking if needed,
            # but we just include all persistences inside the results.
            for model_name, metrics in unified_mase[h].items():
                std_metrics["results"][h][model_name] = {
                    "mae": metrics["mae"],
                    "rmse": None,
                    "mase_unified": metrics["mase_hyndman"],
                    "r2": None, # Skipped for v9 as MASE/MAE are prioritized
                    "n_samples": metrics.get("n_samples"),
                    "freq": metrics.get("freq"),
                    "source": "v9_final"
                }

    # Save standardized metrics
    std_path = PROJECT_ROOT / "research" / "experiments" / "standardized_metrics.json"
    with open(std_path, "w") as f:
        json.dump(std_metrics, f, indent=2)
    print(f"✅ Saved {std_path}")

    # 4. Create v9 snapshot for dashboard
    snapshot = {
        "version": "v9_multi_resolution",
        "description": "Final v9 pipeline with 15m/30m/1h resolutions and Ensembles",
        "timestamp": datetime.now().isoformat(),
        "metrics": std_metrics["results"]
    }

    snapshot_path = PROJECT_ROOT / "research" / "experiments" / "dashboard_runs" / "v9_multi_resolution.json"
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"✅ Saved {snapshot_path}")

    # 5. Re-generate manifest
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.generate_manifest import main as gen_manifest
    print("\nRe-generating manifest...")
    gen_manifest()

if __name__ == "__main__":
    main()
