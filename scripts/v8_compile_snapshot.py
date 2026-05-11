"""Compile v8 snapshot and standardized metrics."""
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "research" / "experiments" / "v8_final"

def main():
    # 1. Load unified MASE
    with open(EXP_DIR / "unified_mase.json", encoding="utf-8") as f:
        unified_mase = json.load(f)

    # 2. Load R2
    with open(EXP_DIR / "r2_multi_horizon.json", encoding="utf-8") as f:
        r2 = json.load(f)

    # 3. Create standardized metrics format
    std_metrics = {
        "_metadata": {
            "description": "Standardized metrics with unified Persistence baseline",
            "generated": datetime.now().isoformat(),
            "unified_test": "val_end to end, real-only, consistent across all models",
            "note": "mase_unified uses common Persistence MAE; mase_original uses each run's own Persistence"
        },
        "unified_persistence": {},
        "results": {}
    }

    horizons = ["1h", "6h", "24h"]
    for h in horizons:
        std_metrics["results"][h] = {}
        # Get persistence from unified_mase
        if h in unified_mase and "Persistence" in unified_mase[h]:
            pers = unified_mase[h]["Persistence"]
            std_metrics["unified_persistence"][h] = {
                "mae": pers["mae"],
                "mase": pers["mase_hyndman"],
                "n_samples": pers["n_samples"]
            }
            std_metrics["results"][h]["Persistence"] = {
                "mae": pers["mae"],
                "rmse": None,
                "mase": 1.0,
                "mase_unified": 1.0,
                "r2": r2.get(h, {}).get("Persistence", {}).get("r2"),
                "n_test": pers["n_samples"],
                "source": "unified"
            }

        # Populate models
        if h in unified_mase:
            for model_name, metrics in unified_mase[h].items():
                if model_name == "Persistence":
                    continue
                std_metrics["results"][h][model_name] = {
                    "mae": metrics["mae"],
                    "rmse": None,
                    "mase_unified": metrics["mase_hyndman"],
                    "r2": r2.get(h, {}).get(model_name, {}).get("r2"),
                    "source": "v8_final"
                }

    # Save standardized metrics
    std_path = PROJECT_ROOT / "research" / "experiments" / "standardized_metrics.json"
    with open(std_path, "w", encoding="utf-8") as f:
        json.dump(std_metrics, f, indent=2)
    print(f"✅ Saved {std_path}")

    # 4. Create v8 snapshot for dashboard
    snapshot = {
        "version": "v8_scientific_audit",
        "description": "Final v8 pipeline after KNN look-ahead fix and unified MASE",
        "timestamp": datetime.now().isoformat(),
        "metrics": std_metrics["results"]
    }

    snapshot_path = PROJECT_ROOT / "research" / "experiments" / "dashboard_runs" / "v8_cqr_aci.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
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
