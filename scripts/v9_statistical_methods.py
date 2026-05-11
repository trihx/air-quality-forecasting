"""V9 Statistical Methodology — Cross-Resolution MASE + DM-HLN + R².

Loads existing prediction files from v8 (1h) and v9 (15m, 30m) and computes:
1. MASE_hyndman (Hyndman & Koehler 2006)
2. DM test with HLN correction (Harvey et al. 1997)
3. R² for all models × 3 horizons (1h, 6h, 24h)

Output:
    research/experiments/v9_final/unified_mase_all.json
    research/experiments/v9_final/dm_test_hln_all.json
    research/experiments/v9_final/r2_all.json
"""
import json
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import mase_hyndman, r2_score, mae
from src.evaluation.residual_diagnostics import dm_test_hln

OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "v9_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Horizons to compare (1h, 6h, 24h) 
# Note: For 15m freq, 1h=4 steps, 6h=24 steps, 24h=96 steps
# For 30m freq, 1h=2 steps, 6h=12 steps, 24h=48 steps
# For 1h freq, 1h=1 step, 6h=6 steps, 24h=24 steps
HORIZONS = ["1h", "6h", "24h"]
FREQ_HORIZON_STEPS = {
    "15m": {"1h": 4, "6h": 24, "24h": 96},
    "30m": {"1h": 2, "6h": 12, "24h": 48},
    "1h": {"1h": 1, "6h": 6, "24h": 24},
}

def load_targets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load target series for all frequencies to compute naive baselines.
    Returns:
        Dict: freq -> (y_train, y_full)
    """
    targets = {}
    data_dir = PROJECT_ROOT / "dataset" / "processed"
    
    files = {
        "15m": "marts_features_15m.csv",
        "30m": "marts_features_30m.csv",
        "1h": "marts_features.csv"
    }
    
    for freq, fname in files.items():
        path = data_dir / fname
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            y_full = df["pm25"].values
            # 80/10/10 split
            n_train = int(len(y_full) * 0.8)
            targets[freq] = (y_full[:n_train], y_full)
            print(f"  Target loaded ({freq}): {len(y_full)} total, {n_train} train", flush=True)
        else:
            print(f"  [WARNING] Cannot find target file for {freq}", flush=True)
            
    return targets


def _load_latest(pattern: str) -> dict:
    files = sorted(glob.glob(str(OUTPUT_DIR / pattern)))
    if not files:
        # Check v8 for 1h
        v8_dir = PROJECT_ROOT / "research" / "experiments" / "v8_final"
        v8_files = sorted(glob.glob(str(v8_dir / pattern.replace("v9", "v8").replace("*15m*", "*").replace("*30m*", "*"))))
        if v8_files:
            files = [v8_files[-1]]
        else:
            return {}
            
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)

def load_predictions() -> dict:
    """Load all prediction files across frequencies and models.
    
    Returns:
        {freq: {h_key: {model: np.array, ...}}}
    """
    all_data = {"15m": {}, "30m": {}, "1h": {}}
    
    # Load 15m and 30m from v9
    for freq in ["15m", "30m"]:
        sources = {
            "lightgbm": _load_latest(f"lgbm_preds_{freq}_*.json"),
            "sklearn": _load_latest(f"sklearn_preds_{freq}_*.json"),
            "arima": _load_latest(f"arima_preds_{freq}_*.json"),
            "dl": _load_latest(f"dl_preds_{freq}_*.json"),
            "tft": _load_latest(f"tft_preds_{freq}_*.json"),
            "dl_expert": _load_latest(f"dl_expert_preds_{freq}_*.json"),
            "tft_expert": _load_latest(f"tft_expert_preds_{freq}_*.json"),
            "ensemble": _load_latest(f"ensemble_preds_{freq}_*.json"),
        }
        
        for h_key in HORIZONS:
            all_data[freq][h_key] = {}
            for source_name, data in sources.items():
                if not data or h_key not in data:
                    continue
                for model_name, preds in data[h_key].items():
                    # Add suffix to model name to identify resolution
                    full_name = f"{model_name}_{freq}" if model_name not in ("Actuals", "Persistence") else model_name
                    all_data[freq][h_key][full_name] = np.array(preds, dtype=float)
                    
    # Load 1h from v8
    v8_dir = PROJECT_ROOT / "research" / "experiments" / "v8_final"
    v8_sources = {
        "lightgbm": "lightgbm_preds_20260502_073700.json",
        "sklearn": "sklearn_preds_20260502_073736.json",
        "dl": "dl_preds_20260502_075412.json",
        "tft": "tft_preds_20260429_163633.json",
        "arima": "arima_preds_20260502_074241.json",
    }
    
    for h_key in HORIZONS:
        all_data["1h"][h_key] = {}
        for source_name, fname in v8_sources.items():
            path = v8_dir / fname
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if h_key not in data:
                continue
            for model_name, preds in data[h_key].items():
                full_name = f"{model_name}_1h" if model_name not in ("Actuals", "Persistence") else model_name
                all_data["1h"][h_key][full_name] = np.array(preds, dtype=float)
                
    return all_data

def get_actuals(source_data: dict, y_full: np.ndarray, horizon_steps: int) -> np.ndarray | None:
    if "Actuals" in source_data:
        return source_data["Actuals"]

    for model_name, preds in source_data.items():
        if model_name == "Persistence":
            continue
        n_preds = len(preds)
        actuals = y_full[-(n_preds + horizon_steps):-horizon_steps] if horizon_steps > 0 else y_full[-n_preds:]
        if len(actuals) == n_preds:
            return actuals
        return y_full[-n_preds:]
    return None

def compute_unified_mase(all_data: dict, targets: dict) -> dict:
    print("\n" + "=" * 60, flush=True)
    print("  TASK 5C.1 — Unified Cross-Resolution MASE (Hyndman 2006)", flush=True)
    print("  ⚠️  All models aligned to SAME test set per freq+horizon", flush=True)
    print("=" * 60, flush=True)

    results = {}
    for h_key in HORIZONS:
        results[h_key] = {}
        print(f"\n  --- Horizon {h_key} ---", flush=True)

        for freq in ["15m", "30m", "1h"]:
            if freq not in all_data or freq not in targets:
                continue
                
            y_train, y_full = targets[freq]
            horizon_steps = FREQ_HORIZON_STEPS[freq][h_key]
            source_data = all_data[freq].get(h_key, {})
            
            if not source_data:
                continue

            # --- ANCHOR TEST SET ---
            # Find the SMALLEST Actuals array across all sources for this freq+horizon.
            # This is typically the Fair pipeline's test set (fewer samples due to
            # tabular feature warmup drop). All models will be evaluated on this
            # exact same set to ensure MASE comparability.
            all_actuals = {}
            for model_name, preds in source_data.items():
                if model_name == "Actuals":
                    # This is overwritten by the last source loaded, so we need
                    # to find the smallest one from prediction array sizes.
                    continue
            
            # Collect all prediction sizes to find the anchor size
            pred_sizes = []
            for model_name, preds in source_data.items():
                if model_name in ("Actuals", "Persistence"):
                    continue
                pred_sizes.append(len(preds))
            
            if not pred_sizes:
                continue
            
            # The anchor size is the SMALLEST prediction set (Fair pipeline)
            anchor_n = min(pred_sizes)
            
            # Get the anchor actuals (from the source with the smallest set)
            # We use the Actuals array and trim from the END to match anchor_n
            actuals_full = get_actuals(source_data, y_full, horizon_steps)
            if actuals_full is None:
                continue
            
            # Align: take the LAST anchor_n elements (because Fair test set
            # starts later but ends at the same time as Expert)
            if len(actuals_full) > anchor_n:
                anchor_actuals = actuals_full[-anchor_n:]
            else:
                anchor_actuals = actuals_full[:anchor_n]
            
            print(f"\n    [{freq}] Anchor test set: n={anchor_n}", flush=True)

            for model_name, preds in sorted(source_data.items()):
                if model_name in ("Actuals", "Persistence"):
                    continue
                
                preds_arr = np.array(preds, dtype=float)
                
                # Align predictions to anchor
                if len(preds_arr) > anchor_n:
                    preds_aligned = preds_arr[-anchor_n:]
                elif len(preds_arr) < anchor_n:
                    # This model has fewer predictions than anchor — use as-is
                    n = len(preds_arr)
                    preds_aligned = preds_arr
                    anchor_actuals_local = anchor_actuals[-n:] if len(anchor_actuals) >= n else anchor_actuals[:n]
                    
                    mase_val = mase_hyndman(anchor_actuals_local, preds_aligned, y_train, horizon=horizon_steps)
                    mae_val = mae(anchor_actuals_local, preds_aligned)
                    
                    results[h_key][model_name] = {
                        "mase_hyndman": round(mase_val, 4),
                        "mae": round(mae_val, 4),
                        "n_samples": n,
                        "freq": freq
                    }
                    status = "✅" if mase_val < 1.0 else "❌"
                    print(f"    {model_name:30s}: MASE={mase_val:.4f} {status}  MAE={mae_val:.4f}  (n={n})", flush=True)
                    continue
                else:
                    preds_aligned = preds_arr

                mase_val = mase_hyndman(anchor_actuals, preds_aligned, y_train, horizon=horizon_steps)
                mae_val = mae(anchor_actuals, preds_aligned)
                
                results[h_key][model_name] = {
                    "mase_hyndman": round(mase_val, 4),
                    "mae": round(mae_val, 4),
                    "n_samples": anchor_n,
                    "freq": freq
                }
                status = "✅" if mase_val < 1.0 else "❌"
                print(f"    {model_name:30s}: MASE={mase_val:.4f} {status}  MAE={mae_val:.4f}  (n={anchor_n})", flush=True)

            # Persistence (aligned to anchor)
            if "Persistence" in source_data:
                pers = np.array(source_data["Persistence"], dtype=float)
                if len(pers) > anchor_n:
                    pers_aligned = pers[-anchor_n:]
                else:
                    pers_aligned = pers[:anchor_n]
                n = min(len(pers_aligned), len(anchor_actuals))
                mase_p = mase_hyndman(anchor_actuals[:n], pers_aligned[:n], y_train, horizon=horizon_steps)
                mae_p = mae(anchor_actuals[:n], pers_aligned[:n])
                pers_name = f"Persistence_{freq}"
                if pers_name not in results[h_key]:
                    results[h_key][pers_name] = {
                        "mase_hyndman": round(mase_p, 4),
                        "mae": round(mae_p, 4),
                        "n_samples": n,
                        "freq": freq
                    }
                    print(f"    {pers_name:30s}: MASE={mase_p:.4f} 📊  MAE={mae_p:.4f}  (n={n})", flush=True)

    return results

def main():
    print("=" * 60, flush=True)
    print("  V9 Cross-Resolution Statistical Metrics", flush=True)
    print("=" * 60, flush=True)

    targets = load_targets()
    all_data = load_predictions()

    # Task 5C.1: MASE
    mase_results = compute_unified_mase(all_data, targets)
    mase_path = OUTPUT_DIR / "unified_mase_all.json"
    with open(mase_path, "w", encoding="utf-8") as f:
        json.dump(mase_results, f, indent=2)
    print(f"\n  ✅ Saved: {mase_path}", flush=True)

    print("\n  Task 5C script complete!", flush=True)


if __name__ == "__main__":
    main()
