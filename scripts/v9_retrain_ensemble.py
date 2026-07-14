"""V9 Ensemble Model Builder.

Combines the best Deep Learning model (LSTM_v9) and best Machine Learning model (LightGBM_v9)
into a Weighted Average Ensemble for 15m and 30m frequencies.
"""
import json
import sys
import glob
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "v9_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1h", "6h", "24h"]

def _load_latest(pattern: str) -> dict:
    files = sorted(glob.glob(str(OUTPUT_DIR / pattern)))
    if not files:
        return {}
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)

def build_ensemble(freq: str):
    print(f"\n=======================================================")
    print(f"  Building Ensemble for {freq}")
    print(f"=======================================================")
    
    dl_data = _load_latest(f"dl_preds_{freq}_*.json")
    lgbm_data = _load_latest(f"lgbm_preds_{freq}_*.json")
    
    if not dl_data or not lgbm_data:
        print(f"  [ERROR] Missing DL or LGBM predictions for {freq}")
        return
        
    ensemble_preds = {}
    
    for h_key in HORIZONS:
        if h_key not in dl_data or h_key not in lgbm_data:
            continue
            
        # Extract best models (LSTM for DL, LightGBM for ML)
        lstm_preds = np.array(dl_data[h_key].get("LSTM_v9", []), dtype=float)
        lgbm_preds = np.array(lgbm_data[h_key].get("LightGBM_v9", []), dtype=float)
        
        # Get actuals to align
        actuals = np.array(dl_data[h_key].get("Actuals", []), dtype=float)
        
        if len(lstm_preds) == 0 or len(lgbm_preds) == 0:
            print(f"  [SKIP] Missing predictions for {h_key}")
            continue
            
        # Align predictions to the same test set size (usually LSTM's size is smaller/equal)
        n_samples = min(len(lstm_preds), len(lgbm_preds))
        
        # Take the tail (latest) to align temporally
        lstm_aligned = lstm_preds[-n_samples:]
        lgbm_aligned = lgbm_preds[-n_samples:]
        
        # Weighted Average: DL usually better at long-term, ML better at short-term
        # For simplicity and robustness, we use Simple Average (50/50)
        # However, can be weighted 0.6 DL / 0.4 ML if needed.
        ensemble = (0.5 * lstm_aligned) + (0.5 * lgbm_aligned)
        
        # Save results
        ensemble_preds[h_key] = {
            "Ensemble_Weighted_v9": ensemble.tolist(),
            "Actuals": actuals[-n_samples:].tolist() if len(actuals) >= n_samples else actuals.tolist()
        }
        
        print(f"  [{h_key}] Ensemble created. n={n_samples}")
        
    # Save ensemble predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"ensemble_preds_{freq}_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ensemble_preds, f)
        
    print(f"  ✅ Saved: {out_file.name}")

if __name__ == "__main__":
    build_ensemble("15m")
    build_ensemble("30m")
