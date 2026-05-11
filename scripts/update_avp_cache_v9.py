import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "research" / "cache"
V9_DIR = PROJECT_ROOT / "research" / "experiments" / "v9_final"

from src.snapshot_adapter import load_all_normalized

def main():
    print("Updating AVP Cache with v9 Models (Full Resolution Align)...")
    snapshots = load_all_normalized()
    v9_data = snapshots.get("v9_multi_resolution", {})
    top_n = v9_data.get("top_n", {})

    v9_preds = {} # (horizon_int, full_model_name) -> preds
    v9_actuals = {} # (horizon_int, resolution) -> actuals

    all_preds_files = list(V9_DIR.glob("*_preds_*.json"))

    for fpath in all_preds_files:
        name = fpath.name
        if "15m" in name:
            res = "15m"
        elif "30m" in name:
            res = "30m"
        elif "1h" in name:
            res = "1h"
        else:
            continue
            
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
            for h_key, h_data in data.items():
                if not h_key.endswith("h"): continue
                h_int = int(h_key.replace("h", ""))
                
                for m_name, preds in h_data.items():
                    if m_name.lower() == "actuals":
                        v9_actuals[(h_int, res)] = preds
                        continue
                    
                    full_name = f"{m_name}_{res}"
                    v9_preds[(h_int, full_name)] = preds

    for h in [1, 6, 24]:
        cache_path = CACHE_DIR / f"avp_{h}h.json"
        if not cache_path.exists():
            continue
        
        with open(cache_path, encoding="utf-8") as f:
            cache_data = json.load(f)

        if "model_preds" not in cache_data:
            cache_data["model_preds"] = {}
        if "actuals_multi" not in cache_data:
            cache_data["actuals_multi"] = {}

        h_key = f"{h}h"
        top_models = top_n.get(h_key, [])
        
        # Save the actuals for the resolutions used
        for res in ["15m", "30m", "1h"]:
            if (h, res) in v9_actuals:
                cache_data["actuals_multi"][res] = v9_actuals[(h, res)]
        
        # Ensure 1h actuals is always available (from legacy cache)
        if not cache_data["actuals_multi"].get("1h") and cache_data.get("actuals"):
            cache_data["actuals_multi"]["1h"] = cache_data["actuals"]

        # Also save Persistence for each resolution if available
        for res in ["15m", "30m", "1h"]:
            persist_name = f"Persistence_{res}"
            if (h, persist_name) in v9_preds:
                cache_data["model_preds"][persist_name] = v9_preds[(h, persist_name)]
                print(f"Added {persist_name} (len {len(v9_preds[(h, persist_name)])}) to {h}h cache.")

        # Ensure Persistence_1h is always available (from legacy cache)
        if "Persistence_1h" not in cache_data["model_preds"] and cache_data.get("persistence"):
            cache_data["model_preds"]["Persistence_1h"] = cache_data["persistence"]

        for tm in top_models:
            model_name = tm["model"]
            if (h, model_name) in v9_preds:
                # Do NOT truncate! Keep full length so we can plot it against its own actuals!
                preds = v9_preds[(h, model_name)]
                cache_data["model_preds"][model_name] = preds
                print(f"Added {model_name} (len {len(preds)}) to {h}h cache.")
                
                # Add to metrics if not exists
                metrics_list = cache_data.get("metrics", [])
                existing = [m for m in metrics_list if m.get("Mô hình") == model_name]
                if not existing:
                    metrics_list.append({
                        "Mô hình": model_name,
                        "MAE": f"{tm['mae']:.2f}",
                        "MASE": f"{tm['mase']:.2f}"
                    })
            else:
                pass

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
        print(f"Saved {cache_path.name}")

if __name__ == "__main__":
    main()
