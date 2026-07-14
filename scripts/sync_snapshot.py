"""Sync dashboard snapshot from standardized_metrics.json."""
import json

std_path = "research/experiments/standardized_metrics.json"
snap_path = "research/experiments/dashboard_runs/v9_multi_resolution.json"

std = json.load(open(std_path, encoding="utf-8"))
snap = json.load(open(snap_path, encoding="utf-8"))

# Overwrite metrics block in snapshot with standardized_metrics results
snap["metrics"] = std["results"]

json.dump(snap, open(snap_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# Verify
snap2 = json.load(open(snap_path, encoding="utf-8"))
for h in ["1h", "6h", "24h"]:
    models = snap2["metrics"][h]
    has_rmse = sum(1 for m in models.values() if m.get("rmse") is not None)
    has_r2 = sum(1 for m in models.values() if m.get("r2") is not None)
    has_da = sum(1 for m in models.values() if m.get("da") is not None)
    print(f"{h}: {len(models)} models | RMSE: {has_rmse} | R2: {has_r2} | DA: {has_da}")

print("Done! Dashboard snapshot synced with standardized_metrics.json")
