"""Quick audit script to list all models ranked by MASE per horizon."""
import json

with open("research/experiments/standardized_metrics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]

for horizon in ["1h", "6h", "24h"]:
    if horizon not in results:
        continue
    models = list(results[horizon].keys())
    print(f"\n{'='*60}")
    print(f"  {horizon} ({len(models)} models)")
    print(f"{'='*60}")
    ranked = sorted(models, key=lambda m: results[horizon][m].get("mase_unified", 999))
    for i, m in enumerate(ranked):
        r = results[horizon][m]
        mase = r.get("mase_unified", "N/A")
        mae = r.get("mae", "N/A")
        freq = r.get("freq", "?")
        print(f"  {i+1:2d}. {m:40s} MASE={mase:<8} MAE={mae:<10} freq={freq}")
