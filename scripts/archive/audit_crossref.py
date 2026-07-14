"""Cross-reference v6 vs v7 TFT + full improvement summary."""
import json
from pathlib import Path

# Load v6
with open("research/experiments/dashboard_runs/v6_pca_tft.json", encoding="utf-8") as f:
    v6 = json.load(f)
r6 = v6.get("results", v6.get("data", {}).get("results", {}))
print("=== v6 TFT ===")
for h in ["1h", "6h", "24h"]:
    tft = r6.get(h, {}).get("TFT", {})
    print(f"  {h}: {tft}")

# Load v7
with open("research/experiments/dashboard_runs/v7_cqr.json", encoding="utf-8") as f:
    v7 = json.load(f)
r7 = v7.get("results", {})
print("\n=== v7 TFT ===")
for h in ["1h", "6h", "24h"]:
    tft = r7.get(h, {}).get("TFT", {})
    print(f"  {h}: {tft}")

# Compare
v6_mae_1h = r6.get("1h", {}).get("TFT", {}).get("mae", 0)
v7_mae_1h = r7.get("1h", {}).get("TFT", {}).get("mae", 0)
if v6_mae_1h > 0 and v7_mae_1h > 0:
    pct = ((v7_mae_1h - v6_mae_1h) / v6_mae_1h) * 100
    print(f"\n  TFT 1h: v6={v6_mae_1h} -> v7={v7_mae_1h} = {pct:+.1f}%")

# Full v7 improvement table
print("\n=== v7 Improvement % (1-MASE)*100 ===")
for h in ["1h", "6h", "24h"]:
    h_data = r7.get(h, {})
    pers_mae = h_data.get("Persistence", {}).get("mae", 0)
    print(f"\n  {h} (Pers MAE={pers_mae:.3f}):")
    entries = []
    for m, d in h_data.items():
        if m == "Persistence":
            continue
        mase = d.get("mase_unified", d.get("mase", 0))
        if mase:
            imp = (1 - mase) * 100
            entries.append((m, d.get("mae", 0), mase, imp))
    entries.sort(key=lambda x: x[2])
    for name, mae, mase, imp in entries:
        marker = "BEATS" if mase < 1.0 else "LOSES"
        print(f"    {name:22s}: MAE={mae:.3f} MASE={mase:.4f} ({imp:+.1f}%) [{marker}]")

# Check Ensemble_GRU at 6h
eg6 = r7.get("6h", {}).get("Ensemble_GRU", {})
print(f"\n=== Ensemble_GRU 6h: {eg6} ===")
# Check Ensemble_Stack at 24h
es24 = r7.get("24h", {}).get("Ensemble_Stack", {})
print(f"=== Ensemble_Stack 24h: {es24} ===")
