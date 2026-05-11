"""Post-fix verification: validate JSON and re-audit content."""
import json
import re
from pathlib import Path

# 1. Validate JSON
try:
    with open("research/experiments/dashboard_content.json", encoding="utf-8") as f:
        content = json.load(f)
    print("JSON valid ✅")
except Exception as e:
    print(f"JSON INVALID ❌: {e}")
    exit(1)

# 2. Re-audit
print("\n=== POST-FIX AUDIT ===")

# Check limitations[0]
lim0 = content["versions"]["v7_cqr"]["overview"]["limitations"][0]
if "ngoài TFT" in lim0 or "TFT vẫn thua" in lim0:
    print(f"RED BUG STILL EXISTS: {lim0}")
elif "Tất cả" in lim0 and "GRU" in lim0:
    print(f"✅ limitations[0] FIXED: {lim0[:80]}")
else:
    print(f"⚠️ limitations[0] unclear: {lim0[:80]}")

# Check multi_horizon_findings
findings = content["global"]["info_cards"]["multi_horizon_findings"]
if "TFT" in findings and "0.987" in findings:
    print("RED BUG STILL EXISTS in multi_horizon_findings")
elif "GRU" in findings and "1.009" in findings and "0.649" in findings:
    print("✅ multi_horizon_findings FIXED")
else:
    print(f"⚠️ multi_horizon_findings needs manual check")

# Check overview_improvements
improvements = content["global"]["info_cards"]["overview_improvements"]
if "Ensemble_GRU" in improvements and "0.750" in improvements:
    print("RED BUG STILL EXISTS in overview_improvements")
elif "GRU champion" in improvements and "0.649" in improvements:
    print("✅ overview_improvements FIXED")
else:
    print(f"⚠️ overview_improvements needs manual check")

# Check experiment_runs_guide placeholder
guide = content["global"]["info_cards"]["experiment_runs_guide"]
if "{len(snapshots)}" in guide:
    print("RED BUG STILL EXISTS in experiment_runs_guide")
elif "{len}" in guide:
    print("✅ experiment_runs_guide placeholder FIXED")
else:
    print(f"⚠️ experiment_runs_guide needs manual check")

# Check for any remaining hardcoded MASE that's wrong
print("\n=== Remaining hardcoded metrics check ===")
info_cards = content["global"]["info_cards"]
for key, val in info_cards.items():
    # Check for the OLD wrong MASE values
    for bad in ["0.987", "0.745", "0.676", "0.812", "0.750", "0.696"]:
        if bad in val and key not in ["multi_horizon_references"]:
            print(f"  ⚠️ {key} still contains old MASE {bad}")

print("\n=== AUDIT COMPLETE ===")
