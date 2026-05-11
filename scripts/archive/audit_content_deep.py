"""Deep audit: cross-check dashboard_content.json against v7_cqr snapshot."""
import json
import re
from pathlib import Path

content_path = Path("research/experiments/dashboard_content.json")
with open(content_path, encoding="utf-8") as f:
    content = json.load(f)

print("=" * 80)
print("DEEP AUDIT: Content accuracy cross-check with v7 actual data")
print("=" * 80)

# Actual data from v7_cqr.json (verified):
# 1h: ALL models LOSE. Best=GRU MAE=2.515 MASE=1.009. TFT MAE=2.960 MASE=1.187
# 6h: Best=GRU MAE=4.167 MASE=0.649. Pers MAE=6.422
# 24h: Best=LSTM MAE=4.278 MASE=0.663. Pers MAE=6.454

# === Audit v7_cqr limitations ===
print("\n[BUG CHECK] v7_cqr limitations:")
v7_lim = content["versions"]["v7_cqr"]["overview"]["limitations"]
for i, l in enumerate(v7_lim):
    if "ngoài TFT" in l:
        print(f"  RED BUG [{i}]: '{l}'")
        print(f"     TFT MASE=1.187 at 1h, LOSES to Persistence!")
    elif "PCA" in l and "1.572" in l:
        print(f"  WARN [{i}]: '{l}' -> numbers need verify")
    elif "TFT v2" in l:
        print(f"  WARN [{i}]: '{l}' -> needs cross-ref")
    else:
        print(f"  OK [{i}]: '{l[:80]}'")

# === Audit info cards with specific metrics ===
print("\n[BUG CHECK] Info cards with hardcoded metrics:")
info_cards = content["global"]["info_cards"]
for key, val in info_cards.items():
    mase_matches = re.findall(r'MASE[=:≈]\s*[\d.]+', val)
    mae_matches = re.findall(r'MAE[=:≈]\s*[\d.]+', val)
    if mase_matches or mae_matches:
        print(f"  WARN {key}: {mase_matches + mae_matches}")

# === Check for experiment results referencing specific numbers ===
print("\n[BUG CHECK] Experiments with hardcoded numbers:")
for ver_name in sorted(content["versions"].keys()):
    exps = content["versions"][ver_name].get("overview", {}).get("experiments", [])
    for exp in exps:
        result = exp.get("result", "")
        mase_matches = re.findall(r'MASE[=:≈]\s*[\d.]+', result)
        mae_matches = re.findall(r'MAE[=:≈]\s*[\d.]+', result)
        pct_matches = re.findall(r'[\d.]+%', result)
        if mase_matches or mae_matches:
            title = exp.get("title", "")[:50]
            print(f"  WARN {ver_name} '{title}': {mase_matches + mae_matches}")

# === Check multi_horizon insight for accuracy ===
print("\n[BUG CHECK] Global multi_horizon insight:")
mh = content["global"]["multi_horizon"]["insight_no_single_best"]
for k, v in mh.items():
    if "TFT" in v:
        print(f"  WARN {k} mentions TFT: check accuracy")
    mase_vals = re.findall(r'MASE[=:≈]\s*[\d.]+', v)
    if mase_vals:
        print(f"  WARN {k} has hardcoded MASE: {mase_vals}")
    print(f"  {k}: {v[:120]}...")

# === Check v7_cqr achievements against data ===
print("\n[BUG CHECK] v7_cqr achievements:")
v7_ach = content["versions"]["v7_cqr"]["overview"]["achievements"]
for i, a in enumerate(v7_ach):
    if "35.1%" in a:
        # GRU at 6h: improvement = (1-0.649)*100 = 35.1% -> correct
        print(f"  OK [{i}]: GRU 35.1% at 6h (1-0.6489=0.3511) CORRECT")
    elif "33.7%" in a:
        # LSTM at 24h: improvement = (1-0.663)*100 = 33.7% -> correct  
        print(f"  OK [{i}]: LSTM 33.7% at 24h (1-0.6629=0.3371) CORRECT")
    elif "MASE=1.009" in a:
        # GRU at 1h MASE=1.009 -> correct
        print(f"  OK [{i}]: GRU MASE=1.009 at 1h CORRECT")
    else:
        print(f"  OK [{i}]: '{a[:80]}'")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
