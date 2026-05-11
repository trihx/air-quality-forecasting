import json
from pathlib import Path

runs_dir = Path("research/experiments/dashboard_runs")
snapshots = {}
for jpath in sorted(runs_dir.glob("*.json")):
    try:
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("version", jpath.stem)
        snapshots[version] = data
    except (json.JSONDecodeError, KeyError):
        continue

print(list(snapshots.keys()))

v1 = snapshots[list(snapshots.keys())[0]]
v2 = snapshots[list(snapshots.keys())[-1]]

v1_features = v1.get("feature_set", {})
v2_features = v2.get("feature_set", {})

print(type(v1_features))
print(type(v2_features))
try:
    all_keys = sorted(set(list(v1_features.keys()) + list(v2_features.keys())))
    print("Success")
except Exception as e:
    print(f"Error: {e}")

