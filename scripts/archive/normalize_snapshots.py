import json
import glob
from pathlib import Path

runs_dir = Path("research/experiments/dashboard_runs")

for p in runs_dir.glob("*.json"):
    with open(p, "r") as f:
        data = json.load(f)
    
    modified = False
    
    # 1. Ensure models_included exists
    if "models_included" not in data:
        if "results" in data:
            data["models_included"] = list(data["results"].keys())
        elif "data" in data and isinstance(data["data"], dict):
            # Try to infer from data if it contains model names
            if "h1" in data["data"] and isinstance(data["data"]["h1"], dict):
                data["models_included"] = list(data["data"]["h1"].keys())
            else:
                data["models_included"] = []
        else:
            data["models_included"] = []
        modified = True

    # 2. Ensure parent_version exists
    if "parent_version" not in data:
        v = data.get("version", "")
        if v == "v1_baseline":
            data["parent_version"] = "—"
        elif v == "v2_enhanced":
            data["parent_version"] = "v1_baseline"
        else:
            data["parent_version"] = "unknown"
        modified = True
        
    # 3. Ensure changes block exists and is complete
    if "changes" not in data or not isinstance(data["changes"], dict):
        data["changes"] = {
            "what": data.get("description", ""),
            "why": "—",
            "result": "—",
            "conclusion": ""
        }
        modified = True
    else:
        for k in ["what", "why", "result", "conclusion"]:
            if k not in data["changes"]:
                data["changes"][k] = "" if k == "conclusion" else "—"
                modified = True

    if modified:
        with open(p, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Normalized {p.name}")

print("Done normalizing.")
