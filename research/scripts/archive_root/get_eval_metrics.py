import json
from pathlib import Path

def get_metrics():
    dir_path = Path("research/experiments/prediction_intervals")
    if not dir_path.exists():
        return {}
        
    json_files = list(dir_path.glob("prediction_intervals_*.json"))
    if not json_files:
        return {}
        
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    with open(latest_file, encoding="utf-8") as f:
        data = json.load(f)
        
    metrics = {}
    for row in data:
        model = row["model"]
        horizon = row["horizon"]
        # prefer conformal_prediction or quantile_regression
        method = row["method"]
        key = f"{model}_{horizon}"
        if key not in metrics or method == "conformal_prediction":
            metrics[key] = {
                "avg_width": row["avg_width"],
                "mae": row["mae"],
                "coverage": row.get("coverage", 0.9),
            }
    return metrics

print(get_metrics())
