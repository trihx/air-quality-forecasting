import json
from pathlib import Path

file_path = Path("research/experiments/dashboard_content.json")
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if "global" in data and "info_cards" in data["global"]:
    del data["global"]["info_cards"]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("info_cards removed.")
else:
    print("info_cards not found.")
