import json

with open("research/experiments/dashboard_runs/v7_cqr_20260428.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)

old_results = old_data.get("data", {}).get("results", {})

# Load new data
with open("research/experiments/v7_retrain/sklearn_20260428_161819.json", "r", encoding="utf-8") as f:
    ml_data = json.load(f)
with open("research/experiments/v7_retrain/dl_20260428_161819.json", "r", encoding="utf-8") as f:
    dl_data = json.load(f)
with open("research/experiments/ensemble/ensemble_20260428_170601.json", "r", encoding="utf-8") as f:
    ensemble_data = json.load(f)

models_to_check = ["Persistence", "LightGBM_tuned", "RandomForest", "GradientBoosting", "LSTM", "GRU", "Ensemble_Stack"]
horizons = ["1h", "6h", "24h"]

print(f"{'Model':<20} | {'Horizon':<10} | {'Old MAE':<10} | {'New MAE':<10} | {'Diff':<10}")
print("-" * 65)

for model in models_to_check:
    for h in horizons:
        old_mae = old_results.get(h, {}).get(model, {}).get("mae")
        
        # Get new mae
        new_mae = None
        if model in ml_data.get(f"{h}", {}):
            new_mae = ml_data[f"{h}"][model].get("mae")
        elif model in dl_data.get(f"{h}", {}):
            new_mae = dl_data[f"{h}"][model].get("mae")
        elif model in ensemble_data.get(f"{h}", {}):
            new_mae = ensemble_data[f"{h}"][model].get("mae")
        elif model == "GRU_v2_log" and "GRU_v2_log" in dl_data.get(f"{h}", {}):
             new_mae = dl_data[f"{h}"]["GRU_v2_log"].get("mae") # Handle names
        elif model == "LSTM_v2_log" and "LSTM_v2_log" in dl_data.get(f"{h}", {}):
             new_mae = dl_data[f"{h}"]["LSTM_v2_log"].get("mae") # Handle names
        
        if model == "GRU" and "GRU_v2_log" in dl_data.get(f"{h}", {}):
             new_mae = dl_data[f"{h}"]["GRU_v2_log"].get("mae")
        if model == "LSTM" and "LSTM_v2_log" in dl_data.get(f"{h}", {}):
             new_mae = dl_data[f"{h}"]["LSTM_v2_log"].get("mae")

        if old_mae is not None and new_mae is not None:
            diff = new_mae - old_mae
            print(f"{model:<20} | {h:<10} | {old_mae:<10.4f} | {new_mae:<10.4f} | {diff:<10.4f}")
        elif new_mae is not None:
            print(f"{model:<20} | {h:<10} | {'N/A':<10} | {new_mae:<10.4f} | {'N/A':<10}")
