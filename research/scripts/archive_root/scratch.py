import json
import numpy as np

for horizon in ['1h', '6h', '24h']:
    print(f"\n--- Horizon: {horizon} ---")
    try:
        with open(f'research/cache/avp_{horizon}.json') as f:
            data = json.load(f)
            
        actual = np.array(data['actuals'])
        persist = np.array(data['persistence'])
        
        if 'TFT' in data['model_preds']:
            tft = np.array(data['model_preds']['TFT'])
        else:
            print("TFT not in model_preds")
            continue
            
        valid_idx = [i for i in range(len(actual)) if actual[i] is not None and persist[i] is not None and tft[i] is not None]
        actual_v = actual[valid_idx]
        persist_v = persist[valid_idx]
        tft_v = tft[valid_idx]
        
        mae_persist = np.abs(actual_v - persist_v).mean()
        mae_tft = np.abs(actual_v - tft_v).mean()
        
        print(f"Calculated MAE Persist: {mae_persist:.4f}")
        print(f"Calculated MAE TFT:     {mae_tft:.4f}")
    except Exception as e:
        print(f"Error: {e}")
