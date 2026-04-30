import json
import numpy as np
import pandas as pd

# Load snapshot
with open('research/experiments/dashboard_runs/v7_cqr_20260428.json') as f:
    snapshot = json.load(f)

snapshot_results = snapshot.get('results', snapshot) # handle potential nesting

mismatches = []
all_results = []

horizons = ['1h', '6h', '24h']
for h in horizons:
    print(f"\n{'='*40}")
    print(f"AUDITING HORIZON: {h}")
    print(f"{'='*40}")
    
    # Load cache
    try:
        with open(f'research/cache/avp_{h}.json') as f:
            cache = json.load(f)
    except Exception as e:
        print(f"Error loading cache for {h}: {e}")
        continue
        
    actual = np.array(cache.get('actuals', []))
    persist = np.array(cache.get('persistence', []))
    model_preds = cache.get('model_preds', {})
    
    if len(actual) == 0:
        print(f"No actuals found in cache for {h}")
        continue

    # Evaluate Persistence manually
    valid_idx_p = [i for i in range(len(actual)) if actual[i] is not None and persist[i] is not None]
    mae_persist_cache = np.abs(actual[valid_idx_p] - persist[valid_idx_p]).mean()
    
    snap_p = snapshot_results.get(h, {}).get('Persistence', {})
    mae_persist_snap = snap_p.get('mae', None)
    
    diff_p = abs(mae_persist_cache - mae_persist_snap) if mae_persist_snap is not None else None
    
    print(f"{'Persistence':<20} | Cache MAE: {mae_persist_cache:.4f} | Snap MAE: {mae_persist_snap} | Diff: {diff_p}")
    if diff_p is not None and diff_p > 0.01:
        mismatches.append((h, 'Persistence', mae_persist_cache, mae_persist_snap, diff_p))
        
    # Evaluate all other models
    for model_name, preds in model_preds.items():
        preds = np.array(preds)
        valid_idx = [i for i in range(len(actual)) if actual[i] is not None and preds[i] is not None]
        
        if len(valid_idx) == 0:
            print(f"{model_name:<20} | ALL NULL PREDICTIONS IN CACHE")
            continue
            
        mae_cache = np.abs(actual[valid_idx] - preds[valid_idx]).mean()
        
        snap_m = snapshot_results.get(h, {}).get(model_name, {})
        mae_snap = snap_m.get('mae', None)
        
        if mae_snap is None:
            print(f"{model_name:<20} | Cache MAE: {mae_cache:.4f} | Snap MAE: MISSING (Not in snapshot)")
            continue
            
        diff = abs(mae_cache - mae_snap)
        status = "❌ MISMATCH" if diff > 0.01 else "✅ OK"
        
        print(f"{model_name:<20} | Cache MAE: {mae_cache:.4f} | Snap MAE: {mae_snap:.4f} | Diff: {diff:.4f} | {status}")
        
        if diff > 0.01:
            mismatches.append((h, model_name, mae_cache, mae_snap, diff))

print(f"\n{'='*40}")
print(f"AUDIT SUMMARY")
print(f"{'='*40}")
print(f"Total Mismatches Found: {len(mismatches)}")
for m in mismatches:
    print(f"- [{m[0]}] {m[1]}: Cache={m[2]:.4f}, Snapshot={m[3]:.4f} (Diff: {m[4]:.4f})")
