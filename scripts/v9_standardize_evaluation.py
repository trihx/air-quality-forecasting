"""V9 Standardize Evaluation — Resolve MASE confound.

This script ensures that the MASE comparison between the Fair and Expert
pipelines is scientifically valid by strictly evaluating both models on
the exact same test set indices.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V9_DIR = PROJECT_ROOT / "research" / "experiments" / "v9_final"

def mase_manual(y_true, y_pred, naive_mae):
    """Compute MASE using a pre-calculated naive MAE denominator."""
    if naive_mae == 0:
        return float('inf')
    mae_model = np.mean(np.abs(y_true - y_pred))
    return mae_model / naive_mae

def load_latest(pattern):
    files = sorted(V9_DIR.glob(pattern))
    if not files:
        return None, None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f), files[-1]

def main():
    print("===============================================================")
    print("  PHASE 6: STANDARDIZE EVALUATION (FAIR VS EXPERT)")
    print("===============================================================")

    STEPS = {"15m": {"1h": 4, "6h": 24, "24h": 96},
             "30m": {"1h": 2, "6h": 12, "24h": 48}}

    for freq in ["30m", "15m"]:
        # 1. Load Fair Dataset to compute Naive Baseline (The Source of Truth)
        data_file = f"marts_features_{freq}.csv"
        df_fair = pd.read_csv(PROJECT_ROOT / "dataset" / "processed" / data_file, 
                              index_col=0, parse_dates=True)
        
        y_train_fair = df_fair["pm25"].values[:int(len(df_fair) * 0.8)]
        
        fair_dl, fair_path = load_latest(f"dl_preds_{freq}_*.json")
        expert_dl, expert_path = load_latest(f"dl_expert_preds_{freq}_*.json")
        
        if not fair_dl or not expert_dl:
            print(f"Skipping {freq} - missing predictions")
            continue
            
        print(f"\n--- {freq} Standardization ---")
        
        for h_key in ["1h", "6h", "24h"]:
            h_steps = STEPS[freq][h_key]
            
            # Naive MAE on FAIR y_train (This is the anchor for MASE)
            naive_mae_train = np.mean(np.abs(y_train_fair[h_steps:] - y_train_fair[:-h_steps]))
            
            if h_key not in fair_dl or h_key not in expert_dl:
                continue
                
            # Both predictions must have timestamps to align!
            # Since our DL JSON currently doesn't store dates, we rely on the fact 
            # that Fair test is a SUBSET of Expert test (from the end).
            # The Fair pipeline drops early sequence rows, so the Fair test set 
            # starts LATER than the Expert test set, but they both end at the same time.
            
            fair_actuals = np.array(fair_dl[h_key].get("Actuals", []))
            expert_actuals = np.array(expert_dl[h_key].get("Actuals", []))
            
            n_fair = len(fair_actuals)
            n_expert = len(expert_actuals)
            
            if n_expert < n_fair:
                print(f"  [{h_key}] Error: Expert n < Fair n. Cannot align.")
                continue
                
            # Align from the END
            expert_actuals_aligned = expert_actuals[-n_fair:]
            
            # Sanity Check Alignment
            if not np.allclose(fair_actuals, expert_actuals_aligned, rtol=1e-5):
                print(f"  [{h_key}] ❌ ALIGNMENT FAILED! Actuals do not match.")
                # We need to find the actual offset instead of assuming it's at the end
                match_idx = -1
                for i in range(len(expert_actuals) - n_fair + 1):
                    if np.allclose(fair_actuals[:10], expert_actuals[i:i+10], rtol=1e-5):
                        match_idx = i
                        break
                
                if match_idx != -1:
                     expert_actuals_aligned = expert_actuals[match_idx:match_idx+n_fair]
                     print(f"  [{h_key}] ✅ Alignment found at offset {match_idx}")
                else:
                    print(f"  [{h_key}] ❌ Alignment completely failed. Skip.")
                    continue
            else:
                 print(f"  [{h_key}] ✅ Aligned correctly from the end. Offset = {n_expert - n_fair}")
                 match_idx = n_expert - n_fair
                 
            # Compute Aligned Metrics
            print(f"    Denominator (Naive MAE from Fair): {naive_mae_train:.4f}")
            
            # Fair models
            for model_name in ["GRU_v9", "LSTM_v9"]:
                preds = fair_dl[h_key].get(model_name, [])
                if not preds: continue
                preds = np.array(preds)[:n_fair]
                mae = np.mean(np.abs(fair_actuals - preds))
                mase = mase_manual(fair_actuals, preds, naive_mae_train)
                print(f"    Fair   {model_name:10s} | MAE: {mae:.4f} | MASE: {mase:.4f}")

            # Expert models
            for model_name in ["GRU_v9_expert", "LSTM_v9_expert"]:
                preds = expert_dl[h_key].get(model_name, [])
                if not preds: continue
                preds_aligned = np.array(preds)[match_idx:match_idx+n_fair]
                mae = np.mean(np.abs(expert_actuals_aligned - preds_aligned))
                mase = mase_manual(expert_actuals_aligned, preds_aligned, naive_mae_train)
                print(f"    Expert {model_name:10s} | MAE: {mae:.4f} | MASE: {mase:.4f}")

if __name__ == "__main__":
    main()
