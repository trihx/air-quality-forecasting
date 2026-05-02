"""V9 Expert Results Audit — Validate MASE integrity.

Checks:
1. Test set consistency: Fair vs Expert same Actuals?
2. Persistence baseline: Is the denominator inflated?
3. Leakage red flags: MASE < 0.1 or implausible improvements
4. Overfitting signals: Train vs Test gap analysis
5. Sample count discrepancy: Why n_samples differ?
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

V9_DIR = PROJECT_ROOT / "research" / "experiments" / "v9_final"

# =====================================================================
# Helper: MASE from scratch (no library dependency for audit)
# =====================================================================
def mase_manual(y_true, y_pred, y_train, h=1):
    """Compute MASE manually for audit."""
    mae_model = np.mean(np.abs(y_true - y_pred))
    # Naive forecast MAE on training set
    naive_errors = np.abs(y_train[h:] - y_train[:-h])
    mae_naive = np.mean(naive_errors)
    if mae_naive == 0:
        return float('inf')
    return mae_model / mae_naive


def load_latest(pattern):
    files = sorted(V9_DIR.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def audit_test_set_consistency():
    """CHECK 1: Are Fair and Expert evaluated on the SAME test samples?"""
    print("=" * 70)
    print("  CHECK 1: Test Set Consistency (Fair vs Expert)")
    print("=" * 70)
    
    for freq in ["30m", "15m"]:
        fair_dl = load_latest(f"dl_preds_{freq}_*.json")
        expert_dl = load_latest(f"dl_expert_preds_{freq}_*.json")
        fair_tft = load_latest(f"tft_preds_{freq}_*.json")
        expert_tft = load_latest(f"tft_expert_preds_{freq}_*.json")
        
        print(f"\n  --- {freq} ---")
        
        for h_key in ["1h", "6h", "24h"]:
            print(f"\n  Horizon {h_key}:")
            
            # DL comparison
            if fair_dl and expert_dl and h_key in fair_dl and h_key in expert_dl:
                fair_actuals = np.array(fair_dl[h_key].get("Actuals", []))
                expert_actuals = np.array(expert_dl[h_key].get("Actuals", []))
                
                n_fair = len(fair_actuals)
                n_expert = len(expert_actuals)
                
                print(f"    DL Fair n={n_fair}, Expert n={n_expert}", end="")
                
                if n_fair == n_expert:
                    if np.allclose(fair_actuals, expert_actuals, rtol=1e-6):
                        print(" → ✅ SAME test set")
                    else:
                        print(" → ⚠️ SAME SIZE but DIFFERENT values!")
                        # Check overlap
                        overlap = 0
                        for v in fair_actuals:
                            if v in expert_actuals:
                                overlap += 1
                        print(f"       Overlap: {overlap}/{n_fair} ({100*overlap/max(n_fair,1):.0f}%)")
                else:
                    print(f" → ❌ DIFFERENT test set sizes! Δ={n_expert - n_fair}")
                    # Check if one is a subset
                    if n_fair < n_expert:
                        # Check if fair actuals appear in expert actuals
                        match_count = 0
                        for i, v in enumerate(fair_actuals):
                            if v in expert_actuals:
                                match_count += 1
                        print(f"       Fair values found in Expert: {match_count}/{n_fair}")
                    
                    # Compare statistics
                    print(f"       Fair  Actuals: mean={fair_actuals.mean():.2f}, std={fair_actuals.std():.2f}, "
                          f"min={fair_actuals.min():.2f}, max={fair_actuals.max():.2f}")
                    print(f"       Expert Actuals: mean={expert_actuals.mean():.2f}, std={expert_actuals.std():.2f}, "
                          f"min={expert_actuals.min():.2f}, max={expert_actuals.max():.2f}")
            
            # TFT comparison
            if fair_tft and expert_tft and h_key in fair_tft and h_key in expert_tft:
                fair_a = np.array(fair_tft[h_key].get("Actuals", []))
                expert_a = np.array(expert_tft[h_key].get("Actuals", []))
                n_f, n_e = len(fair_a), len(expert_a)
                print(f"    TFT Fair n={n_f}, Expert n={n_e}", end="")
                if n_f == n_e and np.allclose(fair_a, expert_a, rtol=1e-6):
                    print(" → ✅ SAME")
                elif n_f != n_e:
                    print(f" → ❌ DIFFERENT sizes! Δ={n_e - n_f}")
                else:
                    print(" → ⚠️ DIFFERENT values")


def audit_persistence_baseline():
    """CHECK 2: Is the Persistence baseline correct and consistent?"""
    print("\n" + "=" * 70)
    print("  CHECK 2: Persistence Baseline Audit")
    print("=" * 70)
    
    for freq in ["30m", "15m"]:
        fair_dl = load_latest(f"dl_preds_{freq}_*.json")
        expert_dl = load_latest(f"dl_expert_preds_{freq}_*.json")
        
        print(f"\n  --- {freq} ---")
        for h_key in ["1h", "6h", "24h"]:
            if not fair_dl or not expert_dl:
                continue
            if h_key not in fair_dl or h_key not in expert_dl:
                continue
                
            fair_pers = np.array(fair_dl[h_key].get("Persistence", []))
            expert_pers = np.array(expert_dl[h_key].get("Persistence", []))
            fair_act = np.array(fair_dl[h_key].get("Actuals", []))
            expert_act = np.array(expert_dl[h_key].get("Actuals", []))
            
            # Compute MAE of persistence for each
            if len(fair_pers) > 0 and len(fair_act) > 0:
                n = min(len(fair_pers), len(fair_act))
                fair_pers_mae = np.mean(np.abs(fair_act[:n] - fair_pers[:n]))
            else:
                fair_pers_mae = float('nan')
                
            if len(expert_pers) > 0 and len(expert_act) > 0:
                n = min(len(expert_pers), len(expert_act))
                expert_pers_mae = np.mean(np.abs(expert_act[:n] - expert_pers[:n]))
            else:
                expert_pers_mae = float('nan')
            
            print(f"    {h_key}: Fair Pers MAE={fair_pers_mae:.4f} (n={len(fair_pers)}), "
                  f"Expert Pers MAE={expert_pers_mae:.4f} (n={len(expert_pers)})")
            
            if abs(fair_pers_mae - expert_pers_mae) > 0.5:
                print(f"         ⚠️ Persistence MAE differs by {abs(fair_pers_mae - expert_pers_mae):.4f}!")
                print(f"         → MASE denominator is DIFFERENT → scores NOT directly comparable!")


def audit_mase_recompute():
    """CHECK 3: Recompute MASE from scratch using raw predictions."""
    print("\n" + "=" * 70)
    print("  CHECK 3: MASE Recomputation from Scratch")
    print("=" * 70)
    
    import pandas as pd
    
    STEPS = {"15m": {"1h": 4, "6h": 24, "24h": 96},
             "30m": {"1h": 2, "6h": 12, "24h": 48}}
    
    for freq in ["30m", "15m"]:
        # Load training target for naive baseline
        data_file = f"marts_features_{freq}.csv"
        base_file = f"marts_features_{freq}_base.csv"
        
        df_fair = pd.read_csv(PROJECT_ROOT / "dataset" / "processed" / data_file, 
                              index_col=0, parse_dates=True)
        df_base = pd.read_csv(PROJECT_ROOT / "dataset" / "processed" / base_file,
                              index_col=0, parse_dates=True)
        
        # y_train for Fair and Expert (80% split)
        y_fair_train = df_fair["pm25"].values[:int(len(df_fair) * 0.8)]
        y_base_train = df_base["pm25"].values[:int(len(df_base) * 0.8)]
        
        expert_dl = load_latest(f"dl_expert_preds_{freq}_*.json")
        fair_dl = load_latest(f"dl_preds_{freq}_*.json")
        
        print(f"\n  --- {freq} ---")
        print(f"  Fair train samples: {len(y_fair_train):,}")
        print(f"  Expert (base) train samples: {len(y_base_train):,}")
        
        for h_key in ["1h", "6h", "24h"]:
            h_steps = STEPS[freq][h_key]
            
            print(f"\n    Horizon {h_key} (h={h_steps} steps):")
            
            # --- Fair Pipeline ---
            if fair_dl and h_key in fair_dl:
                actuals_f = np.array(fair_dl[h_key].get("Actuals", []))
                for model_name in ["GRU_v9", "LSTM_v9"]:
                    preds = fair_dl[h_key].get(model_name, [])
                    if not preds:
                        continue
                    preds = np.array(preds)
                    n = min(len(preds), len(actuals_f))
                    
                    # MASE with Fair y_train
                    mase_fair_train = mase_manual(actuals_f[:n], preds[:n], y_fair_train, h=h_steps)
                    # MASE with Base y_train (WRONG but let's check)
                    mase_base_train = mase_manual(actuals_f[:n], preds[:n], y_base_train, h=h_steps)
                    
                    mae_val = np.mean(np.abs(actuals_f[:n] - preds[:n]))
                    
                    print(f"      {model_name} (Fair):")
                    print(f"        MAE={mae_val:.4f}")
                    print(f"        MASE (fair y_train): {mase_fair_train:.4f}")
                    print(f"        MASE (base y_train): {mase_base_train:.4f}")
            
            # --- Expert Pipeline ---
            if expert_dl and h_key in expert_dl:
                actuals_e = np.array(expert_dl[h_key].get("Actuals", []))
                for model_name in ["GRU_v9_expert", "LSTM_v9_expert"]:
                    preds = expert_dl[h_key].get(model_name, [])
                    if not preds:
                        continue
                    preds = np.array(preds)
                    n = min(len(preds), len(actuals_e))
                    
                    # MASE with Base y_train (correct for Expert)
                    mase_base_train = mase_manual(actuals_e[:n], preds[:n], y_base_train, h=h_steps)
                    # MASE with Fair y_train (check sensitivity)
                    mase_fair_train = mase_manual(actuals_e[:n], preds[:n], y_fair_train, h=h_steps)
                    
                    mae_val = np.mean(np.abs(actuals_e[:n] - preds[:n]))
                    
                    print(f"      {model_name} (Expert):")
                    print(f"        MAE={mae_val:.4f}")
                    print(f"        MASE (base y_train): {mase_base_train:.4f}")
                    print(f"        MASE (fair y_train): {mase_fair_train:.4f}")
                    
                    # RED FLAG check
                    if mase_base_train < 0.1:
                        print(f"        🚨 RED FLAG: MASE < 0.1 → LEAKAGE SUSPECTED!")
                    if mase_base_train < 0.3:
                        print(f"        ⚠️ WARNING: MASE unusually low — verify no leakage")


def audit_overfitting_check():
    """CHECK 4: Compare MAE values — does Expert MAE actually improve?"""
    print("\n" + "=" * 70)
    print("  CHECK 4: MAE Sanity Check (Resolution-Independent)")
    print("=" * 70)
    print("  If MASE improved but MAE didn't, the MASE denominator is suspect.")
    
    for freq in ["30m", "15m"]:
        fair_dl = load_latest(f"dl_preds_{freq}_*.json")
        expert_dl = load_latest(f"dl_expert_preds_{freq}_*.json")
        
        print(f"\n  --- {freq} ---")
        for h_key in ["1h", "6h", "24h"]:
            if not fair_dl or not expert_dl:
                continue
            if h_key not in fair_dl or h_key not in expert_dl:
                continue
                
            actuals_f = np.array(fair_dl[h_key].get("Actuals", []))
            actuals_e = np.array(expert_dl[h_key].get("Actuals", []))
            
            print(f"\n    {h_key}:")
            
            # For each model pair
            for fair_name, expert_name in [("GRU_v9", "GRU_v9_expert"), ("LSTM_v9", "LSTM_v9_expert")]:
                preds_f = fair_dl[h_key].get(fair_name, [])
                preds_e = expert_dl[h_key].get(expert_name, [])
                
                if not preds_f or not preds_e:
                    continue
                    
                preds_f = np.array(preds_f)
                preds_e = np.array(preds_e)
                
                mae_f = np.mean(np.abs(actuals_f[:len(preds_f)] - preds_f))
                mae_e = np.mean(np.abs(actuals_e[:len(preds_e)] - preds_e))
                
                delta = mae_e - mae_f
                pct = (delta / mae_f) * 100 if mae_f > 0 else 0
                
                status = "✅ Lower (better)" if delta < 0 else "⚠️ Higher (worse!)"
                print(f"      {fair_name:15s}: Fair MAE={mae_f:.4f} vs Expert MAE={mae_e:.4f} "
                      f"(Δ={delta:+.4f}, {pct:+.1f}%) {status}")


def audit_statistical_methods_mase():
    """CHECK 5: What y_train did v9_statistical_methods.py use?"""
    print("\n" + "=" * 70)
    print("  CHECK 5: v9_statistical_methods.py — MASE y_train Source Audit")
    print("=" * 70)
    print("  The script uses marts_features_{freq}.csv (FAIR dataset) for ALL models.")
    print("  This means Expert models have MASE computed with FAIR y_train.")
    print("  → If base y_train has HIGHER naive MAE, Expert MASE would be LOWER.")
    print("  → This is a CONFOUND, not a real improvement!")
    
    import pandas as pd
    
    for freq in ["15m", "30m"]:
        STEPS = {"15m": {"1h": 4, "6h": 24, "24h": 96},
                 "30m": {"1h": 2, "6h": 12, "24h": 48}}
        
        df_fair = pd.read_csv(PROJECT_ROOT / "dataset" / "processed" / f"marts_features_{freq}.csv",
                              index_col=0, parse_dates=True)
        df_base = pd.read_csv(PROJECT_ROOT / "dataset" / "processed" / f"marts_features_{freq}_base.csv",
                              index_col=0, parse_dates=True)
        
        y_fair = df_fair["pm25"].values[:int(len(df_fair) * 0.8)]
        y_base = df_base["pm25"].values[:int(len(df_base) * 0.8)]
        
        print(f"\n  --- {freq} ---")
        print(f"  Fair y_train: {len(y_fair):,} samples")
        print(f"  Base y_train: {len(y_base):,} samples")
        
        for h_key in ["1h", "6h", "24h"]:
            h = STEPS[freq][h_key]
            
            naive_fair = np.mean(np.abs(y_fair[h:] - y_fair[:-h]))
            naive_base = np.mean(np.abs(y_base[h:] - y_base[:-h]))
            
            pct_diff = ((naive_base - naive_fair) / naive_fair) * 100
            
            print(f"    {h_key} (h={h}): Naive MAE fair={naive_fair:.4f}, base={naive_base:.4f} "
                  f"(Δ={pct_diff:+.1f}%)")
            
            if abs(pct_diff) > 5:
                print(f"         ⚠️ >5% difference in naive baseline!")
                print(f"         → Expert MASE denominator would be {'LARGER' if pct_diff > 0 else 'SMALLER'}")
                print(f"         → Expert MASE would appear {'BETTER' if pct_diff > 0 else 'WORSE'} "
                      f"even with SAME model MAE!")


def main():
    print("=" * 70)
    print("  v9 EXPERT RESULTS AUDIT — Overfitting & Leakage Check")
    print(f"  Rule: MASE < 0.1 = leakage. Too-good = verify.")
    print("=" * 70)
    
    audit_test_set_consistency()
    audit_persistence_baseline()
    audit_mase_recompute()
    audit_overfitting_check()
    audit_statistical_methods_mase()
    
    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
