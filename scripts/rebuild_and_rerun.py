"""Rebuild marts data and re-run ML pipeline after leakage fix.

Usage:
    uv run python scripts/rebuild_and_rerun.py
"""

import sys
from datetime import datetime
from pathlib import Path

print("=" * 60, flush=True)
print("🔧 REBUILD PIPELINE — Post Leakage Fix", flush=True)
print(f"   Time: {datetime.now()}", flush=True)
print("=" * 60, flush=True)

# Step 1: Load cleaned data
print("\n[1/5] Loading cleaned hourly data...", flush=True)

import pandas as pd

cleaned_path = Path("dataset/interim/cleaned_hourly.csv")
if not cleaned_path.exists():
    print(f"❌ Not found: {cleaned_path}", flush=True)
    sys.exit(1)

df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
print(f"  ✅ Loaded: {len(df):,} rows × {len(df.columns)} cols", flush=True)

# Step 2: Rebuild features (with leakage fixes)
print("\n[2/5] Rebuilding features (anti-leakage)...", flush=True)

from src.features.builder import build_features, save_marts_data

df_features = build_features(df)
print(f"  ✅ Features: {len(df_features):,} rows × {len(df_features.columns)} cols", flush=True)

# Step 3: Save new marts data
print("\n[3/5] Saving new marts data...", flush=True)

marts_path = save_marts_data(df_features, output_path="dataset/processed/marts_features.csv")
print(f"  ✅ Saved: {marts_path}", flush=True)

# Step 4: Quick leakage verification
print("\n[4/5] Quick leakage verification...", flush=True)

import numpy as np
from src.data.loader import TARGET_COL

target = df_features[TARGET_COL]
leakage_found = False

# Check diff features
if "pm25_diff_1h" in df_features.columns and "pm25_lag_1h" in df_features.columns:
    d = df_features["pm25_diff_1h"]
    l = df_features["pm25_lag_1h"]
    recon = d + l
    valid = recon.notna()
    if valid.any() and np.allclose(recon[valid], target[valid], rtol=1e-10):
        print("  🔴 LEAKAGE STILL PRESENT: diff_1h + lag_1h = target!", flush=True)
        leakage_found = True
    else:
        print("  ✅ diff_1h: no longer reconstructs target", flush=True)

# Check domain features correlation
for col in ["co2_pm25_ratio", "pm25_aqi_cat"]:
    if col in df_features.columns:
        corr = abs(float(df_features[col].corr(target)))
        status = "🔴 HIGH" if corr > 0.95 else "✅ OK"
        print(f"  {status} {col}: corr={corr:.4f}", flush=True)
        if corr > 0.95:
            leakage_found = True

# Check all features
print("\n  Top correlated features with target:", flush=True)
feat_cols = [c for c in df_features.columns if c != TARGET_COL]
numeric_feats = df_features[feat_cols].select_dtypes(include=[np.number])
corrs = numeric_feats.corrwith(target).abs().sort_values(ascending=False)
for col, val in corrs.head(10).items():
    flag = " ⚠️" if val > 0.95 else ""
    print(f"    {col:40s} corr={val:.4f}{flag}", flush=True)

if leakage_found:
    print("\n  ⚠️ SOME LEAKAGE MAY STILL EXIST — review features above", flush=True)
else:
    print("\n  ✅ No obvious leakage detected", flush=True)

# Step 5: Re-run ML models
print("\n[5/5] Re-running ML models...", flush=True)

from src.models.run_ml import main as run_ml_main

run_ml_main()

print("\n" + "=" * 60, flush=True)
print("✅ REBUILD & RE-RUN COMPLETE", flush=True)
print("=" * 60, flush=True)
