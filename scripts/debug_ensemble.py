"""Debug script — find exact crash point in ensemble pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("STEP 1: imports...", flush=True)
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
print("  OK", flush=True)

print("STEP 2: load data...", flush=True)
from src.data.cleaner import _clip_physical_bounds, _handle_outliers, _remove_duplicates, _resample, _set_datetime_index
from src.data.imputer import impute_missing_data
from src.data.loader import TARGET_COL, load_raw_data
from src.features.builder import build_features

df_raw = load_raw_data()
df = _remove_duplicates(df_raw)
df = _set_datetime_index(df)
df, _ = _clip_physical_bounds(df)
df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
df = _resample(df, freq="1h")
df_hybrid = impute_missing_data(df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24, knn_neighbors=5, verbose=False)
print(f"  Data: {len(df_hybrid)} rows", flush=True)

print("STEP 3: features...", flush=True)
is_imputed = df_hybrid["is_imputed"].values.copy()
df_features = build_features(df_hybrid.drop(columns=["is_imputed"], errors="ignore"), drop_na=True)
is_imp_feat = is_imputed[len(is_imputed) - len(df_features):]
print(f"  Features: {df_features.shape}, imp: {is_imp_feat.shape}", flush=True)

print("STEP 4: prepare LightGBM data...", flush=True)
df_lgbm = df_features.copy()
print("  4a: copy done", flush=True)
df_lgbm["_is_imputed"] = is_imp_feat[:len(df_lgbm)]
print("  4b: _is_imputed assigned", flush=True)
df_lgbm["target"] = df_lgbm[TARGET_COL].shift(-1)
print("  4c: target shifted", flush=True)
df_lgbm = df_lgbm.dropna(subset=["target"])
print(f"  4d: dropna done, {len(df_lgbm)} rows", flush=True)

feature_cols = [c for c in df_lgbm.columns if c not in [TARGET_COL, "target", "is_imputed", "_is_imputed"]]
X = df_lgbm[feature_cols].values
y = df_lgbm["target"].values
imp_aligned = df_lgbm["_is_imputed"].values
print(f"  4e: X={X.shape}, y={y.shape}, NaN_X={np.isnan(X).any()}, NaN_y={np.isnan(y).any()}", flush=True)

n = len(X)
tr_end = int(n * 0.8)
val_end = int(n * 0.9)
X_tr, y_tr = X[:tr_end], y[:tr_end]
X_te, y_te = X[val_end:], y[val_end:]
print(f"  4f: train={tr_end}, test={n-val_end}", flush=True)

print("STEP 5: LightGBM fit (n=300)...", flush=True)
lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=8,
                     num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                     min_child_samples=20, verbose=-1)
lgbm.fit(X_tr, y_tr)
print(f"  5a: fit done", flush=True)

pred = lgbm.predict(X_te)
print(f"  5b: predict done, shape={pred.shape}", flush=True)

real_mask = ~imp_aligned[val_end:]
y_real = y_te[real_mask]
pred_real = pred[real_mask]
persist = df_lgbm[TARGET_COL].values[val_end:val_end+len(X_te)]
persist_real = persist[real_mask]
mae = float(np.mean(np.abs(y_real - pred_real)))
p_mae = float(np.mean(np.abs(y_real - persist_real)))
mase = mae / p_mae if p_mae > 0 else float("inf")
print(f"  5c: MAE={mae:.3f}, MASE={mase:.3f}", flush=True)

print("STEP 6: PyTorch import...", flush=True)
import torch
print(f"  torch={torch.__version__}, MPS={torch.backends.mps.is_available()}", flush=True)

print("\n✅ ALL STEPS PASSED — no crash!", flush=True)
