"""Pre-compute Actual vs Predicted data — fully isolated version.

Each model type runs in its own subprocess to prevent LightGBM + torch segfault.
Step 1: Generate persistence baseline + LightGBM preds
Step 2: Add GRU preds (separate process)

Usage:
    uv run python scripts/precompute_avp_safe.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "research" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable


def _run_script(script_code: str, label: str, timeout: int = 120) -> bool:
    """Run inline Python script in isolated subprocess."""
    print(f"  [{label}] Starting...", flush=True)
    result = subprocess.run(
        [PYTHON, "-u", "-c", script_code],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        timeout=timeout,
    )
    if result.returncode == 0:
        print(f"  [{label}] ✅ Done", flush=True)
        return True
    else:
        print(f"  [{label}] ❌ Failed (exit={result.returncode})", flush=True)
        return False


LGBM_SCRIPT_TEMPLATE = '''
import gc, json, sys, time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("{project_root}")
sys.path.insert(0, str(PROJECT_ROOT))
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"
CACHE_DIR = PROJECT_ROOT / "research" / "cache"
horizon = {horizon}

from src.data.cleaner import _clip_physical_bounds, _handle_outliers, _remove_duplicates, _resample, _set_datetime_index
from src.data.imputer import impute_missing_data
from src.data.loader import load_raw_data, TARGET_COL

df_raw = load_raw_data()
df = _remove_duplicates(df_raw)
df = _set_datetime_index(df)
df, _ = _clip_physical_bounds(df)
df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
df = _resample(df, freq="1h")
df_hybrid = impute_missing_data(df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24, knn_neighbors=5, verbose=False)
del df_raw, df
gc.collect()

target = df_hybrid[TARGET_COL].values
is_imputed = df_hybrid["is_imputed"].values
n = len(target)
val_end = int(n * 0.9)

test_actuals, test_persist = [], []
for i in range(val_end, n - horizon):
    if is_imputed[i + horizon]:
        continue
    test_actuals.append(target[i + horizon])
    test_persist.append(target[i])

test_actuals = np.array(test_actuals)
test_persist = np.array(test_persist)
persist_mae = float(np.mean(np.abs(test_actuals - test_persist)))

result = {{
    "horizon": horizon, "n_test": len(test_actuals),
    "actuals": test_actuals.tolist(), "persistence": test_persist.tolist(),
    "gru_preds": None, "lgbm_preds": None,
    "metrics": [{{"Mô hình": "Persistence", "MAE": f"{{persist_mae:.2f}}", "MASE": "1.00"}}],
}}
print(f"  h={{horizon}}: {{len(test_actuals)}} test samples, Persistence MAE={{persist_mae:.2f}}", flush=True)

lgbm_path = EXPORT_DIR / f"lgbm_{{horizon}}h.txt"
if lgbm_path.exists():
    import lightgbm as lgb
    from src.features.builder import build_features
    booster = lgb.Booster(model_file=str(lgbm_path))
    df_feat = build_features(df_hybrid)
    feat_names_path = EXPORT_DIR / f"lgbm_{{horizon}}h_features.json"
    if feat_names_path.exists():
        with open(feat_names_path, encoding="utf-8") as f:
            feat_info = json.load(f)
        feat_cols = [c for c in feat_info["features"] if c in df_feat.columns]
    else:
        feat_cols = [c for c in df_feat.columns if c not in [TARGET_COL, "is_imputed"]]
    X_all = df_feat[feat_cols].values
    lgbm_preds = []
    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        if i < len(X_all):
            lgbm_preds.append(float(booster.predict(X_all[i:i+1])[0]))
        else:
            lgbm_preds.append(None)
    result["lgbm_preds"] = lgbm_preds
    lgbm_arr = np.array([p for p in lgbm_preds if p is not None], dtype=float)
    if len(lgbm_arr) > 0:
        lgbm_mae = float(np.mean(np.abs(test_actuals[:len(lgbm_arr)] - lgbm_arr)))
        result["metrics"].append({{"Mô hình": "LightGBM", "MAE": f"{{lgbm_mae:.2f}}", "MASE": f"{{lgbm_mae/persist_mae:.2f}}"}})
        print(f"  LightGBM MAE={{lgbm_mae:.2f}}, MASE={{lgbm_mae/persist_mae:.3f}}", flush=True)

out_path = CACHE_DIR / f"avp_{{horizon}}h.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f)
print(f"  Saved {{out_path.name}} ({{out_path.stat().st_size/1024:.0f}} KB)", flush=True)
'''

GRU_SCRIPT_TEMPLATE = '''
import gc, json, sys, time
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("{project_root}")
sys.path.insert(0, str(PROJECT_ROOT))
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"
CACHE_DIR = PROJECT_ROOT / "research" / "cache"
horizon = {horizon}

from src.data.cleaner import _clip_physical_bounds, _handle_outliers, _remove_duplicates, _resample, _set_datetime_index
from src.data.imputer import impute_missing_data
from src.data.loader import load_raw_data, TARGET_COL

df_raw = load_raw_data()
df = _remove_duplicates(df_raw)
df = _set_datetime_index(df)
df, _ = _clip_physical_bounds(df)
df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
df = _resample(df, freq="1h")
df_hybrid = impute_missing_data(df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24, knn_neighbors=5, verbose=False)
del df_raw, df
gc.collect()

target = df_hybrid[TARGET_COL].values
is_imputed = df_hybrid["is_imputed"].values
n = len(target)
val_end = int(n * 0.9)

# Load existing cache
cache_path = CACHE_DIR / f"avp_{{horizon}}h.json"
with open(cache_path, encoding="utf-8") as f:
    result = json.load(f)

persist_mae = float(result["metrics"][0]["MAE"])

gru_path = EXPORT_DIR / f"gru_{{horizon}}h.pt"
scaler_path = EXPORT_DIR / f"scalers_{{horizon}}h.json"
if gru_path.exists() and scaler_path.exists():
    import torch
    model_gru = torch.jit.load(str(gru_path), map_location="cpu")
    model_gru.eval()

    with open(scaler_path, encoding="utf-8") as f:
        sc = json.load(f)

    feat_cols_dl = sc["features"]
    available = [c for c in feat_cols_dl if c in df_hybrid.columns]
    features = df_hybrid[available].values
    feat_scaler = StandardScaler()
    train_end = int(n * 0.8)
    feat_scaler.fit(features[:train_end])
    features_scaled = feat_scaler.transform(features)
    lb = sc.get("lookback", 72)

    valid_indices, valid_windows = [], []
    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        if i < lb:
            continue
        window = features_scaled[i - lb + 1 : i + 1]
        if len(window) < lb:
            continue
        valid_indices.append(i)
        valid_windows.append(window)

    gru_preds_dict = {{}}
    for cs in range(0, len(valid_windows), 128):
        ce = min(cs + 128, len(valid_windows))
        batch = np.array(valid_windows[cs:ce])
        x = torch.FloatTensor(batch)
        with torch.no_grad():
            ps = model_gru(x).numpy().flatten()
        for j, pred_s in enumerate(ps):
            idx = valid_indices[cs + j]
            gru_preds_dict[idx] = float(pred_s) * sc["target_scaler_scale"] + sc["target_scaler_mean"]

    gru_preds = []
    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        gru_preds.append(gru_preds_dict.get(i))

    result["gru_preds"] = gru_preds
    gru_arr = np.array([p for p in gru_preds if p is not None], dtype=float)
    test_actuals = np.array(result["actuals"])
    if len(gru_arr) > 0:
        gru_mae = float(np.mean(np.abs(test_actuals[:len(gru_arr)] - gru_arr)))
        result["metrics"].append({{"Mô hình": "GRU", "MAE": f"{{gru_mae:.2f}}", "MASE": f"{{gru_mae/persist_mae:.2f}}"}})
        print(f"  GRU MAE={{gru_mae:.2f}}, MASE={{gru_mae/persist_mae:.3f}}", flush=True)

with open(cache_path, "w", encoding="utf-8") as f:
    json.dump(result, f)
print(f"  Updated {{cache_path.name}}", flush=True)
'''


def main():
    t_start = time.time()
    print("=" * 60)
    print("  Pre-compute Actual vs Predicted (fully isolated)")
    print("=" * 60)

    for h in [1, 6, 24]:
        print(f"\n--- Horizon {h}h ---")

        # Step 1: Persistence + LightGBM (no torch)
        lgbm_script = LGBM_SCRIPT_TEMPLATE.format(project_root=PROJECT_ROOT, horizon=h)
        ok1 = _run_script(lgbm_script, f"h={h} LightGBM", timeout=120)
        if not ok1:
            print(f"  ⚠️ LightGBM failed for h={h}, skipping GRU")
            continue

        # Step 2: GRU (torch only, no lightgbm)
        gru_script = GRU_SCRIPT_TEMPLATE.format(project_root=PROJECT_ROOT, horizon=h)
        _run_script(gru_script, f"h={h} GRU", timeout=120)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done in {total:.1f}s | Cache: {CACHE_DIR}")

    # Verify
    for h in [1, 6, 24]:
        p = CACHE_DIR / f"avp_{h}h.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            models = [m["Mô hình"] for m in d["metrics"]]
            print(f"  h={h}: {', '.join(models)} ({p.stat().st_size/1024:.0f} KB)")
        else:
            print(f"  h={h}: ❌ Not found")
    print("=" * 60)


if __name__ == "__main__":
    main()
