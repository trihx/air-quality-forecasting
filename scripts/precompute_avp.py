"""Pre-compute Actual vs Predicted data for all horizons.

Saves results as JSON files to research/cache/ so the Streamlit dashboard
can render charts instantly without running heavy inference.

PERFORMANCE FIX: The main crash cause in the dashboard was running
the full data pipeline + model inference on the Streamlit event loop.
This script runs once (offline) and caches the results.

Usage:
    uv run python scripts/precompute_avp.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPORT_DIR = PROJECT_ROOT / "models" / "exported"
CACHE_DIR = PROJECT_ROOT / "research" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _detect_device() -> str:
    """Detect best available PyTorch device."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def precompute_horizon(horizon: int, df_hybrid, device_name: str) -> dict:
    """Pre-compute actual vs predicted for a single horizon."""
    from sklearn.preprocessing import StandardScaler
    from src.data.loader import TARGET_COL

    print(f"\n{'=' * 60}")
    print(f"  Horizon: {horizon}h | Device: {device_name}")
    print(f"{'=' * 60}")

    target = df_hybrid[TARGET_COL].values
    is_imputed = df_hybrid["is_imputed"].values
    n = len(target)
    val_end = int(n * 0.9)

    # Actual test values (only real data, skip imputed)
    test_actuals = []
    test_persist = []
    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        test_actuals.append(target[i + horizon])
        test_persist.append(target[i])

    test_actuals = np.array(test_actuals)
    test_persist = np.array(test_persist)

    print(f"  Test samples: {len(test_actuals)} (real data only)")

    result = {
        "horizon": horizon,
        "n_test": len(test_actuals),
        "actuals": test_actuals.tolist(),
        "persistence": test_persist.tolist(),
        "gru_preds": None,
        "lgbm_preds": None,
        "metrics": [],
    }

    persist_mae = float(np.mean(np.abs(test_actuals - test_persist)))
    result["metrics"].append(
        {
            "Mô hình": "Persistence",
            "MAE": f"{persist_mae:.2f}",
            "MASE": "1.00",
        }
    )
    print(f"  Persistence MAE: {persist_mae:.2f}")

    # ── LightGBM Predictions (MUST import BEFORE torch to avoid segfault on MPS) ──
    lgbm_path = EXPORT_DIR / f"lgbm_{horizon}h.txt"
    if lgbm_path.exists():
        t0 = time.time()
        print("  Loading LightGBM model...")

        import lightgbm as lgb
        from src.data.loader import TARGET_COL
        from src.features.builder import build_features

        booster = lgb.Booster(model_file=str(lgbm_path))
        df_feat = build_features(df_hybrid)
        feat_names_path = EXPORT_DIR / f"lgbm_{horizon}h_features.json"
        if feat_names_path.exists():
            with open(feat_names_path) as f:
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
                pred = booster.predict(X_all[i : i + 1])[0]
                lgbm_preds.append(float(pred))
            else:
                lgbm_preds.append(None)

        result["lgbm_preds"] = lgbm_preds

        lgbm_arr = np.array([p for p in lgbm_preds if p is not None], dtype=float)
        if len(lgbm_arr) > 0:
            lgbm_mae = float(np.mean(np.abs(test_actuals[: len(lgbm_arr)] - lgbm_arr)))
            result["metrics"].append(
                {
                    "Mô hình": "LightGBM",
                    "MAE": f"{lgbm_mae:.2f}",
                    "MASE": f"{lgbm_mae / persist_mae:.2f}",
                }
            )
            print(f"  LightGBM MAE: {lgbm_mae:.2f} | MASE: {lgbm_mae / persist_mae:.3f}")

        elapsed = time.time() - t0
        print(f"  LightGBM done in {elapsed:.1f}s")

        del booster, df_feat, X_all
        gc.collect()
    else:
        print(f"  LightGBM model not found at {lgbm_path}")

    # ── GRU Predictions (import torch AFTER LightGBM) ──
    gru_path = EXPORT_DIR / f"gru_{horizon}h.pt"
    scaler_path = EXPORT_DIR / f"scalers_{horizon}h.json"
    if gru_path.exists() and scaler_path.exists():
        t0 = time.time()
        print("  Loading GRU model...")

        import torch

        # Use CPU for TorchScript stability (MPS can segfault with jit)
        model_gru = torch.jit.load(str(gru_path), map_location="cpu")
        model_gru.eval()

        with open(scaler_path) as f:
            sc = json.load(f)

        feat_cols_dl = sc["features"]
        available = [c for c in feat_cols_dl if c in df_hybrid.columns]
        features = df_hybrid[available].values

        feat_scaler = StandardScaler()
        train_end = int(n * 0.8)
        feat_scaler.fit(features[:train_end])
        features_scaled = feat_scaler.transform(features)

        lb = sc.get("lookback", 72)

        # Batch inference for efficiency
        chunk_size = 128
        valid_indices = []
        valid_windows = []

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

        print(f"  GRU: {len(valid_windows)} valid windows, batch_size={chunk_size}")
        gru_preds_dict = {}
        for chunk_start in range(0, len(valid_windows), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(valid_windows))
            batch = np.array(valid_windows[chunk_start:chunk_end])
            x = torch.FloatTensor(batch)
            with torch.no_grad():
                preds_scaled = model_gru(x).numpy().flatten()
            for j, pred_s in enumerate(preds_scaled):
                idx = valid_indices[chunk_start + j]
                pred = float(pred_s) * sc["target_scaler_scale"] + sc["target_scaler_mean"]
                gru_preds_dict[idx] = pred

            if (chunk_start // chunk_size) % 5 == 0:
                progress = min(100, int(chunk_start / max(1, len(valid_windows)) * 100))
                print(f"    GRU progress: {progress}% ({chunk_start}/{len(valid_windows)})", flush=True)

        # Re-align with test indices
        gru_preds = []
        for i in range(val_end, n - horizon):
            if is_imputed[i + horizon]:
                continue
            gru_preds.append(gru_preds_dict.get(i))

        result["gru_preds"] = gru_preds

        gru_arr = np.array([p for p in gru_preds if p is not None], dtype=float)
        if len(gru_arr) > 0:
            gru_mae = float(np.mean(np.abs(test_actuals[: len(gru_arr)] - gru_arr)))
            result["metrics"].append(
                {
                    "Mô hình": "GRU (CPU batch)",
                    "MAE": f"{gru_mae:.2f}",
                    "MASE": f"{gru_mae / persist_mae:.2f}",
                }
            )
            print(f"  GRU MAE: {gru_mae:.2f} | MASE: {gru_mae / persist_mae:.3f}")

        elapsed = time.time() - t0
        print(f"  GRU done in {elapsed:.1f}s")

        del model_gru, features_scaled
        gc.collect()
    else:
        print(f"  GRU model not found at {gru_path}")

    return result


def main():
    t_start = time.time()
    print("=" * 60)
    print("  Pre-compute Actual vs Predicted Data")
    print("  Saves to research/cache/avp_*.json")
    print("=" * 60)

    device = _detect_device()
    print(f"\nDevice: {device}")

    # Load data pipeline
    print("\nStep 1/2: Loading data pipeline...")
    from src.data.cleaner import (
        _clip_physical_bounds,
        _handle_outliers,
        _remove_duplicates,
        _resample,
        _set_datetime_index,
    )
    from src.data.imputer import impute_missing_data
    from src.data.loader import load_raw_data

    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")
    df_hybrid = impute_missing_data(
        df,
        strategy="hybrid",
        max_gap_interp=6,
        max_gap_ml=24,
        knn_neighbors=5,
        verbose=False,
    )
    del df_raw, df
    gc.collect()

    print(f"  Pipeline loaded: {len(df_hybrid)} rows")

    # Pre-compute for each horizon
    print("\nStep 2/2: Computing predictions...")
    for h in [1, 6, 24]:
        result = precompute_horizon(h, df_hybrid, device)

        out_path = CACHE_DIR / f"avp_{h}h.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"  ✅ Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  All done in {total:.1f}s")
    print(f"  Cache files in: {CACHE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
