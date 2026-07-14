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
import lightgbm as lgb  # Import early to prevent MPS segfault with torch

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
        "model_preds": {},  # model_name -> prediction array
        "metrics": [],
    }

    persist_mae = float(np.mean(np.abs(test_actuals - test_persist)))
    persist_rmse = float(np.sqrt(np.mean((test_actuals - test_persist) ** 2)))
    result["metrics"].append(
        {
            "Mô hình": "Persistence",
            "MAE": f"{persist_mae:.2f}",
            "RMSE": f"{persist_rmse:.2f}",
            "MASE": "1.00",
        }
    )
    print(f"  Persistence MAE: {persist_mae:.2f} | RMSE: {persist_rmse:.2f}")

    # ── LightGBM Predictions (MUST import BEFORE torch to avoid segfault on MPS) ──
    lgbm_path = EXPORT_DIR / f"lgbm_{horizon}h.txt"
    if lgbm_path.exists():
        t0 = time.time()
        print("  Loading LightGBM model...")
        from src.data.loader import TARGET_COL
        from src.features.builder import build_features

        booster = lgb.Booster(model_file=str(lgbm_path))
        df_feat = build_features(df_hybrid)
        feat_names_path = EXPORT_DIR / f"lgbm_{horizon}h_features.json"
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
                pred = booster.predict(X_all[i : i + 1])[0]
                lgbm_preds.append(float(pred))
            else:
                lgbm_preds.append(None)

        result["lgbm_preds"] = lgbm_preds
        result["model_preds"]["LightGBM"] = lgbm_preds

        lgbm_arr = np.array([p for p in lgbm_preds if p is not None], dtype=float)
        if len(lgbm_arr) > 0:
            lgbm_mae = float(np.mean(np.abs(test_actuals[: len(lgbm_arr)] - lgbm_arr)))
            lgbm_rmse = float(np.sqrt(np.mean((test_actuals[: len(lgbm_arr)] - lgbm_arr) ** 2)))
            result["metrics"].append(
                {
                    "Mô hình": "LightGBM",
                    "MAE": f"{lgbm_mae:.2f}",
                    "RMSE": f"{lgbm_rmse:.2f}",
                    "MASE": f"{lgbm_mae / persist_mae:.2f}",
                }
            )
            print(f"  LightGBM MAE: {lgbm_mae:.2f} | RMSE: {lgbm_rmse:.2f} | MASE: {lgbm_mae / persist_mae:.3f}")

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
        result["model_preds"]["GRU"] = gru_preds

        gru_arr = np.array([p for p in gru_preds if p is not None], dtype=float)
        if len(gru_arr) > 0:
            gru_mae = float(np.mean(np.abs(test_actuals[: len(gru_arr)] - gru_arr)))
            gru_rmse = float(np.sqrt(np.mean((test_actuals[: len(gru_arr)] - gru_arr) ** 2)))
            result["metrics"].append(
                {
                    "Mô hình": "GRU",
                    "MAE": f"{gru_mae:.2f}",
                    "RMSE": f"{gru_rmse:.2f}",
                    "MASE": f"{gru_mae / persist_mae:.2f}",
                }
            )
            print(f"  GRU MAE: {gru_mae:.2f} | RMSE: {gru_rmse:.2f} | MASE: {gru_mae / persist_mae:.3f}")

        elapsed = time.time() - t0
        print(f"  GRU done in {elapsed:.1f}s")

        del model_gru, features_scaled
        gc.collect()
    else:
        print(f"  GRU model not found at {gru_path}")

    # ── Ensemble Predictions (derived from GRU + LightGBM) ──
    _compute_ensemble_preds(result, horizon, persist_mae)

    # ── External model predictions (TFT, ARIMA, SARIMA) ──
    _load_external_preds(result, horizon, persist_mae)

    return result


def _compute_ensemble_preds(result: dict, horizon: int, persist_mae: float) -> None:
    """Compute Ensemble predictions from GRU + LightGBM using weights from ensemble JSON."""
    gru_preds = result.get("gru_preds")
    lgbm_preds = result.get("lgbm_preds")

    if not gru_preds or not lgbm_preds:
        print("  Skipping Ensemble: need both GRU and LightGBM predictions")
        return

    # Load ensemble weights
    ens_files = sorted((PROJECT_ROOT / "research" / "experiments" / "ensemble").glob("ensemble_*.json"), reverse=True)
    if not ens_files:
        print("  Skipping Ensemble: no ensemble JSON found")
        return

    with open(ens_files[0], encoding="utf-8") as f:
        ens_data = json.load(f)

    h_key = f"{horizon}h"
    if h_key not in ens_data:
        print(f"  Skipping Ensemble: no {h_key} data in ensemble JSON")
        return

    # Convert to numpy, handling None values
    gru_arr = np.array([p if p is not None else np.nan for p in gru_preds], dtype=float)
    lgbm_arr = np.array([p if p is not None else np.nan for p in lgbm_preds], dtype=float)
    n_common = min(len(gru_arr), len(lgbm_arr))
    gru_arr = gru_arr[:n_common]
    lgbm_arr = lgbm_arr[:n_common]
    valid = ~np.isnan(gru_arr) & ~np.isnan(lgbm_arr)

    if valid.sum() < 10:
        print(f"  Skipping Ensemble: only {valid.sum()} valid common points")
        return

    actuals_arr = np.array(result["actuals"][:n_common], dtype=float)

    # ── Ensemble_Stack (Ridge weights) ──
    stack_info = ens_data[h_key].get("Ensemble_Stack", {})
    weights_str = stack_info.get("weights", "")
    try:
        import re
        w_lgbm = float(re.search(r'LightGBM=([\-\d.]+)', weights_str).group(1))
        w_gru = float(re.search(r'GRU=([\-\d.]+)', weights_str).group(1))
        intercept = float(re.search(r'intercept=([\-\d.]+)', weights_str).group(1))

        stack_preds = np.where(valid, w_lgbm * lgbm_arr + w_gru * gru_arr + intercept, np.nan)
        result["model_preds"]["Ensemble_Stack"] = [float(p) if not np.isnan(p) else None for p in stack_preds]

        stack_valid = stack_preds[valid]
        stack_mae = float(np.mean(np.abs(actuals_arr[valid] - stack_valid)))
        stack_rmse = float(np.sqrt(np.mean((actuals_arr[valid] - stack_valid) ** 2)))
        result["metrics"].append({
            "Mô hình": "Ensemble_Stack",
            "MAE": f"{stack_mae:.2f}",
            "RMSE": f"{stack_rmse:.2f}",
            "MASE": f"{stack_mae / persist_mae:.2f}" if persist_mae > 0 else "N/A",
        })
        print(f"  Ensemble_Stack MAE: {stack_mae:.2f} | RMSE: {stack_rmse:.2f} | MASE: {stack_mae / persist_mae:.3f}")
    except (AttributeError, ValueError) as e:
        print(f"  Skipping Ensemble_Stack: failed to parse weights: {e}")

    # ── Ensemble_GRU (GRU standalone from ensemble = same as exported GRU)
    # In the snapshot, Ensemble_GRU = GRU from the ensemble script.
    # Since the exported GRU is from dl_multi_horizon (different training run),
    # we use the exported GRU predictions directly as Ensemble_GRU proxy.
    result["model_preds"]["Ensemble_GRU"] = result["gru_preds"]

    # Add metric for Ensemble_GRU so it appears in cache rankings
    gru_preds_clean = np.array([p for p in result["gru_preds"] if p is not None], dtype=float)
    if len(gru_preds_clean) > 0:
        n_valid = min(len(gru_preds_clean), valid.sum())
        ens_gru_mae = float(np.mean(np.abs(actuals_arr[valid][:n_valid] - gru_preds_clean[:n_valid])))
        ens_gru_rmse = float(np.sqrt(np.mean((actuals_arr[valid][:n_valid] - gru_preds_clean[:n_valid]) ** 2)))
        result["metrics"].append({
            "Mô hình": "Ensemble_GRU",
            "MAE": f"{ens_gru_mae:.2f}",
            "RMSE": f"{ens_gru_rmse:.2f}",
            "MASE": f"{ens_gru_mae / persist_mae:.2f}" if persist_mae > 0 else "N/A",
        })
        print(f"  Ensemble_GRU MAE: {ens_gru_mae:.2f} | RMSE: {ens_gru_rmse:.2f} | MASE: {ens_gru_mae / persist_mae:.3f}")
    else:
        print(f"  Ensemble_GRU: using GRU predictions as proxy (no metric)")

    # ── Ensemble_Weighted ──
    weighted_info = ens_data[h_key].get("Ensemble_Weighted", {})
    weight_str = weighted_info.get("best_weight", "")
    try:
        import re
        w = float(re.search(r'LightGBM=([\d.]+)', weight_str).group(1))
        weighted_preds = np.where(valid, w * lgbm_arr + (1 - w) * gru_arr, np.nan)
        result["model_preds"]["Ensemble_Weighted"] = [float(p) if not np.isnan(p) else None for p in weighted_preds]

        w_valid = weighted_preds[valid]
        w_mae = float(np.mean(np.abs(actuals_arr[valid] - w_valid)))
        w_rmse = float(np.sqrt(np.mean((actuals_arr[valid] - w_valid) ** 2)))
        result["metrics"].append({
            "Mô hình": "Ensemble_Weighted",
            "MAE": f"{w_mae:.2f}",
            "RMSE": f"{w_rmse:.2f}",
            "MASE": f"{w_mae / persist_mae:.2f}" if persist_mae > 0 else "N/A",
        })
        print(f"  Ensemble_Weighted MAE: {w_mae:.2f} | RMSE: {w_rmse:.2f} | MASE: {w_mae / persist_mae:.3f}")
    except (AttributeError, ValueError) as e:
        print(f"  Skipping Ensemble_Weighted: failed to parse weights: {e}")


def _load_external_preds(result: dict, horizon: int, persist_mae: float) -> None:
    """Load pre-exported predictions from TFT, ARIMA, SARIMA .json files."""
    # Model name -> file prefix mapping
    external_models = {
        "TFT": f"tft_preds_{horizon}h.json",
        "ARIMA": f"arima_preds_{horizon}h.json",
        "SARIMA": f"sarima_preds_{horizon}h.json",
    }

    actuals = np.array(result["actuals"], dtype=float)
    n_test = len(actuals)

    for model_name, filename in external_models.items():
        pred_path = CACHE_DIR / filename
        if not pred_path.exists():
            continue

        try:
            with open(pred_path, encoding="utf-8") as f:
                pred_data = json.load(f)

            predictions = pred_data.get("predictions", [])
            if not predictions:
                print(f"  {model_name}: empty predictions, skipping")
                continue

            # Trim/pad to match cache test size
            if len(predictions) > n_test:
                predictions = predictions[:n_test]
            elif len(predictions) < n_test:
                predictions = predictions + [None] * (n_test - len(predictions))

            result["model_preds"][model_name] = predictions

            # Compute metrics from aligned predictions
            valid_pairs = [
                (a, p) for a, p in zip(actuals, predictions)
                if p is not None and not np.isnan(a)
            ]

            if len(valid_pairs) > 0:
                act_arr = np.array([a for a, _ in valid_pairs])
                pred_arr = np.array([p for _, p in valid_pairs])
                mae = float(np.mean(np.abs(act_arr - pred_arr)))
                rmse = float(np.sqrt(np.mean((act_arr - pred_arr) ** 2)))
                mase = mae / persist_mae if persist_mae > 0 else float("inf")

                result["metrics"].append({
                    "Mô hình": model_name,
                    "MAE": f"{mae:.2f}",
                    "RMSE": f"{rmse:.2f}",
                    "MASE": f"{mase:.2f}",
                })
                n_valid = len(valid_pairs)
                print(
                    f"  {model_name} MAE: {mae:.2f} | RMSE: {rmse:.2f} | "
                    f"MASE: {mase:.3f} | n={n_valid}/{n_test}"
                )
            else:
                print(f"  {model_name}: no valid predictions")
        except Exception as e:
            print(f"  {model_name}: failed to load {filename}: {e}")


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
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f)
        print(f"  ✅ Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  All done in {total:.1f}s")
    print(f"  Cache files in: {CACHE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
