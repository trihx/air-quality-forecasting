"""Standardize Persistence Baseline & Build AVP Cache (v8_final).

Reads individual model family files from v8_final/ and ensemble/ directories,
re-calculates MASE against a unified Persistence baseline, and produces
avp_*.json cache files for the Streamlit dashboard.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from src.data.cleaner import (
    _clip_physical_bounds,
    _handle_outliers,
    _remove_duplicates,
    _resample,
    _set_datetime_index,
)
from src.data.imputer import impute_missing_data
from src.data.loader import TARGET_COL, load_raw_data

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HORIZONS = [1, 6, 24]
EXPERIMENT_DIR = PROJECT_ROOT / "research" / "experiments"
CACHE_DIR = PROJECT_ROOT / "research" / "cache"


def _load_latest(prefix: str, dir_path: Path) -> dict:
    """Load the latest JSON file matching prefix_TIMESTAMP.json in dir_path.
    
    Only matches files where the character after 'prefix_' is a digit,
    avoiding e.g. lightgbm_preds_*.json when prefix='lightgbm'.
    """
    files = sorted(dir_path.glob(f"{prefix}_*.json"))
    # Only keep files where the suffix after prefix_ starts with a digit (timestamp)
    exact = []
    for f in files:
        suffix = f.stem[len(prefix) + 1:]  # part after "prefix_"
        if suffix and suffix[0].isdigit():
            exact.append(f)
    if not exact:
        print(f"  ⚠️ Not found: {prefix}_TIMESTAMP.json in {dir_path}", flush=True)
        return {}
    with open(exact[-1], encoding="utf-8") as f:
        return json.load(f)


def _prepare_hybrid_data() -> pd.DataFrame:
    """Run the full data pipeline to get hybrid imputed data with is_imputed column."""
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")

    return impute_missing_data(
        df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24, knn_neighbors=5, verbose=False
    )


def main() -> None:
    print("=" * 70, flush=True)
    print("STANDARDIZE BASELINE & BUILD AVP CACHE", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Compute unified Persistence baseline ──
    df_hybrid = _prepare_hybrid_data()
    is_imputed = df_hybrid["is_imputed"].values
    target = df_hybrid[TARGET_COL].values
    n = len(df_hybrid)
    val_end = int(n * 0.9)

    unified_persist = {}
    avp_cache_data = {1: {}, 6: {}, 24: {}}

    for h in HORIZONS:
        unified_true, unified_persist_vals = [], []
        for i in range(val_end, n - h):
            if is_imputed[i + h]:
                continue
            actual = target[i + h]
            persist = target[i]
            if not np.isnan(actual) and not np.isnan(persist):
                unified_true.append(actual)
                unified_persist_vals.append(persist)

        unified_mae = float(np.mean(np.abs(np.array(unified_true) - np.array(unified_persist_vals))))
        unified_rmse = float(np.sqrt(np.mean((np.array(unified_true) - np.array(unified_persist_vals)) ** 2)))

        unified_persist[h] = {
            "mae": round(unified_mae, 4),
            "rmse": round(unified_rmse, 4),
            "n_samples": len(unified_true),
        }
        avp_cache_data[h] = {
            "horizon": h,
            "n_test": len(unified_true),
            "actuals": unified_true,
            "persistence": unified_persist_vals,
            "model_preds": {},
            "metrics": [{"Mô hình": "Persistence", "MAE": f"{unified_mae:.4f}", "RMSE": f"{unified_rmse:.4f}", "MASE": "1.0000"}],
        }

    print(f"\n  Unified Persistence baselines:", flush=True)
    for h in HORIZONS:
        p = unified_persist[h]
        print(f"    {h}h: MAE={p['mae']:.4f}, RMSE={p['rmse']:.4f}, n={p['n_samples']}", flush=True)

    # ── Step 2: Load all individual model family files ──
    v8_dir = EXPERIMENT_DIR / "v8_final"

    lgbm_metrics = _load_latest("lightgbm", v8_dir)
    lgbm_preds = _load_latest("lightgbm_preds", v8_dir)
    sklearn_metrics = _load_latest("sklearn", v8_dir)
    sklearn_preds = _load_latest("sklearn_preds", v8_dir)
    arima_metrics = _load_latest("arima", v8_dir)
    arima_preds = _load_latest("arima_preds", v8_dir)
    dl_metrics = _load_latest("dl", v8_dir)
    dl_preds = _load_latest("dl_preds", v8_dir)
    tft_multi = _load_latest("tft_multi_horizon", v8_dir)
    tft_preds = _load_latest("tft_preds", v8_dir)
    ensemble = _load_latest("ensemble", EXPERIMENT_DIR / "ensemble")
    ensemble_preds = _load_latest("ensemble_preds", EXPERIMENT_DIR / "ensemble")

    print(f"\n  Sources loaded:")
    print(f"    LightGBM: {bool(lgbm_metrics)}, Sklearn: {bool(sklearn_metrics)}, ARIMA: {bool(arima_metrics)}")
    print(f"    DL: {bool(dl_metrics)}, TFT: {bool(tft_multi)}, Ensemble: {bool(ensemble)}")

    # ── Step 3: Build standardized results + AVP cache ──
    standardized = {}

    for h in HORIZONS:
        h_key = f"{h}h"
        persist_mae = unified_persist[h]["mae"]

        h_results = {
            "Persistence": {
                "mae": persist_mae,
                "rmse": unified_persist[h]["rmse"],
                "mase": 1.0,
                "n_test": unified_persist[h]["n_samples"],
                "source": "unified",
            }
        }

        h_model_preds = {}

        # ── LightGBM ──
        if h_key in lgbm_metrics:
            orig = lgbm_metrics[h_key].get("LightGBM_tuned", {})
            if orig:
                new_mase = round(orig["mae"] / persist_mae, 4)
                h_results["LightGBM_tuned"] = {
                    "mae": orig["mae"],
                    "rmse": orig.get("rmse"),
                    "mase_unified": new_mase,
                    "source": "v8_final",
                }
                h_results["LightGBM"] = h_results["LightGBM_tuned"]
                avp_cache_data[h]["metrics"].append(
                    {"Mô hình": "LightGBM", "MAE": f"{orig['mae']:.4f}", "RMSE": f"{orig.get('rmse', 0):.4f}", "MASE": f"{new_mase:.4f}"}
                )

            if lgbm_preds and h_key in lgbm_preds:
                h_model_preds["LightGBM_tuned"] = lgbm_preds[h_key].get("LightGBM_tuned")
                h_model_preds["LightGBM"] = h_model_preds["LightGBM_tuned"]

        # ── Sklearn ──
        if h_key in sklearn_metrics:
            for m_key in ["RandomForest", "GradientBoosting", "Stacking", "Ensemble_Weighted"]:
                if m_key in sklearn_metrics[h_key]:
                    orig = sklearn_metrics[h_key][m_key]
                    new_mase = round(orig["mae"] / persist_mae, 4)
                    h_results[m_key] = {
                        "mae": orig["mae"],
                        "rmse": orig.get("rmse"),
                        "mase_unified": new_mase,
                        "source": "v8_final",
                    }
                    avp_cache_data[h]["metrics"].append(
                        {"Mô hình": m_key, "MAE": f"{orig['mae']:.4f}", "RMSE": f"{orig.get('rmse', 0):.4f}", "MASE": f"{new_mase:.4f}"}
                    )

            if sklearn_preds and h_key in sklearn_preds:
                for m_key in ["RandomForest", "GradientBoosting", "Stacking", "Ensemble_Weighted"]:
                    if m_key in sklearn_preds[h_key]:
                        h_model_preds[m_key] = sklearn_preds[h_key][m_key]

        # ── ARIMA ──
        if h_key in arima_metrics:
            for m_key in ["ARIMA", "SARIMA"]:
                if m_key in arima_metrics[h_key]:
                    orig = arima_metrics[h_key][m_key]
                    new_mase = round(orig["mae"] / persist_mae, 4)
                    h_results[m_key] = {
                        "mae": orig["mae"],
                        "rmse": orig.get("rmse"),
                        "mase_unified": new_mase,
                        "source": "v8_final",
                    }
                    avp_cache_data[h]["metrics"].append(
                        {"Mô hình": m_key, "MAE": f"{orig['mae']:.4f}", "RMSE": f"{orig.get('rmse', 0):.4f}", "MASE": f"{new_mase:.4f}"}
                    )

            if arima_preds and h_key in arima_preds:
                for m_key in ["ARIMA", "SARIMA"]:
                    if m_key in arima_preds[h_key]:
                        h_model_preds[m_key] = arima_preds[h_key][m_key]

        # ── DL (GRU, LSTM) ──
        if h_key in dl_metrics:
            dl_map = {"GRU_v2_log": "GRU", "LSTM_v2_log": "LSTM"}
            for k, m_key in dl_map.items():
                if k in dl_metrics[h_key]:
                    orig = dl_metrics[h_key][k]
                    new_mase = round(orig["mae"] / persist_mae, 4)
                    h_results[m_key] = {
                        "mae": orig["mae"],
                        "rmse": orig.get("rmse"),
                        "mase_unified": new_mase,
                        "source": "v8_final",
                    }
                    avp_cache_data[h]["metrics"].append(
                        {"Mô hình": m_key, "MAE": f"{orig['mae']:.4f}", "RMSE": f"{orig.get('rmse', 0):.4f}", "MASE": f"{new_mase:.4f}"}
                    )

            if dl_preds and h_key in dl_preds:
                for k, m_key in dl_map.items():
                    if k in dl_preds[h_key]:
                        h_model_preds[m_key] = dl_preds[h_key][k]

        # ── TFT ──
        if tft_multi and h_key in tft_multi and "TFT" in tft_multi[h_key]:
            orig = tft_multi[h_key]["TFT"]
            new_mase = round(orig["mae"] / persist_mae, 4)
            h_results["TFT"] = {
                "mae": orig["mae"],
                "rmse": orig.get("rmse"),
                "mase_unified": new_mase,
                "source": "v8_final",
            }
            avp_cache_data[h]["metrics"].append(
                {"Mô hình": "TFT", "MAE": f"{orig['mae']:.4f}", "RMSE": f"{orig.get('rmse', 0):.4f}", "MASE": f"{new_mase:.4f}"}
            )

        if tft_preds and h_key in tft_preds and "TFT" in tft_preds[h_key]:
            h_model_preds["TFT"] = tft_preds[h_key]["TFT"]

        # ── Ensemble (from ensemble/ directory) ──
        if ensemble and h_key in ensemble:
            for m_key in ["Ensemble_Stack", "Ensemble_GRU", "Ensemble_Weighted"]:
                if m_key in ensemble[h_key]:
                    orig = ensemble[h_key][m_key]
                    new_mase = round(orig["mae"] / persist_mae, 4)
                    h_results[m_key] = {
                        "mae": orig["mae"],
                        "rmse": orig.get("rmse"),
                        "mase_unified": new_mase,
                        "source": "ensemble",
                    }
                    avp_cache_data[h]["metrics"].append(
                        {"Mô hình": m_key, "MAE": f"{orig['mae']:.4f}", "RMSE": f"{orig.get('rmse', 0):.4f}", "MASE": f"{new_mase:.4f}"}
                    )

            if ensemble_preds and h_key in ensemble_preds:
                for m_key in ["Ensemble_Stack", "Ensemble_Weighted"]:
                    if m_key in ensemble_preds[h_key]:
                        h_model_preds[m_key] = ensemble_preds[h_key][m_key]

        standardized[h_key] = h_results

        # Save AVP Cache for this horizon
        avp_cache_data[h]["model_preds"] = h_model_preds
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"avp_{h}h.json"

        def _convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.generic): return obj.item()
            return obj

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(avp_cache_data[h], f, indent=2, ensure_ascii=False, default=_convert)
        
        n_models = len(h_model_preds)
        print(f"  💾 Saved Cache: {cache_path} ({n_models} model preds)", flush=True)

    # ── Print summary table ──
    print(f"\n{'Hz':<6} {'Model':<22} {'MAE':>8} {'RMSE':>8} {'MASE*':>8}", flush=True)
    print("─" * 60, flush=True)
    for h in HORIZONS:
        h_key = f"{h}h"
        for m_name, m_data in standardized[h_key].items():
            mae = m_data.get("mae", "?")
            rmse = m_data.get("rmse", "?")
            mase = m_data.get("mase", m_data.get("mase_unified", "?"))
            print(f"{h_key:<6} {m_name:<22} {mae:>8.4f} {str(rmse):>8} {str(mase):>8}", flush=True)
        print()

    # ── Save dashboard snapshot ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EXPERIMENT_DIR / "dashboard_runs" / f"v8_final_cqr_{ts}.json"

    output = {
        "version": f"v8_final_cqr_{ts}",
        "timestamp": datetime.now().isoformat(),
        "description": "Unified pipeline (v8_final) with exact cache alignment.",
        "feature_set": {
            "lag": True,
            "rolling": True,
            "ewm": True,
            "diff": True,
            "calendar": True,
            "domain": True,
            "fourier": True,
            "interaction": True,
            "log_transform": True,
            "rolling_range": True,
            "cv_features": True,
            "pca_reduction": True,
        },
        "_metadata": {
            "description": "Standardized metrics with unified Persistence baseline",
            "generated": ts,
            "unified_test": "val_end to end, real-only, consistent across all models",
        },
        "unified_persistence": {f"{h}h": unified_persist[h] for h in HORIZONS},
        "results": standardized,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved Metrics: {out_path}", flush=True)


if __name__ == "__main__":
    main()
