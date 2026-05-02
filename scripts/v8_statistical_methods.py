"""V8 Statistical Methodology — Unified MASE + DM-HLN + R² Multi-Horizon.

Loads existing prediction files and computes:
1. MASE_hyndman (Hyndman & Koehler 2006) — in-sample naive MAE denominator
2. DM test with HLN correction (Harvey et al. 1997)
3. R² for all models × 3 horizons

Handles different test set sizes across ML/DL/ARIMA sources by
loading ground truth directly from marts_features.csv.

Output:
    research/experiments/v8_final/unified_mase.json
    research/experiments/v8_final/dm_test_hln.json
    research/experiments/v8_final/r2_multi_horizon.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import mase_hyndman, r2_score, mae
from src.evaluation.residual_diagnostics import dm_test_hln

OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "v8_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prediction files — use latest timestamps
PRED_FILES = {
    "lightgbm": "lightgbm_preds_20260502_073700.json",
    "sklearn": "sklearn_preds_20260502_073736.json",
    "dl": "dl_preds_20260502_075412.json",
    "tft": "tft_preds_20260429_163633.json",  # Keep old TFT for now as it wasn't retrained
    "arima": "arima_preds_20260502_074241.json",
}

HORIZONS = ["1h", "6h", "24h"]
HORIZON_MAP = {"1h": 1, "6h": 6, "24h": 24}


def load_target_series() -> tuple[np.ndarray, np.ndarray]:
    """Load PM2.5 target and split into train/full.

    Returns:
        (y_train, y_full) — training subset and full series.
    """
    marts_path = PROJECT_ROOT / "dataset" / "processed" / "marts_features.csv"
    if marts_path.exists():
        df = pd.read_csv(marts_path, index_col=0, parse_dates=True)
        target = df["pm25"].values
    else:
        raw_path = PROJECT_ROOT / "dataset" / "raw" / "final_dataset.csv"
        df = pd.read_csv(raw_path)
        target = df["pm25"].dropna().values

    n_train = int(len(target) * 0.8)
    y_train = target[:n_train]
    print(f"  Target loaded: {len(target)} total, {n_train} train", flush=True)
    return y_train, target


def load_predictions_per_source() -> dict:
    """Load prediction files keeping source separation.

    Returns:
        {source: {horizon: {model: np.array, ...}}}
    """
    all_data = {}
    for source, fname in PRED_FILES.items():
        fpath = OUTPUT_DIR / fname
        if not fpath.exists():
            print(f"  [SKIP] {fname} not found", flush=True)
            continue
        with open(fpath) as f:
            data = json.load(f)
        all_data[source] = {}
        for h_key in HORIZONS:
            if h_key not in data:
                continue
            all_data[source][h_key] = {}
            for model_name, preds in data[h_key].items():
                all_data[source][h_key][model_name] = np.array(preds, dtype=float)
    return all_data


def get_actuals_for_preds(source_data: dict, h_key: str, y_full: np.ndarray, horizon: int) -> np.ndarray | None:
    """Get matching actuals for a set of predictions.

    If source has 'Actuals' key, use it.
    Otherwise, compute from y_full using tail alignment.
    """
    if "Actuals" in source_data:
        return source_data["Actuals"]

    # Get any model's prediction length to determine test set size
    for model_name, preds in source_data.items():
        if model_name == "Persistence":
            continue
        n_preds = len(preds)
        # Actuals = shifted target → take from tail
        actuals = y_full[-(n_preds + horizon):-horizon] if horizon > 0 else y_full[-n_preds:]
        if len(actuals) == n_preds:
            return actuals
        # Fallback: just take tail
        return y_full[-n_preds:]
    return None


def compute_unified_mase(all_sources: dict, y_train: np.ndarray, y_full: np.ndarray) -> dict:
    """Task 2.1: Compute MASE_hyndman for all models × horizons."""
    print("\n" + "=" * 60, flush=True)
    print("  TASK 2.1 — Unified MASE (Hyndman 2006)", flush=True)
    print("=" * 60, flush=True)

    results = {}
    for h_key in HORIZONS:
        horizon = HORIZON_MAP[h_key]
        results[h_key] = {}
        print(f"\n  --- Horizon {h_key} ---", flush=True)

        for source, source_data_all in all_sources.items():
            if h_key not in source_data_all:
                continue
            source_data = source_data_all[h_key]
            actuals = get_actuals_for_preds(source_data, h_key, y_full, horizon)
            if actuals is None:
                continue

            for model_name, preds in sorted(source_data.items()):
                if model_name in ("Actuals", "Persistence"):
                    continue
                n = min(len(preds), len(actuals))
                y_true = actuals[:n]
                y_pred = preds[:n]

                mase_val = mase_hyndman(y_true, y_pred, y_train, horizon=horizon)
                mae_val = mae(y_true, y_pred)
                results[h_key][model_name] = {
                    "mase_hyndman": round(mase_val, 4),
                    "mae": round(mae_val, 4),
                    "n_samples": n,
                }
                status = "✅" if mase_val < 1.0 else "❌"
                print(f"  {model_name:25s}: MASE_h={mase_val:.4f} {status}  MAE={mae_val:.4f}  (n={n})", flush=True)

            # Persistence from this source
            if "Persistence" in source_data:
                pers = source_data["Persistence"]
                n = min(len(pers), len(actuals))
                mase_p = mase_hyndman(actuals[:n], pers[:n], y_train, horizon=horizon)
                mae_p = mae(actuals[:n], pers[:n])
                if "Persistence" not in results[h_key]:
                    results[h_key]["Persistence"] = {
                        "mase_hyndman": round(mase_p, 4),
                        "mae": round(mae_p, 4),
                        "n_samples": n,
                    }
                    print(f"  {'Persistence':25s}: MASE_h={mase_p:.4f} 📊  MAE={mae_p:.4f}  (n={n})", flush=True)

    return results


def compute_dm_hln(all_sources: dict, y_full: np.ndarray) -> dict:
    """Task 2.2: DM test with HLN correction for key model pairs."""
    print("\n" + "=" * 60, flush=True)
    print("  TASK 2.2 — DM Test with HLN Correction", flush=True)
    print("=" * 60, flush=True)

    # Flatten all predictions per horizon
    flat = {}  # {h_key: {model: (preds, actuals, n)}}
    for h_key in HORIZONS:
        horizon = HORIZON_MAP[h_key]
        flat[h_key] = {}
        for source, source_data_all in all_sources.items():
            if h_key not in source_data_all:
                continue
            source_data = source_data_all[h_key]
            actuals = get_actuals_for_preds(source_data, h_key, y_full, horizon)
            if actuals is None:
                continue
            for model_name, preds in source_data.items():
                if model_name == "Actuals":
                    continue
                n = min(len(preds), len(actuals))
                flat[h_key][model_name] = (preds[:n], actuals[:n])

    comparison_pairs = [
        ("LightGBM_tuned", "Persistence"),
        ("GRU_v2_log", "Persistence"),
        ("LSTM_v2_log", "Persistence"),
        ("GRU_v2_log", "LightGBM_tuned"),
        ("LSTM_v2_log", "LightGBM_tuned"),
        ("Ensemble_Weighted", "Persistence"),
        ("Ensemble_Weighted", "LightGBM_tuned"),
        ("Ensemble_Weighted", "GRU_v2_log"),
        ("RandomForest", "Persistence"),
        ("ARIMA", "Persistence"),
        ("TFT", "Persistence"),
    ]

    results = {}
    for h_key in HORIZONS:
        horizon = HORIZON_MAP[h_key]
        results[h_key] = []
        print(f"\n  --- Horizon {h_key} ---", flush=True)

        for model_a, model_b in comparison_pairs:
            if model_a not in flat[h_key] or model_b not in flat[h_key]:
                continue

            preds_a, actuals_a = flat[h_key][model_a]
            preds_b, actuals_b = flat[h_key][model_b]

            # Align to same length (use shorter)
            n = min(len(preds_a), len(preds_b))
            # Use same actuals — take from the source with matching length
            act_a = actuals_a[-n:]
            preds_a_aligned = preds_a[-n:]
            preds_b_aligned = preds_b[-n:]

            e1 = act_a - preds_a_aligned
            e2 = act_a - preds_b_aligned

            dm_result = dm_test_hln(e1, e2, horizon=horizon)
            dm_result["model_a"] = model_a
            dm_result["model_b"] = model_b

            results[h_key].append(dm_result)

            sig = "***" if dm_result["HLN_p_value"] < 0.01 else (
                "**" if dm_result["HLN_p_value"] < 0.05 else (
                    "*" if dm_result["HLN_p_value"] < 0.10 else "ns"))
            print(
                f"  {model_a:20s} vs {model_b:20s}: "
                f"DM={dm_result['DM_statistic']:+.3f}  "
                f"HLN={dm_result['HLN_statistic']:+.3f}  "
                f"p_HLN={dm_result['HLN_p_value']:.4f} {sig}",
                flush=True,
            )

    return results


def compute_r2_multi_horizon(all_sources: dict, y_full: np.ndarray) -> dict:
    """Task 2.3: R² for all models × horizons."""
    print("\n" + "=" * 60, flush=True)
    print("  TASK 2.3 — R² Multi-Horizon", flush=True)
    print("=" * 60, flush=True)

    results = {}
    for h_key in HORIZONS:
        horizon = HORIZON_MAP[h_key]
        results[h_key] = {}
        print(f"\n  --- Horizon {h_key} ---", flush=True)

        for source, source_data_all in all_sources.items():
            if h_key not in source_data_all:
                continue
            source_data = source_data_all[h_key]
            actuals = get_actuals_for_preds(source_data, h_key, y_full, horizon)
            if actuals is None:
                continue

            for model_name, preds in sorted(source_data.items()):
                if model_name == "Actuals":
                    continue
                n = min(len(preds), len(actuals))
                r2 = r2_score(actuals[:n], preds[:n])
                results[h_key][model_name] = {
                    "r2": round(r2, 4),
                    "n_samples": n,
                }
                print(f"  {model_name:25s}: R²={r2:.4f}  (n={n})", flush=True)

    return results


def main():
    print("=" * 60, flush=True)
    print("  V8 Statistical Methodology Script (v2 — aligned)", flush=True)
    print("=" * 60, flush=True)

    y_train, y_full = load_target_series()
    all_sources = load_predictions_per_source()

    print(f"\n  Loaded sources: {list(all_sources.keys())}", flush=True)
    for source, horizons in all_sources.items():
        for h_key, models in horizons.items():
            names = [m for m in models.keys() if m != "Actuals"]
            print(f"  {source}/{h_key}: {names}", flush=True)

    # Task 2.1
    mase_results = compute_unified_mase(all_sources, y_train, y_full)
    mase_path = OUTPUT_DIR / "unified_mase.json"
    with open(mase_path, "w") as f:
        json.dump(mase_results, f, indent=2)
    print(f"\n  ✅ Saved: {mase_path}", flush=True)

    # Task 2.2
    dm_results = compute_dm_hln(all_sources, y_full)
    dm_path = OUTPUT_DIR / "dm_test_hln.json"
    with open(dm_path, "w") as f:
        json.dump(dm_results, f, indent=2)
    print(f"  ✅ Saved: {dm_path}", flush=True)

    # Task 2.3
    r2_results = compute_r2_multi_horizon(all_sources, y_full)
    r2_path = OUTPUT_DIR / "r2_multi_horizon.json"
    with open(r2_path, "w") as f:
        json.dump(r2_results, f, indent=2)
    print(f"  ✅ Saved: {r2_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  Phase 2 Script Complete!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
