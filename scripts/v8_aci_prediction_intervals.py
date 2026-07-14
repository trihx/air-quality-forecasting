"""V8 ACI Prediction Intervals — Compare CQR vs ACI.

Loads GRU Quantile models, runs inference on test set,
then compares CQR (static) vs ACI (adaptive) coverage.

Output: research/experiments/v8_final/aci_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "v8_final"
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"
LOOKBACK = 72
HORIZONS = [1, 6, 24]


def load_and_split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed data and split into train/cal/test."""
    marts_path = PROJECT_ROOT / "dataset" / "processed" / "marts_features.csv"
    df = pd.read_csv(marts_path, index_col=0, parse_dates=True)

    n = len(df)
    n_train = int(n * 0.7)
    n_cal = int(n * 0.1)
    # train=70%, cal=10%, test=20%

    train = df.iloc[:n_train]
    cal = df.iloc[n_train: n_train + n_cal]
    test = df.iloc[n_train + n_cal:]

    print(f"  Data split: train={len(train)}, cal={len(cal)}, test={len(test)}", flush=True)
    return train, cal, test


def run_quantile_inference(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run GRU quantile model on all windows in df.

    Returns:
        (q_lower, q_median, q_upper) arrays — raw quantile predictions.
    """
    import torch

    model_path = EXPORT_DIR / f"gru_quantile_{horizon}h.pt"
    config_path = EXPORT_DIR / f"gru_quantile_{horizon}h_config.json"

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    feat_mean = np.array(cfg["feature_scaler_mean"])
    feat_scale = np.array(cfg["feature_scaler_scale"])
    tgt_mean = cfg["target_scaler_mean"]
    tgt_scale = cfg["target_scaler_scale"]
    features = cfg["features"]

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    values = df[features].values.astype(np.float64)
    n_samples = len(df) - LOOKBACK
    q_lower = np.zeros(n_samples)
    q_median = np.zeros(n_samples)
    q_upper = np.zeros(n_samples)

    for i in range(n_samples):
        window = values[i: i + LOOKBACK]
        scaled = (window - feat_mean) / feat_scale
        x = torch.FloatTensor(scaled).unsqueeze(0)

        with torch.no_grad():
            out = model(x).cpu().numpy().flatten()

        q_vals = out * tgt_scale + tgt_mean
        q_lower[i] = q_vals[0]
        q_median[i] = q_vals[1]
        q_upper[i] = q_vals[2]

    return q_lower, q_median, q_upper


def main():
    from src.evaluation.adaptive_conformal import adaptive_conformal_inference

    print("=" * 60, flush=True)
    print("  V8 ACI Prediction Intervals", flush=True)
    print("=" * 60, flush=True)

    train, cal, test = load_and_split_data()
    features = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]

    all_results = []

    for horizon in HORIZONS:
        print(f"\n  --- Horizon {horizon}h ---", flush=True)

        # Run quantile inference on calibration set
        print(f"  Running quantile inference on cal set...", flush=True)
        cal_lower, cal_median, cal_upper = run_quantile_inference(cal, horizon)
        cal_actuals = cal["pm25"].values[LOOKBACK:]
        n_cal = min(len(cal_actuals), len(cal_lower))
        cal_lower = cal_lower[:n_cal]
        cal_upper = cal_upper[:n_cal]
        cal_actuals = cal_actuals[:n_cal]
        print(f"  Cal: {n_cal} samples", flush=True)

        # Run quantile inference on test set
        print(f"  Running quantile inference on test set...", flush=True)
        test_lower, test_median, test_upper = run_quantile_inference(test, horizon)
        test_actuals = test["pm25"].values[LOOKBACK:]
        n_test = min(len(test_actuals), len(test_lower))
        test_lower = test_lower[:n_test]
        test_upper = test_upper[:n_test]
        test_actuals = test_actuals[:n_test]
        print(f"  Test: {n_test} samples", flush=True)

        # CQR (static) — use conformal_adjustment from config
        config_path = EXPORT_DIR / f"gru_quantile_{horizon}h_config.json"
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cqr_adj = cfg.get("conformal_adjustment", 0.0)

        cqr_lower_adj = test_lower - cqr_adj
        cqr_upper_adj = test_upper + cqr_adj
        cqr_covered = (test_actuals >= cqr_lower_adj) & (test_actuals <= cqr_upper_adj)
        cqr_coverage = float(np.mean(cqr_covered))
        cqr_width = float(np.mean(cqr_upper_adj - cqr_lower_adj))

        print(f"  CQR: coverage={cqr_coverage:.4f}, avg_width={cqr_width:.2f}", flush=True)

        # ACI — adaptive
        for gamma in [0.001, 0.005, 0.01, 0.02]:
            aci_result = adaptive_conformal_inference(
                y_true=test_actuals,
                y_pred_lower=test_lower,
                y_pred_upper=test_upper,
                y_cal_true=cal_actuals,
                y_cal_lower=cal_lower,
                y_cal_upper=cal_upper,
                alpha=0.10,
                gamma=gamma,
            )

            summary = {
                "horizon": horizon,
                "method": "aci",
                "gamma": gamma,
                "coverage": aci_result["coverage"],
                "avg_width": aci_result["avg_width"],
                "n_test": n_test,
                "n_calibration": n_cal,
                "cqr_coverage": round(cqr_coverage, 4),
                "cqr_avg_width": round(cqr_width, 4),
                "improvement_over_cqr": round(aci_result["coverage"] - cqr_coverage, 4),
            }
            all_results.append(summary)

            delta = aci_result["coverage"] - cqr_coverage
            arrow = "↑" if delta > 0 else "↓"
            print(
                f"  ACI(γ={gamma}): coverage={aci_result['coverage']:.4f} "
                f"({arrow}{abs(delta):.4f} vs CQR), "
                f"width={aci_result['avg_width']:.2f}",
                flush=True,
            )

    # Save results
    out_path = OUTPUT_DIR / "aci_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✅ Saved: {out_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  ACI Script Complete!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
