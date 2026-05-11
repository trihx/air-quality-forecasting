"""Compute Conformal Prediction Intervals for GRU.

Phương án A: Post-hoc conformal — no retraining required.
Uses validation set as calibration set to compute conformal width.
Coverage guaranteed >= (1-alpha) under exchangeability assumption.

Usage:
    uv run python research/scripts/compute_gru_conformal.py

References:
    - Vovk et al. (2005) "Algorithmic Learning in a Random World"
    - Romano et al. (2019) "Conformalized Quantile Regression"
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Setup paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import TARGET_COL

LOOKBACK = 72
HORIZONS = [1, 6, 24]
ALPHA = 0.10  # target 90% coverage
FEATURE_COLS = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]


def load_pipeline_data() -> pd.DataFrame:
    """Load and clean data through full pipeline."""
    from src.data.cleaner import (
        _clip_physical_bounds,
        _handle_outliers,
        _remove_duplicates,
        _resample,
        _set_datetime_index,
    )
    from src.data.imputer import impute_missing_data
    from src.data.loader import load_raw_data

    logger.info("Loading raw data...")
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
        verbose=True,
    )
    logger.info(f"Pipeline data: {len(df_hybrid)} rows")
    return df_hybrid


def create_sequences(data: np.ndarray, target: np.ndarray, 
                     lookback: int, horizon: int):
    """Create windowed sequences for GRU prediction."""
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i])
        y.append(target[i + horizon - 1])
    return np.array(X), np.array(y)


def compute_conformal_for_horizon(
    df: pd.DataFrame, horizon: int, alpha: float = 0.10
) -> dict:
    """Compute conformal prediction intervals for GRU at given horizon."""
    import torch

    logger.info(f"\n{'='*60}")
    logger.info(f"Computing Conformal PI for GRU h={horizon}")
    logger.info(f"{'='*60}")

    # ── Load GRU model and scalers ──
    model_path = PROJECT_ROOT / "models" / "exported" / f"gru_{horizon}h.pt"
    scaler_path = PROJECT_ROOT / "models" / "exported" / f"scalers_{horizon}h.json"

    if not model_path.exists():
        logger.warning(f"GRU model not found: {model_path}, skipping h={horizon}")
        return None

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    with open(scaler_path, encoding="utf-8") as f:
        sc = json.load(f)
    feat_mean = np.array(sc["feature_scaler_mean"])
    feat_scale = np.array(sc["feature_scaler_scale"])
    tgt_mean = sc["target_scaler_mean"]
    tgt_scale = sc["target_scaler_scale"]
    features = sc["features"]

    # ── Prepare data ──
    df_feat = df[features].dropna()
    values = df_feat.values.astype(np.float64)
    target = df["pm25"].loc[df_feat.index].values.astype(np.float64)

    # ── Temporal split 80/10/10 ──
    n = len(values)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Calibration set = validation portion
    cal_values = values[train_end:val_end]
    cal_target = target[train_end:val_end]
    
    # Test set for coverage evaluation
    test_values = values[val_end:]
    test_target = target[val_end:]

    logger.info(f"Calibration set: {len(cal_values)} rows")
    logger.info(f"Test set: {len(test_values)} rows")

    def predict_batch(data_values, data_target, lookback, horizon):
        """Run GRU predictions on a data split."""
        X_seq, y_true = create_sequences(data_values, data_target, lookback, horizon)
        if len(X_seq) == 0:
            return np.array([]), np.array([])

        # Scale features
        X_scaled = (X_seq - feat_mean) / feat_scale
        X_tensor = torch.FloatTensor(X_scaled)

        # Batch predict
        preds_scaled = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                out = model(batch).cpu().numpy().flatten()
                preds_scaled.extend(out)

        preds_scaled = np.array(preds_scaled)
        # Inverse scale
        preds = preds_scaled * tgt_scale + tgt_mean
        return preds, y_true

    # ── Step 1: Calibration — compute nonconformity scores ──
    cal_preds, cal_y = predict_batch(cal_values, cal_target, LOOKBACK, horizon)
    if len(cal_preds) == 0:
        logger.error(f"No calibration predictions for h={horizon}")
        return None

    cal_scores = np.abs(cal_y - cal_preds)
    cal_mae = float(np.mean(cal_scores))

    # Conformal quantile (Vovk 2005: use (n+1-ceil((n+1)*alpha)) / n+1)
    n_cal = len(cal_scores)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / (n_cal + 1)
    q_level = min(q_level, 1.0)
    conformal_width = float(np.quantile(cal_scores, q_level))

    logger.info(f"Calibration MAE: {cal_mae:.3f}")
    logger.info(f"Conformal width (α={alpha}): ±{conformal_width:.3f} µg/m³")

    # ── Step 2: Evaluate coverage on TEST set ──
    test_preds, test_y = predict_batch(test_values, test_target, LOOKBACK, horizon)
    if len(test_preds) == 0:
        logger.error(f"No test predictions for h={horizon}")
        return None

    test_lower = test_preds - conformal_width
    test_upper = test_preds + conformal_width
    covered = np.sum((test_y >= test_lower) & (test_y <= test_upper))
    coverage = covered / len(test_y)
    test_mae = float(np.mean(np.abs(test_y - test_preds)))
    avg_width = conformal_width * 2  # total interval width

    logger.info(f"Test MAE: {test_mae:.3f}")
    logger.info(f"Test Coverage: {coverage:.4f} ({covered}/{len(test_y)})")
    logger.info(f"Avg Interval Width: {avg_width:.3f} µg/m³")
    logger.info(f"Target coverage: {1 - alpha:.0%} → Achieved: {coverage:.1%}")

    return {
        "method": "conformal_prediction",
        "model": "GRU",
        "horizon": horizon,
        "alpha": alpha,
        "coverage": round(coverage, 4),
        "avg_width": round(avg_width, 3),
        "conformal_width": round(conformal_width, 3),
        "mae": round(test_mae, 3),
        "n_calibration": n_cal,
        "n_test": len(test_y),
        "cal_mae": round(cal_mae, 3),
    }


def main():
    """Run conformal prediction for all horizons."""
    logger.info("=" * 70)
    logger.info("GRU Conformal Prediction Intervals — Phương án A")
    logger.info("=" * 70)

    df = load_pipeline_data()
    results = []

    for h in HORIZONS:
        result = compute_conformal_for_horizon(df, h, ALPHA)
        if result:
            results.append(result)
            logger.success(
                f"h={h}: coverage={result['coverage']:.1%}, "
                f"width=±{result['conformal_width']:.2f} µg/m³"
            )

    if not results:
        logger.error("No results computed!")
        return

    # ── Save results ──
    out_dir = PROJECT_ROOT / "research" / "experiments" / "prediction_intervals"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing PI file and merge
    existing_file = out_dir / "prediction_intervals_20260405_100353.json"
    if existing_file.exists():
        with open(existing_file, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    # Remove old GRU conformal entries (if re-running)
    existing = [
        r for r in existing
        if not (r["model"] == "GRU" and r["method"] == "conformal_prediction")
    ]

    # Add new results
    merged = existing + results

    # Save back to same file (keeps things simple)
    with open(existing_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    logger.success(f"Saved {len(results)} new entries to {existing_file}")

    # ── Summary table ──
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: GRU Conformal Prediction vs MC Dropout")
    logger.info("=" * 70)
    logger.info(f"{'Horizon':>8} {'Method':>25} {'Coverage':>10} {'Width':>10} {'MAE':>8}")
    logger.info("-" * 70)
    for r in merged:
        if r["model"] == "GRU":
            logger.info(
                f"{r['horizon']:>8} {r['method']:>25} "
                f"{r['coverage']:>10.1%} {r['avg_width']:>10.3f} {r['mae']:>8.3f}"
            )


if __name__ == "__main__":
    main()
