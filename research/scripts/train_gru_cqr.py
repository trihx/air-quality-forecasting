"""CQR — Conformalized Quantile Regression for GRU.

Phương án C: Train GRU with Pinball Loss (quantile q=0.05, 0.50, 0.95),
then calibrate with Conformal Prediction for guaranteed coverage.

Workflow:
  1. Train GRUQuantile model → output (q_low, q_median, q_high)
  2. On calibration set: compute scores = max(q_low - y, y - q_high)
  3. q = quantile(scores, 1-α) → adjustment factor
  4. Final interval = [q_low - q, q_high + q]
  → Guaranteed coverage + Adaptive width

Usage:
    uv run python research/scripts/train_gru_cqr.py | tee research/logs/cqr_training.log

References:
    - Romano et al. (2019) "Conformalized Quantile Regression", NeurIPS
    - Gal & Ghahramani (2016) "Dropout as Bayesian Approximation", ICML

IMPORTANT: torch lazy import — see LESSONS_LEARNED [2026-04-12]
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Avoid OMP threading crash on Apple Silicon
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOOKBACK = 72
HORIZONS = [1, 6, 24]
QUANTILES = [0.05, 0.50, 0.95]  # 90% coverage target
ALPHA = 0.10
FEATURE_COLS = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]
SEED = 42


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


def create_sequences(data, target, lookback, horizon):
    """Create windowed sequences."""
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i])
        y.append(target[i + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_gru_quantile(horizon: int, df: pd.DataFrame) -> dict:
    """Train GRU with Pinball Loss + CQR calibration for one horizon."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger.info(f"\n{'='*70}")
    logger.info(f"CQR Training — GRU Quantile h={horizon}")
    logger.info(f"{'='*70}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data preparation ──
    available = [c for c in FEATURE_COLS if c in df.columns]
    features = df[available].values.astype(np.float64)
    target = df["pm25"].values.astype(np.float64)
    is_imputed = df["is_imputed"].values if "is_imputed" in df.columns else np.zeros(len(target))

    n = len(features)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # ── Scale (fit on TRAIN ONLY — per AGENTS.md rule) ──
    feat_scaler = StandardScaler()
    feat_scaled = np.zeros_like(features)
    feat_scaled[:train_end] = feat_scaler.fit_transform(features[:train_end])
    feat_scaled[train_end:] = feat_scaler.transform(features[train_end:])

    tgt_scaler = StandardScaler()
    tgt_scaler.fit(target[:train_end].reshape(-1, 1))
    tgt_scaled = tgt_scaler.transform(target.reshape(-1, 1)).flatten()

    # ── Create sequences ──
    X_all, y_all = create_sequences(feat_scaled, tgt_scaled, LOOKBACK, horizon)
    # Track which indices belong to which split
    idx_offsets = np.arange(LOOKBACK, len(feat_scaled) - horizon + 1)
    # The target index for each sequence
    tgt_indices = idx_offsets + horizon - 1

    # Split masks
    train_mask = tgt_indices < train_end
    val_mask = (tgt_indices >= train_end) & (tgt_indices < val_end)
    # Calibration = first half of val, Test = second half of val + test portion
    cal_mask = val_mask.copy()
    test_mask = tgt_indices >= val_end

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    # Real-only mask for test
    test_tgt_idx = tgt_indices[test_mask]
    real_mask_test = np.array([is_imputed[i] == 0 for i in test_tgt_idx])

    logger.info(f"Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)} (real: {real_mask_test.sum()})")

    # ── GRU Quantile Model ──
    class GRUQuantile(nn.Module):
        """GRU with 3-quantile output (q05, q50, q95)."""

        def __init__(self, in_dim, hidden=64, layers=2, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(
                in_dim, hidden, layers,
                dropout=dropout if layers > 1 else 0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 3),  # 3 quantiles
            )

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    # ── Pinball Loss ──
    def pinball_loss(preds, target, quantiles=QUANTILES):
        """Combined pinball loss for multiple quantiles.

        Args:
            preds: (batch_size, n_quantiles) — model output
            target: (batch_size,) — actual values (same y for all quantiles)
        """
        losses = []
        for i, q in enumerate(quantiles):
            error = target - preds[:, i]  # both (batch_size,)
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        return sum(losses) / len(quantiles)

    # ── Training setup ──
    model = GRUQuantile(len(available), hidden=64, layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

    cal_ds = TensorDataset(torch.FloatTensor(X_cal), torch.FloatTensor(y_cal))
    cal_loader = DataLoader(cal_ds, batch_size=256, shuffle=False)

    # ── Training loop ──
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    max_epochs = 100
    patience = 10

    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = pinball_loss(preds, yb)  # yb is (batch,), preds is (batch, 3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # Validation on calibration set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in cal_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = pinball_loss(preds, yb)  # yb is (batch,), preds is (batch, 3)
                val_loss += loss.item()
        val_loss /= max(len(cal_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch+1:3d}/{max_epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"lr={lr_now:.6f}", flush=True
            )

    training_time = time.time() - t0
    epochs_trained = epoch + 1

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    logger.info(f"Training complete: {epochs_trained} epochs, {training_time:.1f}s")

    # ── Step 1: Get quantile predictions on calibration set ──
    cal_preds_list = []
    with torch.no_grad():
        for xb, _ in cal_loader:
            out = model(xb.to(device)).cpu().numpy()
            cal_preds_list.append(out)
    cal_preds = np.concatenate(cal_preds_list, axis=0)  # (n_cal, 3)

    # Inverse scale
    cal_q05 = cal_preds[:, 0] * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]
    cal_q50 = cal_preds[:, 1] * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]
    cal_q95 = cal_preds[:, 2] * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]
    cal_y_real = y_cal * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]

    # ── Step 2: CQR calibration — compute nonconformity scores ──
    # Score = max(q_low - y, y - q_high)  [Romano et al. 2019]
    cal_scores = np.maximum(cal_q05 - cal_y_real, cal_y_real - cal_q95)
    n_cal = len(cal_scores)
    q_level = np.ceil((n_cal + 1) * (1 - ALPHA)) / (n_cal + 1)
    q_level = min(q_level, 1.0)
    conformal_adjustment = float(np.quantile(cal_scores, q_level))

    logger.info(f"CQR calibration: n_cal={n_cal}, adjustment={conformal_adjustment:.3f}")

    # ── Step 3: Evaluate on TEST set ──
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    test_preds_list = []
    with torch.no_grad():
        for xb, _ in test_loader:
            out = model(xb.to(device)).cpu().numpy()
            test_preds_list.append(out)
    test_preds = np.concatenate(test_preds_list, axis=0)

    # Inverse scale
    test_q05 = test_preds[:, 0] * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]
    test_q50 = test_preds[:, 1] * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]
    test_q95 = test_preds[:, 2] * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]
    test_y_real = y_test * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]

    # CQR interval = [q_low - adjustment, q_high + adjustment]
    cqr_lower = test_q05 - conformal_adjustment
    cqr_upper = test_q95 + conformal_adjustment

    # Coverage & width (on real data only)
    y_real = test_y_real[real_mask_test]
    lower_real = cqr_lower[real_mask_test]
    upper_real = cqr_upper[real_mask_test]
    median_real = test_q50[real_mask_test]

    covered = np.sum((y_real >= lower_real) & (y_real <= upper_real))
    coverage = covered / len(y_real)
    avg_width = float(np.mean(upper_real - lower_real))
    mae_point = float(mean_absolute_error(y_real, median_real))

    # Also compute raw quantile coverage (before CQR)
    raw_lower = test_q05[real_mask_test]
    raw_upper = test_q95[real_mask_test]
    raw_covered = np.sum((y_real >= raw_lower) & (y_real <= raw_upper))
    raw_coverage = raw_covered / len(y_real)
    raw_width = float(np.mean(raw_upper - raw_lower))

    # Persistence MAE for MASE
    persist_preds = np.array([target[idx] for idx in test_tgt_idx[real_mask_test] - horizon])
    persist_mae = float(mean_absolute_error(y_real, persist_preds))
    mase = mae_point / persist_mae if persist_mae > 0 else float("inf")

    logger.info(f"\n--- Results h={horizon} ---")
    logger.info(f"MAE (median):    {mae_point:.3f} µg/m³")
    logger.info(f"MASE:            {mase:.4f}")
    logger.info(f"Persist MAE:     {persist_mae:.3f}")
    logger.info(f"Raw QR coverage: {raw_coverage:.1%} (width={raw_width:.2f})")
    logger.info(f"CQR coverage:    {coverage:.1%} (width={avg_width:.2f})")
    logger.info(f"Target coverage: {1-ALPHA:.0%}")

    # ── Export model ──
    export_dir = PROJECT_ROOT / "models" / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save TorchScript
    model.cpu().eval()
    example_input = torch.randn(1, LOOKBACK, len(available))
    scripted = torch.jit.trace(model, example_input)
    model_path = export_dir / f"gru_quantile_{horizon}h.pt"
    scripted.save(str(model_path))
    logger.info(f"Exported model: {model_path}")

    # Save scalers + CQR config
    cqr_config = {
        "feature_scaler_mean": feat_scaler.mean_.tolist(),
        "feature_scaler_scale": feat_scaler.scale_.tolist(),
        "target_scaler_mean": float(tgt_scaler.mean_[0]),
        "target_scaler_scale": float(tgt_scaler.scale_[0]),
        "features": available,
        "lookback": LOOKBACK,
        "horizon": horizon,
        "quantiles": QUANTILES,
        "conformal_adjustment": conformal_adjustment,
        "alpha": ALPHA,
    }
    config_path = export_dir / f"gru_quantile_{horizon}h_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cqr_config, f, indent=2)
    logger.info(f"Exported config: {config_path}")

    # Free GPU memory
    import gc
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

    return {
        "method": "cqr",
        "model": "GRU",
        "horizon": horizon,
        "alpha": ALPHA,
        "quantiles": QUANTILES,
        "coverage_raw": round(raw_coverage, 4),
        "coverage_cqr": round(coverage, 4),
        "avg_width_raw": round(raw_width, 3),
        "avg_width_cqr": round(avg_width, 3),
        "conformal_adjustment": round(conformal_adjustment, 3),
        "mae": round(mae_point, 3),
        "mase": round(mase, 4),
        "persist_mae": round(persist_mae, 3),
        "n_calibration": n_cal,
        "n_test": int(real_mask_test.sum()),
        "epochs_trained": epochs_trained,
        "training_time_s": round(training_time, 1),
        # For PI JSON compatibility
        "coverage": round(coverage, 4),
        "avg_width": round(avg_width, 3),
        "conformal_width": round(avg_width / 2, 3),  # half-width for ± display
    }


def main():
    logger.info("=" * 70)
    logger.info("CQR — Conformalized Quantile Regression for GRU")
    logger.info("Romano et al. (2019), NeurIPS")
    logger.info("=" * 70)

    df = load_pipeline_data()
    all_results = []

    for h in HORIZONS:
        try:
            result = train_gru_quantile(h, df)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Failed h={h}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        logger.error("No results!")
        return

    # ── Update prediction_intervals JSON ──
    pi_dir = PROJECT_ROOT / "research" / "experiments" / "prediction_intervals"
    pi_file = pi_dir / "prediction_intervals_20260405_100353.json"
    if pi_file.exists():
        with open(pi_file, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    # Remove old GRU mc_dropout entries (replaced by CQR)
    existing = [r for r in existing if not (r["model"] == "GRU" and r["method"] in ("mc_dropout", "cqr", "conformal_prediction"))]

    # Add CQR results
    for r in all_results:
        pi_entry = {
            "method": "cqr",
            "model": "GRU",
            "horizon": r["horizon"],
            "alpha": r["alpha"],
            "coverage": r["coverage"],
            "avg_width": r["avg_width"],
            "conformal_width": r["conformal_width"],
            "mae": r["mae"],
            "n_test": r["n_test"],
            "conformal_adjustment": r["conformal_adjustment"],
        }
        existing.append(pi_entry)

    with open(pi_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.success(f"Updated {pi_file}")

    # ── Save detailed CQR results ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = pi_dir / f"cqr_results_{ts}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.success(f"Detailed results: {detail_path}")

    # ── Summary ──
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY: GRU CQR vs MC Dropout (old)")
    logger.info("=" * 70)
    logger.info(f"{'Horizon':>8} {'Coverage(CQR)':>15} {'Width(CQR)':>12} {'MAE':>8} {'MASE':>8}")
    logger.info("-" * 60)
    for r in all_results:
        logger.info(
            f"{r['horizon']:>8} {r['coverage_cqr']:>15.1%} "
            f"{r['avg_width_cqr']:>12.2f} {r['mae']:>8.3f} {r['mase']:>8.4f}"
        )
    logger.info("-" * 60)
    logger.info("Old MC Dropout: h=1: 36.8%, h=6: 7.6%, h=24: 25.7%")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
