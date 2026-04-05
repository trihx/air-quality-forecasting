"""Prediction Intervals — Conformal Prediction + GRU Bootstrap.

Implements:
  1. Conformal Prediction (split conformal) for LightGBM
  2. MC Dropout Bootstrap for GRU
  3. Quantile Regression for LightGBM

Output: prediction intervals at 90% confidence for each horizon.

Usage:
    uv run python scripts/prediction_intervals.py 2>&1 | tee research/logs/prediction_intervals.log
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.cleaner import (
    _clip_physical_bounds,
    _handle_outliers,
    _remove_duplicates,
    _resample,
    _set_datetime_index,
)
from src.data.imputer import impute_missing_data
from src.data.loader import TARGET_COL, load_raw_data
from src.features.builder import build_features

warnings.filterwarnings("ignore")

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "prediction_intervals"
FIG_DIR = PROJECT_ROOT / "research" / "figures" / "prediction_intervals"
HORIZONS = [1, 6, 24]
ALPHA = 0.10  # 90% prediction intervals
LOOKBACK = 72
GRU_HIDDEN = 64
GRU_LAYERS = 2
GRU_DROPOUT = 0.2
MC_SAMPLES = 50  # MC Dropout forward passes
FEATURE_COLS = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]


# ══════════════════════════════════════════════════════════════════════
# Data Preparation (reuse from other scripts)
# ══════════════════════════════════════════════════════════════════════


def prepare_data():
    """Load raw → clean → impute → features."""
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")

    df_hybrid = impute_missing_data(
        df, strategy="hybrid",
        max_gap_interp=6, max_gap_ml=24, knn_neighbors=5,
        verbose=True,
    )
    return df_hybrid


def prepare_ml_data(df_hybrid, horizon):
    """Build features and split for ML models."""
    df_feat = build_features(df_hybrid)
    is_imputed = (
        df_feat["is_imputed"].values
        if "is_imputed" in df_feat.columns
        else np.zeros(len(df_feat), dtype=bool)
    )

    feature_cols = [c for c in df_feat.columns if c not in [TARGET_COL, "is_imputed"]]
    X = df_feat[feature_cols].values
    y = df_feat[TARGET_COL].values

    # Create target at horizon h
    y_target = pd.Series(y).shift(-horizon).values

    # Remove NaN from shift
    valid = ~np.isnan(y_target) & ~np.isnan(X).any(axis=1)
    X, y_target, is_imputed = X[valid], y_target[valid], is_imputed[valid]

    # Temporal split 80/10/10
    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Test set: real data only
    test_mask = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imputed[i]:
            test_mask[i] = True

    return X, y_target, train_end, val_end, test_mask, feature_cols


# ══════════════════════════════════════════════════════════════════════
# 1. Conformal Prediction for LightGBM
# ══════════════════════════════════════════════════════════════════════


def conformal_prediction_lgbm(X, y, train_end, val_end, test_mask, horizon, alpha=ALPHA):
    """Split conformal prediction using calibration set residuals."""
    print(f"\n  [Conformal] LightGBM h={horizon}h (alpha={alpha})...", flush=True)

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[test_mask], y[test_mask]

    # Train LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.6,
        min_child_samples=30, reg_alpha=0.05, reg_lambda=0.5,
        num_leaves=64, verbose=-1, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Calibration: compute conformity scores (absolute residuals)
    cal_preds = model.predict(X_cal)
    cal_residuals = np.abs(y_cal - cal_preds)

    # Quantile of residuals at (1-alpha)(1 + 1/n_cal)
    n_cal = len(cal_residuals)
    q = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q = min(q, 1.0)
    conformal_width = float(np.quantile(cal_residuals, q))

    # Test predictions with intervals
    test_preds = model.predict(X_test)
    lower = test_preds - conformal_width
    upper = test_preds + conformal_width

    # Coverage
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    avg_width = float(np.mean(upper - lower))
    mae = float(np.mean(np.abs(y_test - test_preds)))

    print(f"    Conformal width: ±{conformal_width:.2f} µg/m³", flush=True)
    print(f"    Coverage: {coverage:.1%} (target: {1 - alpha:.0%})", flush=True)
    print(f"    Avg interval width: {avg_width:.2f} µg/m³", flush=True)
    print(f"    MAE: {mae:.3f} µg/m³", flush=True)

    return {
        "method": "conformal_prediction",
        "model": "LightGBM",
        "horizon": horizon,
        "alpha": alpha,
        "conformal_width": round(conformal_width, 3),
        "coverage": round(coverage, 4),
        "avg_width": round(avg_width, 3),
        "mae": round(mae, 3),
        "n_cal": n_cal,
        "n_test": len(y_test),
        "predictions": test_preds.tolist()[:20],  # sample for JSON
        "lower": lower.tolist()[:20],
        "upper": upper.tolist()[:20],
        "y_true": y_test.tolist()[:20],
    }


# ══════════════════════════════════════════════════════════════════════
# 2. Quantile Regression for LightGBM
# ══════════════════════════════════════════════════════════════════════


def quantile_regression_lgbm(X, y, train_end, val_end, test_mask, horizon, alpha=ALPHA):
    """Train separate LightGBM models for lower and upper quantiles."""
    print(f"\n  [Quantile] LightGBM h={horizon}h (alpha={alpha})...", flush=True)

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[test_mask], y[test_mask]

    base_params = {
        "n_estimators": 300, "max_depth": 3, "learning_rate": 0.01,
        "subsample": 0.8, "colsample_bytree": 0.6,
        "min_child_samples": 30, "num_leaves": 64,
        "verbose": -1, "n_jobs": -1,
    }

    # Lower quantile
    model_lower = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha / 2, **base_params
    )
    model_lower.fit(X_train, y_train)

    # Upper quantile
    model_upper = lgb.LGBMRegressor(
        objective="quantile", alpha=1 - alpha / 2, **base_params
    )
    model_upper.fit(X_train, y_train)

    # Median
    model_median = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, **base_params
    )
    model_median.fit(X_train, y_train)

    lower = model_lower.predict(X_test)
    upper = model_upper.predict(X_test)
    median_pred = model_median.predict(X_test)

    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    avg_width = float(np.mean(upper - lower))
    mae = float(np.mean(np.abs(y_test - median_pred)))

    print(f"    Coverage: {coverage:.1%} (target: {1 - alpha:.0%})", flush=True)
    print(f"    Avg interval width: {avg_width:.2f} µg/m³", flush=True)
    print(f"    MAE (median): {mae:.3f} µg/m³", flush=True)

    return {
        "method": "quantile_regression",
        "model": "LightGBM",
        "horizon": horizon,
        "alpha": alpha,
        "coverage": round(coverage, 4),
        "avg_width": round(avg_width, 3),
        "mae": round(mae, 3),
        "n_test": len(y_test),
    }


# ══════════════════════════════════════════════════════════════════════
# 3. MC Dropout for GRU
# ══════════════════════════════════════════════════════════════════════


def mc_dropout_gru(df_hybrid, horizon, alpha=ALPHA):
    """Monte Carlo Dropout for GRU prediction intervals."""
    # Lazy import to avoid MPS/LightGBM conflict
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    print(f"\n  [MC Dropout] GRU h={horizon}h ({MC_SAMPLES} samples)...", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Prepare data
    available = [c for c in FEATURE_COLS if c in df_hybrid.columns]
    features = df_hybrid[available].values
    target = df_hybrid[TARGET_COL].values
    is_imputed = (
        df_hybrid["is_imputed"].values
        if "is_imputed" in df_hybrid.columns
        else np.zeros(len(df_hybrid), dtype=bool)
    )

    n = len(features)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Scale
    feat_scaler = StandardScaler()
    features_scaled = np.zeros_like(features)
    features_scaled[:train_end] = feat_scaler.fit_transform(features[:train_end])
    features_scaled[train_end:] = feat_scaler.transform(features[train_end:])

    tgt_scaler = StandardScaler()
    tgt_scaler.fit_transform(target[:train_end].reshape(-1, 1))
    target_all_scaled = tgt_scaler.transform(target.reshape(-1, 1)).flatten()

    # Dataset class
    class SeqDataset(Dataset):
        def __init__(self, feats, tgts, lb, h, mask=None):
            self.feats, self.tgts, self.lb, self.h = feats, tgts, lb, h
            self.indices = []
            for i in range(len(feats) - lb - h):
                ti = i + lb + h - 1
                if ti < len(tgts) and (mask is None or mask[ti]):
                    self.indices.append(i)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i = self.indices[idx]
            x = self.feats[i: i + self.lb]
            y = self.tgts[i + self.lb + self.h - 1]
            return torch.FloatTensor(x), torch.FloatTensor([y])

    # GRU with dropout enabled at inference
    class GRUDropout(nn.Module):
        def __init__(self, input_dim, hidden, layers, drop):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, layers, dropout=drop if layers > 1 else 0, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    # Build datasets
    real_mask = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imputed[i]:
            real_mask[i] = True

    train_ds = SeqDataset(features_scaled, target_all_scaled, LOOKBACK, horizon)
    train_ds.indices = [i for i in train_ds.indices if i + LOOKBACK + horizon - 1 < train_end]
    val_ds = SeqDataset(features_scaled, target_all_scaled, LOOKBACK, horizon)
    val_ds.indices = [i for i in val_ds.indices if train_end <= i + LOOKBACK + horizon - 1 < val_end]
    test_ds = SeqDataset(features_scaled, target_all_scaled, LOOKBACK, horizon, mask=real_mask)
    test_ds.indices = [i for i in test_ds.indices if i + LOOKBACK + horizon - 1 >= val_end]

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    print(f"    Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}", flush=True)

    # Train
    model = GRUDropout(len(available), GRU_HIDDEN, GRU_LAYERS, GRU_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()
    best_val, best_state, patience = float("inf"), None, 0

    for ep in range(100):
        model.train()
        tl = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= max(len(train_loader), 1)

        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                vl += criterion(model(xb.to(device)), yb.to(device)).item()
        vl /= max(len(val_loader), 1)
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            print(f"    Early stop epoch {ep + 1}, val={best_val:.4f}", flush=True)
            break
        if (ep + 1) % 25 == 0:
            print(f"    Epoch {ep + 1}/100: train={tl:.4f}, val={vl:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)

    # MC Dropout inference: enable dropout at test time
    def enable_dropout(m):
        for module in m.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    model.eval()
    enable_dropout(model)

    # Collect MC samples
    all_samples = []
    all_targets = []
    for _ in range(MC_SAMPLES):
        preds_run = []
        tgts_run = []
        with torch.no_grad():
            for xb, yb in test_loader:
                p = model(xb.to(device)).cpu().numpy().flatten()
                preds_run.extend(p)
                if len(all_targets) == 0:
                    tgts_run.extend(yb.numpy().flatten())
        all_samples.append(preds_run)
        if len(all_targets) == 0:
            all_targets = tgts_run

    samples = np.array(all_samples)  # (MC_SAMPLES, n_test)
    y_test_scaled = np.array(all_targets)

    # Inverse transform
    mean_pred_scaled = samples.mean(axis=0)
    lower_scaled = np.quantile(samples, alpha / 2, axis=0)
    upper_scaled = np.quantile(samples, 1 - alpha / 2, axis=0)

    mean_pred = tgt_scaler.inverse_transform(mean_pred_scaled.reshape(-1, 1)).flatten()
    lower = tgt_scaler.inverse_transform(lower_scaled.reshape(-1, 1)).flatten()
    upper = tgt_scaler.inverse_transform(upper_scaled.reshape(-1, 1)).flatten()
    y_test = tgt_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    avg_width = float(np.mean(upper - lower))
    mae = float(np.mean(np.abs(y_test - mean_pred)))

    print(f"    Coverage: {coverage:.1%} (target: {1 - alpha:.0%})", flush=True)
    print(f"    Avg interval width: {avg_width:.2f} µg/m³", flush=True)
    print(f"    MAE (mean): {mae:.3f} µg/m³", flush=True)

    return {
        "method": "mc_dropout",
        "model": "GRU",
        "horizon": horizon,
        "alpha": alpha,
        "mc_samples": MC_SAMPLES,
        "coverage": round(coverage, 4),
        "avg_width": round(avg_width, 3),
        "mae": round(mae, 3),
        "n_test": len(y_test),
    }


# ══════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════


def plot_intervals(results, fig_dir):
    """Plot sample prediction intervals."""
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        if "predictions" not in r:
            continue
        n = len(r["predictions"])
        x = range(n)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(
            x, r["lower"], r["upper"], alpha=0.3,
            color="steelblue",
            label=f"{(1 - r['alpha']) * 100:.0f}% PI",
        )
        ax.plot(x, r["y_true"], "k-", lw=1, label="Actual")
        ax.plot(x, r["predictions"], "r--", lw=1, label="Predicted")
        ax.set_title(f"{r['method']} — {r['model']} h={r['horizon']}h (coverage={r['coverage']:.1%})")
        ax.set_xlabel("Test sample")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = fig_dir / f"pi_{r['method']}_{r['model']}_{r['horizon']}h.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {fname.name}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("PREDICTION INTERVALS — PM2.5 Multi-Horizon", flush=True)
    print("Methods: Conformal Prediction, Quantile Regression, MC Dropout", flush=True)
    print(f"Horizons: {HORIZONS}h | Confidence: {(1 - ALPHA) * 100:.0f}%", flush=True)
    print("=" * 70, flush=True)

    # Prepare data
    print("\n[1/4] Preparing data...", flush=True)
    df_hybrid = prepare_data()
    print(f"  Hybrid data: {len(df_hybrid)} rows", flush=True)

    all_results = []

    for h in HORIZONS:
        print(f"\n{'─' * 60}", flush=True)
        print(f"  HORIZON = {h}h", flush=True)
        print(f"{'─' * 60}", flush=True)

        # ML data
        X, y, train_end, val_end, test_mask, feat_names = prepare_ml_data(df_hybrid, h)

        # 1. Conformal
        r1 = conformal_prediction_lgbm(X, y, train_end, val_end, test_mask, h)
        all_results.append(r1)

        # 2. Quantile
        r2 = quantile_regression_lgbm(X, y, train_end, val_end, test_mask, h)
        all_results.append(r2)

        # 3. MC Dropout GRU
        r3 = mc_dropout_gru(df_hybrid, h)
        all_results.append(r3)

    # Plot
    print("\n[3/4] Plotting...", flush=True)
    plot_intervals(all_results, FIG_DIR)

    # Summary
    print(f"\n{'═' * 70}", flush=True)
    print("[4/4] SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)
    print(f"\n{'Method':<25} {'Model':<12} {'Horizon':<8} {'Coverage':<10} {'Width':<10} {'MAE':<8}", flush=True)
    print("─" * 75, flush=True)
    for r in all_results:
        print(
            f"{r['method']:<25} {r['model']:<12} {r['horizon']}h{'':<5} "
            f"{r['coverage']:.1%}{'':<5} {r['avg_width']:<10.2f} {r['mae']:<8.3f}",
            flush=True,
        )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"prediction_intervals_{ts}.json"

    # Clean for JSON serialization
    clean_results = []
    for r in all_results:
        cr = {k: v for k, v in r.items() if k not in ["predictions", "lower", "upper", "y_true"]}
        clean_results.append(cr)

    with open(out_path, "w") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'═' * 70}", flush=True)
    print(f"COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)", flush=True)
    print(f"{'═' * 70}", flush=True)


if __name__ == "__main__":
    main()
