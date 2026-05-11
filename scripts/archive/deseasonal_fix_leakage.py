"""P0-8 FIX — Re-run STL Residual with TRAIN-ONLY STL fitting.

Issue: Previous run fitted STL on entire series (incl. test) → look-ahead bias.
Fix: Fit STL on train (80%), extract 24h seasonal pattern, apply to val/test.

Usage:
    uv run python scripts/deseasonal_fix_leakage.py 2>&1 | tee research/logs/deseasonal_fix.log
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from torch.utils.data import DataLoader, Dataset

from src.data.cleaner import (
    _clip_physical_bounds,
    _handle_outliers,
    _remove_duplicates,
    _resample,
    _set_datetime_index,
)
from src.data.imputer import impute_missing_data
from src.data.loader import TARGET_COL, load_raw_data
from src.evaluation.metrics import forecast_bias, medae
from src.evaluation.residual_diagnostics import run_residual_diagnostics
from src.features.builder import build_features

warnings.filterwarnings("ignore")

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "deseasonal"
DIAG_DIR = PROJECT_ROOT / "research" / "diagnostics"

HORIZON = 6
LOOKBACK = 72
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Reuse Dataset + GRU from deseasonal_experiment.py ──

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, lookback, horizon, real_mask=None):
        self.features = features
        self.targets = targets
        self.lookback = lookback
        self.horizon = horizon
        self.valid_indices = []
        max_start = len(features) - lookback - horizon
        for i in range(max_start):
            target_idx = i + lookback + horizon - 1
            if target_idx < len(targets):
                if real_mask is not None and not real_mask[target_idx]:
                    continue
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x = self.features[i : i + self.lookback]
        y = self.targets[i + self.lookback + self.horizon - 1]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def train_model(model, train_loader, val_loader, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, n_b = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1
        train_loss /= max(n_b, 1)

        model.eval()
        val_loss, n_v = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(x_batch), y_batch).item()
                n_v += 1
        val_loss /= max(n_v, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS} | Train={train_loss:.4f} Val={val_loss:.4f} patience={patience_counter}/{PATIENCE}", flush=True)

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1} (best={best_val_loss:.4f})", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════
# LEAKAGE-FREE STL Decomposition
# ══════════════════════════════════════════════════════════════════════


def create_leakage_free_stl_target(
    pm25: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create STL residual target WITHOUT leakage.

    Strategy:
        1. Fit STL on TRAIN data only (indices 0..train_end)
        2. Extract the 24h seasonal PATTERN from train
        3. Apply this pattern to val/test by tiling the last full cycle
        4. Trend at val/test: use last known train trend (flat extrapolation)

    Args:
        pm25: Full PM2.5 series.
        train_end: Index where train ends.

    Returns:
        (residual_target, seasonal_trend_component) — same length as pm25.
    """
    n = len(pm25)

    # 1. Fit STL on TRAIN only
    train_series = pd.Series(pm25[:train_end])
    stl = STL(train_series, period=24, robust=True)
    result = stl.fit()

    train_trend = result.trend.values
    train_seasonal = result.seasonal.values
    train_resid = result.resid.values

    # 2. Extract 24h seasonal pattern (average from last 30 days of train)
    #    This is the repeating daily pattern
    last_30d = train_seasonal[-720:]  # last 30 days = 720 hours
    seasonal_pattern = np.zeros(24)
    for h in range(24):
        seasonal_pattern[h] = np.nanmean(last_30d[h::24])

    print(f"    [LeakFree] Train STL fitted on {train_end} points", flush=True)
    print(f"    [LeakFree] Seasonal pattern range: [{seasonal_pattern.min():.2f}, {seasonal_pattern.max():.2f}]", flush=True)

    # 3. Build full seasonal component
    full_seasonal = np.zeros(n)
    full_seasonal[:train_end] = train_seasonal

    # For val/test: tile the 24h pattern
    for i in range(train_end, n):
        hour_of_day = i % 24
        full_seasonal[i] = seasonal_pattern[hour_of_day]

    # 4. Build full trend component
    full_trend = np.zeros(n)
    full_trend[:train_end] = train_trend

    # For val/test: flat extrapolation from last known trend
    last_trend = np.nanmean(train_trend[-168:])  # average of last week
    full_trend[train_end:] = last_trend
    print(f"    [LeakFree] Trend extrapolated: {last_trend:.2f} (last week avg)", flush=True)

    # 5. Compute residual = original - (trend + seasonal)
    seasonal_trend = full_trend + full_seasonal
    residual = pm25 - seasonal_trend

    print(f"    [LeakFree] Residual: mean={np.nanmean(residual):.2f}, std={np.nanstd(residual):.2f}", flush=True)

    return residual, seasonal_trend


# ══════════════════════════════════════════════════════════════════════
# Main Experiment
# ══════════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("P0-8 FIX: STL RESIDUAL — LEAKAGE-FREE RE-RUN", flush=True)
    print(f"Horizon: {HORIZON}h | Lookback: {LOOKBACK}h | Device: {DEVICE}", flush=True)
    print("FIX: STL fitted on TRAIN ONLY, seasonal pattern extrapolated", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Prepare data ──
    print("\n[1/4] Preparing dataset...", flush=True)
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")

    df_hybrid = impute_missing_data(
        df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24,
        knn_neighbors=5, verbose=True,
    )

    is_imputed = df_hybrid["is_imputed"].copy()
    df_for_feat = df_hybrid.drop(columns=["is_imputed"])

    print("  Building v2 features...", flush=True)
    df_feat = build_features(df_for_feat, include_fourier=True, fourier_order=3)
    df_feat["is_imputed"] = is_imputed.reindex(df_feat.index).fillna(False)

    n = len(df_feat)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    is_imp = df_feat["is_imputed"].values

    print(f"  Shape: {df_feat.shape}, train_end={train_end}, val_end={val_end}", flush=True)

    # ── Step 2: Create leakage-free STL target ──
    print("\n[2/4] Creating LEAKAGE-FREE STL target...", flush=True)
    pm25 = df_feat[TARGET_COL].values
    stl_residual, stl_seasonal_trend = create_leakage_free_stl_target(pm25, train_end)

    # ── Step 3: Feature selection + scaling ──
    exclude = {"is_imputed", TARGET_COL}
    exclude.update(c for c in df_feat.columns if c.startswith("target_"))
    feature_cols = [
        c for c in df_feat.columns
        if c not in exclude and df_feat[c].dtype in ("float64", "float32", "int64")
    ]
    features_df = df_feat[feature_cols].fillna(0)

    scaler = StandardScaler()
    features_scaled = np.zeros_like(features_df.values, dtype=np.float32)
    features_scaled[:train_end] = scaler.fit_transform(features_df.values[:train_end])
    features_scaled[train_end:] = scaler.transform(features_df.values[train_end:])
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale STL residual target
    target_safe = np.nan_to_num(stl_residual, nan=0.0)
    target_scaler = StandardScaler()
    target_scaler.fit(target_safe[:train_end].reshape(-1, 1))
    target_all_scaled = target_scaler.transform(target_safe.reshape(-1, 1)).flatten()

    print(f"  Features: {len(feature_cols)} cols", flush=True)

    # Persistence baseline (on ORIGINAL pm25)
    y_true_list, y_persist_list = [], []
    for i in range(val_end, n - HORIZON):
        if is_imp[i + HORIZON]:
            continue
        actual = pm25[i + HORIZON]
        persist = pm25[i]
        if not np.isnan(actual) and not np.isnan(persist):
            y_true_list.append(actual)
            y_persist_list.append(persist)
    y_true_baseline = np.array(y_true_list)
    y_persist = np.array(y_persist_list)
    persist_mae = float(np.mean(np.abs(y_true_baseline - y_persist)))
    print(f"  Persistence {HORIZON}h: MAE={persist_mae:.3f}", flush=True)

    # Datasets
    real_mask_test = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imp[i]:
            real_mask_test[i] = True

    train_ds = TimeSeriesDataset(features_scaled, target_all_scaled, LOOKBACK, HORIZON)
    train_ds.valid_indices = [i for i in train_ds.valid_indices if i + LOOKBACK + HORIZON - 1 < train_end]

    val_ds = TimeSeriesDataset(features_scaled, target_all_scaled, LOOKBACK, HORIZON)
    val_ds.valid_indices = [i for i in val_ds.valid_indices if train_end <= i + LOOKBACK + HORIZON - 1 < val_end]

    test_ds = TimeSeriesDataset(features_scaled, target_all_scaled, LOOKBACK, HORIZON, real_mask=real_mask_test)
    test_ds.valid_indices = [i for i in test_ds.valid_indices if i + LOOKBACK + HORIZON - 1 >= val_end]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)

    # ── Step 3: Train GRU ──
    print(f"\n[3/4] Training GRU_stl_leakfree (h={HORIZON})...", flush=True)
    t0 = time.time()
    input_dim = features_scaled.shape[1]
    model = GRUModel(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} | Input dim: {input_dim}", flush=True)

    model = train_model(model, train_loader, val_loader, "GRU_stl_leakfree")
    train_time = time.time() - t0

    # ── Evaluate ──
    model.eval()
    all_preds, all_targets = [], []
    test_valid_indices_used = []
    batch_start = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            pred = model(x_batch).cpu().numpy().flatten()
            bs = len(pred)
            for j in range(bs):
                all_preds.append(pred[j])
                all_targets.append(y_batch[j].item())
                test_valid_indices_used.append(test_ds.valid_indices[batch_start + j])
            batch_start += bs

    # Inverse transform
    preds_resid = target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    targets_resid = target_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()

    # Add seasonal+trend back
    pred_target_indices = [i + LOOKBACK + HORIZON - 1 for i in test_valid_indices_used]
    seasonal_at_pred = np.array([
        stl_seasonal_trend[idx] if idx < len(stl_seasonal_trend) else stl_seasonal_trend[-1]
        for idx in pred_target_indices
    ])

    preds_original = np.clip(preds_resid + seasonal_at_pred, 0, None)
    targets_original = np.clip(targets_resid + seasonal_at_pred, 0, None)

    # ── Metrics ──
    mae_val = float(np.mean(np.abs(targets_original - preds_original)))
    rmse_val = float(np.sqrt(np.mean((targets_original - preds_original) ** 2)))
    mase = round(mae_val / persist_mae, 4) if persist_mae > 0 else float("inf")
    fb = forecast_bias(targets_original, preds_original)
    med_ae = medae(targets_original, preds_original)

    status = "✅" if mase < 1.0 else "❌"
    print(f"\n  {status} GRU_stl_leakfree h={HORIZON}: MAE={mae_val:.3f}, MASE={mase:.3f}, FB={fb:.4f}, MedAE={med_ae:.3f} ({train_time:.0f}s)", flush=True)

    # Residual diagnostics
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    diag = run_residual_diagnostics(
        y_true=targets_original, y_pred=preds_original,
        model_name="GRU_stl_leakfree", horizon=HORIZON,
        output_dir=str(DIAG_DIR),
    )

    # ── Step 4: Comparison ──
    print(f"\n{'═' * 70}", flush=True)
    print("[4/4] COMPARISON: Leaky vs Leak-Free STL", flush=True)
    print(f"{'═' * 70}", flush=True)
    print(f"  STL (LEAKY — full data):     MASE = 0.507 ⚠️ INVALID", flush=True)
    print(f"  STL (LEAK-FREE — train only): MASE = {mase:.3f} {'✅' if mase < 1.0 else '❌'}", flush=True)
    print(f"  seasonal_diff (y-y[t-24]):    MASE = 0.903 ✅ VALID", flush=True)
    print(f"  raw (no transform):           MASE = 0.731 ✅ VALID", flush=True)
    print(f"  GRU v2+log (reference):       MASE = 0.692 ✅ REFERENCE", flush=True)

    delta = ((mase - 0.692) / 0.692) * 100
    print(f"\n  vs v2+log reference: {delta:+.1f}%", flush=True)

    if mase < 0.692:
        print("  🎉 STL leak-free BEATS v2+log!", flush=True)
    elif mase < 0.731:
        print("  📊 STL leak-free better than raw but worse than v2+log", flush=True)
    else:
        print("  📊 STL leak-free ≈ raw → deseasonalizing doesn't help with Fourier features", flush=True)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "meta": {
            "description": "P0-8 FIX: STL Residual with TRAIN-ONLY fitting (no leakage)",
            "fix": "STL fitted on train only, seasonal pattern extrapolated to val/test",
            "timestamp": datetime.now().isoformat(),
        },
        "result": {
            "model": "GRU_stl_leakfree",
            "horizon": HORIZON,
            "mae": round(mae_val, 4),
            "rmse": round(rmse_val, 4),
            "mase": mase,
            "forecast_bias": round(fb, 4),
            "medae": round(med_ae, 4),
            "params": n_params,
            "train_time_s": round(train_time, 1),
            "n_test": len(all_preds),
            "persistence_mae": round(persist_mae, 4),
            "diagnostics_verdict": diag["verdict"],
        },
    }
    out_path = OUTPUT_DIR / f"stl_leakfree_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}", flush=True)

    total = time.time() - t_start
    print(f"\nCOMPLETE — Total: {total:.0f}s ({total / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
