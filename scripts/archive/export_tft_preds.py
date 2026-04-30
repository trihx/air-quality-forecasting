"""Export TFT predictions for AVP dashboard integration.

Trains the Simplified TFT model at each horizon (1h, 6h, 24h) and saves
the prediction arrays as .npy files for precompute_avp.py to merge into cache.

IMPORTANT: This script reuses the EXACT same data pipeline and model architecture
as tft_multi_horizon.py to ensure metric consistency.

Usage:
    export OMP_NUM_THREADS=1
    uv run python scripts/export_tft_preds.py 2>&1 | tee research/logs/export_tft_preds.log
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "research" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Same config as tft_multi_horizon.py
HORIZONS = [1, 6, 24]
LOOKBACK = 72
HIDDEN_DIM = 32
NUM_HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 15

TEMPORAL_COLS = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]
STATIC_COLS_MAP = {
    "hour_sin": lambda idx: np.sin(2 * np.pi * idx.hour / 24),
    "hour_cos": lambda idx: np.cos(2 * np.pi * idx.hour / 24),
    "dow_sin": lambda idx: np.sin(2 * np.pi * idx.dayofweek / 7),
    "dow_cos": lambda idx: np.cos(2 * np.pi * idx.dayofweek / 7),
}


def prepare_data():
    """Load → clean → impute → add static features (same as tft_multi_horizon.py)."""
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
        verbose=True,
    )

    idx = df_hybrid.index
    for name, func in STATIC_COLS_MAP.items():
        df_hybrid[name] = func(idx)

    return df_hybrid


def export_horizon(df_hybrid, horizon: int, components: dict) -> dict | None:
    """Train TFT and export aligned predictions for a single horizon."""
    import torch
    import torch.nn as nn

    from src.data.loader import TARGET_COL

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"  TFT Export: Horizon {horizon}h | Device: {device}")
    print(f"{'=' * 60}", flush=True)

    available_temporal = [c for c in TEMPORAL_COLS if c in df_hybrid.columns]
    static_cols = list(STATIC_COLS_MAP.keys())

    temporal = df_hybrid[available_temporal].values
    static = df_hybrid[static_cols].values
    target = df_hybrid[TARGET_COL].values
    is_imputed = (
        df_hybrid["is_imputed"].values if "is_imputed" in df_hybrid.columns
        else np.zeros(len(df_hybrid), dtype=bool)
    )

    n = len(temporal)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Scale (fit on train only — anti-leakage)
    temp_scaler = StandardScaler()
    temporal_scaled = np.zeros_like(temporal)
    temporal_scaled[:train_end] = temp_scaler.fit_transform(temporal[:train_end])
    temporal_scaled[train_end:] = temp_scaler.transform(temporal[train_end:])

    tgt_scaler = StandardScaler()
    tgt_scaler.fit(target[:train_end].reshape(-1, 1))
    target_scaled = tgt_scaler.transform(target.reshape(-1, 1)).flatten()

    # Real-only mask for test
    real_mask = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imputed[i]:
            real_mask[i] = True

    # Build datasets
    TFTDataset = components["TFTDataset"]
    DataLoader = components["DataLoader"]

    train_ds = TFTDataset(temporal_scaled, static, target_scaled, LOOKBACK, horizon)
    train_ds.indices = [i for i in train_ds.indices if i + LOOKBACK + horizon - 1 < train_end]

    val_ds = TFTDataset(temporal_scaled, static, target_scaled, LOOKBACK, horizon)
    val_ds.indices = [i for i in val_ds.indices if train_end <= i + LOOKBACK + horizon - 1 < val_end]

    test_ds = TFTDataset(temporal_scaled, static, target_scaled, LOOKBACK, horizon, mask=real_mask)
    test_ds.indices = [i for i in test_ds.indices if i + LOOKBACK + horizon - 1 >= val_end]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} (real only)", flush=True)

    if len(train_ds) < BATCH_SIZE:
        print("  ⚠️ Too few samples, skipping", flush=True)
        return None

    # Build + Train model
    SimplifiedTFT = components["SimplifiedTFT"]
    model = SimplifiedTFT(
        temporal_dim=len(available_temporal),
        static_dim=len(static_cols),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  TFT Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    criterion = nn.MSELoss()
    best_val, best_state, patience_counter = float("inf"), None, 0

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        tl = 0
        for t_batch, s_batch, y_batch in train_loader:
            t_batch = t_batch.to(device)
            s_batch = s_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(t_batch, s_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= max(len(train_loader), 1)

        model.eval()
        vl = 0
        with torch.no_grad():
            for t_batch, s_batch, y_batch in val_loader:
                vl += criterion(
                    model(t_batch.to(device), s_batch.to(device)),
                    y_batch.to(device),
                ).item()
        vl /= max(len(val_loader), 1)
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (ep + 1) % 20 == 0 or ep == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"    Epoch {ep + 1:3d}/{EPOCHS} "
                f"train={tl:.4f} val={vl:.4f} "
                f"lr={lr:.1e} patience={patience_counter}/{PATIENCE}",
                flush=True,
            )

        if patience_counter >= PATIENCE:
            print(f"    Early stop epoch {ep + 1} (best val={best_val:.4f})", flush=True)
            break

    train_time = time.time() - t0

    if best_state:
        model.load_state_dict(best_state)

    # ── Predict on test set ──
    model.eval()
    all_preds_scaled = []
    all_targets_scaled = []
    with torch.no_grad():
        for t_batch, s_batch, y_batch in test_loader:
            pred = model(t_batch.to(device), s_batch.to(device)).cpu().numpy().flatten()
            all_preds_scaled.extend(pred)
            all_targets_scaled.extend(y_batch.numpy().flatten())

    # Inverse transform
    preds = tgt_scaler.inverse_transform(np.array(all_preds_scaled).reshape(-1, 1)).flatten()
    targets = tgt_scaler.inverse_transform(np.array(all_targets_scaled).reshape(-1, 1)).flatten()

    mae = float(np.mean(np.abs(targets - preds)))
    rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
    print(f"  TFT {horizon}h: MAE={mae:.3f} | RMSE={rmse:.3f} | n={len(preds)} ({train_time:.0f}s)", flush=True)

    # ── Align predictions with AVP cache test indices ──
    # AVP cache uses: for i in range(val_end, n - horizon): if not is_imputed[i + horizon]
    # TFT uses: TFTDataset with lookback window + real_mask
    # We need to map TFT predictions back to the AVP cache index order.

    # Rebuild the exact AVP cache test index list
    avp_test_indices = []
    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        avp_test_indices.append(i)

    # TFT test indices (from dataset)
    tft_test_target_indices = set()
    for ds_idx in test_ds.indices:
        target_idx = ds_idx + LOOKBACK + horizon - 1
        tft_test_target_indices.add(target_idx)

    # Create aligned prediction array (None where TFT can't predict due to lookback)
    aligned_preds = []
    pred_iter = iter(preds)
    for avp_idx in avp_test_indices:
        if avp_idx in tft_test_target_indices:
            try:
                aligned_preds.append(float(next(pred_iter)))
            except StopIteration:
                aligned_preds.append(None)
        else:
            aligned_preds.append(None)

    n_filled = sum(1 for p in aligned_preds if p is not None)
    print(f"  Aligned: {n_filled}/{len(aligned_preds)} predictions mapped", flush=True)

    # Save
    out_path = CACHE_DIR / f"tft_preds_{horizon}h.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": "TFT",
            "horizon": horizon,
            "predictions": aligned_preds,
            "n_total": len(aligned_preds),
            "n_valid": n_filled,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "train_time_s": round(train_time, 1),
        }, f)
    print(f"  ✅ Saved: {out_path}", flush=True)

    return {"mae": mae, "rmse": rmse, "n": n_filled}


def main():
    t_start = time.time()
    print("=" * 60)
    print("  EXPORT TFT PREDICTIONS FOR AVP DASHBOARD")
    print(f"  Horizons: {HORIZONS}")
    print("=" * 60, flush=True)

    # Import TFT model components (lazy import torch)
    print("\n[1/3] Building TFT components...", flush=True)
    # Reuse the EXACT model architecture from tft_multi_horizon.py
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from tft_multi_horizon import _build_model_components
    components = _build_model_components()

    # Prepare data
    print("\n[2/3] Preparing data...", flush=True)
    df_hybrid = prepare_data()
    print(f"  Data: {len(df_hybrid)} rows", flush=True)

    # Export for each horizon
    print("\n[3/3] Training + exporting predictions...", flush=True)
    results = {}
    for h in HORIZONS:
        r = export_horizon(df_hybrid, h, components)
        if r:
            results[f"{h}h"] = r

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  All done in {total:.0f}s ({total / 60:.1f} min)")
    print(f"  Files saved in: {CACHE_DIR}")
    for h in HORIZONS:
        p = CACHE_DIR / f"tft_preds_{h}h.json"
        if p.exists():
            print(f"    ✅ {p.name} ({p.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"    ❌ {p.name} (not created)")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
