"""Model Export — Convert best models to portable formats.

Exports:
  1. GRU → TorchScript (.pt) — cross-platform inference
  2. LightGBM → ONNX (.onnx) — if onnxmltools available, else native .txt

Usage:
    uv run python scripts/export_models.py 2>&1 | tee research/logs/export_models.log
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
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"
HORIZONS = [1, 6, 24]
LOOKBACK = 72
GRU_HIDDEN = 64
GRU_LAYERS = 2
GRU_DROPOUT = 0.2
FEATURE_COLS = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]


# ══════════════════════════════════════════════════════════════════════
# Data (reuse from pipeline)
# ══════════════════════════════════════════════════════════════════════


def prepare_data():
    """Load and prepare hybrid data."""
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")
    df_hybrid = impute_missing_data(
        df, strategy="hybrid",
        max_gap_interp=6, max_gap_ml=24, knn_neighbors=5, verbose=True,
    )
    return df_hybrid


# ══════════════════════════════════════════════════════════════════════
# 1. Export GRU → TorchScript
# ══════════════════════════════════════════════════════════════════════


def export_gru(df_hybrid, horizon):
    """Train GRU and export to TorchScript."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    print(f"\n  [GRU → TorchScript] h={horizon}h...", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    available = [c for c in FEATURE_COLS if c in df_hybrid.columns]
    features = df_hybrid[available].values
    target = df_hybrid[TARGET_COL].values
    # is_imputed tracked but not used in export (test-on-real is eval-only)

    n = len(features)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Scale
    feat_scaler = StandardScaler()
    features_scaled = np.zeros_like(features)
    features_scaled[:train_end] = feat_scaler.fit_transform(features[:train_end])
    features_scaled[train_end:] = feat_scaler.transform(features[train_end:])

    tgt_scaler = StandardScaler()
    tgt_scaler.fit(target[:train_end].reshape(-1, 1))
    target_all_scaled = tgt_scaler.transform(target.reshape(-1, 1)).flatten()

    # Simple dataset
    class SeqDataset(Dataset):
        def __init__(self, feats, tgts, lb, h):
            self.feats, self.tgts, self.lb, self.h = feats, tgts, lb, h
            self.indices = [i for i in range(len(feats) - lb - h) if i + lb + h - 1 < len(tgts)]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i = self.indices[idx]
            x = torch.FloatTensor(self.feats[i:i + self.lb])
            y = torch.FloatTensor([self.tgts[i + self.lb + self.h - 1]])
            return x, y

    # GRU model
    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden, layers, drop):
            super().__init__()
            drop_gru = drop if layers > 1 else 0
            self.gru = nn.GRU(
                input_dim, hidden, layers,
                dropout=drop_gru, batch_first=True,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Dropout(drop), nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    # Build datasets
    train_ds = SeqDataset(features_scaled[:train_end], target_all_scaled[:train_end], LOOKBACK, horizon)
    val_ds = SeqDataset(features_scaled[:val_end], target_all_scaled[:val_end], LOOKBACK, horizon)
    val_ds.indices = [i for i in val_ds.indices if i + LOOKBACK + horizon - 1 >= train_end]

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    print(f"    Train={len(train_ds)}, Val={len(val_ds)}", flush=True)

    # Train
    model = GRUModel(len(available), GRU_HIDDEN, GRU_LAYERS, GRU_DROPOUT).to(device)
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
            print(f"    Early stop epoch {ep + 1}", flush=True)
            break
        if (ep + 1) % 25 == 0:
            print(f"    Epoch {ep + 1}/100 val={vl:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)

    # Export to TorchScript (CPU for portability)
    model = model.cpu()
    model.eval()
    example_input = torch.randn(1, LOOKBACK, len(available))
    scripted = torch.jit.trace(model, example_input)

    out_path = EXPORT_DIR / f"gru_{horizon}h.pt"
    scripted.save(str(out_path))
    print(f"    ✅ Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # Also save scaler params
    scaler_path = EXPORT_DIR / f"scalers_{horizon}h.json"
    scaler_info = {
        "feature_scaler_mean": feat_scaler.mean_.tolist(),
        "feature_scaler_scale": feat_scaler.scale_.tolist(),
        "target_scaler_mean": float(tgt_scaler.mean_[0]),
        "target_scaler_scale": float(tgt_scaler.scale_[0]),
        "features": available,
        "lookback": LOOKBACK,
        "horizon": horizon,
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_info, f, indent=2, ensure_ascii=False)
    print(f"    ✅ Scalers: {scaler_path.name}", flush=True)

    return str(out_path)


# ══════════════════════════════════════════════════════════════════════
# 2. Export LightGBM → Native format
# ══════════════════════════════════════════════════════════════════════


def export_lgbm(df_hybrid, horizon):
    """Train and export LightGBM to native .txt format."""
    print(f"\n  [LightGBM → Native] h={horizon}h...", flush=True)

    df_feat = build_features(df_hybrid)

    feature_cols = [c for c in df_feat.columns if c not in [TARGET_COL, "is_imputed"]]
    X = df_feat[feature_cols].values
    y = df_feat[TARGET_COL].values

    y_target = pd.Series(y).shift(-horizon).values
    valid = ~np.isnan(y_target) & ~np.isnan(X).any(axis=1)
    X, y_target = X[valid], y_target[valid]

    train_end = int(len(X) * 0.8)
    X_train, y_train = X[:train_end], y_target[:train_end]

    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.6,
        min_child_samples=30, reg_alpha=0.05, reg_lambda=0.5,
        num_leaves=64, verbose=-1, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Native format (portable)
    out_path = EXPORT_DIR / f"lgbm_{horizon}h.txt"
    model.booster_.save_model(str(out_path))
    print(f"    ✅ Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # Feature names
    names_path = EXPORT_DIR / f"lgbm_{horizon}h_features.json"
    with open(names_path, "w") as f:
        json.dump({"features": feature_cols, "horizon": horizon, "n_features": len(feature_cols)}, f, indent=2)
    print(f"    ✅ Features: {names_path.name}", flush=True)

    return str(out_path)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("MODEL EXPORT — GRU (TorchScript) + LightGBM (Native)", flush=True)
    print(f"Horizons: {HORIZONS}h", flush=True)
    print("=" * 70, flush=True)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    df_hybrid = prepare_data()

    exported = []
    for h in HORIZONS:
        print(f"\n{'─' * 60}", flush=True)
        print(f"  HORIZON = {h}h", flush=True)
        print(f"{'─' * 60}", flush=True)

        # GRU → TorchScript
        gru_path = export_gru(df_hybrid, h)
        exported.append({"model": "GRU", "format": "TorchScript", "horizon": h, "path": gru_path})

        # LightGBM → Native
        lgbm_path = export_lgbm(df_hybrid, h)
        exported.append({"model": "LightGBM", "format": "Native (.txt)", "horizon": h, "path": lgbm_path})

    # Summary
    print(f"\n{'═' * 70}", flush=True)
    print("EXPORT SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)
    for e in exported:
        size = Path(e["path"]).stat().st_size / 1024
        name = Path(e["path"]).name
        print(
            f"  {e['model']:<12} {e['format']:<18} "
            f"{e['horizon']}h → {name} ({size:.1f} KB)",
            flush=True,
        )

    # Save manifest
    manifest = EXPORT_DIR / "manifest.json"
    with open(manifest, "w") as f:
        json.dump({
            "exported_at": datetime.now().isoformat(),
            "models": exported,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Manifest: {manifest}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'═' * 70}", flush=True)
    print(f"COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)", flush=True)
    print(f"{'═' * 70}", flush=True)


if __name__ == "__main__":
    main()
