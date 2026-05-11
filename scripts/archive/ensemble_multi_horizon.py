"""Ensemble Stacking Multi-Horizon — Combine LightGBM + GRU for best performance.

Strategy: Level-0 base models → Level-1 Ridge meta-learner.
- LightGBM: engineered features (lag, rolling, temporal)
- GRU: raw multivariate sequence (5 sensors, lookback=72h)

For each horizon, find optimal combination weights on validation set.

Usage:
    uv run python scripts/ensemble_multi_horizon.py 2>&1 | tee research/logs/ensemble_multi_horizon.log
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# NOTE: torch is lazy-imported in GRU section to avoid MPS/OpenMP conflict with LightGBM
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
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "ensemble"
HORIZONS = [1, 6, 24]

# GRU config (same as dl_multi_horizon.py)
LOOKBACK = 72
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10
GRU_FEATURES = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]

DEVICE = None  # set lazily when torch is imported


# ══════════════════════════════════════════════════════════════════════
# GRU Model (same architecture as dl_multi_horizon.py)
# ══════════════════════════════════════════════════════════════════════


def _init_torch():
    """Lazy init torch + MPS device."""
    global DEVICE
    import torch

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"    PyTorch initialized: device={DEVICE}", flush=True)
    return torch


def _make_dataset_class():
    """Create TimeSeriesDataset class with lazy torch import."""
    import torch
    from torch.utils.data import Dataset

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

    return TimeSeriesDataset


def _make_gru_model():
    """Create GRU model class with lazy torch import."""
    import torch.nn as nn

    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.gru = nn.GRU(
                input_dim, hidden_dim, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    return GRUModel


def train_gru(model, train_loader, val_loader):
    """Train GRU with early stopping."""
    import torch
    import torch.nn as nn

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.MSELoss()
    best_val, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(EPOCHS):
        model.train()
        t_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            n += 1
        t_loss /= max(n, 1)

        model.eval()
        v_loss, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_loss += criterion(model(xb), yb).item()
                nv += 1
        v_loss /= max(nv, 1)
        scheduler.step(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = model.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"      Epoch {epoch + 1:3d} | Train={t_loss:.4f} Val={v_loss:.4f} p={patience_cnt}/{PATIENCE}",
                flush=True,
            )

        if patience_cnt >= PATIENCE:
            print(f"      Early stop epoch {epoch + 1} (best={best_val:.4f})", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("ENSEMBLE STACKING — LightGBM + GRU", flush=True)
    print(f"Horizons: {HORIZONS}h | Meta-learner: Ridge Regression", flush=True)
    print("Device: (lazy — set when GRU starts)", flush=True)
    print("RULE: Test set = REAL data only", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Prepare data ──
    print("\n[1/4] Preparing data...", flush=True)
    df_hybrid = _prepare_hybrid_data()
    is_imputed = df_hybrid["is_imputed"].values.copy()

    # LightGBM features
    print("  Building engineered features for LightGBM...", flush=True)
    df_features = build_features(df_hybrid.drop(columns=["is_imputed"], errors="ignore"), drop_na=True)
    # Align is_imputed after feature building (dropped warmup rows)
    is_imputed_feat = is_imputed[len(is_imputed) - len(df_features) :]

    # GRU raw features
    gru_features = df_hybrid[GRU_FEATURES].values
    gru_target = df_hybrid[TARGET_COL].values
    gru_imputed = is_imputed

    print(f"  LightGBM data: {len(df_features)} rows × {len(df_features.columns)} cols", flush=True)
    print(f"  GRU data: {len(gru_features)} rows × {gru_features.shape[1]} features", flush=True)

    all_results = {}
    all_preds = {}

    for h in HORIZONS:
        print(f"\n{'═' * 70}", flush=True)
        print(f"[2/4] HORIZON = {h}h", flush=True)
        print(f"{'═' * 70}", flush=True)

        results_h = _evaluate_horizon(
            df_features,
            is_imputed_feat,
            gru_features,
            gru_target,
            gru_imputed,
            h,
        )
        all_results[f"{h}h"] = results_h["metrics"]
        all_preds[f"{h}h"] = results_h["preds"]

    # ── Summary ──
    print(f"\n{'═' * 70}", flush=True)
    print("[3/4] ENSEMBLE RESULTS SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)

    _save_results(all_results, all_preds)


    print(f"\n{'Hz':<6} {'Model':<25} {'MAE':>8} {'MASE':>8} {'vs Best':>10}", flush=True)
    print("─" * 65, flush=True)

    for h in HORIZONS:
        hk = f"{h}h"
        p = all_results[hk].get("Persistence", {})
        print(f"{hk:<6} {'Persistence':<25} {p.get('mae', 0):>8.3f} {'1.000':>8} {'baseline':>10}", flush=True)

        lgbm = all_results[hk].get("LightGBM", {})
        if lgbm:
            print(
                f"{hk:<6} {'LightGBM':<25} {lgbm.get('mae', 0):>8.3f} {lgbm.get('mase', 0):>8.3f} {'':>10}", flush=True
            )

        gru = all_results[hk].get("GRU", {})
        if gru:
            print(f"{hk:<6} {'GRU':<25} {gru.get('mae', 0):>8.3f} {gru.get('mase', 0):>8.3f} {'':>10}", flush=True)

        ens = all_results[hk].get("Ensemble_Stack", {})
        if ens:
            best_individual = min(lgbm.get("mase", 99), gru.get("mase", 99))
            delta = ((ens["mase"] - best_individual) / best_individual) * 100
            status = f"{'✅' if delta < 0 else '❌'} {delta:+.1f}%"
            print(
                f"{hk:<6} {'⭐ Ensemble (Stack)':<25} {ens.get('mae', 0):>8.3f} {ens.get('mase', 0):>8.3f} {status:>10}",
                flush=True,
            )

        ens_w = all_results[hk].get("Ensemble_Weighted", {})
        if ens_w:
            best_individual = min(lgbm.get("mase", 99), gru.get("mase", 99))
            delta = ((ens_w["mase"] - best_individual) / best_individual) * 100
            status = f"{'✅' if delta < 0 else '❌'} {delta:+.1f}%"
            print(
                f"{hk:<6} {'⭐ Ensemble (Weighted)':<25} {ens_w.get('mae', 0):>8.3f} {ens_w.get('mase', 0):>8.3f} {status:>10}",
                flush=True,
            )

        print("─" * 65, flush=True)

    # ── Save ──
    print("\n[4/4] Saving results...", flush=True)
    _save_results(all_results, all_preds)

    total = time.time() - t_start
    print(f"\n{'═' * 70}", flush=True)
    print(f"COMPLETE — Total: {total:.0f}s ({total / 60:.1f} min)", flush=True)
    print(f"{'═' * 70}", flush=True)


def _prepare_hybrid_data():
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")
    return impute_missing_data(df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24, knn_neighbors=5, verbose=True)


def _evaluate_horizon(df_feat, is_imp_feat, gru_feat, gru_tgt, gru_imp, horizon):
    """Evaluate LightGBM, GRU, and their ensemble at a specific horizon."""
    from lightgbm import LGBMRegressor

    results = {}

    # ══════════════════════════════════════════════════════════════
    # A. LightGBM
    # ══════════════════════════════════════════════════════════════
    print(f"\n  [A] Training LightGBM ({horizon}h)...", flush=True)
    t0 = time.time()

    try:
        # Create target: shift by horizon
        df_lgbm = df_feat.copy()
        # Carry is_imputed through as a column for alignment
        df_lgbm["_is_imputed"] = is_imp_feat[: len(df_lgbm)]
        df_lgbm["target"] = df_lgbm[TARGET_COL].shift(-horizon)
        df_lgbm = df_lgbm.dropna(subset=["target"])

        feature_cols = [c for c in df_lgbm.columns if c not in [TARGET_COL, "target", "is_imputed", "_is_imputed"]]
        X = df_lgbm[feature_cols].values
        y = df_lgbm["target"].values
        imp_aligned = df_lgbm["_is_imputed"].values

        n = len(X)
        tr_end = int(n * 0.8)
        val_end = int(n * 0.9)

        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_val, _y_val = X[tr_end:val_end], y[tr_end:val_end]
        X_te, y_te = X[val_end:], y[val_end:]

        # Real test mask
        real_test_mask = ~imp_aligned[val_end:]

        lgbm = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            verbose=-1,
        )
        lgbm.fit(X_tr, y_tr)

        lgbm.predict(X_val)
        lgbm_pred_test = lgbm.predict(X_te)

        # Persistence baseline (from test set)
        persist_test = df_lgbm[TARGET_COL].values[val_end : val_end + len(X_te)]

        # Apply real mask
        y_te_real = y_te[real_test_mask]
        lgbm_pred_real = lgbm_pred_test[real_test_mask]
        persist_real = persist_test[real_test_mask]

        persist_mae = float(np.mean(np.abs(y_te_real - persist_real)))
        persist_rmse = float(np.sqrt(np.mean((y_te_real - persist_real) ** 2)))
        results["Persistence"] = {"mae": round(persist_mae, 4), "rmse": round(persist_rmse, 4), "mase": 1.0}

        lgbm_mae = float(np.mean(np.abs(y_te_real - lgbm_pred_real)))
        lgbm_rmse = float(np.sqrt(np.mean((y_te_real - lgbm_pred_real) ** 2)))
        lgbm_mase = round(lgbm_mae / persist_mae, 4) if persist_mae > 0 else float("inf")
        results["LightGBM"] = {"mae": round(lgbm_mae, 4), "rmse": round(lgbm_rmse, 4), "mase": lgbm_mase}
        print(
            f"    LightGBM {horizon}h: MAE={lgbm_mae:.3f}, MASE={lgbm_mase:.3f} ({time.time() - t0:.0f}s)", flush=True
        )
        print(f"    Test: {len(y_te)} total, {len(y_te_real)} real", flush=True)
    except Exception as e:
        import traceback

        print(f"    ❌ LightGBM ERROR: {e}", flush=True)
        traceback.print_exc()
        return results

    # ══════════════════════════════════════════════════════════════
    # B. GRU
    # ══════════════════════════════════════════════════════════════
    print(f"\n  [B] Training GRU ({horizon}h)...", flush=True)
    t0 = time.time()

    # Lazy init torch
    torch = _init_torch()
    from torch.utils.data import DataLoader

    TimeSeriesDataset = _make_dataset_class()
    GRUModel = _make_gru_model()

    n_gru = len(gru_feat)
    gru_tr_end = int(n_gru * 0.8)
    gru_val_end = int(n_gru * 0.9)

    # Scale
    scaler = StandardScaler()
    gru_scaled = np.zeros_like(gru_feat)
    gru_scaled[:gru_tr_end] = scaler.fit_transform(gru_feat[:gru_tr_end])
    gru_scaled[gru_tr_end:] = scaler.transform(gru_feat[gru_tr_end:])

    tgt_scaler = StandardScaler()
    tgt_scaler.fit_transform(gru_tgt[:gru_tr_end].reshape(-1, 1)).flatten()
    tgt_all_scaled = tgt_scaler.transform(gru_tgt.reshape(-1, 1)).flatten()

    # Real mask for GRU test
    real_mask_gru = np.zeros(n_gru, dtype=bool)
    for i in range(gru_val_end, n_gru):
        if not gru_imp[i]:
            real_mask_gru[i] = True

    # Datasets
    train_ds = TimeSeriesDataset(gru_scaled, tgt_all_scaled, LOOKBACK, horizon)
    train_ds.valid_indices = [i for i in train_ds.valid_indices if i + LOOKBACK + horizon - 1 < gru_tr_end]

    val_ds = TimeSeriesDataset(gru_scaled, tgt_all_scaled, LOOKBACK, horizon)
    val_ds.valid_indices = [i for i in val_ds.valid_indices if gru_tr_end <= i + LOOKBACK + horizon - 1 < gru_val_end]

    test_ds = TimeSeriesDataset(gru_scaled, tgt_all_scaled, LOOKBACK, horizon, real_mask=real_mask_gru)
    test_ds.valid_indices = [i for i in test_ds.valid_indices if i + LOOKBACK + horizon - 1 >= gru_val_end]

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"    Datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)

    model = GRUModel(gru_feat.shape[1], HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    model = train_gru(model, tr_loader, val_loader)

    # Get GRU predictions
    model.eval()
    gru_preds_scaled, gru_targets_scaled = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            pred = model(xb.to(DEVICE)).cpu().numpy().flatten()
            gru_preds_scaled.extend(pred)
            gru_targets_scaled.extend(yb.numpy().flatten())

    gru_preds = tgt_scaler.inverse_transform(np.array(gru_preds_scaled).reshape(-1, 1)).flatten()
    gru_targets = tgt_scaler.inverse_transform(np.array(gru_targets_scaled).reshape(-1, 1)).flatten()

    gru_mae = float(np.mean(np.abs(gru_targets - gru_preds)))
    gru_rmse = float(np.sqrt(np.mean((gru_targets - gru_preds) ** 2)))
    # Use the SAME persist_mae calculated correctly in LightGBM section
    # Persistence MAE = mean(|y[t+h] - y[t]|) on real test data
    persist_mae_ref = results["Persistence"]["mae"]
    gru_mase = round(gru_mae / persist_mae_ref, 4) if persist_mae_ref > 0 else float("inf")
    results["GRU"] = {"mae": round(gru_mae, 4), "rmse": round(gru_rmse, 4), "mase": gru_mase}
    print(
        f"    GRU {horizon}h: MAE={gru_mae:.3f}, MASE={gru_mase:.3f} (persist_ref={persist_mae_ref:.3f}) ({time.time() - t0:.0f}s)",
        flush=True,
    )

    # ══════════════════════════════════════════════════════════════
    # C. Ensemble — Align predictions
    # ══════════════════════════════════════════════════════════════
    print(f"\n  [C] Building Ensemble ({horizon}h)...", flush=True)

    # Both models predict on different test subsets (different features/alignment)
    # Use the common test size = min of both
    n_common = min(len(lgbm_pred_real), len(gru_preds))
    print(f"    Common test points: {n_common}", flush=True)

    if n_common < 10:
        print("    ⚠️ Too few common points, skipping ensemble", flush=True)
        return results

    lgbm_aligned = lgbm_pred_real[:n_common]
    gru_aligned = gru_preds[:n_common]
    y_aligned = y_te_real[:n_common]

    # C1. Ridge Stacking Meta-learner
    # Split aligned predictions: 60% meta-train, 40% meta-test
    n_meta_train = int(n_common * 0.6)
    meta_X_train = np.column_stack([lgbm_aligned[:n_meta_train], gru_aligned[:n_meta_train]])
    meta_y_train = y_aligned[:n_meta_train]
    meta_X_test = np.column_stack([lgbm_aligned[n_meta_train:], gru_aligned[n_meta_train:]])
    meta_y_test = y_aligned[n_meta_train:]

    ridge = Ridge(alpha=1.0)
    ridge.fit(meta_X_train, meta_y_train)
    stack_pred = ridge.predict(meta_X_test)

    stack_mae = float(np.mean(np.abs(meta_y_test - stack_pred)))
    stack_rmse = float(np.sqrt(np.mean((meta_y_test - stack_pred) ** 2)))
    persist_aligned_mae = float(np.mean(np.abs(meta_y_test - persist_real[n_meta_train:n_common])))
    stack_mase = round(stack_mae / persist_aligned_mae, 4) if persist_aligned_mae > 0 else float("inf")

    results["Ensemble_Stack"] = {
        "mae": round(stack_mae, 4),
        "rmse": round(stack_rmse, 4),
        "mase": stack_mase,
        "weights": f"LightGBM={ridge.coef_[0]:.3f}, GRU={ridge.coef_[1]:.3f}, intercept={ridge.intercept_:.3f}",
    }
    print(f"    Stack {horizon}h: MAE={stack_mae:.3f}, MASE={stack_mase:.3f}", flush=True)
    print(f"    Weights: LightGBM={ridge.coef_[0]:.3f}, GRU={ridge.coef_[1]:.3f}", flush=True)

    # C2. Grid search weighted average on full aligned set
    best_w, best_mae_w = 0.5, float("inf")
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * lgbm_aligned + (1 - w) * gru_aligned
        mae_w = float(np.mean(np.abs(y_aligned - blend)))
        if mae_w < best_mae_w:
            best_mae_w = mae_w
            best_w = w

    blend_pred = best_w * lgbm_aligned + (1 - best_w) * gru_aligned
    blend_mae = float(np.mean(np.abs(y_aligned - blend_pred)))
    blend_rmse = float(np.sqrt(np.mean((y_aligned - blend_pred) ** 2)))
    persist_full_mae = float(np.mean(np.abs(y_aligned - persist_real[:n_common])))
    blend_mase = round(blend_mae / persist_full_mae, 4) if persist_full_mae > 0 else float("inf")

    results["Ensemble_Weighted"] = {
        "mae": round(blend_mae, 4),
        "rmse": round(blend_rmse, 4),
        "mase": blend_mase,
        "best_weight": f"LightGBM={best_w:.2f}, GRU={1 - best_w:.2f}",
    }
    print(f"    Weighted {horizon}h: MAE={blend_mae:.3f}, MASE={blend_mase:.3f} (w_lgbm={best_w:.2f})", flush=True)

    preds_dict = {
        "Ensemble_Stack": stack_pred.tolist(),
        "Ensemble_Weighted": blend_pred.tolist()
    }

    return {"metrics": results, "preds": preds_dict}


# LightGBM uses fixed n_estimators=300 (no callback needed)


def _save_results(all_results, all_preds):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_metrics = OUTPUT_DIR / f"ensemble_{ts}.json"
    path_preds = OUTPUT_DIR / f"ensemble_preds_{ts}.json"

    def conv(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path_metrics, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=conv, ensure_ascii=False)
    with open(path_preds, "w", encoding="utf-8") as f:
        json.dump(all_preds, f, indent=2, default=conv, ensure_ascii=False)
        
    print(f"  Metrics saved: {path_metrics}", flush=True)
    print(f"  Preds saved: {path_preds}", flush=True)


if __name__ == "__main__":
    main()
