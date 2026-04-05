"""LSTM/GRU Multi-Horizon Evaluation — Level 4 Deep Learning Models.

Compare LSTM and GRU with Persistence, LightGBM, and SARIMA
at 1h, 6h, 24h horizons.

Strategy: Direct forecasting — one model per horizon.
Data: Hybrid imputation | Test = REAL data only.

Usage:
    uv run python scripts/dl_multi_horizon.py 2>&1 | tee research/logs/dl_multi_horizon.log
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

warnings.filterwarnings("ignore")

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "dl"
HORIZONS = [1, 6, 24]

# Model hyperparameters
LOOKBACK = 72        # 3 days of history as input sequence
HIDDEN_DIM = 64      # LSTM/GRU hidden dimension
NUM_LAYERS = 2       # stacked layers
DROPOUT = 0.2        # dropout between layers
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100         # max epochs (early stopping will trigger earlier)
PATIENCE = 10        # early stopping patience

# Features to use (multivariate)
FEATURE_COLS = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════


class TimeSeriesDataset(Dataset):
    """Sliding window dataset for sequence-to-one forecasting."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lookback: int,
        horizon: int,
        real_mask: np.ndarray | None = None,
    ):
        self.features = features
        self.targets = targets
        self.lookback = lookback
        self.horizon = horizon
        self.real_mask = real_mask

        # Build valid indices
        self.valid_indices = []
        max_start = len(features) - lookback - horizon
        for i in range(max_start):
            target_idx = i + lookback + horizon - 1
            if target_idx < len(targets):
                # If real_mask provided, only include real test points
                if real_mask is not None and not real_mask[target_idx]:
                    continue
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x = self.features[i : i + self.lookback]  # (lookback, n_features)
        y = self.targets[i + self.lookback + self.horizon - 1]  # scalar
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ══════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════


class LSTMModel(nn.Module):
    """LSTM for time series forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out)


class GRUModel(nn.Module):
    """GRU for time series forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    horizon: int,
) -> nn.Module:
    """Train with early stopping on validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()
                n_val += 1

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            stop_info = f"patience={patience_counter}/{PATIENCE}"
            print(
                f"    Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"Train={train_loss:.4f} Val={val_loss:.4f} "
                f"LR={lr:.1e} {stop_info}",
                flush=True,
            )

        if patience_counter >= PATIENCE:
            print(
                f"    Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})",
                flush=True,
            )
            break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("LSTM/GRU MULTI-HORIZON EVALUATION", flush=True)
    print(f"Horizons: {HORIZONS}h | Lookback: {LOOKBACK}h | Device: {DEVICE}", flush=True)
    print(f"Architecture: hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, dropout={DROPOUT}", flush=True)
    print(f"Training: epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}", flush=True)
    print("RULE: Test set = REAL data only", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Prepare data ──
    print("\n[1/5] Preparing Hybrid dataset...", flush=True)
    df_hybrid = _prepare_hybrid_data()
    is_imputed = df_hybrid["is_imputed"].values

    # Select features
    available_features = [c for c in FEATURE_COLS if c in df_hybrid.columns]
    features_df = df_hybrid[available_features].copy()
    target = df_hybrid[TARGET_COL].values
    print(f"  Features: {available_features}", flush=True)
    print(f"  Data: {len(features_df)} rows × {len(available_features)} features", flush=True)

    # ── Step 2: Temporal split ──
    print("\n[2/5] Splitting data...", flush=True)
    n = len(features_df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Scale features
    scaler = StandardScaler()
    features_scaled = np.zeros_like(features_df.values)
    features_scaled[:train_end] = scaler.fit_transform(features_df.values[:train_end])
    features_scaled[train_end:] = scaler.transform(features_df.values[train_end:])

    # Scale target separately for inverse transform
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(target[:train_end].reshape(-1, 1)).flatten()
    target_all_scaled = target_scaler.transform(target.reshape(-1, 1)).flatten()

    print(f"  Train: {train_end} | Val: {val_end - train_end} | Test: {n - val_end}", flush=True)
    print(f"  Feature scaling: StandardScaler (fit on train only)", flush=True)

    # Reference results from previous experiments
    ref_results = {
        # Reference from multi_horizon v2 (post-audit ground truth)
        "1h": {"lgbm_mase": 1.492, "sarima_mase": 1.283},
        "6h": {"lgbm_mase": 0.745, "sarima_mase": 0.762},
        "24h": {"lgbm_mase": 0.842, "sarima_mase": 0.813},
    }

    all_results = {}

    for h in HORIZONS:
        print(f"\n{'═' * 70}", flush=True)
        print(f"[3/5] HORIZON = {h}h", flush=True)
        print(f"{'═' * 70}", flush=True)

        results_h = _evaluate_horizon(
            features_scaled, target_all_scaled, target, is_imputed,
            train_end, val_end, h, target_scaler,
        )
        all_results[f"{h}h"] = results_h

        # Quick comparison
        r = ref_results[f"{h}h"]
        lstm_mase = results_h.get("LSTM", {}).get("mase", 99)
        gru_mase = results_h.get("GRU", {}).get("mase", 99)
        print(f"\n  📊 Quick comparison ({h}h):", flush=True)
        print(f"    Persistence: 1.000 | LightGBM: {r['lgbm_mase']:.3f} | SARIMA: {r['sarima_mase']:.3f}", flush=True)
        print(f"    LSTM: {lstm_mase:.3f} | GRU: {gru_mase:.3f}", flush=True)

    # ── Summary ──
    print(f"\n{'═' * 70}", flush=True)
    print("[4/5] FULL MODEL COMPARISON SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)

    print(f"\n{'Horizon':<10} {'Model':<20} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'Status':>12}", flush=True)
    print("─" * 75, flush=True)

    for h in HORIZONS:
        h_key = f"{h}h"

        # Persistence
        p = all_results[h_key].get("Persistence", {})
        print(f"{h}h{'':<7} {'Persistence':<20} {p.get('mae', 0):>8.3f} {p.get('rmse', 0):>8.3f} {'1.000':>8} {'baseline':>12}", flush=True)

        # LSTM
        lstm = all_results[h_key].get("LSTM", {})
        if lstm:
            s = "✅ BEATS!" if lstm.get("mase", 99) < 1.0 else "❌ MASE>1"
            print(f"{h}h{'':<7} {'LSTM':<20} {lstm.get('mae', 0):>8.3f} {lstm.get('rmse', 0):>8.3f} {lstm.get('mase', 0):>8.3f} {s:>12}", flush=True)

        # GRU
        gru = all_results[h_key].get("GRU", {})
        if gru:
            s = "✅ BEATS!" if gru.get("mase", 99) < 1.0 else "❌ MASE>1"
            print(f"{h}h{'':<7} {'GRU':<20} {gru.get('mae', 0):>8.3f} {gru.get('rmse', 0):>8.3f} {gru.get('mase', 0):>8.3f} {s:>12}", flush=True)

        # References
        r = ref_results[h_key]
        print(f"{h}h{'':<7} {'SARIMA (ref)':<20} {'—':>8} {'—':>8} {r['sarima_mase']:>8.3f} {'':>12}", flush=True)
        print(f"{h}h{'':<7} {'LightGBM (ref)':<20} {'—':>8} {'—':>8} {r['lgbm_mase']:>8.3f} {'':>12}", flush=True)
        print("─" * 75, flush=True)

    # ── Save ──
    print(f"\n[5/5] Saving results...", flush=True)
    _save_results(all_results)

    total_time = time.time() - t_start
    print(f"\n{'═' * 70}", flush=True)
    print(f"COMPLETE — Total: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print(f"{'═' * 70}", flush=True)


def _prepare_hybrid_data() -> pd.DataFrame:
    """Load raw data and apply Hybrid imputation strategy."""
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


def _evaluate_horizon(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    target_original: np.ndarray,
    is_imputed: np.ndarray,
    train_end: int,
    val_end: int,
    horizon: int,
    target_scaler: StandardScaler,
) -> dict:
    """Evaluate LSTM and GRU at a specific horizon."""
    results = {}
    n = len(features_scaled)
    input_dim = features_scaled.shape[1]

    # ── Persistence baseline ──
    print(f"  Evaluating Persistence baseline ({horizon}h)...", flush=True)
    test_start = val_end + LOOKBACK  # need lookback history
    y_true_list = []
    y_persist_list = []

    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        actual = target_original[i + horizon]
        persist = target_original[i]
        if not np.isnan(actual) and not np.isnan(persist):
            y_true_list.append(actual)
            y_persist_list.append(persist)

    y_true = np.array(y_true_list)
    y_persist = np.array(y_persist_list)
    persist_mae = float(np.mean(np.abs(y_true - y_persist)))
    persist_rmse = float(np.sqrt(np.mean((y_true - y_persist) ** 2)))
    results["Persistence"] = {"mae": round(persist_mae, 4), "rmse": round(persist_rmse, 4), "mase": 1.0}
    print(f"    Persistence {horizon}h: MAE={persist_mae:.3f}, RMSE={persist_rmse:.3f}", flush=True)

    # ── Datasets (real-only mask for test) ──
    real_mask_test = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imputed[i]:
            real_mask_test[i] = True

    train_dataset = TimeSeriesDataset(features_scaled, target_scaled, LOOKBACK, horizon)
    # Filter train to only use indices within training range
    train_dataset.valid_indices = [
        i for i in train_dataset.valid_indices if i + LOOKBACK + horizon - 1 < train_end
    ]

    val_dataset = TimeSeriesDataset(features_scaled, target_scaled, LOOKBACK, horizon)
    val_dataset.valid_indices = [
        i for i in val_dataset.valid_indices
        if train_end <= i + LOOKBACK + horizon - 1 < val_end
    ]

    test_dataset = TimeSeriesDataset(
        features_scaled, target_scaled, LOOKBACK, horizon,
        real_mask=real_mask_test,
    )
    test_dataset.valid_indices = [
        i for i in test_dataset.valid_indices if i + LOOKBACK + horizon - 1 >= val_end
    ]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)} (real only)", flush=True)

    if len(train_dataset) < BATCH_SIZE:
        print(f"  ⚠️ Too few training samples, skipping", flush=True)
        return results

    # ── Train LSTM ──
    for model_name, ModelClass in [("LSTM", LSTMModel), ("GRU", GRUModel)]:
        print(f"\n  Training {model_name} ({horizon}h)...", flush=True)
        t0 = time.time()

        model = ModelClass(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params:,}", flush=True)

        model = train_model(model, train_loader, val_loader, model_name, horizon)
        train_time = time.time() - t0

        # ── Evaluate on test set ──
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(DEVICE)
                pred = model(x_batch).cpu().numpy().flatten()
                all_preds.extend(pred)
                all_targets.extend(y_batch.numpy().flatten())

        if len(all_preds) == 0:
            print(f"    ⚠️ No predictions generated", flush=True)
            continue

        # Inverse transform to original scale
        preds_original = target_scaler.inverse_transform(
            np.array(all_preds).reshape(-1, 1)
        ).flatten()
        targets_original = target_scaler.inverse_transform(
            np.array(all_targets).reshape(-1, 1)
        ).flatten()

        mae = float(np.mean(np.abs(targets_original - preds_original)))
        rmse = float(np.sqrt(np.mean((targets_original - preds_original) ** 2)))
        mase = round(mae / persist_mae, 4) if persist_mae > 0 else float("inf")

        results[model_name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mase": mase,
            "params": n_params,
            "train_time_s": round(train_time, 1),
            "n_test": len(all_preds),
        }

        status = "✅" if mase < 1.0 else "❌"
        print(
            f"    {status} {model_name} {horizon}h: MAE={mae:.3f}, MASE={mase:.3f} "
            f"({train_time:.0f}s, {len(all_preds)} test points)",
            flush=True,
        )

    return results


def _save_results(all_results: dict) -> None:
    """Save results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"dl_multi_horizon_{timestamp}.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"  Results saved: {json_path}", flush=True)


if __name__ == "__main__":
    main()
