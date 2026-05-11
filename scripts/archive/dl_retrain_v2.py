"""DL Retrain v2 — GRU/LSTM with Enhanced Features + Log Transform Comparison.

3 Tasks:
  1. Retrain GRU/LSTM with Fourier + interaction features (v2 enhanced)
  2. Add CV features (std/mean) with safeguard (clamp min mean=1.0)
  3. Compare log1p transform vs raw target for each model

Strategy: Direct forecasting — one model per horizon.
Data: Hybrid imputation | Test = REAL data only.

Usage:
    uv run python scripts/dl_retrain_v2.py 2>&1 | tee research/logs/dl_retrain_v2.log
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
from src.features.builder import build_features

warnings.filterwarnings("ignore")

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "dl_v2"
HORIZONS = [1, 6, 24]

# Model hyperparameters
LOOKBACK = 72
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# CV feature windows (same as rolling windows)
CV_WINDOWS = [6, 12, 24]


# ══════════════════════════════════════════════════════════════════════
# CV Features with Safeguard
# ══════════════════════════════════════════════════════════════════════


def add_cv_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Add Coefficient of Variation features with safeguard.

    CV = std / mean. When mean is near 0, CV explodes.
    Safeguard: clamp |mean| >= 1.0 µg/m³ before division.
    Uses pm25_lag_1h to prevent leakage.
    """
    df = df.copy()
    lag_col = f"{target_col}_lag_1h"
    if lag_col not in df.columns:
        return df

    for w in CV_WINDOWS:
        roll = df[lag_col].shift(1).rolling(window=w, min_periods=max(w // 2, 2))
        roll_std = roll.std()
        roll_mean = roll.mean()

        # Safeguard: clamp mean away from 0 to prevent CV explosion
        safe_mean = roll_mean.abs().clip(lower=1.0)
        cv = roll_std / safe_mean

        # Additional safeguard: clip extreme values
        cv = cv.clip(upper=5.0)  # CV > 5.0 = noise, not signal

        col_name = f"{target_col}_cv_{w}h"
        df[col_name] = cv

    return df


# ══════════════════════════════════════════════════════════════════════
# Dataset & Models (same architecture as dl_multi_horizon.py)
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


class LSTMModel(nn.Module):
    """LSTM for time series forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
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
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    """GRU for time series forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
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
        return self.fc(out[:, -1, :])


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"    Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"Train={train_loss:.4f} Val={val_loss:.4f} "
                f"LR={lr:.1e} patience={patience_counter}/{PATIENCE}",
                flush=True,
            )

        if patience_counter >= PATIENCE:
            print(
                f"    Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ══════════════════════════════════════════════════════════════════════
# Core: Prepare + Evaluate
# ══════════════════════════════════════════════════════════════════════


def prepare_data() -> pd.DataFrame:
    """Load → clean → impute → build_features (v2 enhanced) → add CV."""
    print("  Loading and cleaning raw data...", flush=True)
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")

    print("  Imputing (hybrid)...", flush=True)
    df_hybrid = impute_missing_data(
        df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24,
        knn_neighbors=5, verbose=True,
    )

    # Save is_imputed before feature building
    is_imputed = df_hybrid["is_imputed"].copy()
    df_for_feat = df_hybrid.drop(columns=["is_imputed"])

    print("  Building v2 features (Fourier + interactions)...", flush=True)
    df_feat = build_features(df_for_feat, include_fourier=True, fourier_order=3)

    # Add CV features with safeguard
    print("  Adding CV features (std/mean with safeguard)...", flush=True)
    df_feat = add_cv_features(df_feat, TARGET_COL)

    # Restore is_imputed
    df_feat["is_imputed"] = is_imputed.reindex(df_feat.index).fillna(False)

    cv_cols = [c for c in df_feat.columns if "_cv_" in c]
    print(f"  CV features added: {cv_cols}", flush=True)
    print(f"  Total features: {len(df_feat.columns)} cols × {len(df_feat):,} rows", flush=True)

    return df_feat


def select_dl_features(df: pd.DataFrame) -> list[str]:
    """Select features suitable for DL models.

    DL needs numerical features only. Exclude target, is_imputed, and
    features that would leak target info.
    """
    exclude = {"is_imputed", TARGET_COL}
    # Also exclude any target_{horizon}h columns
    exclude.update(c for c in df.columns if c.startswith("target_"))

    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ("float64", "float32", "int64")
    ]
    return sorted(feature_cols)


def evaluate_horizon(
    df_feat: pd.DataFrame,
    horizon: int,
    use_log: bool,
    label_suffix: str = "",
) -> dict:
    """Evaluate GRU+LSTM at a specific horizon.

    Args:
        df_feat: DataFrame with all features.
        horizon: Forecast horizon (1, 6, 24).
        use_log: Whether to use log1p target transform.
        label_suffix: Suffix for model names (e.g., "_log" or "_raw").

    Returns:
        Dict of model results with MAE, MASE, etc.
    """
    results = {}

    # ── Target ──
    is_imputed = df_feat["is_imputed"].values
    target = df_feat[TARGET_COL].values

    # ── Feature selection ──
    feature_cols = select_dl_features(df_feat)
    features_df = df_feat[feature_cols].fillna(0)

    n = len(features_df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # ── Scale features (fit on train only) ──
    scaler = StandardScaler()
    features_scaled = np.zeros_like(features_df.values, dtype=np.float32)
    features_scaled[:train_end] = scaler.fit_transform(features_df.values[:train_end])
    features_scaled[train_end:] = scaler.transform(features_df.values[train_end:])

    # ── Scale target ──
    if use_log:
        target_transformed = np.log1p(np.clip(target, 0, None))
    else:
        target_transformed = target.copy()

    target_scaler = StandardScaler()
    target_scaler.fit(target_transformed[:train_end].reshape(-1, 1))
    target_all_scaled = target_scaler.transform(target_transformed.reshape(-1, 1)).flatten()

    # Handle NaN/inf in scaled features
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Features: {len(feature_cols)} cols | log={use_log}", flush=True)
    print(f"  Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}", flush=True)

    # ── Persistence baseline ──
    y_true_list, y_persist_list = [], []
    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        actual = target[i + horizon]
        persist = target[i]
        if not np.isnan(actual) and not np.isnan(persist):
            y_true_list.append(actual)
            y_persist_list.append(persist)

    y_true = np.array(y_true_list)
    y_persist = np.array(y_persist_list)
    persist_mae = float(np.mean(np.abs(y_true - y_persist)))
    persist_rmse = float(np.sqrt(np.mean((y_true - y_persist) ** 2)))
    results[f"Persistence{label_suffix}"] = {
        "mae": round(persist_mae, 4), "rmse": round(persist_rmse, 4), "mase": 1.0,
    }
    print(f"  Persistence {horizon}h: MAE={persist_mae:.3f}", flush=True)

    # ── Datasets ──
    real_mask_test = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imputed[i]:
            real_mask_test[i] = True

    train_dataset = TimeSeriesDataset(features_scaled, target_all_scaled, LOOKBACK, horizon)
    train_dataset.valid_indices = [
        i for i in train_dataset.valid_indices if i + LOOKBACK + horizon - 1 < train_end
    ]

    val_dataset = TimeSeriesDataset(features_scaled, target_all_scaled, LOOKBACK, horizon)
    val_dataset.valid_indices = [
        i for i in val_dataset.valid_indices
        if train_end <= i + LOOKBACK + horizon - 1 < val_end
    ]

    test_dataset = TimeSeriesDataset(
        features_scaled, target_all_scaled, LOOKBACK, horizon,
        real_mask=real_mask_test,
    )
    test_dataset.valid_indices = [
        i for i in test_dataset.valid_indices if i + LOOKBACK + horizon - 1 >= val_end
    ]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}", flush=True)

    if len(train_dataset) < BATCH_SIZE:
        print("  ⚠️ Too few training samples, skipping", flush=True)
        return results

    input_dim = features_scaled.shape[1]

    # ── Train GRU + LSTM ──
    for model_name, ModelClass in [("GRU", GRUModel), ("LSTM", LSTMModel)]:
        full_name = f"{model_name}{label_suffix}"
        print(f"\n  Training {full_name} ({horizon}h, log={use_log})...", flush=True)
        t0 = time.time()

        model = ModelClass(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params:,} | Input dim: {input_dim}", flush=True)

        model = train_model(model, train_loader, val_loader, full_name)
        train_time = time.time() - t0

        # ── Evaluate ──
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(DEVICE)
                pred = model(x_batch).cpu().numpy().flatten()
                all_preds.extend(pred)
                all_targets.extend(y_batch.numpy().flatten())

        if len(all_preds) == 0:
            print("    ⚠️ No predictions", flush=True)
            continue

        # Inverse transform: scaler → then expm1 if log
        preds_scaled = target_scaler.inverse_transform(
            np.array(all_preds).reshape(-1, 1)
        ).flatten()
        targets_scaled = target_scaler.inverse_transform(
            np.array(all_targets).reshape(-1, 1)
        ).flatten()

        if use_log:
            preds_original = np.clip(np.expm1(preds_scaled), 0, None)
            targets_original = np.clip(np.expm1(targets_scaled), 0, None)
        else:
            preds_original = np.clip(preds_scaled, 0, None)
            targets_original = np.clip(targets_scaled, 0, None)

        mae = float(np.mean(np.abs(targets_original - preds_original)))
        rmse = float(np.sqrt(np.mean((targets_original - preds_original) ** 2)))
        mase = round(mae / persist_mae, 4) if persist_mae > 0 else float("inf")

        results[full_name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mase": mase,
            "params": n_params,
            "input_dim": input_dim,
            "train_time_s": round(train_time, 1),
            "n_test": len(all_preds),
            "use_log": use_log,
            "features": "v2_enhanced+cv",
        }

        status = "✅" if mase < 1.0 else "❌"
        print(
            f"    {status} {full_name} {horizon}h: MAE={mae:.3f}, MASE={mase:.3f} "
            f"({train_time:.0f}s, {len(all_preds)} pts)",
            flush=True,
        )

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("DL RETRAIN v2 — Enhanced Features + Log Transform Comparison", flush=True)
    print(f"Horizons: {HORIZONS}h | Lookback: {LOOKBACK}h | Device: {DEVICE}", flush=True)
    print(f"Architecture: hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, dropout={DROPOUT}", flush=True)
    print(f"Training: epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}", flush=True)
    print(f"New: CV features (windows={CV_WINDOWS}), Fourier, Interactions", flush=True)
    print("RULE: Test set = REAL data only", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Prepare data ──
    print("\n[1/4] Preparing v2 enhanced dataset...", flush=True)
    df_feat = prepare_data()

    # ── Step 2: Run all horizons ──
    all_results = {}

    for h in HORIZONS:
        print(f"\n{'═' * 70}", flush=True)
        print(f"[2/4] HORIZON = {h}h", flush=True)
        print(f"{'═' * 70}", flush=True)

        # A) Raw target
        print(f"\n  ── A) RAW target ──", flush=True)
        raw_results = evaluate_horizon(df_feat, h, use_log=False, label_suffix="_raw")

        # B) Log1p target
        print(f"\n  ── B) LOG1P target ──", flush=True)
        log_results = evaluate_horizon(df_feat, h, use_log=True, label_suffix="_log")

        # Merge
        merged = {}
        merged["Persistence"] = raw_results.get("Persistence_raw", {})
        merged.update(raw_results)
        merged.update(log_results)
        # Remove duplicate Persistence entries
        for k in list(merged.keys()):
            if k.startswith("Persistence"):
                if k != "Persistence":
                    merged.pop(k, None)

        all_results[f"{h}h"] = merged

    # ── Step 3: Summary ──
    print(f"\n{'═' * 70}", flush=True)
    print("[3/4] LOG TRANSFORM COMPARISON", flush=True)
    print(f"{'═' * 70}", flush=True)

    print(f"\n{'Horizon':<8} {'Model':<18} {'MAE':<10} {'MASE':<10} {'Status':<10}", flush=True)
    print("─" * 60, flush=True)

    for h in HORIZONS:
        h_key = f"{h}h"
        h_results = all_results.get(h_key, {})

        # Persistence
        p = h_results.get("Persistence", {})
        print(f"{h}h       {'Persistence':<18} {p.get('mae', 0):<10.3f} {'1.000':<10} {'baseline':<10}", flush=True)

        # Compare raw vs log for each model
        for model in ["GRU", "LSTM"]:
            raw = h_results.get(f"{model}_raw", {})
            log = h_results.get(f"{model}_log", {})

            if raw:
                s = "✅" if raw.get("mase", 99) < 1 else "❌"
                print(f"{h}h       {f'{model}_raw':<18} {raw.get('mae', 0):<10.3f} {raw.get('mase', 0):<10.3f} {s:<10}", flush=True)
            if log:
                s = "✅" if log.get("mase", 99) < 1 else "❌"
                print(f"{h}h       {f'{model}_log':<18} {log.get('mae', 0):<10.3f} {log.get('mase', 0):<10.3f} {s:<10}", flush=True)

        # Who wins: raw or log?
        for model in ["GRU", "LSTM"]:
            raw_mase = h_results.get(f"{model}_raw", {}).get("mase", 99)
            log_mase = h_results.get(f"{model}_log", {}).get("mase", 99)
            winner = "RAW" if raw_mase <= log_mase else "LOG"
            diff = abs(raw_mase - log_mase)
            print(f"         {f'→ {model} winner':<18} {winner:<10} delta={diff:.4f}", flush=True)

        print("─" * 60, flush=True)

    # ── Reference comparison ──
    print(f"\n{'═' * 70}", flush=True)
    print("REFERENCE: v1 DL (5 features) vs v2 DL (v2 enhanced + CV)", flush=True)
    print(f"{'═' * 70}", flush=True)

    # v1 reference from RUNS_LOG
    v1_ref = {
        "1h": {"GRU": 1.173, "LSTM": 1.560},
        "6h": {"GRU": 0.812, "LSTM": 0.914},
        "24h": {"GRU": 0.727, "LSTM": 0.830},
    }

    print(f"{'Horizon':<8} {'Model':<12} {'v1 MASE':<10} {'v2_raw':<10} {'v2_log':<10} {'Δ best':<10}", flush=True)
    print("─" * 55, flush=True)

    for h in HORIZONS:
        h_key = f"{h}h"
        for model in ["GRU", "LSTM"]:
            v1_mase = v1_ref[h_key][model]
            raw_mase = all_results.get(h_key, {}).get(f"{model}_raw", {}).get("mase", 99)
            log_mase = all_results.get(h_key, {}).get(f"{model}_log", {}).get("mase", 99)
            best_v2 = min(raw_mase, log_mase)
            delta = best_v2 - v1_mase
            pct = (delta / v1_mase) * 100
            print(
                f"{h}h       {model:<12} {v1_mase:<10.3f} {raw_mase:<10.3f} {log_mase:<10.3f} {pct:>+8.1f}%",
                flush=True,
            )

    # ── Step 4: Save ──
    print(f"\n[4/4] Saving results...", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"dl_retrain_v2_{ts}.json"

    meta = {
        "description": "DL retrain with v2 enhanced features + CV + log transform comparison",
        "feature_set": "v2_enhanced + cv_features (std/mean, safeguard mean>=1.0)",
        "cv_windows": CV_WINDOWS,
        "lookback": LOOKBACK,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat(),
    }

    output = {"meta": meta, **all_results}

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"  Results saved: {out_path}", flush=True)

    total = time.time() - t_start
    print(f"\nCOMPLETE — Total: {total:.0f}s ({total / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
