"""DL v3 — PCA Feature Selection for 1h + TFT Retrain with v2 Features.

Tasks:
  1. PCA/feature selection for 1h: reduce 117→top-N features, retrain GRU/LSTM
  2. TFT retrain: use v2 enhanced features instead of 5 raw features

Strategy:
  - PCA: Fit on training data, select N components that explain 95% variance
  - Feature importance: Use LightGBM feature importance to select top-N features
  - TFT: Extended with v2 features (temporal + static)

Usage:
    uv run python scripts/dl_v3_pca_tft.py 2>&1 | tee research/logs/dl_v3_pca_tft.log
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
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
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "dl_v3"
HORIZONS = [1, 6, 24]

# DL Config
LOOKBACK = 72
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10

# PCA Config
PCA_VARIANCE_THRESHOLD = 0.95  # Keep 95% variance
TOP_N_FEATURES = [10, 20, 40]  # Feature selection candidates

# TFT Config
TFT_HIDDEN = 32
TFT_HEADS = 4
TFT_DROPOUT = 0.1
TFT_BATCH = 128
TFT_PATIENCE = 15

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CV_WINDOWS = [6, 12, 24]


# ══════════════════════════════════════════════════════════════════════
# CV Features (same as dl_retrain_v2.py)
# ══════════════════════════════════════════════════════════════════════


def add_cv_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Add Coefficient of Variation features with safeguard."""
    df = df.copy()
    lag_col = f"{target_col}_lag_1h"
    if lag_col not in df.columns:
        return df
    for w in CV_WINDOWS:
        roll = df[lag_col].shift(1).rolling(window=w, min_periods=max(w // 2, 2))
        safe_mean = roll.mean().abs().clip(lower=1.0)
        cv = roll.std() / safe_mean
        df[f"{target_col}_cv_{w}h"] = cv.clip(upper=5.0)
    return df


# ══════════════════════════════════════════════════════════════════════
# Data Preparation
# ══════════════════════════════════════════════════════════════════════


def prepare_data() -> pd.DataFrame:
    """Load → clean → impute → build_features (v2) → add CV."""
    print("  Loading raw data...", flush=True)
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
    is_imputed = df_hybrid["is_imputed"].copy()
    df_for_feat = df_hybrid.drop(columns=["is_imputed"])

    print("  Building v2 features...", flush=True)
    df_feat = build_features(df_for_feat, include_fourier=True, fourier_order=3)
    df_feat = add_cv_features(df_feat, TARGET_COL)
    df_feat["is_imputed"] = is_imputed.reindex(df_feat.index).fillna(False)

    print(f"  Data ready: {len(df_feat):,} rows × {len(df_feat.columns)} cols", flush=True)
    return df_feat


def select_dl_features(df: pd.DataFrame) -> list[str]:
    """Select numerical features, excluding target and metadata."""
    exclude = {"is_imputed", TARGET_COL}
    exclude.update(c for c in df.columns if c.startswith("target_"))
    return sorted([
        c for c in df.columns
        if c not in exclude and df[c].dtype in ("float64", "float32", "int64")
    ])


# ══════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, lookback, horizon, real_mask=None):
        self.features = features
        self.targets = targets
        self.lookback = lookback
        self.horizon = horizon
        self.valid_indices = []
        for i in range(len(features) - lookback - horizon):
            ti = i + lookback + horizon - 1
            if ti < len(targets):
                if real_mask is not None and not real_mask[ti]:
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
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ══════════════════════════════════════════════════════════════════════
# TFT Components (simplified, from tft_multi_horizon.py)
# ══════════════════════════════════════════════════════════════════════


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(self.glu(self.fc2(h)))
        return self.layer_norm(h + residual)


class SimplifiedTFT(nn.Module):
    """TFT with v2 features: more temporal inputs + static calendar."""

    def __init__(self, temporal_dim, static_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_grn = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.post_lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        # Multi-head attention
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.post_attn_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)
        self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def _attention(self, x):
        B, S, _ = x.shape
        Q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(ctx)

    def forward(self, temporal, static):
        static_ctx = self.static_grn(static).unsqueeze(1)
        t_emb = self.temporal_proj(temporal) + static_ctx
        lstm_out, _ = self.lstm(t_emb)
        gated = self.post_lstm_gate(lstm_out)
        enriched = self.post_lstm_norm(gated + t_emb)
        attn_out = self._attention(enriched)
        attn_out = self.post_attn_gate(attn_out)
        enriched = self.post_attn_norm(attn_out + enriched)
        last = enriched[:, -1, :]
        return self.fc_out(self.dropout(self.output_grn(last)))


class TFTDataset(Dataset):
    def __init__(self, temporal, static, targets, lookback, horizon, mask=None):
        self.temporal = temporal
        self.static = static
        self.targets = targets
        self.lookback = lookback
        self.horizon = horizon
        self.indices = []
        for i in range(len(temporal) - lookback - horizon):
            ti = i + lookback + horizon - 1
            if ti < len(targets) and (mask is None or mask[ti]):
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        t = self.temporal[i : i + self.lookback]
        s = self.static[i + self.lookback - 1]
        y = self.targets[i + self.lookback + self.horizon - 1]
        return torch.FloatTensor(t), torch.FloatTensor(s), torch.FloatTensor([y])


# ══════════════════════════════════════════════════════════════════════
# Training utilities
# ══════════════════════════════════════════════════════════════════════


def train_dl_model(model, train_loader, val_loader, name, patience=PATIENCE):
    """Train with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, 5)
    criterion = nn.MSELoss()
    best_val, best_state, pc = float("inf"), None, 0

    for ep in range(EPOCHS):
        model.train()
        tl, nb = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            y = batch[-1].to(DEVICE)
            # TFT has 3 elements, GRU/LSTM has 2
            if len(batch) == 3:
                s = batch[1].to(DEVICE)
                pred = model(x, s)
            else:
                pred = model(x)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
            nb += 1
        tl /= max(nb, 1)

        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                y = batch[-1].to(DEVICE)
                if len(batch) == 3:
                    s = batch[1].to(DEVICE)
                    pred = model(x, s)
                else:
                    pred = model(x)
                vl += criterion(pred, y).item()
                nv += 1
        vl /= max(nv, 1)
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = model.state_dict().copy()
            pc = 0
        else:
            pc += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {ep+1:3d}/{EPOCHS} train={tl:.4f} val={vl:.4f} lr={lr:.1e} pat={pc}/{patience}",
                  flush=True)

        if pc >= patience:
            print(f"    Early stop ep {ep+1} (best={best_val:.4f})", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════
# Task 1: PCA/Feature Selection for 1h
# ══════════════════════════════════════════════════════════════════════


def run_pca_experiment(df_feat: pd.DataFrame) -> dict:
    """Test PCA + top-N feature selection at 1h horizon ONLY."""
    print(f"\n{'═' * 70}", flush=True)
    print("TASK 1: PCA / Feature Selection for h=1", flush=True)
    print(f"{'═' * 70}", flush=True)

    feature_cols = select_dl_features(df_feat)
    features_df = df_feat[feature_cols].fillna(0)
    target = df_feat[TARGET_COL].values
    is_imputed = df_feat["is_imputed"].values

    n = len(features_df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    horizon = 1

    # Scale all features
    scaler = StandardScaler()
    X_scaled = np.zeros_like(features_df.values, dtype=np.float32)
    X_scaled[:train_end] = scaler.fit_transform(features_df.values[:train_end])
    X_scaled[train_end:] = scaler.transform(features_df.values[train_end:])
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Target scaler
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(target[:train_end].reshape(-1, 1))
    target_scaled = tgt_scaler.transform(target.reshape(-1, 1)).flatten()

    # Persistence
    y_true_list, y_persist_list = [], []
    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        a, p = target[i + horizon], target[i]
        if not np.isnan(a) and not np.isnan(p):
            y_true_list.append(a)
            y_persist_list.append(p)
    persist_mae = float(np.mean(np.abs(np.array(y_true_list) - np.array(y_persist_list))))
    print(f"  Persistence 1h: MAE={persist_mae:.3f}", flush=True)

    # Real mask
    real_mask = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imputed[i]:
            real_mask[i] = True

    results = {"Persistence": {"mae": round(persist_mae, 4), "mase": 1.0}}

    # ── A) PCA ──
    print(f"\n  ── A) PCA (95% variance) ──", flush=True)
    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=42)
    X_pca_train = pca.fit_transform(X_scaled[:train_end])
    X_pca_all = pca.transform(X_scaled)
    n_components = X_pca_train.shape[1]
    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"    PCA: {len(feature_cols)} → {n_components} components ({explained:.1f}% variance)", flush=True)

    for model_name, ModelClass in [("GRU_pca", GRUModel), ("LSTM_pca", LSTMModel)]:
        r = _train_and_eval(
            X_pca_all.astype(np.float32), target_scaled, target, is_imputed,
            real_mask, train_end, val_end, horizon, tgt_scaler, persist_mae,
            model_name, ModelClass, n_components,
        )
        results[model_name] = r

    # ── B) Top-N feature importance ──
    print(f"\n  ── B) Feature Importance Selection ──", flush=True)

    # Use LightGBM to get feature importance
    import lightgbm as lgb

    lgbm = lgb.LGBMRegressor(n_estimators=200, max_depth=5, verbose=-1, n_jobs=1)
    lgbm.fit(features_df.values[:train_end], target[:train_end])
    importances = pd.Series(lgbm.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False)
    print(f"    Top-10 features: {list(top_features.head(10).index)}", flush=True)

    for top_n in TOP_N_FEATURES:
        selected = list(top_features.head(top_n).index)
        sel_indices = [feature_cols.index(c) for c in selected]
        X_sel = X_scaled[:, sel_indices]
        print(f"\n    Top-{top_n} features:", flush=True)

        for model_name, ModelClass in [(f"GRU_top{top_n}", GRUModel)]:
            r = _train_and_eval(
                X_sel, target_scaled, target, is_imputed,
                real_mask, train_end, val_end, horizon, tgt_scaler, persist_mae,
                model_name, ModelClass, top_n,
            )
            results[model_name] = r

    return results


def _train_and_eval(
    features_scaled, target_scaled, target_orig, is_imputed,
    real_mask, train_end, val_end, horizon, tgt_scaler, persist_mae,
    model_name, ModelClass, input_dim,
):
    """Train and evaluate a single DL model."""
    print(f"\n  Training {model_name} (h={horizon}, dim={input_dim})...", flush=True)
    t0 = time.time()

    train_ds = TimeSeriesDataset(features_scaled, target_scaled, LOOKBACK, horizon)
    train_ds.valid_indices = [i for i in train_ds.valid_indices if i + LOOKBACK + horizon - 1 < train_end]
    val_ds = TimeSeriesDataset(features_scaled, target_scaled, LOOKBACK, horizon)
    val_ds.valid_indices = [i for i in val_ds.valid_indices if train_end <= i + LOOKBACK + horizon - 1 < val_end]
    test_ds = TimeSeriesDataset(features_scaled, target_scaled, LOOKBACK, horizon, real_mask=real_mask)
    test_ds.valid_indices = [i for i in test_ds.valid_indices if i + LOOKBACK + horizon - 1 >= val_end]

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"    Datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)

    if len(train_ds) < BATCH_SIZE:
        print("    ⚠️ Too few samples", flush=True)
        return {"mae": float("nan"), "mase": float("nan")}

    model = ModelClass(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params:,}", flush=True)

    model = train_dl_model(model, train_ld, val_ld, model_name)
    train_time = time.time() - t0

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_ld:
            pred = model(x.to(DEVICE)).cpu().numpy().flatten()
            preds.extend(pred)
            targets.extend(y.numpy().flatten())

    preds_orig = np.clip(tgt_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten(), 0, None)
    targets_orig = np.clip(tgt_scaler.inverse_transform(np.array(targets).reshape(-1, 1)).flatten(), 0, None)

    mae = float(np.mean(np.abs(targets_orig - preds_orig)))
    rmse = float(np.sqrt(np.mean((targets_orig - preds_orig) ** 2)))
    mase = round(mae / persist_mae, 4) if persist_mae > 0 else float("inf")

    status = "✅" if mase < 1.0 else "❌"
    print(f"    {status} {model_name}: MAE={mae:.3f}, MASE={mase:.3f} ({train_time:.0f}s)", flush=True)

    return {
        "mae": round(mae, 4), "rmse": round(rmse, 4), "mase": mase,
        "params": n_params, "input_dim": input_dim,
        "train_time_s": round(train_time, 1), "n_test": len(preds),
    }


# ══════════════════════════════════════════════════════════════════════
# Task 2: TFT Retrain with v2 Features
# ══════════════════════════════════════════════════════════════════════


def run_tft_retrain(df_feat: pd.DataFrame) -> dict:
    """Retrain TFT with v2 enhanced features at all horizons."""
    print(f"\n{'═' * 70}", flush=True)
    print("TASK 2: TFT Retrain with v2 Enhanced Features", flush=True)
    print(f"{'═' * 70}", flush=True)

    # Temporal = all numerical features except target/is_imputed/calendar sin/cos
    feature_cols = select_dl_features(df_feat)
    # Static = calendar cyclical features (pre-computed by build_features)
    static_cols = [c for c in feature_cols if c.startswith(("hour_", "day_", "month_"))]
    temporal_cols = [c for c in feature_cols if c not in static_cols]

    print(f"  Temporal: {len(temporal_cols)} features", flush=True)
    print(f"  Static: {len(static_cols)} features → {static_cols}", flush=True)

    target = df_feat[TARGET_COL].values
    is_imputed = df_feat["is_imputed"].values
    n = len(df_feat)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Scale temporal
    temp_scaler = StandardScaler()
    temporal = df_feat[temporal_cols].fillna(0).values.astype(np.float32)
    temporal_scaled = np.zeros_like(temporal)
    temporal_scaled[:train_end] = temp_scaler.fit_transform(temporal[:train_end])
    temporal_scaled[train_end:] = temp_scaler.transform(temporal[train_end:])
    temporal_scaled = np.nan_to_num(temporal_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Static (already normalized sin/cos)
    static = df_feat[static_cols].fillna(0).values.astype(np.float32)

    # Target scaler
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(target[:train_end].reshape(-1, 1))
    target_scaled = tgt_scaler.transform(target.reshape(-1, 1)).flatten()

    # Real mask
    real_mask = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imputed[i]:
            real_mask[i] = True

    # v1 reference TFT results
    v1_ref = {"1h": 1.029, "6h": 0.821, "24h": 0.811}

    all_results = {}

    for h in HORIZONS:
        print(f"\n  {'─' * 50}", flush=True)
        print(f"  TFT v2 — Horizon {h}h", flush=True)
        print(f"  {'─' * 50}", flush=True)

        # Persistence
        y_true_list, y_persist_list = [], []
        for i in range(val_end, n - h):
            if is_imputed[i + h]:
                continue
            a, p = target[i + h], target[i]
            if not np.isnan(a) and not np.isnan(p):
                y_true_list.append(a)
                y_persist_list.append(p)
        persist_mae = float(np.mean(np.abs(np.array(y_true_list) - np.array(y_persist_list))))
        print(f"    Persistence {h}h: MAE={persist_mae:.3f}", flush=True)

        # Datasets
        train_ds = TFTDataset(temporal_scaled, static, target_scaled, LOOKBACK, h)
        train_ds.indices = [i for i in train_ds.indices if i + LOOKBACK + h - 1 < train_end]
        val_ds = TFTDataset(temporal_scaled, static, target_scaled, LOOKBACK, h)
        val_ds.indices = [i for i in val_ds.indices if train_end <= i + LOOKBACK + h - 1 < val_end]
        test_ds = TFTDataset(temporal_scaled, static, target_scaled, LOOKBACK, h, mask=real_mask)
        test_ds.indices = [i for i in test_ds.indices if i + LOOKBACK + h - 1 >= val_end]

        train_ld = DataLoader(train_ds, batch_size=TFT_BATCH, shuffle=True, drop_last=True)
        val_ld = DataLoader(val_ds, batch_size=TFT_BATCH, shuffle=False)
        test_ld = DataLoader(test_ds, batch_size=TFT_BATCH, shuffle=False)

        print(f"    Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}", flush=True)

        if len(train_ds) < TFT_BATCH:
            print("    ⚠️ Too few samples", flush=True)
            all_results[f"{h}h"] = {"Persistence": {"mae": persist_mae, "mase": 1.0}}
            continue

        # Build TFT
        model = SimplifiedTFT(
            temporal_dim=len(temporal_cols),
            static_dim=len(static_cols),
            hidden_dim=TFT_HIDDEN,
            num_heads=TFT_HEADS,
            dropout=TFT_DROPOUT,
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    TFT params: {n_params:,} | temporal={len(temporal_cols)}, static={len(static_cols)}", flush=True)

        t0 = time.time()
        model = train_dl_model(model, train_ld, val_ld, f"TFT_v2_{h}h", patience=TFT_PATIENCE)
        train_time = time.time() - t0

        # Evaluate
        model.eval()
        preds, targets_list = [], []
        with torch.no_grad():
            for t_b, s_b, y_b in test_ld:
                pred = model(t_b.to(DEVICE), s_b.to(DEVICE)).cpu().numpy().flatten()
                preds.extend(pred)
                targets_list.extend(y_b.numpy().flatten())

        preds_orig = np.clip(tgt_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten(), 0, None)
        targets_orig = np.clip(tgt_scaler.inverse_transform(np.array(targets_list).reshape(-1, 1)).flatten(), 0, None)

        mae = float(np.mean(np.abs(targets_orig - preds_orig)))
        rmse = float(np.sqrt(np.mean((targets_orig - preds_orig) ** 2)))
        mase = round(mae / persist_mae, 4) if persist_mae > 0 else float("inf")

        status = "✅" if mase < 1.0 else "❌"
        ref = v1_ref.get(f"{h}h", 99)
        delta = ((mase - ref) / ref) * 100
        print(f"    {status} TFT_v2 {h}h: MAE={mae:.3f}, MASE={mase:.3f} (v1={ref:.3f} Δ={delta:+.1f}%) ({train_time:.0f}s)",
              flush=True)

        all_results[f"{h}h"] = {
            "Persistence": {"mae": round(persist_mae, 4), "mase": 1.0},
            "TFT_v2": {
                "mae": round(mae, 4), "rmse": round(rmse, 4), "mase": mase,
                "params": n_params, "temporal_dim": len(temporal_cols),
                "static_dim": len(static_cols), "train_time_s": round(train_time, 1),
                "n_test": len(preds), "features": "v2_enhanced+cv",
            },
        }

    return all_results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("DL v3 — PCA Feature Selection (1h) + TFT Retrain (v2 Features)", flush=True)
    print(f"Device: {DEVICE} | Lookback: {LOOKBACK}h", flush=True)
    print(f"PCA: {PCA_VARIANCE_THRESHOLD*100}% variance | Top-N: {TOP_N_FEATURES}", flush=True)
    print(f"TFT: hidden={TFT_HIDDEN}, heads={TFT_HEADS}, dropout={TFT_DROPOUT}", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/3] Preparing data...", flush=True)
    df_feat = prepare_data()

    print("\n[2/3] Running experiments...", flush=True)
    pca_results = run_pca_experiment(df_feat)
    tft_results = run_tft_retrain(df_feat)

    # Summary
    print(f"\n{'═' * 70}", flush=True)
    print("[3/3] FINAL SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)

    print("\n  ── PCA/Feature Selection @ 1h ──", flush=True)
    print(f"  {'Method':<20} {'MASE':<10} {'MAE':<10} {'Dim':<6}", flush=True)
    print("  " + "─" * 50, flush=True)
    for name, r in sorted(pca_results.items(), key=lambda x: x[1].get("mase", 99)):
        dim = r.get("input_dim", "—")
        print(f"  {name:<20} {r.get('mase', '—'):<10} {r.get('mae', '—'):<10} {dim:<6}", flush=True)

    print("\n  ── TFT v2 vs v1 ──", flush=True)
    v1_ref = {"1h": 1.029, "6h": 0.821, "24h": 0.811}
    print(f"  {'Horizon':<10} {'v1 MASE':<10} {'v2 MASE':<10} {'Δ':<10}", flush=True)
    print("  " + "─" * 40, flush=True)
    for h in HORIZONS:
        tft = tft_results.get(f"{h}h", {}).get("TFT_v2", {})
        v1 = v1_ref.get(f"{h}h", 99)
        v2 = tft.get("mase", "—")
        delta = f"{((v2 - v1) / v1) * 100:+.1f}%" if isinstance(v2, float) else "—"
        print(f"  {h}h{'':<7} {v1:<10.3f} {str(v2):<10} {delta:<10}", flush=True)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"dl_v3_pca_tft_{ts}.json"

    def _c(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    output = {
        "meta": {
            "description": "PCA feature selection for 1h + TFT retrain with v2 features",
            "timestamp": datetime.now().isoformat(),
            "device": str(DEVICE),
            "pca_variance": PCA_VARIANCE_THRESHOLD,
            "top_n_candidates": TOP_N_FEATURES,
        },
        "pca_1h": pca_results,
        "tft_v2": tft_results,
    }
    with open(out, "w") as f:
        json.dump(output, f, indent=2, default=_c, ensure_ascii=False)
    print(f"\n  Results saved: {out}", flush=True)

    total = time.time() - t_start
    print(f"\nCOMPLETE — {total:.0f}s ({total / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
