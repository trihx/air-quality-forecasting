"""v9 Phase 5B.4 — Retrain DL Models (GRU/LSTM).

Retrains PyTorch models (GRU, LSTM v2+log) on high-resolution segment-aware data.
Uses a segment-aware sequence generator to prevent False Continuity.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import TARGET_COL
from src.evaluation.metrics import evaluate_forecast_full
from scripts.v9_retrain_lgbm import _save

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HORIZONS_HOURS = [1, 6, 24]
LOOKBACK_HOURS = 72
USE_LOG_TRANSFORM = True

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, rnn_type="GRU"):
        super().__init__()
        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        # Take the output of the last time step
        return self.fc(out[:, -1, :]).squeeze(-1)


def create_sequences_segment_aware(df: pd.DataFrame, lookback: int, horizon: int, feature_cols: list[str], target_col: str):
    """Create sequences strictly within segments to prevent False Continuity."""
    values = df[feature_cols].values
    target_idx = feature_cols.index(target_col)
    segments = df['segment_id'].values

    X, y, persist, actual = [], [], [], []

    for i in range(lookback, len(df) - horizon):
        # Only create sequence if the entire window (from start of lookback to target horizon)
        # falls within the same contiguous segment.
        if segments[i - lookback] == segments[i + horizon - 1]:
            X.append(values[i - lookback:i])
            y.append(values[i + horizon - 1, target_idx])
            persist.append(values[i - 1, target_idx])  # Persistence is value at time t
            actual.append(values[i + horizon - 1, target_idx])

    return np.array(X), np.array(y), np.array(persist), np.array(actual)


def train_model(model, train_loader, epochs=15, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def evaluate_dl(model, X_tensor, y_actual, persist_vals, name, h_hours):
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(DEVICE)
        preds = model(X_tensor).cpu().numpy()
        del X_tensor
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
    if USE_LOG_TRANSFORM:
        preds = np.clip(np.expm1(preds), 0, None)
    else:
        preds = np.clip(preds, 0, None)

    elapsed = time.time() - t0
    metrics = evaluate_forecast_full(y_actual, preds, persist_vals, name, h_hours)
    metrics["train_time_s"] = round(elapsed, 2)
    print(f"    {name}: MAE={metrics['mae']}, MASE={metrics['mase']}", flush=True)
    return metrics, preds


def run_pipeline(freq: str, df_feat: pd.DataFrame):
    print(f"\n{'=' * 70}", flush=True)
    print(f"[DL] GRU & LSTM - {freq} Dataset", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"{'=' * 70}", flush=True)

    steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(freq.replace("m", "min")))
    lookback_steps = LOOKBACK_HOURS * steps_per_hour

    # DL uses raw numerical features, avoiding lagged tabular features to prevent redundancy
    raw_features = ["nhiet_do", "do_am", "diem_suong", "co2", TARGET_COL]
    avail_features = [c for c in raw_features if c in df_feat.columns]

    # DO NOT filter real data here. Deep Learning needs continuous sequences!
    # Imputed data will be used for training, but filtered out during test/evaluation.
    df_dl = df_feat.dropna(subset=avail_features).copy()

    n = len(df_dl)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # We scale the features using train data stats
    scaler = StandardScaler()
    scaler.fit(df_dl[avail_features].iloc[:train_end].values)

    df_scaled = df_dl.copy()
    df_scaled[avail_features] = scaler.transform(df_dl[avail_features].values)
    df_scaled["pm25_unscaled"] = df_dl[TARGET_COL]
    df_scaled["is_imputed"] = df_dl["is_imputed"]
    
    results = {}
    preds = {}

    for h_hours in HORIZONS_HOURS:
        horizon_steps = h_hours * steps_per_hour
        print(f"\n  ── Horizon {h_hours}h ({horizon_steps} steps, lookback {lookback_steps} steps) ──", flush=True)

        X_all, y_all_scaled, persist_all, y_actual = create_sequences_segment_aware(
            df_scaled, lookback_steps, horizon_steps, avail_features, TARGET_COL
        )
        
        # The target in `create_sequences_segment_aware` will be scaled because it uses avail_features.
        # We need the unscaled actual targets. Let's get it by adding the unscaled target to the function or doing it here.
        # Actually, let's just re-extract the actual unscaled targets.
        _, _, _, y_actual_unscaled = create_sequences_segment_aware(
            df_scaled, lookback_steps, horizon_steps, ["pm25_unscaled", "segment_id"], "pm25_unscaled"
        )
        _, _, _, is_imputed_seq = create_sequences_segment_aware(
            df_scaled, lookback_steps, horizon_steps, ["is_imputed", "segment_id"], "is_imputed"
        )
        
        persist_unscaled = persist_all * scaler.scale_[avail_features.index(TARGET_COL)] + scaler.mean_[avail_features.index(TARGET_COL)]

        # Split sequences chronologically
        n_seq = len(X_all)
        tr_end = int(n_seq * 0.8)
        va_end = int(n_seq * 0.9)

        if USE_LOG_TRANSFORM:
            # We log-transform the UNSCALED target
            y_train_t = torch.FloatTensor(np.log1p(y_actual_unscaled[:tr_end]))
        else:
            y_train_t = torch.FloatTensor(y_actual_unscaled[:tr_end])

        X_train_t = torch.FloatTensor(X_all[:tr_end])
        X_test_t = torch.FloatTensor(X_all[va_end:])
        y_test_orig = y_actual_unscaled[va_end:]
        persist_test = persist_unscaled[va_end:]
        is_imputed_test = is_imputed_seq[va_end:].astype(bool)

        # Apply Test-on-Real-Only Rule: Filter out imputed targets from test set
        real_test_mask = ~is_imputed_test
        X_test_t = X_test_t[real_test_mask]
        y_test_orig = y_test_orig[real_test_mask]
        persist_test = persist_test[real_test_mask]

        print(f"    Train sequences: {tr_end:,} | Test sequences (real only): {len(y_test_orig):,}")

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        h_results = {}
        h_preds = {}

        # 1. GRU
        gru = RNNModel(input_dim=len(avail_features), rnn_type="GRU").to(DEVICE)
        train_model(gru, train_loader, epochs=15)
        m_gru, p_gru = evaluate_dl(gru, X_test_t, y_test_orig, persist_test, "GRU_v9", h_hours)
        h_results["GRU_v9"] = m_gru
        h_preds["GRU_v9"] = p_gru.tolist()

        # 2. LSTM
        lstm = RNNModel(input_dim=len(avail_features), rnn_type="LSTM").to(DEVICE)
        train_model(lstm, train_loader, epochs=15)
        m_lstm, p_lstm = evaluate_dl(lstm, X_test_t, y_test_orig, persist_test, "LSTM_v9", h_hours)
        h_results["LSTM_v9"] = m_lstm
        h_preds["LSTM_v9"] = p_lstm.tolist()

        h_preds["Persistence"] = persist_test.tolist()
        h_preds["Actuals"] = y_test_orig.tolist()

        results[f"{h_hours}h"] = h_results
        preds[f"{h_hours}h"] = h_preds
        
        del X_all, y_all_scaled, persist_all, y_actual, X_train_t, y_train_t, X_test_t, train_ds, train_loader, gru, lstm
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save
    _save(results, "dl_metrics", freq)
    _save(preds, "dl_preds", freq)
    print(f"\n[v9] ✅ DL ({freq}) saved.", flush=True)


def main():
    print("=" * 70, flush=True)
    print(f"v9 DL RETRAIN — Segment-Aware", flush=True)
    print(f"Timestamp: {TIMESTAMP}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()

    data_dir = PROJECT_ROOT / "dataset" / "processed"

    # Process 30m
    path_30m = data_dir / "marts_features_30m.csv"
    if path_30m.exists():
        df_30m = pd.read_csv(path_30m, index_col=0, parse_dates=True)
        run_pipeline("30m", df_30m)
    else:
        print(f"⚠️ {path_30m} not found.")

    # Process 15m
    path_15m = data_dir / "marts_features_15m.csv"
    if path_15m.exists():
        df_15m = pd.read_csv(path_15m, index_col=0, parse_dates=True)
        run_pipeline("15m", df_15m)
    else:
        print(f"⚠️ {path_15m} not found.")

    print(f"\n✅ Total DL time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
