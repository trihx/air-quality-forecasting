"""v9 Phase 5B.5 — Retrain TFT Model.

Retrains Temporal Fusion Transformer on high-resolution segment-aware data.
Uses a segment-aware sequence generator to prevent False Continuity.
"""

from __future__ import annotations

import json
import sys
import time
import math
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


class GatedLinearUnit(nn.Module):
    """GLU activation: splits input, applies sigmoid gate."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))

class GatedResidualNetwork(nn.Module):
    """GRN: core building block of TFT."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None

    def forward(self, x, context=None):
        residual = self.skip(x)
        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            h = h + self.context_fc(context)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        return self.layer_norm(h + residual)

class VariableSelectionNetwork(nn.Module):
    """VSN: learns which variables are important."""

    def __init__(self, input_dim, num_vars, hidden_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.num_vars = num_vars
        self.var_dim = input_dim // num_vars

        # Per-variable GRNs
        self.var_grns = nn.ModuleList(
            [GatedResidualNetwork(self.var_dim, hidden_dim, hidden_dim, dropout) for _ in range(num_vars)]
        )

        # Softmax weights
        self.weight_grn = GatedResidualNetwork(input_dim, hidden_dim, num_vars, dropout, context_dim=context_dim)

    def forward(self, x, context=None):
        # x: (batch, seq, input_dim) or (batch, input_dim)
        has_time = x.dim() == 3

        # Variable selection weights
        flat = x.reshape(x.shape[0] * x.shape[1], -1) if has_time else x

        ctx = context.repeat(x.shape[1], 1) if (context is not None and has_time) else context
        weights = torch.softmax(
            self.weight_grn(flat, context=ctx),
            dim=-1,
        )

        # Process each variable through its GRN
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            start = i * self.var_dim
            end = start + self.var_dim
            var_input = x[:, :, start:end].reshape(-1, self.var_dim) if has_time else x[:, start:end]
            var_outputs.append(grn(var_input))

        var_outputs = torch.stack(var_outputs, dim=-1)  # (batch*seq, hidden, num_vars)
        weights = weights.unsqueeze(1)  # (batch*seq, 1, num_vars)
        combined = (var_outputs * weights).sum(dim=-1)  # (batch*seq, hidden)

        if has_time:
            combined = combined.reshape(x.shape[0], x.shape[1], -1)

        return combined

class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretable weights."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(context), attn

class SimplifiedTFT(nn.Module):
    """Simplified Temporal Fusion Transformer.

    Key components:
    1. Static covariate encoder (GRN)
    2. Variable Selection Network for temporal inputs
    3. LSTM encoder
    4. Multi-head attention (interpretable)
    5. Gated skip connections
    6. Dense output
    """

    def __init__(self, temporal_dim, static_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Static encoder
        self.static_grn = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout)

        # Temporal input projection
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)

        # LSTM encoder (replaces full VSN for simplicity on small data)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )

        # Post-LSTM gate
        self.post_lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        # Multi-head attention
        self.attention = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.post_attn_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)

        # Output
        self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal, static):
        # Static context
        static_ctx = self.static_grn(static)  # (batch, hidden)

        # Temporal projection
        temporal_emb = self.temporal_proj(temporal)  # (batch, seq, hidden)

        # Add static context to temporal
        temporal_emb = temporal_emb + static_ctx.unsqueeze(1)

        # LSTM encoding
        lstm_out, _ = self.lstm(temporal_emb)

        # Post-LSTM gating with skip connection
        gated = self.post_lstm_gate(lstm_out)
        lstm_enriched = self.post_lstm_norm(gated + temporal_emb)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_enriched, lstm_enriched, lstm_enriched)
        attn_out = self.post_attn_gate(attn_out)
        enriched = self.post_attn_norm(attn_out + lstm_enriched)

        # Take last timestep
        last = enriched[:, -1, :]  # (batch, hidden)
        out = self.output_grn(last)
        out = self.dropout(out)
        return self.fc_out(out)


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

def train_tft(model, train_loader, epochs=15, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for batch_t, batch_s, batch_y in train_loader:
            batch_t, batch_s, batch_y = batch_t.to(DEVICE), batch_s.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_t, batch_s)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

def evaluate_tft(model, X_temp_t, X_stat_t, y_actual, persist_vals, name, h_hours):
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        X_temp_t = X_temp_t.to(DEVICE)
        X_stat_t = X_stat_t.to(DEVICE)
        preds = model(X_temp_t, X_stat_t).cpu().numpy().flatten()
        del X_temp_t, X_stat_t
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
    print(f"[DL] Temporal Fusion Transformer - {freq} Dataset", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"{'=' * 70}", flush=True)

    steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(freq.replace("m", "min")))
    lookback_steps = LOOKBACK_HOURS * steps_per_hour

    temporal_cols = ["nhiet_do", "do_am", "diem_suong", "co2", TARGET_COL]
    avail_temporal = [c for c in temporal_cols if c in df_feat.columns]
    static_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]

    avail_features = avail_temporal + static_cols

    # DO NOT filter real data here. Temporal Fusion Transformer needs continuous sequences!
    # Imputed data will be used for training, but filtered out during test/evaluation.
    df_dl = df_feat.dropna(subset=avail_features).copy()

    n = len(df_dl)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    scaler = StandardScaler()
    scaler.fit(df_dl[avail_temporal].iloc[:train_end].values)

    df_scaled = df_dl.copy()
    df_scaled[avail_temporal] = scaler.transform(df_dl[avail_temporal].values)
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
        
        _, _, _, y_actual_unscaled = create_sequences_segment_aware(
            df_scaled, lookback_steps, horizon_steps, ["pm25_unscaled", "segment_id"], "pm25_unscaled"
        )
        _, _, _, is_imputed_seq = create_sequences_segment_aware(
            df_scaled, lookback_steps, horizon_steps, ["is_imputed", "segment_id"], "is_imputed"
        )
        
        persist_unscaled = persist_all * scaler.scale_[avail_temporal.index(TARGET_COL)] + scaler.mean_[avail_temporal.index(TARGET_COL)]

        n_seq = len(X_all)
        tr_end = int(n_seq * 0.8)
        va_end = int(n_seq * 0.9)

        if USE_LOG_TRANSFORM:
            y_train_t = torch.FloatTensor(np.log1p(y_actual_unscaled[:tr_end]))
        else:
            y_train_t = torch.FloatTensor(y_actual_unscaled[:tr_end])

        X_train_t = torch.FloatTensor(X_all[:tr_end])
        X_test_t = torch.FloatTensor(X_all[va_end:])
        
        # Split into temporal and static
        num_temp = len(avail_temporal)
        
        X_train_temp = X_train_t[:, :, :num_temp]
        X_train_stat = X_train_t[:, -1, num_temp:]  # Take static from last timestep
        
        X_test_temp = X_test_t[:, :, :num_temp]
        X_test_stat = X_test_t[:, -1, num_temp:]

        y_test_orig = y_actual_unscaled[va_end:]
        persist_test = persist_unscaled[va_end:]
        is_imputed_test = is_imputed_seq[va_end:].astype(bool)

        # Apply Test-on-Real-Only Rule
        real_test_mask = ~is_imputed_test
        X_test_temp = X_test_temp[real_test_mask]
        X_test_stat = X_test_stat[real_test_mask]
        y_test_orig = y_test_orig[real_test_mask]
        persist_test = persist_test[real_test_mask]

        print(f"    Train sequences: {tr_end:,} | Test sequences (real only): {len(y_test_orig):,}")

        train_ds = TensorDataset(X_train_temp, X_train_stat, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        h_results = {}
        h_preds = {}

        # TFT
        tft = SimplifiedTFT(
            temporal_dim=len(avail_temporal),
            static_dim=len(static_cols),
            hidden_dim=32,
            num_heads=4,
            dropout=0.1
        ).to(DEVICE)
        
        train_tft(tft, train_loader, epochs=15)
        m_tft, p_tft = evaluate_tft(tft, X_test_temp, X_test_stat, y_test_orig, persist_test, "TFT_v9", h_hours)
        
        h_results["TFT_v9"] = m_tft
        h_preds["TFT_v9"] = p_tft.tolist()
        
        h_preds["Persistence"] = persist_test.tolist()
        h_preds["Actuals"] = y_test_orig.tolist()

        results[f"{h_hours}h"] = h_results
        preds[f"{h_hours}h"] = h_preds

        del X_all, y_all_scaled, persist_all, y_actual, X_train_t, y_train_t, X_test_t, train_ds, train_loader, tft
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save
    _save(results, "tft_metrics", freq)
    _save(preds, "tft_preds", freq)
    print(f"\n[v9] ✅ TFT ({freq}) saved.", flush=True)

def main():
    print("=" * 70, flush=True)
    print(f"v9 TFT RETRAIN — Segment-Aware", flush=True)
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

    print(f"\n✅ Total TFT time: {time.time() - t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
