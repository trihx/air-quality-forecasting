"""Temporal Fusion Transformer (TFT) Multi-Horizon Evaluation — Level 5.

Simplified TFT implementation using PyTorch (no pytorch-forecasting dependency).
Incorporates key TFT innovations:
  - Variable Selection Networks (VSN)
  - Gated Residual Networks (GRN)
  - Multi-head attention over temporal dimension
  - Static covariate encoders (calendar features)

Compared against: Persistence, LightGBM, GRU at 1h, 6h, 24h.

Usage:
    uv run python scripts/tft_multi_horizon.py 2>&1 | tee research/logs/tft_multi_horizon.log
"""

from __future__ import annotations

import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
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

warnings.filterwarnings("ignore")

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "tft"
HORIZONS = [1, 6, 24]

# Architecture
LOOKBACK = 72
HIDDEN_DIM = 32  # Smaller than GRU (64) to avoid overfit on small dataset
NUM_HEADS = 4  # Multi-head attention heads
DROPOUT = 0.1
BATCH_SIZE = 128  # Balanced for M1 Pro MPS
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 15

# Features
TEMPORAL_COLS = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]
STATIC_COLS_MAP = {
    "hour_sin": lambda idx: np.sin(2 * np.pi * idx.hour / 24),
    "hour_cos": lambda idx: np.cos(2 * np.pi * idx.hour / 24),
    "dow_sin": lambda idx: np.sin(2 * np.pi * idx.dayofweek / 7),
    "dow_cos": lambda idx: np.cos(2 * np.pi * idx.dayofweek / 7),
}


# ══════════════════════════════════════════════════════════════════════
# TFT Model Components (PyTorch)
# ══════════════════════════════════════════════════════════════════════


def _build_model_components():
    """Lazy import torch + define TFT components."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

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

    class TFTDataset(Dataset):
        """Dataset providing temporal features + static (calendar) features."""

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
            import torch

            i = self.indices[idx]
            t = self.temporal[i : i + self.lookback]
            s = self.static[i + self.lookback - 1]  # Static at prediction time
            y = self.targets[i + self.lookback + self.horizon - 1]
            return torch.FloatTensor(t), torch.FloatTensor(s), torch.FloatTensor([y])

    return {
        "SimplifiedTFT": SimplifiedTFT,
        "TFTDataset": TFTDataset,
        "DataLoader": DataLoader,
    }


# ══════════════════════════════════════════════════════════════════════
# Data Preparation
# ══════════════════════════════════════════════════════════════════════


def prepare_data():
    """Load → clean → impute → add static features."""
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

    # Add static calendar features
    idx = df_hybrid.index
    for name, func in STATIC_COLS_MAP.items():
        df_hybrid[name] = func(idx)

    return df_hybrid


# ══════════════════════════════════════════════════════════════════════
# Training + Evaluation
# ══════════════════════════════════════════════════════════════════════


def evaluate_horizon(df_hybrid, horizon, components):
    """Train and evaluate TFT at a specific horizon."""
    import torch
    import torch.nn as nn

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Training TFT ({horizon}h) on {device}...", flush=True)

    # Prepare arrays
    available_temporal = [c for c in TEMPORAL_COLS if c in df_hybrid.columns]
    static_cols = list(STATIC_COLS_MAP.keys())

    temporal = df_hybrid[available_temporal].values
    static = df_hybrid[static_cols].values
    target = df_hybrid[TARGET_COL].values
    is_imputed = (
        df_hybrid["is_imputed"].values if "is_imputed" in df_hybrid.columns else np.zeros(len(df_hybrid), dtype=bool)
    )

    n = len(temporal)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Scale temporal
    temp_scaler = StandardScaler()
    temporal_scaled = np.zeros_like(temporal)
    temporal_scaled[:train_end] = temp_scaler.fit_transform(temporal[:train_end])
    temporal_scaled[train_end:] = temp_scaler.transform(temporal[train_end:])

    # Scale target
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(target[:train_end].reshape(-1, 1))
    target_scaled = tgt_scaler.transform(target.reshape(-1, 1)).flatten()

    # Static features are already normalized (sin/cos in [-1, 1])

    # Real-only mask for test
    real_mask = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imputed[i]:
            real_mask[i] = True

    # Persistence baseline
    y_true_list, y_persist_list = [], []
    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        actual = target[i + horizon]
        persist = target[i]
        if not np.isnan(actual) and not np.isnan(persist):
            y_true_list.append(actual)
            y_persist_list.append(persist)

    y_true_arr = np.array(y_true_list)
    y_persist_arr = np.array(y_persist_list)
    persist_mae = float(np.mean(np.abs(y_true_arr - y_persist_arr)))
    print(f"    Persistence {horizon}h: MAE={persist_mae:.3f}", flush=True)

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

    print(f"    Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} (real only)", flush=True)

    if len(train_ds) < BATCH_SIZE:
        print("    ⚠️ Too few samples, skipping", flush=True)
        return {"Persistence": {"mae": persist_mae, "mase": 1.0}}

    # Build model
    SimplifiedTFT = components["SimplifiedTFT"]
    model = SimplifiedTFT(
        temporal_dim=len(available_temporal),
        static_dim=len(static_cols),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    TFT Parameters: {n_params:,}", flush=True)

    # Train
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

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for t_batch, s_batch, y_batch in test_loader:
            pred = model(t_batch.to(device), s_batch.to(device)).cpu().numpy().flatten()
            all_preds.extend(pred)
            all_targets.extend(y_batch.numpy().flatten())

    preds = tgt_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    targets = tgt_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()

    mae = float(np.mean(np.abs(targets - preds)))
    rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
    mase = round(mae / persist_mae, 4) if persist_mae > 0 else float("inf")

    status = "✅" if mase < 1.0 else "❌"
    print(
        f"    {status} TFT {horizon}h: MAE={mae:.3f} "
        f"RMSE={rmse:.3f} MASE={mase:.3f} "
        f"({train_time:.0f}s, {len(all_preds)} pts)",
        flush=True,
    )

    # ── Align predictions with test set indices ──
    avp_test_indices = []
    for i in range(val_end, n - horizon):
        if is_imputed[i + horizon]:
            continue
        avp_test_indices.append(i)

    tft_test_target_indices = set()
    for ds_idx in test_ds.indices:
        target_idx = ds_idx + LOOKBACK + horizon - 1
        tft_test_target_indices.add(target_idx)

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

    return {
        "metrics": {
            "Persistence": {"mae": round(persist_mae, 4), "rmse": 0, "mase": 1.0},
            "TFT": {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mase": mase,
                "params": n_params,
                "train_time_s": round(train_time, 1),
                "n_test": len(all_preds),
            },
        },
        "preds": {
            "TFT": aligned_preds
        }
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("TEMPORAL FUSION TRANSFORMER — Multi-Horizon Evaluation", flush=True)
    print(f"Horizons: {HORIZONS}h | Lookback: {LOOKBACK}h", flush=True)
    print(f"Architecture: hidden={HIDDEN_DIM}, heads={NUM_HEADS}, dropout={DROPOUT}", flush=True)
    print(f"Training: epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}", flush=True)
    print("=" * 70, flush=True)

    # Lazy build model components
    print("\n[1/4] Building TFT components...", flush=True)
    components = _build_model_components()

    # Prepare data
    print("\n[2/4] Preparing data...", flush=True)
    df_hybrid = prepare_data()
    print(f"  Data: {len(df_hybrid)} rows", flush=True)

    # Reference results
    ref_results = {
        "1h": {"gru_mase": 1.173, "lgbm_mase": 1.492},
        "6h": {"gru_mase": 0.812, "lgbm_mase": 0.745},
        "24h": {"gru_mase": 0.727, "lgbm_mase": 0.842},
    }

    all_results = {}
    all_preds = {}
    for h in HORIZONS:
        print(f"\n{'═' * 70}", flush=True)
        print(f"[3/4] HORIZON = {h}h", flush=True)
        print(f"{'═' * 70}", flush=True)

        results_h = evaluate_horizon(df_hybrid, h, components)
        all_results[f"{h}h"] = results_h["metrics"]
        all_preds[f"{h}h"] = results_h["preds"]

        # Compare
        r = ref_results[f"{h}h"]
        tft_mase = results_h["metrics"].get("TFT", {}).get("mase", 99)
        print(f"\n  📊 Comparison ({h}h):", flush=True)
        print(
            f"    Persistence: 1.000 | LightGBM: {r['lgbm_mase']:.3f} | GRU: {r['gru_mase']:.3f} | TFT: {tft_mase:.3f}",
            flush=True,
        )

    # Summary
    print(f"\n{'═' * 70}", flush=True)
    print("[4/4] FINAL COMPARISON SUMMARY", flush=True)
    print(f"{'═' * 70}", flush=True)
    print(f"\n{'Horizon':<10} {'Persist':<10} {'LightGBM':<10} {'GRU':<10} {'TFT':<10} {'Best':<12}", flush=True)
    print("─" * 62, flush=True)

    for h in HORIZONS:
        r = ref_results[f"{h}h"]
        tft = all_results[f"{h}h"].get("TFT", {})
        tft_mase = tft.get("mase", "—")
        vals = {"Persistence": 1.0, "LightGBM": r["lgbm_mase"], "GRU": r["gru_mase"]}
        if isinstance(tft_mase, float):
            vals["TFT"] = tft_mase
        best = min(vals, key=vals.get)
        tft_str = f"{tft_mase:.3f}" if isinstance(tft_mase, float) else "—"
        print(
            f"{h}h{'':<7} {'1.000':<10} {r['lgbm_mase']:<10.3f} {r['gru_mase']:<10.3f} {tft_str:<10} {best:<12}",
            flush=True,
        )

    # Save
    v8_dir = PROJECT_ROOT / "research" / "experiments" / "v8_final"
    v8_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = v8_dir / f"tft_multi_horizon_{ts}.json"
    preds_path = v8_dir / f"tft_preds_{ts}.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_convert, ensure_ascii=False)
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(all_preds, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}", flush=True)
    print(f"  Preds saved: {preds_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'═' * 70}", flush=True)
    print(f"COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)", flush=True)
    print(f"{'═' * 70}", flush=True)


if __name__ == "__main__":
    main()
