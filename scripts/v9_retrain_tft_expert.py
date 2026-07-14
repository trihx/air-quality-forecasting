"""v9 Phase 5B-Expert — TFT Expert Pipeline.

Uses the BASE dataset (more rows, no lag/rolling warmup loss).
Temporal: 5 raw sensors. Static: calendar+Fourier features.
Ablation study counterpart to v9_retrain_tft.py (Fair Pipeline).
"""

from __future__ import annotations

import gc
import json
import math
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

# Import TFT architecture from existing script
from scripts.v9_retrain_tft import (
    SimplifiedTFT,
    create_sequences_segment_aware,
)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HORIZONS_HOURS = [1, 6, 24]
LOOKBACK_HOURS = 72
USE_LOG_TRANSFORM = True

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


def run_pipeline(freq: str, df_base: pd.DataFrame):
    print(f"\n{'=' * 70}", flush=True)
    print(f"[TFT-EXPERT] Temporal Fusion Transformer - {freq} Base Dataset", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"{'=' * 70}", flush=True)

    steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(freq.replace("m", "min")))
    lookback_steps = LOOKBACK_HOURS * steps_per_hour

    # TFT Expert: temporal = raw sensors, static = calendar + Fourier
    temporal_cols = ["nhiet_do", "do_am", "diem_suong", "co2", TARGET_COL]
    avail_temporal = [c for c in temporal_cols if c in df_base.columns]

    static_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
    # Also add Fourier as static (they repeat cyclically, known future)
    fourier_cols = [c for c in df_base.columns if c.startswith("fourier_")]
    all_static = [c for c in static_cols + fourier_cols if c in df_base.columns]

    avail_features = avail_temporal + all_static
    print(f"  Temporal features ({len(avail_temporal)}): {avail_temporal}", flush=True)
    print(f"  Static features ({len(all_static)}): {all_static}", flush=True)

    df_dl = df_base.dropna(subset=[c for c in temporal_cols if c in df_base.columns]).copy()
    print(f"  Dataset rows: {len(df_dl):,} (vs Fair pipeline's tabular-engineered dataset)", flush=True)

    n = len(df_dl)
    train_end = int(n * 0.8)

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

        pm25_idx = avail_temporal.index(TARGET_COL)
        persist_unscaled = persist_all * scaler.scale_[pm25_idx] + scaler.mean_[pm25_idx]

        n_seq = len(X_all)
        tr_end = int(n_seq * 0.8)
        va_end = int(n_seq * 0.9)

        if USE_LOG_TRANSFORM:
            y_train_t = torch.FloatTensor(np.log1p(y_actual_unscaled[:tr_end]))
        else:
            y_train_t = torch.FloatTensor(y_actual_unscaled[:tr_end])

        X_train_t = torch.FloatTensor(X_all[:tr_end])
        X_test_t = torch.FloatTensor(X_all[va_end:])

        num_temp = len(avail_temporal)

        X_train_temp = X_train_t[:, :, :num_temp]
        X_train_stat = X_train_t[:, -1, num_temp:]

        X_test_temp = X_test_t[:, :, :num_temp]
        X_test_stat = X_test_t[:, -1, num_temp:]

        y_test_orig = y_actual_unscaled[va_end:]
        persist_test = persist_unscaled[va_end:]
        is_imputed_test = is_imputed_seq[va_end:].astype(bool)

        # Test-on-Real-Only Rule
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

        tft = SimplifiedTFT(
            temporal_dim=len(avail_temporal),
            static_dim=len(all_static),
            hidden_dim=32,
            num_heads=4,
            dropout=0.1
        ).to(DEVICE)

        train_tft(tft, train_loader, epochs=15)
        m_tft, p_tft = evaluate_tft(tft, X_test_temp, X_test_stat, y_test_orig, persist_test, "TFT_v9_expert", h_hours)

        h_results["TFT_v9_expert"] = m_tft
        h_preds["TFT_v9_expert"] = p_tft.tolist()
        h_preds["Persistence"] = persist_test.tolist()
        h_preds["Actuals"] = y_test_orig.tolist()

        results[f"{h_hours}h"] = h_results
        preds[f"{h_hours}h"] = h_preds

        del X_all, y_all_scaled, persist_all, y_actual, X_train_t, y_train_t, X_test_t, train_ds, train_loader, tft
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    _save(results, "tft_expert_metrics", freq)
    _save(preds, "tft_expert_preds", freq)
    print(f"\n[v9] ✅ TFT Expert ({freq}) saved.", flush=True)


def main():
    print("=" * 70, flush=True)
    print(f"v9 TFT EXPERT RETRAIN — Ablation Study (Base Dataset)", flush=True)
    print(f"Timestamp: {TIMESTAMP}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    data_dir = PROJECT_ROOT / "dataset" / "processed"

    path_30m = data_dir / "marts_features_30m_base.csv"
    if path_30m.exists():
        df_30m = pd.read_csv(path_30m, index_col=0, parse_dates=True)
        run_pipeline("30m", df_30m)
    else:
        print(f"⚠️ {path_30m} not found. Run v9_rebuild_data.py first.")

    path_15m = data_dir / "marts_features_15m_base.csv"
    if path_15m.exists():
        df_15m = pd.read_csv(path_15m, index_col=0, parse_dates=True)
        run_pipeline("15m", df_15m)
    else:
        print(f"⚠️ {path_15m} not found. Run v9_rebuild_data.py first.")

    print(f"\n✅ Total TFT Expert time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
