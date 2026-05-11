"""v10 Ablation Study — Phase 2: Retrain 7 models on IQR-truncated data.

Retrains Persistence, ElasticNet, LightGBM, GRU, LSTM, TFT, Ensemble
on v10 ablation data (IQR for PM2.5) at 30m resolution.
Uses CUDA (RTX 3050) for DL models.

Usage:
    uv run python scripts/v10_ablation_retrain.py 2>&1 | tee research/logs/v10_retrain.log
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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import TARGET_COL
from src.evaluation.metrics import evaluate_forecast_full

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HORIZONS_HOURS = [1, 6, 24]
LOOKBACK_HOURS = 72
USE_LOG_TRANSFORM = True
FREQ = "30m"
RANDOM_SEED = 42

OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "v10_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "dataset" / "processed" / "v10_ablation"


def _save(data: dict, prefix: str):
    path = OUTPUT_DIR / f"{prefix}_{FREQ}_{TIMESTAMP}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path.name}", flush=True)


# ============================================================
# ML UTILITIES (from v9_retrain_lgbm)
# ============================================================

def split_data_segment_aware(df: pd.DataFrame, horizon_steps: int):
    """Split data with segment awareness and Test-on-Real-Only rule."""
    feature_cols = [c for c in df.columns if c not in [TARGET_COL, "segment_id", "is_imputed"]]
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Create target by shifting within segments
    df = df.copy()
    df["target"] = df.groupby("segment_id")[TARGET_COL].shift(-horizon_steps)
    df = df.dropna(subset=["target"])

    if USE_LOG_TRANSFORM:
        df["target"] = np.log1p(df["target"])

    X = df[feature_cols].values
    y = df["target"].values
    naive = df[TARGET_COL].values  # Persistence = current value
    actual_raw = np.expm1(y) if USE_LOG_TRANSFORM else y
    is_imp = df["is_imputed"].values if "is_imputed" in df.columns else np.zeros(len(df))

    # Recalculate split indices after dropna
    n2 = len(df)
    tr = int(n2 * 0.8)
    va = int(n2 * 0.9)

    # Test-on-Real-Only
    test_mask = np.arange(n2) >= va
    real_mask = ~is_imp.astype(bool)
    final_mask = test_mask & real_mask

    X_train, y_train = X[:tr], y[:tr]
    X_test = X[final_mask]
    y_true = actual_raw[final_mask]
    y_naive = naive[final_mask]

    return X_train, y_train, X_test, y_true, y_naive


# ============================================================
# 1. PERSISTENCE BASELINE
# ============================================================

def run_persistence(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"[1/7] Persistence Baseline", flush=True)
    print(f"{'='*60}", flush=True)

    steps_per_hour = 2  # 30m
    results = {}

    for h in HORIZONS_HOURS:
        steps = h * steps_per_hour
        _, _, _, y_true, y_naive = split_data_segment_aware(df, steps)
        metrics = evaluate_forecast_full(y_true, y_naive, y_naive, "Persistence_v10", h)
        results[f"{h}h"] = {"Persistence_v10": metrics}
        print(f"  {h}h: MAE={metrics['mae']}, MASE={metrics['mase']}", flush=True)

    return results


# ============================================================
# 2-3. SKLEARN MODELS (ElasticNet, LightGBM)
# ============================================================

def run_ml_models(df: pd.DataFrame) -> dict:
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'='*60}", flush=True)
    print(f"[2-3/7] ElasticNet + LightGBM", flush=True)
    print(f"{'='*60}", flush=True)

    steps_per_hour = 2
    results = {}
    preds_all = {}

    for h in HORIZONS_HOURS:
        steps = h * steps_per_hour
        print(f"\n  -- Horizon {h}h --", flush=True)
        X_train, y_train, X_test, y_true, y_naive = split_data_segment_aware(df, steps)

        h_results = {}
        h_preds = {"Actuals": y_true.tolist(), "Persistence": y_naive.tolist()}

        # ElasticNet
        scaler = StandardScaler()
        scaler.fit(X_train)
        en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED)
        en.fit(scaler.transform(X_train), y_train)
        pred = en.predict(scaler.transform(X_test))
        pred_orig = np.clip(np.expm1(pred), 0, None) if USE_LOG_TRANSFORM else np.clip(pred, 0, None)
        m = evaluate_forecast_full(y_true, pred_orig, y_naive, "ElasticNet_v10", h)
        h_results["ElasticNet_v10"] = m
        h_preds["ElasticNet_v10"] = pred_orig.tolist()
        print(f"    ElasticNet: MAE={m['mae']}, MASE={m['mase']}", flush=True)

        # LightGBM (lazy import)
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=RANDOM_SEED, verbose=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred_orig = np.clip(np.expm1(pred), 0, None) if USE_LOG_TRANSFORM else np.clip(pred, 0, None)
        m = evaluate_forecast_full(y_true, pred_orig, y_naive, "LightGBM_v10", h)
        h_results["LightGBM_v10"] = m
        h_preds["LightGBM_v10"] = pred_orig.tolist()
        print(f"    LightGBM:   MAE={m['mae']}, MASE={m['mase']}", flush=True)

        results[f"{h}h"] = h_results
        preds_all[f"{h}h"] = h_preds

    return results, preds_all


# ============================================================
# 4-5. DL MODELS (GRU, LSTM) — CUDA
# ============================================================

def run_dl_models(df_base: pd.DataFrame) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}", flush=True)
    print(f"[4-5/7] GRU + LSTM (Device: {DEVICE})", flush=True)
    print(f"{'='*60}", flush=True)

    class RNNModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, rnn_type="GRU"):
            super().__init__()
            if rnn_type == "GRU":
                self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            else:
                self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
            )

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    def create_sequences(df, lookback, horizon, feature_cols, target_col):
        values = df[feature_cols].values
        target_idx = feature_cols.index(target_col)
        segments = df["segment_id"].values
        X, y, persist, actual = [], [], [], []
        for i in range(lookback, len(df) - horizon):
            if segments[i - lookback] == segments[i + horizon - 1]:
                X.append(values[i - lookback:i])
                y.append(values[i + horizon - 1, target_idx])
                persist.append(values[i - 1, target_idx])
                actual.append(values[i + horizon - 1, target_idx])
        return np.array(X), np.array(y), np.array(persist), np.array(actual)

    steps_per_hour = 2
    lookback_steps = LOOKBACK_HOURS * steps_per_hour
    raw_features = ["nhiet_do", "do_am", "diem_suong", "co2", TARGET_COL]
    avail = [c for c in raw_features if c in df_base.columns]

    from sklearn.preprocessing import StandardScaler
    df_dl = df_base.dropna(subset=avail).copy()
    n = len(df_dl)
    train_end = int(n * 0.8)

    scaler = StandardScaler()
    scaler.fit(df_dl[avail].iloc[:train_end].values)
    df_scaled = df_dl.copy()
    df_scaled[avail] = scaler.transform(df_dl[avail].values)
    df_scaled["pm25_unscaled"] = df_dl[TARGET_COL]
    df_scaled["is_imputed"] = df_dl["is_imputed"]

    results = {}
    preds_all = {}

    for h in HORIZONS_HOURS:
        horizon_steps = h * steps_per_hour
        print(f"\n  -- Horizon {h}h ({horizon_steps} steps) --", flush=True)

        X_all, _, persist_all, _ = create_sequences(df_scaled, lookback_steps, horizon_steps, avail, TARGET_COL)
        _, _, _, y_unscaled = create_sequences(
            df_scaled, lookback_steps, horizon_steps, ["pm25_unscaled", "segment_id"], "pm25_unscaled"
        )
        _, _, _, is_imp_seq = create_sequences(
            df_scaled, lookback_steps, horizon_steps, ["is_imputed", "segment_id"], "is_imputed"
        )
        persist_unsc = persist_all * scaler.scale_[avail.index(TARGET_COL)] + scaler.mean_[avail.index(TARGET_COL)]

        n_seq = len(X_all)
        tr = int(n_seq * 0.8)
        va = int(n_seq * 0.9)

        y_train_t = torch.FloatTensor(np.log1p(y_unscaled[:tr]) if USE_LOG_TRANSFORM else y_unscaled[:tr])
        X_train_t = torch.FloatTensor(X_all[:tr])
        X_test_t = torch.FloatTensor(X_all[va:])
        y_test = y_unscaled[va:]
        persist_test = persist_unsc[va:]
        is_imp_test = is_imp_seq[va:].astype(bool)

        real_mask = ~is_imp_test
        X_test_t = X_test_t[real_mask]
        y_test = y_test[real_mask]
        persist_test = persist_test[real_mask]

        print(f"    Train: {tr:,} | Test (real): {len(y_test):,}", flush=True)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        h_results = {}
        h_preds = {"Actuals": y_test.tolist(), "Persistence": persist_test.tolist()}

        for rnn_type in ["GRU", "LSTM"]:
            torch.manual_seed(RANDOM_SEED)
            model = RNNModel(input_dim=len(avail), rnn_type=rnn_type).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(15):
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_test_t.to(DEVICE)).cpu().numpy()
            preds = np.clip(np.expm1(preds), 0, None) if USE_LOG_TRANSFORM else np.clip(preds, 0, None)

            name = f"{rnn_type}_v10"
            m = evaluate_forecast_full(y_test, preds, persist_test, name, h)
            h_results[name] = m
            h_preds[name] = preds.tolist()
            print(f"    {rnn_type}: MAE={m['mae']}, MASE={m['mase']}", flush=True)

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        results[f"{h}h"] = h_results
        preds_all[f"{h}h"] = h_preds

        del X_all, X_train_t, y_train_t, X_test_t
        import gc; gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results, preds_all


# ============================================================
# 6. TFT — CUDA
# ============================================================

def run_tft(df_base: pd.DataFrame) -> dict:
    import torch
    import torch.nn as nn
    import math
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}", flush=True)
    print(f"[6/7] TFT (Device: {DEVICE})", flush=True)
    print(f"{'='*60}", flush=True)

    # Import TFT architecture from v9
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from v9_retrain_tft import SimplifiedTFT, create_sequences_segment_aware

    steps_per_hour = 2
    lookback_steps = LOOKBACK_HOURS * steps_per_hour

    temporal_cols = ["nhiet_do", "do_am", "diem_suong", "co2", TARGET_COL]
    avail_temporal = [c for c in temporal_cols if c in df_base.columns]
    static_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
    avail_features = avail_temporal + static_cols

    df_dl = df_base.dropna(subset=avail_features).copy()
    n = len(df_dl)
    train_end = int(n * 0.8)

    scaler = StandardScaler()
    scaler.fit(df_dl[avail_temporal].iloc[:train_end].values)
    df_scaled = df_dl.copy()
    df_scaled[avail_temporal] = scaler.transform(df_dl[avail_temporal].values)
    df_scaled["pm25_unscaled"] = df_dl[TARGET_COL]
    df_scaled["is_imputed"] = df_dl["is_imputed"]

    results = {}
    preds_all = {}

    for h in HORIZONS_HOURS:
        horizon_steps = h * steps_per_hour
        print(f"\n  -- Horizon {h}h --", flush=True)

        X_all, _, persist_all, _ = create_sequences_segment_aware(
            df_scaled, lookback_steps, horizon_steps, avail_features, TARGET_COL
        )
        _, _, _, y_unscaled = create_sequences_segment_aware(
            df_scaled, lookback_steps, horizon_steps, ["pm25_unscaled", "segment_id"], "pm25_unscaled"
        )
        _, _, _, is_imp_seq = create_sequences_segment_aware(
            df_scaled, lookback_steps, horizon_steps, ["is_imputed", "segment_id"], "is_imputed"
        )
        persist_unsc = persist_all * scaler.scale_[avail_temporal.index(TARGET_COL)] + scaler.mean_[avail_temporal.index(TARGET_COL)]

        n_seq = len(X_all)
        tr = int(n_seq * 0.8)
        va = int(n_seq * 0.9)
        num_temp = len(avail_temporal)

        y_train_t = torch.FloatTensor(np.log1p(y_unscaled[:tr]))
        X_train_t = torch.FloatTensor(X_all[:tr])
        X_test_t = torch.FloatTensor(X_all[va:])

        X_train_temp = X_train_t[:, :, :num_temp]
        X_train_stat = X_train_t[:, -1, num_temp:]
        X_test_temp = X_test_t[:, :, :num_temp]
        X_test_stat = X_test_t[:, -1, num_temp:]

        y_test = y_unscaled[va:]
        persist_test = persist_unsc[va:]
        is_imp_test = is_imp_seq[va:].astype(bool)

        real_mask = ~is_imp_test
        X_test_temp = X_test_temp[real_mask]
        X_test_stat = X_test_stat[real_mask]
        y_test = y_test[real_mask]
        persist_test = persist_test[real_mask]

        print(f"    Train: {tr:,} | Test (real): {len(y_test):,}", flush=True)

        torch.manual_seed(RANDOM_SEED)
        tft = SimplifiedTFT(
            temporal_dim=num_temp, static_dim=len(static_cols),
            hidden_dim=32, num_heads=4, dropout=0.1
        ).to(DEVICE)

        optimizer = torch.optim.Adam(tft.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        train_ds = TensorDataset(X_train_temp, X_train_stat, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        tft.train()
        for epoch in range(15):
            for bt, bs, by in train_loader:
                bt, bs, by = bt.to(DEVICE), bs.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(tft(bt, bs), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tft.parameters(), 1.0)
                optimizer.step()

        tft.eval()
        with torch.no_grad():
            preds = tft(X_test_temp.to(DEVICE), X_test_stat.to(DEVICE)).cpu().numpy().flatten()
        preds = np.clip(np.expm1(preds), 0, None)

        m = evaluate_forecast_full(y_test, preds, persist_test, "TFT_v10", h)
        h_results = {"TFT_v10": m}
        h_preds = {"TFT_v10": preds.tolist(), "Actuals": y_test.tolist(), "Persistence": persist_test.tolist()}
        print(f"    TFT: MAE={m['mae']}, MASE={m['mase']}", flush=True)

        results[f"{h}h"] = h_results
        preds_all[f"{h}h"] = h_preds

        del X_all, X_train_t, X_test_t, tft
        import gc; gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results, preds_all


# ============================================================
# 7. ENSEMBLE (Weighted Average LSTM + LightGBM)
# ============================================================

def build_ensemble(dl_preds: dict, ml_preds: dict) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"[7/7] Ensemble Weighted (LSTM + LightGBM)", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}
    for h_key in ["1h", "6h", "24h"]:
        lstm = np.array(dl_preds.get(h_key, {}).get("LSTM_v10", []))
        lgbm = np.array(ml_preds.get(h_key, {}).get("LightGBM_v10", []))
        actuals = np.array(dl_preds.get(h_key, {}).get("Actuals", []))
        persist = np.array(dl_preds.get(h_key, {}).get("Persistence", []))

        if len(lstm) == 0 or len(lgbm) == 0:
            print(f"  [{h_key}] SKIP — missing predictions", flush=True)
            continue

        n = min(len(lstm), len(lgbm))
        ensemble = 0.5 * lstm[-n:] + 0.5 * lgbm[-n:]
        act = actuals[-n:]
        per = persist[-n:]

        h_hours = int(h_key.replace("h", ""))
        m = evaluate_forecast_full(act, ensemble, per, "Ensemble_v10", h_hours)
        results[h_key] = {"Ensemble_v10": m}
        print(f"  [{h_key}] Ensemble: MAE={m['mae']}, MASE={m['mase']}", flush=True)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70, flush=True)
    print(f"v10 ABLATION STUDY — Full Retrain Pipeline", flush=True)
    print(f"Timestamp: {TIMESTAMP}", flush=True)
    print(f"Device: CUDA={__import__('torch').cuda.is_available()}", flush=True)
    print("=" * 70, flush=True)

    t_total = time.time()

    # Load v10 ablation data
    feat_path = DATA_DIR / "marts_features_30m.csv"
    base_path = DATA_DIR / "marts_features_30m_base.csv"
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found. Run v10_ablation_rebuild_data.py first!")
        sys.exit(1)

    df_feat = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    df_base = pd.read_csv(base_path, index_col=0, parse_dates=True)
    print(f"Data loaded: features={df_feat.shape}, base={df_base.shape}", flush=True)

    all_results = {}

    # 1. Persistence
    persist_results = run_persistence(df_feat)
    for h_key, v in persist_results.items():
        all_results.setdefault(h_key, {}).update(v)

    # 2-3. ML models
    ml_results, ml_preds = run_ml_models(df_feat)
    for h_key, v in ml_results.items():
        all_results.setdefault(h_key, {}).update(v)

    # 4-5. DL models (GRU, LSTM)
    dl_results, dl_preds = run_dl_models(df_base)
    for h_key, v in dl_results.items():
        all_results.setdefault(h_key, {}).update(v)

    # 6. TFT
    tft_results, tft_preds = run_tft(df_base)
    for h_key, v in tft_results.items():
        all_results.setdefault(h_key, {}).update(v)

    # 7. Ensemble
    ens_results = build_ensemble(dl_preds, ml_preds)
    for h_key, v in ens_results.items():
        all_results.setdefault(h_key, {}).update(v)

    # Save consolidated results
    _save(all_results, "ablation_metrics")

    elapsed = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"v10 ABLATION COMPLETE — {elapsed:.0f}s total", flush=True)
    print(f"{'='*70}", flush=True)

    # Print summary table
    print(f"\n{'Model':<25} {'1h MASE':<12} {'6h MASE':<12} {'24h MASE':<12}", flush=True)
    print("-" * 60, flush=True)
    model_names = set()
    for h_key in ["1h", "6h", "24h"]:
        model_names.update(all_results.get(h_key, {}).keys())
    for name in sorted(model_names):
        row = f"{name:<25}"
        for h_key in ["1h", "6h", "24h"]:
            mase = all_results.get(h_key, {}).get(name, {}).get("mase", "—")
            row += f" {mase:<12}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
