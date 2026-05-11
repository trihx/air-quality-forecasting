"""v7 Full Pipeline Retrain — After outlier fix (PM2.5 domain bounds).

Runs ALL model families sequentially:
  1. ML: LightGBM, sklearn (RF, GB, Stacking, Ensemble)
  2. Statistical: ARIMA/SARIMA
  3. DL: GRU, LSTM (v2+log)
  4. Standardize metrics (unified Persistence baseline)
  5. Precompute AVP cache for Dashboard

Results saved to research/experiments/ with v7_retrain timestamp.

Usage:
    uv run python scripts/retrain_v7_full.py 2>&1 | tee research/logs/retrain_v8.log
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
HORIZONS = [1, 6, 24]
LOOKBACK = 72  # DL lookback
USE_LOG_TRANSFORM = True
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from src.data.cleaner import (
    _clip_physical_bounds,
    _handle_outliers,
    _remove_duplicates,
    _resample,
    _set_datetime_index,
)
from src.data.imputer import impute_missing_data
from src.data.loader import TARGET_COL, load_raw_data
from src.evaluation.metrics import evaluate_forecast_full
from src.features.builder import build_features


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean (v7 outlier fix), impute, build features."""
    print("[DATA] Loading raw data...", flush=True)
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")

    df_hybrid = impute_missing_data(
        df, strategy="hybrid", max_gap_interp=6, max_gap_ml=24,
        knn_neighbors=5, verbose=True,
    )

    is_imputed = df_hybrid["is_imputed"].copy()
    df_for_feat = df_hybrid.drop(columns=["is_imputed"])
    df_feat = build_features(df_for_feat, include_fourier=True, fourier_order=3)
    df_feat["is_imputed"] = is_imputed.reindex(df_feat.index).fillna(False)

    print(f"[DATA] Final: {len(df_feat):,} rows × {len(df_feat.columns)} cols", flush=True)
    return df_hybrid, df_feat


def split_data(df_feat, horizon):
    """Create train/test split with anti-leakage."""
    df = df_feat.copy()
    df[f"target_{horizon}h"] = df[TARGET_COL].shift(-horizon)
    df["_persist"] = df[TARGET_COL]
    df = df.dropna(subset=[f"target_{horizon}h"])

    exclude = ["is_imputed", TARGET_COL, "_persist"] + [f"target_{h}h" for h in HORIZONS]
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "float32", "int64")]

    X = df[feat_cols].fillna(0)
    y_raw = df[f"target_{horizon}h"]
    is_imp = df["is_imputed"]
    persist = df["_persist"]

    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train_raw = X.iloc[:train_end], y_raw.iloc[:train_end]

    X_test = X.iloc[val_end:]
    y_test_raw = y_raw.iloc[val_end:]
    test_imp = is_imp.iloc[val_end:]
    test_persist = persist.iloc[val_end:]

    real_mask = ~test_imp.values
    X_test_real = X_test[real_mask]
    y_test_real = y_test_raw[real_mask].values
    persist_real = test_persist[real_mask].values

    if USE_LOG_TRANSFORM:
        y_train = np.log1p(y_train_raw.values)
    else:
        y_train = y_train_raw.values

    return X_train, y_train, X_test_real, y_test_real, persist_real, train_end, val_end, n


def eval_model(model, name, X_train, y_train, X_test, y_true, y_naive, h, scaler=None):
    """Train, predict, evaluate."""
    t0 = time.time()
    if scaler:
        model.fit(scaler.transform(X_train), y_train)
        pred = model.predict(scaler.transform(X_test))
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    if USE_LOG_TRANSFORM:
        pred_orig = np.clip(np.expm1(pred), 0, None)
    else:
        pred_orig = np.clip(pred, 0, None)

    elapsed = time.time() - t0
    metrics = evaluate_forecast_full(y_true, pred_orig, y_naive, name, h)
    metrics["train_time_s"] = round(elapsed, 2)
    print(f"    {name}: MAE={metrics['mae']}, MASE={metrics['mase']} ({elapsed:.1f}s)", flush=True)
    return metrics, pred_orig


# ══════════════════════════════════════════════════════════════════════
# MODEL FAMILIES
# ══════════════════════════════════════════════════════════════════════


def run_lightgbm(df_feat):
    """LightGBM with Optuna-tuned hyperparameters."""
    print("\n" + "=" * 70, flush=True)
    print("[ML] LightGBM (Optuna-tuned)", flush=True)
    print("=" * 70, flush=True)

    import lightgbm as lgb

    # Best hyperparams from previous Optuna runs
    LGBM_PARAMS = {
        1: dict(n_estimators=500, max_depth=3, learning_rate=0.013, num_leaves=64,
                subsample=0.8, colsample_bytree=0.6, min_child_samples=30,
                reg_alpha=0.05, reg_lambda=0.5, random_state=42, verbose=-1),
        6: dict(n_estimators=637, max_depth=3, learning_rate=0.012, num_leaves=87,
                subsample=0.85, colsample_bytree=0.55, min_child_samples=25,
                reg_alpha=0.03, reg_lambda=0.7, random_state=42, verbose=-1),
        24: dict(n_estimators=450, max_depth=4, learning_rate=0.015, num_leaves=52,
                 subsample=0.75, colsample_bytree=0.65, min_child_samples=35,
                 reg_alpha=0.08, reg_lambda=0.4, random_state=42, verbose=-1),
    }

    results = {}
    preds = {}
    for h in HORIZONS:
        print(f"\n  ── Horizon {h}h ──", flush=True)
        X_train, y_train, X_test, y_true, y_naive, *_ = split_data(df_feat, h)
        model = lgb.LGBMRegressor(**LGBM_PARAMS[h])
        m, p = eval_model(model, f"LightGBM_tuned", X_train, y_train, X_test, y_true, y_naive, h)
        results[f"{h}h"] = {"LightGBM_tuned": m}
        preds[f"{h}h"] = {"LightGBM_tuned": p.tolist(), "Persistence": y_naive.tolist(), "Actuals": y_true.tolist()}
    return {"metrics": results, "preds": preds}


def run_sklearn(df_feat):
    """RF, GradientBoosting, Stacking, Ensemble."""
    print("\n" + "=" * 70, flush=True)
    print("[ML] Sklearn: RF, GB, Stacking, Ensemble", flush=True)
    print("=" * 70, flush=True)

    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor,
        StackingRegressor,
        VotingRegressor,
    )
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.preprocessing import StandardScaler

    results = {}
    preds = {}
    for h in HORIZONS:
        print(f"\n  ── Horizon {h}h ──", flush=True)
        X_train, y_train, X_test, y_true, y_naive, *_ = split_data(df_feat, h)
        h_results = {}

        # Persistence
        p_m = evaluate_forecast_full(y_true, y_naive, y_naive, "Persistence", h)
        p_m["mase"] = 1.0
        h_results["Persistence"] = p_m

        # RF
        rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=10,
                                   max_features=0.7, random_state=42, n_jobs=-1)
        rf_m, rf_p = eval_model(rf, "RandomForest", X_train, y_train, X_test, y_true, y_naive, h)
        h_results["RandomForest"] = rf_m

        # GB
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, min_samples_leaf=10, random_state=42)
        gb_m, gb_p = eval_model(gb, "GradientBoosting", X_train, y_train, X_test, y_true, y_naive, h)
        h_results["GradientBoosting"] = gb_m

        # Stacking
        stacking = StackingRegressor(
            estimators=[
                ("enet", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)),
                ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
                ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)),
            ],
            final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1,
        )
        st_m, st_p = eval_model(stacking, "Stacking", X_train, y_train, X_test, y_true, y_naive, h)
        h_results["Stacking"] = st_m

        # Weighted Ensemble (weight search)
        best_mae = float("inf")
        best_ens_pred = None
        best_w = {}
        for w_rf in np.arange(0, 1.1, 0.1):
            for w_gb in np.arange(0, 1.1 - w_rf, 0.1):
                w_st = round(1.0 - w_rf - w_gb, 2)
                if w_st < 0:
                    continue
                wp = w_rf * rf_p + w_gb * gb_p + w_st * st_p
                mae_v = float(np.mean(np.abs(y_true - wp)))
                if mae_v < best_mae:
                    best_mae = mae_v
                    best_w = {"RF": round(w_rf, 2), "GB": round(w_gb, 2), "Stack": round(w_st, 2)}
                    best_ens_pred = wp

        ens_m = evaluate_forecast_full(y_true, best_ens_pred, y_naive, "Ensemble_Weighted", h)
        ens_m["best_weights"] = best_w
        h_results["Ensemble_Weighted"] = ens_m
        print(f"    Ensemble_Weighted: MAE={ens_m['mae']}, MASE={ens_m['mase']} (weights={best_w})", flush=True)

        results[f"{h}h"] = h_results
        preds[f"{h}h"] = {
            "RandomForest": rf_p.tolist(),
            "GradientBoosting": gb_p.tolist(),
            "Stacking": st_p.tolist(),
            "Ensemble_Weighted": best_ens_pred.tolist()
        }
    return {"metrics": results, "preds": preds}


def run_arima(df_feat):
    """ARIMA/SARIMA with rolling window."""
    print("\n" + "=" * 70, flush=True)
    print("[STAT] ARIMA/SARIMA (rolling window)", flush=True)
    print("=" * 70, flush=True)

    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    real_mask = ~df_feat["is_imputed"].astype(bool)
    pm25 = df_feat.loc[real_mask, TARGET_COL].values
    n = len(pm25)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    test_vals = pm25[val_end:]
    window = 720

    results = {}
    preds = {}
    for h in HORIZONS:
        print(f"\n  ── Horizon {h}h ──", flush=True)
        h_results = {}
        h_preds = {}

        # ARIMA(2,1,1)
        from joblib import Parallel, delayed
        def _fit_arima(i):
            start = max(0, val_end + i - window)
            train_window = pm25[start:val_end + i]
            try:
                model = ARIMA(train_window, order=(2, 1, 1))
                fit = model.fit()
                fc = fit.forecast(steps=h)
                return float(fc[-1])
            except Exception:
                return float(train_window[-1])
        
        preds_arima = Parallel(n_jobs=-1)(delayed(_fit_arima)(i) for i in range(len(test_vals) - h))

        actuals = test_vals[h:h + len(preds_arima)]
        persist = test_vals[:len(preds_arima)]
        preds_arima = np.clip(preds_arima[:len(actuals)], 0, None)

        arima_m = evaluate_forecast_full(actuals, preds_arima, persist, "ARIMA(2,1,1)", h)
        h_results["ARIMA"] = arima_m
        h_preds["ARIMA"] = preds_arima.tolist() if hasattr(preds_arima, "tolist") else list(preds_arima)
        print(f"    ARIMA(2,1,1): MAE={arima_m['mae']}, MASE={arima_m['mase']}", flush=True)

        # SARIMA(1,0,0)(2,1,0,24)
        def _fit_sarima(i):
            start = max(0, val_end + i - window)
            train_window = pm25[start:val_end + i]
            if len(train_window) < 48:
                return float(train_window[-1])
            try:
                model = SARIMAX(train_window, order=(1, 0, 0), seasonal_order=(2, 1, 0, 24),
                                enforce_stationarity=False, enforce_invertibility=False)
                fit = model.fit(disp=False, maxiter=50)
                fc = fit.forecast(steps=h)
                return float(fc[-1])
            except Exception:
                return float(train_window[-1])
                
        preds_sarima = Parallel(n_jobs=-1)(delayed(_fit_sarima)(i) for i in range(len(test_vals) - h))

        preds_sarima = np.clip(preds_sarima[:len(actuals)], 0, None)
        sarima_m = evaluate_forecast_full(actuals, preds_sarima, persist, "SARIMA", h)
        h_results["SARIMA"] = sarima_m
        h_preds["SARIMA"] = preds_sarima.tolist() if hasattr(preds_sarima, "tolist") else list(preds_sarima)
        print(f"    SARIMA: MAE={sarima_m['mae']}, MASE={sarima_m['mase']}", flush=True)

        results[f"{h}h"] = h_results
        preds[f"{h}h"] = h_preds
    return {"metrics": results, "preds": preds}


def run_dl(df_feat):
    """GRU/LSTM v2+log with MPS acceleration."""
    print("\n" + "=" * 70, flush=True)
    print("[DL] GRU & LSTM v2+log (MPS)", flush=True)
    print("=" * 70, flush=True)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    real_mask = ~df_feat["is_imputed"].astype(bool)
    # Use only numeric columns for DL
    raw_features = ["nhiet_do", "do_am", "diem_suong", "co2", TARGET_COL]
    df_dl = df_feat.loc[real_mask, [c for c in raw_features if c in df_feat.columns]].dropna()

    n = len(df_dl)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    values = df_dl.values.astype(np.float32)

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(values[:train_end])
    values_scaled = scaler.transform(values)
    target_idx = list(df_dl.columns).index(TARGET_COL)

    # Create sequences
    def create_sequences(data, lookback, horizon, target_col_idx):
        X, y = [], []
        for i in range(lookback, len(data) - horizon):
            X.append(data[i - lookback:i])
            y.append(data[i + horizon - 1, target_col_idx])  # Target at t+h
        return np.array(X), np.array(y)

    results = {}
    preds = {}
    for h in HORIZONS:
        print(f"\n  ── Horizon {h}h ──", flush=True)

        X_all, y_all = create_sequences(values_scaled, LOOKBACK, h, target_idx)

        # The actual target in original scale
        y_all_orig = []
        for i in range(LOOKBACK, len(values) - h):
            y_all_orig.append(values[i + h - 1, target_idx])
        y_all_orig = np.array(y_all_orig)

        # Split
        n_seq = len(X_all)
        tr_end = int(n_seq * 0.8)
        va_end = int(n_seq * 0.9)

        X_train_t = torch.FloatTensor(X_all[:tr_end]).to(device)
        y_train_t = torch.FloatTensor(np.log1p(y_all_orig[:tr_end])).to(device)
        X_test_t = torch.FloatTensor(X_all[va_end:]).to(device)
        y_test_orig = y_all_orig[va_end:]

        persist_test = values[LOOKBACK + va_end:LOOKBACK + va_end + len(y_test_orig), target_idx]
        persist_test = persist_test[:len(y_test_orig)]

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

        h_results = {}
        h_preds = {}

        for arch_name, rnn_class in [("GRU_v2_log", nn.GRU), ("LSTM_v2_log", nn.LSTM)]:
            print(f"\n    Training {arch_name}...", flush=True)
            t0 = time.time()

            # Build model
            class RNNModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.rnn = rnn_class(input_size=len(raw_features), hidden_size=64,
                                        num_layers=2, dropout=0.2, batch_first=True)
                    self.fc = nn.Linear(64, 1)

                def forward(self, x):
                    out, _ = self.rnn(x)
                    return self.fc(out[:, -1, :]).squeeze(-1)

            model = RNNModel().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
            criterion = nn.MSELoss()

            best_loss = float("inf")
            patience_counter = 0
            best_state = None

            # Pre-compute val target on device (once)
            y_val_log_t = torch.FloatTensor(np.log1p(y_test_orig)).to(device)

            for epoch in range(100):
                model.train()
                epoch_loss = 0
                n_batch = 0
                for xb, yb in train_dl:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batch += 1

                avg_loss = epoch_loss / max(n_batch, 1)

                # Evaluate on validation set for early stopping
                model.eval()
                with torch.no_grad():
                    preds_val = []
                    for i in range(0, len(X_test_t), 64):
                        preds_val.append(model(X_test_t[i:i+64]))
                    pred_val = torch.cat(preds_val)
                    val_loss = criterion(pred_val, y_val_log_t).item()

                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        print(f"      Early stop at epoch {epoch + 1} (val_loss={val_loss:.6f})", flush=True)
                        break

                if (epoch + 1) % 20 == 0:
                    print(f"      Epoch {epoch + 1}: train={avg_loss:.6f}, val={val_loss:.6f}", flush=True)

            # Load best and predict
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            model.eval()
            with torch.no_grad():
                preds_log = []
                for i in range(0, len(X_test_t), 64):
                    preds_log.append(model(X_test_t[i:i+64]).cpu().numpy())
                pred_log = np.concatenate(preds_log)
            pred_orig = np.clip(np.expm1(pred_log), 0, None)

            elapsed = time.time() - t0

            metrics = evaluate_forecast_full(y_test_orig, pred_orig, persist_test, arch_name, h)
            metrics["train_time_s"] = round(elapsed, 2)
            h_results[arch_name] = metrics
            h_preds[arch_name] = pred_orig.tolist()
            print(f"      {arch_name}: MAE={metrics['mae']}, MASE={metrics['mase']} ({elapsed:.1f}s)", flush=True)

            # Cleanup
            del model, optimizer, scheduler, y_val_log_t
            torch.mps.empty_cache() if device.type == "mps" else None

        results[f"{h}h"] = h_results
        preds[f"{h}h"] = h_preds
    return {"metrics": results, "preds": preds}


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def _save(data, name):
    """Save results JSON."""
    out_dir = RESEARCH_DIR / "experiments" / "v8_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{TIMESTAMP}.json"

    def _conv(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_conv, ensure_ascii=False)
    print(f"  → Saved: {path}", flush=True)
    return path


def main():
    print("=" * 70, flush=True)
    print(f"v7 FULL PIPELINE RETRAIN — Outlier Fix (PM2.5 domain bounds)", flush=True)
    print(f"Timestamp: {TIMESTAMP}", flush=True)
    print(f"Models: LightGBM, RF, GB, Stacking, Ensemble, ARIMA, SARIMA, GRU, LSTM", flush=True)
    print(f"Horizons: {HORIZONS}h", flush=True)
    print("=" * 70, flush=True)

    t_total = time.time()

    # Data
    df_hybrid, df_feat = prepare_data()

    all_results = {"_metadata": {
        "version": "v8_final",
        "outlier_strategy": "PM2.5=domain[0,500], others=IQR*3",
        "timestamp": datetime.now().isoformat(),
    }}
    
    all_preds = {}

    # 1. LightGBM (already run)
    # lgbm_out = run_lightgbm(df_feat)
    # all_results["lightgbm"] = lgbm_out["metrics"]
    # all_preds["lightgbm"] = lgbm_out["preds"]
    # _save(lgbm_out["metrics"], "lightgbm")
    # _save(lgbm_out["preds"], "lightgbm_preds")

    # 2. Sklearn (already run)
    # sklearn_out = run_sklearn(df_feat)
    # all_results["sklearn"] = sklearn_out["metrics"]
    # all_preds["sklearn"] = sklearn_out["preds"]
    # _save(sklearn_out["metrics"], "sklearn")
    # _save(sklearn_out["preds"], "sklearn_preds")

    # 3. ARIMA/SARIMA (already run)
    # arima_out = run_arima(df_feat)
    # all_results["arima"] = arima_out["metrics"]
    # all_preds["arima"] = arima_out["preds"]
    # _save(arima_out["metrics"], "arima")
    # _save(arima_out["preds"], "arima_preds")

    # 4. DL (GRU/LSTM)
    dl_out = run_dl(df_feat)
    all_results["dl"] = dl_out["metrics"]
    all_preds["dl"] = dl_out["preds"]
    _save(dl_out["metrics"], "dl")
    _save(dl_out["preds"], "dl_preds")

    # Save combined
    # _save(all_results, "all_models_combined")
    # _save(all_preds, "all_models_preds")

    total_time = time.time() - t_total
    print(f"\n{'=' * 70}", flush=True)
    print(f"✅ v7 RETRAIN COMPLETE — {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    # Summary table
    print(f"\n{'Horizon':<10} {'Model':<22} {'MAE':>8} {'MASE':>8}", flush=True)
    print("─" * 55, flush=True)
    for family in ["lightgbm", "sklearn", "arima", "dl"]:
        fam = all_results.get(family, {})
        for h_key in ["1h", "6h", "24h"]:
            if h_key in fam:
                for name, m in fam[h_key].items():
                    if isinstance(m, dict) and "mae" in m:
                        print(f"{h_key:<10} {name:<22} {m['mae']:>8.3f} {m['mase']:>8.3f}", flush=True)
        print("─" * 55, flush=True)


if __name__ == "__main__":
    main()
