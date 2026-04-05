"""Interactive Model Training Service for Dashboard.

Provides trainers for LightGBM and GRU that:
  1. Accept hyperparameters from the user
  2. Train on the hybrid dataset
  3. Return metrics (MAE, MASE, RMSE, R²)
  4. Optionally save the model

Usage:
    trainer = LightGBMTrainer(horizon=6, params={...})
    result = trainer.train(progress_callback=st_callback)
    trainer.save_model()
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.loader import TARGET_COL

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"
USER_MODELS_DIR = PROJECT_ROOT / "models" / "user_trained"
DASHBOARD_RUNS_DIR = PROJECT_ROOT / "research" / "experiments" / "dashboard_runs"
LOOKBACK = 72
FEATURE_COLS_DL = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]


def _prepare_hybrid_data():
    """Load and prepare hybrid dataset."""
    from src.data.cleaner import (
        _clip_physical_bounds,
        _handle_outliers,
        _remove_duplicates,
        _resample,
        _set_datetime_index,
    )
    from src.data.imputer import impute_missing_data
    from src.data.loader import load_raw_data

    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")
    return impute_missing_data(
        df, strategy="hybrid",
        max_gap_interp=6, max_gap_ml=24, knn_neighbors=5,
        verbose=False,
    )


def get_default_params(model_type: str) -> dict:
    """Get recommended default hyperparameters."""
    if model_type == "LightGBM":
        return {
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.013,
            "num_leaves": 64,
            "subsample": 0.80,
            "colsample_bytree": 0.60,
            "min_child_samples": 30,
            "reg_alpha": 0.05,
            "reg_lambda": 0.50,
        }
    elif model_type == "GRU":
        return {
            "lookback": 72,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "batch_size": 256,
            "learning_rate": 0.001,
            "epochs": 100,
            "patience": 10,
        }
    return {}


class LightGBMTrainer:
    """Train LightGBM with user-specified params."""

    def __init__(self, horizon: int, params: dict):
        self.horizon = horizon
        self.params = params
        self.model = None
        self.metrics = None
        self._feature_cols = None
        self._train_time = None

    def train(self, progress_callback=None) -> dict:
        """Train and evaluate LightGBM.

        Args:
            progress_callback: Callable(step, total, message) for UI progress.
        """
        import lightgbm as lgb

        from src.features.builder import build_features

        t_start = time.time()

        if progress_callback:
            progress_callback(0, 5, "Loading data...")

        df_hybrid = _prepare_hybrid_data()

        if progress_callback:
            progress_callback(1, 5, "Building features...")

        df_feat = build_features(df_hybrid)
        self._feature_cols = [
            c for c in df_feat.columns if c not in [TARGET_COL, "is_imputed"]
        ]

        X = df_feat[self._feature_cols].values
        y = df_feat[TARGET_COL].values
        is_imputed = df_feat["is_imputed"].values if "is_imputed" in df_feat.columns else np.zeros(len(y))

        # Create target at horizon
        y_target = pd.Series(y).shift(-self.horizon).values
        valid = ~np.isnan(y_target) & ~np.isnan(X).any(axis=1)
        X, y_target, is_imputed_valid = X[valid], y_target[valid], is_imputed[valid]

        # Split 80/10/10
        n = len(X)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        X_train, y_train = X[:train_end], y_target[:train_end]
        X_val, y_val = X[train_end:val_end], y_target[train_end:val_end]
        X_test, y_test = X[val_end:], y_target[val_end:]
        test_imputed = is_imputed_valid[val_end:]

        if progress_callback:
            progress_callback(2, 5, "Training LightGBM...")

        self.model = lgb.LGBMRegressor(
            **self.params,
            verbose=-1, n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)],
        )

        if progress_callback:
            progress_callback(3, 5, "Evaluating...")

        # Test on real data only
        real_mask = test_imputed == 0
        y_test_real = y_test[real_mask]
        y_pred_real = self.model.predict(X_test[real_mask])

        # Persistence baseline
        y_orig = y[valid]
        persist_mae = float(np.mean(np.abs(
            y_test_real - y_orig[val_end:][real_mask]
        )))

        mae = float(mean_absolute_error(y_test_real, y_pred_real))
        rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
        r2 = float(r2_score(y_test_real, y_pred_real))
        mase = mae / persist_mae if persist_mae > 0 else float("inf")

        self._train_time = time.time() - t_start
        self.metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mase": round(mase, 4),
            "persist_mae": round(persist_mae, 4),
            "n_test": int(real_mask.sum()),
            "training_time_s": round(self._train_time, 1),
        }

        if progress_callback:
            progress_callback(4, 5, "Done!")

        self._log_run()

        if progress_callback:
            progress_callback(5, 5, "Complete!")

        return self.metrics

    def save_model(self) -> str:
        """Save model to user_trained directory."""
        if self.model is None:
            raise RuntimeError("Train first before saving.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = USER_MODELS_DIR / f"{ts}_lgbm_{self.horizon}h"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Model
        model_path = save_dir / f"lgbm_{self.horizon}h.txt"
        self.model.booster_.save_model(str(model_path))

        # Features
        with open(save_dir / "features.json", "w") as f:
            json.dump({"features": self._feature_cols, "horizon": self.horizon}, f, indent=2)

        # Params + metrics
        with open(save_dir / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

        return str(save_dir)

    def _log_run(self):
        """Log run to dashboard_runs."""
        DASHBOARD_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log = {
            "model": "LightGBM",
            "horizon": self.horizon,
            "params": self.params,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat(),
            "source": "dashboard",
        }
        with open(DASHBOARD_RUNS_DIR / f"run_{ts}.json", "w") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)


class GRUTrainer:
    """Train GRU with user-specified params."""

    def __init__(self, horizon: int, params: dict):
        self.horizon = horizon
        self.params = params
        self.model = None
        self.metrics = None
        self._train_time = None
        self._model_state = None

    def train(self, progress_callback=None) -> dict:
        """Train and evaluate GRU."""
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, Dataset

        t_start = time.time()

        if progress_callback:
            progress_callback(0, 10, "Loading data...")

        df_hybrid = _prepare_hybrid_data()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        available = [c for c in FEATURE_COLS_DL if c in df_hybrid.columns]
        features = df_hybrid[available].values
        target = df_hybrid[TARGET_COL].values
        is_imputed = df_hybrid["is_imputed"].values

        n = len(features)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        if progress_callback:
            progress_callback(1, 10, "Scaling features...")

        # Scale
        feat_scaler = StandardScaler()
        features_scaled = np.zeros_like(features)
        features_scaled[:train_end] = feat_scaler.fit_transform(features[:train_end])
        features_scaled[train_end:] = feat_scaler.transform(features[train_end:])

        tgt_scaler = StandardScaler()
        tgt_scaler.fit(target[:train_end].reshape(-1, 1))
        target_scaled = tgt_scaler.transform(target.reshape(-1, 1)).flatten()

        lb = self.params.get("lookback", LOOKBACK)
        h = self.horizon

        # Dataset
        class SeqDS(Dataset):
            def __init__(self, feats, tgts, lb_, h_):
                self.feats, self.tgts = feats, tgts
                self.indices = [i for i in range(len(feats) - lb_ - h_)
                                if i + lb_ + h_ - 1 < len(tgts)]
                self.lb_, self.h_ = lb_, h_

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                i = self.indices[idx]
                x = torch.FloatTensor(self.feats[i:i + self.lb_])
                y = torch.FloatTensor([self.tgts[i + self.lb_ + self.h_ - 1]])
                return x, y

        train_ds = SeqDS(features_scaled[:train_end], target_scaled[:train_end], lb, h)
        val_ds = SeqDS(features_scaled[:val_end], target_scaled[:val_end], lb, h)
        val_ds.indices = [i for i in val_ds.indices if i + lb + h - 1 >= train_end]

        bs = self.params.get("batch_size", 256)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

        if progress_callback:
            progress_callback(2, 10, f"Building GRU (hidden={self.params.get('hidden_dim', 64)})...")

        hidden = self.params.get("hidden_dim", 64)
        layers = self.params.get("num_layers", 2)
        drop = self.params.get("dropout", 0.2)
        drop_gru = drop if layers > 1 else 0

        class GRUModel(nn.Module):
            def __init__(self, in_dim, hid, lay, dr):
                super().__init__()
                self.gru = nn.GRU(in_dim, hid, lay, dropout=dr if lay > 1 else 0,
                                  batch_first=True)
                self.fc = nn.Sequential(
                    nn.Linear(hid, hid // 2), nn.ReLU(),
                    nn.Dropout(dr), nn.Linear(hid // 2, 1),
                )

            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])

        model = GRUModel(len(available), hidden, layers, drop).to(device)
        lr = self.params.get("learning_rate", 1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        epochs = self.params.get("epochs", 100)
        patience = self.params.get("patience", 10)
        best_val, best_state, wait = float("inf"), None, 0

        for ep in range(epochs):
            model.train()
            tl = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tl += loss.item()
            tl /= max(len(train_loader), 1)

            model.eval()
            vl = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    vl += criterion(model(xb.to(device)), yb.to(device)).item()
            vl /= max(len(val_loader), 1)
            scheduler.step(vl)

            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                break

            if progress_callback and (ep + 1) % 5 == 0:
                pct = min(2 + int(7 * (ep + 1) / epochs), 9)
                progress_callback(pct, 10, f"Epoch {ep + 1}/{epochs} — val_loss: {vl:.4f}")

        if best_state:
            model.load_state_dict(best_state)

        # Evaluate on test set (real-only)
        model.eval()
        model = model.to(device)
        test_ds = SeqDS(features_scaled, target_scaled, lb, h)
        test_ds.indices = [i for i in test_ds.indices if i + lb + h - 1 >= val_end]
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

        preds_scaled, actuals_scaled = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb.to(device)).cpu().numpy()
                preds_scaled.append(pred)
                actuals_scaled.append(yb.numpy())

        preds_s = np.concatenate(preds_scaled).flatten()
        actuals_s = np.concatenate(actuals_scaled).flatten()

        # Inverse scale
        preds = preds_s * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]
        actuals = actuals_s * tgt_scaler.scale_[0] + tgt_scaler.mean_[0]

        # Filter real-only
        # Build imputed mask for test indices
        test_indices_end = [i + lb + h - 1 for i in test_ds.indices]
        real_mask = np.array([is_imputed[idx] == 0 for idx in test_indices_end])
        preds_real = preds[real_mask]
        actuals_real = actuals[real_mask]

        # Persistence
        persist_preds = np.array([target[i] for i in test_ds.indices])[real_mask]
        persist_mae = float(np.mean(np.abs(actuals_real - persist_preds)))

        mae = float(mean_absolute_error(actuals_real, preds_real))
        rmse = float(np.sqrt(mean_squared_error(actuals_real, preds_real)))
        r2 = float(r2_score(actuals_real, preds_real))
        mase = mae / persist_mae if persist_mae > 0 else float("inf")

        self._train_time = time.time() - t_start
        self.model = model
        self._model_state = best_state
        self._feat_scaler = feat_scaler
        self._tgt_scaler = tgt_scaler
        self._available_features = available

        self.metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mase": round(mase, 4),
            "persist_mae": round(persist_mae, 4),
            "n_test": int(real_mask.sum()),
            "epochs_trained": ep + 1,
            "best_val_loss": round(best_val, 6),
            "training_time_s": round(self._train_time, 1),
        }

        self._log_run()

        if progress_callback:
            progress_callback(10, 10, "Complete!")

        return self.metrics

    def save_model(self) -> str:
        """Export GRU to TorchScript and save."""
        import torch

        if self.model is None:
            raise RuntimeError("Train first before saving.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = USER_MODELS_DIR / f"{ts}_gru_{self.horizon}h"
        save_dir.mkdir(parents=True, exist_ok=True)

        # TorchScript
        self.model.cpu().eval()
        lb = self.params.get("lookback", LOOKBACK)
        example = torch.randn(1, lb, len(self._available_features))
        scripted = torch.jit.trace(self.model, example)
        model_path = save_dir / f"gru_{self.horizon}h.pt"
        scripted.save(str(model_path))

        # Scalers
        scaler_info = {
            "feature_scaler_mean": self._feat_scaler.mean_.tolist(),
            "feature_scaler_scale": self._feat_scaler.scale_.tolist(),
            "target_scaler_mean": float(self._tgt_scaler.mean_[0]),
            "target_scaler_scale": float(self._tgt_scaler.scale_[0]),
            "features": self._available_features,
            "lookback": lb,
            "horizon": self.horizon,
        }
        with open(save_dir / f"scalers_{self.horizon}h.json", "w") as f:
            json.dump(scaler_info, f, indent=2)

        # Params + metrics
        with open(save_dir / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

        return str(save_dir)

    def _log_run(self):
        """Log run to dashboard_runs."""
        DASHBOARD_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log = {
            "model": "GRU",
            "horizon": self.horizon,
            "params": self.params,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat(),
            "source": "dashboard",
        }
        with open(DASHBOARD_RUNS_DIR / f"run_{ts}.json", "w") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
