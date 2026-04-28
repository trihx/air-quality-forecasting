"""PM2.5 Prediction Service — Load exported models and run inference.

Supports:
  - GRU (TorchScript .pt) with feature scaling
  - LightGBM (native .txt) with feature engineering

Usage:
    predictor = GRUPredictor(horizon=6)
    result = predictor.predict(recent_data_df)
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: TC002

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"
FEATURE_COLS = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]
LOOKBACK = 72


class GRUPredictor:
    """Load exported GRU TorchScript model and predict PM2.5."""

    def __init__(self, horizon: int, model_dir: Path | None = None):
        self.horizon = horizon
        model_dir = model_dir or EXPORT_DIR

        self.model_path = model_dir / f"gru_{horizon}h.pt"
        self.scaler_path = model_dir / f"scalers_{horizon}h.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"GRU model not found: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scalers not found: {self.scaler_path}")

        # Lazy import torch to avoid MPS/CUDA conflicts
        import torch

        self.torch = torch

        # Default to CPU, can be overridden in predict()
        self.model = torch.jit.load(str(self.model_path), map_location="cpu")
        self.model.eval()

        # Load scalers
        with open(self.scaler_path) as f:
            sc = json.load(f)
        self.feat_mean = np.array(sc["feature_scaler_mean"])
        self.feat_scale = np.array(sc["feature_scaler_scale"])
        self.tgt_mean = sc["target_scaler_mean"]
        self.tgt_scale = sc["target_scaler_scale"]
        self.features = sc["features"]

    def predict(self, recent_data: pd.DataFrame, device: str = "cpu") -> dict:
        """Predict PM2.5 from recent data.

        Args:
            recent_data: DataFrame with at least LOOKBACK rows and FEATURE_COLS.
            device: Torch device string — ignored for TorchScript models.
                    TorchScript-traced GRU hard-codes hidden state device
                    at trace time (CPU), so MPS causes device mismatch.

        Returns:
            Dict with predicted_pm25, model, horizon, timestamp.
        """
        torch = self.torch

        # Validate
        available = [c for c in self.features if c in recent_data.columns]
        if len(available) < len(self.features):
            missing = set(self.features) - set(available)
            raise ValueError(f"Missing columns: {missing}")

        if len(recent_data) < LOOKBACK:
            raise ValueError(f"Need at least {LOOKBACK} rows, got {len(recent_data)}")

        # Take last LOOKBACK rows
        window = recent_data[self.features].tail(LOOKBACK).values.astype(np.float64)

        # Scale features
        scaled = (window - self.feat_mean) / self.feat_scale

        # TorchScript-traced GRU: hidden state device baked at trace time (CPU).
        # Moving model to MPS only moves weights → RuntimeError device mismatch.
        # Force CPU for safe, portable inference.
        model = self.model.to("cpu")
        x = torch.FloatTensor(scaled).unsqueeze(0)
        device = "cpu"

        # Predict
        with torch.no_grad():
            pred_scaled = model(x).cpu().item()

        # Free GPU memory if used
        if device == "mps":
            with contextlib.suppress(AttributeError):
                torch.mps.empty_cache()

        # Inverse scale
        pred_pm25 = pred_scaled * self.tgt_scale + self.tgt_mean

        return {
            "predicted_pm25": round(float(pred_pm25), 2),
            "model": "GRU",
            "horizon": self.horizon,
            "timestamp": datetime.now().isoformat(),
            "input_rows": len(recent_data),
            "last_pm25": round(float(recent_data["pm25"].iloc[-1]), 2),
            "device": device,
        }


class GRUQuantilePredictor:
    """Load exported GRU Quantile TorchScript model with CQR calibration.

    Provides:
      - Point prediction (median quantile q=0.50)
      - Prediction intervals [q_low - adj, q_high + adj] with CQR guarantee

    CPU-only inference for Docker portability.
    Reference: Romano et al. (2019) "Conformalized Quantile Regression"
    """

    def __init__(self, horizon: int, model_dir: Path | None = None):
        self.horizon = horizon
        model_dir = model_dir or EXPORT_DIR

        self.model_path = model_dir / f"gru_quantile_{horizon}h.pt"
        self.config_path = model_dir / f"gru_quantile_{horizon}h_config.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"GRU Quantile model not found: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"CQR config not found: {self.config_path}")

        # Lazy import torch — avoid OMP crash with LightGBM
        import torch

        self.torch = torch

        # Load model on CPU (Docker-safe, TorchScript portable)
        self.model = torch.jit.load(str(self.model_path), map_location="cpu")
        self.model.eval()

        # Load scalers + CQR config
        with open(self.config_path) as f:
            cfg = json.load(f)
        self.feat_mean = np.array(cfg["feature_scaler_mean"])
        self.feat_scale = np.array(cfg["feature_scaler_scale"])
        self.tgt_mean = cfg["target_scaler_mean"]
        self.tgt_scale = cfg["target_scaler_scale"]
        self.features = cfg["features"]
        self.conformal_adj = cfg.get("conformal_adjustment", 0.0)
        self.quantiles = cfg.get("quantiles", [0.05, 0.50, 0.95])

    def predict(self, recent_data: pd.DataFrame, device: str = "cpu") -> dict:
        """Predict PM2.5 with CQR prediction intervals.

        Args:
            recent_data: DataFrame with at least LOOKBACK rows.
            device: Ignored — always uses CPU for Docker compatibility.

        Returns:
            Dict with predicted_pm25, lower/upper bounds, coverage info.
        """
        torch = self.torch

        # Validate
        available = [c for c in self.features if c in recent_data.columns]
        if len(available) < len(self.features):
            missing = set(self.features) - set(available)
            raise ValueError(f"Missing columns: {missing}")

        if len(recent_data) < LOOKBACK:
            raise ValueError(f"Need at least {LOOKBACK} rows, got {len(recent_data)}")

        # Take last LOOKBACK rows and scale
        window = recent_data[self.features].tail(LOOKBACK).values.astype(np.float64)
        scaled = (window - self.feat_mean) / self.feat_scale
        x = torch.FloatTensor(scaled).unsqueeze(0)

        # Predict 3 quantiles (CPU inference)
        with torch.no_grad():
            out = self.model(x).cpu().numpy().flatten()

        # Inverse scale all 3 quantiles
        q_vals = out * self.tgt_scale + self.tgt_mean
        q_low, q_median, q_high = float(q_vals[0]), float(q_vals[1]), float(q_vals[2])

        # Apply CQR conformal adjustment for guaranteed coverage
        cqr_lower = q_low - self.conformal_adj
        cqr_upper = q_high + self.conformal_adj

        return {
            "predicted_pm25": round(q_median, 2),
            "model": "GRU",
            "horizon": self.horizon,
            "timestamp": datetime.now().isoformat(),
            "input_rows": len(recent_data),
            "last_pm25": round(float(recent_data["pm25"].iloc[-1]), 2),
            "device": "cpu",
            # CQR-specific fields
            "pi_method": "cqr",
            "pi_lower": round(max(cqr_lower, 0.0), 2),  # PM2.5 >= 0
            "pi_upper": round(cqr_upper, 2),
            "pi_width": round(cqr_upper - cqr_lower, 2),
            "conformal_adjustment": round(self.conformal_adj, 3),
            "quantile_raw_lower": round(q_low, 2),
            "quantile_raw_upper": round(q_high, 2),
        }


class LightGBMPredictor:
    """Load exported LightGBM model and predict PM2.5."""

    def __init__(self, horizon: int, model_dir: Path | None = None):
        import lightgbm as lgb

        self.horizon = horizon
        model_dir = model_dir or EXPORT_DIR

        self.model_path = model_dir / f"lgbm_{horizon}h.txt"
        self.features_path = model_dir / f"lgbm_{horizon}h_features.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"LightGBM model not found: {self.model_path}")

        # Load model
        self.model = lgb.Booster(model_file=str(self.model_path))

        # Load feature names
        if self.features_path.exists():
            with open(self.features_path) as f:
                info = json.load(f)
            self.feature_names = info["features"]
        else:
            self.feature_names = None

    def predict(self, feature_row: pd.DataFrame) -> dict:
        """Predict PM2.5 from a single feature-engineered row.

        Args:
            feature_row: DataFrame with 1+ rows, all engineered feature columns.
                         Uses the LAST row for prediction.

        Returns:
            Dict with predicted_pm25, model, horizon, timestamp.
        """
        if self.feature_names:
            available = [c for c in self.feature_names if c in feature_row.columns]
            if len(available) < len(self.feature_names) * 0.8:
                raise ValueError(f"Too few matching features: {len(available)}/{len(self.feature_names)}")
            row = feature_row[available].tail(1).values
        else:
            row = feature_row.tail(1).values

        pred = self.model.predict(row)[0]

        return {
            "predicted_pm25": round(float(pred), 2),
            "model": "LightGBM",
            "horizon": self.horizon,
            "timestamp": datetime.now().isoformat(),
            "n_features_used": row.shape[1],
        }


def get_latest_data(n_rows: int = LOOKBACK) -> pd.DataFrame:
    """Load the latest N rows from the processed hybrid dataset.

    Returns a DataFrame ready for GRU prediction (raw features).
    """
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
    df_hybrid = impute_missing_data(
        df,
        strategy="hybrid",
        max_gap_interp=6,
        max_gap_ml=24,
        knn_neighbors=5,
        verbose=False,
    )
    return df_hybrid.tail(n_rows)


def get_suggestion_values() -> dict:
    """Get pre-filled suggestion values (last known data point)."""
    try:
        df = get_latest_data(1)
        row = df.iloc[-1]
        return {
            "pm25": round(float(row.get("pm25", 10.0)), 1),
            "nhiet_do": round(float(row.get("nhiet_do", 28.0)), 1),
            "do_am": round(float(row.get("do_am", 75.0)), 1),
            "diem_suong": round(float(row.get("diem_suong", 24.0)), 1),
            "co2": round(float(row.get("co2", 400.0)), 1),
        }
    except Exception:
        return {
            "pm25": 10.0,
            "nhiet_do": 28.0,
            "do_am": 75.0,
            "diem_suong": 24.0,
            "co2": 400.0,
        }
