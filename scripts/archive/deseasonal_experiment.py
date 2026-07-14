"""P0-8 — Deseasonalizing Transform Experiment.

Hypothesis: PM2.5 has strong daily cycle (STL seasonal strength = 0.343).
    If we remove the seasonal component BEFORE modeling, the model only
    needs to learn trend + residual → should be easier.

Strategy:
    A) seasonal_diff: y_deseas[t] = y[t] - y[t-24]  (simple seasonal diff)
    B) stl_residual: y_deseas[t] = STL_residual[t]   (use STL to remove season)
    C) baseline: y[t] (raw, same as v2 experiments)

    Train GRU on each → compare at h=6 (our best horizon).
    After prediction, inverse transform: ŷ[t] = ŷ_deseas[t] + seasonal[t]

Reference: Manu Joseph Ch.7, DL Cookbook Ch.1
Data: Same as v2 pipeline (hybrid imputation, anti-leakage, test=real only)

Usage:
    uv run python scripts/deseasonal_experiment.py 2>&1 | tee research/logs/deseasonal_experiment.log
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
from statsmodels.tsa.seasonal import STL
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
from src.evaluation.metrics import evaluate_forecast, forecast_bias, medae
from src.evaluation.residual_diagnostics import run_residual_diagnostics
from src.features.builder import build_features

warnings.filterwarnings("ignore")

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "deseasonal"
DIAG_DIR = PROJECT_ROOT / "research" / "diagnostics"

HORIZON = 6  # Best performing horizon from v6
LOOKBACK = 72
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════
# Dataset & GRU Model (reuse from dl_retrain_v2)
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
# Data Preparation
# ══════════════════════════════════════════════════════════════════════


def prepare_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load → clean → impute → build_features (v2)."""
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

    is_imputed = df_hybrid["is_imputed"].copy()
    df_for_feat = df_hybrid.drop(columns=["is_imputed"])

    print("  Building v2 features...", flush=True)
    df_feat = build_features(df_for_feat, include_fourier=True, fourier_order=3)
    df_feat["is_imputed"] = is_imputed.reindex(df_feat.index).fillna(False)

    print(f"  Shape: {df_feat.shape}", flush=True)
    return df_feat, is_imputed.reindex(df_feat.index).fillna(False).values


def create_deseasonalized_targets(
    pm25: pd.Series,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Create deseasonalized target variants.

    Returns:
        Dict of name -> (target, seasonal_component):
            - "raw": (y, zeros) — baseline
            - "seasonal_diff": (y[t]-y[t-24], y[t-24])
            - "stl_residual": (STL_residual, trend + seasonal)
    """
    print("\n  Creating deseasonalized targets...", flush=True)

    y = pm25.values.copy()
    n = len(y)

    # A) Raw baseline (no transform)
    raw_seasonal = np.zeros(n)

    # B) Seasonal differencing: y_d[t] = y[t] - y[t-24]
    #    CRITICAL: We use the VALUE at t-24 (known past data), NO leakage.
    seasonal_lag = np.full(n, np.nan)
    seasonal_lag[24:] = y[:-24]  # y[t-24]
    seasonal_diff_target = y - seasonal_lag  # y[t] - y[t-24]

    # C) STL decomposition → use residual as target
    stl = STL(pm25.dropna(), period=24, robust=True)
    result = stl.fit()
    stl_residual = result.resid.values
    stl_seasonal_trend = result.trend.values + result.seasonal.values

    print(f"    Raw: mean={np.nanmean(y):.2f}, std={np.nanstd(y):.2f}", flush=True)
    print(f"    Seasonal diff: mean={np.nanmean(seasonal_diff_target):.2f}, std={np.nanstd(seasonal_diff_target):.2f}", flush=True)
    print(f"    STL residual: mean={np.nanmean(stl_residual):.2f}, std={np.nanstd(stl_residual):.2f}", flush=True)

    return {
        "raw": (y, raw_seasonal),
        "seasonal_diff": (seasonal_diff_target, seasonal_lag),
        "stl_residual": (stl_residual, stl_seasonal_trend),
    }


# ══════════════════════════════════════════════════════════════════════
# Experiment Runner
# ══════════════════════════════════════════════════════════════════════


def run_experiment(
    df_feat: pd.DataFrame,
    is_imputed: np.ndarray,
    target_name: str,
    target_values: np.ndarray,
    seasonal_component: np.ndarray,
    original_pm25: np.ndarray,
) -> dict:
    """Run GRU experiment with a specific target transform.

    Args:
        df_feat: Feature DataFrame.
        is_imputed: Boolean mask for imputed data.
        target_name: Name of the transform (raw, seasonal_diff, stl_residual).
        target_values: Transformed target values.
        seasonal_component: Seasonal values to add back for inverse transform.
        original_pm25: Original PM2.5 values for final MAE calculation.

    Returns:
        Result dict with MAE, MASE, etc.
    """
    # ── Feature selection (exclude target and is_imputed) ──
    exclude = {"is_imputed", TARGET_COL}
    exclude.update(c for c in df_feat.columns if c.startswith("target_"))
    feature_cols = [
        c for c in df_feat.columns
        if c not in exclude and df_feat[c].dtype in ("float64", "float32", "int64")
    ]
    features_df = df_feat[feature_cols].fillna(0)

    n = len(features_df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Valid mask: need both target and features to be non-NaN
    valid = ~np.isnan(target_values)
    print(f"    Valid points: {np.sum(valid)}/{n} ({np.sum(valid)/n*100:.1f}%)", flush=True)

    # ── Scale features (fit on train only) ──
    scaler = StandardScaler()
    features_scaled = np.zeros_like(features_df.values, dtype=np.float32)
    features_scaled[:train_end] = scaler.fit_transform(features_df.values[:train_end])
    features_scaled[train_end:] = scaler.transform(features_df.values[train_end:])
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Scale target ──
    target_safe = np.nan_to_num(target_values, nan=0.0)
    target_scaler = StandardScaler()
    target_scaler.fit(target_safe[:train_end].reshape(-1, 1))
    target_all_scaled = target_scaler.transform(target_safe.reshape(-1, 1)).flatten()

    print(f"    Features: {len(feature_cols)} cols", flush=True)
    print(f"    Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}", flush=True)

    # ── Persistence baseline (on ORIGINAL pm25) ──
    y_true_list, y_persist_list = [], []
    for i in range(val_end, n - HORIZON):
        if is_imputed[i + HORIZON]:
            continue
        actual = original_pm25[i + HORIZON]
        persist = original_pm25[i]
        if not np.isnan(actual) and not np.isnan(persist):
            y_true_list.append(actual)
            y_persist_list.append(persist)

    y_true_baseline = np.array(y_true_list)
    y_persist = np.array(y_persist_list)
    persist_mae = float(np.mean(np.abs(y_true_baseline - y_persist)))
    print(f"    Persistence {HORIZON}h: MAE={persist_mae:.3f}", flush=True)

    # ── Datasets ──
    real_mask_test = np.zeros(n, dtype=bool)
    for i in range(val_end, n):
        if not is_imputed[i] and valid[i]:
            real_mask_test[i] = True

    # Combined validity mask: target must be valid AND not imputed (for test)
    train_valid = np.ones(n, dtype=bool)
    train_valid[~valid] = False

    train_dataset = TimeSeriesDataset(features_scaled, target_all_scaled, LOOKBACK, HORIZON)
    train_dataset.valid_indices = [
        i for i in train_dataset.valid_indices
        if i + LOOKBACK + HORIZON - 1 < train_end and valid[i + LOOKBACK + HORIZON - 1]
    ]

    val_dataset = TimeSeriesDataset(features_scaled, target_all_scaled, LOOKBACK, HORIZON)
    val_dataset.valid_indices = [
        i for i in val_dataset.valid_indices
        if train_end <= i + LOOKBACK + HORIZON - 1 < val_end and valid[i + LOOKBACK + HORIZON - 1]
    ]

    test_dataset = TimeSeriesDataset(
        features_scaled, target_all_scaled, LOOKBACK, HORIZON,
        real_mask=real_mask_test,
    )
    test_dataset.valid_indices = [
        i for i in test_dataset.valid_indices
        if i + LOOKBACK + HORIZON - 1 >= val_end
    ]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"    Datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}", flush=True)

    if len(train_dataset) < BATCH_SIZE:
        print("    ⚠️ Too few training samples, skipping", flush=True)
        return {"error": "too few samples"}

    input_dim = features_scaled.shape[1]

    # ── Train GRU ──
    model_name = f"GRU_deseas_{target_name}"
    print(f"\n    Training {model_name} (h={HORIZON})...", flush=True)
    t0 = time.time()

    model = GRUModel(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,} | Input dim: {input_dim}", flush=True)

    model = train_model(model, train_loader, val_loader, model_name)
    train_time = time.time() - t0

    # ── Evaluate ──
    model.eval()
    all_preds, all_targets, all_indices = [], [], []
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(DEVICE)
            pred = model(x_batch).cpu().numpy().flatten()
            all_preds.extend(pred)
            all_targets.extend(y_batch.numpy().flatten())

    if len(all_preds) == 0:
        print("    ⚠️ No predictions", flush=True)
        return {"error": "no predictions"}

    # ── Inverse transform ──
    # 1. Undo target scaling
    preds_unscaled = target_scaler.inverse_transform(
        np.array(all_preds).reshape(-1, 1)
    ).flatten()
    targets_unscaled = target_scaler.inverse_transform(
        np.array(all_targets).reshape(-1, 1)
    ).flatten()

    # 2. Undo deseasonalization (add seasonal component back)
    #    For test set, we need the seasonal values at the prediction indices
    test_indices = [
        test_dataset.valid_indices[batch_idx * BATCH_SIZE + j]
        for batch_idx in range(len(all_preds) // BATCH_SIZE + 1)
        for j in range(min(BATCH_SIZE, len(all_preds) - batch_idx * BATCH_SIZE))
    ][:len(all_preds)]

    # Get the corresponding seasonal values
    pred_target_indices = [i + LOOKBACK + HORIZON - 1 for i in test_indices]

    if target_name == "raw":
        # No inverse transform needed
        preds_original = np.clip(preds_unscaled, 0, None)
        targets_original = np.clip(targets_unscaled, 0, None)
    elif target_name == "seasonal_diff":
        # ŷ_original = ŷ_diff + y[t-24]
        # For the target: y_original = y_diff + y[t-24] (should match original)
        seasonal_at_pred = np.array([
            seasonal_component[idx] if idx < len(seasonal_component) and not np.isnan(seasonal_component[idx])
            else 0.0
            for idx in pred_target_indices
        ])
        preds_original = np.clip(preds_unscaled + seasonal_at_pred, 0, None)
        targets_original = np.clip(targets_unscaled + seasonal_at_pred, 0, None)
    elif target_name == "stl_residual":
        # ŷ_original = ŷ_residual + (trend + seasonal)
        seasonal_at_pred = np.array([
            seasonal_component[idx] if idx < len(seasonal_component) else 0.0
            for idx in pred_target_indices
        ])
        preds_original = np.clip(preds_unscaled + seasonal_at_pred, 0, None)
        targets_original = np.clip(targets_unscaled + seasonal_at_pred, 0, None)

    # ── Metrics (on ORIGINAL scale) ──
    mae_val = float(np.mean(np.abs(targets_original - preds_original)))
    rmse_val = float(np.sqrt(np.mean((targets_original - preds_original) ** 2)))
    mase = round(mae_val / persist_mae, 4) if persist_mae > 0 else float("inf")
    fb = forecast_bias(targets_original, preds_original)
    med_ae = medae(targets_original, preds_original)

    status = "✅" if mase < 1.0 else "❌"
    print(
        f"\n    {status} {model_name} h={HORIZON}: MAE={mae_val:.3f}, MASE={mase:.3f}, "
        f"FB={fb:.4f}, MedAE={med_ae:.3f} ({train_time:.0f}s)",
        flush=True,
    )

    # ── Residual Diagnostics ──
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    diag = run_residual_diagnostics(
        y_true=targets_original,
        y_pred=preds_original,
        model_name=model_name,
        horizon=HORIZON,
        output_dir=str(DIAG_DIR),
    )

    return {
        "model": model_name,
        "target_transform": target_name,
        "horizon": HORIZON,
        "mae": round(mae_val, 4),
        "rmse": round(rmse_val, 4),
        "mase": mase,
        "forecast_bias": round(fb, 4),
        "medae": round(med_ae, 4),
        "params": n_params,
        "input_dim": input_dim,
        "train_time_s": round(train_time, 1),
        "n_test": len(all_preds),
        "persistence_mae": round(persist_mae, 4),
        "diagnostics": {
            "verdict": diag["verdict"],
            "lb_pass_rate": diag["lb_pass_rate"],
            "residual_mean": diag["residual_stats"]["mean"],
            "residual_std": diag["residual_stats"]["std"],
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("P0-8: DESEASONALIZING TRANSFORM EXPERIMENT", flush=True)
    print(f"Horizon: {HORIZON}h | Lookback: {LOOKBACK}h | Device: {DEVICE}", flush=True)
    print(f"Architecture: hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, dropout={DROPOUT}", flush=True)
    print("Transforms: raw | seasonal_diff (y-y[t-24]) | stl_residual", flush=True)
    print("RULE: Test set = REAL data only", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Prepare data ──
    print("\n[1/4] Preparing dataset...", flush=True)
    df_feat, is_imputed = prepare_data()

    # ── Step 2: Create deseasonalized targets ──
    print("\n[2/4] Creating deseasonalized targets...", flush=True)
    pm25_series = df_feat[TARGET_COL]
    targets = create_deseasonalized_targets(pm25_series)

    # ── Step 3: Run experiments ──
    print(f"\n[3/4] Running experiments...", flush=True)
    all_results = {}
    original_pm25 = df_feat[TARGET_COL].values

    for name, (target_vals, seasonal_comp) in targets.items():
        print(f"\n{'═' * 60}", flush=True)
        print(f"  Transform: {name}", flush=True)
        print(f"{'═' * 60}", flush=True)

        result = run_experiment(
            df_feat=df_feat,
            is_imputed=is_imputed,
            target_name=name,
            target_values=target_vals,
            seasonal_component=seasonal_comp,
            original_pm25=original_pm25,
        )
        all_results[name] = result

    # ── Step 4: Comparison Summary ──
    print(f"\n{'═' * 70}", flush=True)
    print("[4/4] EXPERIMENT COMPARISON", flush=True)
    print(f"{'═' * 70}", flush=True)

    print(f"\n{'Transform':<20} {'MAE':>8} {'MASE':>8} {'FB':>8} {'MedAE':>8} {'Verdict':<12}", flush=True)
    print("─" * 70, flush=True)

    # Reference: GRU v2+log at 6h → MASE=0.692
    ref_mase = 0.692

    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<20} ERROR: {r['error']}", flush=True)
            continue
        status = "✅" if r["mase"] < 1.0 else "❌"
        diag_verdict = r.get("diagnostics", {}).get("verdict", "N/A")[:20]
        print(
            f"{name:<20} {r['mae']:>8.3f} {r['mase']:>8.3f} {r['forecast_bias']:>8.4f} "
            f"{r['medae']:>8.3f} {diag_verdict:<12} {status}",
            flush=True,
        )

    print(f"\n{'─' * 70}", flush=True)
    print(f"Reference: GRU_v2+log (6h) MASE = {ref_mase}", flush=True)

    # Find best
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    if valid_results:
        best_name = min(valid_results, key=lambda k: valid_results[k]["mase"])
        best = valid_results[best_name]
        delta_vs_ref = ((best["mase"] - ref_mase) / ref_mase) * 100
        print(f"Best transform: {best_name} (MASE={best['mase']:.3f})", flush=True)
        print(f"vs v2+log reference: {delta_vs_ref:+.1f}%", flush=True)

        if best["mase"] < ref_mase:
            print("🎉 NEW BEST! Deseasonalizing IMPROVES over v2+log!", flush=True)
        else:
            print("📊 Deseasonalizing does NOT improve over v2+log at 6h.", flush=True)
            print("   This is expected if Fourier features already capture seasonality.", flush=True)

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"deseasonal_experiment_{ts}.json"

    meta = {
        "description": "P0-8 Deseasonalizing Transform Experiment",
        "reference": "Manu Joseph Ch.7, DL Cookbook Ch.1",
        "hypothesis": "Removing seasonal component before modeling simplifies learning",
        "horizon": HORIZON,
        "lookback": LOOKBACK,
        "hidden_dim": HIDDEN_DIM,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat(),
    }

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {"meta": meta, "results": all_results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}", flush=True)

    total = time.time() - t_start
    print(f"\nCOMPLETE — Total: {total:.0f}s ({total / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
