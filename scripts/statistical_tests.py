"""Statistical Significance & Residual Diagnostics.

Generates:
1. Diebold-Mariano tests: GRU vs Persistence, LightGBM vs Persistence
2. Residual diagnostics: Ljung-Box, Q-Q plots, ACF of residuals
3. Results saved to research/diagnostics/

Usage:
    uv run python scripts/statistical_tests.py
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "diagnostics" / "statistical_tests"
FIGURES_DIR = OUTPUT_DIR / "figures"
HORIZONS = [1, 6, 24]


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical Tests")
    parser.add_argument("--resolution", type=str, default="1h", help="Data resolution (e.g., 1h, 30m, 15m)")
    args = parser.parse_args()
    
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"STATISTICAL SIGNIFICANCE & RESIDUAL DIAGNOSTICS ({args.resolution})", flush=True)
    print("=" * 70, flush=True)

    # ── 1. Prepare data ──
    print("\n[1/4] Preparing data...", flush=True)
    df_hybrid = _prepare_hybrid_data(freq=args.resolution)
    is_imp_col = df_hybrid["is_imputed"].copy()
    df_feat = build_features(df_hybrid.drop(columns=["is_imputed"]))
    df_feat["is_imputed"] = is_imp_col.reindex(df_feat.index).fillna(False)
    print(f"  Features: {len(df_feat)} rows × {len(df_feat.columns)} cols", flush=True)

    # ── 2. Generate predictions for all models ──
    print("\n[2/4] Generating predictions...", flush=True)
    all_results = {}

    for h in HORIZONS:
        print(f"\n{'─' * 60}", flush=True)
        print(f"  HORIZON = {h}h", flush=True)
        print(f"{'─' * 60}", flush=True)
        result = _generate_predictions(df_feat, df_hybrid, horizon=h)
        all_results[f"{h}h"] = result

    # ── 3. Diebold-Mariano tests ──
    print("\n[3/4] Diebold-Mariano Tests...", flush=True)
    dm_results = {}
    for h in HORIZONS:
        key = f"{h}h"
        r = all_results[key]
        dm_results[key] = _diebold_mariano_tests(r, horizon=h)

    # ── 4. Residual diagnostics ──
    print("\n[4/4] Residual Diagnostics...", flush=True)
    diag_results = {}
    for h in HORIZONS:
        key = f"{h}h"
        r = all_results[key]
        diag_results[key] = _residual_diagnostics(r, horizon=h)

    # ── Save results ──
    total = time.time() - t_start
    results_json = {
        "diebold_mariano": dm_results,
        "residual_diagnostics": diag_results,
    }
    json_path = OUTPUT_DIR / "statistical_tests_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'═' * 70}", flush=True)
    print(f"COMPLETE — Total: {total:.0f}s ({total / 60:.1f} min)", flush=True)
    print(f"Results: {json_path}", flush=True)
    print(f"Figures: {FIGURES_DIR}", flush=True)
    print(f"{'═' * 70}", flush=True)


def _prepare_hybrid_data(freq: str = "1h") -> pd.DataFrame:
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq=freq)
    return impute_missing_data(
        df,
        strategy="hybrid",
        max_gap_interp=6,
        max_gap_ml=24,
        knn_neighbors=5,
        verbose=True,
    )


def _generate_predictions(
    df_feat: pd.DataFrame,
    df_hybrid: pd.DataFrame,
    horizon: int,
) -> dict:
    """Generate predictions from LightGBM and GRU for a given horizon."""
    from lightgbm import LGBMRegressor

    # ── LightGBM ──
    df = df_feat.copy()
    df["target"] = df[TARGET_COL].shift(-horizon)
    df["_persist"] = df[TARGET_COL]
    df = df.dropna(subset=["target"])

    exclude = ["is_imputed", TARGET_COL, "target", "_persist"]
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "float32", "int64")]

    X = df[feature_cols].fillna(0)
    y = df["target"]
    persist = df["_persist"]
    imp = df["is_imputed"]

    n = len(X)
    tr_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train = X.iloc[:tr_end], y.iloc[:tr_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    persist_test = persist.iloc[val_end:]
    test_real = ~imp.iloc[val_end:].values

    # Filter to real data only
    X_test_real = X_test[test_real]
    y_test_real = y_test[test_real].values
    persist_test_real = persist_test[test_real].values
    test_index = y_test[test_real].index

    # Train LightGBM
    print(f"  Training LightGBM ({horizon}h)...", flush=True)
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    lgbm_pred = model.predict(X_test_real)

    lgbm_mae = float(np.mean(np.abs(y_test_real - lgbm_pred)))
    persist_mae = float(np.mean(np.abs(y_test_real - persist_test_real)))
    print(f"    LightGBM MAE={lgbm_mae:.3f}, Persistence MAE={persist_mae:.3f}", flush=True)

    # ── GRU ──
    print(f"  Training GRU ({horizon}h)...", flush=True)
    gru_pred = _train_gru_predict(df_hybrid, horizon)

    # Align GRU predictions to same test indices
    n_gru = len(gru_pred)
    n_lgbm = len(y_test_real)
    # Use the minimum overlap
    n_common = min(n_gru, n_lgbm)
    print(f"    GRU pred: {n_gru}, LightGBM test: {n_lgbm}, common: {n_common}", flush=True)

    return {
        "y_true": y_test_real[-n_common:],
        "persist_pred": persist_test_real[-n_common:],
        "lgbm_pred": lgbm_pred[-n_common:],
        "gru_pred": gru_pred[-n_common:],
        "test_index": test_index[-n_common:],
    }


def _train_gru_predict(df_hybrid: pd.DataFrame, horizon: int) -> np.ndarray:
    """Train GRU and return test predictions on real data."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    target = df_hybrid[TARGET_COL].values.astype(np.float32)
    is_imputed = df_hybrid["is_imputed"].values
    n = len(target)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    mean_t = float(target[:train_end].mean())
    std_t = float(target[:train_end].std())
    target_norm = (target - mean_t) / std_t

    feat_cols = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]
    feat_data = df_hybrid[feat_cols].values.astype(np.float32)
    feat_mean = feat_data[:train_end].mean(axis=0)
    feat_std = feat_data[:train_end].std(axis=0)
    feat_std[feat_std == 0] = 1
    feat_norm = (feat_data - feat_mean) / feat_std

    lookback = 72
    batch_size = 256

    valid_range = n - horizon - lookback
    indices = np.arange(lookback, lookback + valid_range)
    X_all = np.stack([feat_norm[i - lookback : i] for i in indices])
    y_all = target_norm[indices + horizon]
    imp_all = is_imputed[indices + horizon]

    tr_idx = train_end - lookback
    te_idx = val_end - lookback
    X_train, y_train = X_all[:tr_idx], y_all[:tr_idx]
    X_test, _y_test = X_all[te_idx:], y_all[te_idx:]
    imp_test = imp_all[te_idx:]
    real_mask = ~imp_test

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden, layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = GRUModel(len(feat_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(50):
        epoch_loss, nb = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        scheduler.step(epoch_loss / nb)
        if (epoch + 1) % 25 == 0:
            print(f"      GRU Epoch {epoch + 1}/50: loss={epoch_loss / nb:.4f}", flush=True)

    model.eval()
    X_te = torch.from_numpy(X_test[real_mask]).to(device)
    with torch.no_grad():
        y_pred = model(X_te).cpu().numpy()
    return y_pred * std_t + mean_t


# ═══════════════════════════════════════════════════════════════
# DIEBOLD-MARIANO TEST
# ═══════════════════════════════════════════════════════════════


def _diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano test for predictive accuracy.

    H0: E[d_t] = 0 (equal predictive accuracy)
    H1: E[d_t] ≠ 0 (different predictive accuracy)

    Using absolute loss: L(e) = |e|
    Harvey, Leybourne, Newbold (1997) small-sample correction.

    Args:
        e1: errors from model 1
        e2: errors from model 2
        h: forecast horizon (for autocovariance truncation)

    Returns:
        dict with DM statistic, p-value, and interpretation
    """
    d = np.abs(e1) - np.abs(e2)  # loss differential
    n = len(d)
    d_mean = np.mean(d)

    # Autocovariance with Newey-West truncation
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        var_d = gamma_0 / n  # fallback

    dm_stat = d_mean / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold small-sample correction
    hln_correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corrected = dm_stat * hln_correction

    # Two-sided p-value (t-distribution for small samples)
    p_value = 2 * stats.t.sf(np.abs(dm_corrected), df=n - 1)

    return {
        "dm_statistic": round(float(dm_corrected), 4),
        "p_value": round(float(p_value), 6),
        "n_samples": n,
        "mean_loss_diff": round(float(d_mean), 4),
        "significant_0.05": p_value < 0.05,
        "significant_0.01": p_value < 0.01,
    }


def _diebold_mariano_tests(preds: dict, horizon: int) -> dict:
    """Run DM tests for all model pairs."""
    y_true = preds["y_true"]
    persist_err = y_true - preds["persist_pred"]
    lgbm_err = y_true - preds["lgbm_pred"]
    gru_err = y_true - preds["gru_pred"]

    results = {}

    # Test 1: GRU vs Persistence
    dm = _diebold_mariano(gru_err, persist_err, h=horizon)
    results["GRU_vs_Persistence"] = dm
    sig = "✅ YES" if dm["significant_0.05"] else "❌ NO"
    print(f"\n  DM Test ({horizon}h): GRU vs Persistence", flush=True)
    print(f"    DM stat = {dm['dm_statistic']:.4f}, p = {dm['p_value']:.6f}", flush=True)
    print(f"    Mean |e_GRU| - |e_Persist| = {dm['mean_loss_diff']:.4f}", flush=True)
    print(f"    Significant (α=0.05)? {sig}", flush=True)

    # Test 2: LightGBM vs Persistence
    dm2 = _diebold_mariano(lgbm_err, persist_err, h=horizon)
    results["LightGBM_vs_Persistence"] = dm2
    sig2 = "✅ YES" if dm2["significant_0.05"] else "❌ NO"
    print(f"\n  DM Test ({horizon}h): LightGBM vs Persistence", flush=True)
    print(f"    DM stat = {dm2['dm_statistic']:.4f}, p = {dm2['p_value']:.6f}", flush=True)
    print(f"    Mean |e_LGB| - |e_Persist| = {dm2['mean_loss_diff']:.4f}", flush=True)
    print(f"    Significant (α=0.05)? {sig2}", flush=True)

    # Test 3: GRU vs LightGBM
    dm3 = _diebold_mariano(gru_err, lgbm_err, h=horizon)
    results["GRU_vs_LightGBM"] = dm3
    sig3 = "✅ YES" if dm3["significant_0.05"] else "❌ NO"
    print(f"\n  DM Test ({horizon}h): GRU vs LightGBM", flush=True)
    print(f"    DM stat = {dm3['dm_statistic']:.4f}, p = {dm3['p_value']:.6f}", flush=True)
    print(f"    Mean |e_GRU| - |e_LGB| = {dm3['mean_loss_diff']:.4f}", flush=True)
    print(f"    Significant (α=0.05)? {sig3}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# RESIDUAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════


def _residual_diagnostics(preds: dict, horizon: int) -> dict:
    """Residual diagnostics: Ljung-Box, normality, ACF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.stats.diagnostic import acorr_ljungbox

    results = {}
    models = {
        "GRU": preds["gru_pred"],
        "LightGBM": preds["lgbm_pred"],
        "Persistence": preds["persist_pred"],
    }

    for model_name, y_pred in models.items():
        print(f"\n  Residual Diagnostics ({horizon}h) — {model_name}", flush=True)

        residuals = preds["y_true"] - y_pred
        n = len(residuals)

        # ── Basic stats ──
        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals, ddof=1))
        res_skew = float(stats.skew(residuals))
        res_kurt = float(stats.kurtosis(residuals))

        print(f"    Mean={res_mean:.3f}, Std={res_std:.3f}", flush=True)
        print(f"    Skewness={res_skew:.3f}, Kurtosis={res_kurt:.3f}", flush=True)

        # ── Ljung-Box test (H0: no autocorrelation) ──
        max_lag = min(24, n // 5)
        try:
            lb_result = acorr_ljungbox(residuals, lags=[6, 12, max_lag], return_df=True)
            lb_dict = {}
            for lag, row in lb_result.iterrows():
                lb_dict[f"lag_{lag}"] = {
                    "statistic": round(float(row["lb_stat"]), 4),
                    "p_value": round(float(row["lb_pvalue"]), 6),
                    "autocorrelated": float(row["lb_pvalue"]) < 0.05,
                }
                ac = "✅ autocorr" if float(row["lb_pvalue"]) < 0.05 else "❌ no autocorr"
                print(f"    Ljung-Box(lag={lag}): Q={row['lb_stat']:.2f}, p={row['lb_pvalue']:.4f} → {ac}", flush=True)
        except Exception as e:
            lb_dict = {"error": str(e)}
            print(f"    Ljung-Box error: {e}", flush=True)

        # ── Normality tests ──
        if n > 8:
            shapiro_stat, shapiro_p = stats.shapiro(residuals[: min(5000, n)])
            jb_stat, jb_p = stats.jarque_bera(residuals)
            print(f"    Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.6f}", flush=True)
            print(f"    Jarque-Bera:  JB={jb_stat:.2f}, p={jb_p:.6f}", flush=True)
            normality = {
                "shapiro_wilk": {"statistic": round(float(shapiro_stat), 4), "p_value": round(float(shapiro_p), 6)},
                "jarque_bera": {"statistic": round(float(jb_stat), 4), "p_value": round(float(jb_p), 6)},
                "is_normal_0.05": float(shapiro_p) > 0.05 and float(jb_p) > 0.05,
            }
        else:
            normality = {"error": "insufficient samples"}

        results[model_name] = {
            "n": n,
            "mean": round(res_mean, 4),
            "std": round(res_std, 4),
            "skewness": round(res_skew, 4),
            "kurtosis": round(res_kurt, 4),
            "ljung_box": lb_dict,
            "normality": normality,
        }

        # ── Q-Q Plot ──
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Q-Q
        stats.probplot(residuals, dist="norm", plot=axes[0])
        axes[0].set_title(f"Q-Q Plot — {model_name} ({horizon}h)")
        axes[0].grid(True, alpha=0.3)

        # Histogram
        axes[1].hist(residuals, bins=40, density=True, alpha=0.7, color="#3498db", edgecolor="white")
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1].plot(x_range, stats.norm.pdf(x_range, res_mean, res_std), "r-", lw=2, label="Normal fit")
        axes[1].set_title(f"Residual Distribution — {model_name} ({horizon}h)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ACF of residuals
        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(residuals, lags=min(30, n // 3), ax=axes[2], alpha=0.05)
        axes[2].set_title(f"ACF of Residuals — {model_name} ({horizon}h)")

        plt.tight_layout()
        fig_path = FIGURES_DIR / f"diagnostics_{horizon}h_{model_name}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {fig_path.name}", flush=True)

    return results


if __name__ == "__main__":
    main()
