"""SHAP Explainability — Feature importance analysis for best models.

Generates:
1. SHAP values for LightGBM (TreeExplainer — fast, exact)
2. Permutation importance for GRU (model-agnostic)
3. Summary plots saved to research/figures/

Usage:
    uv run python scripts/shap_explainability.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
FIGURES_DIR = PROJECT_ROOT / "research" / "figures" / "shap"
HORIZONS = [1, 6, 24]


def main() -> None:
    t_start = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("SHAP EXPLAINABILITY ANALYSIS", flush=True)
    print(f"Horizons: {HORIZONS}h | Models: LightGBM (SHAP) + GRU (Permutation)", flush=True)
    print("=" * 70, flush=True)

    # ── 1. Prepare data ──
    print("\n[1/4] Preparing Hybrid dataset...", flush=True)
    df_hybrid = _prepare_hybrid_data()
    df_hybrid["is_imputed"].values.copy()
    n = len(df_hybrid)
    print(f"  Hybrid data: {n} rows", flush=True)

    # ── 2. Build features ──
    print("\n[2/4] Building features...", flush=True)
    is_imp_col = df_hybrid["is_imputed"].copy()
    df_feat = build_features(df_hybrid.drop(columns=["is_imputed"]))
    df_feat["is_imputed"] = is_imp_col.reindex(df_feat.index).fillna(False)
    print(f"  Features: {len(df_feat)} rows × {len(df_feat.columns)} cols", flush=True)

    # ── 3. SHAP for LightGBM at each horizon ──
    print("\n[3/4] Computing SHAP values (LightGBM)...", flush=True)
    all_shap_results = {}

    for h in HORIZONS:
        print(f"\n{'─' * 60}", flush=True)
        print(f"  HORIZON = {h}h", flush=True)
        print(f"{'─' * 60}", flush=True)

        result = _shap_lightgbm(df_feat, horizon=h)
        if result:
            all_shap_results[f"{h}h"] = result

    # ── 4. Permutation importance for GRU ──
    print("\n[4/4] Permutation importance (GRU)...", flush=True)
    for h in HORIZONS:
        _permutation_gru(df_hybrid, horizon=h)

    # ── Summary ──
    total = time.time() - t_start
    print(f"\n{'═' * 70}", flush=True)
    print(f"COMPLETE — Total: {total:.0f}s ({total / 60:.1f} min)", flush=True)
    print(f"Figures saved to: {FIGURES_DIR}", flush=True)
    print(f"{'═' * 70}", flush=True)

    # Save SHAP results JSON
    json_path = FIGURES_DIR / "shap_results.json"
    with open(json_path, "w") as f:
        json.dump(all_shap_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results JSON: {json_path}", flush=True)


def _prepare_hybrid_data() -> pd.DataFrame:
    df_raw = load_raw_data()
    df = _remove_duplicates(df_raw)
    df = _set_datetime_index(df)
    df, _ = _clip_physical_bounds(df)
    df, _ = _handle_outliers(df, method="iqr", threshold=3.0)
    df = _resample(df, freq="1h")
    return impute_missing_data(
        df,
        strategy="hybrid",
        max_gap_interp=6,
        max_gap_ml=24,
        knn_neighbors=5,
        verbose=True,
    )


def _shap_lightgbm(df_feat: pd.DataFrame, horizon: int) -> dict | None:
    """Train LightGBM and compute SHAP values."""
    import shap
    from lightgbm import LGBMRegressor

    try:
        # ── Prepare data ──
        df = df_feat.copy()
        df["target"] = df[TARGET_COL].shift(-horizon)
        df["_persist"] = df[TARGET_COL]
        df = df.dropna(subset=["target"])

        exclude = ["is_imputed", TARGET_COL, "target", "_persist"]
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "float32", "int64")]

        X = df[feature_cols].fillna(0)
        y = df["target"]
        imp = df["is_imputed"]

        n = len(X)
        tr_end = int(n * 0.8)
        val_end = int(n * 0.9)

        X_train, y_train = X.iloc[:tr_end], y.iloc[:tr_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        test_real = ~imp.iloc[val_end:].values

        X_test_real = X_test[test_real]
        y_test_real = y_test[test_real]

        # ── Train ──
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

        y_pred = model.predict(X_test_real)
        mae = float(np.mean(np.abs(y_test_real.values - y_pred)))
        print(f"    MAE={mae:.3f} (n_test={len(y_test_real)})", flush=True)

        # ── SHAP TreeExplainer ──
        print("  Computing SHAP values...", flush=True)
        t0 = time.time()
        explainer = shap.TreeExplainer(model)

        # Use subset for speed (max 500 samples)
        n_explain = min(500, len(X_test_real))
        X_explain = X_test_real.iloc[:n_explain]
        shap_values = explainer.shap_values(X_explain)
        print(f"    SHAP computed for {n_explain} samples ({time.time() - t0:.1f}s)", flush=True)

        # ── Mean absolute SHAP ──
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)

        print(f"\n  Top-15 SHAP features ({horizon}h):", flush=True)
        for feat, val in shap_importance.head(15).items():
            print(f"    {feat:<40s} SHAP={val:.4f}", flush=True)

        # ── Save SHAP summary plot ──
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_explain, plot_type="bar", max_display=20, show=False)
        plt.title(f"SHAP Feature Importance — LightGBM {horizon}h", fontsize=14)
        plt.tight_layout()
        bar_path = FIGURES_DIR / f"shap_bar_{horizon}h.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {bar_path.name}", flush=True)

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_explain, max_display=20, show=False)
        plt.title(f"SHAP Beeswarm — LightGBM {horizon}h", fontsize=14)
        plt.tight_layout()
        bee_path = FIGURES_DIR / f"shap_beeswarm_{horizon}h.png"
        plt.savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {bee_path.name}", flush=True)

        # Top feature dependence plots (top 3)
        top3 = shap_importance.head(3).index.tolist()
        for feat in top3:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(feat, shap_values, X_explain, show=False, ax=ax)
            plt.title(f"SHAP Dependence — {feat} ({horizon}h)", fontsize=12)
            plt.tight_layout()
            dep_path = FIGURES_DIR / f"shap_dep_{horizon}h_{feat}.png"
            plt.savefig(dep_path, dpi=150, bbox_inches="tight")
            plt.close()
        print("    Saved: 3 dependence plots", flush=True)

        # ── Built-in importance comparison ──
        builtin_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

        # Compare top-10 SHAP vs built-in
        print(f"\n  SHAP vs Built-in importance ({horizon}h):", flush=True)
        print(f"    {'Feature':<35s} {'SHAP rank':>10} {'Built-in rank':>14}", flush=True)
        shap_ranks = {f: i + 1 for i, f in enumerate(shap_importance.index)}
        builtin_ranks = {f: i + 1 for i, f in enumerate(builtin_imp.index)}
        for feat in shap_importance.head(10).index:
            sr = shap_ranks.get(feat, "—")
            br = builtin_ranks.get(feat, "—")
            print(f"    {feat:<35s} {sr:>10} {br:>14}", flush=True)

        return {
            "top_15_shap": {str(k): round(float(v), 4) for k, v in shap_importance.head(15).items()},
            "mae": round(mae, 4),
            "n_test": len(y_test_real),
        }

    except Exception as e:
        import traceback

        print(f"  ❌ SHAP error ({horizon}h): {e}", flush=True)
        traceback.print_exc()
        return None


def _permutation_gru(df_hybrid: pd.DataFrame, horizon: int) -> None:
    """Compute permutation importance for GRU model.

    Optimized: MPS GPU (Apple Silicon M1 Pro) + DataLoader mini-batches.
    """
    # Lazy import to avoid MPS/OpenMP conflict with LightGBM
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Device: MPS if available (Apple Silicon), else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n  GRU Permutation Importance ({horizon}h) — 🚀 MPS GPU", flush=True)
    else:
        device = torch.device("cpu")
        print(f"\n  GRU Permutation Importance ({horizon}h) — CPU", flush=True)

    try:
        t0 = time.time()
        target = df_hybrid[TARGET_COL].values.astype(np.float32)
        is_imputed = df_hybrid["is_imputed"].values
        n = len(target)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        # Normalize
        mean_t = float(target[:train_end].mean())
        std_t = float(target[:train_end].std())
        target_norm = (target - mean_t) / std_t

        # Feature columns (include pm25 history — strongest signal)
        feat_cols = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]
        feat_data = df_hybrid[feat_cols].values.astype(np.float32)
        feat_mean = feat_data[:train_end].mean(axis=0)
        feat_std = feat_data[:train_end].std(axis=0)
        feat_std[feat_std == 0] = 1
        feat_norm = (feat_data - feat_mean) / feat_std

        lookback = 72
        batch_size = 256

        # Create sequences (vectorized for speed)
        valid_range = n - horizon - lookback
        indices = np.arange(lookback, lookback + valid_range)
        X_all = np.stack([feat_norm[i - lookback : i] for i in indices])
        y_all = target_norm[indices + horizon]
        imp_all = is_imputed[indices + horizon]

        # Split
        tr_idx = train_end - lookback
        te_idx = val_end - lookback
        X_train, y_train = X_all[:tr_idx], y_all[:tr_idx]
        X_test, y_test = X_all[te_idx:], y_all[te_idx:]
        imp_test = imp_all[te_idx:]
        real_mask = ~imp_test
        X_test_real, y_test_real = X_test[real_mask], y_test[real_mask]

        print(f"    Train: {len(X_train)}, Test(real): {len(X_test_real)}", flush=True)

        # DataLoader for mini-batch training
        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # GRU model
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

        # Training with mini-batches on MPS — more epochs for convergence
        n_epochs = 50
        best_loss = float("inf")
        patience_counter = 0
        print(f"    Training GRU ({n_epochs} epochs, batch={batch_size})...", flush=True)
        model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / n_batches
            scheduler.step(avg_loss)
            if (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"      Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.4f}, lr={lr:.1e}", flush=True)
            # Early stopping on train loss plateau
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 10:
                print(f"      Early stop at epoch {epoch + 1} (no improvement for 10 epochs)", flush=True)
                break

        train_time = time.time() - t0
        print(f"    Training done ({train_time:.1f}s)", flush=True)

        # Baseline MAE (batch inference)
        model.eval()
        X_te = torch.from_numpy(X_test_real).to(device)
        with torch.no_grad():
            y_pred = model(X_te).cpu().numpy()
        y_pred_orig = y_pred * std_t + mean_t
        y_test_orig = y_test_real * std_t + mean_t
        baseline_mae = float(np.mean(np.abs(y_test_orig - y_pred_orig)))
        print(f"    Baseline MAE: {baseline_mae:.3f}", flush=True)

        # Permutation importance per feature
        print("    Computing permutation importance (5 rounds)...", flush=True)
        perm_importance = {}

        for fi, feat in enumerate(feat_cols):
            mae_permuted_list = []
            for _ in range(5):
                X_perm = X_test_real.copy()
                perm_idx = np.random.permutation(len(X_perm))
                X_perm[:, :, fi] = X_perm[perm_idx, :, fi]

                X_pe = torch.from_numpy(X_perm).to(device)
                with torch.no_grad():
                    y_perm = model(X_pe).cpu().numpy()
                y_perm_orig = y_perm * std_t + mean_t
                mae_perm = float(np.mean(np.abs(y_test_orig - y_perm_orig)))
                mae_permuted_list.append(mae_perm)

            mean_perm_mae = float(np.mean(mae_permuted_list))
            importance = mean_perm_mae - baseline_mae
            perm_importance[feat] = round(importance, 4)
            pct = (importance / baseline_mae) * 100
            print(f"      {feat:<20s} Δ MAE={importance:+.3f} ({pct:+.1f}%)", flush=True)

        # Sort and display
        sorted_imp = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\n    GRU Permutation Ranking ({horizon}h):", flush=True)
        for rank, (feat, imp) in enumerate(sorted_imp, 1):
            print(f"      {rank}. {feat:<20s} Δ MAE={imp:+.4f}", flush=True)

        # Save plot
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feats = [x[0] for x in sorted_imp]
        imps = [x[1] for x in sorted_imp]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in imps]
        ax.barh(range(len(feats)), imps, color=colors)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats)
        ax.set_xlabel("Δ MAE (higher = more important)")
        ax.set_title(f"GRU Permutation Importance — {horizon}h", fontsize=14)
        ax.invert_yaxis()
        plt.tight_layout()

        perm_path = FIGURES_DIR / f"gru_permutation_{horizon}h.png"
        plt.savefig(perm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {perm_path.name}", flush=True)

        total = time.time() - t0
        print(f"    ✅ Done ({total:.1f}s)", flush=True)

    except Exception as e:
        import traceback

        print(f"  ❌ GRU Permutation error ({horizon}h): {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
