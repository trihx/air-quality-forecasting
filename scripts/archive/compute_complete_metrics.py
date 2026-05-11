"""
Compute R², RMSE, MAPE for ALL models across ALL horizons.
Generates a complete metrics table for thesis comparison.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import json
from pathlib import Path

import numpy as np


def main():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from src.data.cleaner import clean_data
    from src.data.imputer import impute_missing_data
    from src.data.loader import load_raw_data
    from src.features.builder import build_features

    print("=" * 70)
    print("  COMPLETE METRICS TABLE FOR THESIS")
    print("=" * 70)

    # --- Load & prepare data ---
    print("\n[1/3] Loading data...")
    raw = load_raw_data()
    cleaned = clean_data(raw)
    imputed = impute_missing_data(cleaned, strategy="hybrid")
    featured = build_features(imputed)

    TARGET = "pm25"
    feature_cols = [c for c in featured.columns if c != TARGET and c != "is_imputed"]

    results = {}

    for horizon in [1, 6, 24]:
        print(f"\n{'─' * 60}")
        print(f"  HORIZON = {horizon}h")
        print(f"{'─' * 60}")

        df = featured.copy()
        df["target"] = df[TARGET].shift(-horizon)
        df = df.dropna(subset=["target"])

        # Filter test to real data only
        if "is_imputed" in df.columns:
            test_mask = df["is_imputed"] == False  # noqa
        else:
            [True] * len(df)

        n = len(df)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        X_train = df[feature_cols].iloc[:train_end]
        y_train = df["target"].iloc[:train_end]
        X_test = df[feature_cols].iloc[val_end:]
        y_test = df["target"].iloc[val_end:]

        # Only real data in test
        test_real_mask = (
            df["is_imputed"].iloc[val_end:] == False if "is_imputed" in df.columns else [True] * len(y_test)
        )  # noqa
        X_test = X_test[test_real_mask]
        y_test = y_test[test_real_mask]

        y_true = y_test.values
        h_results = {}

        # --- Persistence ---
        y_persist = df[TARGET].iloc[val_end:][test_real_mask].values[: len(y_true)]
        persist_mae = mean_absolute_error(y_true, y_persist)
        persist_rmse = np.sqrt(mean_squared_error(y_true, y_persist))
        persist_r2 = r2_score(y_true, y_persist)
        persist_mape = np.mean(np.abs((y_true - y_persist) / np.where(y_true == 0, 1e-8, y_true))) * 100

        h_results["Persistence"] = {
            "MAE": round(persist_mae, 3),
            "RMSE": round(persist_rmse, 3),
            "R2": round(persist_r2, 4),
            "MAPE": round(persist_mape, 2),
            "MASE": 1.000,
        }
        print(
            f"  Persistence: MAE={persist_mae:.3f}, RMSE={persist_rmse:.3f}, R²={persist_r2:.4f}, MAPE={persist_mape:.2f}%"
        )

        # --- LightGBM (Optuna tuned) ---
        try:
            import lightgbm as lgb

            # Load best params from experiments
            with open("research/experiments/multi_horizon/multi_horizon_20260404_215251.json", encoding="utf-8") as f:
                mh_data = json.load(f)

            hkey = f"{horizon}h"
            if hkey in mh_data and "LightGBM_tuned" in mh_data[hkey]:
                best_params = mh_data[hkey]["LightGBM_tuned"].get("optuna_best_params", {})
            elif str(horizon) in mh_data and "LightGBM_tuned" in mh_data[str(horizon)]:
                best_params = mh_data[str(horizon)]["LightGBM_tuned"].get("optuna_best_params", {})
            else:
                best_params = {}

            params = {**best_params, "verbose": -1, "random_state": 42}
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            y_pred_lgb = model.predict(X_test)

            lgb_mae = mean_absolute_error(y_true, y_pred_lgb)
            lgb_rmse = np.sqrt(mean_squared_error(y_true, y_pred_lgb))
            lgb_r2 = r2_score(y_true, y_pred_lgb)
            lgb_mase = lgb_mae / persist_mae
            lgb_mape = np.mean(np.abs((y_true - y_pred_lgb) / np.where(y_true == 0, 1e-8, y_true))) * 100

            h_results["LightGBM"] = {
                "MAE": round(lgb_mae, 3),
                "RMSE": round(lgb_rmse, 3),
                "R2": round(lgb_r2, 4),
                "MAPE": round(lgb_mape, 2),
                "MASE": round(lgb_mase, 3),
            }
            print(
                f"  LightGBM:    MAE={lgb_mae:.3f}, RMSE={lgb_rmse:.3f}, R²={lgb_r2:.4f}, MAPE={lgb_mape:.2f}%, MASE={lgb_mase:.3f}"
            )
        except Exception as e:
            print(f"  LightGBM FAILED: {e}")

        # --- GRU ---
        try:
            import torch
            import torch.nn as nn

            class GRUModel(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
                    super().__init__()
                    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    out, _ = self.gru(x)
                    return self.fc(out[:, -1, :])

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            # Standardize
            from sklearn.preprocessing import StandardScaler

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_tr_sc = scaler_X.fit_transform(X_train.values)
            y_tr_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            X_te_sc = scaler_X.transform(X_test.values)

            # Create sequences (seq_len=24)
            seq_len = 24

            def make_sequences(X, y, seq_len):
                Xs, ys = [], []
                for i in range(seq_len, len(X)):
                    Xs.append(X[i - seq_len : i])
                    ys.append(y[i])
                return np.array(Xs), np.array(ys)

            X_tr_seq, y_tr_seq = make_sequences(X_tr_sc, y_tr_sc, seq_len)
            X_te_seq, _ = make_sequences(X_te_sc, np.zeros(len(X_te_sc)), seq_len)
            y_true_gru = y_true[seq_len:]

            X_tr_t = torch.FloatTensor(X_tr_seq).to(device)
            y_tr_t = torch.FloatTensor(y_tr_seq).to(device)
            X_te_t = torch.FloatTensor(X_te_seq).to(device)

            model_gru = GRUModel(X_tr_seq.shape[2], hidden_size=64, num_layers=2, dropout=0.2).to(device)
            optimizer = torch.optim.Adam(model_gru.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Train
            model_gru.train()
            batch_size = 64
            for epoch in range(50):
                perm = torch.randperm(len(X_tr_t))
                epoch_loss = 0
                n_batches = 0
                for i in range(0, len(X_tr_t), batch_size):
                    idx = perm[i : i + batch_size]
                    xb, yb = X_tr_t[idx], y_tr_t[idx]
                    pred = model_gru(xb).squeeze()
                    loss = criterion(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                if (epoch + 1) % 25 == 0:
                    print(f"      GRU Epoch {epoch + 1}/50: loss={epoch_loss / n_batches:.4f}")

            # Predict
            model_gru.eval()
            with torch.no_grad():
                y_pred_sc = model_gru(X_te_t).cpu().numpy().ravel()
            y_pred_gru = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()

            gru_mae = mean_absolute_error(y_true_gru, y_pred_gru)
            gru_rmse = np.sqrt(mean_squared_error(y_true_gru, y_pred_gru))
            gru_r2 = r2_score(y_true_gru, y_pred_gru)
            gru_mase = gru_mae / persist_mae
            gru_mape = np.mean(np.abs((y_true_gru - y_pred_gru) / np.where(y_true_gru == 0, 1e-8, y_true_gru))) * 100

            h_results["GRU"] = {
                "MAE": round(gru_mae, 3),
                "RMSE": round(gru_rmse, 3),
                "R2": round(gru_r2, 4),
                "MAPE": round(gru_mape, 2),
                "MASE": round(gru_mase, 3),
            }
            print(
                f"  GRU:         MAE={gru_mae:.3f}, RMSE={gru_rmse:.3f}, R²={gru_r2:.4f}, MAPE={gru_mape:.2f}%, MASE={gru_mase:.3f}"
            )
        except Exception as e:
            print(f"  GRU FAILED: {e}")
            import traceback

            traceback.print_exc()

        results[f"{horizon}h"] = h_results

    # Save results
    out_path = Path("research/diagnostics/complete_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS SAVED → {out_path}")
    print(f"{'=' * 70}")

    # Print summary table
    print("\n\n  SUMMARY TABLE FOR THESIS")
    print(f"  {'Model':<20} {'Horizon':<8} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE':>8} {'MASE':>8}")
    print(f"  {'─' * 72}")
    for h, models in results.items():
        for model, m in models.items():
            print(
                f"  {model:<20} {h:<8} {m['MAE']:>8.3f} {m['RMSE']:>8.3f} {m['R2']:>8.4f} {m.get('MAPE', 'N/A'):>8} {m['MASE']:>8.3f}"
            )
        print()


if __name__ == "__main__":
    main()
