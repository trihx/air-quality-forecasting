"""
Bootstrap Confidence Intervals for MASE.
Computes 95% CI via block bootstrap (preserves temporal structure).
Output: JSON + thesis-ready table.
"""

import json
import numpy as np
from pathlib import Path

# --- Config ---
PROJECT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT / "research" / "experiments" / "v9_final"
OUTPUT = PROJECT / "research" / "diagnostics" / "bootstrap_mase_ci.json"

HORIZONS = ["1h", "6h", "24h"]
N_BOOTSTRAP = 2000
BLOCK_SIZE = 24  # Block bootstrap: 24-step blocks to preserve autocorrelation
CONFIDENCE = 0.95
SEED = 42

# Unified Persistence MAE (anchor test set — consistent with standardized_metrics.json)
# Source: research/experiments/v8_final/unified_mase.json
UNIFIED_PERSIST_MAE = {
    "1h": 2.5964,
    "6h": 6.9323,
    "24h": 6.3273,
}

# Model -> prediction file mapping (v9, 30m resolution = best)
MODEL_FILES = {
    "GRU_v9_30m": ("dl_preds_30m_20260502_123725.json", "GRU_v9"),
    "LSTM_v9_30m": ("dl_preds_30m_20260502_123725.json", "LSTM_v9"),
    "LightGBM_v9_30m": ("lgbm_preds_30m_20260502_084115.json", "LightGBM_v9"),
    "Ensemble_30m": ("ensemble_preds_30m_20260502_193322.json", "Ensemble_Weighted_v9"),
}


def load_predictions(filename: str, model_key: str, horizon: str):
    """Load actual/predicted pairs from JSON."""
    path = PRED_DIR / filename
    with open(path) as f:
        data = json.load(f)

    h_data = data[horizon]
    actuals = np.array(h_data["Actuals"])
    preds = np.array(h_data[model_key])

    if len(actuals) == 0 or len(preds) == 0:
        return None, None, None

    persist_key = "Persistence"
    if persist_key in h_data:
        persistence = np.array(h_data[persist_key])
    else:
        h_steps = {"1h": 2, "6h": 12, "24h": 48}  # 30m resolution
        step = h_steps[horizon]
        persistence = np.roll(actuals, step)
        persistence[:step] = np.nan

    return actuals, preds, persistence


def compute_mase_unified(actuals, preds, horizon):
    """Compute MASE = MAE_model / Unified_MAE_persistence."""
    mask = ~(np.isnan(actuals) | np.isnan(preds))
    a, p = actuals[mask], preds[mask]

    mae_model = np.mean(np.abs(a - p))
    mae_persist = UNIFIED_PERSIST_MAE[horizon]

    return mae_model / mae_persist


def block_bootstrap_mase(actuals, preds, horizon, n_boot, block_size, rng):
    """Block bootstrap for MASE using unified Persistence denominator."""
    mask = ~(np.isnan(actuals) | np.isnan(preds))
    a, p = actuals[mask], preds[mask]
    mae_persist = UNIFIED_PERSIST_MAE[horizon]

    n = len(a)
    n_blocks = max(1, n // block_size)
    mase_samples = []

    for _ in range(n_boot):
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, min(s + block_size, n)) for s in block_starts])
        indices = indices[:n]

        a_boot = a[indices]
        p_boot = p[indices]

        mae_model = np.mean(np.abs(a_boot - p_boot))
        mase_samples.append(mae_model / mae_persist)

    return np.array(mase_samples)


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    alpha = 1 - CONFIDENCE

    print(f"Bootstrap MASE CI (n={N_BOOTSTRAP}, block_size={BLOCK_SIZE}, confidence={CONFIDENCE})")
    print("=" * 80)

    for model_name, (filename, model_key) in MODEL_FILES.items():
        results[model_name] = {}
        for horizon in HORIZONS:
            actuals, preds, persistence = load_predictions(filename, model_key, horizon)
            if actuals is None:
                print(f"  SKIP: {model_name} @ {horizon} -- no data")
                continue

            point_mase = compute_mase_unified(actuals, preds, horizon)
            boot_samples = block_bootstrap_mase(
                actuals, preds, horizon, N_BOOTSTRAP, BLOCK_SIZE, rng
            )

            ci_lower = np.percentile(boot_samples, 100 * alpha / 2)
            ci_upper = np.percentile(boot_samples, 100 * (1 - alpha / 2))
            std = np.std(boot_samples)

            results[model_name][horizon] = {
                "mase_point": round(float(point_mase), 4),
                "ci_lower": round(float(ci_lower), 4),
                "ci_upper": round(float(ci_upper), 4),
                "ci_width": round(float(ci_upper - ci_lower), 4),
                "std": round(float(std), 4),
                "n_samples": len(actuals),
                "n_bootstrap": N_BOOTSTRAP,
                "block_size": BLOCK_SIZE,
            }

            print(
                f"  {model_name:25s} @ {horizon}: "
                f"MASE = {point_mase:.4f} "
                f"[{ci_lower:.4f}, {ci_upper:.4f}] "
                f"(width={ci_upper - ci_lower:.4f}, n={len(actuals)})"
            )

    # Save results
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT}")

    # Print thesis-ready table
    print("\n" + "=" * 80)
    print("THESIS TABLE: Bootstrap 95% CI for MASE (v9, 30m resolution)")
    print("=" * 80)
    print(f"| {'Model':25s} | {'h=1 MASE [95% CI]':30s} | {'h=6 MASE [95% CI]':30s} | {'h=24 MASE [95% CI]':30s} |")
    print(f"|{'-'*27}|{'-'*32}|{'-'*32}|{'-'*32}|")

    for model_name in MODEL_FILES:
        row = f"| {model_name:25s} |"
        for horizon in HORIZONS:
            if horizon in results.get(model_name, {}):
                r = results[model_name][horizon]
                cell = f" {r['mase_point']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
                row += f"{cell:31s} |"
            else:
                row += f" {'N/A':30s} |"
        print(row)


if __name__ == "__main__":
    main()
