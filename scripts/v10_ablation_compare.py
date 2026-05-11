"""v10 Ablation Study — Phase 3: Compare v9 (Domain Bounds) vs v10 (IQR).

Reads v9 standardized_metrics.json + v10 ablation metrics, 
generates comparison table JSON and grouped bar chart.

Usage:
    $env:PYTHONIOENCODING='utf-8'; uv run python scripts/v10_ablation_compare.py
"""
from __future__ import annotations

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "research" / "experiments" / "v10_ablation"

# Model mapping: v9 JSON key -> v10 JSON key -> display name
MODEL_MAP = {
    "Persistence_30m": "Persistence_v10",
    "ElasticNet_v9_30m": "ElasticNet_v10",
    "LightGBM_v9_30m": "LightGBM_v10",
    "GRU_v9_30m": "GRU_v10",
    "LSTM_v9_30m": "LSTM_v10",
    "TFT_v9_30m": "TFT_v10",
    "Ensemble_Weighted_v9_30m": "Ensemble_v10",
}

DISPLAY_NAMES = {
    "Persistence_30m": "Persistence",
    "ElasticNet_v9_30m": "ElasticNet",
    "LightGBM_v9_30m": "LightGBM",
    "GRU_v9_30m": "GRU",
    "LSTM_v9_30m": "LSTM",
    "TFT_v9_30m": "TFT",
    "Ensemble_Weighted_v9_30m": "Ensemble",
}


def load_v9_metrics() -> dict:
    """Load v9 standardized metrics."""
    path = PROJECT_ROOT / "research" / "experiments" / "standardized_metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data)


def load_v10_metrics() -> dict:
    """Load latest v10 ablation metrics."""
    files = sorted(OUTPUT_DIR.glob("ablation_metrics_*.json"))
    if not files:
        print("ERROR: No v10 ablation metrics found!")
        sys.exit(1)
    path = files[-1]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_v9_model(v9_data: dict, model_name: str, horizon: str) -> dict | None:
    """Extract metrics for a model from v9 standardized structure.
    
    Structure: results -> {1h, 6h, 24h} -> ModelName_v9_30m -> {mae, mase, ...}
    """
    horizon_data = v9_data.get(horizon, {})
    return horizon_data.get(model_name)


def main():
    print("=" * 70, flush=True)
    print("v10 ABLATION COMPARISON: v9 (Domain Bounds) vs v10 (IQR)", flush=True)
    print("=" * 70, flush=True)

    v9 = load_v9_metrics()
    v10 = load_v10_metrics()

    comparison = {}
    horizons = ["1h", "6h", "24h"]

    print(f"\n{'Model':<15} {'Horizon':<8} {'v9 MAE':<10} {'v10 MAE':<10} {'Delta_MAE':<10} {'v9 MASE':<10} {'v10 MASE':<10} {'Delta_MASE':<11} {'Note'}", flush=True)
    print("-" * 110, flush=True)

    for v9_name, v10_name in MODEL_MAP.items():
        for h in horizons:
            if h not in comparison:
                comparison[h] = {}
                
            v9_m = extract_v9_model(v9, v9_name, h)
            v10_h = v10.get(h, {})
            v10_m = v10_h.get(v10_name, None)

            if v9_m is None or v10_m is None:
                continue

            v9_mae = v9_m.get("mae", 0)
            v10_mae = v10_m.get("mae", 0)
            v9_mase = v9_m.get("mase_unified", v9_m.get("mase", 0))
            v10_mase = v10_m.get("mase", 0)
            delta_mae = v10_mae - v9_mae
            delta_mase = v10_mase - v9_mase

            note = ""
            if delta_mase < 0:
                note = "FALSE ACCURACY"
            else:
                note = "DOMAIN BETTER"

            comparison[h][v9_name] = {
                "v9_mae": v9_mae, "v10_mae": v10_mae, "delta_mae": round(delta_mae, 4),
                "v9_mase": v9_mase, "v10_mase": v10_mase, "delta_mase": round(delta_mase, 4),
                "note": note
            }

            print(f"{v9_name:<25} {h:<8} {v9_mase:<10.4f} {v10_mase:<10.4f} {delta_mase:<+11.4f} {note}", flush=True)

    # Save comparison
    comp_path = OUTPUT_DIR / "comparison_table.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {comp_path}", flush=True)

    # Generate grouped bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # Use Times New Roman if available
        for fname in font_manager.findSystemFonts():
            if "times" in fname.lower() and "new" in fname.lower():
                font_manager.fontManager.addfont(fname)
                plt.rcParams["font.family"] = "Times New Roman"
                break

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
        models_v9 = list(MODEL_MAP.keys())
        display_models = [DISPLAY_NAMES[m] for m in models_v9]
        x = np.arange(len(models_v9))
        width = 0.35

        for idx, h in enumerate(horizons):
            v9_mases = []
            v10_mases = []
            for m in models_v9:
                data = comparison.get(h, {}).get(m, {})
                v9_mases.append(data.get("v9_mase", 0))
                v10_mases.append(data.get("v10_mase", 0))

            ax = axes[idx]
            bars1 = ax.bar(x - width/2, v9_mases, width, label="v9 Domain Bounds", color="#2196F3", alpha=0.85)
            bars2 = ax.bar(x + width/2, v10_mases, width, label="v10 IQR (flawed)", color="#FF5722", alpha=0.85)
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="MASE = 1.0 (Naive)")
            ax.set_xlabel("Model", fontsize=11)
            ax.set_ylabel("MASE", fontsize=11)
            ax.set_title(f"Horizon {h}", fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(display_models, rotation=45, ha="right", fontsize=9)
            ax.legend(fontsize=8)

        fig.suptitle("Ablation Study: Outlier Removal Strategy Impact on MASE", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig_path = PROJECT_ROOT / "research" / "figures" / "ablation_outlier_impact.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved: {fig_path}", flush=True)
        plt.close()
    except Exception as e:
        print(f"Chart generation failed: {e}", flush=True)

    # Print Markdown table for thesis
    print(f"\n{'='*70}", flush=True)
    print("MARKDOWN TABLE (copy to thesis):", flush=True)
    print(f"{'='*70}\n", flush=True)
    print("| Horizon | Model | v9 MASE (Domain Bounds) | v10 MASE (IQR Flawed) | Delta MASE | Đánh giá |")
    print("|---------|-------|-------------------------|-----------------------|------------|----------|")
    for h in horizons:
        for m in models_v9:
            d = comparison.get(h, {}).get(m, {})
            if d:
                status = "Ảo giác (MASE giảm ảo)" if d['note'] == "FALSE ACCURACY" else "Domain tốt hơn"
                print(f"| {h} | {m} | {d['v9_mase']:.4f} | {d['v10_mase']:.4f} | {d['delta_mase']:+.4f} | {status} |")


if __name__ == "__main__":
    main()
