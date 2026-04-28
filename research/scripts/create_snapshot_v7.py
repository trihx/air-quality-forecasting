"""Create v7_pre_cqr snapshot — preserving current state before CQR upgrade.

Usage:
    uv run python research/scripts/create_snapshot_v7.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # Load current standardized metrics
    metrics_path = PROJECT_ROOT / "research" / "experiments" / "standardized_metrics.json"
    with open(metrics_path) as f:
        std_data = json.load(f)

    # Load current prediction intervals
    pi_path = PROJECT_ROOT / "research" / "experiments" / "prediction_intervals" / "prediction_intervals_20260405_100353.json"
    pi_data = []
    if pi_path.exists():
        with open(pi_path) as f:
            pi_data = json.load(f)

    # Build snapshot
    snapshot = {
        "version": "v7_pre_cqr",
        "timestamp": datetime.now().isoformat(),
        "parent_version": "v6_pca_tft",
        "description": "Snapshot trước khi nâng cấp CQR. Lưu lại trạng thái MC Dropout (overconfident) để so sánh.",
        "changes": {
            "what": "Snapshot bảo toàn trước CQR upgrade. MC Dropout coverage: h=1: 36.8%, h=6: 7.6%, h=24: 25.7%",
            "why": "Lưu lại bản cũ để so sánh phiên bản trước/sau CQR trên dashboard",
            "result": "MC Dropout overconfident — khoảng dự báo quá hẹp, không đạt target 90%",
            "conclusion": "Cần nâng cấp sang CQR (Romano et al., 2019) để đạt coverage guarantee",
        },
        "feature_set": {
            "lag": True,
            "rolling": True,
            "ewm": True,
            "diff": True,
            "calendar": True,
            "domain": True,
            "fourier": True,
            "interaction": True,
            "anti_leakage": True,
            "purging_gap": True
        },
        "models_included": list(std_data.get("results", {}).get("1h", {}).keys()),
        "new_in_this_version": ["Snapshot preservation before CQR"],
        "prediction_intervals": {
            "methods": ["conformal_prediction", "quantile_regression", "mc_dropout"],
            "gru_coverage": {
                "1h": 0.3675,
                "6h": 0.0762,
                "24h": 0.2566,
            },
            "note": "MC Dropout coverage FAR below 90% target",
        },
        "data": {
            "results": std_data.get("results", {}),
        },
    }

    # Save
    out_dir = PROJECT_ROOT / "research" / "experiments" / "dashboard_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v7_pre_cqr_20260428.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved v7 snapshot: {out_path}")
    print(f"   Models: {len(snapshot['models_included'])}")
    print(f"   MC Dropout coverage: {snapshot['prediction_intervals']['gru_coverage']}")


if __name__ == "__main__":
    main()
