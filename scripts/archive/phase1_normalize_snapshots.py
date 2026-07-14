"""Phase 1: Normalize snapshot versions — rename files, update version keys, merge v7.

Steps:
  1. Rename v1–v6 files: strip date suffix
  2. Update 'version' key inside each JSON to match new filename
  3. Merge v7_retrain (13 models) INTO v7_cqr (6 models) → keep v7_cqr.json
  4. Delete v7_retrain file

Author: trihx
"""

import json
import shutil
from pathlib import Path

RUNS_DIR = Path("research/experiments/dashboard_runs")

# ── Step 1+2: Rename v1-v6 and update version key ──

RENAME_MAP = {
    "v1_baseline_20260411.json": ("v1_baseline.json", "v1_baseline"),
    "v2_enhanced_20260411.json": ("v2_enhanced.json", "v2_enhanced"),
    "v3_sklearn_ensemble_20260412.json": ("v3_sklearn_ensemble.json", "v3_sklearn_ensemble"),
    "v4_roc_auc_20260412.json": ("v4_roc_auc.json", "v4_roc_auc"),
    "v5_dl_retrain_20260412.json": ("v5_dl_retrain.json", "v5_dl_retrain"),
    "v6_pca_tft_20260412.json": ("v6_pca_tft.json", "v6_pca_tft"),
}


def rename_and_update(old_name: str, new_name: str, new_version: str) -> bool:
    """Rename file and update version key inside JSON."""
    old_path = RUNS_DIR / old_name
    new_path = RUNS_DIR / new_name

    if not old_path.exists():
        print(f"  ⚠️  SKIP (not found): {old_name}")
        return False

    # Load, update version, write
    data = json.loads(old_path.read_text(encoding="utf-8"))
    old_version = data.get("version", "?")
    data["version"] = new_version

    new_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Remove old file (if names differ)
    if old_path != new_path:
        old_path.unlink()

    print(f"  ✅ {old_name} → {new_name}  (version: '{old_version}' → '{new_version}')")
    return True


# ── Step 3: Merge v7_retrain into v7_cqr ──


def merge_v7():
    """Merge v7_retrain (13 models) INTO v7_cqr (6 models).

    Strategy:
    - Start with v7_cqr as base (has CQR metadata, changes block)
    - For each horizon: merge model results from v7_retrain INTO v7_cqr
    - If model exists in both → use v7_retrain (newer, retrained)
    - Update models_included, version, parent_version
    """
    cqr_path = RUNS_DIR / "v7_cqr_20260428.json"
    retrain_path = RUNS_DIR / "v7_retrain_cqr_20260429_203805.json"

    if not cqr_path.exists():
        print(f"  ❌ v7_cqr base not found: {cqr_path}")
        return False
    if not retrain_path.exists():
        print(f"  ❌ v7_retrain not found: {retrain_path}")
        return False

    cqr = json.loads(cqr_path.read_text(encoding="utf-8"))
    retrain = json.loads(retrain_path.read_text(encoding="utf-8"))

    # Count models before merge
    cqr_results = cqr.get("results", {})
    retrain_results = retrain.get("results", {})

    models_before = set()
    for h in ("1h", "6h", "24h"):
        models_before.update(cqr_results.get(h, {}).keys())

    # Merge results: retrain overwrites cqr for same model
    for h in ("1h", "6h", "24h"):
        if h not in cqr_results:
            cqr_results[h] = {}
        for model_name, model_data in retrain_results.get(h, {}).items():
            cqr_results[h][model_name] = model_data

    cqr["results"] = cqr_results

    # Count models after merge
    models_after = set()
    for h in ("1h", "6h", "24h"):
        models_after.update(cqr_results.get(h, {}).keys())

    # Update metadata
    cqr["version"] = "v7_cqr"
    cqr["parent_version"] = "v6_pca_tft"
    cqr["models_included"] = sorted(models_after)

    # Merge unified_persistence from retrain if cqr doesn't have it complete
    if retrain.get("unified_persistence"):
        cqr["unified_persistence"] = retrain["unified_persistence"]

    # Ensure changes block exists
    if not cqr.get("changes"):
        cqr["changes"] = {
            "what": "Unified Persistence baseline + CQR + retrain toàn bộ 13 models",
            "why": "Chuẩn hoá baseline MASE cho so sánh công bằng, thêm CQR prediction intervals",
            "result": "13 models đánh giá trên cùng test set, GRU best 6h (MASE=0.649), LSTM best 24h (MASE=0.663)",
            "conclusion": "Pipeline v7 hoàn thiện cho luận văn: unified baseline + full model zoo + CQR",
        }

    # Write merged file with new name
    out_path = RUNS_DIR / "v7_cqr.json"
    out_path.write_text(json.dumps(cqr, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  ✅ Merged v7_retrain ({len(retrain_results.get('1h', {}))} models)")
    print(f"     INTO v7_cqr ({len(cqr_results.get('1h', {}))} models)")
    print(f"     Models before: {len(models_before)} → after: {len(models_after)}")
    print(f"     Output: {out_path.name}")

    return True


def cleanup_old_v7():
    """Remove old v7 files after merge."""
    old_files = [
        RUNS_DIR / "v7_cqr_20260428.json",
        RUNS_DIR / "v7_retrain_cqr_20260429_203805.json",
    ]
    for f in old_files:
        if f.exists():
            f.unlink()
            print(f"  🗑️  Deleted: {f.name}")


# ── Main ──

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: Normalize Snapshot Versions")
    print("=" * 60)

    print("\n📁 Step 1+2: Rename files & update version keys")
    for old_name, (new_name, new_version) in RENAME_MAP.items():
        rename_and_update(old_name, new_name, new_version)

    print("\n🔀 Step 3: Merge v7_retrain → v7_cqr")
    merge_ok = merge_v7()

    if merge_ok:
        print("\n🗑️  Step 4: Cleanup old v7 files")
        cleanup_old_v7()

    print("\n📋 Final state:")
    for f in sorted(RUNS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        version = data.get("version", "?")
        results = data.get("results", data.get("data", {}).get("results", {}))
        n_models = len(set().union(*(results.get(h, {}).keys() for h in ("1h", "6h", "24h"))))
        print(f"  {f.name:40s} version='{version}' models={n_models}")

    print("\n✅ Phase 1 complete!")
