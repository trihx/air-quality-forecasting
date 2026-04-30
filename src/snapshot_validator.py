"""Snapshot Validator — Validates JSON schema for dashboard snapshots.

Ensures every snapshot file in dashboard_runs/ conforms to the required
contract: has all 3 horizons, Persistence baseline, valid MASE/MAE values.

Usage:
    python src/snapshot_validator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HORIZONS = ("1h", "6h", "24h")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "research" / "experiments" / "dashboard_runs"


def validate_snapshot(filepath: Path) -> list[str]:
    """Validate a single snapshot file against the schema contract.

    Returns:
        List of error strings (empty = valid).
    """
    errors = []

    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    # ── Required top-level keys ──
    if "version" not in data:
        errors.append("Missing 'version' key")

    # ── Results structure ──
    results = data.get("results", data.get("data", {}).get("results", {}))
    if not results:
        errors.append("No 'results' block found")
        return errors

    for h in HORIZONS:
        if h not in results:
            errors.append(f"Missing horizon '{h}' in results")
            continue

        h_data = results[h]
        if not isinstance(h_data, dict):
            errors.append(f"results['{h}'] is not a dict")
            continue

        # Check Persistence exists
        if "Persistence" not in h_data:
            errors.append(f"Missing 'Persistence' baseline in {h}")

        # Validate model metrics
        for model_name, model_data in h_data.items():
            if not isinstance(model_data, dict):
                errors.append(f"{h}/{model_name}: model data is not a dict")
                continue

            mae = model_data.get("mae")
            if mae is None:
                errors.append(f"{h}/{model_name}: missing 'mae'")
            elif not isinstance(mae, (int, float)) or mae < 0:
                errors.append(f"{h}/{model_name}: invalid MAE={mae}")

            # Check MASE (try multiple keys)
            mase = None
            for key in ("mase_unified", "mase_original", "mase"):
                if key in model_data:
                    mase = model_data[key]
                    break
            if mase is None:
                errors.append(f"{h}/{model_name}: missing MASE (checked mase_unified, mase_original, mase)")
            elif not isinstance(mase, (int, float)) or mase < 0:
                errors.append(f"{h}/{model_name}: invalid MASE={mase}")
            elif mase > 5.0:
                errors.append(f"{h}/{model_name}: suspicious MASE={mase} (>5.0)")

    return errors


def main():
    print("=" * 60)
    print("Snapshot Validator — Schema Check")
    print("=" * 60)

    if not RUNS_DIR.exists():
        print(f"❌ Runs directory not found: {RUNS_DIR}")
        sys.exit(1)

    all_errors = {}
    json_files = sorted(RUNS_DIR.glob("*.json"))

    if not json_files:
        print("⚠️  No snapshot files found")
        sys.exit(1)

    for filepath in json_files:
        errors = validate_snapshot(filepath)
        all_errors[filepath.name] = errors
        status = "✅ VALID" if not errors else f"❌ {len(errors)} error(s)"
        print(f"  {filepath.name:40s} {status}")
        for e in errors:
            print(f"    → {e}")

    total_errors = sum(len(e) for e in all_errors.values())
    if total_errors > 0:
        print(f"\n❌ Total: {total_errors} error(s) across {len(json_files)} files")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(json_files)} snapshot files are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
