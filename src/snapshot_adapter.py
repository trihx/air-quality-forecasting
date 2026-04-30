"""Snapshot Adapter — Normalize all dashboard snapshot formats into a unified structure.

All dashboard code should read snapshot data through this adapter instead of
parsing raw JSON directly. This eliminates format inconsistencies across v1–v7.

Normalized output format:
    {
        "version": str,
        "timestamp": str,
        "description": str,
        "feature_set": dict,
        "models": list[str],
        "changes": dict,
        "results": {
            "1h": { "ModelName": {"mae": float, "rmse": float|None, "mase": float}, ... },
            "6h": { ... },
            "24h": { ... },
        },
        "best_models": {
            "1h": {"model": str, "mae": float, "mase": float},
            ...
        },
        "top_n": {
            "1h": [{"model": str, "mae": float, "mase": float}, ...],
            ...
        },
    }
"""

from __future__ import annotations

import json
from pathlib import Path

HORIZONS = ("1h", "6h", "24h")
TOP_N = 3  # Number of top models to surface (parameterized)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "research" / "experiments" / "dashboard_runs"


# ── MASE extraction ──────────────────────────────────────────────────


def extract_mase(model_data: dict) -> float:
    """Extract MASE value from a model result dict with key fallback.

    Snapshot formats use different keys for MASE:
      - Persistence baseline uses "mase"
      - Other models use "mase_unified" (preferred) or "mase_original"

    Priority: mase_unified → mase_original → mase → 0.0
    """
    for key in ("mase_unified", "mase_original", "mase"):
        val = model_data.get(key)
        if val is not None:
            return float(val)
    return 0.0


# ── Results normalization ────────────────────────────────────────────


def _extract_results(raw: dict) -> dict[str, dict]:
    """Extract results dict from raw snapshot, handling nesting differences.

    v1–v6: results live under raw["data"]["results"]
    v7:    results live under raw["results"]
    """
    # Try top-level first (v7 format)
    if "results" in raw and isinstance(raw["results"], dict):
        # Verify it's actual results (has horizon keys), not something else
        sample = raw["results"]
        if any(h in sample for h in HORIZONS):
            return sample

    # Try nested under "data" (v1–v6 format)
    data_block = raw.get("data", {})
    if isinstance(data_block, dict) and "results" in data_block:
        return data_block["results"]

    return {}


def _normalize_model_entry(model_name: str, model_data: dict) -> dict:
    """Normalize a single model's metrics to the standard format.

    Strips classification, optuna params, and other non-core fields.
    Keeps only: mae, rmse, mase.
    """
    return {
        "mae": float(model_data.get("mae", 0)),
        "rmse": float(model_data["rmse"]) if model_data.get("rmse") is not None else None,
        "mase": extract_mase(model_data),
    }


def _normalize_results(raw_results: dict) -> dict[str, dict]:
    """Normalize results for all horizons and models."""
    normalized: dict[str, dict] = {}
    for h in HORIZONS:
        h_data = raw_results.get(h, {})
        normalized[h] = {}
        for model_name, model_data in h_data.items():
            if isinstance(model_data, dict):
                normalized[h][model_name] = _normalize_model_entry(model_name, model_data)
    return normalized


# ── Top-N computation ────────────────────────────────────────────────


def _compute_top_n(
    results: dict[str, dict],
    n: int = TOP_N,
    exclude: tuple[str, ...] = ("Persistence",),
) -> dict[str, list[dict]]:
    """Compute top N models per horizon, sorted by MAE ascending.

    Args:
        results: Normalized results dict.
        n: Number of top models to return.
        exclude: Model names to exclude from ranking (e.g., baseline).

    Returns:
        Dict mapping horizon → list of top N model dicts.
    """
    top: dict[str, list[dict]] = {}
    for h in HORIZONS:
        h_models = results.get(h, {})
        ranked = []
        for model_name, metrics in h_models.items():
            if model_name in exclude:
                continue
            ranked.append({
                "model": model_name,
                "mae": metrics["mae"],
                "mase": metrics["mase"],
            })
        ranked.sort(key=lambda x: x["mae"])
        top[h] = ranked[:n]
    return top


def _compute_best_models(results: dict[str, dict]) -> dict[str, dict]:
    """Compute the single best model per horizon (lowest MAE, excluding Persistence)."""
    top_1 = _compute_top_n(results, n=1)
    best: dict[str, dict] = {}
    for h in HORIZONS:
        if top_1.get(h):
            best[h] = top_1[h][0]
        else:
            best[h] = {"model": "N/A", "mae": 0.0, "mase": 0.0}
    return best


# ── Models list derivation ───────────────────────────────────────────


def _derive_models(results: dict[str, dict]) -> list[str]:
    """Derive unique model list from results, sorted alphabetically."""
    models: set[str] = set()
    for h_data in results.values():
        models.update(h_data.keys())
    return sorted(models)


# ── Main adapter ─────────────────────────────────────────────────────


def normalize_snapshot(raw: dict) -> dict:
    """Convert a raw snapshot JSON dict into the normalized format.

    This is the single entry point for all snapshot data consumption.
    """
    raw_results = _extract_results(raw)
    results = _normalize_results(raw_results)
    models = _derive_models(results)

    return {
        "version": raw.get("version", ""),
        "timestamp": raw.get("timestamp", ""),
        "description": raw.get("description", ""),
        "feature_set": raw.get("feature_set", {}),
        "models": models,
        "changes": raw.get("changes", {}),
        "results": results,
        "best_models": _compute_best_models(results),
        "top_n": _compute_top_n(results, n=TOP_N),
    }


def load_all_normalized() -> dict[str, dict]:
    """Load and normalize all snapshot files from dashboard_runs/.

    Returns:
        Dict mapping version name → normalized snapshot dict.
    """
    snapshots: dict[str, dict] = {}
    if not RUNS_DIR.exists():
        return snapshots
    for jpath in sorted(RUNS_DIR.glob("*.json")):
        try:
            raw = json.loads(jpath.read_text(encoding="utf-8"))
            normalized = normalize_snapshot(raw)
            version = normalized["version"] or jpath.stem
            snapshots[version] = normalized
        except (json.JSONDecodeError, KeyError):
            continue
    return snapshots
