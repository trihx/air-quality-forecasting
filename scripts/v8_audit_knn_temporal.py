"""V8 Audit: KNN Imputation Temporal Order Check.

Purpose: Verify whether KNNImputer uses future data points as neighbors
when imputing missing values. This is a look-ahead bias (data leakage) check.

Method:
  1. Reproduce the KNN imputation pipeline
  2. For each imputed point at position `i`, identify which rows KNN used
     as neighbors by manually computing distances
  3. Report % of imputed points where any neighbor index > i (future leak)

Output: research/experiments/v8_final/knn_audit_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import FEATURE_COLS, TARGET_COL


def build_knn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate _build_knn_features from imputer.py."""
    features = pd.DataFrame(index=df.index)

    for col in FEATURE_COLS:
        if col in df.columns:
            features[col] = df[col]

    if hasattr(df.index, "hour"):
        features["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        features["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        features["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    return features


def identify_gaps(series: pd.Series) -> pd.DataFrame:
    """Replicate _identify_gaps from imputer.py."""
    is_nan = series.isna()
    if not is_nan.any():
        return pd.DataFrame(columns=["group_id", "start_idx", "end_idx", "length"])

    groups = (is_nan != is_nan.shift()).cumsum()
    gap_groups = groups[is_nan]

    records = []
    for gid, indices in gap_groups.groupby(gap_groups).groups.items():
        idx_positions = [series.index.get_loc(i) for i in indices]
        records.append({
            "group_id": gid,
            "start_idx": min(idx_positions),
            "end_idx": max(idx_positions),
            "length": len(idx_positions),
        })
    return pd.DataFrame(records)


def audit_knn_temporal_order(
    df: pd.DataFrame,
    max_gap_ml: int = 24,
    knn_neighbors: int = 5,
) -> dict:
    """Audit KNN imputation for temporal order violations.

    Returns:
        Dict with audit results: total imputed, n_with_future, pct_future, details.
    """
    print("=" * 70, flush=True)
    print("V8 AUDIT: KNN Imputation Temporal Order Check", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Identify gaps that KNN would fill (medium gaps 6-24h)
    gap_info = identify_gaps(df[TARGET_COL])
    if len(gap_info) == 0:
        print("No gaps found in data. KNN not used.", flush=True)
        return {"status": "NO_GAPS", "total_imputed": 0}

    # Filter to medium gaps (what hybrid strategy D sends to KNN)
    medium_gaps = gap_info[(gap_info["length"] > 0) & (gap_info["length"] <= max_gap_ml)]
    print(f"\nTotal gap segments: {len(gap_info)}", flush=True)
    print(f"Medium gaps (≤{max_gap_ml}h, sent to KNN): {len(medium_gaps)}", flush=True)

    if len(medium_gaps) == 0:
        print("No medium gaps for KNN. Audit not needed.", flush=True)
        return {"status": "NO_MEDIUM_GAPS", "total_imputed": 0}

    # Collect positions to fill
    positions_to_fill = set()
    for _, gap in medium_gaps.iterrows():
        positions_to_fill.update(range(gap["start_idx"], gap["end_idx"] + 1))

    print(f"Total positions KNN would fill: {len(positions_to_fill)}", flush=True)

    # Step 2: Build feature matrix (same as imputer.py)
    knn_features = build_knn_features(df)
    knn_data = knn_features.copy()
    knn_data[TARGET_COL] = df[TARGET_COL]

    feature_cols = [c for c in knn_data.columns if c != TARGET_COL]

    # Fill feature NaNs with column mean (same as imputer.py)
    for col in feature_cols:
        if knn_data[col].isna().any():
            knn_data[col] = knn_data[col].fillna(knn_data[col].mean())

    # Standardize (same as imputer.py)
    scaler = StandardScaler()
    knn_data[feature_cols] = scaler.fit_transform(knn_data[feature_cols])

    # Step 3: Find neighbors for each imputed point
    # We use NearestNeighbors directly to inspect which rows are used
    # Only COMPLETE rows (non-NaN target) can be donors
    complete_mask = knn_data[TARGET_COL].notna()
    complete_indices = np.where(complete_mask)[0]  # positional indices of complete rows

    print(f"\nComplete rows (potential donors): {len(complete_indices)}", flush=True)
    print(f"Rows with NaN target: {(~complete_mask).sum()}", flush=True)

    # Build NN model on complete rows' features
    complete_features = knn_data.iloc[complete_indices][feature_cols].values
    nn = NearestNeighbors(n_neighbors=knn_neighbors, metric="euclidean")
    nn.fit(complete_features)

    # Step 4: For each imputed position, find neighbors and check temporal order
    n_with_future = 0
    n_audited = 0
    future_leak_details = []

    sorted_positions = sorted(positions_to_fill)
    print(f"\nAuditing {len(sorted_positions)} imputed positions...", flush=True)

    for i, pos in enumerate(sorted_positions):
        if pos >= len(knn_data):
            continue

        # Get features for this position
        point_features = knn_data.iloc[pos][feature_cols].values.reshape(1, -1)

        # Find k nearest neighbors among complete rows
        distances, neighbor_local_indices = nn.kneighbors(point_features)

        # Map local indices back to original DataFrame indices
        neighbor_global_indices = complete_indices[neighbor_local_indices[0]]

        # Check: any neighbor from the future?
        future_neighbors = [int(idx) for idx in neighbor_global_indices if idx > pos]
        has_future = len(future_neighbors) > 0

        if has_future:
            n_with_future += 1
            if len(future_leak_details) < 20:  # Keep first 20 examples
                future_leak_details.append({
                    "imputed_pos": int(pos),
                    "neighbor_positions": [int(x) for x in neighbor_global_indices],
                    "future_neighbor_positions": future_neighbors,
                    "n_future": len(future_neighbors),
                    "max_future_offset": int(max(future_neighbors) - pos),
                })

        n_audited += 1

        if (i + 1) % 100 == 0:
            print(f"  Audited {i+1}/{len(sorted_positions)}...", flush=True)

    # Step 5: Compile results
    pct_future = round(n_with_future / n_audited * 100, 2) if n_audited > 0 else 0.0

    print(f"\n{'=' * 70}", flush=True)
    print("AUDIT RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"Total imputed positions audited: {n_audited}", flush=True)
    print(f"Positions using future neighbors: {n_with_future}", flush=True)
    print(f"Percentage with future leak:      {pct_future}%", flush=True)

    if pct_future > 5:
        print(f"\n🔴 CRITICAL: {pct_future}% of imputed points use future data!", flush=True)
        print("   → KNN imputation has LOOK-AHEAD BIAS.", flush=True)
        print("   → Pipeline MUST be rebuilt with temporal restriction.", flush=True)
        verdict = "FAIL_LOOK_AHEAD_BIAS"
    elif pct_future > 0:
        print(f"\n🟡 WARNING: {pct_future}% of imputed points use future data.", flush=True)
        print("   → Minor look-ahead bias detected.", flush=True)
        verdict = "WARN_MINOR_BIAS"
    else:
        print(f"\n✅ PASS: No future data leakage detected.", flush=True)
        verdict = "PASS"

    result = {
        "audit": "KNN Temporal Order",
        "verdict": verdict,
        "total_gap_segments": len(gap_info),
        "medium_gap_segments": len(medium_gaps),
        "total_imputed_positions": n_audited,
        "positions_with_future_neighbors": n_with_future,
        "pct_future_leak": pct_future,
        "knn_neighbors": knn_neighbors,
        "max_gap_ml": max_gap_ml,
        "complete_donor_rows": len(complete_indices),
        "examples": future_leak_details[:10],
    }

    return result


def main():
    """Run the KNN temporal audit."""
    # Load cleaned + resampled data (pre-imputation state)
    # We need data WITH NaN gaps to simulate what imputer sees
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean_data

    print("Step 1/3: Loading raw data...", flush=True)
    raw = load_raw_data()

    print("Step 2/3: Cleaning data...", flush=True)
    cleaned = clean_data(raw)

    # Resample to hourly (this creates NaN gaps)
    print("Step 3/3: Resampling to hourly...", flush=True)
    cleaned[TARGET_COL] = pd.to_numeric(cleaned[TARGET_COL], errors="coerce")
    for col in FEATURE_COLS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Set datetime index if not already
    if "ngay_tao" in cleaned.columns:
        cleaned = cleaned.set_index("ngay_tao")
    cleaned = cleaned.resample("1h").mean()

    print(f"Resampled: {len(cleaned)} rows, {cleaned[TARGET_COL].isna().sum()} NaN PM2.5", flush=True)

    # Run audit
    result = audit_knn_temporal_order(cleaned)

    # Save results
    output_dir = PROJECT_ROOT / "research" / "experiments" / "v8_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "knn_audit_report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nReport saved to: {output_path}", flush=True)
    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get("verdict") == "PASS" else 1)
