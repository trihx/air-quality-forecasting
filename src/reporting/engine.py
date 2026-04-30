"""Reporting Engine — Single Source of Truth for Dashboard metrics.

Loads normalized snapshot data and provides parameterized access to:
- KPI values (best model per horizon)
- Ranking tables (auto-sorted, auto-starred)
- Chart data (MASE/MAE dicts ready for Plotly)
- Insight text (auto-generated from actual data, zero hardcode)

Usage:
    from src.reporting import ReportingEngine
    rpt = ReportingEngine(snapshot_data)
    best_6h = rpt.get_best_model("6h")  # {"model": "GRU", "mase": 0.649, ...}
"""

from __future__ import annotations

import pandas as pd

# ── Model Type Registry (parameterized) ──────────────────────────────

MODEL_TYPES: dict[str, str] = {
    "Persistence": "Baseline",
    "ARIMA": "Statistical",
    "SARIMA": "Statistical",
    "LightGBM": "ML",
    "LightGBM_tuned": "ML",
    "RandomForest": "ML",
    "GradientBoosting": "ML",
    "Stacking": "Ensemble",
    "Ensemble_Weighted": "Ensemble",
    "Ensemble_Stack": "Ensemble",
    "Ensemble_GRU": "Ensemble",
    "GRU": "Deep Learning",
    "LSTM": "Deep Learning",
    "TFT": "Transformer",
}

HORIZONS = ("1h", "6h", "24h")


class ReportingEngine:
    """Parameterized reporting engine — all metrics derived from snapshot data.

    Attributes:
        version: Snapshot version name (e.g. "v7_cqr").
        results: Normalized results dict {horizon: {model: {mae, rmse, mase}}}.
        models: Sorted list of all model names.
    """

    def __init__(self, snapshot_data: dict):
        """Initialize from a normalized snapshot (output of snapshot_adapter).

        Args:
            snapshot_data: Dict with keys: version, results, models, etc.
        """
        self._raw = snapshot_data
        self.version: str = snapshot_data.get("version", "unknown")
        self.results: dict = snapshot_data.get("results", {})
        self.models: list[str] = snapshot_data.get("models", [])
        self.description: str = snapshot_data.get("description", "")
        self.changes: dict = snapshot_data.get("changes", {})
        self.timestamp: str = snapshot_data.get("timestamp", "")

    # ── Best Model per Horizon ───────────────────────────────────────

    def get_best_model(self, horizon: str) -> dict:
        """Get the best model for a given horizon (lowest MAE, excluding Persistence).

        Returns:
            Dict with keys: model, mae, mase, type, improvement_pct
        """
        h_data = self.results.get(horizon, {})
        if not h_data:
            return {"model": "N/A", "mae": 0.0, "mase": 0.0, "type": "?", "improvement_pct": 0.0}

        # Get Persistence MAE for improvement calculation
        persistence_mae = h_data.get("Persistence", {}).get("mae", 0.0)

        # Find best non-Persistence model by MAE
        best = None
        for model_name, metrics in h_data.items():
            if model_name == "Persistence":
                continue
            mae = metrics.get("mae", float("inf"))
            if best is None or mae < best["mae"]:
                best = {
                    "model": model_name,
                    "mae": mae,
                    "mase": metrics.get("mase", 0.0),
                    "type": MODEL_TYPES.get(model_name, "Unknown"),
                }

        if best is None:
            return {"model": "N/A", "mae": 0.0, "mase": 0.0, "type": "?", "improvement_pct": 0.0}

        # Calculate improvement vs Persistence
        if persistence_mae > 0:
            best["improvement_pct"] = (1.0 - best["mase"]) * 100
        else:
            best["improvement_pct"] = 0.0

        return best

    def get_best_models_all(self) -> dict[str, dict]:
        """Get best model for all horizons.

        Returns:
            {"1h": {...}, "6h": {...}, "24h": {...}}
        """
        return {h: self.get_best_model(h) for h in HORIZONS}

    # ── Ranking Table ────────────────────────────────────────────────

    def get_ranking_table(self, top_n: int = 6) -> pd.DataFrame:
        """Generate ranking table sorted by MAE, with star markers for best.

        Args:
            top_n: Number of top models to include per horizon ranking.

        Returns:
            DataFrame with columns: Model, Type, 1h_MASE, 6h_MASE, 24h_MASE
        """
        # Collect all models across horizons
        all_models = set()
        for h in HORIZONS:
            all_models.update(self.results.get(h, {}).keys())

        rows = []
        for model in sorted(all_models):
            row = {
                "Model": model,
                "Type": MODEL_TYPES.get(model, "Unknown"),
            }
            for h in HORIZONS:
                mase = self.results.get(h, {}).get(model, {}).get("mase", None)
                row[f"{h}_MASE"] = mase
                row[f"{h}_MAE"] = self.results.get(h, {}).get(model, {}).get("mae", None)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by average MASE across horizons (lower = better)
        def avg_mase(row):
            vals = [row.get(f"{h}_MASE") for h in HORIZONS]
            valid = [v for v in vals if v is not None]
            return sum(valid) / len(valid) if valid else float("inf")

        df["_avg_mase"] = df.apply(avg_mase, axis=1)
        df = df.sort_values("_avg_mase").head(top_n).drop(columns=["_avg_mase"])

        return df.reset_index(drop=True)

    def get_ranking_display(self, top_n: int = 6) -> pd.DataFrame:
        """Get ranking table formatted for display with ⭐ markers.

        Returns DataFrame with string-formatted MASE values and star markers.
        """
        df = self.get_ranking_table(top_n=top_n)

        # Find best per horizon (excluding Persistence)
        best_per_h = {}
        for h in HORIZONS:
            col = f"{h}_MASE"
            non_persist = df[df["Model"] != "Persistence"]
            if not non_persist.empty and non_persist[col].notna().any():
                best_idx = non_persist[col].idxmin()
                best_per_h[h] = df.loc[best_idx, "Model"]

        # Format with stars
        for h in HORIZONS:
            col = f"{h}_MASE"
            df[col] = df.apply(
                lambda row: (
                    f"⭐ {row[col]:.3f}" if row["Model"] == best_per_h.get(h) and row[col] is not None
                    else (f"{row[col]:.3f}" if row[col] is not None else "—")
                ),
                axis=1,
            )
            # Drop MAE columns (internal, not for display)
            mae_col = f"{h}_MAE"
            if mae_col in df.columns:
                df = df.drop(columns=[mae_col])

        return df

    # ── Chart Data ───────────────────────────────────────────────────

    def get_mase_data(self, model_filter: list[str] | None = None) -> dict[str, list[float]]:
        """Get MASE values formatted for chart plotting.

        Args:
            model_filter: Optional list of model names to include.
                         If None, includes all models in snapshot.

        Returns:
            {"Persistence": [1.0, 1.0, 1.0], "GRU": [1.009, 0.649, 0.726], ...}
        """
        models_to_use = model_filter or sorted(self.results.get("1h", {}).keys())
        data = {}
        for model in models_to_use:
            vals = []
            for h in HORIZONS:
                mase = self.results.get(h, {}).get(model, {}).get("mase", 0.0)
                vals.append(round(mase, 4))
            data[model] = vals
        return data

    def get_mae_data(self, model_filter: list[str] | None = None) -> dict[str, list[float]]:
        """Get MAE values formatted for chart plotting.

        Returns:
            {"Persistence": [2.493, 6.422, 6.454], "GRU": [2.515, 4.167, 4.683], ...}
        """
        models_to_use = model_filter or sorted(self.results.get("1h", {}).keys())
        data = {}
        for model in models_to_use:
            vals = []
            for h in HORIZONS:
                mae = self.results.get(h, {}).get(model, {}).get("mae", 0.0)
                vals.append(round(mae, 3))
            data[model] = vals
        return data

    # ── Auto-Generated Insights ──────────────────────────────────────

    def generate_insights(self) -> dict[str, str]:
        """Generate insight text strings from actual data.

        Returns dict with keys: main, h1, h6, h24, kpi_subtitle
        """
        bests = self.get_best_models_all()
        b1 = bests["1h"]
        b6 = bests["6h"]
        b24 = bests["24h"]

        # Check if any model beats Persistence at 1h
        h1_beats_persistence = b1["mase"] < 1.0

        if h1_beats_persistence:
            h1_text = (
                f"<b>{b1['model']}</b> phá vỡ Autocorrelation Trap ở 1h: "
                f"<b>MASE={b1['mase']:.3f}</b> — model duy nhất thắng Persistence! ⭐"
            )
        else:
            h1_text = (
                f"Persistence vẫn thắng ở 1h (ACF≈0.97). "
                f"<b>{b1['model']}</b> gần nhất với MASE={b1['mase']:.3f}"
            )

        h6_text = (
            f"<b>{b6['model']}</b> giảm <b>{abs(b6['improvement_pct']):.1f}%</b> lỗi "
            f"so với Persistence tại 6h (MASE={b6['mase']:.3f}) ⭐"
        )

        h24_text = (
            f"<b>{b24['model']}</b> giảm <b>{abs(b24['improvement_pct']):.1f}%</b> lỗi "
            f"tại 24h (MASE={b24['mase']:.3f}) — long-range champion ⭐⭐"
        )

        main_insight = (
            f"{h1_text}<br>"
            f"• {h6_text}<br>"
            f"• {h24_text}<br><br>"
            f"Tổng cộng <b>{len(self.models)} mô hình</b> được đánh giá trên 3 horizons "
            f"với Persistence baseline thống nhất."
        )

        # KPI subtitle for sidebar
        kpi_subtitle = (
            f"6h: {b6['model']} ↓{abs(b6['improvement_pct']):.0f}% | "
            f"24h: {b24['model']} ↓{abs(b24['improvement_pct']):.0f}%"
        )

        return {
            "main": main_insight,
            "h1": h1_text,
            "h6": h6_text,
            "h24": h24_text,
            "kpi_subtitle": kpi_subtitle,
        }

    # ── KPI Cards Data ───────────────────────────────────────────────

    def get_kpi_data(self) -> dict:
        """Get all KPI card data in a single dict.

        Returns dict with keys: best_1h, best_6h, best_24h, n_models, version
        """
        bests = self.get_best_models_all()
        return {
            "best_1h": bests["1h"],
            "best_6h": bests["6h"],
            "best_24h": bests["24h"],
            "n_models": len(self.models),
            "version": self.version,
        }

    # ── Cross-Version Comparison ─────────────────────────────────────

    @staticmethod
    def compare_versions(snapshots: dict[str, dict]) -> pd.DataFrame:
        """Build a cross-version comparison DataFrame.

        Args:
            snapshots: Dict mapping version_name → normalized snapshot dict
                       (output of snapshot_adapter.load_all_normalized).

        Returns:
            DataFrame with columns: Version, 1h_MASE, 6h_MASE, 24h_MASE,
                                     Best_1h, Best_6h, Best_24h
        """
        rows = []
        for ver_name in sorted(snapshots.keys()):
            snap = snapshots[ver_name]
            rpt = ReportingEngine(snap)
            bests = rpt.get_best_models_all()
            row = {
                "Version": ver_name,
                "Description": snap.get("description", "")[:60],
                "Models": len(rpt.models),
            }
            for h in HORIZONS:
                b = bests[h]
                row[f"{h}_Best"] = b["model"]
                row[f"{h}_MASE"] = round(b["mase"], 3) if b["mase"] else None
                row[f"{h}_MAE"] = round(b["mae"], 2) if b["mae"] else None
            rows.append(row)

        return pd.DataFrame(rows)
