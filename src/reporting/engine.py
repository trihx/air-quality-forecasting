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

def get_model_type(name: str) -> str:
    """Resolve model type dynamically based on its name."""
    name_lower = name.lower()
    if "persistence" in name_lower or "naive" in name_lower: return "Baseline"
    if "arima" in name_lower: return "Statistical"
    if "ensemble" in name_lower or "stacking" in name_lower or "voting" in name_lower: return "Ensemble"
    if "tft" in name_lower: return "Transformer"
    if "gru" in name_lower or "lstm" in name_lower: return "Deep Learning"
    if "lightgbm" in name_lower or "randomforest" in name_lower or "gradientboosting" in name_lower or "elasticnet" in name_lower: return "ML"
    return MODEL_TYPES.get(name, "Unknown")


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

        # Find Persistence model MASE for improvement calculation
        persist_mase = 1.0
        for model_name, metrics in h_data.items():
            if "Persistence" in model_name:
                persist_mase = metrics.get("mase", 1.0)
                break

        # Find best non-baseline model by MASE
        best = None
        for model_name, metrics in h_data.items():
            if get_model_type(model_name) == "Baseline":
                continue
            mase = metrics.get("mase", float("inf"))
            if best is None or mase < best["mase"]:
                best = {
                    "model": model_name,
                    "mae": metrics.get("mae", float("inf")),
                    "mase": mase,
                    "type": get_model_type(model_name),
                }

        if best is None:
            return {"model": "N/A", "mae": 0.0, "mase": 0.0, "type": "?", "improvement_pct": 0.0, "persist_mase": 1.0}

        # Calculate improvement vs Baseline
        if best["mase"] != float("inf"):
            best["improvement_pct"] = ((persist_mase - best["mase"]) / persist_mase) * 100
        else:
            best["improvement_pct"] = 0.0

        best["persist_mase"] = persist_mase
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
                "Type": get_model_type(model),
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
        if df.empty or "Model" not in df.columns:
            return df

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

    # ── Representative Models (Top-5 per family) ────────────────────

    def get_representative_models(self) -> list[str]:
        """Get best model per family for clean Top-5 charts.

        Selects the model with the lowest average MASE across all horizons
        for each model type (Baseline, Statistical, ML, Deep Learning,
        Ensemble, Transformer). Always includes Persistence as baseline.

        Returns:
            Sorted list of 5-6 representative model names.
        """
        # Collect all models with their average MASE
        family_best: dict[str, tuple[str, float]] = {}
        all_models = set()
        for h in HORIZONS:
            all_models.update(self.results.get(h, {}).keys())

        for model in all_models:
            mtype = get_model_type(model)
            vals = []
            for h in HORIZONS:
                mase = self.results.get(h, {}).get(model, {}).get("mase")
                if mase is not None:
                    vals.append(mase)
            if not vals:
                continue
            avg_mase = sum(vals) / len(vals)

            if mtype not in family_best or avg_mase < family_best[mtype][1]:
                family_best[mtype] = (model, avg_mase)

        # Build list: always include best Baseline first
        reps = []
        for mtype in ("Baseline", "Statistical", "ML", "Deep Learning", "Transformer", "Ensemble"):
            if mtype in family_best:
                name = family_best[mtype][0]
                if name not in reps:
                    reps.append(name)

        return reps

    # ── MAE Ranking Table ─────────────────────────────────────────

    def get_mae_ranking_table(self, horizon: str, top_n: int = 10) -> pd.DataFrame:
        """Generate MAE ranking table for a specific horizon.

        Args:
            horizon: One of '1h', '6h', '24h'.
            top_n: Number of top models to include.

        Returns:
            DataFrame with columns: Rank, Model, Type, MAE (µg/m³),
            MASE, vs Persistence (%).
        """
        h_data = self.results.get(horizon, {})
        if not h_data:
            return pd.DataFrame()

        # Find Persistence MAE for comparison
        persist_mae = None
        for model_name, metrics in h_data.items():
            if "Persistence" in model_name:
                persist_mae = metrics.get("mae")
                break

        rows = []
        for model_name, metrics in h_data.items():
            mae = metrics.get("mae")
            mase = metrics.get("mase")
            if mae is None:
                continue

            vs_persist = None
            if persist_mae and persist_mae > 0:
                vs_persist = round(((mae - persist_mae) / persist_mae) * 100, 1)

            rmse = metrics.get("rmse")
            r2 = metrics.get("r2")
            da = metrics.get("da")
            bias = metrics.get("forecast_bias")

            rows.append({
                "Model": model_name,
                "Type": get_model_type(model_name),
                "MAE (µg/m³)": round(mae, 3),
                "RMSE (µg/m³)": round(rmse, 3) if rmse is not None and rmse != float("inf") else None,
                "MASE": round(mase, 3) if mase is not None else None,
                "R²": round(r2, 4) if r2 is not None else None,
                "DA (%)": round(da, 1) if da is not None else None,
                "vs Persistence (%)": vs_persist,
            })

        df = pd.DataFrame(rows).sort_values("MAE (µg/m³)").head(top_n)
        df.insert(0, "Rank", range(1, len(df) + 1))
        # Add medal emoji for top 3
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        df["Rank"] = df["Rank"].map(lambda x: f"{medals.get(x, '')} {x}".strip())
        return df.reset_index(drop=True)

    def get_models_ranked_by_mae(self, horizon: str) -> list[str]:
        """Get all model names sorted by MAE for a given horizon (best first).

        Useful for populating ranked multiselect dropdowns in the UI.

        Args:
            horizon: One of '1h', '6h', '24h'.

        Returns:
            List of model names sorted by MAE ascending.
        """
        h_data = self.results.get(horizon, {})
        if not h_data:
            return []

        ranked = sorted(
            h_data.items(),
            key=lambda item: item[1].get("mae", float("inf")),
        )
        return [name for name, _ in ranked]

    # ── MASE Ranking Table ────────────────────────────────────────

    def get_mase_ranking_table(self, horizon: str, top_n: int = 10) -> pd.DataFrame:
        """Generate MASE ranking table for a specific horizon.

        Args:
            horizon: One of '1h', '6h', '24h'.
            top_n: Number of top models to include.

        Returns:
            DataFrame with columns: Rank, Model, Type, MASE,
            MAE (µg/m³), vs Persistence (%).
        """
        h_data = self.results.get(horizon, {})
        if not h_data:
            return pd.DataFrame()

        # Find Persistence MASE for comparison
        persist_mase = 1.0
        for model_name, metrics in h_data.items():
            if "Persistence" in model_name:
                persist_mase = metrics.get("mase", 1.0)
                break

        rows = []
        for model_name, metrics in h_data.items():
            mase = metrics.get("mase")
            mae = metrics.get("mae")
            if mase is None:
                continue

            vs_persist = round(((mase - persist_mase) / persist_mase) * 100, 1) if persist_mase > 0 else None

            rmse = metrics.get("rmse")
            r2 = metrics.get("r2")
            da = metrics.get("da")

            rows.append({
                "Model": model_name,
                "Type": get_model_type(model_name),
                "MASE": round(mase, 3),
                "MAE (µg/m³)": round(mae, 3) if mae is not None else None,
                "RMSE (µg/m³)": round(rmse, 3) if rmse is not None and rmse != float("inf") else None,
                "R²": round(r2, 4) if r2 is not None else None,
                "DA (%)": round(da, 1) if da is not None else None,
                "vs Persistence (%)": vs_persist,
            })

        df = pd.DataFrame(rows).sort_values("MASE").head(top_n)
        df.insert(0, "Rank", range(1, len(df) + 1))
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        df["Rank"] = df["Rank"].map(lambda x: f"{medals.get(x, '')} {x}".strip())
        return df.reset_index(drop=True)

    def get_models_ranked_by_mase(self, horizon: str) -> list[str]:
        """Get all model names sorted by MASE for a given horizon (best first).

        Args:
            horizon: One of '1h', '6h', '24h'.

        Returns:
            List of model names sorted by MASE ascending.
        """
        h_data = self.results.get(horizon, {})
        if not h_data:
            return []

        ranked = sorted(
            h_data.items(),
            key=lambda item: item[1].get("mase", float("inf")),
        )
        return [name for name, _ in ranked]

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
        h1_beats_persistence = b1["mase"] < b1.get("persist_mase", 1.0)

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
