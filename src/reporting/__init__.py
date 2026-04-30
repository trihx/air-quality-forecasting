"""Reporting Framework — Unified Data + Visualization for Dashboard.

Provides:
- ReportingEngine: parameterized metrics, rankings, insights from snapshot JSON.
- charts: Plotly chart factory using VTF for consistent styling.

Usage:
    from src.reporting import ReportingEngine
    from src.reporting import charts

    rpt = ReportingEngine(snapshot_data)
    fig = charts.plot_mase_comparison(rpt)
"""

from src.reporting.engine import MODEL_TYPES, ReportingEngine

__all__ = [
    "ReportingEngine",
    "MODEL_TYPES",
]
