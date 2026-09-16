"""Shared UI components, helpers, and path constants for Streamlit Dashboard.

Design Direction: "Scientific Observatory"
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from src.viz.theme import (
    PALETTE_CATEGORICAL,
    PALETTE_SEMANTIC,
)

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
FIGURES_DIR = RESEARCH_DIR / "figures"
SHAP_DIR = FIGURES_DIR / "shap"
PI_DIR = FIGURES_DIR / "prediction_intervals"
EDA_DIR = RESEARCH_DIR / "eda" / "visualizations"

# ── Color Palette ──
COLORS = {
    "primary": PALETTE_SEMANTIC["primary"],
    "secondary": PALETTE_SEMANTIC["secondary"],
    "accent": PALETTE_SEMANTIC["accent"],
    "warning": PALETTE_SEMANTIC["warning"],
    "bg_dark": "#0E1117",
    "card_bg": "var(--secondary-background-color)",
    "text": "#FAFAFA",
    "text_muted": "#71717A",
    "success": PALETTE_SEMANTIC["success"],
    "danger": PALETTE_SEMANTIC["danger"],
}

CHART_COLORS = PALETTE_CATEGORICAL


# ── Caching & Data Loaders ──
@st.cache_data(ttl=3600)
def _count_tests() -> int:
    """Count total test functions in tests/ directory (cached 1h)."""
    import ast

    tests_dir = PROJECT_ROOT / "tests"
    count = 0
    for py_file in tests_dir.rglob("test_*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    count += 1
        except Exception:  # noqa: S112
            continue
    return count


@st.cache_data(ttl=3600)
def _get_pipeline_metrics() -> dict[str, Any]:
    """Compute pipeline metrics from actual data files — zero hardcode.

    Returns dict with keys:
        resolutions: {label: {"rows": int, "cols": int, "size_mb": float, "modified": str}}
        features_count: int (from 1h main dataset)
        total_rows: int (sum all resolutions)
    """
    from datetime import datetime

    processed = PROJECT_ROOT / "dataset" / "processed"
    datasets = [
        ("marts_features.csv", "1h"),
        ("marts_features_30m.csv", "30m"),
        ("marts_features_15m.csv", "15m"),
        ("marts_features_30m_base.csv", "30m_base"),
        ("marts_features_15m_base.csv", "15m_base"),
    ]

    resolutions: dict[str, dict[str, Any]] = {}
    features_count = 0

    for filename, label in datasets:
        path = processed / filename
        if not path.exists():
            continue
        try:
            stat = path.stat()
            size_mb = round(stat.st_size / (1024 * 1024), 1)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            with open(path, encoding="utf-8") as f:
                header = f.readline()
                cols = len(header.strip().split(","))
                rows = sum(1 for _ in f)

            resolutions[label] = {
                "rows": rows,
                "cols": cols,
                "size_mb": size_mb,
                "modified": modified,
                "filename": filename,
            }

            if label == "1h":
                features_count = cols
        except Exception:  # noqa: S112
            continue

    total_rows = sum(int(v["rows"]) for v in resolutions.values())

    return {
        "resolutions": resolutions,
        "features_count": features_count,
        "total_rows": total_rows,
    }


@st.cache_data
def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """Load JSON from path safely."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_experiment_results() -> dict[str, Any]:
    """Load all experiment metrics and configurations across modules."""
    results: dict[str, Any] = {}
    for name, subdir in [
        ("multi_horizon", "multi_horizon"),
        ("dl", "dl"),
        ("ensemble", "ensemble"),
        ("prediction_intervals", "prediction_intervals"),
    ]:
        d = RESEARCH_DIR / "experiments" / subdir
        if d.exists():
            jsons = sorted(d.glob("*.json"))
            if jsons:
                results[name] = load_json(jsons[-1])

    cfg = RESEARCH_DIR / "best_models_configs.json"
    if cfg.exists():
        results["configs"] = load_json(cfg)

    shap_json = SHAP_DIR / "shap_results.json"
    if shap_json.exists():
        results["shap"] = load_json(shap_json)

    aci_json = RESEARCH_DIR / "experiments" / "v8_final" / "aci_results.json"
    if aci_json.exists():
        results["aci_intervals"] = load_json(aci_json)

    return results


# ── Visual UI Elements ──
def kpi_card(
    label: str,
    value: Any,
    delta: str | None = None,
    delta_class: str = "positive",
) -> str:
    """Render a styled KPI card HTML block."""
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""<div class="kpi-card">
<div class="kpi-label">{label}</div>
<div class="kpi-value">{value}</div>
{delta_html}
</div>"""


def section_header(icon: str, title: str) -> None:
    """Render a section header with icon and subtle styling."""
    st.markdown(
        f"""
    <div class="section-header">
        <span class="icon">{icon}</span>
        <span class="title">{title}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def insight_card(title: str, text: str, card_type: str = "default") -> None:
    """Render an insight callout card with markdown parsing."""
    cls = "warning" if card_type == "warning" else ""
    parsed_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    parsed_text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", parsed_text)
    st.markdown(
        f"""
    <div class="insight-card {cls}">
        <h4>{title}</h4>
        <div class="insight-text">{parsed_text}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
