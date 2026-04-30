"""Unified Visualization Theme Framework (VTF) for PM2.5 Dashboard.

Provides centralized design tokens, color palettes, and theme helpers
for consistent styling across Matplotlib (static), Plotly (interactive),
and Streamlit HTML/CSS components.

Usage:
    from src.viz.theme import apply_mpl_theme, annotation_bbox, get_plotly_template
"""

from src.viz.theme import (
    PALETTE_CATEGORICAL,
    PALETTE_SEMANTIC,
    THEMES,
    TOKENS,
    annotation_bbox,
    apply_mpl_theme,
    get_plotly_template,
    get_theme,
)

__all__ = [
    "TOKENS",
    "PALETTE_CATEGORICAL",
    "PALETTE_SEMANTIC",
    "THEMES",
    "get_theme",
    "apply_mpl_theme",
    "annotation_bbox",
    "get_plotly_template",
]
