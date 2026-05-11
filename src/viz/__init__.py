"""Unified Visualization Theme Framework (VTF) for PM2.5 Dashboard.

Provides centralized design tokens, color palettes, chart factory,
and theme helpers for consistent styling across Plotly (interactive)
and Matplotlib (static) charts.

Quick start::

    from src.viz.chart_factory import chart, render_chart, styled_bar, styled_line
    from src.viz.palettes import get_trace_style, COLORSCALES

    fig = chart(title="My Chart", yaxis_title="MAE (µg/m³)")
    fig.add_trace(styled_bar(0, name="GRU", x=["1h"], y=[3.2], textfmt=".2f"))
    render_chart(fig, filename="my_chart")

Legacy API (still works but prefer chart_factory)::

    from src.viz.theme import apply_mpl_theme, get_plotly_template
"""

from src.viz.theme import (
    PALETTE_CATEGORICAL,
    PALETTE_SEMANTIC,
    THEMES,
    TOKENS,
    annotation_bbox,
    apply_mpl_theme,
    apply_plotly_style,
    get_plotly_annotation_style,
    get_plotly_config,
    get_plotly_template,
    get_theme,
)

__all__ = [
    # Design tokens & palettes
    "TOKENS",
    "PALETTE_CATEGORICAL",
    "PALETTE_SEMANTIC",
    "THEMES",
    "get_theme",
    # Matplotlib
    "apply_mpl_theme",
    "annotation_bbox",
    # Plotly (legacy — prefer chart_factory)
    "get_plotly_template",
    "get_plotly_config",
    "get_plotly_annotation_style",
    "apply_plotly_style",
]
