"""Unified Visualization Theme — Single Source of Truth.

Centralized design tokens, color palettes, and helpers for Matplotlib,
Plotly, and Streamlit HTML/CSS theming.

References:
    - Shen (2023) "Statistics and Data Visualization in Climate Science"
    - Cole Nussbaumer Knaflic "Storytelling with Data"
    - SciencePlots (github.com/garrettj403/SciencePlots)
    - Matplotlib Custom Style Sheets (matplotlib.org)
    - WCAG 2.1 AA Contrast Guidelines

Author: trihx
"""

from __future__ import annotations

from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Design Tokens (theme-neutral)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOKENS = {
    # Font
    "font_family": "Inter, Arial, sans-serif",
    "font_size_base": 10,
    "font_size_title": 13,
    "font_size_label": 11,
    "font_size_tick": 9,
    "font_size_annotation": 9,
    "font_size_legend": 9,
    # Lines
    "line_width_primary": 1.5,
    "line_width_secondary": 0.8,
    "line_width_thin": 0.5,
    # Grid
    "grid_alpha": 0.25,
    # DPI
    "dpi_screen": 150,
    "dpi_print": 300,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Color Palettes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Categorical: 8-color accessible cycle (colorblind-friendly)
PALETTE_CATEGORICAL = [
    "#00D4AA",  # Teal (primary)
    "#FF6B6B",  # Coral
    "#4ECDC4",  # Sea green
    "#FFE66D",  # Yellow
    "#A78BFA",  # Purple
    "#FB923C",  # Orange
    "#60A5FA",  # Blue
    "#F472B6",  # Pink
]

# Semantic: role-based colors
PALETTE_SEMANTIC = {
    "primary": "#00D4AA",
    "secondary": "#4ECDC4",
    "accent": "#FF6B6B",
    "warning": "#FFE66D",
    "success": "#00D4AA",
    "danger": "#FF6B6B",
    "info": "#60A5FA",
}

# EDA accent colors (for annotation markers / vlines)
ACCENT_COLORS = {
    "blue": "#4A9EF5",
    "orange": "#F59E4A",
    "green": "#4AF5A3",
    "red": "#F54A4A",
    "purple": "#A34AF5",
}

# Domain-specific (sensor variables)
PALETTE_DOMAIN = {
    "pm25": "#e74c3c",
    "nhiet_do": "#e67e22",
    "do_am": "#3498db",
    "diem_suong": "#2ecc71",
    "co2": "#9b59b6",
    "who_line": "#c0392b",
    "vn_line": "#f39c12",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Theme Modes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THEMES = {
    "dark": {
        "figure_facecolor": "#0E1117",
        "axes_facecolor": "#1A1D23",
        "text_color": "#EAEAEA",
        "text_muted": "#71717A",
        "grid_color": "#2D3139",
        "spine_color": "#2D3139",
        "annotation_bg": "#1A1D23",
        "annotation_edge": "none",
    },
    "light": {
        "figure_facecolor": "#FFFFFF",
        "axes_facecolor": "#FAFAFA",
        "text_color": "#373737",
        "text_muted": "#6B7280",
        "grid_color": "#E5E7EB",
        "spine_color": "#D1D5DB",
        "annotation_bg": "#FFFFFF",
        "annotation_edge": "#D1D5DB",
    },
}

# Style files directory
_STYLES_DIR = Path(__file__).parent / "styles"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_theme(mode: str = "light") -> dict:
    """Get theme dict by mode name.

    Args:
        mode: 'light' or 'dark'.

    Returns:
        Dictionary with theme color values.
    """
    return THEMES.get(mode, THEMES["light"])


def apply_mpl_theme(mode: str = "light") -> None:
    """Apply matplotlib rcParams for the given mode.

    Sets figure/axes colors, font sizes, grid, and spines
    from the centralized design tokens + theme colors.

    Args:
        mode: 'light' or 'dark'.
    """
    import matplotlib.pyplot as plt

    theme = get_theme(mode)
    plt.rcParams.update(
        {
            # Colors
            "figure.facecolor": theme["figure_facecolor"],
            "axes.facecolor": theme["axes_facecolor"],
            "axes.edgecolor": theme["spine_color"],
            "axes.labelcolor": theme["text_color"],
            "text.color": theme["text_color"],
            "xtick.color": theme["text_muted"],
            "ytick.color": theme["text_muted"],
            "grid.color": theme["grid_color"],
            "grid.alpha": TOKENS["grid_alpha"],
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
            "font.size": TOKENS["font_size_base"],
            "axes.titlesize": TOKENS["font_size_title"],
            "axes.labelsize": TOKENS["font_size_label"],
            "xtick.labelsize": TOKENS["font_size_tick"],
            "ytick.labelsize": TOKENS["font_size_tick"],
            "legend.fontsize": TOKENS["font_size_legend"],
            # Lines
            "lines.linewidth": TOKENS["line_width_primary"],
            "lines.markersize": 5,
            # Grid & Spines
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # DPI & Layout
            "figure.dpi": TOKENS["dpi_screen"],
            # Color cycle
            "axes.prop_cycle": plt.cycler(color=PALETTE_CATEGORICAL),
        }
    )


def annotation_bbox(mode: str = "light") -> dict:
    """Get standard annotation bbox props for text boxes.

    Returns a dict suitable for matplotlib's `bbox` parameter
    in `ax.text()` and `ax.annotate()` calls.

    Args:
        mode: 'light' or 'dark'.

    Returns:
        Dict with boxstyle, facecolor, edgecolor, alpha.
    """
    theme = get_theme(mode)
    return {
        "boxstyle": "round,pad=0.3",
        "facecolor": theme["annotation_bg"],
        "edgecolor": theme["annotation_edge"],
        "alpha": 0.85,
    }


def get_plotly_annotation_style(mode: str = "light", overrides: dict | None = None) -> dict:
    """Get standard Plotly annotation style dict for consistent chart labels.

    Designed to support high contrast (Grayscale printing) by using strong
    white backgrounds and dark text to avoid gridline overlap.

    Args:
        mode: 'light' or 'dark' (defaults to light for best print contrast).
        overrides: Optional dict to override default values.

    Returns:
        Dict suitable for unpacking into fig.add_annotation(**kwargs).
    """
    theme = get_theme(mode)
    base_style = {
        "showarrow": False,
        "font": dict(
            size=TOKENS["font_size_annotation"] + 1,  # +1 for slightly better readability
            color="#111111" if mode == "light" else theme["text_color"],
        ),
        "bgcolor": "rgba(255, 255, 255, 0.9)" if mode == "light" else "rgba(26, 29, 35, 0.9)",
        "bordercolor": "rgba(0, 0, 0, 0.2)" if mode == "light" else "rgba(255, 255, 255, 0.2)",
        "borderwidth": 1,
        "borderpad": 3,
    }

    if overrides:
        # Deep update for nested dicts like font
        for k, v in overrides.items():
            if k == "font" and isinstance(v, dict):
                base_style["font"].update(v)
            else:
                base_style[k] = v

    return base_style


# ══════════════════════════════════════════════════════════════════════
# DEPRECATED — Legacy helpers kept for backward compatibility.
# New code MUST use `src.viz.chart_factory` instead.
# ══════════════════════════════════════════════════════════════════════


def get_plotly_config(filename: str = "chart", scale: int = 3) -> dict:
    """Get standardized Plotly configuration for the dashboard.

    .. deprecated::
        Use ``src.viz.chart_factory.render_chart()`` which embeds config automatically.
    """
    return {"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": filename, "scale": scale}}


def get_plotly_template(mode: str = "light") -> dict:
    """Get Plotly layout template for the given mode.

    .. deprecated::
        Use ``src.viz.chart_factory.chart()`` which applies template automatically.
    """
    theme = get_theme(mode)
    return {
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {
                "color": theme["text_color"],
                "family": TOKENS["font_family"],
                "size": TOKENS["font_size_base"],
            },
            "xaxis": {
                "gridcolor": theme["grid_color"],
                "zerolinecolor": theme["grid_color"],
            },
            "yaxis": {
                "gridcolor": theme["grid_color"],
                "zerolinecolor": theme["grid_color"],
            },
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "orientation": "h",
                "yanchor": "top",
                "y": -0.2,
                "xanchor": "center",
                "x": 0.5,
            },
            "hovermode": "x unified",
        }
    }


def detect_streamlit_mode() -> str:
    """Detect the current Streamlit theme (light/dark).

    Uses st.get_option to check the configured theme base.
    Falls back to 'dark' if detection fails.

    Returns:
        'light' or 'dark'.
    """
    try:
        import streamlit as st

        base = st.get_option("theme.base")
        if base == "light":
            return "light"
    except Exception:
        pass
    return "dark"


def apply_plotly_style(fig, height=450):
    """Apply standard Plotly template to a figure based on current Streamlit mode.

    .. deprecated::
        Use ``src.viz.chart_factory.chart()`` + ``render_chart()`` instead.
    """
    mode = detect_streamlit_mode()
    _template = get_plotly_template(mode)
    fig.update_layout(
        **_template["layout"],
        margin=dict(l=20, r=20, t=50, b=80),
        height=height,
    )
    # Force text annotations on traces to use the main text color instead of inheriting trace color
    theme = get_theme(mode)
    # fig.update_traces(textfont_color="var(--text-color)")
    fig.update_traces(outsidetextfont_color="#71717A", insidetextfont_color="#71717A", selector=dict(type="bar"))
    fig.update_traces(outsidetextfont_color="#71717A", insidetextfont_color="#71717A", selector=dict(type="pie"))
    return fig
