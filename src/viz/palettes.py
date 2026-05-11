"""Grayscale-safe Palette System for Scientific Publications.

Provides marker shapes, dash patterns, and hatch fills that remain
distinguishable when printed in black-and-white.  Works alongside the
existing ``PALETTE_CATEGORICAL`` colors defined in ``theme.py``.

Design principles:
    * Each trace index gets a unique ``(color, marker, dash)`` triple.
    * Heatmap colorscales are tested for monotonic luminance so that
      cells stay readable in grayscale.
    * All constants live here — ``chart_factory.py`` reads them.

References:
    * Crameri et al. (2020) "The misuse of colour in science communication"
    * WCAG 2.1 AA contrast guidelines
"""

from __future__ import annotations

from src.viz.theme import PALETTE_CATEGORICAL

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Grayscale-safe Marker/Dash combos (one per trace index)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GRAYSCALE_MARKERS: list[dict[str, str]] = [
    {"symbol": "circle",       "dash": "solid",       "hatch": ""},
    {"symbol": "square",       "dash": "dash",        "hatch": "/"},
    {"symbol": "diamond",      "dash": "dot",         "hatch": "\\"},
    {"symbol": "triangle-up",  "dash": "dashdot",     "hatch": "x"},
    {"symbol": "cross",        "dash": "longdash",    "hatch": "+"},
    {"symbol": "star",         "dash": "longdashdot", "hatch": "."},
    {"symbol": "hexagon",      "dash": "solid",       "hatch": "|"},
    {"symbol": "pentagon",     "dash": "dash",        "hatch": "-"},
]


def get_trace_style(index: int) -> dict:
    """Return a ``(color, marker_symbol, dash)`` dict for trace *index*.

    Cycles through both the categorical palette and the marker set so
    that no two traces look alike even in grayscale.

    Args:
        index: Zero-based trace index.

    Returns:
        Dict with keys ``color``, ``symbol``, ``dash``.
    """
    palette = PALETTE_CATEGORICAL
    markers = GRAYSCALE_MARKERS
    return {
        "color": palette[index % len(palette)],
        "symbol": markers[index % len(markers)]["symbol"],
        "dash": markers[index % len(markers)]["dash"],
        "hatch": markers[index % len(markers)]["hatch"],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Heatmap Colorscales (grayscale-safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Viridis: monotonic luminance → excellent grayscale contrast
# RdBu_r : symmetric diverging, prints as dark→light→dark
# PM2.5 AQI: custom domain scale with good luminance spread
COLORSCALES: dict[str, str | list] = {
    "sequential": "Viridis",
    "diverging": "RdBu_r",
    "pm25_aqi": [
        [0.00, "#2E7D32"],   # Good         (0-12 µg/m³)  — dark green
        [0.24, "#66BB6A"],   # Moderate low  (12-25)       — lighter green
        [0.50, "#FDD835"],   # Moderate high (25-35)       — yellow
        [0.75, "#FB8C00"],   # Unhealthy     (35-55)       — orange
        [1.00, "#C62828"],   # Hazardous     (>55)         — dark red
    ],
    "correlation": "RdBu_r",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model-family color map (domain-specific)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_FAMILY_COLORS: dict[str, str] = {
    "Baseline":       "#FF6B6B",   # Coral red
    "Statistical":    "#60A5FA",   # Blue
    "ML":             "#00D4AA",   # Teal
    "Deep Learning":  "#A78BFA",   # Purple
    "Transformer":    "#F472B6",   # Pink
    "Ensemble":       "#FFE66D",   # Yellow/Gold
}
