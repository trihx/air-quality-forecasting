"""Plotly Chart Factory — Single Source of Truth for all dashboard charts.

Every Plotly figure on the PM2.5 dashboard **must** be created through
this module.  It enforces:

* Consistent fonts (Inter for web, Times New Roman for print export)
* Grayscale-safe trace styling (marker shapes + dash patterns)
* IEEE-standard legend placement (bottom-center, horizontal)
* Unified hover templates with Vietnamese labels
* High-DPI PNG download via modebar (300 DPI, scale=3)

Usage::

    from src.viz.chart_factory import chart, render_chart, add_baseline

    fig = chart(
        title="MASE Comparison",
        xaxis_title="Forecast Horizon",
        yaxis_title="MASE",
        height=500,
    )
    fig.add_trace(go.Bar(name="GRU", x=["1h","6h","24h"], y=[0.9,0.5,0.6]))
    add_baseline(fig, y=1.0, label="Persistence (MASE=1.0)")
    render_chart(fig, filename="mase_comparison")

Author: trihx
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.viz.palettes import get_trace_style
from src.viz.theme import (
    PALETTE_SEMANTIC,
    TOKENS,
    detect_streamlit_mode,
    get_plotly_annotation_style,
    get_theme,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_FONT_WEB = "Inter, Arial, sans-serif"
_FONT_PRINT = "Times New Roman, serif"

# Modebar buttons to show (keep it clean)
_MODEBAR_BUTTONS = [
    "toImage",
    "zoom2d",
    "pan2d",
    "resetScale2d",
    "zoomIn2d",
    "zoomOut2d",
]

# Standard margins (pixels) — generous enough for Vietnamese axis labels
_MARGINS = dict(l=60, r=30, t=60, b=80)

# Muted text color for annotations that work in both light & dark
_MUTED_TEXT = "#71717A"  # Zinc 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def chart(
    *,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 450,
    mode: str | None = None,
    showlegend: bool = True,
    barmode: str | None = None,
    hovermode: str = "x unified",
    margin: dict | None = None,
    layout_overrides: dict | None = None,
) -> go.Figure:
    """Create a pre-styled Plotly Figure.

    This is the **only** sanctioned way to create charts on the dashboard.
    It auto-detects the Streamlit theme and applies consistent styling.

    Per QĐ 1799 Phụ lục 8: "không đặt tựa cho hình vì đã có tên hình
    đầy đủ nghĩa ngay phía dưới hình". The ``title`` parameter is
    **ignored** — use ``figure_caption()`` after ``render_chart()`` instead.

    Args:
        title: **Deprecated/ignored.** Kept for backward compatibility.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        height: Figure height in pixels.
        mode: Force ``'light'`` or ``'dark'``.  ``None`` = auto-detect.
        showlegend: Whether to show the legend.
        barmode: ``'group'``, ``'stack'``, ``'overlay'``, or ``None``.
        hovermode: Plotly hover mode string.
        margin: Override default margins ``dict(l,r,t,b)``.
        layout_overrides: Arbitrary extra ``fig.update_layout(**kwargs)``.

    Returns:
        A styled ``go.Figure`` ready for ``add_trace()`` calls.
    """
    if mode is None:
        mode = detect_streamlit_mode()

    theme = get_theme(mode)

    fig = go.Figure()

    layout_kwargs: dict[str, Any] = {
        # Background
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(
            family=_FONT_WEB,
            size=TOKENS["font_size_base"],
        ),
        # Title — suppressed per QĐ 1799 Phụ lục 8
        # Caption goes below chart via figure_caption().
        # Axes
        "xaxis": dict(
            title=dict(text=xaxis_title, font=dict(size=TOKENS["font_size_label"])),
            gridcolor=theme["grid_color"],
            zerolinecolor=theme["grid_color"],
            tickfont=dict(size=TOKENS["font_size_tick"]),
            showgrid=True,
        ),
        "yaxis": dict(
            title=dict(text=yaxis_title, font=dict(size=TOKENS["font_size_label"])),
            gridcolor=theme["grid_color"],
            zerolinecolor=theme["grid_color"],
            tickfont=dict(size=TOKENS["font_size_tick"]),
            showgrid=True,
        ),
        # Legend (IEEE standard: bottom-center, horizontal)
        "legend": dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=TOKENS["font_size_legend"]),
        ),
        "showlegend": showlegend,
        # Hover
        "hovermode": hovermode,
        # Size
        "height": height,
        "margin": margin or _MARGINS,
    }

    if barmode:
        layout_kwargs["barmode"] = barmode

    # Remove None values
    layout_kwargs = {k: v for k, v in layout_kwargs.items() if v is not None}

    fig.update_layout(**layout_kwargs)

    if layout_overrides:
        fig.update_layout(**layout_overrides)

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Render helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_chart(
    fig: go.Figure,
    *,
    filename: str = "chart",
    scale: int = 3,
    key: str | None = None,
) -> None:
    """Render a Plotly figure into Streamlit with standardized config.

    This wraps ``st.plotly_chart()`` with:
    * High-DPI PNG export (scale=3 → 300 DPI equivalent)
    * Clean modebar (only useful buttons)
    * Full container width

    Args:
        fig: The Plotly figure to render.
        filename: Default filename for PNG download.
        scale: Resolution multiplier for exported PNG.
        key: Optional Streamlit widget key for deduplication.
    """
    config = {
        "displayModeBar": True,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "autoScale2d",
        ],
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "scale": scale,
        },
        "displaylogo": False,
    }

    # Reduce top margin since title is no longer rendered inside
    try:
        current_margin = fig.layout.margin
        t_margin = getattr(current_margin, "t", 60)
        if isinstance(t_margin, tuple) and len(t_margin) > 0:
            t_margin = t_margin[0]
        if isinstance(t_margin, (int, float)) and t_margin >= 50:
            fig.update_layout(margin_t=20)
    except Exception:
        pass

    # ── Print Mode: auto-convert to B&W if toggled in sidebar ──
    try:
        if st.session_state.get("print_mode", False):
            fig = to_bw(fig)
    except Exception:
        pass  # Graceful fallback if to_bw fails

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=config,
        key=key,
    )


def figure_caption(text: str) -> None:
    """Render a figure caption below a chart.

    Per QĐ 1799 Phụ lục 8: "đã có tên hình đầy đủ nghĩa ngay phía
    dưới hình".  Call this **immediately after** ``render_chart()``.

    Args:
        text: Caption text (e.g. "Phân bổ khoảng gaps theo kích thước").
    """
    st.markdown(
        f"<div style='text-align:center; margin-top:-10px; margin-bottom:18px;'>"
        f"<span style='font-size:13px; color:#374151;'>{text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def figure_caption_numbered(
    chapter: int,
    number: int,
    text: str,
) -> None:
    """Render a numbered figure caption per QĐ 1799 format.

    Format: "Hình {chapter}.{number}. {text}"
    Example: "Hình 4.1. Bootstrap 95% CI cho MASE"

    Args:
        chapter: Thesis chapter number (e.g. 4 for Results).
        number: Figure number within the chapter.
        text: Caption description.
    """
    st.markdown(
        f"<div style='text-align:center; margin-top:-10px; margin-bottom:18px;'>"
        f"<span style='font-size:13px; color:#374151;'>"
        f"<b>Hình {chapter}.{number}.</b> {text}"
        f"</span></div>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trace helpers (grayscale-safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def styled_line(
    index: int,
    *,
    name: str,
    x: Any,
    y: Any,
    width: float = 2.0,
    mode: str = "lines+markers",
    marker_size: int = 7,
    text: list | None = None,
    hovertemplate: str | None = None,
    **kwargs: Any,
) -> go.Scatter:
    """Create a ``go.Scatter`` trace with grayscale-safe styling.

    Automatically assigns color + marker symbol + dash pattern from the
    palette index so that traces remain distinguishable in B&W print.

    Args:
        index: Zero-based index into the palette.
        name: Legend label.
        x, y: Data arrays.
        width: Line width.
        mode: Plotly scatter mode.
        marker_size: Marker diameter.
        text: Optional text labels per point.
        hovertemplate: Custom hover template.
        **kwargs: Passed through to ``go.Scatter()``.

    Returns:
        A styled ``go.Scatter`` trace.
    """
    style = get_trace_style(index)
    trace_kwargs: dict[str, Any] = {
        "name": name,
        "x": x,
        "y": y,
        "mode": mode,
        "line": dict(color=style["color"], width=width, dash=style["dash"]),
        "marker": dict(
            color=style["color"],
            size=marker_size,
            symbol=style["symbol"],
            line=dict(width=1, color="white"),
        ),
    }
    if text is not None:
        trace_kwargs["text"] = text
        trace_kwargs["textposition"] = "top center"
    if hovertemplate:
        trace_kwargs["hovertemplate"] = hovertemplate
    trace_kwargs.update(kwargs)
    return go.Scatter(**trace_kwargs)


def styled_bar(
    index: int,
    *,
    name: str,
    x: Any,
    y: Any,
    text: list | None = None,
    textposition: str = "outside",
    textfmt: str = "",
    orientation: str = "v",
    color: str | None = None,
    pattern: bool = True,
    **kwargs: Any,
) -> go.Bar:
    """Create a ``go.Bar`` trace with consistent styling.

    Automatically applies a grayscale-safe hatch pattern from the
    palette so bars remain distinguishable in B&W print (QĐ 1799
    Phụ lục 8: "texture/pattern của cột khác nhau để dễ phân biệt
    nhất là khi in trắng đen").

    Args:
        index: Zero-based palette index.
        name: Legend label.
        x, y: Data arrays.
        text: Value labels (auto-formatted if *textfmt* given).
        textposition: ``'outside'``, ``'inside'``, ``'none'``.
        textfmt: Python format spec, e.g. ``'.3f'``.
        orientation: ``'v'`` (vertical) or ``'h'`` (horizontal).
        color: Override palette color.
        pattern: Whether to apply grayscale hatch pattern.
        **kwargs: Passed through to ``go.Bar()``.

    Returns:
        A styled ``go.Bar`` trace.
    """
    style = get_trace_style(index)
    bar_color = color or style["color"]

    if text is None and textfmt:
        text = [f"{v:{textfmt}}" for v in (y if orientation == "v" else x)]

    marker_kwargs: dict[str, Any] = {"color": bar_color}
    if pattern and style.get("hatch"):
        marker_kwargs["pattern"] = {"shape": style["hatch"], "solidity": 0.6}

    trace_kwargs: dict[str, Any] = {
        "name": name,
        "x": x,
        "y": y,
        "marker": marker_kwargs,
        "orientation": orientation,
        "textposition": textposition,
        "textfont": {"size": 10, "color": _MUTED_TEXT},
    }
    if text is not None:
        trace_kwargs["text"] = text
    trace_kwargs.update(kwargs)
    return go.Bar(**trace_kwargs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Annotation helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def add_baseline(
    fig: go.Figure,
    *,
    y: float = 1.0,
    label: str = "Persistence Baseline (MASE = 1.0)",
    color: str | None = None,
    line_width: float = 2.0,
    line_dash: str = "dash",
) -> None:
    """Add a horizontal reference line with label.

    Common use: MASE = 1.0 baseline, WHO guideline = 25 µg/m³, etc.

    Args:
        fig: Target figure.
        y: Y-axis value for the line.
        label: Annotation text.
        color: Line color.  Defaults to ``PALETTE_SEMANTIC['accent']``.
        line_width: Line width.
        line_dash: Dash pattern.
    """
    _color = color or PALETTE_SEMANTIC["accent"]
    fig.add_hline(
        y=y,
        line_dash=line_dash,
        line_color=_color,
        line_width=line_width,
        annotation_text=f"<b>{label}</b>",
        annotation_font_size=11,
        annotation_font_color=_color,
        annotation_bgcolor="rgba(255,255,255,0.85)",
        layer="below",
    )


def add_value_annotations(
    fig: go.Figure,
    *,
    models: list[str],
    data: dict[str, list[float]],
    categories: list,
    fmt: str = ".3f",
    yshift: int = 12,
) -> None:
    """Add white-background value labels above grouped bars.

    Uses numerical x-axis offsets so annotations align perfectly
    over bars regardless of container width.

    Args:
        fig: Target figure.
        models: Ordered list of model names (bar groups).
        data: ``{model_name: [val_per_category]}``.
        categories: Category labels (horizons, etc.).
        fmt: Number format spec.
        yshift: Pixels to shift label upward.
    """
    n = len(models)
    if n == 0:
        return
    bar_width = 0.8 / n
    offsets = [(i - (n - 1) / 2) * bar_width for i in range(n)]

    mode = detect_streamlit_mode()
    annot_style = get_plotly_annotation_style(mode, overrides={"yshift": yshift})

    for mi, model in enumerate(models):
        vals = data[model]
        for ci in range(len(categories)):
            if ci < len(vals):
                fig.add_annotation(
                    x=ci + offsets[mi],
                    y=vals[ci],
                    text=f"<b>{vals[ci]:{fmt}}</b>",
                    **annot_style,
                )


def add_simple_bar_labels(
    fig: go.Figure,
    *,
    orientation: str = "v",
    fmt: str = "",
    yshift: int = 5,
    xshift: int = 5,
) -> None:
    """Add styled white-box annotations to all bar traces in a figure.

    This replaces native Plotly text labels with high-contrast boxes.
    It reads the existing `x` and `y` from the bar traces. If `text` is
    present on the trace, it uses that. Otherwise, it formats the value
    (y for vertical, x for horizontal) using `fmt`.

    Args:
        fig: Target figure.
        orientation: ``'v'`` or ``'h'``.
        fmt: Format string if trace.text is missing (e.g. ``'.2f'``).
        yshift: Pixels to shift up (for 'v').
        xshift: Pixels to shift right (for 'h').
    """
    mode = detect_streamlit_mode()

    if orientation == "v":
        annot_style = get_plotly_annotation_style(mode, overrides={"yanchor": "bottom", "yshift": yshift})
    else:
        annot_style = get_plotly_annotation_style(mode, overrides={"xanchor": "left", "xshift": xshift})

    is_grouped = fig.layout.barmode == "group"
    bar_traces = [t for t in fig.data if getattr(t, "type", "") == "bar" or isinstance(t, go.Bar)]
    n_traces = len(bar_traces)

    for trace_idx, trace in enumerate(bar_traces):
        x_vals = trace.x
        y_vals = trace.y
        texts = trace.text

        if x_vals is None or y_vals is None:
            continue

        offset = 0
        if is_grouped and n_traces > 1:
            bar_width = 0.8 / n_traces
            offset = (trace_idx - (n_traces - 1) / 2) * bar_width

        for i, (x_raw, y_raw) in enumerate(zip(x_vals, y_vals)):
            # Determine text to display
            if hasattr(texts, "__iter__") and not isinstance(texts, str) and i < len(texts):
                label = texts[i]
                if isinstance(label, float):
                    label = f"{label:{fmt}}" if fmt else str(label)
            else:
                val = y_raw if orientation == "v" else x_raw
                label = f"{val:{fmt}}" if fmt else str(val)

            if label is None or str(label).strip() in ("nan", "", "None"):
                continue

            x_annot = (i + offset) if (orientation == "v" and is_grouped) else x_raw
            y_annot = (i + offset) if (orientation == "h" and is_grouped) else y_raw

            fig.add_annotation(x=x_annot, y=y_annot, text=f"<b>{label}</b>", **annot_style)

        # Clear native text to prevent double rendering
        trace.text = None
        trace.texttemplate = None


def add_rangeslider(fig: go.Figure) -> None:
    """Add an interactive range slider to the x-axis.

    Best for time series charts with datetime x-axis.
    """
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider_thickness=0.06,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B&W Print Export (Dual-mode: Color dashboard + B&W print)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Grayscale palette — high contrast shades for B&W printing
_BW_GRAYS = [
    "#1A1A1A",  # Near-black
    "#666666",  # Dark gray
    "#999999",  # Medium gray
    "#CCCCCC",  # Light gray
    "#4D4D4D",  # Charcoal
    "#B3B3B3",  # Silver
    "#333333",  # Dark
    "#808080",  # Mid
]

_BW_HATCHES = ["", "/", "\\", "x", "+", ".", "|", "-"]
_BW_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
_BW_MARKERS = ["circle", "square", "diamond", "triangle-up", "cross", "star"]


def to_bw(fig: go.Figure) -> go.Figure:
    """Convert a Plotly figure to grayscale-safe version for thesis printing.

    Transforms:
    - Background → white
    - Font → Times New Roman (print standard)
    - Bar traces → grayscale fills + hatch patterns
    - Line traces → distinct dash patterns + marker symbols
    - Annotations → black text, placed outside chart elements
    - Grid → light gray, subtle

    The original figure is NOT modified; a deep copy is returned.

    Args:
        fig: Source Plotly figure (color version).

    Returns:
        New ``go.Figure`` optimized for B&W printing.
    """
    import copy

    bw = copy.deepcopy(fig)

    # ── Layout: white background, black text, Times New Roman ──
    bw.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family=_FONT_PRINT,
            color="black",
            size=TOKENS["font_size_base"],
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.12)",
            zerolinecolor="rgba(0,0,0,0.25)",
            linecolor="black",
            linewidth=1,
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black")),
            showline=True,
            mirror=True,
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.12)",
            zerolinecolor="rgba(0,0,0,0.25)",
            linecolor="black",
            linewidth=1,
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black")),
            showline=True,
            mirror=True,
        ),
        legend=dict(
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(color="black"),
        ),
    )

    # ── Traces: apply grayscale styling ──
    bar_idx = 0
    line_idx = 0

    for trace in bw.data:
        trace_type = getattr(trace, "type", "")

        if trace_type == "bar":
            gray = _BW_GRAYS[bar_idx % len(_BW_GRAYS)]
            hatch = _BW_HATCHES[bar_idx % len(_BW_HATCHES)]

            trace.marker.color = gray
            trace.marker.line = dict(color="black", width=1)

            if hatch:
                trace.marker.pattern = dict(
                    shape=hatch,
                    bgcolor="white",
                    fgcolor="black",
                    size=6,
                    solidity=0.5,
                )
            bar_idx += 1

        elif trace_type in ("scatter", "scattergl"):
            gray = _BW_GRAYS[line_idx % len(_BW_GRAYS)]
            dash = _BW_DASHES[line_idx % len(_BW_DASHES)]
            symbol = _BW_MARKERS[line_idx % len(_BW_MARKERS)]

            if hasattr(trace, "line") and trace.line is not None:
                trace.line.color = gray
                trace.line.dash = dash
                trace.line.width = 2

            if hasattr(trace, "marker") and trace.marker is not None:
                trace.marker.color = gray
                trace.marker.symbol = symbol
                trace.marker.size = 7
                trace.marker.line = dict(color="black", width=1)

            # Show markers for line charts in B&W
            if trace.mode in (None, "lines"):
                trace.mode = "lines+markers"

            line_idx += 1

        elif trace_type == "heatmap":
            trace.colorscale = "Greys"
            if hasattr(trace, "colorbar") and trace.colorbar is not None:
                trace.colorbar.tickfont = dict(color="black")

    # ── Annotations: black text on white background box, outside elements ──
    if bw.layout.annotations:
        new_annotations = []
        for ann in bw.layout.annotations:
            ann_dict = ann.to_plotly_json()
            font_size = ann_dict.get("font", {}).get("size", TOKENS["font_size_annotation"])
            ann_dict["font"] = dict(
                color="black",
                family=_FONT_PRINT,
                size=font_size,
            )
            # White background box behind text → readable over any element
            ann_dict["bgcolor"] = "rgba(255,255,255,0.92)"
            ann_dict["bordercolor"] = "rgba(0,0,0,0.3)"
            ann_dict["borderpad"] = 3
            ann_dict["borderwidth"] = 1
            # Push text outside bars to avoid overlay
            if ann_dict.get("yshift", 0) < 14:
                ann_dict["yshift"] = 16
            new_annotations.append(ann_dict)
        bw.layout.annotations = new_annotations

    return bw


def render_bw_download(
    fig: go.Figure,
    *,
    filename: str = "thesis_figure",
    label: str = "🖨️ Tải bản in (B&W)",
    width: int = 1200,
    height: int = 500,
) -> None:
    """Render a download button for B&W print-optimized PNG.

    Converts the figure to grayscale via ``to_bw()``, exports as
    high-DPI PNG (300 DPI equivalent), and provides a Streamlit
    download button.

    Args:
        fig: Source figure (color version).
        filename: Download filename (without extension).
        label: Button label text.
        width: Export width in pixels.
        height: Export height in pixels.
    """
    try:
        bw_fig = to_bw(fig)

        try:
            img_bytes = bw_fig.to_image(
                format="png",
                width=width,
                height=height,
                scale=3,  # 300 DPI equivalent
            )
        except Exception:
            img_bytes = bw_fig.to_image(
                format="png",
                width=width,
                height=height,
            )

        st.download_button(
            label=label,
            data=img_bytes,
            file_name=f"{filename}_bw.png",
            mime="image/png",
            key=f"bw_{filename}",
        )
    except Exception:
        # Fallback gracefully — do not pollute dashboard UI with error boxes
        st.caption("💡 *Tải ảnh B&W: Bấm biểu tượng máy ảnh ở góc trên bên phải biểu đồ Plotly.*")
