# VTF Chart Guide — Plotly Chart Factory

> **Module:** `src.viz.chart_factory`
> **Version:** v1.0 (2026-05-07)
> **Author:** trihx

## Overview

All Plotly charts on the PM2.5 dashboard **must** be created through the Chart Factory.
It enforces:

| Concern | Standard |
|---|---|
| **Font** | Inter (web), Times New Roman (print fallback) |
| **Legend** | Bottom-center, horizontal (IEEE standard) |
| **Grayscale** | Marker shapes + dash patterns via `GRAYSCALE_MARKERS` |
| **Heatmaps** | `Viridis` colorscale (monotonic, grayscale-readable) |
| **Export** | PNG @300 DPI (scale=3), SVG-ready layouts |
| **Colors** | `PALETTE_CATEGORICAL` (10 colors) for traces |

---

## Quick Start

```python
from src.viz.chart_factory import chart, render_chart, add_baseline

# 1. Create figure
fig = chart(
    title="MASE Comparison",
    xaxis_title="Forecast Horizon",
    yaxis_title="MASE",
    height=500,
)

# 2. Add traces
fig.add_trace(go.Bar(name="GRU", x=["1h","6h","24h"], y=[0.9,0.5,0.6]))

# 3. Add annotations
add_baseline(fig, y=1.0, label="Persistence (MASE=1.0)")

# 4. Render
render_chart(fig, filename="mase_comparison")
```

---

## API Reference

### `chart(**kwargs) → go.Figure`

Create a pre-styled Plotly Figure with consistent theming.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `title` | `str` | `""` | Chart title (supports HTML) |
| `xaxis_title` | `str` | `""` | X-axis label |
| `yaxis_title` | `str` | `""` | Y-axis label |
| `height` | `int` | `450` | Figure height in pixels |
| `mode` | `str\|None` | `None` | Force `'light'`/`'dark'`, `None` = auto |
| `showlegend` | `bool` | `True` | Show legend |
| `barmode` | `str\|None` | `None` | `'group'`, `'stack'`, `'overlay'` |
| `hovermode` | `str` | `"x unified"` | Plotly hover mode |
| `margin` | `dict\|None` | `None` | Override `dict(l,r,t,b)` |
| `layout_overrides` | `dict\|None` | `None` | Extra `update_layout()` kwargs |

### `render_chart(fig, *, filename, scale, key) → None`

Render a figure into Streamlit with standardized export config.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fig` | `go.Figure` | required | The figure to render |
| `filename` | `str` | `"chart"` | Default PNG download name |
| `scale` | `int` | `3` | DPI multiplier (3 = 300 DPI) |
| `key` | `str\|None` | `None` | Streamlit widget key |

### Trace Helpers

| Function | Purpose |
|---|---|
| `styled_line(index, *, name, x, y, ...)` | Grayscale-safe `go.Scatter` with marker + dash |
| `styled_bar(index, *, name, x, y, ...)` | Consistent `go.Bar` with palette color |

### Annotation Helpers

| Function | Purpose |
|---|---|
| `add_baseline(fig, *, y, label, color)` | Horizontal reference line (e.g. MASE=1.0) |
| `add_value_annotations(fig, *, models, data, categories)` | Value labels above grouped bars |
| `add_rangeslider(fig)` | Interactive range slider for time series |

---

## Chart Type Recipes

### Time Series Line Chart

```python
fig = chart(title="PM2.5 72h History", xaxis_title="Time", yaxis_title="µg/m³")
fig.add_trace(styled_line(0, name="Actual", x=dates, y=values))
fig.add_trace(styled_line(1, name="Predicted", x=dates, y=preds))
render_chart(fig, filename="timeseries")
```

### Grouped Bar Chart

```python
fig = chart(title="MASE by Horizon", barmode="group")
for i, model in enumerate(models):
    fig.add_trace(styled_bar(i, name=model, x=horizons, y=mase_values[model]))
add_baseline(fig, y=1.0)
render_chart(fig, filename="mase_grouped")
```

### Heatmap (Correlation / Calendar)

```python
fig = chart(title="Correlation Matrix", height=450, hovermode="closest")
fig.add_trace(go.Heatmap(
    z=corr_matrix, x=labels, y=labels,
    colorscale="Viridis",  # ← ALWAYS Viridis for grayscale
    colorbar=dict(title="r", thickness=15),
))
render_chart(fig, filename="correlation")
```

### Radar (Polar) Chart

```python
fig = chart(title="", height=400, showlegend=False)
fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself'))
fig.update_layout(polar=dict(
    radialaxis=dict(visible=True, range=[0, 1]),
    bgcolor='rgba(0,0,0,0)',
))
render_chart(fig, filename="radar")
```

### Sankey Diagram

```python
fig = go.Figure(go.Sankey(...))  # Sankey uses go.Figure directly
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Arial, sans-serif", size=10),
    margin=dict(l=60, r=30, t=60, b=80), height=650,
)
render_chart(fig, filename="pipeline_sankey")
```

---

## Grayscale Palette

The `GRAYSCALE_MARKERS` system ensures charts remain readable in B&W thesis prints:

| Index | Color | Symbol | Dash |
|---|---|---|---|
| 0 | `#00D4AA` (Teal) | circle | solid |
| 1 | `#FF6B6B` (Coral) | square | dash |
| 2 | `#4ECDC4` (Cyan) | diamond | dot |
| 3 | `#FFE66D` (Yellow) | triangle-up | dashdot |
| 4 | `#A78BFA` (Purple) | x | longdash |
| 5 | `#FB923C` (Orange) | star | longdashdot |
| 6 | `#60A5FA` (Blue) | hexagram | solid |
| 7 | `#F472B6` (Pink) | pentagon | dash |

For heatmaps: **always use `Viridis`** — it is monotonic and prints cleanly in grayscale.

---

## Migration from Legacy API

### Before (❌ Deprecated)

```python
from src.viz.theme import get_plotly_template, apply_plotly_style

fig = go.Figure()
fig.add_trace(...)
_tpl = get_plotly_template(st.session_state.get("theme", "light"))["layout"]
fig.update_layout(**_tpl, title="...", height=450)
st.plotly_chart(fig, use_container_width=True)
```

### After (✅ Chart Factory)

```python
from src.viz.chart_factory import chart as _chart, render_chart as _render_chart

fig = _chart(title="...", height=450)
fig.add_trace(...)
_render_chart(fig, filename="my_chart")
```

### Key Differences

| Legacy | Chart Factory |
|---|---|
| Manual `get_plotly_template()` | Auto-detected in `chart()` |
| Manual `st.plotly_chart()` | `render_chart()` with export config |
| Ad-hoc legend/margin | IEEE standard legend, consistent margins |
| No export config | PNG @300 DPI + clean modebar |
| Hardcoded `template="plotly_dark"` | Theme-agnostic (transparent bg) |

---

## File Inventory

| File | Charts | Status |
|---|---|---|
| `src/reporting/charts.py` | 6 | ✅ Migrated |
| `src/explainability_hub.py` | 7 + 2 Sankey | ✅ Migrated |
| `app.py` | 25 | ✅ Migrated |
| `pages.py` | 4 | ✅ Migrated |
| `src/pipeline_walkthrough.py` | 1 | ✅ Migrated |
| **Total** | **~38 charts** | **100% migrated** |
