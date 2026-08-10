"""Reporting Charts — Plotly chart factory for ReportingEngine data.

All charts are created via ``chart_factory.chart()`` for consistent theming
(legend position, margins, fonts, grayscale-safe markers).

Usage::

    from src.reporting.charts import plot_mase_comparison
    fig = plot_mase_comparison(engine)
    render_chart(fig, filename="mase_comparison")
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.viz.chart_factory import (
    add_baseline,
    add_value_annotations,
    chart,
    styled_bar,
)
from src.viz.palettes import MODEL_FAMILY_COLORS

HORIZONS = ("1h", "6h", "24h")


def plot_mase_comparison(
    engine,
    height: int = 520,
    model_filter: list[str] | None = None,
) -> go.Figure:
    """Create MASE comparison bar chart across all horizons.

    Args:
        engine: ReportingEngine instance.
        height: Chart height in pixels.
        model_filter: Optional list of models to display.

    Returns:
        Fully styled Plotly Figure ready for ``render_chart()``.
    """
    mase_data = engine.get_mase_data(model_filter=model_filter)

    fig = chart(
        xaxis_title="Forecast Horizon",
        yaxis_title="MASE (thấp hơn = tốt hơn)",
        height=height,
        barmode="group",
    )

    for i, (model, vals) in enumerate(mase_data.items()):
        fig.add_trace(
            styled_bar(
                i,
                name=model,
                x=list(HORIZONS),
                y=vals,
                textfmt=".3f",
            )
        )

    # Baseline reference line
    add_baseline(fig, y=1.0, label="Baseline (MASE = 1.0)")

    return fig


def plot_mae_trend(
    engine,
    height: int = 450,
    model_filter: list[str] | None = None,
) -> go.Figure:
    """Create MAE trend line chart across horizons.

    Args:
        engine: ReportingEngine instance.
        height: Chart height in pixels.
        model_filter: Optional list of models to display.

    Returns:
        Fully styled Plotly Figure.
    """
    from src.viz.chart_factory import styled_line

    mae_data = engine.get_mae_data(model_filter=model_filter)

    fig = chart(
        xaxis_title="Forecast Horizon",
        yaxis_title="MAE (µg/m³)",
        height=height,
    )

    for i, (model, vals) in enumerate(mae_data.items()):
        fig.add_trace(
            styled_line(
                i,
                name=model,
                x=list(HORIZONS),
                y=vals,
                text=[f"{v:.2f}" for v in vals],
            )
        )

    return fig


# ── Family color mapping for representative chart ──────────────
_FAMILY_COLORS = MODEL_FAMILY_COLORS


def plot_mae_trend_top5(
    engine,
    height: int = 520,
) -> go.Figure:
    """Create MAE grouped bar chart with only top representative models.

    Shows best model per family (Baseline, Statistical, ML, DL,
    Transformer, Ensemble) sorted by average MAE ascending.

    Args:
        engine: ReportingEngine instance.
        height: Chart height in pixels.

    Returns:
        Fully styled Plotly Figure.
    """
    from src.reporting.engine import get_model_type

    reps = engine.get_representative_models()
    mae_data = engine.get_mae_data(model_filter=reps)

    # Sort models by average MAE ascending (best first → leftmost bar)
    avg_mae = {m: sum(v) / len(v) for m, v in mae_data.items()}
    sorted_models = sorted(mae_data.keys(), key=lambda m: avg_mae[m])

    x_vals = list(range(len(HORIZONS)))

    fig = chart(
        xaxis_title="Forecast Horizon",
        yaxis_title="MAE (µg/m³)",
        height=height,
        barmode="group",
        layout_overrides={
            "xaxis": dict(
                tickmode="array",
                tickvals=x_vals,
                ticktext=list(HORIZONS),
            ),
        },
    )

    for model in sorted_models:
        vals = mae_data[model]
        mtype = get_model_type(model)
        color = _FAMILY_COLORS.get(mtype, "#888888")
        fig.add_trace(
            go.Bar(
                name=f"{model} ({mtype})",
                x=x_vals,
                y=vals,
                marker_color=color,
                textposition="none",  # annotations handle text
            )
        )

    # Add value annotations with white background
    add_value_annotations(
        fig,
        models=sorted_models,
        data=mae_data,
        categories=list(HORIZONS),
        fmt=".2f",
    )

    max_val = max(v for vals in mae_data.values() for v in vals)
    fig.update_layout(yaxis_range=[0, max_val * 1.20])

    return fig


def plot_mase_comparison_top5(
    engine,
    height: int = 520,
) -> go.Figure:
    """Create MASE grouped bar chart with only top representative models.

    Shows best model per family (Baseline, Statistical, ML, DL,
    Transformer, Ensemble) sorted by MASE ascending within each horizon.
    Includes MASE=1.0 baseline reference line.

    Args:
        engine: ReportingEngine instance.
        height: Chart height in pixels.

    Returns:
        Fully styled Plotly Figure.
    """
    from src.reporting.engine import get_model_type

    reps = engine.get_representative_models()
    mase_data = engine.get_mase_data(model_filter=reps)

    # Sort models by average MASE ascending (best first → leftmost bar)
    avg_mase = {m: sum(v) / len(v) for m, v in mase_data.items()}
    sorted_models = sorted(mase_data.keys(), key=lambda m: avg_mase[m])

    x_vals = list(range(len(HORIZONS)))

    fig = chart(
        xaxis_title="Forecast Horizon",
        yaxis_title="MASE (thấp hơn = tốt hơn)",
        height=height,
        barmode="group",
        layout_overrides={
            "xaxis": dict(
                tickmode="array",
                tickvals=x_vals,
                ticktext=list(HORIZONS),
            ),
        },
    )

    for model in sorted_models:
        vals = mase_data[model]
        mtype = get_model_type(model)
        color = _FAMILY_COLORS.get(mtype, "#888888")
        fig.add_trace(
            go.Bar(
                name=f"{model} ({mtype})",
                x=x_vals,
                y=vals,
                marker_color=color,
                textposition="none",  # annotations handle text
            )
        )

    # Add value annotations with white background
    add_value_annotations(
        fig,
        models=sorted_models,
        data=mase_data,
        categories=list(HORIZONS),
        fmt=".3f",
    )

    # Baseline reference line — bold and prominent
    add_baseline(fig, y=1.0, label="▶ Persistence Baseline (MASE = 1.0)")

    # Add extra headroom for text labels
    max_val = max(v for vals in mase_data.values() for v in vals)
    fig.update_layout(yaxis_range=[0, max_val * 1.20])

    return fig


def plot_dm_test_heatmap(dm_results: list[dict], height: int = 400) -> go.Figure:
    """Create Diebold-Mariano test result heatmap.

    Args:
        dm_results: List of DM test result dicts.
        height: Chart height.

    Returns:
        Plotly heatmap figure.
    """
    if not dm_results:
        fig = chart(height=height)
        fig.add_annotation(text="Chưa có dữ liệu DM test", showarrow=False)
        return fig

    import pandas as pd


    df = pd.DataFrame(dm_results)
    pivot = df.pivot_table(index="model_1", columns="model_2", values="p_value")

    fig = chart(
        xaxis_title="Model 2",
        yaxis_title="Model 1",
        height=height,
        hovermode="closest",
    )
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=0.1,
            text=pivot.applymap(lambda x: f"{x:.3f}" if pd.notna(x) else "").values,
            texttemplate="%{text}",
        )
    )

    return fig
