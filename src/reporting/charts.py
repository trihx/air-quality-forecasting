"""Reporting Charts — Unified Plotly chart factory using VTF.

Creates fully-styled Plotly figures from ReportingEngine data.
All charts use apply_plotly_style() from VTF for consistent theming
(legend bottom, margins, colors, fonts).

Usage:
    from src.reporting import ReportingEngine
    from src.reporting.charts import plot_mase_comparison
    rpt = ReportingEngine(snapshot_data)
    fig = plot_mase_comparison(rpt)
    st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.viz.theme import PALETTE_CATEGORICAL, PALETTE_SEMANTIC, apply_plotly_style

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
        Fully styled Plotly Figure ready for st.plotly_chart().
    """
    mase_data = engine.get_mase_data(model_filter=model_filter)

    fig = go.Figure()
    for i, (model, vals) in enumerate(mase_data.items()):
        fig.add_trace(go.Bar(
            name=model,
            x=list(HORIZONS),
            y=vals,
            marker_color=PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont={"size": 10},
        ))

    # Baseline reference line
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color=PALETTE_SEMANTIC["accent"],
        line_width=2,
        annotation_text="Baseline (MASE = 1.0)",
        annotation_font_color=PALETTE_SEMANTIC["accent"],
    )

    fig.update_layout(
        barmode="group",
        yaxis_title="MASE (thấp hơn = tốt hơn)",
        xaxis_title="Forecast Horizon",
        title={"text": "MASE < 1.0 → Mô hình vượt trội Persistence Baseline", "font": {"size": 15}},
    )

    # Apply VTF theme (fixes legend position, margins, colors)
    fig = apply_plotly_style(fig, height=height)
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
    mae_data = engine.get_mae_data(model_filter=model_filter)

    fig = go.Figure()
    for i, (model, vals) in enumerate(mae_data.items()):
        fig.add_trace(go.Scatter(
            name=model,
            x=list(HORIZONS),
            y=vals,
            mode="lines+markers",
            line=dict(color=PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)], width=2),
            marker=dict(size=8),
            text=[f"{v:.2f}" for v in vals],
            textposition="top center",
        ))

    fig.update_layout(
        yaxis_title="MAE (µg/m³)",
        xaxis_title="Forecast Horizon",
        title={"text": "MAE Trend — Lỗi Tuyệt Đối Theo Horizon", "font": {"size": 15}},
    )

    fig = apply_plotly_style(fig, height=height)
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
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu DM test", showarrow=False)
        return apply_plotly_style(fig, height=height)

    import pandas as pd
    df = pd.DataFrame(dm_results)

    fig = go.Figure(data=go.Heatmap(
        z=df.pivot_table(index="model_1", columns="model_2", values="p_value").values,
        x=df["model_2"].unique().tolist(),
        y=df["model_1"].unique().tolist(),
        colorscale="RdYlGn_r",
        zmin=0, zmax=0.1,
        text=df.pivot_table(index="model_1", columns="model_2", values="p_value").applymap(
            lambda x: f"{x:.3f}" if pd.notna(x) else ""
        ).values,
        texttemplate="%{text}",
    ))

    fig.update_layout(
        title={"text": "Diebold-Mariano Test — p-values", "font": {"size": 15}},
        xaxis_title="Model 2",
        yaxis_title="Model 1",
    )

    return apply_plotly_style(fig, height=height)
