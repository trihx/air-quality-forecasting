"""Unit tests verifying Dashboard bugfixes (EDA scatter, Detail Sankey, etc.)."""

from unittest.mock import patch

from src.eda_page import _get_scatter_dispersion_data
from src.explainability_hub import _render_detail_sankey


def test_get_scatter_dispersion_data():
    """Verify that dispersion helper loads a valid dataframe with required columns."""
    df = _get_scatter_dispersion_data()
    assert df is not None
    assert not df.empty
    assert "pm25" in df.columns
    assert "h1" in df.columns
    assert "h24" in df.columns
    assert len(df) <= 2000


def test_render_detail_sankey_handles_empty_pm():
    """Verify that _render_detail_sankey succeeds with empty pm data without failing or link=0 collapse."""
    with patch("streamlit.plotly_chart") as mock_chart, patch("streamlit.markdown"):
        # Test with empty pm
        _render_detail_sankey("30m", {}, {})
        assert mock_chart.called
        fig = mock_chart.call_args[0][0]
        sankey_trace = fig.data[0]
        # Verify links are non-empty and all values > 0
        assert len(sankey_trace.link.value) > 0
        assert all(v > 0 for v in sankey_trace.link.value)
        # Verify all nodes are present
        assert len(sankey_trace.node.label) >= 20


def test_render_detail_sankey_data_and_feature_counts():
    """Verify that detail Sankey displays audited 209,594 raw rows and 119 model features."""
    with patch("streamlit.plotly_chart") as mock_chart, patch("streamlit.markdown"):
        _render_detail_sankey("30m", {}, {})
        assert mock_chart.called
        fig = mock_chart.call_args[0][0]
        labels = fig.data[0].node.label
        assert any("209,594" in lbl for lbl in labels), f"Expected 209,594 in labels, got: {labels[:2]}"
        assert any("119 features" in lbl for lbl in labels), f"Expected 119 features in labels, got: {labels[4:6]}"
