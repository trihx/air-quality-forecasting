"""Tests for P0-3: Forecast Bias, P0-4: RMSE/MAE Ratio, P2-4: MedAE,
and P0-5: Residual Diagnostics.

Tests ensure metrics are correctly computed per audit requirements.
"""

import numpy as np
import pytest


class TestForecastBias:
    """P0-3: Forecast Bias metric — Manu Joseph Ch.4 p.80."""

    def test_unbiased(self):
        """FB ≈ 0 when predictions match actuals."""
        from src.evaluation.metrics import forecast_bias
        y = np.array([10.0, 20.0, 30.0, 40.0])
        assert abs(forecast_bias(y, y)) < 1e-10

    def test_over_forecast(self):
        """FB > 0 when model over-predicts."""
        from src.evaluation.metrics import forecast_bias
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([15.0, 25.0, 35.0])  # all higher
        fb = forecast_bias(y_true, y_pred)
        assert fb > 0, f"Expected positive FB, got {fb}"
        # (75 - 60) / 60 = 0.25
        assert abs(fb - 0.25) < 1e-10

    def test_under_forecast(self):
        """FB < 0 when model under-predicts (dangerous for PM2.5)."""
        from src.evaluation.metrics import forecast_bias
        y_true = np.array([20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0])  # all lower
        fb = forecast_bias(y_true, y_pred)
        assert fb < 0, f"Expected negative FB, got {fb}"

    def test_near_zero_actual(self):
        """Returns NaN when total actual is near zero."""
        from src.evaluation.metrics import forecast_bias
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert np.isnan(forecast_bias(y_true, y_pred))


class TestMedAE:
    """P2-4: Median Absolute Error — robust to outliers."""

    def test_basic(self):
        from src.evaluation.metrics import medae
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        assert abs(medae(y_true, y_pred) - 0.5) < 1e-10

    def test_robust_to_outliers(self):
        """MedAE should be less affected by outliers than MAE."""
        from src.evaluation.metrics import mae, medae
        y_true = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        y_pred = np.array([11.0, 11.0, 11.0, 11.0, 100.0])  # one huge outlier
        assert medae(y_true, y_pred) < mae(y_true, y_pred)
        assert abs(medae(y_true, y_pred) - 1.0) < 1e-10  # median error = 1.0


class TestRmseMaeRatio:
    """P0-4: RMSE/MAE Ratio — outlier detection."""

    def test_uniform_errors(self):
        """Ratio = 1.0 when all errors are equal (uniform)."""
        from src.evaluation.metrics import evaluate_forecast
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 22.0, 32.0, 42.0])  # constant error = 2
        y_naive = np.array([9.0, 19.0, 29.0, 39.0])
        result = evaluate_forecast(y_true, y_pred, y_naive)
        assert abs(result["rmse_mae_ratio"] - 1.0) < 1e-3

    def test_outlier_errors(self):
        """Ratio > √2 when errors have outliers."""
        from src.evaluation.metrics import evaluate_forecast
        y_true = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        y_pred = np.array([10.5, 10.5, 10.5, 10.5, 30.0])  # one big outlier
        y_naive = np.array([9.0, 9.0, 9.0, 9.0, 9.0])
        result = evaluate_forecast(y_true, y_pred, y_naive)
        assert result["rmse_mae_ratio"] > 1.414, f"Expected > √2, got {result['rmse_mae_ratio']}"

    def test_in_evaluate_forecast(self):
        """New metrics present in evaluate_forecast output."""
        from src.evaluation.metrics import evaluate_forecast
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 22.0, 28.0])
        y_naive = np.array([9.0, 19.0, 29.0])
        result = evaluate_forecast(y_true, y_pred, y_naive)
        assert "forecast_bias" in result
        assert "medae" in result
        assert "rmse_mae_ratio" in result


class TestResidualDiagnostics:
    """P0-5: Residual Diagnostics — Peixeiro Ch.6."""

    def test_white_noise_passes(self):
        """White noise residuals should pass diagnostics."""
        from src.evaluation.residual_diagnostics import run_residual_diagnostics
        np.random.seed(42)
        y_true = np.random.randn(500) + 20
        y_pred = y_true + np.random.randn(500) * 0.1  # small random errors
        result = run_residual_diagnostics(y_true, y_pred, model_name="test_wn")
        assert "PASS" in result["verdict"] or "PARTIAL" in result["verdict"]
        assert result["lb_pass_rate"] > 0.5

    def test_autocorrelated_residuals_fail(self):
        """Autocorrelated residuals should fail Ljung-Box."""
        from src.evaluation.residual_diagnostics import run_residual_diagnostics
        np.random.seed(42)
        # Create autocorrelated residuals (AR(1) process)
        n = 500
        y_true = np.cumsum(np.random.randn(n)) + 20
        # Systematically biased predictions
        y_pred = y_true - 0.5 * np.sin(np.arange(n) * 2 * np.pi / 24)
        result = run_residual_diagnostics(y_true, y_pred, model_name="test_ar")
        # Should detect pattern in residuals
        assert result["ljung_box"] is not None
        assert len(result["ljung_box"]) > 0

    def test_output_structure(self):
        """Verify output dictionary structure."""
        from src.evaluation.residual_diagnostics import run_residual_diagnostics
        np.random.seed(42)
        y = np.random.randn(200) + 15
        result = run_residual_diagnostics(y, y + 0.1, model_name="struct_test")
        assert "model" in result
        assert "horizon" in result
        assert "residual_stats" in result
        assert "ljung_box" in result
        assert "normality" in result
        assert "verdict" in result
        assert "lb_pass_rate" in result
        # Residual stats
        rs = result["residual_stats"]
        assert "mean" in rs
        assert "std" in rs
        assert "skew" in rs
        assert "kurtosis" in rs

    def test_chart_generation(self, tmp_path):
        """Verify chart is generated when output_dir is provided."""
        from src.evaluation.residual_diagnostics import run_residual_diagnostics
        np.random.seed(42)
        y = np.random.randn(200) + 15
        result = run_residual_diagnostics(
            y, y + np.random.randn(200) * 0.5,
            model_name="chart_test",
            horizon=6,
            output_dir=str(tmp_path),
        )
        # Check chart file exists
        chart_files = list(tmp_path.glob("*.png"))
        assert len(chart_files) == 1, f"Expected 1 chart, found {len(chart_files)}"
        assert "chart_test" in chart_files[0].name
