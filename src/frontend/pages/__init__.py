"""Dashboard presentation pages for PM2.5 forecasting application."""

from __future__ import annotations

from src.frontend.pages.actual_vs_predicted import page_actual_vs_predicted
from src.frontend.pages.audit import page_scientific_audit
from src.frontend.pages.benchmarks import page_scientific_benchmark
from src.frontend.pages.content_manager import page_content_manager
from src.frontend.pages.experiment_runs import page_experiment_runs
from src.frontend.pages.forecast import page_forecast
from src.frontend.pages.hyperparams import page_hyperparams
from src.frontend.pages.multi_horizon import page_multi_horizon
from src.frontend.pages.overview import page_overview
from src.frontend.pages.prediction_intervals import page_prediction_intervals
from src.frontend.pages.training import page_training

__all__ = [
    "page_actual_vs_predicted",
    "page_content_manager",
    "page_experiment_runs",
    "page_forecast",
    "page_hyperparams",
    "page_multi_horizon",
    "page_overview",
    "page_prediction_intervals",
    "page_scientific_audit",
    "page_scientific_benchmark",
    "page_training",
]
