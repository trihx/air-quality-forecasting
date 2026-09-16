"""Backward-compatible facade for dashboard presentation pages.

All page implementations have been modularized into `src/frontend/pages/`:
  - `page_forecast`: src.frontend.pages.forecast
  - `page_actual_vs_predicted`: src.frontend.pages.actual_vs_predicted
  - `page_experiment_runs`: src.frontend.pages.experiment_runs
  - `page_training`: src.frontend.pages.training
"""

from __future__ import annotations

from src.frontend.pages.actual_vs_predicted import (
    _load_avp_cache,
    _render_avp_chart,
    page_actual_vs_predicted,
)
from src.frontend.pages.experiment_runs import (
    _render_all_runs,
    _render_version_comparison,
    page_experiment_runs,
)
from src.frontend.pages.forecast import (
    WHO_LEVELS,
    _cached_pipeline_data,
    _detect_available_models,
    _get_torch_device,
    _pm25_color,
    page_forecast,
)
from src.frontend.pages.training import (
    page_training,
)

__all__ = [
    "WHO_LEVELS",
    "_cached_pipeline_data",
    "_detect_available_models",
    "_get_torch_device",
    "_load_avp_cache",
    "_pm25_color",
    "_render_all_runs",
    "_render_avp_chart",
    "_render_version_comparison",
    "page_actual_vs_predicted",
    "page_experiment_runs",
    "page_forecast",
    "page_training",
]
