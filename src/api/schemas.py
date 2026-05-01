"""Pydantic schemas for API request/response validation.

Follows the Pydantic V2 pattern:
    - *Create schemas for POST/PUT requests
    - *Response schemas for GET responses
    - *Base schemas for shared fields
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ExperimentCreate(BaseModel):
    """Create a new experiment."""

    name: str = Field(..., min_length=1, max_length=255, examples=["v8_final_cqr"])
    description: str | None = Field(None, examples=["Final pipeline with CQR calibration"])
    pipeline_version: str | None = Field(None, examples=["v8"])
    data_hash_md5: str | None = Field(None, min_length=32, max_length=32)
    config: dict | None = None


class ExperimentResponse(BaseModel):
    """Experiment response."""

    id: int
    name: str
    description: str | None
    pipeline_version: str | None
    data_hash_md5: str | None
    config: dict | None
    created_at: datetime
    run_count: int = 0

    model_config = {"from_attributes": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RunCreate(BaseModel):
    """Create a new run within an experiment."""

    experiment_id: int
    horizon: int = Field(..., ge=1, le=168, examples=[1, 6, 24])


class RunResponse(BaseModel):
    """Run response."""

    id: int
    experiment_id: int
    horizon: int
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    model_count: int = 0

    model_config = {"from_attributes": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RunModel Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RunModelCreate(BaseModel):
    """Log a model within a run."""

    run_id: int
    model_name: str = Field(..., max_length=100, examples=["GRU", "LightGBM", "LSTM"])
    training_time_s: float | None = None
    hyperparameters: dict | None = None
    feature_set: list[str] | None = None
    weight_hash_md5: str | None = None
    weight_path: str | None = None


class RunModelResponse(BaseModel):
    """RunModel response."""

    id: int
    run_id: int
    model_name: str
    training_time_s: float | None
    hyperparameters: dict | None
    feature_set: list | None
    weight_hash_md5: str | None
    weight_path: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metric Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MetricCreate(BaseModel):
    """Log metrics for a model."""

    run_model_id: int
    mae: float | None = None
    rmse: float | None = None
    mase: float | None = None
    r2: float | None = None
    mape: float | None = None
    smape: float | None = None
    forecast_bias: float | None = None
    medae: float | None = None
    extra_metrics: dict | None = None


class MetricResponse(BaseModel):
    """Metric response."""

    id: int
    run_model_id: int
    mae: float | None
    rmse: float | None
    mase: float | None
    r2: float | None
    mape: float | None
    smape: float | None
    forecast_bias: float | None
    medae: float | None
    extra_metrics: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inference Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PredictionRequest(BaseModel):
    """Request body for PM2.5 prediction."""

    horizon: int = Field(..., ge=1, le=168, examples=[1, 6, 24])
    model_name: str = Field(
        default="gru", examples=["gru", "gru_quantile", "lightgbm"]
    )


class PredictionResponse(BaseModel):
    """Prediction response."""

    predicted_pm25: float
    model: str
    horizon: int
    timestamp: str
    input_rows: int | None = None
    last_pm25: float | None = None
    device: str | None = None
    # CQR fields (optional)
    pi_method: str | None = None
    pi_lower: float | None = None
    pi_upper: float | None = None
    pi_width: float | None = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Audit Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DataHashResponse(BaseModel):
    """Data file hash for audit."""

    file_path: str
    hash_md5: str
    file_size_bytes: int
    computed_at: str


class ModelWeightResponse(BaseModel):
    """Model weight audit info."""

    model_name: str
    horizon: int
    weight_path: str
    hash_md5: str
    file_size_bytes: int
    training_device: str | None = None


class AuditReportResponse(BaseModel):
    """Full audit report."""

    data_hashes: list[DataHashResponse]
    model_weights: list[ModelWeightResponse]
    test_suite_status: str | None = None
    computed_at: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# General
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    database: str = "connected"
    models_loaded: int = 0
