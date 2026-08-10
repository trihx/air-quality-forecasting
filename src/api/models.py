"""ORM models for experiment tracking.

Schema design: Relational + JSONB (hybrid).
- Fixed columns for high-frequency queries (metrics, timestamps).
- JSONB columns for flexible data (hyperparameters, configs).

Tables:
    experiments → runs → run_models → metrics
                                    → feature_importances
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.database import Base


class Experiment(Base):
    """Top-level experiment grouping.

    One experiment = one pipeline execution (e.g., "v8_final_cqr").
    """

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    data_hash_md5: Mapped[str | None] = mapped_column(String(32), nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    runs: Mapped[list[Run]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class Run(Base):
    """A single training/evaluation run within an experiment.

    One run = one horizon (e.g., 1h, 6h, 24h).
    """

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending, running, completed, failed
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    run_models: Mapped[list[RunModel]] = relationship(back_populates="run", cascade="all, delete-orphan")


class RunModel(Base):
    """A specific model trained/evaluated within a run.

    One run can have multiple models (e.g., LightGBM, GRU, LSTM).
    """

    __tablename__ = "run_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    training_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    hyperparameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feature_set: Mapped[list | None] = mapped_column(JSON, nullable=True)
    weight_hash_md5: Mapped[str | None] = mapped_column(String(32), nullable=True)
    weight_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    run: Mapped[Run] = relationship(back_populates="run_models")
    metrics: Mapped[list[Metric]] = relationship(back_populates="run_model", cascade="all, delete-orphan")
    feature_importances: Mapped[list[FeatureImportance]] = relationship(
        back_populates="run_model", cascade="all, delete-orphan"
    )


class Metric(Base):
    """Evaluation metrics for a model.

    Fixed columns for fast queries (MAE, RMSE, MASE, R²).
    JSONB for extra/custom metrics.
    """

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("run_models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    mase: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2: Mapped[float | None] = mapped_column(Float, nullable=True)
    mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    smape: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast_bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    medae: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    run_model: Mapped[RunModel] = relationship(back_populates="metrics")


class FeatureImportance(Base):
    """Feature importance scores for a model."""

    __tablename__ = "feature_importances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("run_models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    feature_name: Mapped[str] = mapped_column(String(200), nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    run_model: Mapped[RunModel] = relationship(back_populates="feature_importances")


class InfoCard(Base):
    """Dashboard info card content stored in DB.

    Each card has a unique key (e.g., "overview_guide") and belongs to a page
    group (e.g., "overview"). Content is Markdown text rendered by Streamlit.

    Design rationale:
    - card_key (unique index): O(1) lookup for single card fetch.
    - page (indexed): Filtered queries by dashboard page.
    - display_order: Controls rendering sequence within a page.
    - content (Text): Unlimited Markdown, no VARCHAR truncation risk.
    """

    __tablename__ = "info_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    card_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    display_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
