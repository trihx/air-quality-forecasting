"""Centralized experiment tracking — write to DB or API.

Provides ExperimentLogger class with two modes:
  - "db": Direct SQLAlchemy insert (default, fast, no server needed)
  - "api": HTTP POST to FastAPI backend (for Docker/remote)

Usage:
    from src.api.experiment_logger import ExperimentLogger

    logger = ExperimentLogger(mode="db")
    logger.log_full_result(
        experiment_name="v8_dl_20260501",
        result_dict=json_data,
        description="DL models retrain",
        version="v8",
    )
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from loguru import logger as log


def _sanitize_float(val: Any) -> float | None:
    """Convert NaN/Infinity to None for DB storage."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


class ExperimentLogger:
    """Centralized experiment tracking — write to DB or API."""

    def __init__(self, mode: str = "db"):
        self.mode = mode
        if mode == "db":
            self._init_db()
        elif mode == "api":
            self._init_api()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'db' or 'api'.")

    # ── Initialization ──

    def _init_db(self):
        """Initialize direct DB connection via SQLAlchemy."""
        from src.api.database import Base, SessionLocal, engine
        from src.api.models import (  # noqa: F401
            Experiment,
            FeatureImportance,
            Metric,
            Run,
            RunModel,
        )

        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
        self._SessionLocal = SessionLocal
        log.info("ExperimentLogger: DB mode initialized")

    def _init_api(self):
        """Initialize HTTP API client."""
        import os

        self._api_base = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
        log.info(f"ExperimentLogger: API mode → {self._api_base}")

    # ── Core Methods ──

    def log_experiment(
        self,
        name: str,
        description: str | None = None,
        version: str | None = None,
        data_hash_md5: str | None = None,
        config: dict | None = None,
    ) -> int:
        """Create experiment, return experiment_id."""
        if self.mode == "db":
            return self._db_create_experiment(name, description, version, data_hash_md5, config)
        return self._api_create_experiment(name, description, version, data_hash_md5, config)

    def log_run(
        self,
        experiment_id: int,
        horizon: int,
        status: str = "completed",
    ) -> int:
        """Create run within experiment, return run_id."""
        if self.mode == "db":
            return self._db_create_run(experiment_id, horizon, status)
        return self._api_create_run(experiment_id, horizon, status)

    def log_model(
        self,
        run_id: int,
        model_name: str,
        training_time_s: float | None = None,
        hyperparameters: dict | None = None,
        feature_set: list[str] | None = None,
        weight_hash_md5: str | None = None,
        weight_path: str | None = None,
    ) -> int:
        """Log a model within a run, return run_model_id."""
        if self.mode == "db":
            return self._db_create_model(
                run_id,
                model_name,
                training_time_s,
                hyperparameters,
                feature_set,
                weight_hash_md5,
                weight_path,
            )
        return self._api_create_model(
            run_id,
            model_name,
            training_time_s,
            hyperparameters,
            feature_set,
            weight_hash_md5,
            weight_path,
        )

    def log_metrics(
        self,
        run_model_id: int,
        mae: float | None = None,
        rmse: float | None = None,
        mase: float | None = None,
        r2: float | None = None,
        mape: float | None = None,
        smape: float | None = None,
        forecast_bias: float | None = None,
        medae: float | None = None,
        extra_metrics: dict | None = None,
    ) -> int:
        """Log metrics for a model, return metric_id."""
        if self.mode == "db":
            return self._db_create_metrics(
                run_model_id,
                mae,
                rmse,
                mase,
                r2,
                mape,
                smape,
                forecast_bias,
                medae,
                extra_metrics,
            )
        return self._api_create_metrics(
            run_model_id,
            mae,
            rmse,
            mase,
            r2,
            mape,
            smape,
            forecast_bias,
            medae,
            extra_metrics,
        )

    def experiment_exists(self, name: str) -> bool:
        """Check if experiment with given name already exists."""
        if self.mode == "db":
            from src.api.models import Experiment

            db = self._SessionLocal()
            try:
                return db.query(Experiment).filter(Experiment.name == name).first() is not None
            finally:
                db.close()
        return False  # API mode: always allow (server handles dedup)

    # ── Convenience: Parse full JSON result ──

    def log_full_result(
        self,
        experiment_name: str,
        result_dict: dict,
        description: str | None = None,
        version: str | None = None,
        config: dict | None = None,
        source_file: str | None = None,
    ) -> int | None:
        """Parse a full JSON result dict → Experiment + Runs + Models + Metrics.

        Expected JSON structure:
            {
                "1h": {"ModelA": {"mae": ..., "rmse": ...}, ...},
                "6h": {...},
                "24h": {...},
                "_metadata": {...}  # optional, skipped
            }

        Returns experiment_id or None if skipped.
        """
        # Check duplicate
        if self.experiment_exists(experiment_name):
            log.warning(f"Experiment '{experiment_name}' already exists — skipping")
            return None

        # Extract metadata if present
        metadata = result_dict.get("_metadata", {})
        merged_config = {**(config or {}), **(metadata or {})}
        if source_file:
            merged_config["source_file"] = source_file

        # Create experiment
        exp_id = self.log_experiment(
            name=experiment_name,
            description=description,
            version=version,
            config=merged_config if merged_config else None,
        )
        log.info(f"Created experiment: {experiment_name} (id={exp_id})")

        # Parse horizons
        horizon_map = {"1h": 1, "6h": 6, "24h": 24}
        total_models = 0

        for h_key, horizon in horizon_map.items():
            h_data = result_dict.get(h_key, {})
            if not h_data or not isinstance(h_data, dict):
                continue

            # Create run for this horizon
            run_id = self.log_run(exp_id, horizon, status="completed")

            for model_name, metrics in h_data.items():
                if not isinstance(metrics, dict):
                    continue

                # Extract training time
                train_time = _sanitize_float(metrics.get("train_time_s"))

                # Extract classification data as extra_metrics
                extra = {}
                if "classification" in metrics:
                    extra["classification"] = metrics["classification"]
                if "best_weights" in metrics:
                    extra["best_weights"] = metrics["best_weights"]
                if "params" in metrics:
                    extra["params"] = metrics["params"]
                if "n_test" in metrics:
                    extra["n_test"] = metrics["n_test"]

                # Create model
                rm_id = self.log_model(
                    run_id=run_id,
                    model_name=model_name,
                    training_time_s=train_time,
                )

                # Create metrics
                self.log_metrics(
                    run_model_id=rm_id,
                    mae=_sanitize_float(metrics.get("mae")),
                    rmse=_sanitize_float(metrics.get("rmse")),
                    mase=_sanitize_float(metrics.get("mase")),
                    r2=_sanitize_float(metrics.get("r2")),
                    mape=_sanitize_float(metrics.get("mape")),
                    smape=_sanitize_float(metrics.get("smape")),
                    forecast_bias=_sanitize_float(metrics.get("forecast_bias")),
                    medae=_sanitize_float(metrics.get("medae")),
                    extra_metrics=extra if extra else None,
                )
                total_models += 1

        log.info(f"  → Logged {total_models} model records across {len(horizon_map)} horizons")
        return exp_id

    # ══════════════════════════════════════════════
    # DB Mode — Direct SQLAlchemy
    # ══════════════════════════════════════════════

    def _db_create_experiment(self, name, description, version, data_hash_md5, config) -> int:
        from src.api.models import Experiment

        db = self._SessionLocal()
        try:
            exp = Experiment(
                name=name,
                description=description,
                pipeline_version=version,
                data_hash_md5=data_hash_md5,
                config=config,
            )
            db.add(exp)
            db.commit()
            db.refresh(exp)
            return exp.id
        finally:
            db.close()

    def _db_create_run(self, experiment_id, horizon, status) -> int:
        from src.api.models import Run

        db = self._SessionLocal()
        try:
            run = Run(
                experiment_id=experiment_id,
                horizon=horizon,
                status=status,
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            return run.id
        finally:
            db.close()

    def _db_create_model(
        self,
        run_id,
        model_name,
        training_time_s,
        hyperparameters,
        feature_set,
        weight_hash_md5,
        weight_path,
    ) -> int:
        from src.api.models import RunModel

        db = self._SessionLocal()
        try:
            rm = RunModel(
                run_id=run_id,
                model_name=model_name,
                training_time_s=training_time_s,
                hyperparameters=hyperparameters,
                feature_set=feature_set,
                weight_hash_md5=weight_hash_md5,
                weight_path=weight_path,
            )
            db.add(rm)
            db.commit()
            db.refresh(rm)
            return rm.id
        finally:
            db.close()

    def _db_create_metrics(
        self,
        run_model_id,
        mae,
        rmse,
        mase,
        r2,
        mape,
        smape,
        forecast_bias,
        medae,
        extra_metrics,
    ) -> int:
        from src.api.models import Metric

        # Sanitize all float values for DB
        db = self._SessionLocal()
        try:
            metric = Metric(
                run_model_id=run_model_id,
                mae=_sanitize_float(mae),
                rmse=_sanitize_float(rmse),
                mase=_sanitize_float(mase),
                r2=_sanitize_float(r2),
                mape=_sanitize_float(mape),
                smape=_sanitize_float(smape),
                forecast_bias=_sanitize_float(forecast_bias),
                medae=_sanitize_float(medae),
                extra_metrics=extra_metrics,
            )
            db.add(metric)
            db.commit()
            db.refresh(metric)
            return metric.id
        finally:
            db.close()

    # ══════════════════════════════════════════════
    # API Mode — HTTP POST
    # ══════════════════════════════════════════════

    def _api_post(self, endpoint: str, data: dict) -> dict:
        import requests as req

        url = f"{self._api_base}/{endpoint}"
        resp = req.post(url, json=data, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _api_create_experiment(self, name, description, version, data_hash_md5, config) -> int:
        result = self._api_post(
            "experiments",
            {
                "name": name,
                "description": description,
                "pipeline_version": version,
                "data_hash_md5": data_hash_md5,
                "config": config,
            },
        )
        return result["id"]

    def _api_create_run(self, experiment_id, horizon, status) -> int:
        result = self._api_post(
            "runs",
            {
                "experiment_id": experiment_id,
                "horizon": horizon,
            },
        )
        return result["id"]

    def _api_create_model(
        self,
        run_id,
        model_name,
        training_time_s,
        hyperparameters,
        feature_set,
        weight_hash_md5,
        weight_path,
    ) -> int:
        result = self._api_post(
            "run-models",
            {
                "run_id": run_id,
                "model_name": model_name,
                "training_time_s": training_time_s,
                "hyperparameters": hyperparameters,
                "feature_set": feature_set,
                "weight_hash_md5": weight_hash_md5,
                "weight_path": weight_path,
            },
        )
        return result["id"]

    def _api_create_metrics(
        self,
        run_model_id,
        mae,
        rmse,
        mase,
        r2,
        mape,
        smape,
        forecast_bias,
        medae,
        extra_metrics,
    ) -> int:
        result = self._api_post(
            "metrics",
            {
                "run_model_id": run_model_id,
                "mae": _sanitize_float(mae),
                "rmse": _sanitize_float(rmse),
                "mase": _sanitize_float(mase),
                "r2": _sanitize_float(r2),
                "mape": _sanitize_float(mape),
                "smape": _sanitize_float(smape),
                "forecast_bias": _sanitize_float(forecast_bias),
                "medae": _sanitize_float(medae),
                "extra_metrics": extra_metrics,
            },
        )
        return result["id"]
