"""Experiments router — CRUD for experiment tracking.

Endpoints:
    POST   /experiments          — Create experiment
    GET    /experiments          — List all experiments
    GET    /experiments/{id}     — Get experiment with runs
    POST   /runs                 — Create run
    POST   /run-models           — Log a model within a run
    POST   /metrics              — Log metrics for a model
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.models import Experiment, Metric, Run, RunModel
from src.api.schemas import (
    ExperimentCreate,
    ExperimentResponse,
    MetricCreate,
    MetricResponse,
    RunCreate,
    RunModelCreate,
    RunModelResponse,
    RunResponse,
)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
def create_experiment(data: ExperimentCreate, db: Session = Depends(get_db)):
    """Create a new experiment."""
    exp = Experiment(
        name=data.name,
        description=data.description,
        pipeline_version=data.pipeline_version,
        data_hash_md5=data.data_hash_md5,
        config=data.config,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        description=exp.description,
        pipeline_version=exp.pipeline_version,
        data_hash_md5=exp.data_hash_md5,
        config=exp.config,
        created_at=exp.created_at,
        run_count=0,
    )


@router.get("/experiments", response_model=list[ExperimentResponse])
def list_experiments(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """List all experiments, newest first."""
    experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()
    results = []
    for exp in experiments:
        run_count = db.query(func.count(Run.id)).filter(Run.experiment_id == exp.id).scalar()
        results.append(
            ExperimentResponse(
                id=exp.id,
                name=exp.name,
                description=exp.description,
                pipeline_version=exp.pipeline_version,
                data_hash_md5=exp.data_hash_md5,
                config=exp.config,
                created_at=exp.created_at,
                run_count=run_count or 0,
            )
        )
    return results


@router.get("/experiments/latest")
def get_latest_experiment(db: Session = Depends(get_db)):
    """Get the most recent experiment with full nested data."""
    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    if not exp:
        raise HTTPException(status_code=404, detail="No experiments found")
    return _build_experiment_summary(exp, db)


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """Get a single experiment by ID."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    run_count = db.query(func.count(Run.id)).filter(Run.experiment_id == exp.id).scalar()
    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        description=exp.description,
        pipeline_version=exp.pipeline_version,
        data_hash_md5=exp.data_hash_md5,
        config=exp.config,
        created_at=exp.created_at,
        run_count=run_count or 0,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Runs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post("/runs", response_model=RunResponse, status_code=201)
def create_run(data: RunCreate, db: Session = Depends(get_db)):
    """Create a new run within an experiment."""
    exp = db.query(Experiment).filter(Experiment.id == data.experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {data.experiment_id} not found")

    run = Run(
        experiment_id=data.experiment_id,
        horizon=data.horizon,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return RunResponse(
        id=run.id,
        experiment_id=run.experiment_id,
        horizon=run.horizon,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        created_at=run.created_at,
        model_count=0,
    )


@router.get("/experiments/{experiment_id}/runs", response_model=list[RunResponse])
def list_runs(experiment_id: int, db: Session = Depends(get_db)):
    """List runs for an experiment."""
    runs = db.query(Run).filter(Run.experiment_id == experiment_id).all()
    results = []
    for run in runs:
        model_count = db.query(func.count(RunModel.id)).filter(RunModel.run_id == run.id).scalar()
        results.append(
            RunResponse(
                id=run.id,
                experiment_id=run.experiment_id,
                horizon=run.horizon,
                status=run.status,
                started_at=run.started_at,
                finished_at=run.finished_at,
                created_at=run.created_at,
                model_count=model_count or 0,
            )
        )
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post("/run-models", response_model=RunModelResponse, status_code=201)
def create_run_model(data: RunModelCreate, db: Session = Depends(get_db)):
    """Log a model within a run."""
    run = db.query(Run).filter(Run.id == data.run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {data.run_id} not found")

    rm = RunModel(
        run_id=data.run_id,
        model_name=data.model_name,
        training_time_s=data.training_time_s,
        hyperparameters=data.hyperparameters,
        feature_set=data.feature_set,
        weight_hash_md5=data.weight_hash_md5,
        weight_path=data.weight_path,
    )
    db.add(rm)
    db.commit()
    db.refresh(rm)
    return RunModelResponse.model_validate(rm)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post("/metrics", response_model=MetricResponse, status_code=201)
def create_metric(data: MetricCreate, db: Session = Depends(get_db)):
    """Log metrics for a model."""
    rm = db.query(RunModel).filter(RunModel.id == data.run_model_id).first()
    if not rm:
        raise HTTPException(status_code=404, detail=f"RunModel {data.run_model_id} not found")

    metric = Metric(
        run_model_id=data.run_model_id,
        mae=data.mae,
        rmse=data.rmse,
        mase=data.mase,
        r2=data.r2,
        mape=data.mape,
        smape=data.smape,
        forecast_bias=data.forecast_bias,
        medae=data.medae,
        extra_metrics=data.extra_metrics,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return MetricResponse.model_validate(metric)


@router.get("/run-models/{run_model_id}/metrics", response_model=list[MetricResponse])
def list_metrics(run_model_id: int, db: Session = Depends(get_db)):
    """Get metrics for a model."""
    metrics = db.query(Metric).filter(Metric.run_model_id == run_model_id).all()
    return [MetricResponse.model_validate(m) for m in metrics]


@router.get("/experiments/{experiment_id}/summary")
def get_experiment_summary(experiment_id: int, db: Session = Depends(get_db)):
    """Get full experiment summary: experiment → runs → models → metrics."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return _build_experiment_summary(exp, db)


def _build_experiment_summary(exp: Experiment, db: Session) -> dict:
    """Build nested summary dict for an experiment."""
    runs = db.query(Run).filter(Run.experiment_id == exp.id).order_by(Run.horizon).all()

    runs_data = []
    for run in runs:
        models = db.query(RunModel).filter(RunModel.run_id == run.id).all()
        models_data = []
        for rm in models:
            metrics = db.query(Metric).filter(Metric.run_model_id == rm.id).all()
            metrics_data = [
                {
                    "id": m.id,
                    "mae": m.mae,
                    "rmse": m.rmse,
                    "mase": m.mase,
                    "r2": m.r2,
                    "mape": m.mape,
                    "smape": m.smape,
                    "forecast_bias": m.forecast_bias,
                    "medae": m.medae,
                    "extra_metrics": m.extra_metrics,
                }
                for m in metrics
            ]
            models_data.append(
                {
                    "id": rm.id,
                    "model_name": rm.model_name,
                    "training_time_s": rm.training_time_s,
                    "hyperparameters": rm.hyperparameters,
                    "metrics": metrics_data,
                }
            )
        runs_data.append(
            {
                "id": run.id,
                "horizon": run.horizon,
                "status": run.status,
                "models": models_data,
            }
        )

    return {
        "id": exp.id,
        "name": exp.name,
        "description": exp.description,
        "pipeline_version": exp.pipeline_version,
        "config": exp.config,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "runs": runs_data,
    }
