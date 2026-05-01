"""Inference router — PM2.5 prediction endpoints.

Endpoints:
    POST /predict — Run PM2.5 prediction with specified model and horizon.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict PM2.5 for a given horizon using specified model.

    Supported models:
        - gru: GRU TorchScript (point prediction)
        - gru_quantile: GRU Quantile + CQR (prediction intervals)
        - lightgbm: LightGBM gradient boosting
    """
    model_name = request.model_name.lower()
    horizon = request.horizon

    try:
        # Lazy import to avoid MPS/LightGBM conflicts
        from src.inference.predictor import get_latest_data

        recent_data = get_latest_data()

        if model_name == "gru":
            from src.inference.predictor import GRUPredictor

            predictor = GRUPredictor(horizon=horizon)
            result = predictor.predict(recent_data)

        elif model_name == "gru_quantile":
            from src.inference.predictor import GRUQuantilePredictor

            predictor = GRUQuantilePredictor(horizon=horizon)
            result = predictor.predict(recent_data)

        elif model_name == "lightgbm":
            from src.inference.predictor import LightGBMPredictor

            # LightGBM needs feature-engineered data
            from src.features.builder import build_features

            df_features = build_features(recent_data)
            predictor = LightGBMPredictor(horizon=horizon)
            result = predictor.predict(df_features)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_name}. Supported: gru, gru_quantile, lightgbm",
            )

        logger.info(f"Prediction: model={model_name}, horizon={horizon}h, pm25={result['predicted_pm25']}")
        return PredictionResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
