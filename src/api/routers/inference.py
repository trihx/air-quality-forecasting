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

        recent_data = get_latest_data(200)

        if model_name == "gru":
            from src.inference.predictor import GRUPredictor

            gru_pred = GRUPredictor(horizon=horizon)
            result = gru_pred.predict(recent_data)

        elif model_name == "gru_quantile":
            from src.inference.predictor import GRUQuantilePredictor

            q_pred = GRUQuantilePredictor(horizon=horizon)
            result = q_pred.predict(recent_data)

        elif model_name == "lightgbm":
            # LightGBM needs feature-engineered data
            from src.features.builder import build_features
            from src.inference.predictor import LightGBMPredictor

            df_features = build_features(recent_data)
            lgb_pred = LightGBMPredictor(horizon=horizon)
            result = lgb_pred.predict(df_features)

        elif model_name == "ensemble":
            # Weighted Ensemble GRU + LightGBM
            from src.features.builder import build_features
            from src.inference.predictor import GRUQuantilePredictor, LightGBMPredictor

            weights = {1: {"gru": 0.00, "lgbm": 1.00}, 6: {"gru": 0.45, "lgbm": 0.55}, 24: {"gru": 0.70, "lgbm": 0.30}}
            w = weights.get(horizon, {"gru": 0.50, "lgbm": 0.50})

            q_pred = GRUQuantilePredictor(horizon=horizon)
            gru_result = q_pred.predict(recent_data)
            gru_val = gru_result["predicted_pm25"]

            df_features = build_features(recent_data)
            lgb_pred = LightGBMPredictor(horizon=horizon)
            lgb_result = lgb_pred.predict(df_features)
            lgb_val = lgb_result["predicted_pm25"]

            ensemble_val = round(gru_val * w["gru"] + lgb_val * w["lgbm"], 2)
            result = {
                "predicted_pm25": ensemble_val,
                "model": f"Ensemble (GRU×{w['gru']:.0%} + LightGBM×{w['lgbm']:.0%})",
                "horizon": horizon,
                "timestamp": gru_result.get("timestamp"),
                "input_rows": len(recent_data),
                "last_pm25": gru_result.get("last_pm25"),
                "device": "cpu",
                "pi_method": "cqr",
                "pi_lower": gru_result.get("pi_lower", ensemble_val),
                "pi_upper": gru_result.get("pi_upper", ensemble_val),
                "pi_width": gru_result.get("pi_width", 0),
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_name}. Supported: gru, gru_quantile, lightgbm, ensemble",
            )

        logger.info(f"Prediction: model={model_name}, horizon={horizon}h, pm25={result['predicted_pm25']}")
        return PredictionResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") from e
