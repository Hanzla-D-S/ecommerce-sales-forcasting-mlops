from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import (
    predict_single,
    predict_next_n_days,
    load_metrics
)

app = FastAPI(
    title="SmartSales Forecasting API",
    description="E-commerce sales forecasting using XGBoost + MLflow",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────────────────────────────

class SinglePredictionRequest(BaseModel):
    date: str = Field(
        ...,
        description="Date to predict in YYYY-MM-DD format",
        example="2024-01-15"
    )


class SinglePredictionResponse(BaseModel):
    date: str
    predicted_revenue: float
    currency: str = "GBP"


class ForecastRequest(BaseModel):
    days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Number of days to forecast (1-30)"
    )


class ForecastItem(BaseModel):
    date: str
    predicted_revenue: float


class ForecastResponse(BaseModel):
    forecast: List[ForecastItem]
    total_predicted_revenue: float
    currency: str = "GBP"


class ModelInfoResponse(BaseModel):
    model_type: str
    metrics: dict
    status: str


# ── Endpoints ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "SmartSales Forecasting API",
        "docs": "/docs",
        "endpoints": ["/predict", "/forecast", "/model-info"]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=SinglePredictionResponse)
def predict(request: SinglePredictionRequest):
    """Predict revenue for a single date."""
    try:
        target_date = datetime.strptime(request.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )
    try:
        prediction = predict_single(target_date)
        return SinglePredictionResponse(
            date=request.date,
            predicted_revenue=round(prediction, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """Forecast revenue for next N days."""
    try:
        predictions = predict_next_n_days(request.days)
        total = sum(p["predicted_revenue"] for p in predictions)
        return ForecastResponse(
            forecast=[ForecastItem(**p) for p in predictions],
            total_predicted_revenue=round(total, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Get model information and performance metrics."""
    try:
        metrics = load_metrics()
        return ModelInfoResponse(
            model_type="XGBoost Regressor",
            metrics=metrics,
            status="active"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))