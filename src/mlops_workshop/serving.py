"""FastAPI app for the Feast-backed workshop serving demo."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_workshop.feature_store import (
    FEATURE_MODEL_PATH,
    ROUTE_FEATURE_NAME,
    build_request_dataframe,
    get_feature_store,
    get_feature_store_feature_columns,
    get_online_route_features,
)
from mlops_workshop.inference import predict
from mlops_workshop.train import load_model

app = FastAPI(
    title="NYC Taxi Feature Store Demo",
    description="Prediction endpoint backed by Feast online features.",
)


class PredictionRequest(BaseModel):
    trip_miles: float = Field(gt=0)
    PULocationID: int = Field(gt=0)
    DOLocationID: int = Field(gt=0)
    pickup_datetime: datetime


class PredictionResponse(BaseModel):
    route_id: str
    route_avg_duration_24h: float
    predicted_trip_time: float


@lru_cache
def _model():
    if not FEATURE_MODEL_PATH.exists():
        raise FileNotFoundError("Feature-store model not found. Run: make feature-store")
    return load_model(FEATURE_MODEL_PATH)


@lru_cache
def _store():
    return get_feature_store()


@app.get("/health")
def health() -> dict[str, str]:
    _model()
    _store()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_trip(request: PredictionRequest) -> PredictionResponse:
    online_features = get_online_route_features(
        pickup_location_id=request.PULocationID,
        dropoff_location_id=request.DOLocationID,
        store=_store(),
    )
    route_feature = online_features[ROUTE_FEATURE_NAME]
    if route_feature is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No online feature value is available for route {online_features['route_id']}. "
                "Run `make feature-store` or choose a route from the sample dataset."
            ),
        )

    request_df = build_request_dataframe(
        trip_miles=request.trip_miles,
        pickup_location_id=request.PULocationID,
        dropoff_location_id=request.DOLocationID,
        pickup_datetime=request.pickup_datetime,
        route_avg_duration_24h=float(route_feature),
    )
    prediction = float(
        predict(
            request_df,
            model=_model(),
            feature_columns=get_feature_store_feature_columns(),
        ).iloc[0]
    )
    return PredictionResponse(
        route_id=str(online_features["route_id"]),
        route_avg_duration_24h=float(route_feature),
        predicted_trip_time=prediction,
    )
