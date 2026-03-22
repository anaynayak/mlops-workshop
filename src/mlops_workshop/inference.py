"""Batch inference for NYC Taxi trip duration prediction."""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

from mlops_workshop.features import get_feature_columns
from mlops_workshop.train import load_model


def predict(
    df: pd.DataFrame,
    model: RandomForestRegressor | None = None,
    model_path: str | Path | None = None,
) -> pd.Series:
    """Generate predictions for new data.

    Args:
        df: DataFrame with features
        model: Pre-loaded model (optional)
        model_path: Path to saved model (optional)

    Returns:
        Series with predictions
    """
    if model is None and model_path is None:
        raise ValueError("Either model or model_path must be provided")

    if model is None:
        model = load_model(model_path)

    feature_cols = get_feature_columns()
    X = df[feature_cols]

    return pd.Series(model.predict(X), index=df.index)


def run_batch_inference(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
) -> pd.DataFrame:
    """Run batch inference on a parquet file.

    Args:
        input_path: Path to input parquet file
        output_path: Path to save predictions
        model_path: Path to saved model

    Returns:
        DataFrame with input data and predictions
    """
    df = pd.read_parquet(input_path)
    model = load_model(model_path)

    df["predicted_trip_time"] = predict(df, model=model)
    df["prediction_error"] = abs(df["trip_time"] - df["predicted_trip_time"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return df
