#!/usr/bin/env python
"""Batch inference script for workshop."""

import pandas as pd
from pathlib import Path

from mlops_workshop.features import prepare_features, get_feature_columns
from mlops_workshop.train import load_model
from mlops_workshop.evaluate import evaluate_model, print_metrics


def main():
    print("Loading data...")
    df = pd.read_parquet("data/raw/nyc_taxi.parquet")
    print(f"Loaded {len(df):,} rows")

    print("\nPreparing features...")
    df = prepare_features(df)
    print(f"After filtering: {len(df):,} rows")

    print("\nLoading model...")
    model = load_model("models/rf_model.joblib")

    print("\nRunning inference...")
    feature_cols = get_feature_columns()
    predictions = model.predict(df[feature_cols])

    df["predicted_trip_time"] = predictions
    df["prediction_error"] = abs(df["trip_time"] - predictions)

    print("\nEvaluating predictions...")
    metrics = evaluate_model(df["trip_time"], df["predicted_trip_time"])
    print_metrics(metrics)

    print("\nSaving predictions...")
    Path("output").mkdir(exist_ok=True)
    df.to_parquet("output/predictions.parquet", index=False)
    print("Predictions saved to output/predictions.parquet")

    print("\nSample predictions:")
    print(df[["trip_time", "predicted_trip_time", "prediction_error"]].head(10))


if __name__ == "__main__":
    main()
