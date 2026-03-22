#!/usr/bin/env python
"""Train model script for workshop."""

import os
import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn

from mlops_workshop.features import prepare_features
from mlops_workshop.train import train_model, save_model
from mlops_workshop.evaluate import evaluate_model, print_metrics


def main():
    # Configure MLflow to use SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("nyc-taxi-duration")

    print("Loading data...")
    df = pd.read_parquet("data/raw/nyc_taxi.parquet")
    print(f"Loaded {len(df):,} rows")

    print("\nPreparing features...")
    df = prepare_features(df)
    print(f"After filtering: {len(df):,} rows")

    # Training parameters
    n_estimators = 100
    max_depth = 10

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", 0.2)

        print("\nTraining model...")
        model, X_test, y_test = train_model(
            df, n_estimators=n_estimators, max_depth=max_depth
        )

        print("\nEvaluating model...")
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, pd.Series(y_pred))
        print_metrics(metrics)

        # Log metrics
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("rmse_minutes", metrics["rmse_minutes"])
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("r2", metrics["r2"])
        mlflow.log_metric("mape", metrics["mape"])

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("\nSaving model...")
        save_model(model, "models/rf_model.joblib")
        print("Model saved to models/rf_model.joblib")

        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
        print("View experiments: mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    main()
