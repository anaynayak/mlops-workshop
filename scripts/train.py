#!/usr/bin/env python
"""Train model script for workshop."""

import pandas as pd
from pathlib import Path

from mlops_workshop.features import prepare_features
from mlops_workshop.train import train_model, save_model
from mlops_workshop.evaluate import evaluate_model, print_metrics


def main():
    print("Loading data...")
    df = pd.read_parquet("data/raw/nyc_taxi.parquet")
    print(f"Loaded {len(df):,} rows")

    print("\nPreparing features...")
    df = prepare_features(df)
    print(f"After filtering: {len(df):,} rows")

    print("\nTraining model...")
    model, X_test, y_test = train_model(df, n_estimators=100, max_depth=10)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, pd.Series(y_pred))
    print_metrics(metrics)

    print("\nSaving model...")
    save_model(model, "models/rf_model.joblib")
    print("Model saved to models/rf_model.joblib")


if __name__ == "__main__":
    main()
