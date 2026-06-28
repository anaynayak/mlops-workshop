#!/usr/bin/env python
"""Build the Feast demo data and train the serving model."""

import pandas as pd

from mlops_workshop.evaluate import evaluate_model, print_metrics
from mlops_workshop.feature_store import (
    FEATURE_MODEL_PATH,
    FEATURE_STORE_MAX_ROWS,
    RAW_SAMPLE_PATH,
    ROUTE_FEATURE_NAME,
    bootstrap_feature_store,
    build_training_dataframe,
    get_feature_store_feature_columns,
    get_online_route_features,
    sample_demo_data,
)
from mlops_workshop.train import save_model, train_model


def main():
    print("Loading sample data...")
    raw_df = pd.read_parquet(RAW_SAMPLE_PATH)
    print(f"Loaded {len(raw_df):,} rows from {RAW_SAMPLE_PATH.relative_to(RAW_SAMPLE_PATH.parents[2])}")
    demo_df = sample_demo_data(raw_df, max_rows=FEATURE_STORE_MAX_ROWS)
    print(f"Using {len(demo_df):,} rows for the live Feast demo")

    print("\nBuilding Feast repo data...")
    store, route_stats = bootstrap_feature_store(demo_df)
    print(f"Prepared {len(route_stats):,} historical route rows")

    print("\nRetrieving the training set from Feast...")
    training_df = build_training_dataframe(demo_df, store=store)
    print(f"Training rows with {ROUTE_FEATURE_NAME}: {len(training_df):,}")

    print("\nTraining the feature-store model...")
    model, X_test, y_test = train_model(
        training_df,
        n_estimators=100,
        max_depth=10,
        feature_columns=get_feature_store_feature_columns(),
    )

    print("\nEvaluating model...")
    metrics = evaluate_model(y_test, model.predict(X_test))
    print_metrics(metrics)

    print("\nSaving model...")
    save_model(model, FEATURE_MODEL_PATH)
    print(f"Model saved to {FEATURE_MODEL_PATH.relative_to(FEATURE_MODEL_PATH.parents[1])}")

    sample_route = training_df.iloc[-1]
    online_features = get_online_route_features(
        pickup_location_id=int(sample_route["PULocationID"]),
        dropoff_location_id=int(sample_route["DOLocationID"]),
        store=store,
    )
    print(
        "\nSample online lookup: "
        f"{online_features['route_id']} -> {online_features[ROUTE_FEATURE_NAME]:.1f}s"
    )
    print("Start the API with: make serve")


if __name__ == "__main__":
    main()
