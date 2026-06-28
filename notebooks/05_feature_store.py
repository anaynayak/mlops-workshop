import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
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

    return (
        FEATURE_MODEL_PATH,
        FEATURE_STORE_MAX_ROWS,
        RAW_SAMPLE_PATH,
        ROUTE_FEATURE_NAME,
        bootstrap_feature_store,
        build_training_dataframe,
        evaluate_model,
        get_feature_store_feature_columns,
        get_online_route_features,
        pd,
        print_metrics,
        sample_demo_data,
        save_model,
        train_model,
    )


@app.cell
def _():
    print("# Stage 5: Feature Store + Serving")
    print("")
    print("Define one historical route feature, train with Feast, and reuse it in a live endpoint.")
    return


@app.cell
def _(FEATURE_STORE_MAX_ROWS, RAW_SAMPLE_PATH, pd, sample_demo_data):
    print("## Load Sample Data")
    raw_df = pd.read_parquet(RAW_SAMPLE_PATH)
    print(f"Loaded {len(raw_df):,} rows from {RAW_SAMPLE_PATH}")
    demo_df = sample_demo_data(raw_df, max_rows=FEATURE_STORE_MAX_ROWS)
    print(f"Using {len(demo_df):,} rows for the live Feast demo")
    return (demo_df,)


@app.cell
def _(bootstrap_feature_store, demo_df):
    print("## Build and Materialize Feast Features")
    store, route_stats = bootstrap_feature_store(demo_df)
    print(f"Historical route rows: {len(route_stats):,}")
    print(route_stats.head(5))
    return route_stats, store


@app.cell
def _(ROUTE_FEATURE_NAME, build_training_dataframe, demo_df, store):
    print("## Training Set from Feast")
    training_df = build_training_dataframe(demo_df, store=store)
    print(f"Rows with {ROUTE_FEATURE_NAME}: {len(training_df):,}")
    print(training_df.head(5))
    return (training_df,)


@app.cell
def _(
    evaluate_model,
    get_feature_store_feature_columns,
    print_metrics,
    train_model,
    training_df,
):
    print("## Train the Feature-Store Model")
    model, X_test, y_test = train_model(
        training_df,
        n_estimators=100,
        max_depth=10,
        feature_columns=get_feature_store_feature_columns(),
    )
    metrics = evaluate_model(y_test, model.predict(X_test))
    print_metrics(metrics)
    return metrics, model


@app.cell
def _(FEATURE_MODEL_PATH, model, save_model):
    print("## Save Model")
    save_model(model, FEATURE_MODEL_PATH)
    print(f"✓ Model saved to {FEATURE_MODEL_PATH}")
    return


@app.cell
def _(ROUTE_FEATURE_NAME, get_online_route_features, store, training_df):
    print("## Online Feature Lookup")
    sample = training_df.iloc[-1]
    lookup = get_online_route_features(
        pickup_location_id=int(sample["PULocationID"]),
        dropoff_location_id=int(sample["DOLocationID"]),
        store=store,
    )
    print(
        f"Route {lookup['route_id']} -> {ROUTE_FEATURE_NAME}="
        f"{lookup[ROUTE_FEATURE_NAME]:.1f}s"
    )
    return


@app.cell
def _(metrics):
    print("## Summary")
    print(f"RMSE: {metrics['rmse_minutes']:.2f} minutes")
    print("Run `make serve` to launch the API backed by Feast online features.")
    return


if __name__ == "__main__":
    app.run()
