import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    from mlops_workshop import registry
    from mlops_workshop.features import prepare_features, get_feature_columns, get_target_column
    from mlops_workshop.evaluate import evaluate_model, print_metrics

    return (
        Path,
        evaluate_model,
        get_feature_columns,
        get_target_column,
        pd,
        prepare_features,
        print_metrics,
        registry,
    )


@app.cell
def _():
    print("# Stage 4: Inference")
    print("")
    print("Promote a tracked run to the registry, then serve it by alias.")
    return


@app.cell
def _(pd, prepare_features):
    _df_raw = pd.read_parquet("data/raw/nyc_taxi_sample.parquet")
    df = prepare_features(_df_raw)
    print(f"Dataset: {len(df):,} rows (10% sample)")
    return (df,)


@app.cell
def _(registry):
    print("## Pick a Candidate")
    _runs = registry.find_runs(metric="rmse")
    _cols = [c for c in ["run_id", "params.max_depth", "metrics.rmse"] if c in _runs.columns]
    print(_runs[_cols].head())

    best_run_id = _runs.iloc[0]["run_id"]
    print(f"\nBest run: {best_run_id}")
    return (best_run_id,)


@app.cell
def _(best_run_id, registry):
    print("## Register & Promote")
    _mv = registry.register_run(best_run_id)
    registry.set_alias("challenger", _mv.version)   # stage the candidate
    registry.set_alias("champion", _mv.version)     # promote: challenger beats champion
    print(f"✓ Registered version {_mv.version}, alias @champion")
    print(registry.list_versions())
    return


@app.cell
def _(registry):
    print("## Load by Alias")
    model = registry.load_model("champion")
    print("✓ Loaded models:/nyc-taxi-duration@champion")
    return (model,)


@app.cell
def _(df, evaluate_model, get_feature_columns, get_target_column, model, print_metrics):
    print("## Run Batch Inference")
    X = df[get_feature_columns()]
    y_true = df[get_target_column()]
    predictions = model.predict(X)

    print(f"Generated {len(predictions):,} predictions")
    print("\nSample predictions (first 10):")
    for _i in range(min(10, len(predictions))):
        print(f"  Actual: {y_true.iloc[_i]:.0f}s | Predicted: {predictions[_i]:.0f}s")

    print("\n### Evaluation")
    print_metrics(evaluate_model(y_true, predictions))
    return


@app.cell
def _(Path, df, get_feature_columns, model):
    print("## Save Predictions")
    _output = df.copy()
    _output["predicted_trip_time"] = model.predict(df[get_feature_columns()])
    _output["prediction_error"] = abs(_output["trip_time"] - _output["predicted_trip_time"])

    Path("output").mkdir(exist_ok=True)
    _output.to_parquet("output/predictions.parquet", index=False)
    print("✓ Predictions saved to output/predictions.parquet")
    return


@app.cell
def _():
    print("## Summary")
    print("  - Registered the best run and tagged it @champion")
    print("  - Inference loads by alias — never a file path or version")
    print("  - Promote / roll back = move the alias (consumer code unchanged)")
    return


if __name__ == "__main__":
    app.run()
