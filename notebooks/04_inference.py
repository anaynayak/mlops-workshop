import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlops_workshop.features import prepare_features, get_feature_columns, get_target_column
    from mlops_workshop.evaluate import evaluate_model, print_metrics
    from pathlib import Path

    return (
        MlflowClient,
        Path,
        evaluate_model,
        get_feature_columns,
        get_target_column,
        mlflow,
        pd,
        prepare_features,
        print_metrics,
    )


@app.cell
def _():
    print("# Stage 4: Inference")
    print("")
    print("Load models from the registry and run batch predictions.")
    return


@app.cell
def _(pd, prepare_features):
    _df_raw = pd.read_parquet("data/raw/nyc_taxi_sample.parquet")
    df = prepare_features(_df_raw)
    print(f"Dataset: {len(df):,} rows (10% sample)")
    return (df,)


@app.cell
def _(MlflowClient, mlflow):
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    _client = MlflowClient()
    print("## Model Registry")

    _versions = _client.search_model_versions("name='nyc-taxi-model'")
    if _versions:
        print(f"Registered model: nyc-taxi-model ({len(_versions)} versions)")
        for _v in _versions:
            _aliases = _v.aliases if hasattr(_v, 'aliases') else []
            _alias_str = f" [{', '.join(_aliases)}]" if _aliases else ""
            _run = _client.get_run(_v.run_id)
            _rmse = _run.data.metrics.get("rmse", "N/A")
            _rmse_str = f"{_rmse:.2f}" if isinstance(_rmse, float) else str(_rmse)
            print(f"  Version {_v.version}{_alias_str}: RMSE={_rmse_str}s")
    else:
        print("No registered models. Run Stage 3 first.")
    return


@app.cell
def _(mlflow):
    print("## Load Champion Model")
    model = mlflow.sklearn.load_model("models:/nyc-taxi-model@challenger")
    print("✓ Loaded champion model")
    print("  URI: models:/nyc-taxi-model@champion")
    return (model,)


@app.cell
def _(
    df,
    evaluate_model,
    get_feature_columns,
    get_target_column,
    model,
    print_metrics,
):
    print("## Run Batch Inference")
    X = df[get_feature_columns()]
    y_true = df[get_target_column()]
    predictions = model.predict(X)

    print(f"Generated {len(predictions):,} predictions")
    print(f"\nSample predictions (first 10):")
    for _i in range(min(10, len(predictions))):
        print(f"  Actual: {y_true.iloc[_i]:.0f}s | Predicted: {predictions[_i]:.0f}s")

    print("\n### Evaluation")
    _metrics = evaluate_model(y_true, predictions)
    print_metrics(_metrics)
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
def _(MlflowClient):
    print("## Promote Challenger → Champion")
    print("Compare challenger RMSE against champion. Promote only if better.\n")
    _client = MlflowClient()

    _champion = _client.get_model_version_by_alias("nyc-taxi-model", "champion")
    _challenger = _client.get_model_version_by_alias("nyc-taxi-model", "challenger")

    _champion_rmse = _client.get_run(_champion.run_id).data.metrics["rmse"]
    _challenger_rmse = _client.get_run(_challenger.run_id).data.metrics["rmse"]

    print(f"  Champion: Version {_champion.version} (RMSE={_champion_rmse:.2f}s)")
    print(f"  Challenger: Version {_challenger.version} (RMSE={_challenger_rmse:.2f}s)")

    if _challenger_rmse < _champion_rmse:
        _client.set_registered_model_alias("nyc-taxi-model", "champion", int(_challenger.version))
        print(f"\n  ✓ Challenger wins! Version {_challenger.version} is now champion.")
        print(f"    Old champion (v{_champion.version}) kept for rollback.")
    else:
        print(f"\n  ✗ Champion holds. Challenger (v{_challenger.version}) did not beat it.")
    return


@app.cell
def _():
    print("## Summary")
    print("  - Model registry versions every trained model")
    print("  - Champion alias = current production model")
    print("  - Challenger alias = candidate for promotion")
    print("  - Load by alias: models:/nyc-taxi-model@champion")
    print("  - Promote by reassigning the alias")
    print("")
    print("Next: Stage 5 - Operations")
    return


if __name__ == "__main__":
    app.run()
