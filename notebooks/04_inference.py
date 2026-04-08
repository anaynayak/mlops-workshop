import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import mlflow
    from mlops_workshop.features import prepare_features, get_feature_columns, get_target_column
    from mlops_workshop.train import load_model
    from mlops_workshop.evaluate import evaluate_model, print_metrics
    from pathlib import Path

    return (
        Path,
        evaluate_model,
        get_feature_columns,
        get_target_column,
        load_model,
        mlflow,
        pd,
        prepare_features,
        print_metrics,
    )


@app.cell
def _():
    print("# MLOps Workshop - Stage 4: Batch Inference + Model Registry")
    print("")
    print("Load a trained model and run batch predictions.")
    return


@app.cell
def _(pd, prepare_features):
    print("## Load Data")
    _df_raw = pd.read_parquet("data/raw/nyc_taxi_sample.parquet")
    df = prepare_features(_df_raw)
    print(f"Dataset: {len(df):,} rows (10% sample)")
    return (df,)


@app.cell
def _():
    print("## Option 1: Load from Local File")
    return


@app.cell
def _(Path, load_model):
    _model_path = Path("models/rf_model.joblib")
    if _model_path.exists():
        model_local = load_model(_model_path)
        print(f"✓ Loaded model from {_model_path}")
    else:
        print(f"✗ Model not found at {_model_path}")
        print("  Run: make train")
        model_local = None
    return (model_local,)


@app.cell
def _():
    print("## Option 2: Load from MLflow Registry")
    return


@app.cell
def _(mlflow):
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    print("MLflow tracking URI: sqlite:///mlruns/mlflow.db")
    return


@app.cell
def _(mlflow):
    print("### List Runs")
    _client = mlflow.tracking.MlflowClient()
    _experiment = _client.get_experiment_by_name("nyc-taxi-duration")
    if _experiment:
        _runs = _client.search_runs(_experiment.experiment_id, max_results=5)
        print(f"Found {len(_runs)} runs:")
        for _run in _runs:
            _rmse = _run.data.metrics.get("rmse", "N/A")
            _r2 = _run.data.metrics.get("r2", "N/A")
            _rmse_str = f"{_rmse:.2f}" if isinstance(_rmse, float) else str(_rmse)
            _r2_str = f"{_r2:.4f}" if isinstance(_r2, float) else str(_r2)
            print(f"  - {_run.info.run_name}: RMSE={_rmse_str}, R²={_r2_str}")
    else:
        print("No experiments found. Run: make train")
    return


@app.cell
def _():
    print("## Run Batch Inference")
    return


@app.cell
def _(
    df,
    evaluate_model,
    get_feature_columns,
    get_target_column,
    model_local,
    print_metrics,
):
    if model_local is not None:
        print("### Predictions with Local Model")
        X = df[get_feature_columns()]
        y_true = df[get_target_column()]

        predictions = model_local.predict(X)

        print(f"Generated {len(predictions):,} predictions")
        print(f"\nSample predictions (first 10):")
        for _i in range(min(10, len(predictions))):
            print(f"  Actual: {y_true.iloc[_i]:.0f}s | Predicted: {predictions[_i]:.0f}s")

        print("\n### Evaluation on Full Dataset")
        _metrics = evaluate_model(y_true, predictions)
        print_metrics(_metrics)
    else:
        print("No model loaded. Skipping predictions.")
    return


@app.cell
def _():
    print("## Model Registry Concepts")
    print("In production, you would:")
    print("  1. Register best model in MLflow Model Registry")
    print("  2. Promote to 'Staging' for validation")
    print("  3. Promote to 'Production' for serving")
    print("  4. Version models for rollback capability")
    print("Example:")
    print("  mlflow.register_model(model_uri, 'nyc-taxi-model')")
    return


@app.cell
def _():
    return


@app.cell
def _():
    print("## Summary")
    print("")
    print("Stage 4 covered:")
    print("  - Loading models from local files")
    print("  - Loading models from MLflow")
    print("  - Running batch inference")
    print("  - Model registry concepts")
    print("")
    print("Next: Stage 5 - Operations (testing, CI/CD)")
    return


if __name__ == "__main__":
    app.run()
