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
    return mo, pd, mlflow, prepare_features, get_feature_columns, get_target_column, load_model, evaluate_model, print_metrics, Path


@app.cell
def _():
    print("# MLOps Workshop - Stage 4: Batch Inference + Model Registry")
    print("")
    print("Load a trained model and run batch predictions.")


@app.cell
def _(pd, prepare_features):
    print("## Load Data")
    _df_raw = pd.read_parquet("data/raw/nyc_taxi.parquet")
    df = prepare_features(_df_raw)
    print(f"Dataset: {len(df):,} rows")
    return (df,)


@app.cell
def _():
    print("## Option 1: Load from Local File")


@app.cell
def _(load_model, Path):
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


@app.cell
def _(mlflow):
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    print("MLflow tracking URI: sqlite:///mlruns/mlflow.db")


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
            print(f"  - {_run.info.run_name}: RMSE={_rmse:.2f if isinstance(_rmse, float) else _rmse}, R²={_r2:.4f if isinstance(_r2, float) else _r2}")
    else:
        print("No experiments found. Run: make train")


@app.cell
def _():
    print("## Run Batch Inference")


@app.cell
def _(df, model_local, get_feature_columns, get_target_column, evaluate_model, print_metrics):
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
    return (predictions,) if model_local else (None,)


@app.cell
def _(df, predictions, pd, Path):
    if predictions is not None:
        print("## Save Predictions")
        _output = df.copy()
        _output["predicted_trip_time"] = predictions
        _output["prediction_error"] = abs(_output["trip_time"] - predictions)

        Path("output").mkdir(exist_ok=True)
        _output.to_parquet("output/predictions.parquet", index=False)
        print("✓ Predictions saved to output/predictions.parquet")


@app.cell
def _():
    print("## Model Registry Concepts")
    print("")
    print("In production, you would:")
    print("  1. Register best model in MLflow Model Registry")
    print("  2. Promote to 'Staging' for validation")
    print("  3. Promote to 'Production' for serving")
    print("  4. Version models for rollback capability")
    print("")
    print("Example:")
    print("  mlflow.register_model(model_uri, 'nyc-taxi-model')")


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


if __name__ == "__main__":
    app.run()
