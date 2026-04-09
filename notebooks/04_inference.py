import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
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
        mo,
        pd,
        prepare_features,
        print_metrics,
    )


@app.cell
def _():
    print("# Stage 4: Batch Inference")
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
def _(Path, load_model):
    print("## Load Model")
    _model_path = Path("models/rf_model.joblib")
    if _model_path.exists():
        model = load_model(_model_path)
        print(f"✓ Loaded model from {_model_path}")
    else:
        print(f"✗ Model not found. Run: make train")
        model = None
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
    if model is not None:
        print("## Run Predictions")
        X = df[get_feature_columns()]
        y_true = df[get_target_column()]
        predictions = model.predict(X)

        print(f"Generated {len(predictions):,} predictions")
        print(f"\nSample predictions (first 10):")
        for _i in range(min(10, len(predictions))):
            print(f"  Actual: {y_true.iloc[_i]:.0f}s | Predicted: {predictions[_i]:.0f}s")

        print("\n### Evaluation on Full Dataset")
        _metrics = evaluate_model(y_true, predictions)
        print_metrics(_metrics)
    return


@app.cell
def _(Path, df, get_feature_columns, model):
    if model is not None:
        print("## Save Predictions")
        _output = df.copy()
        _output["predicted_trip_time"] = model.predict(df[get_feature_columns()])
        _output["prediction_error"] = abs(_output["trip_time"] - _output["predicted_trip_time"])

        Path("output").mkdir(exist_ok=True)
        _output.to_parquet("output/predictions.parquet", index=False)
        print("✓ Predictions saved to output/predictions.parquet")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Problem

    This works for a single model file. But in production:
    - How do you version models?
    - How do you promote models from staging to production?
    - How do you roll back to a previous version?

    **This is where a Model Registry comes in.**
    """)
    return


if __name__ == "__main__":
    app.run()
