import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    import mlflow
    import mlflow.sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from mlops_workshop.features import prepare_features, get_feature_columns, get_target_column
    from mlops_workshop.evaluate import evaluate_model, print_metrics
    from mlops_workshop.train import save_model
    from pathlib import Path
    return mo, pd, os, mlflow, train_test_split, RandomForestRegressor, prepare_features, get_feature_columns, get_target_column, evaluate_model, print_metrics, save_model, Path


@app.cell
def _():
    print("# MLOps Workshop - Stage 3: Hyperparameter Tuning + MLflow")
    print("")
    print("Tune model hyperparameters and track experiments with MLflow.")


@app.cell
def _(pd, prepare_features):
    print("## Load and Prepare Data")
    _df_raw = pd.read_parquet("data/raw/nyc_taxi_sample.parquet")
    df = prepare_features(_df_raw)
    print(f"Dataset: {len(df):,} rows (10% sample)")
    return (df,)


@app.cell
def _(mlflow, os):
    print("## Configure MLflow")
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("nyc-taxi-duration")
    print("✓ MLflow configured")
    print("  Tracking URI: sqlite:///mlruns/mlflow.db")
    print("  Experiment: nyc-taxi-duration")


@app.cell
def _(df, get_feature_columns, get_target_column, train_test_split):
    print("## Prepare Train/Test Split")
    X = df[get_feature_columns()]
    y = df[get_target_column()]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


@app.cell
def _():
    print("## Hyperparameter Search")
    print("")
    print("Testing different configurations:")
    print("  - n_estimators: [50, 100]")
    print("  - max_depth: [5, 10, 15]")


@app.cell
def _(X_train, y_train, X_test, y_test, mlflow, RandomForestRegressor, evaluate_model):
    print("### Run 1: n_estimators=50, max_depth=5")

    with mlflow.start_run(run_name="rf_shallow"):
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 5)

        _model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        _model.fit(X_train, y_train)

        _pred = _model.predict(X_test)
        _metrics = evaluate_model(y_test, _pred)

        mlflow.log_metric("rmse", _metrics["rmse"])
        mlflow.log_metric("r2", _metrics["r2"])
        mlflow.sklearn.log_model(_model, "model")

        print(f"RMSE: {_metrics['rmse_minutes']:.2f} min | R²: {_metrics['r2']:.4f}")


@app.cell
def _(X_train, y_train, X_test, y_test, mlflow, RandomForestRegressor, evaluate_model):
    print("### Run 2: n_estimators=100, max_depth=10")

    with mlflow.start_run(run_name="rf_baseline"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        _model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        _model.fit(X_train, y_train)

        _pred = _model.predict(X_test)
        _metrics = evaluate_model(y_test, _pred)

        mlflow.log_metric("rmse", _metrics["rmse"])
        mlflow.log_metric("r2", _metrics["r2"])
        mlflow.sklearn.log_model(_model, "model")

        print(f"RMSE: {_metrics['rmse_minutes']:.2f} min | R²: {_metrics['r2']:.4f}")


@app.cell
def _(X_train, y_train, X_test, y_test, mlflow, RandomForestRegressor, evaluate_model):
    print("### Run 3: n_estimators=100, max_depth=15")

    with mlflow.start_run(run_name="rf_deeper"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 15)

        _model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        _model.fit(X_train, y_train)

        _pred = _model.predict(X_test)
        _metrics = evaluate_model(y_test, _pred)

        mlflow.log_metric("rmse", _metrics["rmse"])
        mlflow.log_metric("r2", _metrics["r2"])
        mlflow.sklearn.log_model(_model, "model")

        print(f"RMSE: {_metrics['rmse_minutes']:.2f} min | R²: {_metrics['r2']:.4f}")


@app.cell
def _():
    print("## View Results")
    print("")
    print("Run to view experiments:")
    print("  make mlflow")
    print("")
    print("Or:")
    print("  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db")


@app.cell
def _():
    print("## Summary")
    print("")
    print("MLflow tracks:")
    print("  - Parameters (n_estimators, max_depth)")
    print("  - Metrics (RMSE, R²)")
    print("  - Models (versioned artifacts)")
    print("")
    print("Next: Stage 4 - Batch inference and model registry")


if __name__ == "__main__":
    app.run()
